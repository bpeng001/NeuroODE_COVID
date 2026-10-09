"""
Chain the three-strain model's Texas fit across seasons 2021-22 through
2025-26, using the real Algorithm 10 remap (tsm.end_of_season_remap)
between simulated seasons instead of an empirical AR-based bootstrap for
every transition -- that empirical bootstrap (fit_texas_hosp_2021.py's
build_initial_state_from_2020_21) is kept unchanged and still anchors only
the chain's very first season (2021-22, built from real 2020-21
hospitalization data), because no real Texas hospitalization data exists
for 2019-20 to bootstrap 2020-21 itself the same way (time-series.csv's
earliest row, any location, is 2020-08-08). Every later transition
(2021-22->2022-23->...->2025-26) uses the model's own simulated
end-of-season E/I/H/R directly -- no data-anchored reconstruction needed
once a season has actually been simulated.

Each season is refit independently (beta_H1, beta_H3, beta_B, epsilon_H)
against that season's own reported hospitalizations, rather than assuming
shared ascertainment across seasons (see MODEL.md / conversation history).

Vaccination mu(t) for every season (including a refit of 2021-22) comes
from data_prep.build_overall_vax_daily_from_master, reading the merged
vaccination/state_weekly_vaccination_2020_2026.csv -- this replaces the
raw-CDC-file loader for 2021-22, whose first real Texas observation
(Oct 9/30) is after the season's Oct 1 anchor and so clamped the early
ramp-up to an inflated flat value; the merged file's 2021-22 rows
(scenario-hub reconstructed) genuinely start in August.

Population growth: end_of_season_remap conserves total population exactly
(it only relabels compartments), so real Texas population growth is
injected manually at each season boundary as new naive/unvaccinated (h=N)
individuals, using year-over-year deltas from the Census Bureau's Texas
population series (FRED series TXPOP, https://fred.stlouisfed.org/series/TXPOP).
Deltas are applied on top of fit_texas_hosp_2021.TX_POP (the original
Vintage 2021 estimate already used and validated there) rather than
switching to FRED's absolute levels mid-chain, since FRED's revised 2021
value differs from that estimate by <0.2% -- switching bases would
introduce a spurious one-time jump at the first boundary instead of a
smooth, real year-over-year change.
"""
import datetime as dt

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import data_prep as dp
import three_strain_model as tsm
import fit_texas_hosp_2021 as fit2021

REGION = "Texas"
SEASON_START_YEARS = [2022, 2023, 2024, 2025]   # 2022-23 through 2025-26

# 5th calibrated parameter, t0: the calendar day (relative to the season's
# Oct-1 anchor) at which the fixed Algorithm-11 seed is actually injected.
# Every season previously reseeded at a hardcoded t=0 with a fixed I0,
# forcing the ENTIRE observed rise-to-peak to come from pure exponential
# growth within whatever window that season's real peak happened to fall
# in -- with only beta free, that couples peak *timing* to peak
# *magnitude* (both governed by growth rate), which can't independently
# fit seasons whose real onset ran earlier or later than usual (confirmed:
# 2023-24 needed a later effective start, 2025-26 needed an earlier one).
# t0 decouples "when this season's epidemic effectively took hold" from
# beta's role in growth rate/magnitude. Bounded to +/-30 days: physically
# plausible, and negative t0 needs mu_eff(t)=0 for t<0 (see fit_one_season)
# since no vaccination pressure applies before the season's nominal start.
BOUNDS = ([1e-3, 1e-3, 1e-3, 1e-4, -30.0], [3.0, 3.0, 3.0, 1.0, 30.0])
X_SCALE = [0.3, 0.3, 0.3, 0.03, 10.0]

# Census Bureau resident population estimates for Texas, July 1 of each
# year (FRED series TXPOP -- a single consistently-revised series, not
# mixed vintages). 2026 not yet published as of this writing (Vintage 2026
# releases end of 2026); held flat at the 2025 value as a placeholder.
TX_POP_BY_YEAR = {
    2020: 29_237_895,
    2021: 29_572_672,
    2022: 30_118_002,
    2023: 30_719_247,
    2024: 31_318_578,
    2025: 31_709_821,
    2026: 31_709_821,   # placeholder
}


def add_population_growth(S, from_year, to_year):
    """Inject the real Texas year-over-year population delta (FRED TXPOP)
    as new naive/unvaccinated (h=N) individuals -- end_of_season_remap has
    no growth mechanism of its own, it only relabels existing compartments.
    Also bumps tsm.N_TOTAL to match (it's the force-of-infection
    denominator, must stay in sync with S's total)."""
    delta = TX_POP_BY_YEAR[to_year] - TX_POP_BY_YEAR[from_year]
    idx_N = tsm.H_GROUPS.index("N")
    S = S.copy()
    S[0, idx_N] += max(delta, 0)
    tsm.set_population(tsm.N_TOTAL + delta)
    return S


def build_x0_candidates(x0):
    """A diverse set of starting points for multi-start least_squares, as a
    cross product of beta-scale variants x t0 offsets.

    Varying beta-scale and t0 independently (rather than crossed) missed a
    real regression: for 2024-25, none of the beta-scale variants at t0=0
    happened to land near this season's actual good optimum (moderate
    beta, small |t0|), and none of the t0 variants at the previous
    season's own beta scale did either -- confirmed by comparing costs
    before/after adding t0 (2024-25 got *worse*, 9.97e6 -> 1.86e7, purely
    because the search never revisited the right neighborhood, not because
    a free t0 makes the achievable fit worse). Crossing every beta-scale
    with every t0 offset costs more compute but reliably covers this."""
    beta_h1, beta_h3, beta_b, eps, t0 = x0
    beta_bases = {
        "x0": (np.array([beta_h1, beta_h3, beta_b]), eps),
        "hill": (np.array([0.3913, 0.3917, 0.35815]), 0.03),
        "1.5x": (np.array([beta_h1, beta_h3, beta_b]) * 1.5, eps),
        "2x": (np.array([beta_h1, beta_h3, beta_b]) * 2.0, eps),
        "0.6x": (np.array([beta_h1, beta_h3, beta_b]) * 0.6, eps * 2),
        "0.4x": (np.array([beta_h1, beta_h3, beta_b]) * 0.4, eps * 2),
    }
    t0_options = sorted({round(t0, 4), 0.0, 20.0, -20.0})

    candidates = []
    for betas, e in beta_bases.values():
        for t0_opt in t0_options:
            candidates.append(np.array([betas[0], betas[1], betas[2], e, t0_opt]))
    return [np.clip(c, BOUNDS[0], BOUNDS[1]) for c in candidates]


def build_season_context(season_start_year, region=REGION):
    """Everything about one season that doesn't depend on the trial
    parameter vector: the hospitalization target/dates, season-specific
    VE (set into tsm globally), and the vaccination hazard function.
    Shared by fit_one_season (fitting) and run_season_with_params
    (replaying already-known parameters, e.g. to rebuild a final_state
    without re-running multi-start)."""
    anchor = dt.date(season_start_year, 10, 1)
    day_grid = np.arange(0, tsm.SEASON_DAYS + 1)
    dates, target = dp.build_weekly_hosp_target(season_start_year, subtype_region=region)

    season_alpha = dp.build_season_alpha(season_start_year, region)
    tsm.set_alpha(season_alpha["H1"], season_alpha["H3"], season_alpha["B"])

    overall_cum = dp.build_overall_vax_daily_from_master(season_start_year, day_grid, region)
    mu_daily = dp.hazard_from_daily_cum(overall_cum)

    def mu_eff(t):
        # No vaccination pressure before the season's nominal Oct-1 anchor
        # -- only relevant when a candidate t0 < 0 gives the epidemic a
        # head start before day 0.
        return 0.0 if t < 0 else np.interp(t, day_grid, mu_daily)

    return anchor, dates, target, mu_eff


def run_season_with_params(season_start_year, initial_state_unseeded, params, region=REGION):
    """Run one season forward for a *known* (already-fitted or otherwise
    chosen) parameter vector -- no optimization. Returns (Y_hat,
    final_state, dates, target). Used both inside fit_one_season's
    residuals() and standalone, to rebuild a season's final_state (e.g.
    for forecasting forward) without repeating the multi-start search when
    nothing about that season's fit needs to change."""
    anchor, dates, target, mu_eff = build_season_context(season_start_year, region)
    beta_h1, beta_h3, beta_b, epsilon_H, t0 = params
    tsm.set_betas(beta_h1, beta_h3, beta_b)
    seeded_initial = tsm.seed_infections(*initial_state_unseeded)
    (t_grid, S_traj, E_traj, I_R_traj, I_H_traj,
     H_R_traj, H_D_traj, R_traj, D_traj) = tsm.run_season(
        initial=seeded_initial, mu_func=mu_eff, days=tsm.SEASON_DAYS, t_start=t0)
    tsm.check_positivity(S_traj, E_traj, I_R_traj, I_H_traj, H_R_traj, H_D_traj, R_traj, D_traj)
    tsm.check_population_conservation(S_traj, E_traj, I_R_traj, I_H_traj, H_R_traj, H_D_traj, R_traj, D_traj)
    daily_hosp = tsm.compute_daily_hosp_incidence(I_H_traj)
    weekly_hosp_headcount = dp.aggregate_at_dates(daily_hosp, t_grid, dates, anchor)
    Y_hat = epsilon_H * weekly_hosp_headcount
    final_state = (S_traj[-1], E_traj[-1], I_R_traj[-1], I_H_traj[-1],
                    H_R_traj[-1], H_D_traj[-1], R_traj[-1], D_traj[-1])
    return Y_hat, final_state, dates, target


def fit_one_season(season_start_year, initial_state_unseeded, x0, region=REGION, verbose=True):
    """Fit beta_H1, beta_H3, beta_B, epsilon_H, t0 for one season, chained
    from `initial_state_unseeded` (already remapped + growth-adjusted for
    this season's start, but NOT yet seeded -- see
    tsm.end_of_season_remap(..., reseed=False) -- since the Algorithm 11
    seed is injected here at the calibrated time t0, not always at t=0),
    via multi-start least_squares (see build_x0_candidates). Returns
    (result, Y_hat, final_state, dates, target) for the lowest-cost
    candidate."""
    _, dates, target, _ = build_season_context(season_start_year, region)
    n_weeks = len(dates)

    def run(params):
        Y_hat, final_state, _, _ = run_season_with_params(
            season_start_year, initial_state_unseeded, params, region)
        return Y_hat, final_state

    def residuals(params):
        try:
            Y_hat, _ = run(params)
        except (AssertionError, RuntimeError):
            return np.full(n_weeks * len(tsm.STRAINS), 1e5)
        return (Y_hat - target).ravel()

    best_result = None
    for i, x0_candidate in enumerate(build_x0_candidates(x0)):
        result = least_squares(residuals, x0_candidate, bounds=BOUNDS, xtol=1e-12, ftol=1e-12, x_scale=X_SCALE)
        if verbose:
            print(f"    candidate {i}: x0={np.round(x0_candidate, 4)} -> "
                  f"cost={result.cost:.4e} nfev={result.nfev}")
        if best_result is None or result.cost < best_result.cost:
            best_result = result
    result = best_result
    Y_hat, final_state = run(result.x)
    return result, Y_hat, final_state, dates, target


def plot_season_fit(season_start_year, dates, target, Y_hat, out_path):
    n_weeks = len(dates)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    colors = {"H1": "#1f77b4", "H3": "#d62728", "B": "#2ca02c"}
    season_label = f"{season_start_year}-{(season_start_year + 1) % 100:02d}"

    def format_date_axis(ax):
        ax.set_xlabel(f"Week ending ({n_weeks} reported weeks)")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    ax = axes[0]
    for j, m in enumerate(tsm.STRAINS):
        ax.plot(dates, target[:, j], color=colors[m], linestyle="--",
                 linewidth=1.2, marker="x", markersize=4, label=f"{m} observed")
        ax.plot(dates, Y_hat[:, j], color=colors[m], linewidth=2.0, label=f"{m} fitted")
    format_date_axis(ax)
    ax.set_ylabel("Weekly incident hospitalizations (count)")
    ax.set_title("By subtype")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(dates, target.sum(axis=1), color="#333333", linestyle="--",
             linewidth=1.5, marker="x", markersize=4, label="Observed (all subtypes)")
    ax.plot(dates, Y_hat.sum(axis=1), color="#d62728", linewidth=2.2, label="Simulated (all subtypes)")
    format_date_axis(ax)
    ax.set_ylabel("Weekly incident hospitalizations (count)")
    ax.set_title("Total")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Observed vs. Simulated Flu Hospitalizations -- Texas, {season_label} Season")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


def run_chain():
    # Season 1 (2021-22): reuse fit_texas_hosp_2021.py's own fit, unchanged
    # method (AR bootstrap from real 2020-21 hospitalizations), refit here
    # only because build_mu_func_2021_22 was updated to the corrected
    # master-vax loader.
    result_2021 = fit2021.fit()
    Y_hat_2021, final_state = fit2021.run_full(result_2021.x)
    print(f"2021-22: success={result_2021.success} beta=({result_2021.x[0]:.4f}, "
          f"{result_2021.x[1]:.4f}, {result_2021.x[2]:.4f}) epsilon_H={result_2021.x[3]:.4f}")

    summary = [("2021-22", np.append(result_2021.x, 0.0), result_2021.cost)]
    x0 = np.append(result_2021.x, 0.0)   # append t0=0 -- 2021-22 itself isn't refit with t0
    prev_year = fit2021.FIT_SEASON_START_YEAR   # 2021

    for season_start_year in SEASON_START_YEARS:
        remapped = tsm.end_of_season_remap(*final_state, reseed=False)
        S0, E0, I_R0, I_H0, H_R0, H_D0, R0, D0 = remapped
        S0 = add_population_growth(S0, prev_year, season_start_year)
        initial_state_unseeded = (S0, E0, I_R0, I_H0, H_R0, H_D0, R0, D0)

        result, Y_hat, final_state, dates, target = fit_one_season(
            season_start_year, initial_state_unseeded, x0)
        season_label = f"{season_start_year}-{(season_start_year + 1) % 100:02d}"
        print(f"{season_label}: success={result.success} beta=({result.x[0]:.4f}, "
              f"{result.x[1]:.4f}, {result.x[2]:.4f}) epsilon_H={result.x[3]:.4f} "
              f"t0={result.x[4]:+.1f}d N_TOTAL={tsm.N_TOTAL:,.0f}")

        plot_season_fit(season_start_year, dates, target, Y_hat,
                         f"texas_{season_start_year}_{(season_start_year + 1) % 100:02d}_hosp_fit.png")

        summary.append((season_label, result.x, result.cost))
        x0 = result.x
        prev_year = season_start_year

    print("\nSeason-by-season fitted parameters:")
    print(f"{'Season':<10}{'beta_H1':>10}{'beta_H3':>10}{'beta_B':>10}{'epsilon_H':>12}{'t0':>8}{'cost':>14}")
    for season_label, params, cost in summary:
        print(f"{season_label:<10}{params[0]:>10.4f}{params[1]:>10.4f}{params[2]:>10.4f}"
              f"{params[3]:>12.4f}{params[4]:>+8.1f}{cost:>14.4e}")

    return summary


if __name__ == "__main__":
    run_chain()
