"""
National counterpart to fit_texas_chain.py: chain the three-strain model's
US national fit across seasons 2021-22 through 2025-26, using the real
Algorithm 10 remap (tsm.end_of_season_remap) between simulated seasons.
2021-22 remains the chain's sole data-anchored season
(fit_national_hosp_2021.py's AR bootstrap from real national 2020-21
hospitalizations) -- same reasoning as the Texas chain (no earlier season's
hospitalization data exists to bootstrap 2020-21 itself the same way).
Every later transition (2021-22->2022-23->...->2025-26) uses the model's
own simulated end-of-season E/I_R/I_H/H_R/H_D/R directly.

Structurally identical to fit_texas_chain.py -- same 5-parameter fit
(beta_H1, beta_H3, beta_B, epsilon_H, t0), same multi-start
beta-scale x t0-offset cross product, same population-growth-injection
mechanism -- with every input swapped for its national equivalent:
national hospitalization target (location="US"), national subtype shares
(data_prep.load_national_subtype_shares, a different, simpler-structured
file than the state-level Public Health Labs file -- no unsubtyped bucket
to reallocate), national mu(t) (region="United States", now available for
all 6 seasons in the merged master vax file), and national VE
(data_prep.build_season_alpha("United States") -- CDC VE estimates were
already national to begin with, so this removes a real geography mismatch
the Texas fit had).

Population growth: real US national year-over-year growth (FRED series
POPTHM, https://fred.stlouisfed.org/series/POPTHM), applied on top of
fit_national_hosp_2021.NATIONAL_POP (332,454,000, the same July-2021
POPTHM value, so no base-mismatch jump at the first boundary the way the
Texas chain had to reason about).
"""
import datetime as dt

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import data_prep as dp
import three_strain_model as tsm
import fit_national_hosp_2021 as fit_nat

REGION = "United States"
SEASON_START_YEARS = [2022, 2023, 2024, 2025]   # 2022-23 through 2025-26

BOUNDS = ([1e-3, 1e-3, 1e-3, 1e-4, -30.0], [3.0, 3.0, 3.0, 1.0, 30.0])
X_SCALE = [0.3, 0.3, 0.3, 0.03, 10.0]

# US resident population, July 1 of each year (FRED series POPTHM, a
# single consistently-revised monthly series). 2021 matches
# fit_national_hosp_2021.NATIONAL_POP exactly (same source, same month).
NATIONAL_POP_BY_YEAR = {
    2020: 331_862_000,
    2021: 332_454_000,
    2022: 334_370_000,
    2023: 337_147_000,
    2024: 340_335_000,
    2025: 342_076_000,
    2026: 342_909_000,
}


def add_population_growth(S, from_year, to_year):
    """Inject the real US year-over-year population delta (FRED POPTHM) as
    new naive/unvaccinated (h=N) individuals -- end_of_season_remap has no
    growth mechanism of its own, it only relabels existing compartments.
    Also bumps tsm.N_TOTAL to match (it's the force-of-infection
    denominator, must stay in sync with S's total)."""
    delta = NATIONAL_POP_BY_YEAR[to_year] - NATIONAL_POP_BY_YEAR[from_year]
    idx_N = tsm.H_GROUPS.index("N")
    S = S.copy()
    S[0, idx_N] += max(delta, 0)
    tsm.set_population(tsm.N_TOTAL + delta)
    return S


def build_x0_candidates(x0):
    """Same cross-product multi-start strategy as fit_texas_chain's
    build_x0_candidates (see there for the fuller rationale -- crossing
    beta-scale variants with t0 offsets, not varying them independently,
    after that combination missed a real regression in the Texas chain)."""
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


def fit_one_season(season_start_year, initial_state_unseeded, x0, region=REGION, verbose=True):
    """Fit beta_H1, beta_H3, beta_B, epsilon_H, t0 for one season, chained
    from `initial_state_unseeded` (see fit_texas_chain.fit_one_season for
    the fuller docstring -- identical method, national inputs)."""
    anchor = dt.date(season_start_year, 10, 1)
    day_grid = np.arange(0, tsm.SEASON_DAYS + 1)
    dates, target = dp.build_weekly_hosp_target(
        season_start_year, hosp_location="US", subtype_region=region,
        shares_fn=dp.load_national_subtype_shares)
    n_weeks = len(dates)

    season_alpha = dp.build_season_alpha(season_start_year, region)
    tsm.set_alpha(season_alpha["H1"], season_alpha["H3"], season_alpha["B"])

    overall_cum = dp.build_overall_vax_daily_from_master(season_start_year, day_grid, region)
    mu_daily = dp.hazard_from_daily_cum(overall_cum)

    def mu_eff(t):
        return 0.0 if t < 0 else np.interp(t, day_grid, mu_daily)

    def run(params):
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

    fig.suptitle(f"Observed vs. Simulated Flu Hospitalizations -- United States, {season_label} Season")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


def run_chain():
    result_2021 = fit_nat.fit()
    Y_hat_2021, final_state = fit_nat.run_full(result_2021.x)
    print(f"2021-22: success={result_2021.success} beta=({result_2021.x[0]:.4f}, "
          f"{result_2021.x[1]:.4f}, {result_2021.x[2]:.4f}) epsilon_H={result_2021.x[3]:.4f}")

    summary = [("2021-22", np.append(result_2021.x, 0.0), result_2021.cost)]
    x0 = np.append(result_2021.x, 0.0)   # append t0=0 -- 2021-22 itself isn't refit with t0
    prev_year = fit_nat.FIT_SEASON_START_YEAR   # 2021

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
                         f"national_{season_start_year}_{(season_start_year + 1) % 100:02d}_hosp_fit.png")

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
