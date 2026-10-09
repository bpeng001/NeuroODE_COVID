"""
Fit the three-strain model's beta_H1, beta_H3, beta_B (and a hospitalization
ascertainment/scale parameter epsilon_H) to Texas 2021-22 season flu
hospitalizations, subtype-split via ICL_NREVSS Public Health Labs season
shares.

Model population scale: N_TOTAL is set to TX_POP (a real headcount, via
tsm.set_population), not the module's default normalized 1.0 -- so every
compartment (S,E,I_R,I_H,H_R,H_D,R,D) is in actual numbers of people,
directly comparable to the raw hospitalization counts in time-series.csv. This
requires (and three_strain_model.py implements) Algorithm 4's counts-based
force-of-infection form (lambda_m = beta_m*I_m/N_TOTAL, not beta_m*I_m) --
without that /N_TOTAL, transmission dynamics would be wrong at this scale.

Model-to-data scale: epsilon_H is still a calibrated (not fixed) ascertainment
*probability* in (0,1] -- it's needed, not optional: this MIDAS hub time
series (like FluSurv-NET/COVID-NET) comes from a partial hospital-reporting
network, not 100% statewide capture. Fixing epsilon_H=1 (assuming full
ascertainment) was tried and produces a ~24x overshoot of observed counts --
confirming real, unavoidable under-ascertainment. The fitted value directly
is the completeness fraction now (no separate division by TX_POP needed,
unlike the earlier proportions-based version of this script).

time-series.csv (repo root of this script) is the MIDAS flu-scenario
modeling-hub target-data file (genuinely flu-specific -- distinct from the
COVID-19 hospitalization series at Data/time-series.csv, which was ruled
out). Texas ("48") reporting has a real gap Aug 2021-Jan 2022; only the 34
weeks actually reported (Feb-Sep 2022) are used as the fitting target, not
backfilled with zeros.

Initial state (start of 2021-22 season) is estimated from the 2020-21
season's observed hospitalizations: cumulative hospitalized count per strain
divided by (epsilon_H * p_H) gives an estimated true attack rate (p_H
converts the model's true-infection rate into its hospitalization rate, so
it must be divided back out here since we're inverting from hospitalizations
to infections, not reading model-simulated infections directly). Crossed
with a fixed 2020-21 end-of-season vaccination coverage (46.7%, user-
supplied), assuming infection/vaccination independence, exactly as
Algorithm 10 would from simulated state.

2020-21 was a real, extraordinarily low flu season in Texas (near-zero H1,
tiny H3/B counts) -- consistent with well-documented pandemic-era NPI
suppression of flu circulation; this is not a data artifact.

Current-season (2021-22) vaccination mu_y(t): population-weighted (Census
~27.3% children / ~72.7% adult) blend of real weekly Texas series from both
judz-8etw (children) and sw5n-wg2p (adult) -- both genuinely cover 2021-22,
unlike the 2019-20 fit which needed the Bexar/Houston monthly proxy.

Calibrated parameters: beta_H1, beta_H3, beta_B, epsilon_H (4 total).
"""

import datetime as dt

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import data_prep as dp
import three_strain_model as tsm

PREVIOUS_SEASON_VAX = 0.467   # fixed, user-supplied 2020-21 end-of-season coverage
TX_POP = 29_527_941           # fixed, Census Bureau Vintage 2021 Texas population estimate
INIT_SEASON_START_YEAR = 2020
FIT_SEASON_START_YEAR = 2021
ANCHOR = dt.date(FIT_SEASON_START_YEAR, 10, 1)

tsm.set_population(TX_POP)   # model compartments now in headcounts, not fractions
_ALPHA_2021_22 = dp.build_season_alpha(FIT_SEASON_START_YEAR)   # real, season-specific VE (CDC), not the fixed Hill defaults
tsm.set_alpha(_ALPHA_2021_22["H1"], _ALPHA_2021_22["H3"], _ALPHA_2021_22["B"])

DAY_GRID = np.arange(0, tsm.SEASON_DAYS + 1)
DATES_2021_22, TARGET_HOSP_2021_22 = dp.build_weekly_hosp_target(FIT_SEASON_START_YEAR)
N_FIT_WEEKS = len(DATES_2021_22)


def build_initial_state_from_2020_21(epsilon_H):
    """Empirical analogue of Algorithm 10, driven by observed 2020-21
    hospitalization data instead of simulated E/I/H/R. All quantities here
    are headcounts (N_TOTAL=TX_POP), not fractions."""
    _, target_2020_21 = dp.build_weekly_hosp_target(INIT_SEASON_START_YEAR)
    cumulative_hosp = target_2020_21.sum(axis=0)   # (n_m,) raw counts

    AR = cumulative_hosp / (epsilon_H * tsm.P_H_VEC)     # headcount estimate of true infections, per strain's own IHR
    if np.any(AR < 0) or AR.sum() >= TX_POP:
        # This epsilon_H implies an impossible 2020-21 epidemic (more people
        # infected than exist). Reject the parameter combination via the same
        # RuntimeError path residuals() already uses for other infeasible
        # trial points, instead of silently rescaling AR to fit -- rescaling
        # would flatten AR's dependence on epsilon_H across a wide, plausible
        # part of the search range (anywhere epsilon_H <~ 0.014 here) and
        # could hand the optimizer an artificial attractor.
        raise RuntimeError(
            f"epsilon_H={epsilon_H:.6g} implies an infeasible 2020-21 attack "
            f"rate (AR.sum()={AR.sum():.3e} >= TX_POP={TX_POP})"
        )

    idx = {h: i for i, h in enumerate(tsm.H_GROUPS)}
    S = np.zeros((2, tsm.N_H))
    E = np.zeros((2, tsm.N_M))
    I_R = np.zeros((2, tsm.N_M))
    I_H = np.zeros((2, tsm.N_M))
    H_R = np.zeros((2, tsm.N_M))
    H_D = np.zeros((2, tsm.N_M))
    R = np.zeros((2, tsm.N_M))
    D = np.zeros((2, tsm.N_M))

    V = PREVIOUS_SEASON_VAX
    uninfected = TX_POP - AR.sum()
    S[0, idx["N"]] = uninfected * (1 - V)
    S[0, idx["V"]] = uninfected * V
    for j, m in enumerate(tsm.STRAINS):
        S[0, idx[m]] = AR[j] * (1 - V)
        S[0, idx[m + "V"]] = AR[j] * V

    return tsm.seed_infections(S, E, I_R, I_H, H_R, H_D, R, D)


def build_mu_func_2021_22():
    # build_overall_vax_daily_from_master (reads the merged
    # state_weekly_vaccination_2020_2026.csv) is used here instead of
    # build_overall_vax_daily: the raw CDC weekly files' first real Texas
    # observation for 2021-22 is Oct 9 (children) / Oct 30 (adults), both
    # after the season's Oct 1 anchor, so build_overall_vax_daily clamps
    # every earlier day to that first value (~18.8%/21.6%) instead of
    # showing the campaign ramp up from near-zero. The master file's
    # 2021-22 rows (scenario_hub_reconstructed, from RD2) genuinely start
    # in August, so it interpolates the ramp-up correctly instead of
    # clamping it away.
    overall_cum = dp.build_overall_vax_daily_from_master(FIT_SEASON_START_YEAR, DAY_GRID)
    mu_daily = dp.hazard_from_daily_cum(overall_cum)
    return lambda t: np.interp(t, DAY_GRID, mu_daily)


MU_2021_22 = build_mu_func_2021_22()


def run_full(params):
    """Run the 2021-22 season forward for a trial parameter vector. Returns
    (Y_hat aligned to DATES_2021_22, final-day compartments) -- the latter
    for chaining into 2022-23 via tsm.end_of_season_remap (see
    fit_texas_chain.py)."""
    beta_h1, beta_h3, beta_b, epsilon_H = params
    tsm.set_betas(beta_h1, beta_h3, beta_b)

    initial = build_initial_state_from_2020_21(epsilon_H)
    (t_grid, S_traj, E_traj, I_R_traj, I_H_traj,
     H_R_traj, H_D_traj, R_traj, D_traj) = tsm.run_season(
        initial=initial, mu_func=MU_2021_22, days=tsm.SEASON_DAYS)

    tsm.check_positivity(S_traj, E_traj, I_R_traj, I_H_traj, H_R_traj, H_D_traj, R_traj, D_traj)
    tsm.check_population_conservation(S_traj, E_traj, I_R_traj, I_H_traj, H_R_traj, H_D_traj, R_traj, D_traj)

    daily_hosp = tsm.compute_daily_hosp_incidence(I_H_traj)   # (n_t, n_m) headcount/day
    weekly_hosp_headcount = dp.aggregate_at_dates(daily_hosp, t_grid, DATES_2021_22, ANCHOR)

    Y_hat = epsilon_H * weekly_hosp_headcount
    final_state = (S_traj[-1], E_traj[-1], I_R_traj[-1], I_H_traj[-1],
                    H_R_traj[-1], H_D_traj[-1], R_traj[-1], D_traj[-1])
    return Y_hat, final_state


def simulate(params):
    """Y_hat only (raw hospitalization-count scale) -- for residuals()."""
    Y_hat, _ = run_full(params)
    return Y_hat


def residuals(params):
    try:
        Y_hat = simulate(params)
    except (AssertionError, RuntimeError):
        return np.full(N_FIT_WEEKS * len(tsm.STRAINS), 1e5)
    return (Y_hat - TARGET_HOSP_2021_22).ravel()


_BOUNDS = ([1e-3, 1e-3, 1e-3, 1e-4], [3.0, 3.0, 3.0, 1.0])
_X_SCALE = [0.3, 0.3, 0.3, 0.03]


def _x0_candidates():
    """Multi-start candidates (see fit_texas_chain.build_x0_candidates for
    the fuller rationale): a single warm start is not reliable once the
    model/VE changed -- the Hill-default-only x0 previously used here
    landed in a much worse local optimum (cost ~7.9e5 vs ~1.0e5) after the
    I_R/I_H/H_R/H_D restructuring and real season-specific alpha were
    wired in, purely from lack of starting-point diversity, not a
    structural problem with the model itself (confirmed: known-good old
    beta values plugged in by hand scored the same ~7.9e5, so *some*
    starting points are just stuck in a bad basin under the new dynamics)."""
    hill = [tsm.beta["H1"], tsm.beta["H3"], tsm.beta["B"]]
    candidates = [
        hill + [0.03],
        [b * 1.5 for b in hill] + [0.03],
        [b * 2.0 for b in hill] + [0.03],
        [b * 0.6 for b in hill] + [0.06],
        [b * 0.4 for b in hill] + [0.06],
        [0.3733, 0.3987, 0.3662, 0.0421],   # previously-fitted (pre-restructuring) values
    ]
    return [np.clip(c, _BOUNDS[0], _BOUNDS[1]) for c in candidates]


def fit(verbose=True):
    best_result = None
    for i, x0 in enumerate(_x0_candidates()):
        result = least_squares(residuals, x0, bounds=_BOUNDS, xtol=1e-12, ftol=1e-12, x_scale=_X_SCALE)
        if verbose:
            print(f"    candidate {i}: x0={np.round(x0, 4)} -> cost={result.cost:.4e} nfev={result.nfev}")
        if best_result is None or result.cost < best_result.cost:
            best_result = result
    return best_result


def plot_fit(Y_hat, out_path="texas_2021_22_hosp_fit.png"):
    dates = DATES_2021_22
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    colors = {"H1": "#1f77b4", "H3": "#d62728", "B": "#2ca02c"}

    def format_date_axis(ax):
        ax.set_xlabel(f"Week ending ({N_FIT_WEEKS} reported weeks; Aug 2021-Jan 2022 gap not reported)")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    ax = axes[0]
    for j, m in enumerate(tsm.STRAINS):
        ax.plot(dates, TARGET_HOSP_2021_22[:, j], color=colors[m], linestyle="--",
                 linewidth=1.2, marker="x", markersize=4, label=f"{m} observed")
        ax.plot(dates, Y_hat[:, j], color=colors[m], linewidth=2.0, label=f"{m} fitted")
    format_date_axis(ax)
    ax.set_ylabel("Weekly incident hospitalizations (count)")
    ax.set_title("By subtype")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(dates, TARGET_HOSP_2021_22.sum(axis=1), color="#333333", linestyle="--",
             linewidth=1.5, marker="x", markersize=4, label="Observed (all subtypes)")
    ax.plot(dates, Y_hat.sum(axis=1), color="#d62728", linewidth=2.2, label="Simulated (all subtypes)")
    format_date_axis(ax)
    ax.set_ylabel("Weekly incident hospitalizations (count)")
    ax.set_title("Total")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.suptitle("Observed vs. Simulated Flu Hospitalizations -- Texas, 2021-22 Season")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    result = fit()
    beta_h1, beta_h3, beta_b, epsilon_H = result.x
    print(f"success={result.success}  cost={result.cost:.6e}")
    print(f"beta_H1={beta_h1:.4f}  beta_H3={beta_h3:.4f}  beta_B={beta_b:.4f}  "
          f"epsilon_H={epsilon_H:.4f} (ascertainment fraction)")

    Y_hat = simulate(result.x)
    plot_fit(Y_hat)
