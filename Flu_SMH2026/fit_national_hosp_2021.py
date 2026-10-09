"""
National counterpart to fit_texas_hosp_2021.py: fit the three-strain
model's beta_H1, beta_H3, beta_B (and a hospitalization
ascertainment/scale parameter epsilon_H) to US national 2021-22 season flu
hospitalizations, subtype-split via the national NREVSS weekly file
(data_prep.load_national_subtype_shares) -- NOT the state-level Public
Health Labs file used by the Texas fit, which has no National rows at all.

National vs. Texas data quality (see MODEL.md National-fitting audit):
national hospitalization reporting is MORE complete than Texas for this
season -- one 35-day gap (Oct 9-Nov 13, 2021) instead of Texas's much
larger Aug 2021-Jan 2022 hole, so this fit uses far more of the season's
real weekly data (48 reported weeks vs. Texas's 34).

Model population scale: N_TOTAL is set to NATIONAL_POP (a real headcount,
via tsm.set_population), exactly as fit_texas_hosp_2021.py does for
TX_POP -- every compartment (S,E,I_R,I_H,H_R,H_D,R,D) is in actual numbers
of people, directly comparable to the raw hospitalization counts.

Initial state (start of 2021-22 season) is estimated from the 2020-21
season's observed NATIONAL hospitalizations, same AR = cumulative_hosp /
(epsilon_H * p_H) method as the Texas fit. The previous-season (2020-21)
end-of-season vaccination coverage V_2020_21 is population-weighted
(national child/adult shares, data_prep.get_pop_shares("United States")):
- Children 6mo-17y: REAL national end-of-season 2020-21 coverage, 58.38%
  (unlike Texas, which had no real 2020-21 series to use either way at
  the time that fit was built) -- data_prep.load_children_vax_weekly(2020,
  region="National").
- Adults 18+: 52.1%, a fixed CDC national estimate (the weekly adult
  series, cdc_sw5n-wg2p, doesn't start until 2021-22 nationally either --
  same gap as Texas, just no weekly series to fall back on for this one
  season).
Blended: V_2020_21 = 0.2174*58.38% + 0.7826*52.1% = 53.47%.

Current-season (2021-22) vaccination mu_y(t): population-weighted
(national Census age shares) blend of real weekly NATIONAL series from
both judz-8etw and sw5n-wg2p (data_prep.build_overall_vax_daily_from_master,
region="United States") -- same method as the Texas fit, just national
inputs throughout.

Season-specific VE (alpha): data_prep.build_season_alpha("United States"),
same CDC workbook as the Texas fit -- the VE estimates were already
national to begin with (CDC VE studies are never state-specific), so this
removes a real geography mismatch the Texas fit had (applying national VE
to a Texas-specific model).

Calibrated parameters: beta_H1, beta_H3, beta_B, epsilon_H (4 total).
"""

import datetime as dt

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import data_prep as dp
import three_strain_model as tsm

NATION = "United States"
ADULT_VE_2020_21_CDC = 0.521   # fixed, CDC national adult coverage estimate, 2020-21 season
NATIONAL_POP = 332_454_000     # FRED POPTHM, July 2021 (Census Bureau Vintage 2021 US estimate)
INIT_SEASON_START_YEAR = 2020
FIT_SEASON_START_YEAR = 2021
ANCHOR = dt.date(FIT_SEASON_START_YEAR, 10, 1)


def _compute_v_2020_21():
    """Population-weighted national end-of-2020-21-season vaccination
    coverage: real children's series blended with the fixed CDC adult
    estimate (see module docstring). Note: the raw CDC vaccination files
    use geographic_name="National", not "United States" (the convention
    get_pop_shares/build_season_alpha/the master vax file use) -- same
    naming mismatch documented in build_state_weekly_vax.py's
    VAX_REGION_OVERRIDE."""
    child_dates, child_cum = dp.load_children_vax_weekly(INIT_SEASON_START_YEAR, region="National")
    child_final = child_cum[-1] / 100.0
    child_share, adult_share = dp.get_pop_shares(NATION)
    return child_share * child_final + adult_share * ADULT_VE_2020_21_CDC


PREVIOUS_SEASON_VAX = _compute_v_2020_21()   # ~0.5347

tsm.set_population(NATIONAL_POP)   # model compartments now in headcounts, not fractions
_ALPHA_2021_22 = dp.build_season_alpha(FIT_SEASON_START_YEAR, region=NATION)
tsm.set_alpha(_ALPHA_2021_22["H1"], _ALPHA_2021_22["H3"], _ALPHA_2021_22["B"])

DAY_GRID = np.arange(0, tsm.SEASON_DAYS + 1)
DATES_2021_22, TARGET_HOSP_2021_22 = dp.build_weekly_hosp_target(
    FIT_SEASON_START_YEAR, hosp_location="US", subtype_region=None,
    shares_fn=dp.load_national_subtype_shares)
N_FIT_WEEKS = len(DATES_2021_22)


def build_initial_state_from_2020_21(epsilon_H):
    """Empirical analogue of Algorithm 10, driven by observed national
    2020-21 hospitalization data instead of simulated E/I_R/I_H/H_R/H_D/R.
    All quantities here are headcounts (N_TOTAL=NATIONAL_POP), not
    fractions. See fit_texas_hosp_2021.build_initial_state_from_2020_21
    for the fuller rationale -- identical method, national inputs."""
    _, target_2020_21 = dp.build_weekly_hosp_target(
        INIT_SEASON_START_YEAR, hosp_location="US", subtype_region=None,
        shares_fn=dp.load_national_subtype_shares)
    cumulative_hosp = target_2020_21.sum(axis=0)   # (n_m,) raw counts

    AR = cumulative_hosp / (epsilon_H * tsm.P_H_VEC)     # headcount estimate of true infections, per strain's own IHR
    if np.any(AR < 0) or AR.sum() >= NATIONAL_POP:
        raise RuntimeError(
            f"epsilon_H={epsilon_H:.6g} implies an infeasible 2020-21 attack "
            f"rate (AR.sum()={AR.sum():.3e} >= NATIONAL_POP={NATIONAL_POP})"
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
    uninfected = NATIONAL_POP - AR.sum()
    S[0, idx["N"]] = uninfected * (1 - V)
    S[0, idx["V"]] = uninfected * V
    for j, m in enumerate(tsm.STRAINS):
        S[0, idx[m]] = AR[j] * (1 - V)
        S[0, idx[m + "V"]] = AR[j] * V

    return tsm.seed_infections(S, E, I_R, I_H, H_R, H_D, R, D)


def build_mu_func_2021_22():
    overall_cum = dp.build_overall_vax_daily_from_master(FIT_SEASON_START_YEAR, DAY_GRID, region=NATION)
    mu_daily = dp.hazard_from_daily_cum(overall_cum)
    return lambda t: np.interp(t, DAY_GRID, mu_daily)


MU_2021_22 = build_mu_func_2021_22()


def run_full(params):
    """Run the 2021-22 season forward for a trial parameter vector. Returns
    (Y_hat aligned to DATES_2021_22, final-day compartments) -- the latter
    for chaining into 2022-23 via tsm.end_of_season_remap (see a future
    fit_national_chain.py, mirroring fit_texas_chain.py)."""
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
    """Multi-start candidates (see fit_texas_chain.build_x0_candidates and
    fit_texas_hosp_2021._x0_candidates for the fuller rationale) -- a
    single warm start proved unreliable for the Texas fit under the
    current model, so this starts multi-start from the outset rather than
    waiting to discover the same problem nationally."""
    hill = [tsm.beta["H1"], tsm.beta["H3"], tsm.beta["B"]]
    candidates = [
        hill + [0.03],
        [b * 1.5 for b in hill] + [0.03],
        [b * 2.0 for b in hill] + [0.03],
        [b * 0.6 for b in hill] + [0.06],
        [b * 0.4 for b in hill] + [0.06],
        [0.3733, 0.3987, 0.3662, 0.0421],   # Texas's fitted 2021-22 values, as a diverse extra candidate
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


def plot_fit(Y_hat, out_path="national_2021_22_hosp_fit.png"):
    dates = DATES_2021_22
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    colors = {"H1": "#1f77b4", "H3": "#d62728", "B": "#2ca02c"}

    def format_date_axis(ax):
        ax.set_xlabel(f"Week ending ({N_FIT_WEEKS} reported weeks; Oct 9-Nov 13, 2021 gap not reported)")
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

    fig.suptitle("Observed vs. Simulated Flu Hospitalizations -- United States, 2021-22 Season")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    print(f"V_2020_21 (national, blended) = {PREVIOUS_SEASON_VAX:.4f}")
    result = fit()
    beta_h1, beta_h3, beta_b, epsilon_H = result.x
    print(f"success={result.success}  cost={result.cost:.6e}")
    print(f"beta_H1={beta_h1:.4f}  beta_H3={beta_h3:.4f}  beta_B={beta_b:.4f}  "
          f"epsilon_H={epsilon_H:.4f} (ascertainment fraction)")

    Y_hat = simulate(result.x)
    plot_fit(Y_hat)
