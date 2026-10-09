"""
Pilot: cheaper per-state calibration for scaling the forecast pipeline
beyond Texas (see MODEL.md / conversation plan). Instead of a full
independent 5-parameter multi-start fit per state per season (what
fit_texas_chain.py does), this holds beta_H1/beta_H3/beta_B FIXED at the
already-computed NATIONAL fitted values for each season (MODEL.md SS6.4)
and solves only for that state's own epsilon_H (season 1) or
(epsilon_H, t0) (seasons 2+) -- a 1-D or 2-D least-squares problem instead
of 5-D multi-start, since epsilon_H enters Y_hat linearly and beta is no
longer free. This is the parameter that's actually been found to vary by
geography (Texas's own epsilon_H differed from national's by ~25-30% at
the same season); beta itself was already shown to be similar between
Texas and national (MODEL.md SS6.5).

Usage: calibrate_region("California") or calibrate_region("Vermont").
"""
import datetime as dt

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import data_prep as dp
import three_strain_model as tsm

INIT_SEASON_START_YEAR = 2020
FIT_SEASON_START_YEAR = 2021
SEASON_START_YEARS = [2022, 2023, 2024, 2025]
ADULT_VE_2020_21_CDC = 0.521   # same fixed CDC national adult estimate used for the national fit

# National fitted beta per season (current model, post subtype-specific
# p_H -- MODEL.md SS6.4), held FIXED and shared across every state.
NATIONAL_BETA_BY_SEASON = {
    2021: (0.3728, 0.4049, 0.3918),
    2022: (0.4964, 0.4990, 0.4325),
    2023: (0.4324, 0.4132, 0.4217),
    2024: (0.4728, 0.4223, 0.3901),
    2025: (0.4386, 0.4346, 0.3925),
}


def compute_v_2020_21(region):
    """Population-weighted end-of-2020-21 vaccination coverage for
    `region`: real state children's series blended with the fixed
    national CDC adult estimate (same method as fit_national_hosp_2021,
    generalized to any state)."""
    _, child_cum = dp.load_children_vax_weekly(INIT_SEASON_START_YEAR, region=region)
    child_final = child_cum[-1] / 100.0
    child_share, adult_share = dp.get_pop_shares(region)
    return child_share * child_final + adult_share * ADULT_VE_2020_21_CDC


def build_initial_state_season1(region, epsilon_H, state_pop, v_2020_21):
    beta_h1, beta_h3, beta_b = NATIONAL_BETA_BY_SEASON[FIT_SEASON_START_YEAR]
    tsm.set_betas(beta_h1, beta_h3, beta_b)

    _, target_2020_21 = dp.build_weekly_hosp_target(INIT_SEASON_START_YEAR, hosp_location=STATE_FIPS[region],
                                                      subtype_region=region)
    cumulative_hosp = target_2020_21.sum(axis=0)
    AR = cumulative_hosp / (epsilon_H * tsm.P_H_VEC)
    if np.any(AR < 0) or AR.sum() >= state_pop:
        raise RuntimeError(f"epsilon_H={epsilon_H:.6g} infeasible for {region} 2020-21 "
                            f"(AR.sum()={AR.sum():.3e} >= pop={state_pop})")

    idx = {h: i for i, h in enumerate(tsm.H_GROUPS)}
    S = np.zeros((2, tsm.N_H))
    E = np.zeros((2, tsm.N_M))
    I_R = np.zeros((2, tsm.N_M))
    I_H = np.zeros((2, tsm.N_M))
    H_R = np.zeros((2, tsm.N_M))
    H_D = np.zeros((2, tsm.N_M))
    R = np.zeros((2, tsm.N_M))
    D = np.zeros((2, tsm.N_M))

    uninfected = state_pop - AR.sum()
    S[0, idx["N"]] = uninfected * (1 - v_2020_21)
    S[0, idx["V"]] = uninfected * v_2020_21
    for j, m in enumerate(tsm.STRAINS):
        S[0, idx[m]] = AR[j] * (1 - v_2020_21)
        S[0, idx[m + "V"]] = AR[j] * v_2020_21

    return tsm.seed_infections(S, E, I_R, I_H, H_R, H_D, R, D)


# Per-state 2020 Census FIPS codes needed for time-series.csv's `location`
# column (the hospitalization file is keyed by FIPS, not state name).
STATE_FIPS = {
    "Alabama": "01", "Alaska": "02", "Arizona": "04", "Arkansas": "05", "California": "06",
    "Colorado": "08", "Connecticut": "09", "Delaware": "10", "District of Columbia": "11",
    "Florida": "12", "Georgia": "13", "Hawaii": "15", "Idaho": "16", "Illinois": "17",
    "Indiana": "18", "Iowa": "19", "Kansas": "20", "Kentucky": "21", "Louisiana": "22",
    "Maine": "23", "Maryland": "24", "Massachusetts": "25", "Michigan": "26", "Minnesota": "27",
    "Mississippi": "28", "Missouri": "29", "Montana": "30", "Nebraska": "31", "Nevada": "32",
    "New Hampshire": "33", "New Jersey": "34", "New Mexico": "35", "New York": "36",
    "North Carolina": "37", "North Dakota": "38", "Ohio": "39", "Oklahoma": "40", "Oregon": "41",
    "Pennsylvania": "42", "Rhode Island": "44", "South Carolina": "45", "South Dakota": "46",
    "Tennessee": "47", "Texas": "48", "Utah": "49", "Vermont": "50", "Virginia": "51",
    "Washington": "53", "West Virginia": "54", "Wisconsin": "55", "Wyoming": "56",
}

# Delaware's hospitalization series (time-series.csv) only starts 2022-12-17
# -- zero rows for the entire 2020-21 (season-1 init) and 2021-22 (first
# calibrated season) periods, confirmed by a pre-flight scan of all 51
# states/DC. Every other state had >=20 reported weeks in every season
# 2020-21 through 2025-26. Excluded from the batch run for this reason.
STATES_WITH_DATA_GAPS = {"Delaware": "no hospitalization data before 2022-12-17 "
                                      "(missing 2020-21 and 2021-22 seasons entirely)"}


def compute_fit_metrics(Y_hat, target):
    """Raw and population-size-normalized fit-quality metrics for one
    season. Raw MSE scales with a state's population (Texas's will dwarf
    Vermont's), so cross-state comparison needs the relative (RMSE /
    mean observed) version; reported at both the total and per-subtype
    level since the total can fit well while the subtype split is off
    (seen for Vermont 2024-25 -- beta is shared/fixed nationally, so a
    state whose real subtype mix diverges from the national pattern can't
    locally re-weight it)."""
    total_obs = target.sum(axis=1)
    total_pred = Y_hat.sum(axis=1)
    mse_total = float(np.mean((total_pred - total_obs) ** 2))
    rmse_total = float(np.sqrt(mse_total))
    mean_obs_total = float(total_obs.mean())
    rel_rmse_total = rmse_total / mean_obs_total if mean_obs_total > 1e-9 else float("nan")

    out = {"mse_total": mse_total, "rmse_total": rmse_total, "rel_rmse_total": rel_rmse_total}
    for j, m in enumerate(tsm.STRAINS):
        mse_m = float(np.mean((Y_hat[:, j] - target[:, j]) ** 2))
        rmse_m = float(np.sqrt(mse_m))
        mean_m = float(target[:, j].mean())
        out[f"rel_rmse_{m}"] = rmse_m / mean_m if mean_m > 1e-9 else float("nan")
    return out


def run_one_season(season_start_year, initial_state_unseeded, params, region):
    beta_h1, beta_h3, beta_b = NATIONAL_BETA_BY_SEASON[season_start_year]
    epsilon_H, t0 = params
    tsm.set_betas(beta_h1, beta_h3, beta_b)

    anchor = dt.date(season_start_year, 10, 1)
    day_grid = np.arange(0, tsm.SEASON_DAYS + 1)
    dates, target = dp.build_weekly_hosp_target(season_start_year, hosp_location=STATE_FIPS[region],
                                                  subtype_region=region)

    season_alpha = dp.build_season_alpha(season_start_year, region)
    tsm.set_alpha(season_alpha["H1"], season_alpha["H3"], season_alpha["B"])

    overall_cum = dp.build_overall_vax_daily_from_master(season_start_year, day_grid, region)
    mu_daily = dp.hazard_from_daily_cum(overall_cum)

    def mu_eff(t):
        return 0.0 if t < 0 else np.interp(t, day_grid, mu_daily)

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


def calibrate_season(season_start_year, initial_state_unseeded, region, t0_candidates=(0.0, -20.0, 20.0)):
    """Solve for (epsilon_H, t0) only -- beta fixed at the national value
    for this season. Small multi-start over t0 starting points (beta is
    no longer free, so this should be a much better-behaved landscape
    than the full 5-D search -- that's exactly what this pilot tests)."""
    bounds = ([1e-4, -30.0], [1.0, 30.0])

    def residuals(params):
        try:
            Y_hat, _, dates, target = run_one_season(season_start_year, initial_state_unseeded, params, region)
        except (AssertionError, RuntimeError):
            return np.full(100, 1e5)
        return (Y_hat - target).ravel()

    best = None
    for t0_x0 in t0_candidates:
        result = least_squares(residuals, [0.05, t0_x0], bounds=bounds, xtol=1e-12, ftol=1e-12,
                                x_scale=[0.03, 10.0])
        if best is None or result.cost < best.cost:
            best = result
    Y_hat, final_state, dates, target = run_one_season(season_start_year, initial_state_unseeded, best.x, region)
    return best, Y_hat, final_state, dates, target


def plot_fit(region, season_start_year, dates, target, Y_hat, out_path):
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
        ax.plot(dates, target[:, j], color=colors[m], linestyle="--", linewidth=1.2,
                 marker="x", markersize=4, label=f"{m} observed")
        ax.plot(dates, Y_hat[:, j], color=colors[m], linewidth=2.0, label=f"{m} fitted")
    format_date_axis(ax)
    ax.set_ylabel("Weekly incident hospitalizations (count)")
    ax.set_title("By subtype")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(dates, target.sum(axis=1), color="#333333", linestyle="--", linewidth=1.5,
             marker="x", markersize=4, label="Observed (all subtypes)")
    ax.plot(dates, Y_hat.sum(axis=1), color="#d62728", linewidth=2.2, label="Simulated (all subtypes)")
    format_date_axis(ax)
    ax.set_ylabel("Weekly incident hospitalizations (count)")
    ax.set_title("Total")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Observed vs. Simulated (cheap calibration) -- {region}, {season_label} Season")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"  Saved {out_path}")


def calibrate_region(region, make_plots=True, return_final_state=False):
    print(f"=== {region} ===")
    import csv
    with open("/Users/boyapeng/Desktop/Dissertation/Aim2/Data/Flu/state_population_2020_2025.csv") as f:
        pop_table = {row["state"]: row for row in csv.DictReader(f)}
    state_pop_2021 = float(pop_table[region]["2021"])
    tsm.set_population(state_pop_2021)

    v_2020_21 = compute_v_2020_21(region)
    print(f"  V_2020_21 = {v_2020_21:.4f}")

    # Season 1: 1-D solve for epsilon_H only (beta fixed at national 2021-22 value)
    def residuals_season1(x):
        epsilon_H = x[0]
        try:
            initial = build_initial_state_season1(region, epsilon_H, state_pop_2021, v_2020_21)
            beta_h1, beta_h3, beta_b = NATIONAL_BETA_BY_SEASON[FIT_SEASON_START_YEAR]
            tsm.set_betas(beta_h1, beta_h3, beta_b)
            alpha = dp.build_season_alpha(FIT_SEASON_START_YEAR, region)
            tsm.set_alpha(alpha["H1"], alpha["H3"], alpha["B"])
            day_grid = np.arange(0, tsm.SEASON_DAYS + 1)
            overall_cum = dp.build_overall_vax_daily_from_master(FIT_SEASON_START_YEAR, day_grid, region)
            mu_daily = dp.hazard_from_daily_cum(overall_cum)
            mu_func = lambda t: np.interp(t, day_grid, mu_daily)
            dates, target = dp.build_weekly_hosp_target(FIT_SEASON_START_YEAR, hosp_location=STATE_FIPS[region],
                                                          subtype_region=region)
            anchor = dt.date(FIT_SEASON_START_YEAR, 10, 1)
            t_grid, S_traj, E_traj, I_R_traj, I_H_traj, H_R_traj, H_D_traj, R_traj, D_traj = tsm.run_season(
                initial=initial, mu_func=mu_func, days=tsm.SEASON_DAYS)
            tsm.check_positivity(S_traj, E_traj, I_R_traj, I_H_traj, H_R_traj, H_D_traj, R_traj, D_traj)
            daily_hosp = tsm.compute_daily_hosp_incidence(I_H_traj)
            weekly = dp.aggregate_at_dates(daily_hosp, t_grid, dates, anchor)
            Y_hat = epsilon_H * weekly
            return (Y_hat - target).ravel()
        except (AssertionError, RuntimeError):
            return np.full(100, 1e5)

    result1 = least_squares(residuals_season1, [0.05], bounds=([1e-4], [1.0]), xtol=1e-12, ftol=1e-12, x_scale=[0.03])
    epsilon_H_2021 = result1.x[0]
    print(f"  2021-22: epsilon_H={epsilon_H_2021:.4f}  cost={result1.cost:.4e}")

    initial = build_initial_state_season1(region, epsilon_H_2021, state_pop_2021, v_2020_21)
    beta_h1, beta_h3, beta_b = NATIONAL_BETA_BY_SEASON[FIT_SEASON_START_YEAR]
    tsm.set_betas(beta_h1, beta_h3, beta_b)
    alpha = dp.build_season_alpha(FIT_SEASON_START_YEAR, region)
    tsm.set_alpha(alpha["H1"], alpha["H3"], alpha["B"])
    day_grid = np.arange(0, tsm.SEASON_DAYS + 1)
    overall_cum = dp.build_overall_vax_daily_from_master(FIT_SEASON_START_YEAR, day_grid, region)
    mu_daily = dp.hazard_from_daily_cum(overall_cum)
    mu_func = lambda t: np.interp(t, day_grid, mu_daily)
    dates, target = dp.build_weekly_hosp_target(FIT_SEASON_START_YEAR, hosp_location=STATE_FIPS[region],
                                                  subtype_region=region)
    anchor = dt.date(FIT_SEASON_START_YEAR, 10, 1)
    t_grid, S_traj, E_traj, I_R_traj, I_H_traj, H_R_traj, H_D_traj, R_traj, D_traj = tsm.run_season(
        initial=initial, mu_func=mu_func, days=tsm.SEASON_DAYS)
    daily_hosp = tsm.compute_daily_hosp_incidence(I_H_traj)
    weekly = dp.aggregate_at_dates(daily_hosp, t_grid, dates, anchor)
    Y_hat = epsilon_H_2021 * weekly
    final_state = (S_traj[-1], E_traj[-1], I_R_traj[-1], I_H_traj[-1],
                    H_R_traj[-1], H_D_traj[-1], R_traj[-1], D_traj[-1])
    if make_plots:
        plot_fit(region, FIT_SEASON_START_YEAR, dates, target, Y_hat,
                  f"{region.lower().replace(' ', '_')}_2021_22_hosp_fit.png")

    metrics1 = compute_fit_metrics(Y_hat, target)
    summary = [{"season": "2021-22", "epsilon_H": epsilon_H_2021, "t0": 0.0,
                "cost": result1.cost, **metrics1}]
    prev_year = FIT_SEASON_START_YEAR
    for season_start_year in SEASON_START_YEARS:
        remapped = tsm.end_of_season_remap(*final_state, reseed=False)
        S0, E0, I_R0, I_H0, H_R0, H_D0, R0, D0 = remapped
        pop_this = float(pop_table[region][str(season_start_year)])
        pop_prev = float(pop_table[region][str(prev_year)])
        delta = pop_this - pop_prev
        idx_N = tsm.H_GROUPS.index("N")
        S0 = S0.copy()
        S0[0, idx_N] += max(delta, 0)
        tsm.set_population(tsm.N_TOTAL + delta)
        initial_state_unseeded = (S0, E0, I_R0, I_H0, H_R0, H_D0, R0, D0)

        result, Y_hat, final_state, dates, target = calibrate_season(
            season_start_year, initial_state_unseeded, region)
        season_label = f"{season_start_year}-{(season_start_year + 1) % 100:02d}"
        metrics = compute_fit_metrics(Y_hat, target)
        print(f"  {season_label}: epsilon_H={result.x[0]:.4f} t0={result.x[1]:+.1f}d "
              f"cost={result.cost:.4e} rel_rmse={metrics['rel_rmse_total']:.3f} N_TOTAL={tsm.N_TOTAL:,.0f}")
        if make_plots:
            plot_fit(region, season_start_year, dates, target, Y_hat,
                      f"{region.lower().replace(' ', '_')}_{season_start_year}_{(season_start_year + 1) % 100:02d}_hosp_fit.png")
        summary.append({"season": season_label, "epsilon_H": result.x[0], "t0": result.x[1],
                         "cost": result.cost, **metrics})
        prev_year = season_start_year

    print(f"\n{region} summary:")
    print(f"{'Season':<10}{'epsilon_H':>12}{'t0':>8}{'cost':>14}{'rel_rmse':>10}")
    for row in summary:
        print(f"{row['season']:<10}{row['epsilon_H']:>12.4f}{row['t0']:>+8.1f}"
              f"{row['cost']:>14.4e}{row['rel_rmse_total']:>10.3f}")
    for row in summary:
        row["region"] = region
    if return_final_state:
        return summary, final_state
    return summary


def _state_slug(region):
    return region.lower().replace(" ", "_")


def run_all_states(out_csv="state_calibration_mse_report.csv", skip_regions=None, make_plots=False):
    """Batch-run calibrate_region over every state with usable hospitalization
    data (see STATES_WITH_DATA_GAPS), write one CSV row per state x season
    with epsilon_H, t0, cost, and the MSE/RMSE metrics, and print a summary
    highlighting the worst-fitting states (both overall and subtype-split).

    Each state runs in its OWN subprocess (not a shared loop in this
    process): a 50-state single-process run was killed by the OS for low
    memory partway through (got through 12 states); a fresh subprocess per
    state guarantees memory is fully released between states and, as a
    bonus, one state's crash can't take down the rest of the batch."""
    import json
    import os
    import subprocess
    import sys

    skip_regions = set(skip_regions or [])
    regions = [r for r in STATE_FIPS if r not in STATES_WITH_DATA_GAPS and r not in skip_regions]
    print(f"Running {len(regions)} states (skipping {sorted(set(STATES_WITH_DATA_GAPS) | skip_regions)})\n")

    all_rows = []
    failures = {}
    for i, region in enumerate(regions, 1):
        json_path = f"state_fit_result_{_state_slug(region)}.json"
        if os.path.exists(json_path):
            print(f"[{i}/{len(regions)}] {region} -- already done, reusing {json_path}", flush=True)
            with open(json_path) as f:
                all_rows.extend(json.load(f))
            continue
        print(f"[{i}/{len(regions)}] {region}", flush=True)
        proc = subprocess.run([sys.executable, __file__, region], capture_output=True, text=True)
        print(proc.stdout[-2000:])
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode}): {proc.stderr.strip().splitlines()[-1] if proc.stderr else ''}")
            failures[region] = proc.stderr.strip().splitlines()[-1] if proc.stderr else f"exit {proc.returncode}"
            continue
        with open(json_path) as f:
            summary = json.load(f)
        all_rows.extend(summary)

    fieldnames = ["region", "season", "epsilon_H", "t0", "cost", "mse_total", "rmse_total",
                  "rel_rmse_total", "rel_rmse_H1", "rel_rmse_H3", "rel_rmse_B"]
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row[k] for k in fieldnames})
    print(f"Saved {len(all_rows)} rows to {out_csv}")

    if failures:
        print(f"\n{len(failures)} state(s) failed and were excluded from the CSV:")
        for region, err in failures.items():
            print(f"  {region}: {err}")

    by_region_rel_rmse = {}
    for row in all_rows:
        by_region_rel_rmse.setdefault(row["region"], []).append(row["rel_rmse_total"])
    ranked = sorted(by_region_rel_rmse.items(), key=lambda kv: np.nanmean(kv[1]))
    print(f"\nMean relative RMSE (total) across seasons, best to worst:")
    for region, vals in ranked:
        print(f"  {region:<22}{np.nanmean(vals):.3f}")

    return all_rows, failures


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        run_all_states()
    else:
        region = sys.argv[1] if len(sys.argv) > 1 else "California"
        summary = calibrate_region(region, make_plots=False)
        import json
        with open(f"state_fit_result_{_state_slug(region)}.json", "w") as f:
            json.dump(summary, f)
