"""
2026-27 season-ahead forecast, generalized from the Texas pipeline
(save_texas_2025_26_final_state.py / build_texas_2026_27_initial_state.py /
forecast_texas_2026_27.py / plot_texas_2026_27_forecast.py) to any state.

Per state, this:
  1. Re-runs calibrate_state_chain.calibrate_region(region) (the cheap
     national-beta / state-epsilon_H chain calibration already validated
     across all 50 states -- state_calibration_mse_report.csv) to get that
     state's 2025-26 end-of-season compartment state AND its own fitted
     2025-26 epsilon_H. Re-running (rather than caching) costs about a
     minute per state but needs no new serialization machinery and is
     cheap relative to the historical multi-start Texas-specific fit.
  2. Remaps that state into an unseeded 2026-27 initial state (Algorithm
     10), holding 2025->2026 population flat -- no state-level 2026
     Census projection exists yet (same simplification used for Texas,
     where 2026 was also held flat at 2025's value).
  3. Runs a 300-draw Monte Carlo ensemble: beta_H1/H3/B and t0 sampled
     from the NATIONAL fitted distributions (shared across every state --
     MODEL.md SS6.6 + conversation for t0), epsilon_H FIXED at that
     state's own 2025-26 fitted value (the one parameter shown to vary by
     geography), VE (alpha) and vaccination curve (mu) held flat at that
     state's own 2025-26 values.
  4. Plots the ensemble (median + 95% CI, by-subtype and total) in the
     same two-panel style as texas_2026_27_forecast.png, saved to
     Projection/{state_slug}_2026_27_forecast.png.

Usage: python3 forecast_state_2026_27.py "California"
       python3 forecast_state_2026_27.py all   (resumable batch, all 50
       states except Delaware -- see calibrate_state_chain.STATES_WITH_DATA_GAPS)
"""
import datetime as dt
import json
import os

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import calibrate_state_chain as csc
import data_prep as dp
import three_strain_model as tsm

PROJECTION_DIR = "Projection"
SEASON_START_YEAR = 2026
ANCHOR = dt.date(SEASON_START_YEAR, 10, 1)
N_DRAWS = 300

# National fitted beta/t0 distributions (MODEL.md SS6.6; t0 from the
# conversation's national chain fit), shared across every state -- beta was
# already shown to be similar between Texas and national (MODEL.md SS6.5),
# and epsilon_H (not beta) is the parameter that's state-specific.
BETA_DIST = {"H1": (0.4426, 0.0469), "H3": (0.4348, 0.0375), "B": (0.4057, 0.0199)}
T0_DIST = (-14.875, 9.9356)
CLIP_BOUNDS = ([1e-3, 1e-3, 1e-3, 1e-4, -30.0], [3.0, 3.0, 3.0, 1.0, 30.0])


def _state_slug(region):
    return region.lower().replace(" ", "_")


def sample_params(rng, epsilon_H_fixed, max_tries=1000):
    """Rejection-resample (not clip) beta/t0 draws outside CLIP_BOUNDS --
    clipping would pile mass at the floor/ceiling, contaminating the
    ensemble's tails with a spurious pile-up (see forecast_texas_2026_27.py).
    epsilon_H is fixed, not sampled (see MODEL.md / conversation: it
    reflects reporting-network completeness, not season-to-season
    epidemiological variation)."""
    for _ in range(max_tries):
        beta_h1 = rng.normal(*BETA_DIST["H1"])
        beta_h3 = rng.normal(*BETA_DIST["H3"])
        beta_b = rng.normal(*BETA_DIST["B"])
        t0 = rng.normal(*T0_DIST)
        params = np.array([beta_h1, beta_h3, beta_b, epsilon_H_fixed, t0])
        if np.all(params >= CLIP_BOUNDS[0]) and np.all(params <= CLIP_BOUNDS[1]):
            return params
    raise RuntimeError("sample_params: exceeded max_tries without a valid draw")


def build_weekly_dates(anchor, n_weeks=52):
    return [anchor + dt.timedelta(days=7 * k) for k in range(1, n_weeks + 1)]


def build_region_initial_2026_27(region):
    """Returns (initial_state_unseeded, epsilon_H_2025_26). Re-runs the
    state's chain calibration (fast -- 1D/2D solves, beta fixed at
    national) to get its 2025-26 final compartment state and its own
    fitted 2025-26 epsilon_H, then applies Algorithm 10's between-season
    remap with population held flat into 2026-27."""
    summary, final_state = csc.calibrate_region(region, make_plots=False, return_final_state=True)
    epsilon_H_2025_26 = summary[-1]["epsilon_H"]
    assert summary[-1]["season"] == "2025-26"

    remapped = tsm.end_of_season_remap(*final_state, reseed=False)
    return remapped, epsilon_H_2025_26


def forecast_region(region, seed=20262027):
    initial_state_unseeded, epsilon_H_fixed = build_region_initial_2026_27(region)
    print(f"{region}: 2025-26 fitted epsilon_H={epsilon_H_fixed:.4f}, N_TOTAL={tsm.N_TOTAL:,.0f}")

    alpha_2026_27 = dp.build_season_alpha(2025, region=region)
    tsm.set_alpha(alpha_2026_27["H1"], alpha_2026_27["H3"], alpha_2026_27["B"])

    day_grid = np.arange(0, tsm.SEASON_DAYS + 1)
    overall_cum = dp.build_overall_vax_daily_from_master(2025, day_grid, region)
    mu_daily = dp.hazard_from_daily_cum(overall_cum)

    def mu_eff(t):
        return 0.0 if t < 0 else np.interp(t, day_grid, mu_daily)

    dates = build_weekly_dates(ANCHOR)
    rng = np.random.default_rng(seed=seed)
    n_weeks = len(dates)
    n_strains = len(tsm.STRAINS)
    ensemble_Y = np.zeros((N_DRAWS, n_weeks, n_strains))
    n_failed = 0

    for i in range(N_DRAWS):
        beta_h1, beta_h3, beta_b, epsilon_H, t0 = sample_params(rng, epsilon_H_fixed)
        tsm.set_betas(beta_h1, beta_h3, beta_b)
        try:
            seeded_initial = tsm.seed_infections(*initial_state_unseeded)
            (t_grid, S_traj, E_traj, I_R_traj, I_H_traj,
             H_R_traj, H_D_traj, R_traj, D_traj) = tsm.run_season(
                initial=seeded_initial, mu_func=mu_eff, days=tsm.SEASON_DAYS, t_start=t0)
            tsm.check_positivity(S_traj, E_traj, I_R_traj, I_H_traj, H_R_traj, H_D_traj, R_traj, D_traj)
            daily_hosp = tsm.compute_daily_hosp_incidence(I_H_traj)
            weekly_hosp_headcount = dp.aggregate_at_dates(daily_hosp, t_grid, dates, ANCHOR)
            ensemble_Y[i] = epsilon_H * weekly_hosp_headcount
        except (AssertionError, RuntimeError):
            n_failed += 1
            ensemble_Y[i] = np.nan

    total_per_draw = np.nansum(ensemble_Y, axis=(1, 2))
    peak_week_total = np.nanmax(ensemble_Y.sum(axis=2), axis=1)
    result = {
        "region": region,
        "epsilon_H_2025_26": float(epsilon_H_fixed),
        "n_failed": int(n_failed),
        "season_total_median": float(np.nanmedian(total_per_draw)),
        "season_total_lo": float(np.nanpercentile(total_per_draw, 2.5)),
        "season_total_hi": float(np.nanpercentile(total_per_draw, 97.5)),
        "peak_week_median": float(np.nanmedian(peak_week_total)),
        "peak_week_lo": float(np.nanpercentile(peak_week_total, 2.5)),
        "peak_week_hi": float(np.nanpercentile(peak_week_total, 97.5)),
    }
    print(f"{region}: season-total median={result['season_total_median']:,.0f} "
          f"[{result['season_total_lo']:,.0f}, {result['season_total_hi']:,.0f}], "
          f"peak-week median={result['peak_week_median']:,.0f} "
          f"[{result['peak_week_lo']:,.0f}, {result['peak_week_hi']:,.0f}]")

    plot_region_forecast(region, dates, ensemble_Y)
    return result


def plot_region_forecast(region, dates, ensemble_Y):
    n_weeks = len(dates)
    strains = tsm.STRAINS
    median_by_strain = np.nanmedian(ensemble_Y, axis=0)
    lo_by_strain = np.nanpercentile(ensemble_Y, 2.5, axis=0)
    hi_by_strain = np.nanpercentile(ensemble_Y, 97.5, axis=0)

    total = ensemble_Y.sum(axis=2)
    median_total = np.nanmedian(total, axis=0)
    lo_total = np.nanpercentile(total, 2.5, axis=0)
    hi_total = np.nanpercentile(total, 97.5, axis=0)

    colors = {"H1": "#1f77b4", "H3": "#d62728", "B": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    def format_date_axis(ax):
        ax.set_xlabel(f"Week ending ({n_weeks}-week forecast; no real 2026-27 data exists yet)")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    ax = axes[0]
    for j, m in enumerate(strains):
        ax.plot(dates, median_by_strain[:, j], color=colors[m], linewidth=2.0, label=f"{m} median")
        ax.fill_between(dates, lo_by_strain[:, j], hi_by_strain[:, j],
                         color=colors[m], alpha=0.2, linewidth=0, label=f"{m} 95% CI")
    format_date_axis(ax)
    ax.set_ylabel("Weekly incident hospitalizations (count)")
    ax.set_title("By subtype")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(dates, median_total, color="#d62728", linewidth=2.2, label="Total median")
    ax.fill_between(dates, lo_total, hi_total, color="#d62728", alpha=0.2, linewidth=0, label="Total 95% CI")
    format_date_axis(ax)
    ax.set_ylabel("Weekly incident hospitalizations (count)")
    ax.set_title("Total")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Predicted Flu Hospitalizations -- {region}, 2026-27 Season "
                 "(300-draw Monte Carlo ensemble, median + 95% CI)")
    fig.tight_layout()
    out_path = os.path.join(PROJECTION_DIR, f"{_state_slug(region)}_2026_27_forecast.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def run_all_regions(out_csv="state_2026_27_forecast_summary.csv", skip_regions=None):
    """Resumable batch: one subprocess per state (same rationale as
    calibrate_state_chain.run_all_states -- avoids accumulating memory in
    one long process and survives being killed partway through)."""
    import csv
    import subprocess
    import sys

    skip_regions = set(skip_regions or [])
    regions = [r for r in csc.STATE_FIPS if r not in csc.STATES_WITH_DATA_GAPS and r not in skip_regions]
    print(f"Running {len(regions)} states (skipping {sorted(set(csc.STATES_WITH_DATA_GAPS) | skip_regions)})\n")

    all_rows = []
    failures = {}
    for i, region in enumerate(regions, 1):
        json_path = f"state_forecast_result_{_state_slug(region)}.json"
        png_path = os.path.join(PROJECTION_DIR, f"{_state_slug(region)}_2026_27_forecast.png")
        if os.path.exists(json_path) and os.path.exists(png_path):
            print(f"[{i}/{len(regions)}] {region} -- already done, reusing {json_path}", flush=True)
            with open(json_path) as f:
                all_rows.append(json.load(f))
            continue
        print(f"[{i}/{len(regions)}] {region}", flush=True)
        proc = subprocess.run([sys.executable, __file__, region], capture_output=True, text=True)
        print(proc.stdout[-1500:])
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode}): {proc.stderr.strip().splitlines()[-1] if proc.stderr else ''}")
            failures[region] = proc.stderr.strip().splitlines()[-1] if proc.stderr else f"exit {proc.returncode}"
            continue
        with open(json_path) as f:
            all_rows.append(json.load(f))

    fieldnames = ["region", "epsilon_H_2025_26", "n_failed", "season_total_median", "season_total_lo",
                  "season_total_hi", "peak_week_median", "peak_week_lo", "peak_week_hi"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row[k] for k in fieldnames})
    print(f"\nSaved {len(all_rows)} rows to {out_csv}")

    if failures:
        print(f"\n{len(failures)} state(s) failed:")
        for region, err in failures.items():
            print(f"  {region}: {err}")

    return all_rows, failures


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        run_all_regions()
    else:
        region = sys.argv[1] if len(sys.argv) > 1 else "Texas"
        result = forecast_region(region)
        with open(f"state_forecast_result_{_state_slug(region)}.json", "w") as f:
            json.dump(result, f)
