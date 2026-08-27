"""
national_scenario.py: Vaccination scenario comparison (A–E) × K=1–5 variants.

For each RD20 vaccination scenario:
  - Forward simulations use the scenario's vaccination curve.
  - K=1–5 variant trajectories pooled → single median + 95% CI per scenario.
Beta is calibrated once from Scenario A; backward envelope is shared (Scenario A).
Output: national_scenario.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, "/Users/boyapeng/Desktop/Dissertation/Aim2/epi_model")
os.chdir("/Users/boyapeng/Desktop/Dissertation/Aim2/epi_model")

import national_sto1 as base
import model_utils as mu
from variant_model2 import build_variant_share_series
from variant_model3 import simulate_variant_model_HL

# ── Output / data paths ────────────────────────────────────────────────────────
OUT_PNG  = os.path.join(base.REPO, "national_scenario5.png")
DATA_DIR = "/Users/boyapeng/Desktop/Dissertation/Aim2/Data/variant_pred_dist/new"

# ── Vaccination scenarios ──────────────────────────────────────────────────────
VAC_SCENARIOS = [
    dict(tag="A-2026-05-11", label="Scenario A", color="#1f77b4"),
    dict(tag="B-2026-05-11", label="Scenario B", color="#2ca02c"),
    dict(tag="C-2026-05-11", label="Scenario C", color="#ff7f0e"),
    dict(tag="D-2026-05-11", label="Scenario D", color="#d62728"),
    dict(tag="E-2026-05-11", label="Scenario E", color="#9467bd"),
]

# ── K scenario definitions ─────────────────────────────────────────────────────
K_SCENARIOS = [
    dict(K=1, csv=f"{DATA_DIR}/K=1_N=32.csv", letters=["A"]),
    dict(K=2, csv=f"{DATA_DIR}/K=2_N=61.csv", letters=["A","B"]),
    dict(K=3, csv=f"{DATA_DIR}/K=3_N=77.csv", letters=["A","B","C"]),
    dict(K=4, csv=f"{DATA_DIR}/K=4_N=74.csv", letters=["A","B","C","D"]),
    dict(K=5, csv=f"{DATA_DIR}/K=5_N=56.csv", letters=["A","B","C","D","E"]),
]

SHARE_COLS_ALL = [
    "XFG.1.1_share", "new_b_share", "new_c_share",
    "new_d_share", "new_e_share",
]

# ── sto2_9 fixed params (recalibrated for corrected ε₂ on IHR) ───────────────
FIXED_ALPHA       = 0.0480
FIXED_IC_MULT     = 3.1269
FIXED_WAN_D       = 180
FIXED_IHR_MULT    = 0.8141
FIXED_REINF_SCALE = 0.3406
FIXED_RE_PRO      = 0.10

ANCHOR_DATE     = pd.Timestamp("2026-05-10")
ANCHOR_SHARE    = 0.40
FWD_END         = pd.Timestamp("2027-06-05")
EPOCH_C_MIN     = pd.Timestamp("2026-03-14")
MIN_CROSS_SHARE = 0.01

SEEDS = np.arange(300)
K_SEED_OFFSETS = {1: 0, 2: 32, 3: 93, 4: 170, 5: 244}
ESCAPE_A        = 0.6265
ESCAPE_B_SHP    = 0.4521
XFG11_ESCAPE    = base.variant_escape("XFG.1.1")


# ── Helper functions ───────────────────────────────────────────────────────────
def _apply_sto8_params(params, HDR=None):
    p = dict(params)
    p["IHR_H"]  = base.IHR_H * FIXED_IHR_MULT
    p["IHR_L"]  = base.IHR_L * FIXED_IHR_MULT
    p["reinf"]  = base.REINF * FIXED_REINF_SCALE
    p["re_pro"] = FIXED_RE_PRO
    p["wan"]    = 1.0 / FIXED_WAN_D
    if HDR is not None:
        p["HDR"] = HDR
        # Hd0/Hr0 (dying- vs recovering-track hosp stock) are historical ICs
        # baked in under the old ~0.16 default HDR and never re-derived from
        # a fitted HDR, so a much lower HDR_fit leaves a stale, oversized Hd0
        # stock that produces a multi-week death-rate transient at SIM_START.
        # Re-split each group's total occupancy to be self-consistent with
        # HDR_fit; only affects weekly_death (Hd/Hr never feed weekly_hosp).
        for stem in ("Hd0", "Hd_B0"):
            hd_key, hr_key = stem, stem.replace("Hd", "Hr")
            for suffix in ("_H", "_L"):
                total = p.get(f"{hd_key}{suffix}", 0.0) + p.get(f"{hr_key}{suffix}", 0.0)
                p[f"{hd_key}{suffix}"] = HDR * total
                p[f"{hr_key}{suffix}"] = (1.0 - HDR) * total
    return p


def _find_anchor(series):
    peak = series.idxmax()
    return (series.loc[:peak] - ANCHOR_SHARE).abs().idxmin()


def _crossover_shifted(a_series, b_series, anchor_i, min_date):
    shift  = ANCHOR_DATE - anchor_i
    a_from = a_series.loc[anchor_i:]
    b_from = b_series.loc[anchor_i:]
    a_sh   = pd.Series(a_from.values, index=a_from.index + shift)
    b_sh   = pd.Series(b_from.values, index=b_from.index + shift)
    merged = pd.DataFrame({"a": a_sh, "b": b_sh}).dropna()
    over   = merged[(merged["b"] > merged["a"]) & (merged["b"] >= MIN_CROSS_SHARE)]
    raw    = over.index[0] if len(over) else merged.index[-1]
    return max(raw, min_date)


def _build_extended_marker_K1(variant_marker, ai_series, anchor_i):
    vm = variant_marker.copy()
    vm["Date"] = pd.to_datetime(vm["Date"])
    shift    = ANCHOR_DATE - anchor_i
    af       = ai_series.loc[anchor_i:]
    new_rows = pd.DataFrame({"Date": af.index + shift, "XFG.1.1_share": af.values})
    ext = pd.concat([vm, new_rows], ignore_index=True)
    ext = ext.drop_duplicates(subset=["Date"], keep="last")
    return ext.sort_values("Date").reset_index(drop=True)


def _build_extended_marker_Kn(variant_marker, series_list, anchor_i, share_cols):
    vm = variant_marker.copy()
    vm["Date"] = pd.to_datetime(vm["Date"])
    for col in share_cols[1:]:
        vm[col] = 0.0
    shift    = ANCHOR_DATE - anchor_i
    ref_from = series_list[0][1].loc[anchor_i:]
    new_rows = pd.DataFrame({"Date": ref_from.index + shift})
    for col_name, series in series_list:
        new_rows[col_name] = series.loc[anchor_i:].values
    ext = pd.concat([vm, new_rows], ignore_index=True)
    ext = ext.drop_duplicates(subset=["Date"], keep="last")
    ext = ext.sort_values("Date").reset_index(drop=True)
    for col in share_cols[1:]:
        ext[col] = ext[col].fillna(0.0)
    return ext


def _make_multi_epochs(escapes, crossover_dates, share_cols):
    ep_b_end = (crossover_dates[0] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    epochs = [
        {"start_date": "2025-04-27", "end_date": "2025-10-04",
         "a_share_column": "LP.8.1_share", "b_share_column": "XFG_share",
         "beta_multiplier_b": 1.0,
         "a_escape": base.variant_escape("LP.8.1"),
         "b_escape": base.variant_escape("XFG")},
        {"start_date": "2025-10-05", "end_date": ep_b_end,
         "a_share_column": "XFG_share", "b_share_column": "XFG.1.1_share",
         "beta_multiplier_b": 1.0,
         "a_escape": base.variant_escape("XFG"),
         "b_escape": base.variant_escape("XFG.1.1")},
    ]
    prev_escape = float(XFG11_ESCAPE)
    for j, (esc_j, xover_j) in enumerate(zip(escapes, crossover_dates)):
        mult_j    = 1.0
        start_str = xover_j.strftime("%Y-%m-%d")
        end_str   = (crossover_dates[j+1] - pd.Timedelta(days=1)).strftime("%Y-%m-%d") \
                    if j + 1 < len(crossover_dates) else None
        epochs.append({
            "start_date": start_str, "end_date": end_str,
            "a_share_column": share_cols[j],
            "b_share_column": share_cols[j+1],
            "beta_multiplier_b": mult_j,
            "a_escape": prev_escape,
            "b_escape": float(esc_j),
        })
        prev_escape = float(esc_j)
    return epochs


def _ode_to_arr(det_tot, r_norm, rng, n_weeks):
    det_safe = np.clip(det_tot, 0.0, 1e8)
    noise    = (rng.choice(r_norm, size=n_weeks, replace=True)
                * np.sqrt(np.maximum(det_safe, 1.0)))
    result   = det_tot + noise
    return np.where(np.isfinite(result), np.clip(result, 0.0, None), np.nan)


def _run_forward_k1(k_df, base_epochs, variant_marker, base_params,
                    fwd_ah, vacc_fwd, beta_fit, sigma_beta, r_norm, r_norm_death, HDR_fit,
                    fwd_total_weeks, fwd_horizon_days, fit_window_days, seed_offset):
    sim_cols = k_df.columns.tolist()
    N        = len(sim_cols)
    anchors  = {col: _find_anchor(k_df[col]) for col in sim_cols}
    fwd_bme  = float(np.mean(np.exp(-FIXED_ALPHA * fwd_ah[:fit_window_days])))
    fwd_di   = pd.date_range(base.FIT_START_DATE, periods=fwd_horizon_days, freq="D")
    arr       = np.zeros((N, fwd_total_weeks))
    arr_death = np.zeros((N, fwd_total_weeks))

    for k_idx, col in enumerate(sim_cols):
        rng      = np.random.default_rng(SEEDS[seed_offset + k_idx])
        anchor_i = anchors[col]
        ext_marker = _build_extended_marker_K1(variant_marker, k_df[col], anchor_i)
        fwd_share  = build_variant_share_series(
            ext_marker, fwd_di, base.FIT_START_DATE,
            "XFG.1.1_share", epoch_schedule=base_epochs)
        fwd_cfg = mu.make_cfg(base_epochs, fwd_ah, FIXED_ALPHA, fwd_bme)

        b_rep  = float(np.clip(rng.normal(beta_fit, sigma_beta), 0.05, 3.0))
        fp_rep = _apply_sto8_params(mu.apply_ic_mult(base_params, FIXED_IC_MULT), HDR=HDR_fit)
        r      = simulate_variant_model_HL(
            beta=b_rep, horizon_days=fwd_horizon_days,
            weekly_index=list(range(fwd_total_weeks)),
            vacc_series=vacc_fwd, variant_share_series=fwd_share,
            model_params=fp_rep, variant_cfg=fwd_cfg, stochastic_cfg=None)
        arr[k_idx]       = _ode_to_arr(r["weekly_hosp"],  r_norm,       rng, fwd_total_weeks)
        arr_death[k_idx] = _ode_to_arr(r["weekly_death"], r_norm_death, rng, fwd_total_weeks)
    return arr, arr_death


def _run_forward_kn(k_df, N_sims, letters, share_cols, variant_marker,
                    base_params, fwd_ah, vacc_fwd, beta_fit, sigma_beta,
                    r_norm, r_norm_death, HDR_fit, fwd_total_weeks, fwd_horizon_days, fit_window_days, seed_offset):
    n_xovers = len(letters) - 1
    fwd_bme  = float(np.mean(np.exp(-FIXED_ALPHA * fwd_ah[:fit_window_days])))
    fwd_di   = pd.date_range(base.FIT_START_DATE, periods=fwd_horizon_days, freq="D")
    arr       = np.zeros((N_sims, fwd_total_weeks))
    arr_death = np.zeros((N_sims, fwd_total_weeks))

    for k_idx in range(N_sims):
        rng        = np.random.default_rng(SEEDS[seed_offset + k_idx])
        i          = k_idx + 1
        var_series = [k_df[f"{L}{i}"] for L in letters]
        anchor_i   = _find_anchor(var_series[0])
        escapes    = [float(rng.beta(ESCAPE_A, ESCAPE_B_SHP)) for _ in range(n_xovers)]

        xovers   = []
        prev_min = EPOCH_C_MIN
        for j in range(n_xovers):
            xover = _crossover_shifted(var_series[j], var_series[j+1], anchor_i, prev_min)
            xovers.append(xover)
            prev_min = xover + pd.Timedelta(weeks=1)

        series_list = [(share_cols[j], var_series[j]) for j in range(len(letters))]
        ext_marker  = _build_extended_marker_Kn(
            variant_marker, series_list, anchor_i, share_cols)
        scen_epochs = _make_multi_epochs(escapes, xovers, share_cols)

        fwd_share = build_variant_share_series(
            ext_marker, fwd_di, base.FIT_START_DATE,
            "XFG.1.1_share", epoch_schedule=scen_epochs)
        fwd_cfg = mu.make_cfg(scen_epochs, fwd_ah, FIXED_ALPHA, fwd_bme)

        b_rep  = float(np.clip(rng.normal(beta_fit, sigma_beta), 0.05, 3.0))
        fp_rep = _apply_sto8_params(mu.apply_ic_mult(base_params, FIXED_IC_MULT), HDR=HDR_fit)
        r      = simulate_variant_model_HL(
            beta=b_rep, horizon_days=fwd_horizon_days,
            weekly_index=list(range(fwd_total_weeks)),
            vacc_series=vacc_fwd, variant_share_series=fwd_share,
            model_params=fp_rep, variant_cfg=fwd_cfg, stochastic_cfg=None)
        arr[k_idx]       = _ode_to_arr(r["weekly_hosp"],  r_norm,       rng, fwd_total_weeks)
        arr_death[k_idx] = _ode_to_arr(r["weekly_death"], r_norm_death, rng, fwd_total_weeks)
    return arr, arr_death


# ── Main ───────────────────────────────────────────────────────────────────────
def calibrate():
    """One-time Scenario-A beta calibration plus scenario-independent backward/
    forward config shared by every vaccination scenario.

    Returns a dict consumed by run_scenario_arrays() for each vaccination
    scenario, and by main() for plotting.
    """
    print("=" * 78)
    print("national_scenario: Vaccination scenarios A–E × K=1–5 variants")
    print("=" * 78)

    variant_marker = pd.read_csv(base.VARIANT_CSV)
    GC1            = base.load_extended_observations()

    fit_total_weeks  = (base.FORECAST_END_DATE - base.FIT_START_DATE).days // 7 + 2
    fit_horizon_days = fit_total_weeks * 7
    fit_window_days  = (base.FIT_END_DATE - base.FIT_START_DATE).days

    # ── Calibration: beta fit from Scenario A (baseline) ─────────────────────
    print("\nCalibrating from Scenario A ...")
    vacc_fit_A = mu.load_vacc_hl_weekly(
        "A", base.FIT_START_DATE, fit_total_weeks + 2)

    # vacc_frac_H in base_params is unused when passing 2D vacc series to the ODE;
    # keep FRAC_H (population ratio) as a harmless default.
    base_params = base.make_window_params_HL()
    fit_params  = _apply_sto8_params(mu.apply_ic_mult(base_params, FIXED_IC_MULT))

    fit_ah    = mu.load_ah_pw_national(base.FIT_START_DATE, fit_horizon_days)
    bme_fixed = float(np.mean(np.exp(-FIXED_ALPHA * fit_ah[:fit_window_days])))

    base_epochs = mu.make_base_epochs()
    fit_cfg     = mu.make_cfg(base_epochs, fit_ah, FIXED_ALPHA, bme_fixed)

    fit_daily_idx = pd.date_range(base.FIT_START_DATE, periods=fit_horizon_days, freq="D")
    fit_share     = build_variant_share_series(
        variant_marker, fit_daily_idx, base.FIT_START_DATE,
        "XFG_share", epoch_schedule=base_epochs)

    fit_obs_mask     = ((GC1["date"] >= base.FIT_START_DATE) &
                        (GC1["date"] <= base.FIT_END_DATE))
    fit_obs          = GC1.loc[fit_obs_mask].copy().reset_index(drop=True)
    fit_obs_week_idx = np.array(
        [(d - base.FIT_START_DATE).days // 7 for d in fit_obs["date"]], dtype=int)
    fit_obs_values   = fit_obs["observation"].to_numpy()

    print(f"  Fit obs: {len(fit_obs)} weeks "
          f"({fit_obs['date'].iloc[0].date()} → {fit_obs['date'].iloc[-1].date()})")

    def sim_fit_1p(_x, beta):
        r = simulate_variant_model_HL(
            beta=beta, horizon_days=fit_horizon_days,
            weekly_index=list(range(fit_total_weeks)),
            vacc_series=vacc_fit_A, variant_share_series=fit_share,
            model_params=fit_params, variant_cfg=fit_cfg)
        return r["weekly_hosp"][fit_obs_week_idx]

    popt, pcov = curve_fit(sim_fit_1p, np.arange(len(fit_obs)), fit_obs_values,
                           p0=[0.7571], bounds=([0.05], [3.0]), maxfev=15000)
    beta_fit   = float(popt[0])
    sigma_beta = float(np.sqrt(pcov[0, 0]))
    pred_fit   = sim_fit_1p(None, beta_fit)
    resid_fit  = fit_obs_values - pred_fit
    r_norm     = resid_fit / np.sqrt(np.maximum(pred_fit, 1.0))
    r_norm    -= r_norm.mean()
    rmse       = float(np.sqrt(np.mean(resid_fit**2)))
    print(f"  beta={beta_fit:.4f} ± {sigma_beta:.4f}   RMSE={rmse:.1f}")

    det = simulate_variant_model_HL(
        beta=beta_fit, horizon_days=fit_horizon_days,
        weekly_index=list(range(fit_total_weeks)),
        vacc_series=vacc_fit_A, variant_share_series=fit_share,
        model_params=fit_params, variant_cfg=fit_cfg)
    fit_weekly_det = det["weekly_hosp"]

    # ── HDR fit from inc death (all other params already fixed above) ────────
    # weekly_hosp never depends on HDR (it only gates the post-hospitalization
    # Hd-vs-Hr split -- see variant_model3.py), so beta/alpha/ic_mult/ihr_mult/
    # reinf_scale/wan/re_pro above are untouched by this step.
    print("\nEstimating HDR from inc death ...")
    GC1_death = base.load_extended_death_observations()
    fit_obs_death_mask  = ((GC1_death["date"] >= base.FIT_START_DATE) &
                           (GC1_death["date"] <= base.FIT_END_DATE))
    fit_obs_death        = GC1_death.loc[fit_obs_death_mask].copy().reset_index(drop=True)
    fit_obs_death_week_idx = np.array(
        [(d - base.FIT_START_DATE).days // 7 for d in fit_obs_death["date"]], dtype=int)
    fit_obs_death_values = fit_obs_death["observation"].to_numpy()

    print(f"  Fit obs (death): {len(fit_obs_death)} weeks "
          f"({fit_obs_death['date'].iloc[0].date()} → {fit_obs_death['date'].iloc[-1].date()})")

    def sim_fit_1p_death(_x, HDR):
        p = _apply_sto8_params(mu.apply_ic_mult(base_params, FIXED_IC_MULT), HDR=HDR)
        r = simulate_variant_model_HL(
            beta=beta_fit, horizon_days=fit_horizon_days,
            weekly_index=list(range(fit_total_weeks)),
            vacc_series=vacc_fit_A, variant_share_series=fit_share,
            model_params=p, variant_cfg=fit_cfg)
        return r["weekly_death"][fit_obs_death_week_idx]

    popt_d, pcov_d = curve_fit(
        sim_fit_1p_death, np.arange(len(fit_obs_death)), fit_obs_death_values,
        p0=[base.HDR], bounds=([0.001], [1.0]), maxfev=15000)
    HDR_fit    = float(popt_d[0])
    sigma_HDR  = float(np.sqrt(pcov_d[0, 0]))
    pred_death  = sim_fit_1p_death(None, HDR_fit)
    resid_death = fit_obs_death_values - pred_death
    r_norm_death = resid_death / np.sqrt(np.maximum(pred_death, 1.0))
    r_norm_death -= r_norm_death.mean()
    rmse_death  = float(np.sqrt(np.mean(resid_death**2)))
    print(f"  HDR={HDR_fit:.4f} ± {sigma_HDR:.4f}   RMSE(death)={rmse_death:.1f}")

    det_death = simulate_variant_model_HL(
        beta=beta_fit, horizon_days=fit_horizon_days,
        weekly_index=list(range(fit_total_weeks)),
        vacc_series=vacc_fit_A, variant_share_series=fit_share,
        model_params=_apply_sto8_params(mu.apply_ic_mult(base_params, FIXED_IC_MULT), HDR=HDR_fit),
        variant_cfg=fit_cfg)
    fit_weekly_det_death = det_death["weekly_death"]

    # ── Shared backward/forward config (scenario-independent) ────────────────
    bw_total_weeks  = (base.FIT_END_DATE - base.SIM_START).days // 7 + 2
    bw_horizon_days = bw_total_weeks * 7
    bw_ah  = mu.load_ah_pw_national(base.SIM_START, bw_horizon_days)
    bw_di  = pd.date_range(base.SIM_START, periods=bw_horizon_days, freq="D")
    bw_share = build_variant_share_series(
        variant_marker, bw_di, base.SIM_START,
        "XFG_share", epoch_schedule=base_epochs)
    fwd_total_weeks  = (FWD_END - base.FIT_START_DATE).days // 7 + 1
    fwd_horizon_days = fwd_total_weeks * 7
    fwd_ah = mu.load_ah_pw_national(base.FIT_START_DATE, fwd_horizon_days)

    # Forward pools K=1..5 sims in order, each K's rows seeded SEEDS[offset + k_idx].
    # Offsets/counts are contiguous (K_SEED_OFFSETS + per-K N_sims tile [0, 300) with
    # no gaps/overlaps), so pooled forward row j always uses seed j (SEEDS = arange).
    # Backward reuses the same seed j per row so each world index j is beta/noise-
    # paired across the backward/forward boundary (and across vaccination scenarios).
    k_counts = {}
    for ksc in K_SCENARIOS:
        hdr = pd.read_csv(ksc["csv"], index_col=0, nrows=1)
        k_counts[ksc["K"]] = hdr.shape[1] // len(ksc["letters"])
    N_TOTAL = sum(k_counts.values())
    expected_seeds = np.concatenate([
        K_SEED_OFFSETS[ksc["K"]] + np.arange(k_counts[ksc["K"]])
        for ksc in K_SCENARIOS
    ])
    assert np.array_equal(expected_seeds, np.arange(N_TOTAL)), (
        "K-scenario seed offsets are no longer contiguous with arange(N_TOTAL); "
        "backward/forward pairing by row index j -> seed j no longer holds.")
    print(f"\nForward pool size N_TOTAL = {N_TOTAL} "
          f"(K counts: {k_counts}) — backward will use the same {N_TOTAL} world seeds.")

    return dict(
        variant_marker=variant_marker, GC1=GC1,
        base_params=base_params, base_epochs=base_epochs, bme_fixed=bme_fixed,
        beta_fit=beta_fit, sigma_beta=sigma_beta, r_norm=r_norm, rmse=rmse,
        fit_weekly_det=fit_weekly_det,
        GC1_death=GC1_death, HDR_fit=HDR_fit, sigma_HDR=sigma_HDR,
        r_norm_death=r_norm_death,
        rmse_death=rmse_death, fit_weekly_det_death=fit_weekly_det_death,
        fit_total_weeks=fit_total_weeks, fit_window_days=fit_window_days,
        bw_total_weeks=bw_total_weeks, bw_horizon_days=bw_horizon_days,
        bw_ah=bw_ah, bw_share=bw_share,
        fwd_total_weeks=fwd_total_weeks, fwd_horizon_days=fwd_horizon_days,
        fwd_ah=fwd_ah, N_TOTAL=N_TOTAL, k_counts=k_counts,
    )


def run_scenario_arrays(vs, calib):
    """Backward (N_TOTAL sims) + forward (pooled K=1-5, N_TOTAL sims) for one
    vaccination scenario.

    Returns (bw_sto, bw_death, pooled, pooled_death): bw_sto/bw_death shape
    (bw_total_weeks, N_TOTAL); pooled/pooled_death shape (N_TOTAL, fwd_total_weeks).
    World index i is beta/noise-paired across bw_sto[:, i] and pooled[i, :] (see
    calibrate()'s seed-contiguity note). Death arrays come from the same ODE
    call as hosp (same beta/world, same HDR_fit estimated from inc death) and
    reuse the exact same per-world rng object -- death's bootstrapped noise
    (from r_norm_death) is drawn immediately after hosp's, continuing that
    object's stream rather than reseeding -- so hosp and death stochasticity
    are 1-1 seed-paired per world index, not just beta-paired.
    """
    tag   = vs["tag"]
    label = vs["label"]
    print(f"\n{'─'*60}")
    print(f"Vaccination {label} ({tag})")

    base_params      = calib["base_params"]
    base_epochs      = calib["base_epochs"]
    bme_fixed        = calib["bme_fixed"]
    beta_fit         = calib["beta_fit"]
    sigma_beta       = calib["sigma_beta"]
    r_norm           = calib["r_norm"]
    r_norm_death     = calib["r_norm_death"]
    HDR_fit          = calib["HDR_fit"]
    variant_marker   = calib["variant_marker"]
    bw_total_weeks   = calib["bw_total_weeks"]
    bw_horizon_days  = calib["bw_horizon_days"]
    bw_ah            = calib["bw_ah"]
    bw_share         = calib["bw_share"]
    fwd_total_weeks  = calib["fwd_total_weeks"]
    fwd_horizon_days = calib["fwd_horizon_days"]
    fwd_ah           = calib["fwd_ah"]
    fit_window_days  = calib["fit_window_days"]
    N_TOTAL          = calib["N_TOTAL"]

    # Backward sims for this vaccination scenario.
    # World index i uses seed=i, the same seed forward pooled row i uses
    # (see N_TOTAL/expected_seeds assertion in calibrate()), so backward[:, i] and
    # forward[i, :] are beta/noise-paired into one continuous trajectory.
    vacc_bw = mu.load_vacc_hl_weekly(tag[0], base.SIM_START, bw_total_weeks + 2)
    bw_cfg = mu.make_cfg(base_epochs, bw_ah, FIXED_ALPHA, bme_fixed)
    bw_sto   = np.zeros((bw_total_weeks, N_TOTAL))
    bw_death = np.zeros((bw_total_weeks, N_TOTAL))

    print(f"  Running {N_TOTAL} backward sims (paired with forward world index) ...")
    for i in range(N_TOTAL):
        rng_bw = np.random.default_rng(i)
        b_i    = float(np.clip(rng_bw.normal(beta_fit, sigma_beta), 0.05, 3.0))
        bw_p_i = _apply_sto8_params(mu.apply_ic_mult(base_params, FIXED_IC_MULT), HDR=HDR_fit)
        r      = simulate_variant_model_HL(
            beta=b_i, horizon_days=bw_horizon_days,
            weekly_index=list(range(bw_total_weeks)),
            vacc_series=vacc_bw, variant_share_series=bw_share,
            model_params=bw_p_i, variant_cfg=bw_cfg, stochastic_cfg=None)
        det_tot      = r["weekly_hosp"]
        noise        = (rng_bw.choice(r_norm, size=bw_total_weeks, replace=True)
                        * np.sqrt(np.maximum(det_tot, 1.0)))
        bw_sto[:, i]   = np.clip(det_tot + noise, 0.0, None)

        det_death_tot  = r["weekly_death"]
        noise_death    = (rng_bw.choice(r_norm_death, size=bw_total_weeks, replace=True)
                          * np.sqrt(np.maximum(det_death_tot, 1.0)))
        bw_death[:, i] = np.clip(det_death_tot + noise_death, 0.0, None)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{N_TOTAL} done")

    print(f"  K=1–5 forward sims ...")
    vacc_fwd = mu.load_vacc_hl_weekly(tag[0], base.FIT_START_DATE, fwd_total_weeks + 2)

    all_arrs       = []
    all_arrs_death = []
    for ksc in K_SCENARIOS:
        K           = ksc["K"]
        letters     = ksc["letters"]
        k_df        = pd.read_csv(ksc["csv"], index_col=0, parse_dates=True)
        N_sims      = k_df.shape[1] // len(letters)
        seed_offset = K_SEED_OFFSETS[K]

        print(f"  K={K}: {N_sims} sims × 1 run = {N_sims} ODE runs")

        if K == 1:
            arr, arr_death = _run_forward_k1(
                k_df, base_epochs, variant_marker, base_params,
                fwd_ah, vacc_fwd, beta_fit, sigma_beta, r_norm, r_norm_death, HDR_fit,
                fwd_total_weeks, fwd_horizon_days, fit_window_days, seed_offset)
        else:
            share_cols = SHARE_COLS_ALL[:len(letters)]
            arr, arr_death = _run_forward_kn(
                k_df, N_sims, letters, share_cols, variant_marker,
                base_params, fwd_ah, vacc_fwd, beta_fit, sigma_beta,
                r_norm, r_norm_death, HDR_fit, fwd_total_weeks, fwd_horizon_days, fit_window_days, seed_offset)

        all_arrs.append(arr)
        all_arrs_death.append(arr_death)

    pooled       = np.concatenate(all_arrs, axis=0)
    pooled_death = np.concatenate(all_arrs_death, axis=0)
    return bw_sto, bw_death, pooled, pooled_death


def main():
    calib = calibrate()
    pcts  = [2.5, 25, 50, 75, 97.5]

    GC1                  = calib["GC1"]
    beta_fit             = calib["beta_fit"]
    rmse                 = calib["rmse"]
    fit_weekly_det       = calib["fit_weekly_det"]
    GC1_death            = calib["GC1_death"]
    HDR_fit              = calib["HDR_fit"]
    rmse_death           = calib["rmse_death"]
    fit_weekly_det_death = calib["fit_weekly_det_death"]
    fit_total_weeks      = calib["fit_total_weeks"]
    bw_total_weeks       = calib["bw_total_weeks"]
    fwd_total_weeks      = calib["fwd_total_weeks"]
    N_TOTAL              = calib["N_TOTAL"]

    scenario_fwd_p         = {}   # tag → (fwd_total_weeks, 5)
    scenario_fwd_arr       = {}   # tag → (n_sims, fwd_total_weeks) — full arrays for paired CI
    scenario_bw_p          = {}   # tag → (bw_total_weeks, 5)
    scenario_bw_arr        = {}   # tag → (bw_total_weeks, N_TOTAL) — full arrays for paired CI
    scenario_fwd_p_death   = {}
    scenario_fwd_arr_death = {}
    scenario_bw_p_death    = {}
    scenario_bw_arr_death  = {}

    for vs in VAC_SCENARIOS:
        tag = vs["tag"]
        bw_sto, bw_death, pooled, pooled_death = run_scenario_arrays(vs, calib)
        scenario_bw_p[tag]    = np.percentile(bw_sto, pcts, axis=1).T
        scenario_bw_arr[tag]  = bw_sto
        scenario_fwd_p[tag]   = np.nanpercentile(pooled, pcts, axis=0).T
        scenario_fwd_arr[tag] = pooled          # (n_sims, fwd_total_weeks)

        scenario_bw_p_death[tag]    = np.percentile(bw_death, pcts, axis=1).T
        scenario_bw_arr_death[tag]  = bw_death
        scenario_fwd_p_death[tag]   = np.nanpercentile(pooled_death, pcts, axis=0).T
        scenario_fwd_arr_death[tag] = pooled_death

    # ── Combined plot ─────────────────────────────────────────────────────────
    fwd_week_dates = pd.date_range(base.FIT_START_DATE, periods=fwd_total_weeks, freq="7D")
    bw_week_dates  = pd.date_range(base.SIM_START,      periods=bw_total_weeks,  freq="7D")
    obs_dates      = pd.to_datetime(GC1["date"]).to_numpy()
    fit_win_n      = (base.FIT_END_DATE - base.FIT_START_DATE).days // 7 + 1
    fwd_dates      = fwd_week_dates[fit_win_n:]

    fig, ax = plt.subplots(figsize=(24, 12), dpi=300)
    plt.rcParams.update({"font.size": 28})

    # Fitted deterministic
    fit_week_dates = pd.date_range(base.FIT_START_DATE, periods=fit_total_weeks, freq="7D")
    ax.plot(fit_week_dates[:fit_win_n + 1], fit_weekly_det[:fit_win_n + 1],
            "--", color="#228833", lw=4,
            label=f"Fitted det. (β={beta_fit:.4f})")

    # Per-vaccination-scenario: CI shading (backward + forward), then median lines
    for vs in VAC_SCENARIOS:
        tag   = vs["tag"]
        color = vs["color"]
        bp    = scenario_bw_p[tag]
        fp    = scenario_fwd_p[tag]
        ax.fill_between(bw_week_dates, bp[:, 0], bp[:, 4], color=color, alpha=0.10)
        ax.fill_between(fwd_dates,     fp[fit_win_n:, 0], fp[fit_win_n:, 4],
                        color=color, alpha=0.15)

    for vs in VAC_SCENARIOS:
        tag   = vs["tag"]
        color = vs["color"]
        label = vs["label"]
        bp    = scenario_bw_p[tag]
        fp    = scenario_fwd_p[tag]
        ax.plot(bw_week_dates, bp[:, 2], "-", color=color, lw=2, alpha=0.8)
        ax.plot(fwd_dates, fp[fit_win_n:, 2],
                "-", color=color, lw=3, label=label)

    # Observations
    ax.plot(obs_dates, GC1["observation"], "*", color="black",
            markersize=16, label="Observed", zorder=5)

    ax.axvline(base.SIM_START,    linestyle="--", color="grey",           lw=2)
    ax.axvline(base.FIT_END_DATE, linestyle=":",  color="xkcd:dark grey", lw=3)

    ax.set_ylabel("Weekly hospitalizations", fontsize=28)
    ax.set_xlabel("Date", fontsize=28)
    ax.legend(fontsize=24, loc="upper right", ncol=2)
    ax.grid(alpha=0.4)
    ax.set_ylim(0, 25000)
    ax.set_xlim(base.SIM_START - pd.Timedelta(days=14),
                FWD_END        + pd.Timedelta(days=14))
    ax.set_yticks(range(0, 25001, 5000))
    ax.set_yticklabels(["0"] + [f"{t // 1000}k" for t in range(5000, 25001, 5000)])
    ax.tick_params(axis="both", labelsize=24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUT_PNG}")

    # ── Combined plot: DEATH ───────────────────────────────────────────────────
    obs_dates_death = pd.to_datetime(GC1_death["date"]).to_numpy()

    fig_d, ax_d = plt.subplots(figsize=(24, 12), dpi=300)
    plt.rcParams.update({"font.size": 26})

    ax_d.plot(fit_week_dates[:fit_win_n + 1], fit_weekly_det_death[:fit_win_n + 1],
              "--", color="#228833", lw=4,
              label=f"Fitted det. (HDR={HDR_fit:.4f})")

    for vs in VAC_SCENARIOS:
        tag   = vs["tag"]
        color = vs["color"]
        bp    = scenario_bw_p_death[tag]
        fp    = scenario_fwd_p_death[tag]
        ax_d.fill_between(bw_week_dates, bp[:, 0], bp[:, 4], color=color, alpha=0.10)
        ax_d.fill_between(fwd_dates,     fp[fit_win_n:, 0], fp[fit_win_n:, 4],
                          color=color, alpha=0.15)

    for vs in VAC_SCENARIOS:
        tag   = vs["tag"]
        color = vs["color"]
        label = vs["label"]
        bp    = scenario_bw_p_death[tag]
        fp    = scenario_fwd_p_death[tag]
        ax_d.plot(bw_week_dates, bp[:, 2], "-", color=color, lw=2, alpha=0.8)
        ax_d.plot(fwd_dates, fp[fit_win_n:, 2],
                  "-", color=color, lw=3, label=label)

    ax_d.plot(obs_dates_death, GC1_death["observation"], "*", color="black",
              markersize=16, label="Observed", zorder=5)

    ax_d.axvline(base.SIM_START,    linestyle="--", color="grey",           lw=2)
    ax_d.axvline(base.FIT_END_DATE, linestyle=":",  color="xkcd:dark grey", lw=3)

    # Only observations within the plotted x-range should drive the y-limit --
    # GC1_death spans back to 2024-01-01 and includes a much larger historical
    # peak that is off-screen here (x-range starts at SIM_START).
    plot_xmin = base.SIM_START - pd.Timedelta(days=14)
    plot_xmax = FWD_END        + pd.Timedelta(days=14)
    obs_death_visible_mask = ((GC1_death["date"] >= plot_xmin) &
                              (GC1_death["date"] <= plot_xmax))
    obs_death_visible_max  = float(
        GC1_death.loc[obs_death_visible_mask, "observation"].max())

    ymax_death = max(
        max(scenario_bw_p_death[tag][:, 4].max()  for tag in scenario_bw_p_death),
        max(scenario_fwd_p_death[tag][:, 4].max() for tag in scenario_fwd_p_death),
        obs_death_visible_max,
    ) * 1.15

    ax_d.set_ylabel("Weekly deaths", fontsize=24)
    ax_d.set_xlabel("Date", fontsize=24)
    ax_d.set_title(
        f"Vaccination scenarios A–E × K=1–5 variant uncertainty (deaths)  "
        f"β={beta_fit:.4f}  HDR={HDR_fit:.4f}  RMSE={rmse_death:.1f}",
        fontsize=13)
    ax_d.legend(fontsize=15, loc="upper right", ncol=2)
    ax_d.grid(alpha=0.4)
    ax_d.set_ylim(0, ymax_death)
    ax_d.set_xlim(plot_xmin, plot_xmax)
    ax_d.tick_params(axis="both", labelsize=22)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    OUT_PNG_DEATH = OUT_PNG.replace(".png", "_death.png")
    plt.tight_layout()
    plt.savefig(OUT_PNG_DEATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_PNG_DEATH}")

    # ── Bar plot: cumulative hospitalizations averted vs Scenario A ───────────
    tag_A  = "A-2026-05-11"
    fp_A   = scenario_fwd_p[tag_A]          # (fwd_total_weeks, 5)
    fp_A_f = fp_A[fit_win_n:]               # forward period only

    fwd_start_str = fwd_dates[0].strftime("%Y-%m-%d")
    fwd_end_str   = FWD_END.strftime("%Y-%m-%d")
    n_fwd_weeks   = len(fwd_dates)

    # Paired difference: each simulation row of A minus same row of X.
    # Scenarios share the same rng seed (42) and K-scenario CSVs read in the same
    # order, so row i of A and row i of X are driven by the same variant/beta draw.
    arr_A_f = scenario_fwd_arr[tag_A][:, fit_win_n:]   # (n_sims, n_fwd_weeks)
    med_A_total = float(np.nanmedian(np.nansum(arr_A_f, axis=1)))

    bar_labels, bar_meds, bar_lo, bar_hi, bar_colors, bar_pcts = [], [], [], [], [], []
    for vs in VAC_SCENARIOS[1:]:            # B, C, D, E
        tag   = vs["tag"]
        arr_X_f = scenario_fwd_arr[tag][:, fit_win_n:]  # (n_sims, n_fwd_weeks)

        # Trim to the same number of rows (should be equal, but guard against rounding)
        n = min(arr_A_f.shape[0], arr_X_f.shape[0])
        averted = np.nansum(arr_A_f[:n] - arr_X_f[:n], axis=1)  # (n_sims,)

        med = float(np.nanmedian(averted))
        lo  = float(np.nanpercentile(averted, 2.5))
        hi  = float(np.nanpercentile(averted, 97.5))
        pct = med / med_A_total * 100

        bar_labels.append(vs["label"])
        bar_meds.append(med)
        bar_lo.append(med - lo)             # downward error bar length
        bar_hi.append(hi - med)             # upward error bar length
        bar_colors.append(vs["color"])
        bar_pcts.append(pct)

    fig2, ax2 = plt.subplots(figsize=(10, 7), dpi=300)
    plt.rcParams.update({"font.size": 16})

    x = np.arange(len(bar_labels))
    bars = ax2.bar(x, bar_meds, color=bar_colors, alpha=0.80,
                   edgecolor="black", linewidth=1.2, width=0.55)
    ax2.errorbar(x, bar_meds,
                 yerr=[bar_lo, bar_hi],
                 fmt="none", color="black", capsize=10, capthick=2, linewidth=2)

    for rect, val, pct in zip(bars, bar_meds, bar_pcts):
        sign  = "+" if val >= 0 else ""
        ypos  = val + (max(bar_hi) * 0.04) if val >= 0 else val - (max(bar_hi) * 0.04)
        va    = "bottom" if val >= 0 else "top"
        ax2.text(rect.get_x() + rect.get_width() / 2, ypos,
                 f"{sign}{val:,.0f}\n({pct:.2f}%)",
                 ha="center", va=va, fontsize=13, fontweight="bold")

    ax2.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_labels, fontsize=15)
    ax2.set_ylabel("Cumulative hospitalizations averted\nvs Scenario A", fontsize=14)
    ax2.set_title(
        f"Hospitalizations averted relative to Scenario A\n"
        f"Forward period: {fwd_start_str} → {fwd_end_str}  ({n_fwd_weeks} weeks)\n"
        f"Error bars: 95% CI (paired simulation difference)",
        fontsize=12)
    ax2.grid(axis="y", alpha=0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    OUT_BAR = OUT_PNG.replace(".png", "_averted.png")
    plt.tight_layout()
    plt.savefig(OUT_BAR, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_BAR}")

    # ── Bar plot: cumulative averted over full period (backward + forward) ────
    # Backward and forward are now paired by world index (same seed=i on both
    # sides — see N_TOTAL/expected_seeds above), so each row i already represents
    # one continuous backward+forward trajectory. No independent-index bootstrap
    # is needed to join them; sum row-wise and percentile the paired totals.
    bw_arr_A   = scenario_bw_arr[tag_A]           # (bw_total_weeks, N_TOTAL)
    fwd_arr_Af = scenario_fwd_arr[tag_A][:, fit_win_n:]  # (N_TOTAL, n_fwd_weeks)

    full_start_str = bw_week_dates[0].strftime("%Y-%m-%d")
    full_end_str   = FWD_END.strftime("%Y-%m-%d")
    n_full_weeks   = len(bw_week_dates) + len(fwd_dates)

    full_labels, full_meds, full_lo, full_hi = [], [], [], []
    full_colors, full_pcts = [], []
    med_A_full = float(np.nanmedian(
        np.nansum(bw_arr_A, axis=0) + np.nansum(fwd_arr_Af, axis=1)))

    for vs in VAC_SCENARIOS[1:]:
        tag = vs["tag"]

        bw_arr_X   = scenario_bw_arr[tag]
        fwd_arr_Xf = scenario_fwd_arr[tag][:, fit_win_n:]

        n = min(bw_arr_A.shape[1], bw_arr_X.shape[1],
                fwd_arr_Af.shape[0], fwd_arr_Xf.shape[0])   # == N_TOTAL

        bw_averted  = np.nansum(bw_arr_A[:, :n]  - bw_arr_X[:, :n],  axis=0)  # (n,)
        fwd_averted = np.nansum(fwd_arr_Af[:n]    - fwd_arr_Xf[:n],  axis=1)  # (n,)
        total       = bw_averted + fwd_averted    # paired per world index i

        med = float(np.nanmedian(total))
        lo  = float(np.nanpercentile(total, 2.5))
        hi  = float(np.nanpercentile(total, 97.5))
        pct = med / med_A_full * 100

        full_labels.append(vs["label"])
        full_meds.append(med)
        full_lo.append(med - lo)
        full_hi.append(hi - med)
        full_colors.append(vs["color"])
        full_pcts.append(pct)

    fig3, ax3 = plt.subplots(figsize=(10, 7), dpi=300)
    plt.rcParams.update({"font.size": 16})

    x3   = np.arange(len(full_labels))
    bars = ax3.bar(x3, full_meds, color=full_colors, alpha=0.80,
                   edgecolor="black", linewidth=1.2, width=0.55)
    ax3.errorbar(x3, full_meds,
                 yerr=[full_lo, full_hi],
                 fmt="none", color="black", capsize=10, capthick=2, linewidth=2)

    for rect, val, pct in zip(bars, full_meds, full_pcts):
        sign = "+" if val >= 0 else ""
        ypos = val + (max(full_hi) * 0.04) if val >= 0 else val - (max(full_hi) * 0.04)
        va   = "bottom" if val >= 0 else "top"
        ax3.text(rect.get_x() + rect.get_width() / 2, ypos,
                 f"{sign}{val:,.0f}\n({pct:.2f}%)",
                 ha="center", va=va, fontsize=13, fontweight="bold")

    ax3.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(full_labels, fontsize=15)
    ax3.set_ylabel("Cumulative hospitalizations averted\nvs Scenario A", fontsize=14)
    ax3.set_title(
        f"Hospitalizations averted relative to Scenario A\n"
        f"Full period: {full_start_str} → {full_end_str}  ({n_full_weeks} weeks)\n"
        f"Error bars: 95% CI (paired backward+forward simulation, n={N_TOTAL})",
        fontsize=12)
    ax3.grid(axis="y", alpha=0.4)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    OUT_FULL = OUT_PNG.replace(".png", "_averted_full.png")
    plt.tight_layout()
    plt.savefig(OUT_FULL, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_FULL}")

    # ── Cumulative hospitalization summary (forward period) ───────────────────
    print(f"\n{'='*70}")
    print(f"Cumulative hospitalizations in forward period: "
          f"{fwd_start_str} → {fwd_end_str}  ({n_fwd_weeks} weeks)")
    print(f"{'Scenario':12s}  {'Median':>10s}  {'2.5%':>10s}  {'97.5%':>10s}")
    print(f"{'-'*48}")
    for vs in VAC_SCENARIOS:
        tag  = vs["tag"]
        arr  = scenario_fwd_arr[tag][:, fit_win_n:]   # (n_sims, n_fwd_weeks)
        cum  = np.nansum(arr, axis=1)                  # (n_sims,) total per sim
        med  = np.nanmedian(cum)
        lo   = np.nanpercentile(cum, 2.5)
        hi   = np.nanpercentile(cum, 97.5)
        print(f"{vs['label']:12s}  {med:>10,.0f}  {lo:>10,.0f}  {hi:>10,.0f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
