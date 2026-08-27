"""
Regenerate the stochastic forecast as sto6.png using variant_model3.py
(HIGH / LOW risk stratification) on top of the same humidity-driven beta
seasonality and sto3-derived initial conditions used in sto5.

Differences vs sto5:
  - Texas population is split into HIGH-risk (27.9%) and LOW-risk (72.1%)
    using pop_risk.csv aggregates (18-49_H + 50-64_H + 65+_All for HIGH;
    0-17_All + 18-49_L + 50-64_L for LOW).
  - epsilon1 differs by group (0.40 H, 0.48 L), preserving population-
    weighted average = 0.46.
  - IHR differs by group (0.010 H, 0.005 L), preserving population-weighted
    IHR = 0.0065.
  - epsilon2 = 0.55 in both groups; HDR = 0.16 unchanged.
  - Plot shows stratified HR vs LR weekly hospitalizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from variant_model2 import build_variant_share_series
from variant_model3 import simulate_variant_model_HL

# -------- Paths (inlined) --------
REPO = "/Users/boyapeng/Desktop/Dissertation/Aim2/epi_model"
DATA = "/Users/boyapeng/Desktop/Dissertation/Aim2/Data"
VARIANT_CSV  = os.path.join(REPO, "variant_all_marker1.csv")
ESCAPE_CSV   = os.path.join(REPO, "dominate_variant.csv")
HOSP_CSV     = os.path.join(REPO, "time-series-2026.csv")
VAC_CSV      = os.path.join(DATA, "COVID_RD19_Vaccination_curves.csv")
POP_RISK_CSV = os.path.join(REPO, "pop_risk.csv")
OUT_PNG      = os.path.join(REPO, "sto6.png")
HUMIDITY_CSV = os.path.join(REPO, "Humidity_air_pres_shum_2020-2026.csv")

# -------- Texas population (sto5 baseline N) and H/L split (pop_risk.csv) --------
N = 31_290_831
FRAC_H = 0.279         # Texas: (18-49_H + 50-64_H + 65+) / pop_total
FRAC_L = 1.0 - FRAC_H  # = 0.721
N_H = int(round(N * FRAC_H))
N_L = N - N_H

# -------- Shared model constants --------
EPSILON2    = 0.55
HDR         = 0.16
RE_PRO      = 0.2
WANING_TIME = 120
REINF       = 1 / 30

# -------- Risk-stratified VE and IHR --------
EPSILON1_H = 0.40         # ratio (0.72/0.86) scaled so pop-weighted avg = 0.46
EPSILON1_L = 0.48
IHR_H      = 0.015        # 4.8x LR, gives ~65% HR share of hosp (CDC COVID-NET)
IHR_L      = 0.0031       # pop-weighted average = 0.279*0.015 + 0.721*0.0031 = 0.00642

# -------- Stochastic forecast --------
N_SIM     = 200
STD_STO   = 0.05 * 0.2
KAPPA_STO = 0.15

# -------- Per-variant escape (mid values from dominate_variant.csv) --------
_VARIANT_ESCAPE_MID = {
    "LP.8.1":  0.800,
    "XFG":     0.885,
    "XFG.1.1": 0.525,
}

def variant_escape(name, point="mid"):
    if point != "mid":
        raise NotImplementedError(f"only 'mid' is inlined; got point={point!r}")
    return _VARIANT_ESCAPE_MID[name]


# ------------- Window definition (extended fit window, same as sto5) -------------
SIM_START         = pd.Timestamp("2025-06-08")
FIT_START_DATE    = pd.Timestamp("2025-06-08")
FIT_END_DATE      = pd.Timestamp("2026-06-06")
FORECAST_END_DATE = pd.Timestamp("2027-06-05")

AH_ALPHA = 0.01


def load_ah_series_for_horizon(start_date, n_days, state="Texas"):
    """AH series with calendar-day CLIMATOLOGY for dates past CSV end.

    For dates within the CSV range (2019-12-29 -> 2026-02-22), the CSV value
    is used directly (daily-interpolated). For dates past CSV end, the value
    is the MEAN of that calendar day's AH across all CSV years -- e.g.,
    Mar 15 2026 := mean(Mar 15 in {2020, 2021, ..., 2025}). This gives a
    smoothed seasonal climatology instead of replaying a single year.
    """
    df = pd.read_csv(HUMIDITY_CSV)
    df["date"] = pd.to_datetime(df["date"])
    sub = df[df["state"] == state].sort_values("date")
    sub = sub.drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    ah = sub.set_index("date")["Absolute Humidity"]
    daily_idx = pd.date_range(ah.index.min(), ah.index.max(), freq="D")
    daily_ah = ah.reindex(daily_idx).interpolate(method="time").bfill().ffill()

    # Build calendar-day climatology: mean AH across all CSV years per "MM-DD".
    csv_df = daily_ah.reset_index()
    csv_df.columns = ["date", "ah"]
    csv_df["md"] = csv_df["date"].dt.strftime("%m-%d")
    climatology = csv_df.groupby("md")["ah"].mean()   # Series indexed by "MM-DD"

    # Reindex to target horizon; fill past-CSV gaps from climatology.
    target = pd.date_range(start_date, periods=n_days, freq="D")
    out = daily_ah.reindex(target)
    need_fill = out.isna()
    if need_fill.any():
        md_keys = target[need_fill].strftime("%m-%d")
        clim_vals = climatology.reindex(md_keys).to_numpy()
        out.loc[need_fill] = clim_vals

    # Safety fallback for any residual NaN (e.g. Feb 29 in a non-leap target year)
    out = out.interpolate(method="time").bfill().ffill()
    return out.to_numpy()


def build_variant_cfg(ah_array_full, ah_baseline_mean_exp):
    """Variant cfg for FIT simulation (same as sto5)."""
    transmission_epochs = [
        {"start_date": "2025-06-08", "end_date": "2026-03-13",
         "a_share_column": "LP.8.1_share", "b_share_column": "XFG_share",
         "beta_multiplier_b": 1.0,
         "a_escape": variant_escape("LP.8.1"),
         "b_escape": variant_escape("XFG")},
        {"start_date": "2026-03-14", "end_date": None,
         "a_share_column": "XFG_share", "b_share_column": "XFG.1.1_share",
         "beta_multiplier_b": 0.7,
         "a_escape": variant_escape("XFG"),
         "b_escape": variant_escape("XFG.1.1")},
    ]
    return {
        "transmission_epochs": transmission_epochs,
        "waning_multiplier_B": 0.5,
        "beta_multiplier_B": 1.0,
        "a_escape": 0.0, "b_escape": 0.0,
        "ah_array": ah_array_full,
        "ah_alpha": AH_ALPHA,
        "ah_baseline_mean_exp": ah_baseline_mean_exp,
    }


def build_backward_variant_cfg(ah_array_full, ah_baseline_mean_exp):
    """Backward cfg (same as fit -- both use sto3 ICs in sto6)."""
    return build_variant_cfg(ah_array_full, ah_baseline_mean_exp)


def _split_HL(value, ihr_split=False):
    """Split a sto3-aggregated compartment count into H and L by population fraction.
    If ihr_split=True, weight by IHR ratio (HR over-represented in severe-track flows).
    Returns (H_value, L_value).
    """
    if ihr_split:
        # Severe-track infectious people: HR weight = FRAC_H * IHR_H, normalised.
        wH = FRAC_H * IHR_H
        wL = FRAC_L * IHR_L
        fH = wH / (wH + wL)
    else:
        fH = FRAC_H
    return value * fH, value * (1.0 - fH)


def make_window_params_HL():
    """sto3-derived initial conditions at 2025-06-08, split into H and L
    compartments by population fraction (proportional to pop_risk shares).
    """
    # sto3 aggregate compartment values (from regenerate_sto5 backward cfg)
    sto3_ic = {
        "R0":        969_138,
        "R_B0":       45_901,
        "SxS0":    3_729_016,
        "SxI0":    2_888_249,
        "S_B_prot0":  11_514,
        "S_B_part0":   3_112,
        "Inh0":        2_523,
        "Inr0":      108_302,
        "Imh0":          352,
        "Imr0":        7_156,
        "Inh_B0":        255,
        "Inr_B0":     19_018,
        "Imh_B0":         37,
        "Imr_B0":      2_206,
        "Hd0":           808,
        "Hr0":         3_558,
        "Hd_B0":          25,
        "Hr_B0":         125,
        "D0":         14_963,
    }

    params = {
        # Population partition
        "N_H": float(N_H),
        "N_L": float(N_L),
        # Risk-stratified epsilons / IHR
        "epsilon1_H": EPSILON1_H,
        "epsilon1_L": EPSILON1_L,
        "epsilon2":   EPSILON2,
        "IHR_H": IHR_H,
        "IHR_L": IHR_L,
        "HDR": HDR,
        # Shared progression rates
        "gamma1": (1/6 + 1/5) / 2.0,
        "gamma2": 1 / (7 + 14),
        "delta":  1 / (8 + 10),
        "eta":    1 / (10 + 14),
        "rep":    0.2,
        "re_pro": RE_PRO,
        "reinf":  REINF,
        "wan":    1.0 / WANING_TIME,
        # Vaccinations all drained by 2025-06-08 in sto3 IC; default split.
        "V0_H": 0.0, "V0_L": 0.0,
        "vacc_frac_H": FRAC_H,
    }

    # Non-severe (R, S_*, V, H_recovery-track) compartments: split by population.
    # Severe-track infectious (Inh, Imh) and hospitalized split with IHR weighting
    # so HR is correctly over-represented in flows that matter for hospitalization.
    for name, val in sto3_ic.items():
        ihr_weighted = name.startswith("Inh") or name.startswith("Imh") or name.startswith("Hd") or name.startswith("Hr") or name == "D0"
        vH, vL = _split_HL(val, ihr_split=ihr_weighted)
        params[f"{name}_H"] = vH
        params[f"{name}_L"] = vL

    return params


def load_vacc_annual_extension(start_date, n_weeks_needed):
    """Same as sto5's helper."""
    vac = pd.read_csv(VAC_CSV)
    vac["Date"] = pd.to_datetime(vac["Date"])
    tx = vac[(vac["Geography"] == "Texas") & (vac["Risk_group"] == "Overall")].copy()
    tx["vaccinated_cnt"] = tx["Cum.Coverage.Percent"] * tx["Pop"] / 100.0
    ts = tx.groupby("Date", as_index=False).agg(total_vaccinated=("vaccinated_cnt", "sum"))
    ts = ts.sort_values("Date").reset_index(drop=True)
    ts["total_vaccinated"] = ts["total_vaccinated"].round()
    ts["new_vaccinated"] = ts["total_vaccinated"].diff().fillna(0.0)
    prior = ts[ts["Date"] <= start_date]
    V0 = float(prior.iloc[-1]["total_vaccinated"]) if len(prior) > 0 else 0.0
    week_dates = pd.date_range(start_date + pd.Timedelta(days=1),
                               periods=n_weeks_needed, freq="7D")
    full = (
        ts.set_index("Date").reindex(week_dates).fillna(0.0)
          .reset_index().rename(columns={"index": "Date"})
    )
    base_series = full["new_vaccinated"].astype(float).to_numpy()
    last_data_date = ts["Date"].max()
    last_data_week = max(0, (last_data_date - start_date).days // 7)
    extended = base_series.copy()
    for i in range(last_data_week + 1, len(extended)):
        if i - 52 >= 0:
            extended[i] = extended[i - 52]
    return extended, V0, last_data_week


def load_extended_observations():
    hosp = pd.read_csv(HOSP_CSV)
    g = hosp[(hosp["location"].astype(str) == "48")
             & (hosp["age_group"] == "0-130")
             & (hosp["target"] == "inc hosp")]
    g = g.sort_values("date").reset_index(drop=True)
    g_obs = g[g["date"] >= "2024-01-01"].copy()
    g_obs["date"] = pd.to_datetime(g_obs["date"])
    return g_obs.reset_index(drop=True)


def make_share(variant_marker, variant_cfg, start_date, horizon_days):
    daily_index = pd.date_range(start_date, periods=horizon_days, freq="D")
    return build_variant_share_series(
        variant_marker, daily_index,
        start_date,
        variant_cfg["transmission_epochs"][0]["b_share_column"],
        epoch_schedule=variant_cfg["transmission_epochs"],
    )


def run_stochastic(beta_fit, beta_sd, pred_fit_window, resid_fit_window,
                   horizon_days, n_weeks, vacc, share, params, cfg,
                   seed_table, label):
    """Parameter uncertainty (curve_fit SE on beta) + heteroscedastic
    residual bootstrap (observation noise).

    Each of the N_SIM runs:
      1. Draws beta_i ~ Normal(beta_fit, beta_sd).
      2. Runs the simulator deterministically with that beta_i.
      3. Samples one normalized residual r_norm per week with replacement
         and adds noise = r_norm * sqrt(pred_t) to each week's prediction
         (Poisson-like scaling so noise grows at peaks, shrinks at troughs).
      4. Allocates the noise between HR and LR proportional to the
         deterministic split, so total = HR + LR is preserved.

    Returns (out_total, out_H, out_L) [n_weeks, N_SIM] matrices.
    """
    out_total = np.zeros((n_weeks, N_SIM))
    out_H     = np.zeros((n_weeks, N_SIM))
    out_L     = np.zeros((n_weeks, N_SIM))
    # Normalize residuals by sqrt(prediction) -> dimensionless heteroscedastic
    # noise that scales with local trajectory amplitude during forecasting.
    r_norm = resid_fit_window / np.sqrt(np.maximum(pred_fit_window, 1.0))
    print(f"\nRunning {N_SIM} {label} sims  "
          f"(beta~N({beta_fit:.4f},{beta_sd:.4f}); "
          f"resid bootstrap norm-sd={r_norm.std():.3f}) ...")
    rng = np.random.default_rng(int(seed_table[0]) if np.isfinite(seed_table[0]) else 0)
    for i in range(N_SIM):
        # 1. Parameter draw
        beta_i = max(float(rng.normal(beta_fit, beta_sd)), 1e-4)
        r = simulate_variant_model_HL(
            beta=beta_i, horizon_days=horizon_days,
            weekly_index=list(range(n_weeks)),
            vacc_series=vacc, variant_share_series=share,
            model_params=params, variant_cfg=cfg,
            stochastic_cfg=None)
        det_total = r["weekly_hosp"]
        det_H     = r["weekly_hosp_H"]
        det_L     = r["weekly_hosp_L"]
        # 2. Heteroscedastic residual bootstrap (one draw per week)
        r_sample = rng.choice(r_norm, size=n_weeks, replace=True)
        noise    = r_sample * np.sqrt(np.maximum(det_total, 1.0))
        out_total[:, i] = np.clip(det_total + noise, 0.0, None)
        # 3. Split noise to H / L by deterministic fraction (preserves sum)
        frac_H = det_H / np.maximum(det_total, 1e-9)
        out_H[:, i] = np.clip(det_H + noise * frac_H,         0.0, None)
        out_L[:, i] = np.clip(det_L + noise * (1.0 - frac_H), 0.0, None)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{N_SIM} sims done")
    return out_total, out_H, out_L


def main():
    print("=" * 78)
    print("sto6: HIGH / LOW risk stratification")
    print("=" * 78)
    print(f"Texas N split:  HIGH = {N_H:,} ({100*FRAC_H:.1f}%)   LOW = {N_L:,} ({100*FRAC_L:.1f}%)")
    print(f"epsilon1:       HIGH = {EPSILON1_H}   LOW = {EPSILON1_L}   "
          f"(weighted avg = {FRAC_H*EPSILON1_H + FRAC_L*EPSILON1_L:.4f})")
    print(f"IHR:            HIGH = {IHR_H}   LOW = {IHR_L}   "
          f"(weighted avg = {FRAC_H*IHR_H + FRAC_L*IHR_L:.5f})")
    print(f"epsilon2 = {EPSILON2} (both)  HDR = {HDR} (both)")
    print(f"alpha (humidity) = {AH_ALPHA}")
    print()
    print("Per-variant escape (mid):")
    for v in ("LP.8.1", "XFG", "XFG.1.1"):
        print(f"  {v:>10s}  escape = {variant_escape(v):.3f}")
    print()

    variant_marker = pd.read_csv(VARIANT_CSV)
    GC1 = load_extended_observations()

    if os.path.exists(os.path.join(REPO, "seed.csv")):
        seed_table = np.genfromtxt(os.path.join(REPO, "seed.csv"), delimiter=",")[:, 2]
    else:
        seed_table = np.arange(N_SIM, dtype=float)

    percentiles = [0, 1, 2.5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                   55, 60, 65, 70, 75, 80, 85, 90, 95, 97.5, 99, 100]

    # ====================== FIT SIMULATION ======================
    fit_total_weeks  = (FORECAST_END_DATE - FIT_START_DATE).days // 7 + 2
    fit_horizon_days = fit_total_weeks * 7
    vacc_fit, V0_fit, _ = load_vacc_annual_extension(FIT_START_DATE, fit_total_weeks + 2)
    fit_params = make_window_params_HL()

    fit_ah_array = load_ah_series_for_horizon(FIT_START_DATE, fit_horizon_days)
    fit_window_days = (FIT_END_DATE - FIT_START_DATE).days
    fit_baseline_mean_exp = float(np.mean(np.exp(-AH_ALPHA * fit_ah_array[:fit_window_days])))
    print(f"FIT AH: min={fit_ah_array.min():.2f}, max={fit_ah_array.max():.2f}, "
          f"mean={fit_ah_array.mean():.2f}, baseline_mean_exp={fit_baseline_mean_exp:.4f}")

    variant_cfg = build_variant_cfg(fit_ah_array, fit_baseline_mean_exp)
    fit_share = make_share(variant_marker, variant_cfg, FIT_START_DATE, fit_horizon_days)

    print(f"\n=== FIT SIM ===")
    print(f"  span: {FIT_START_DATE.date()} -> {FORECAST_END_DATE.date()} "
          f"({fit_total_weeks} weeks)")

    fit_obs_mask = (GC1["date"] >= FIT_START_DATE) & (GC1["date"] <= FIT_END_DATE)
    fit_obs = GC1.loc[fit_obs_mask].copy().reset_index(drop=True)
    fit_obs_week_idx = np.array(
        [(d - FIT_START_DATE).days // 7 for d in fit_obs["date"]], dtype=int
    )
    fit_obs_values = fit_obs["observation"].to_numpy()
    print(f"  fit observations: {len(fit_obs)} weeks "
          f"({fit_obs['date'].iloc[0].date()} -> {fit_obs['date'].iloc[-1].date()})")

    def sim_fit(_x, beta):
        result = simulate_variant_model_HL(
            beta=beta, horizon_days=fit_horizon_days,
            weekly_index=list(range(fit_total_weeks)),
            vacc_series=vacc_fit, variant_share_series=fit_share,
            model_params=fit_params, variant_cfg=variant_cfg)
        return result["weekly_hosp"][fit_obs_week_idx]

    print("\nFitting beta on fit-window observations ...")
    popt, pcov = curve_fit(sim_fit, np.arange(len(fit_obs)), fit_obs_values,
                           p0=0.22, bounds=(0.1, 2.0), maxfev=20000)
    beta_fit = float(popt[0])
    beta_sd  = float(np.sqrt(pcov[0, 0]))   # least-squares SE of beta
    pred_fit  = sim_fit(None, beta_fit)               # model predictions at fit weeks
    resid_fit = fit_obs_values - pred_fit             # obs - pred, used for bootstrap
    print(f"  fitted beta = {beta_fit:.4f}  +/- {beta_sd:.4f} (1 SE from curve_fit)")
    print(f"  relative SE = {100*beta_sd/beta_fit:.1f}%")
    print(f"  residuals (obs-pred): mean={resid_fit.mean():+.1f}, "
          f"sd={resid_fit.std():.1f}, range=[{resid_fit.min():.0f}, {resid_fit.max():.0f}]")

    # Deterministic full trace
    det = simulate_variant_model_HL(
        beta=beta_fit, horizon_days=fit_horizon_days,
        weekly_index=list(range(fit_total_weeks)),
        vacc_series=vacc_fit, variant_share_series=fit_share,
        model_params=fit_params, variant_cfg=variant_cfg)
    fit_weekly_det   = det["weekly_hosp"]
    fit_weekly_det_H = det["weekly_hosp_H"]
    fit_weekly_det_L = det["weekly_hosp_L"]

    # Stochastic fit + forward
    fit_sto, fit_sto_H, fit_sto_L = run_stochastic(
        beta_fit, beta_sd, pred_fit, resid_fit,
        fit_horizon_days, fit_total_weeks,
        vacc_fit, fit_share, fit_params, variant_cfg,
        seed_table, "fit+forward")
    fit_p   = np.percentile(fit_sto,   percentiles, axis=1).T
    fit_p_H = np.percentile(fit_sto_H, percentiles, axis=1).T
    fit_p_L = np.percentile(fit_sto_L, percentiles, axis=1).T

    # ====================== BACKWARD SIMULATION ======================
    bw_total_weeks  = (FIT_END_DATE - SIM_START).days // 7 + 1
    bw_horizon_days = bw_total_weeks * 7
    vacc_bw, V0_bw, _ = load_vacc_annual_extension(SIM_START, bw_total_weeks + 2)
    bw_params = make_window_params_HL()
    bw_ah_array = load_ah_series_for_horizon(SIM_START, bw_horizon_days)
    bw_baseline_mean_exp = float(np.mean(np.exp(-AH_ALPHA * bw_ah_array)))
    bw_cfg = build_backward_variant_cfg(bw_ah_array, bw_baseline_mean_exp)
    bw_share = make_share(variant_marker, bw_cfg, SIM_START, bw_horizon_days)
    print(f"\n=== BACKWARD SIM ===")
    print(f"  span: {SIM_START.date()} -> {FIT_END_DATE.date()} ({bw_total_weeks} weeks)")
    print(f"  using fitted beta = {beta_fit:.4f}")

    bw_sto, bw_sto_H, bw_sto_L = run_stochastic(
        beta_fit, beta_sd, pred_fit, resid_fit,
        bw_horizon_days, bw_total_weeks,
        vacc_bw, bw_share, bw_params, bw_cfg,
        seed_table, "backward")
    bw_p   = np.percentile(bw_sto,   percentiles, axis=1).T
    bw_p_H = np.percentile(bw_sto_H, percentiles, axis=1).T
    bw_p_L = np.percentile(bw_sto_L, percentiles, axis=1).T

    # ====================== PLOT ======================
    fit_week_dates = pd.date_range(FIT_START_DATE, periods=fit_total_weeks, freq="7D")
    bw_week_dates  = pd.date_range(SIM_START,      periods=bw_total_weeks,  freq="7D")
    obs_dates = pd.to_datetime(GC1["date"]).to_numpy()

    fit_window_n_weeks = (FIT_END_DATE - FIT_START_DATE).days // 7 + 1
    forecast_start_idx = fit_window_n_weeks

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(24, 18), dpi=300,
                                          sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    plt.rcParams.update({"font.size": 26})

    # --- TOP: total ---
    ax_top.fill_between(bw_week_dates, bw_p[:, 2], bw_p[:, 22],
                        color="xkcd:grey", alpha=0.13, label="95% CI (backward)")
    ax_top.plot(bw_week_dates, bw_p[:, 12], "-", color="xkcd:grey", lw=4,
                label="Median (backward)")
    ax_top.plot(fit_week_dates[:forecast_start_idx + 1],
                fit_weekly_det[:forecast_start_idx + 1],
                "--", color="xkcd:red", lw=5, label="Fitted line (total)")
    ax_top.fill_between(fit_week_dates[forecast_start_idx:],
                        fit_p[forecast_start_idx:, 2],
                        fit_p[forecast_start_idx:, 22],
                        color="xkcd:red", alpha=0.13, label="95% CI (forward)")
    ax_top.plot(fit_week_dates[forecast_start_idx:], fit_p[forecast_start_idx:, 12],
                "-", color="xkcd:red", lw=5, label="Median (forward)")
    ax_top.plot(obs_dates, GC1["observation"], "*", color="black",
                markersize=16, label="Observed")
    ax_top.axvline(SIM_START,      linestyle="--", color="grey", linewidth=2)
    ax_top.axvline(FIT_END_DATE,   linestyle=":",  color="xkcd:dark grey", linewidth=3)
    ax_top.set_ylabel("Total weekly hospitalizations", fontsize=24)
    ax_top.set_title("sto6: HIGH/LOW risk stratification (top: total, bottom: HR vs LR split)",
                     fontsize=22)
    ax_top.legend(fontsize=18, loc="upper right", ncol=2)
    ax_top.grid()
    ax_top.set_ylim(0, 1500)

    # --- BOTTOM: HR vs LR ---
    ax_bot.fill_between(fit_week_dates, fit_p_H[:, 2], fit_p_H[:, 22],
                        color="xkcd:purple", alpha=0.12, label="HR 95% CI")
    ax_bot.plot(fit_week_dates, fit_p_H[:, 12], "-", color="xkcd:purple", lw=4,
                label="HR median")
    ax_bot.fill_between(fit_week_dates, fit_p_L[:, 2], fit_p_L[:, 22],
                        color="xkcd:teal", alpha=0.12, label="LR 95% CI")
    ax_bot.plot(fit_week_dates, fit_p_L[:, 12], "-", color="xkcd:teal", lw=4,
                label="LR median")
    # Backward bands too (lighter)
    ax_bot.plot(bw_week_dates, bw_p_H[:, 12], "--", color="xkcd:purple", lw=2, alpha=0.7,
                label="HR backward median")
    ax_bot.plot(bw_week_dates, bw_p_L[:, 12], "--", color="xkcd:teal", lw=2, alpha=0.7,
                label="LR backward median")
    ax_bot.axvline(SIM_START,      linestyle="--", color="grey", linewidth=2)
    ax_bot.axvline(FIT_END_DATE,   linestyle=":",  color="xkcd:dark grey", linewidth=3)
    ax_bot.set_ylabel("Weekly hosp by risk", fontsize=24)
    ax_bot.set_xlabel("Date", fontsize=24)
    ax_bot.legend(fontsize=16, loc="upper right", ncol=2)
    ax_bot.grid()

    for ax in (ax_top, ax_bot):
        ax.set_xlim(SIM_START - pd.Timedelta(days=14),
                    FORECAST_END_DATE + pd.Timedelta(days=14))
        ax.tick_params(axis="both", labelsize=22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {OUT_PNG}")

    # ====================== Diagnostics ======================
    fit_pred = fit_weekly_det[fit_obs_week_idx]
    resid = fit_pred - fit_obs_values
    print(f"\nFit-window residual diagnostic:")
    print(f"  RMSE = {np.sqrt(np.mean(resid**2)):.1f}")
    print(f"  mean residual = {np.mean(resid):+.1f}")

    fwd_med = fit_p[forecast_start_idx:, 12]
    if len(fwd_med) > 0:
        pk = int(np.argmax(fwd_med))
        pk_date = fit_week_dates[forecast_start_idx + pk]
        print(f"\nForward projection (total):")
        print(f"  median peak: {fwd_med[pk]:.0f} on {pk_date.date()}")
        print(f"  95% CI at peak: [{fit_p[forecast_start_idx + pk, 2]:.0f}, "
              f"{fit_p[forecast_start_idx + pk, 22]:.0f}]")

    bw_med = bw_p[:, 12]
    bw_pk = int(np.argmax(bw_med))
    print(f"\nBackward projection (total):")
    print(f"  median peak: {bw_med[bw_pk]:.0f} on {bw_week_dates[bw_pk].date()}")
    print(f"  HR share at backward peak: "
          f"{100*bw_p_H[bw_pk, 12]/(bw_p[bw_pk, 12] + 1e-9):.1f}%")

    # HR share of fit-window cumulative hospitalizations
    fit_end_idx = (FIT_END_DATE - FIT_START_DATE).days // 7 + 1
    cum_H = fit_weekly_det_H[:fit_end_idx].sum()
    cum_L = fit_weekly_det_L[:fit_end_idx].sum()
    print(f"\nFit-window cumulative weekly-hosp split (deterministic):")
    print(f"  HR : {cum_H:>10,.0f}   ({100*cum_H/(cum_H+cum_L):.1f}%)")
    print(f"  LR : {cum_L:>10,.0f}   ({100*cum_L/(cum_H+cum_L):.1f}%)")


if __name__ == "__main__":
    main()
