"""
model_utils.py: Shared utilities for national COVID model scripts.
Single source of truth for:
  - cleaned vaccination CSV (COVID_RD20_Vaccine.csv)
  - load_vacc_scenario: H/L weekly new-dose series from any RD20 scenario
  - load_ah_pw_national: population-weighted absolute humidity
  - make_base_epochs / make_cfg: variant epoch configuration
  - apply_ic_mult: scale SxS0/SxI0 at initial conditions
"""

import numpy as np
import pandas as pd
import national_sto1 as base

VAC_CSV    = "/Users/boyapeng/Desktop/Dissertation/Aim2/Data/COVID_RD20_Vaccine.csv"
HL_CSV     = "/Users/boyapeng/Desktop/Dissertation/Aim2/Data/vacc_HL_weekly.csv"
LEAF_H     = ["65+", "18-49 highrisk", "50-64 highrisk"]
LEAF_L     = ["0.5-17", "18-49 lowrisk", "50-64 lowrisk"]
EXCLUDE_AH = {"AK", "HI"}


def _fix_leaf_resets(dates, raw_vals):
    """Return corrected cumulative array for one age group.

    When a new campaign begins the CSV resets coverage to 0, so the raw
    cumulative drops sharply.  We detect each such drop (raw value falls to
    < 50 % of the previous raw value) and offset all subsequent values so
    that campaign doses are additive on top of the prior cumulative level.
    """
    result = raw_vals.copy().astype(float)
    offset = 0.0
    for i in range(1, len(result)):
        prev_raw = raw_vals[i - 1]
        curr_raw = raw_vals[i]
        if curr_raw < prev_raw * 0.5 and prev_raw > 1000:
            offset = result[i - 1]   # carry forward the last fixed level
        result[i] = curr_raw + offset
    return result


def load_vacc_scenario(scenario_tag, start_date, n_weeks):
    """Return weekly new-dose arrays shape (n_weeks, 2) — columns [H, L].

    Each leaf age group is processed independently so that:
      - campaign coverage resets (coverage restarts from 0) are handled
        correctly: the reset week's doses equal the new campaign aggregate
        rather than being clipped to zero.
      - missing report weeks for a single age group (e.g. 0.5-17 skipping
        one Sunday) do not distort the H/L aggregate.

    Parameters
    ----------
    scenario_tag : str  e.g. "A-2026-05-11"
    start_date   : pd.Timestamp
    n_weeks      : int
    """
    leaf_all = LEAF_H + LEAF_L
    df = pd.read_csv(VAC_CSV)
    df["date"] = pd.to_datetime(df["date"])
    nat = df[
        (df["geography"] == "National") &
        (df["scenario"]  == scenario_tag) &
        (df["age_group"].isin(leaf_all))
    ].copy()
    nat["cum_vacc"] = nat["coverage"] / 100.0 * nat["population"]

    week_dates = pd.date_range(start_date + pd.Timedelta(days=1), periods=n_weeks, freq="7D")
    date_min   = nat["date"].min()
    date_max   = max(nat["date"].max(), week_dates.max())
    daily_idx  = pd.date_range(date_min, date_max, freq="D")

    cum_H = pd.Series(0.0, index=daily_idx)
    cum_L = pd.Series(0.0, index=daily_idx)

    for age_group in leaf_all:
        sub = (nat[nat["age_group"] == age_group]
               .set_index("date")["cum_vacc"]
               .sort_index())
        if sub.empty:
            continue

        fixed_vals = _fix_leaf_resets(sub.index, sub.values)
        fixed = pd.Series(fixed_vals, index=sub.index)

        daily = (fixed
                 .reindex(fixed.index.union(daily_idx))
                 .interpolate("time")
                 .reindex(daily_idx)
                 .bfill().ffill().fillna(0.0))

        if age_group in LEAF_H:
            cum_H = cum_H + daily
        else:
            cum_L = cum_L + daily

    new_H = (cum_H.reindex(week_dates).interpolate("time").bfill().ffill().fillna(0.0)
             .diff().fillna(0.0).clip(lower=0.0).to_numpy())
    new_L = (cum_L.reindex(week_dates).interpolate("time").bfill().ffill().fillna(0.0)
             .diff().fillna(0.0).clip(lower=0.0).to_numpy())
    return np.column_stack([new_H, new_L])


def load_vacc_hl_weekly(scenario_label, start_date, n_weeks):
    """Load corrected vaccination from vacc_HL_weekly.csv into model week arrays.

    Parameters
    ----------
    scenario_label : str  single letter "A"–"E"
    start_date     : pd.Timestamp  model day-0
    n_weeks        : int

    Returns
    -------
    np.ndarray shape (n_weeks, 2)  columns [new_H, new_L]
    """
    df = pd.read_csv(HL_CSV)
    df["week_start"] = pd.to_datetime(df["week_start"])
    sub = df[df["scenario"] == scenario_label].set_index("week_start").sort_index()

    # Map each CSV Sunday date to model week index
    sub["k"] = np.clip((sub.index - start_date).days // 7, 0, n_weeks - 1)
    sub = sub[sub["k"] < n_weeks]

    result = np.zeros((n_weeks, 2), dtype=float)

    if sub.empty:
        return result

    # Place CSV values at their week indices
    for _, row in sub.iterrows():
        k = int(row["k"])
        result[k, 0] = row["new_H"]
        result[k, 1] = row["new_L"]

    # Linear interpolation for gaps between reported weeks
    for col in range(2):
        series = pd.Series(result[:, col])
        reported = sub["k"].astype(int).values
        # Mark unreported weeks (between first and last entry) as NaN, then interpolate
        mask = np.zeros(n_weeks, dtype=bool)
        mask[reported] = True
        first, last = reported.min(), reported.max()
        series_gap = series.copy().astype(float)
        series_gap[~mask] = np.nan
        series_gap[:first] = 0.0          # before first report: 0
        series_gap[last + 1:] = 0.0       # after last report: 0
        result[:, col] = series_gap.interpolate().fillna(0.0).clip(lower=0.0).values

    return result


def _get_name_to_abbr():
    df = pd.read_csv(base.POP_RISK_CSV)[["state", "Geography"]]
    df = df[df["state"] != "US"]
    return dict(zip(df["Geography"], df["state"]))


def _build_ah_pw(abbr_set, start_date, n_days):
    name_to_abbr = _get_name_to_abbr()
    df_ah = pd.read_csv(base.HUMIDITY_CSV)
    df_ah["date"] = pd.to_datetime(df_ah["date"])
    df_ah["abbr"] = df_ah["state"].map(name_to_abbr)
    df_ah = df_ah.dropna(subset=["abbr"])
    df_ah = df_ah[df_ah["abbr"].isin(abbr_set)].copy()
    df_pop = (pd.read_csv(base.POP_RISK_CSV)[["state", "pop_total"]]
              .rename(columns={"state": "abbr"}))
    df_pop = df_pop[df_pop["abbr"].isin(abbr_set)]
    merged = df_ah.merge(df_pop, on="abbr")

    def _wavg(g):
        return (g["Absolute Humidity"] * g["pop_total"]).sum() / g["pop_total"].sum()

    pw = merged.groupby("date").apply(_wavg).sort_index()
    daily_idx = pd.date_range(pw.index.min(), pw.index.max(), freq="D")
    daily_ah  = pw.reindex(daily_idx).interpolate(method="time").bfill().ffill()
    clim_df         = daily_ah.reset_index()
    clim_df.columns = ["date", "ah"]
    clim_df["md"]   = clim_df["date"].dt.strftime("%m-%d")
    climatology     = clim_df.groupby("md")["ah"].mean()
    target    = pd.date_range(start_date, periods=n_days, freq="D")
    out       = daily_ah.reindex(target)
    need_fill = out.isna()
    if need_fill.any():
        md_keys            = target[need_fill].strftime("%m-%d")
        out.loc[need_fill] = climatology.reindex(md_keys).to_numpy()
    return out.interpolate(method="time").bfill().ffill().to_numpy()


def load_ah_pw_national(start_date, n_days):
    """Return population-weighted AH array (n_days,) for CONUS (excl. AK, HI)."""
    df_pop = pd.read_csv(base.POP_RISK_CSV)[["state"]].rename(columns={"state": "abbr"})
    conus  = set(df_pop["abbr"]) - EXCLUDE_AH - {"US"}
    return _build_ah_pw(conus, start_date, n_days)


def make_base_epochs():
    """Three-epoch schedule: LP.8.1 → XFG → XFG.1.1 (dominant)."""
    return [
        {"start_date": "2025-04-27", "end_date": "2025-10-04",
         "a_share_column": "LP.8.1_share", "b_share_column": "XFG_share",
         "beta_multiplier_b": 1.0,
         "a_escape": base.variant_escape("LP.8.1"),
         "b_escape": base.variant_escape("XFG")},
        {"start_date": "2025-10-05", "end_date": "2026-03-13",
         "a_share_column": "XFG_share", "b_share_column": "XFG.1.1_share",
         "beta_multiplier_b": 1.0,
         "a_escape": base.variant_escape("XFG"),
         "b_escape": base.variant_escape("XFG.1.1")},
        {"start_date": "2026-03-14", "end_date": None,
         "a_share_column": "XFG.1.1_share", "b_share_column": "XFG.1.1_share",
         "beta_multiplier_b": 1.0,
         "a_escape": base.variant_escape("XFG.1.1"),
         "b_escape": base.variant_escape("XFG.1.1")},
    ]


def make_cfg(epochs, ah_array, ah_alpha, bme):
    """Build variant_cfg dict for simulate_variant_model_HL."""
    return {
        "transmission_epochs":  epochs,
        "waning_multiplier_B":  0.5,
        "beta_multiplier_B":    1.0,
        "a_escape": 0.0, "b_escape": 0.0,
        "ah_array":             ah_array,
        "ah_alpha":             ah_alpha,
        "ah_baseline_mean_exp": bme,
    }


def apply_ic_mult(params, ic_mult):
    """Return copy of params with SxS0 and SxI0 scaled by ic_mult."""
    p = dict(params)
    for suffix in ("_H", "_L"):
        p[f"SxS0{suffix}"] = p[f"SxS0{suffix}"] * ic_mult
        p[f"SxI0{suffix}"] = p[f"SxI0{suffix}"] * ic_mult
    return p
