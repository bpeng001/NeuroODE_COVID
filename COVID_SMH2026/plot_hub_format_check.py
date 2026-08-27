"""
plot_hub_format_check.py: Independent sanity-check plot for the Hub-format export.

Reads 2025-06-08-UTHRPI-EvoSEIR.gz.parquet (written by export_hub_format.py),
computes median + 95% CI across the 300 stochastic_run samples per
scenario/date/target, and overlays observed weekly hospitalizations/deaths from
time-series-2026.csv (via national_sto1.load_extended_observations() /
load_extended_death_observations()). This is purely a read-of-the-exported-file
check -- it does not re-run any ODE simulation -- to confirm the parquet content
matches what national_scenario.py plotted.

Note: "origin_date" is a fixed round identifier (same value on every row, see
export_hub_format.py's docstring) -- the actual per-week date axis is derived
from "horizon" (weeks-ahead from SIM_START on a contiguous weekly grid, since
SIM_START == FIT_START_DATE here).

Output: hub_format_check_hosp.png, hub_format_check_death.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import national_sto1 as base

PARQUET      = os.path.join(base.REPO, "2025-06-08-UTHRPI-EvoSEIR.gz.parquet")
OUT_PNG_HOSP  = os.path.join(base.REPO, "hub_format_check_hosp.png")
OUT_PNG_DEATH = os.path.join(base.REPO, "hub_format_check_death.png")

SCEN_COLORS = {
    "A-2026-05-11": "#1f77b4",
    "B-2026-05-11": "#2ca02c",
    "C-2026-05-11": "#ff7f0e",
    "D-2026-05-11": "#d62728",
    "E-2026-05-11": "#9467bd",
}


def _plot_target(df, target, obs_df, ylabel, out_png, ylim=None):
    sub_all = df[df["target"] == target]

    fig, ax = plt.subplots(figsize=(24, 10), dpi=300)
    plt.rcParams.update({"font.size": 22})

    for scen_id, color in SCEN_COLORS.items():
        sub = sub_all[sub_all["scenario_id"] == scen_id]
        stats = (sub.groupby("date")["value"]
                    .agg(median="median",
                         lo=lambda s: np.percentile(s, 2.5),
                         hi=lambda s: np.percentile(s, 97.5))
                    .sort_index())
        ax.fill_between(stats.index, stats["lo"], stats["hi"], color=color, alpha=0.12)
        ax.plot(stats.index, stats["median"], "-", color=color, lw=2.5,
                label=f"{scen_id} (parquet median)")

    obs_dates = pd.to_datetime(obs_df["date"]).to_numpy()
    ax.plot(obs_dates, obs_df["observation"], "*", color="black",
            markersize=14, label="Observed", zorder=5)

    ax.axvline(base.FIT_END_DATE, linestyle=":", color="xkcd:dark grey", lw=3,
               label="Backward/forward boundary (FIT_END_DATE)")

    xmin = sub_all["date"].min() - pd.Timedelta(days=14)
    xmax = sub_all["date"].max() + pd.Timedelta(days=14)

    if ylim is None:
        obs_visible = obs_df[(pd.to_datetime(obs_df["date"]) >= xmin) &
                              (pd.to_datetime(obs_df["date"]) <= xmax)]
        ci_max = (sub_all.groupby("date")["value"]
                          .apply(lambda s: np.percentile(s, 97.5)).max())
        ylim = (0, max(ci_max, float(obs_visible["observation"].max())) * 1.15)

    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_xlabel("Date", fontsize=22)
    ax.set_title(
        f"Hub-format parquet check ({target}): median + 95% CI per scenario, from "
        "2025-06-08-UTHRPI-EvoSEIR.gz.parquet\n(read directly from the exported "
        "file -- no re-simulation -- overlaid on observed data)",
        fontsize=15)
    ax.legend(fontsize=14, loc="upper right", ncol=2)
    ax.grid(alpha=0.4)
    ax.set_ylim(*ylim)
    ax.set_xlim(xmin, xmax)
    ax.tick_params(axis="both", labelsize=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


def main():
    df = pd.read_parquet(PARQUET)

    origin_dates = df["origin_date"].unique()
    assert len(origin_dates) == 1, f"expected a single fixed origin_date, got {origin_dates}"
    n_weeks = df["horizon"].max()
    week_dates = pd.date_range(base.SIM_START, periods=n_weeks, freq="7D")
    df["date"] = week_dates[df["horizon"].to_numpy() - 1]

    GC1_hosp  = base.load_extended_observations()
    GC1_death = base.load_extended_death_observations()

    _plot_target(df, "inc hosp",  GC1_hosp,  "Weekly hospitalizations",
                 OUT_PNG_HOSP, ylim=(0, 25000))
    _plot_target(df, "inc death", GC1_death, "Weekly deaths",
                 OUT_PNG_DEATH)


if __name__ == "__main__":
    main()
