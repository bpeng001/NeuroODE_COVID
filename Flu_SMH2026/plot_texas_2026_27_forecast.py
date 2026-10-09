"""
Plot the Texas 2026-27 forecast ensemble (forecast_texas_2026_27.py's
texas_2026_27_forecast_ensemble.npz) in the same two-panel style as
fit_texas_chain.plot_season_fit (texas_2025_26_hosp_fit.png etc.) -- by
subtype on the left, total on the right -- but with a median trajectory
and a shaded 95% interval band across the 300 Monte Carlo draws instead
of an observed-vs-fitted comparison, since there's no real 2026-27 data
to compare against yet.
"""
import datetime as dt

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

IN_PATH = "texas_2026_27_forecast_ensemble.npz"
OUT_PATH = "texas_2026_27_forecast.png"


def main():
    data = np.load(IN_PATH, allow_pickle=True)
    Y = data["ensemble_Y"]                      # (n_draws, n_weeks, n_strains)
    strains = [s for s in data["strains"]]
    dates = [dt.date.fromisoformat(s) for s in data["dates"]]
    n_weeks = len(dates)

    median_by_strain = np.median(Y, axis=0)               # (n_weeks, n_strains)
    lo_by_strain = np.percentile(Y, 2.5, axis=0)
    hi_by_strain = np.percentile(Y, 97.5, axis=0)

    total = Y.sum(axis=2)                                  # (n_draws, n_weeks)
    median_total = np.median(total, axis=0)
    lo_total = np.percentile(total, 2.5, axis=0)
    hi_total = np.percentile(total, 97.5, axis=0)

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

    fig.suptitle("Predicted Flu Hospitalizations -- Texas, 2026-27 Season "
                  "(300-draw Monte Carlo ensemble, median + 95% CI)")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)
    print(f"Saved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
