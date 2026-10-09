"""
Population-weighted, cross-age (children 6mo-17y + adults 18+), weekly,
state-level cumulative flu vaccination coverage, for the 2023-24, 2024-25,
and 2025-26 seasons -- the three seasons where the adult series
(cdc_sw5n-wg2p) is genuinely weekly rather than sparse/monthly (see
MODEL.md / conversation: 2021-22 and 2022-23 only have ~9 real monthly
points within the weekly schema; 2023-24 onward is fully weekly).

Per-state child/adult population shares come from
/Users/boyapeng/Desktop/Dissertation/Aim2/Data/pop_risk.csv
(0_17_All / pop_total), replacing the single Texas-derived national
constant (CHILD_POP_SHARE=0.273) used elsewhere in data_prep.py -- this
lets each state use its own actual age structure.

For each state, the two source series (weekly children, weekly adults) are
merged onto the union of their real reported week-ending dates and
linearly interpolated onto that shared grid (same method
data_prep.build_overall_vax_daily uses for the daily grid), since the two
CDC datasets don't always share the exact same week-ending day (e.g. one
Saturday, one Sunday, in one week of the season).

Output: one row per (state, season, week_ending), with the raw children
and adult cumulative %, the state's population weights, and the blended
population-weighted cumulative % vaccinated.
"""
import csv
import sys

import numpy as np

sys.path.insert(0, "/Users/boyapeng/Desktop/Dissertation/Aim2/epi_model/flu")
import data_prep as dp

POP_RISK_PATH = "/Users/boyapeng/Desktop/Dissertation/Aim2/Data/pop_risk.csv"
OUT_PATH = "/Users/boyapeng/Desktop/Dissertation/Aim2/Data/Flu/vaccination/state_weekly_vaccination_2023_2026.csv"
SEASON_START_YEARS = [2023, 2024, 2025]


def load_state_pop_shares():
    shares = {}
    with open(POP_RISK_PATH) as f:
        for row in csv.DictReader(f):
            state = row["Geography"].strip()
            total = float(row["pop_total"])
            child = float(row["0_17_All"])
            shares[state] = {
                "child_share": child / total,
                "adult_share": 1 - child / total,
            }
    return shares


# The raw vaccination CSVs (cdc_judz-8etw, cdc_sw5n-wg2p) use
# geographic_name="National" for the whole-country row, while pop_risk.csv
# (and every other file in this vaccination pipeline) uses "United States"
# -- this maps the pop_risk/output label to the string the raw loaders
# actually need.
VAX_REGION_OVERRIDE = {"United States": "National"}


def blended_weekly_series(start_year, region, child_share, adult_share):
    child_dates, child_cum = dp.load_children_vax_weekly(start_year, region)
    adult_dates, adult_cum = dp.load_adult_vax_weekly(start_year, region)
    if len(child_dates) == 0 or len(adult_dates) == 0:
        return []

    anchor = min(child_dates[0], adult_dates[0])
    grid_dates = sorted(set(child_dates) | set(adult_dates))
    grid_days = np.array([(d - anchor).days for d in grid_dates], dtype=float)

    child_days = np.array([(d - anchor).days for d in child_dates], dtype=float)
    adult_days = np.array([(d - anchor).days for d in adult_dates], dtype=float)
    child_on_grid = np.interp(grid_days, child_days, child_cum, left=child_cum[0], right=child_cum[-1])
    adult_on_grid = np.interp(grid_days, adult_days, adult_cum, left=adult_cum[0], right=adult_cum[-1])
    weighted = child_share * child_on_grid + adult_share * adult_on_grid

    rows = []
    child_date_set = set(child_dates)
    adult_date_set = set(adult_dates)
    for d, c, a, w in zip(grid_dates, child_on_grid, adult_on_grid, weighted):
        rows.append({
            "week_ending": d.isoformat(),
            "child_cum_pct": round(float(c), 4),
            "adult_cum_pct": round(float(a), 4),
            "weighted_cum_pct": round(float(w), 4),
            "child_observed": d in child_date_set,
            "adult_observed": d in adult_date_set,
        })
    return rows


def main():
    pop_shares = load_state_pop_shares()
    out_rows = []
    skipped = []

    for state in sorted(pop_shares):
        child_share = pop_shares[state]["child_share"]
        adult_share = pop_shares[state]["adult_share"]
        vax_region = VAX_REGION_OVERRIDE.get(state, state)
        for start_year in SEASON_START_YEARS:
            season_label = f"{start_year}-{start_year + 1}"
            try:
                rows = blended_weekly_series(start_year, vax_region, child_share, adult_share)
            except Exception as e:
                skipped.append((state, season_label, str(e)))
                continue
            if not rows:
                skipped.append((state, season_label, "no data"))
                continue
            for r in rows:
                out_rows.append({
                    "state": state,
                    "season": season_label,
                    "child_pop_share": round(child_share, 4),
                    "adult_pop_share": round(adult_share, 4),
                    **r,
                })

    fieldnames = ["state", "season", "week_ending", "child_cum_pct", "adult_cum_pct",
                  "child_pop_share", "adult_pop_share", "weighted_cum_pct",
                  "child_observed", "adult_observed"]
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {OUT_PATH}")
    print(f"States processed: {len(pop_shares)}, skipped (state,season): {len(skipped)}")
    for s in skipped:
        print("  skipped:", s)


if __name__ == "__main__":
    main()
