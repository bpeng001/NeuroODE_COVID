"""
Aggregate flu-scenario-model-hub_2020-2024.csv's 7 age bins (6 real,
non-overlapping bins + the file's own redundant "6 Months - 17 Years"
partial-aggregate, which is excluded here to avoid double-counting
population) into the same child (6mo-17y) / adult (18+) buckets used by
state_weekly_vaccination_2023_2026.csv, using each bin's own `population`
column as weights (state-and-age-specific, more precise than the
pop_risk.csv-derived shares used for the 2023-26 file).

Only the 2020-21, 2021-22, and 2022-23 seasons are taken from the
scenario-hub reconstruction -- 2023-24 is dropped from that source and
left to the existing real (cdc_direct) rows in
state_weekly_vaccination_2023_2026.csv, so the two seasons are never
double-defined.

Neither source file is modified in place: the merge is written to a new
file, state_weekly_vaccination_2020_2026.csv, with a `data_source` column
(`cdc_direct` vs `scenario_hub_reconstructed`) preserving provenance.
"""
import csv

D = "/Users/boyapeng/Desktop/Dissertation/Aim2/Data/Flu/vaccination"
SCENARIO_HUB_PATH = f"{D}/flu-scenario-model-hub_2020-2024.csv"
EXISTING_PATH = f"{D}/state_weekly_vaccination_2023_2026.csv"
OUT_PATH = f"{D}/state_weekly_vaccination_2020_2026.csv"

CHILD_AGES = {"6 Months - 4 Years", "5-12 Years", "13-17 Years"}
ADULT_AGES = {"18-49 Years", "50-64 Years", "65+ Years"}

FIELDNAMES = ["state", "season", "week_ending", "child_cum_pct", "adult_cum_pct",
              "child_pop_share", "adult_pop_share", "weighted_cum_pct",
              "child_observed", "adult_observed", "data_source", "source_round"]


def aggregate_scenario_hub():
    groups = {}
    with open(SCENARIO_HUB_PATH) as f:
        for row in csv.DictReader(f):
            if row["season"] == "2023-2024":
                continue  # real cdc_direct data already covers this season
            if row["age"] not in CHILD_AGES and row["age"] not in ADULT_AGES:
                continue  # skip the redundant "6 Months - 17 Years" aggregate row
            key = (row["state"], row["season"], row["week_ending"])
            g = groups.setdefault(key, {"child_num": 0.0, "child_den": 0.0,
                                         "adult_num": 0.0, "adult_den": 0.0,
                                         "source_round": row["source_round"]})
            pop = float(row["population"])
            val = float(row["coverage_pct"])
            if row["age"] in CHILD_AGES:
                g["child_num"] += pop * val
                g["child_den"] += pop
            else:
                g["adult_num"] += pop * val
                g["adult_den"] += pop

    out_rows = []
    for (state, season, week_ending), g in groups.items():
        child_cum = g["child_num"] / g["child_den"]
        adult_cum = g["adult_num"] / g["adult_den"]
        total_pop = g["child_den"] + g["adult_den"]
        child_share = g["child_den"] / total_pop
        adult_share = g["adult_den"] / total_pop
        weighted = (g["child_num"] + g["adult_num"]) / total_pop
        out_rows.append({
            "state": state,
            "season": season,
            "week_ending": week_ending,
            "child_cum_pct": round(child_cum, 4),
            "adult_cum_pct": round(adult_cum, 4),
            "child_pop_share": round(child_share, 4),
            "adult_pop_share": round(adult_share, 4),
            "weighted_cum_pct": round(weighted, 4),
            "child_observed": "",
            "adult_observed": "",
            "data_source": "scenario_hub_reconstructed",
            "source_round": g["source_round"],
        })
    return out_rows


def load_existing():
    with open(EXISTING_PATH) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["data_source"] = "cdc_direct"
        r["source_round"] = ""
    return rows


def main():
    existing_rows = load_existing()
    scenario_rows = aggregate_scenario_hub()
    merged = existing_rows + scenario_rows

    season_order = {"2020-2021": 0, "2021-2022": 1, "2022-2023": 2,
                    "2023-2024": 3, "2024-2025": 4, "2025-2026": 5}
    merged.sort(key=lambda r: (r["state"], season_order[r["season"]], r["week_ending"]))

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(merged)

    print(f"Wrote {len(merged)} rows to {OUT_PATH}")
    print(f"  existing (cdc_direct): {len(existing_rows)}")
    print(f"  new (scenario_hub_reconstructed): {len(scenario_rows)}")
    seasons = sorted(set(r["season"] for r in merged), key=lambda s: season_order[s])
    for s in seasons:
        rows_s = [r for r in merged if r["season"] == s]
        srcs = sorted(set(r["data_source"] for r in rows_s))
        print(f"  {s}: n_rows={len(rows_s)}, sources={srcs}")


if __name__ == "__main__":
    main()
