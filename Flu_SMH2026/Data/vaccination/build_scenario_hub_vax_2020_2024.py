"""
Recover the REAL (unscaled) historical vaccination-coverage curves for
2020-21, 2021-22, 2022-23, and 2023-24 by inverting the known scaling
factors baked into the MIDAS flu-scenario-modeling-hub vaccination-scenario
files (RD1, RD2, RD5, RD6, in Data/Flu/vaccination/).

Each RD file is nominally "for" a later round's season, but its scenario
columns are just a real historical season's curve multiplied by a known
constant:

  RD1 (labeled round: 2022-23) -> real base = 2020-21
      Boost.coverage.rd1.sc_A_B = 1.10 * base
      Boost.coverage.rd1.sc_C_D = 0.90 * base
      (verified: sc_A_B/1.10 == sc_C_D/0.90 exactly, to 4 decimals, for every
      state/age/week checked)

  RD2 (labeled round: 2022-23) -> real base = 2021-22
      flu.coverage.rd2.sc_A_B_C_D = 1.00 * base   (single column, unscaled)

  RD5 (labeled round: 2024-25) -> real base = 2022-23
      flu.coverage.rd2425.sc_A_B = 1.20 * base
      flu.coverage.rd2425.sc_C_D = 1.00 * base
      flu.coverage.rd2425.sc_E_F = 0.80 * base
      (verified: ratios exactly 1.20 / 1.00 / 0.80)

  RD6 (labeled round: 2025-26) -> real base = 2023-24
      flu.coverage.rd2526.sc_A = 1.00 * base ("same coverage as 2023-24")
      flu.coverage.rd2526.sc_B = 0.65 * base, under-65 age groups only
      (RD6's sc_A is used here only as a secondary/cross-check source since
      the real, genuinely-weekly official CDC series already covers 2023-24
      directly -- see cdc_judz-8etw / cdc_sw5n-wg2p.)

  (RD4, labeled round 2023-24, also reconstructs 2021-22 -- its sc_C_D
  column was cross-checked against RD2's column and matches to <0.5%, the
  residual being interpolation-grid rounding, not a real discrepancy. Not
  included in the output to avoid duplicate rows for the same season.)

Each file's Week_Ending_Sat column is a real calendar date, but positioned
on the LABELED round's calendar, not the true source season's calendar. To
recover the real historical date for each row, we convert the label date to
its MMWR (year, week) via the `epiweeks` package, subtract the season gap
from the year while keeping the same MMWR week number, and take that
shifted week's Saturday end-date. This is the CDC epi-week definition and
correctly rolls over the Dec/Jan season-boundary and 53-week years, unlike
a naive "subtract N years" on the calendar date (which drifts off Saturday
because of leap days).

Ages are NOT aggregated -- all rows (6 individual age bins plus the file's
own "6 Months - 17 Years" partial-aggregate row, and all 52 Geography
values including "United States") are passed through as-is, just re-dated
and re-labeled with the true season and a coverage_pct column.

Output: one row per (state, age, week_ending), covering 2020-21 through
2023-24, written to
  Data/Flu/vaccination/flu-scenario-model-hub_2020-2024.csv
"""
import csv
import datetime as dt

from epiweeks import Week

D = "/Users/boyapeng/Desktop/Dissertation/Aim2/Data/Flu/vaccination"
OUT_PATH = f"{D}/flu-scenario-model-hub_2020-2024.csv"

# (source file, coverage column, labeled round start year, true source start year, source round id)
SOURCES = [
    ("RD1_2022_23_Sc_A_B_C_D.csv", "Boost.coverage.rd1.sc_A_B", 2022, 2020, "RD1", 1.10),
    ("RD2_2022_23_Sc_A_B_C_D.csv", "flu.coverage.rd2.sc_A_B_C_D", 2022, 2021, "RD2", 1.00),
    ("RD5_2024_25_Sc_A_B_C_D_E_F.csv", "flu.coverage.rd2425.sc_C_D", 2024, 2022, "RD5", 1.00),
    ("RD6_2025_26_Sc_A_B.csv", "flu.coverage.rd2526.sc_A", 2025, 2023, "RD6", 1.00),
]


def _safe_week_enddate(year, week):
    """Saturday end-date for MMWR (year, week), extrapolating consecutively
    past week 52 if `year` doesn't have a 53rd MMWR week (this only happens
    when shifting a date FROM a 53-week source year, e.g. 2025, onto a
    target year that has just 52, e.g. 2023 -- see MODEL.md's note on the
    ~5-6 year 53-week-season cycle)."""
    try:
        return Week(year, week).enddate()
    except ValueError:
        return Week(year, 52).enddate() + dt.timedelta(weeks=(week - 52))


def redate(label_date, labeled_start_year, true_start_year):
    """Map a Saturday-ending date on the LABELED round's calendar onto the
    corresponding real date in the true historical season, preserving the
    MMWR (epi) week number."""
    wk = Week.fromdate(label_date)
    year_gap = labeled_start_year - true_start_year
    return _safe_week_enddate(wk.year - year_gap, wk.week)


def main():
    out_rows = []
    for fname, col, labeled_start_year, true_start_year, round_id, scale in SOURCES:
        season_label = f"{true_start_year}-{true_start_year + 1}"
        with open(f"{D}/{fname}") as f:
            for row in csv.DictReader(f):
                label_date = dt.datetime.strptime(row["Week_Ending_Sat"], "%Y-%m-%d").date()
                true_date = redate(label_date, labeled_start_year, true_start_year)
                coverage_pct = float(row[col]) / scale
                out_rows.append({
                    "state": row["Geography"],
                    "age": row["Age"],
                    "population": row["Population"],
                    "week_ending": true_date.isoformat(),
                    "season": season_label,
                    "coverage_pct": round(coverage_pct, 6),
                    "source_round": round_id,
                    "source_file": fname,
                })

    out_rows.sort(key=lambda r: (r["season"], r["state"], r["age"], r["week_ending"]))

    fieldnames = ["state", "age", "population", "week_ending", "season",
                  "coverage_pct", "source_round", "source_file"]
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {OUT_PATH}")
    seasons = sorted(set(r["season"] for r in out_rows))
    for s in seasons:
        rows_s = [r for r in out_rows if r["season"] == s]
        dates = sorted(r["week_ending"] for r in rows_s)
        print(f"  {s}: n_rows={len(rows_s)}, date_range={dates[0]} -> {dates[-1]}")


if __name__ == "__main__":
    main()
