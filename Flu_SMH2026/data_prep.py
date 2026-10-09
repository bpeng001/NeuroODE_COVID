"""
Texas-specific data preparation for fitting the three-strain model to the
2019-20 season, with the initial state estimated from the 2018-19 season.

Data sources (all under /Users/boyapeng/Desktop/Dissertation/Aim2/Data/Flu):
  - FluViewPhase2Data/ILINet.csv                      weekly ILI, by state
  - FluViewPhase2Data/ICL_NREVSS_Clinical_Labs.csv     weekly %A/%B positivity, by state
  - FluViewPhase2Data/ICL_NREVSS_Public_Health_Labs.csv  season-total subtype counts, by state
  - cdc_judz-8etw_flu_vaccination.csv                 weekly cumulative vax %, children 6mo-17y
  - CDC_FluVaxView_2019-20_AdultsLocalAreas.xlsx      monthly cumulative vax %, TX adults (2 metro areas)

See the conversation-derived revision plan for the methodology and the
caveats around each of these (state ILI is unweighted-only; subtype shares
are season-level, not weekly; adult vaccination is sub-state and monthly).
"""

import csv
import datetime as dt

import numpy as np
import openpyxl

DATA_DIR = "/Users/boyapeng/Desktop/Dissertation/Aim2/Data/Flu"
ILINET_PATH = f"{DATA_DIR}/FluViewPhase2Data/ILINet.csv"
CLINICAL_LABS_PATH = f"{DATA_DIR}/FluViewPhase2Data/ICL_NREVSS_Clinical_Labs.csv"
PUBLIC_HEALTH_LABS_PATH = f"{DATA_DIR}/FluViewPhase2Data/ICL_NREVSS_Public_Health_Labs.csv"
CHILDREN_VAX_PATH = f"{DATA_DIR}/vaccination/cdc_judz-8etw_flu_vaccination.csv"
ADULT_VAX_WEEKLY_PATH = f"{DATA_DIR}/vaccination/cdc_sw5n-wg2p_flu_vaccination.csv"
ADULT_VAX_XLSX_PATH = f"{DATA_DIR}/vaccination/CDC_FluVaxView_2019-20_AdultsLocalAreas.xlsx"
MASTER_VAX_PATH = f"{DATA_DIR}/vaccination/state_weekly_vaccination_2020_2026.csv"
VE_XLSX_PATH = f"{DATA_DIR}/vaccination/influenza_subtype_VE_by_season_with_children.xlsx"
HOSP_PATH = "/Users/boyapeng/Desktop/Dissertation/Aim2/epi_model/flu/time-series.csv"
TX_FIPS = "48"

REGION = "Texas"
STRAINS = ["H1", "H3", "B"]

# 2020 Census populations, used to weight the two Texas BRFSS metro-area
# adult vaccination series into a single statewide-adult proxy.
BEXAR_COUNTY_POP = 2_009_324
HOUSTON_CITY_POP = 2_304_580

POP_RISK_PATH = "/Users/boyapeng/Desktop/Dissertation/Aim2/Data/pop_risk.csv"


def load_state_pop_shares():
    """Per-state (child 0-17, adult 18+) population shares from Census-derived
    age-group headcounts (`pop_risk.csv`: `0_17_All` / `pop_total`). Returns
    {state_name: (child_share, adult_share)}, including the national row
    ("United States", 0.2174/0.7826) -- previously excluded here, which
    made get_pop_shares("United States") raise KeyError for national
    fitting; kept in now since it's a real, valid row like any state's."""
    shares = {}
    with open(POP_RISK_PATH) as f:
        for row in csv.DictReader(f):
            state = row["Geography"].strip()
            total = float(row["pop_total"])
            child_share = float(row["0_17_All"]) / total
            shares[state] = (child_share, 1 - child_share)
    return shares


STATE_POP_SHARES = load_state_pop_shares()


def get_pop_shares(region):
    """(child_share, adult_share) for `region`, used to blend the
    weekly children's and adult vaccination series into a single
    whole-population proxy."""
    return STATE_POP_SHARES[region]


def season_weeks(start_year):
    """(year, week) pairs for the flu season starting in `start_year`
    (MMWR week 40 of start_year through week 39 of start_year+1)."""
    return [(start_year, w) for w in range(40, 54)] + [(start_year + 1, w) for w in range(1, 40)]


def _in_season(year, week, start_year):
    return (year == start_year and week >= 40) or (year == start_year + 1 and week <= 39)


def load_ili_weekly(start_year, region=REGION):
    """Weekly %UNWEIGHTED ILI for `region`, season starting `start_year`.
    Returns dict {(year, week): ili_fraction}."""
    out = {}
    with open(ILINET_PATH) as f:
        r = csv.reader(f)
        next(r)
        next(r)
        for row in r:
            if row[1] != region:
                continue
            year, week = int(row[2]), int(row[3])
            if not _in_season(year, week, start_year):
                continue
            val = row[5]
            if val == "X" or val == "":
                continue
            out[(year, week)] = float(val) / 100.0
    return out


def load_positivity_weekly(start_year, region=REGION):
    """Weekly total positivity fraction ((%A+%B)/100, == PERCENT POSITIVE/100)
    for `region`, season starting `start_year`. Returns dict {(year,week): frac}."""
    out = {}
    with open(CLINICAL_LABS_PATH) as f:
        r = csv.reader(f)
        next(r)
        next(r)
        for row in r:
            if row[1] != region:
                continue
            year, week = int(row[2]), int(row[3])
            if not _in_season(year, week, start_year):
                continue
            percent_positive = row[7]
            if percent_positive == "" or percent_positive == "X":
                continue
            out[(year, week)] = float(percent_positive) / 100.0
    return out


def load_subtype_shares(start_year, region=REGION):
    """Season-level H1/H3/B shares of total confirmed-positive specimens
    (Public Health Labs), for the season 'Season {start_year}-{(start_year+1)%100:02d}'.

    Unspecified-subtype A counts (Subtyping not Performed + Unable to
    Subtype) are allocated proportionally between H1 and H3 by their
    confirmed-subtype ratio. H3N2v / A(H5) are excluded (negligible).
    Denominator is H1_alloc + H3_alloc + B_alloc (total positives), not just A.
    """
    season_label = f"Season {start_year}-{(start_year + 1) % 100:02d}"
    with open(PUBLIC_HEALTH_LABS_PATH) as f:
        r = csv.reader(f)
        next(r)
        header = next(r)
        row = next(
            row for row in r
            if row[1] == region and row[2].strip() == season_label
        )
    col = {name: i for i, name in enumerate(header)}
    count_cols = ["A (2009 H1N1)", "A (H3)", "A (Subtyping not Performed)", "B", "BVic", "BYam"]
    if any(row[col[c]] == "X" for c in count_cols):
        # CDC small-count suppression (privacy convention, like ILINet's "X"):
        # seen for several small states' 2025-26 row (Alaska, Connecticut,
        # Hawaii, Oklahoma, Oregon, South Dakota, Utah) where every count
        # column is suppressed. No state-specific signal available at all,
        # same situation as the zero-positives case below -- fall back to
        # national shares.
        return load_national_subtype_shares(start_year)
    # Post-2015-16 Public Health Labs reporting only has "A (2009 H1N1)" (no
    # separate seasonal "A (H1)" column -- seasonal H1N1 stopped circulating
    # after the 2009 pandemic, so this is the complete H1 count).
    h1c = float(row[col["A (2009 H1N1)"]])
    h3c = float(row[col["A (H3)"]])
    u = float(row[col["A (Subtyping not Performed)"]])
    b_alloc = float(row[col["B"]]) + float(row[col["BVic"]]) + float(row[col["BYam"]])

    if h1c + h3c + b_alloc + u == 0:
        # Zero positives of ANY type/subtype reported for this state/season
        # -- seen for small states in the near-zero-circulation 2020-21
        # season (e.g. Vermont: 608 specimens tested, all negative for flu
        # entirely, confirmed in the raw file -- a real finding, not a data
        # error, consistent with that season's well-documented pandemic-era
        # NPI suppression). There's no state-specific signal to allocate at
        # all here; fall back to that season's national shares as the most
        # informed available proxy.
        return load_national_subtype_shares(start_year)
    if h1c + h3c == 0:
        # No confirmed-subtype A cases to set a ratio from, but B/unsubtyped
        # positives do exist -- split the unsubtyped pool evenly, same
        # "no information" convention already used elsewhere for this
        # season (see Texas's own 2020-21 edge case below: H3_share=
        # B_share=0.5 when h1c=0 with no ratio to lean on).
        h1_alloc = u / 2
        h3_alloc = u / 2
    else:
        h1_alloc = h1c + u * h1c / (h1c + h3c)
        h3_alloc = h3c + u * h3c / (h1c + h3c)
    total = h1_alloc + h3_alloc + b_alloc

    return {"H1": h1_alloc / total, "H3": h3_alloc / total, "B": b_alloc / total}


NATIONAL_SUBTYPE_PATH = f"{DATA_DIR}/ICL_NREVSS_1997_2026_clean_absolute1.csv"


def load_national_subtype_shares(start_year, region=None):
    """Season-level national H1/H3/B shares of total allocated positives,
    from ICL_NREVSS_1997_2026_clean_absolute1.csv (weekly national counts,
    1997-2026, no unsubtyped 'Subtyping not Performed' bucket to reallocate
    like the state-level load_subtype_shares needs -- TOTAL_ALLOCATED here
    already equals A(H1)+A(H3)+H3N2v+A(H5)+B exactly). H3N2v/A(H5) excluded
    (negligible), matching load_subtype_shares's convention. `region` is
    accepted (unused) only so this has the same call signature as
    load_subtype_shares for build_weekly_hosp_target's shares_fn param."""
    weeks = set(season_weeks(start_year))
    h1 = h3 = b = 0.0
    with open(NATIONAL_SUBTYPE_PATH) as f:
        for row in csv.DictReader(f):
            if (int(row["YEAR"]), int(row["WEEK"])) not in weeks:
                continue
            h1 += float(row["A (H1)"])
            h3 += float(row["A (H3)"])
            b += float(row["B"])
    total = h1 + h3 + b
    return {"H1": h1 / total, "H3": h3 / total, "B": b / total}


def build_weekly_target(start_year, region=REGION):
    """Weekly per-strain ascertained-incidence proxy:
    target_m(week) = ILI%(week) * positivity%(week) * subtype_share_m(season).

    Returns (week_list, target) where week_list is a sorted list of (year,week)
    and target is an (n_weeks, 3) array in STRAINS order.
    """
    ili = load_ili_weekly(start_year, region)
    positivity = load_positivity_weekly(start_year, region)
    shares = load_subtype_shares(start_year, region)

    weeks = [wk for wk in season_weeks(start_year) if wk in ili and wk in positivity]
    target = np.zeros((len(weeks), len(STRAINS)))
    for i, wk in enumerate(weeks):
        base = ili[wk] * positivity[wk]
        for j, m in enumerate(STRAINS):
            target[i, j] = base * shares[m]
    return weeks, target


def load_children_vax_weekly(start_year, region=REGION):
    """Weekly cumulative % vaccinated, children 6mo-17y (judz-8etw), for the
    season starting `start_year`. Returns (dates, cum_pct) sorted by date."""
    season_label = f"{start_year}-{start_year + 1}"
    rows = []
    with open(CHILDREN_VAX_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            if (row["geographic_name"] == region
                    and row["influenza_season"] == season_label
                    and row["demographic_level"] == "Overall"
                    and row["indicator_category_label"] == "Yes"
                    and row["estimate"] != ""):
                d = dt.datetime.strptime(row["week_ending"][:10], "%Y-%m-%d").date()
                rows.append((d, float(row["estimate"])))
    rows.sort()
    dates = [d for d, _ in rows]
    cum_pct = np.array([v for _, v in rows])
    return dates, cum_pct


def load_adult_vax_monthly_2019_20():
    """Monthly cumulative % vaccinated, Texas adults 18+, 2019-20 season,
    population-weighted average of the Bexar County and City of Houston
    BRFSS local-area estimates. Missing/suppressed months ('—§') are
    treated as 0%. Returns (dates, cum_pct), dates = 15th of each month."""
    months = ["July", "August", "September", "October", "November",
              "December", "January", "February", "March", "April", "May"]
    month_num = {"July": 7, "August": 8, "September": 9, "October": 10,
                 "November": 11, "December": 12, "January": 1, "February": 2,
                 "March": 3, "April": 4, "May": 5}
    month_year = {m: (2019 if month_num[m] >= 7 else 2020) for m in months}

    wb = openpyxl.load_workbook(ADULT_VAX_XLSX_PATH, data_only=True)

    def series_for(sheet_name):
        rows = list(wb[sheet_name].iter_rows(values_only=True))
        data_row = next(row for row in rows if row[0] == "≥18 years")
        vals = []
        for i in range(len(months)):
            col = 2 + 2 * i
            v = data_row[col]
            vals.append(0.0 if isinstance(v, str) else float(v))
        return np.array(vals)

    bexar = series_for("TX-Bexar County")
    houston = series_for("TX-City of Houston")
    w_bexar = BEXAR_COUNTY_POP / (BEXAR_COUNTY_POP + HOUSTON_CITY_POP)
    w_houston = 1 - w_bexar
    combined = w_bexar * bexar + w_houston * houston

    dates = [dt.date(month_year[m], month_num[m], 15) for m in months]
    return dates, combined


def load_hosp_weekly(start_year, location=TX_FIPS, age_group="0-130"):
    """Weekly incident flu hospitalizations (raw counts, MIDAS flu-scenario
    modeling-hub target-data schema) for `location` (FIPS), season starting
    `start_year` (Oct 1 of start_year through Sep 30 of start_year+1). Returns
    (dates, counts) sorted by date. Seasons/locations with incomplete
    reporting will have gaps -- returned as-is, not backfilled."""
    lo = dt.date(start_year, 10, 1)
    hi = dt.date(start_year + 1, 9, 30)
    rows = []
    with open(HOSP_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            if (row["location"] == location and row["target"] == "inc hosp"
                    and row["age_group"] == age_group):
                d = dt.datetime.strptime(row["date"], "%Y-%m-%d").date()
                if lo <= d <= hi:
                    rows.append((d, float(row["observation"])))
    rows.sort()
    dates = [d for d, _ in rows]
    counts = np.array([v for _, v in rows])
    return dates, counts


def build_weekly_hosp_target(start_year, hosp_location=TX_FIPS, subtype_region=REGION, shares_fn=load_subtype_shares):
    """Weekly per-strain hospitalization-count proxy: observed weekly total
    hospitalizations x season-level subtype share. Returns (dates, target)
    where target is (n_weeks, 3) in STRAINS order. Only includes weeks
    actually present in the hospitalization series (real reporting gaps are
    not backfilled/assumed zero). Pass `shares_fn=load_national_subtype_shares`
    for national fitting (the state-level Public Health Labs file has no
    National rows at all)."""
    dates, counts = load_hosp_weekly(start_year, hosp_location)
    shares = shares_fn(start_year, subtype_region)
    target = counts[:, None] * np.array([shares[m] for m in STRAINS])[None, :]
    return dates, target


def load_adult_vax_weekly(start_year, region=REGION):
    """Weekly cumulative % vaccinated, adults 18+ (sw5n-wg2p), for the season
    starting `start_year`. This series is only sparsely populated (roughly
    monthly, rest blank) even within a season with weekly rows. Returns
    (dates, cum_pct) sorted by date, blanks dropped, duplicate week_ending
    rows deduplicated (first non-blank kept)."""
    season_label = f"{start_year}-{start_year + 1}"
    seen = {}
    with open(ADULT_VAX_WEEKLY_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            if not (row["geographic_name"] == region
                    and row["influenza_season"] == season_label
                    and row["vaccine"] == "FLU"
                    and row["demographic_level"] == "Overall"
                    and row["demographic_name"] == "18+ years"
                    and row["indicator_category_label"] == "Yes"):
                continue
            if row["estimates"] == "":
                continue
            d = dt.datetime.strptime(row["week_ending"][:10], "%Y-%m-%d").date()
            if d not in seen:
                seen[d] = float(row["estimates"])
    dates = sorted(seen)
    cum_pct = np.array([seen[d] for d in dates])
    return dates, cum_pct


def build_overall_vax_daily(start_year, day_grid, region=REGION, anchor_month_day=(10, 1)):
    """Whole-population daily cumulative %-vaccinated curve for `region`,
    season starting `start_year`, on `day_grid` (day offsets from
    `anchor_month_day` of `start_year`), population-weighting (per-state
    Census age shares, `get_pop_shares`) the weekly children's series
    (judz-8etw) and weekly (sparse pre-2023-24) adult series (sw5n-wg2p).
    Both must cover `start_year`'s season (sw5n-wg2p only covers 2021-22
    onward)."""
    anchor = dt.date(start_year, *anchor_month_day)

    child_dates, child_cum = load_children_vax_weekly(start_year, region)
    child_days = np.array([(d - anchor).days for d in child_dates], dtype=float)
    child_daily = np.interp(day_grid, child_days, child_cum, left=child_cum[0], right=child_cum[-1])

    adult_dates, adult_cum = load_adult_vax_weekly(start_year, region)
    adult_days = np.array([(d - anchor).days for d in adult_dates], dtype=float)
    adult_daily = np.interp(day_grid, adult_days, adult_cum, left=adult_cum[0], right=adult_cum[-1])

    child_share, adult_share = get_pop_shares(region)
    return child_share * child_daily + adult_share * adult_daily


def build_overall_vax_daily_from_master(start_year, day_grid, region=REGION, anchor_month_day=(10, 1)):
    """Whole-population daily cumulative %-vaccinated curve for `region`,
    season starting `start_year`, read directly from the merged
    `vaccination/state_weekly_vaccination_2020_2026.csv` (`weighted_cum_pct`
    column) instead of re-deriving the child/adult blend per source. Works
    uniformly across 2020-21 through 2025-26, regardless of whether that
    season's rows are real (`cdc_direct`) or reconstructed
    (`scenario_hub_reconstructed`, see MODEL.md) -- the merge already
    resolved provenance so every season has exactly one series."""
    season_label = f"{start_year}-{start_year + 1}"
    anchor = dt.date(start_year, *anchor_month_day)

    dates, cum_pct = [], []
    with open(MASTER_VAX_PATH) as f:
        for row in csv.DictReader(f):
            if row["state"] == region and row["season"] == season_label:
                d = dt.datetime.strptime(row["week_ending"], "%Y-%m-%d").date()
                dates.append(d)
                cum_pct.append(float(row["weighted_cum_pct"]))
    order = np.argsort(dates)
    dates = [dates[i] for i in order]
    cum_pct = np.array(cum_pct)[order]

    days = np.array([(d - anchor).days for d in dates], dtype=float)
    return np.interp(day_grid, days, cum_pct, left=cum_pct[0], right=cum_pct[-1])


def build_overall_vax_daily_2019_20(day_grid, season_start_year=2019, anchor_month_day=(10, 1)):
    """Whole-population daily cumulative %-vaccinated curve for the 2019-20
    Texas season, on `day_grid` (day offsets from `anchor_month_day` of
    `season_start_year`), built by population-weighting (per-state Census
    age shares, `get_pop_shares`) the two available series:
      - children 6mo-17y: weekly, statewide (judz-8etw)
      - adults 18+: monthly, Bexar County + City of Houston only (BRFSS)
    Texas-only (the adult proxy has no other-state equivalent). Returns
    cum_pct (same shape as day_grid, in percent)."""
    anchor = dt.date(season_start_year, *anchor_month_day)

    child_dates, child_cum = load_children_vax_weekly(season_start_year)
    child_days = np.array([(d - anchor).days for d in child_dates], dtype=float)
    child_daily = np.interp(day_grid, child_days, child_cum, left=child_cum[0], right=child_cum[-1])

    adult_dates, adult_cum = load_adult_vax_monthly_2019_20()
    adult_days = np.array([(d - anchor).days for d in adult_dates], dtype=float)
    adult_daily = np.interp(day_grid, adult_days, adult_cum, left=adult_cum[0], right=adult_cum[-1])

    child_share, adult_share = get_pop_shares("Texas")
    return child_share * child_daily + adult_share * adult_daily


def aggregate_at_dates(daily_values, day_grid, target_dates, anchor_date):
    """Sum `daily_values` (n_days, ...) over the 7-day window ending at each of
    `target_dates` (calendar dates), using `day_grid` (day offsets from
    `anchor_date`) to locate each window. Handles sparse/gapped target_dates
    (unlike a fixed contiguous weekly binning from day 0)."""
    day_grid = np.asarray(day_grid)
    out = np.zeros((len(target_dates),) + daily_values.shape[1:])
    for i, d in enumerate(target_dates):
        end_day = (d - anchor_date).days
        mask = (day_grid > end_day - 7) & (day_grid <= end_day)
        out[i] = daily_values[mask].sum(axis=0)
    return out


def hazard_from_daily_cum(daily_cum_pct):
    """Convert an already-daily cumulative %-vaccinated curve into a daily
    hazard rate mu(t) [day^-1]: mu(t) = -ln(1 - dV/(1-V(t))), clipped to a
    nonnegative, finite rate. `daily_cum_pct` in percent (0-100)."""
    cum_frac = np.clip(np.array(daily_cum_pct) / 100.0, 0, 0.999999)

    mu = np.zeros_like(cum_frac)
    for k in range(len(cum_frac) - 1):
        v, v_next = cum_frac[k], cum_frac[k + 1]
        remaining = 1 - v
        dv = v_next - v
        if remaining <= 1e-9 or dv <= 0:
            mu[k] = 0.0
        else:
            ratio = dv / remaining
            mu[k] = -np.log(max(1 - ratio, 1e-9))
    mu[-1] = mu[-2] if len(mu) > 1 else 0.0
    return np.clip(mu, 0, None)


def cumulative_to_daily_hazard(dates, cum_pct, day_grid):
    """Convert a cumulative %-vaccinated series (irregular dates) into a daily
    hazard rate mu(t) [day^-1] on `day_grid` (day offsets from dates[0]), via
    linear interpolation to daily resolution then `hazard_from_daily_cum`."""
    date_days = np.array([(d - dates[0]).days for d in dates], dtype=float)
    daily_cum = np.interp(day_grid, date_days, cum_pct, left=cum_pct[0], right=cum_pct[-1])
    return hazard_from_daily_cum(daily_cum)


def load_subtype_ve_quick_reference():
    """Season-start-year -> {'H1': (adult_pct, child_pct), 'H3': (...), 'B': (...)},
    from the "Quick reference" sheet of
    vaccination/influenza_subtype_VE_by_season_with_children.xlsx -- CDC
    outpatient vaccine-effectiveness estimates by season/subtype/age group,
    already resolved to one number per cell (interim-vs-final, multi-network
    averaging, etc. already decided in the workbook itself; see its
    "Model-use notes" sheet). Values are percentages (0-100), not fractions."""
    wb = openpyxl.load_workbook(VE_XLSX_PATH, data_only=True)
    ws = wb["Quick reference"]
    rows = list(ws.iter_rows(values_only=True))
    out = {}
    for row in rows[1:]:
        season_label = row[0]   # e.g. "2021–22"
        start_year = int(str(season_label)[:4])
        out[start_year] = {
            "H1": (row[1], row[4]),
            "H3": (row[2], row[5]),
            "B": (row[3], row[6]),
        }
    return out


def build_season_alpha(season_start_year, region=REGION):
    """Population-weighted (child/adult) current-season vaccine
    effectiveness alpha_m, as fractions in [0,1], for `season_start_year`
    -- from the CDC VE-by-season workbook (`load_subtype_ve_quick_reference`),
    blended using the same per-state child/adult population shares as the
    vaccination-coverage loaders (`get_pop_shares`)."""
    ve = load_subtype_ve_quick_reference()[season_start_year]
    child_share, adult_share = get_pop_shares(region)
    return {m: (adult_share * ve[m][0] + child_share * ve[m][1]) / 100.0 for m in STRAINS}
