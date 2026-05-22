# count_lt10 Fallback Usage  HHS Regions

_Generated: 2026-05-22_  
Source: CDC SARS-CoV-2 Variant Proportions API (`jr58-6ysp`).  
Output files: `variant_hhs1.csv`, `variant_hhs_imputed1.csv`, `variant_hhs_marker1.csv`, and per-region `variant_hhs_marker1_region_XX.csv`.

## Precedence rule

Each `(HHS region, variant, week_ending)` triple is filled with the highest-confidence source available:

1. **Primary**  rows with `count_lt10` in `{0, 0.0}` (>=10 sequenced specimens; tight CIs).
2. **Fallback**  rows with `count_lt10` in `{1, 1.0}` (<10 specimens; wider CIs). Used only when no primary row exists for that triple.

After the precedence filter and per-snapshot aggregation, each `(region, variant, week_ending)` appears exactly once. The retained `nchs_or_count_flag` column on each row indicates which tier it came from (0 = primary, 1 = fallback, NaN = imputed in `variant_hhs_imputed1.csv`).

## Per-region summary

| Region | Date range (full) | Primary ceiling (last `flag=0` week) | Latest week (with fallback) | Fallback weeks | Fallback rows | Fallback variants |
|--------|-------------------|--------------------------------------|-----------------------------|----------------|---------------|--------------------|
| region_01 | 2021-01-30  2026-04-11 | 2026-02-14 | 2026-04-11 | 181 | 8547 | 144 |
| region_02 | 2021-01-30  2026-04-11 | 2026-03-14 | 2026-04-11 | 181 | 8226 | 139 |
| region_03 | 2021-01-30  2026-04-11 | 2025-09-27 | 2026-04-11 | 181 | 8277 | 139 |
| region_04 | 2021-01-30  2026-04-11 | 2025-08-30 | 2026-04-11 | 181 | 8228 | 136 |
| region_05 | 2021-01-30  2026-04-11 | 2026-01-17 | 2026-04-11 | 181 | 8247 | 142 |
| region_06 | 2021-01-30  2026-04-11 | 2026-02-14 | 2026-04-11 | 181 | 8473 | 141 |
| region_07 | 2021-01-30  2026-04-11 | 2026-03-14 | 2026-04-11 | 181 | 8645 | 145 |
| region_08 | 2021-01-30  2026-04-11 | 2026-01-17 | 2026-04-11 | 181 | 8589 | 145 |
| region_09 | 2021-01-30  2026-04-11 | 2026-03-14 | 2026-04-11 | 181 | 7735 | 143 |
| region_10 | 2021-01-30  2026-03-14 | 2025-10-25 | 2026-03-14 | 180 | 8621 | 142 |

## Per-region fallback weeks

`Beyond ceiling` = weeks where no primary data exists for any variant (so fallback drives the row count forward in time).  
`Within ceiling` = weeks where some variants have primary data but specific rare variants only have fallback rows.

### region_01

- Primary ceiling: **2026-02-14**
- Fallback weeks **beyond** ceiling (2): 2026-03-14, 2026-04-11
- Fallback weeks **within** ceiling (179 total): 2021-01-30  2026-02-14

### region_02

- Primary ceiling: **2026-03-14**
- Fallback weeks **beyond** ceiling (1): 2026-04-11
- Fallback weeks **within** ceiling (180 total): 2021-01-30  2026-03-14

### region_03

- Primary ceiling: **2025-09-27**
- Fallback weeks **beyond** ceiling (7): 2025-10-25, 2025-11-22, 2025-12-20, 2026-01-17, 2026-02-14, 2026-03-14, 2026-04-11
- Fallback weeks **within** ceiling (174 total): 2021-01-30  2025-09-27

### region_04

- Primary ceiling: **2025-08-30**
- Fallback weeks **beyond** ceiling (8): 2025-09-27, 2025-10-25, 2025-11-22, 2025-12-20, 2026-01-17, 2026-02-14, 2026-03-14, 2026-04-11
- Fallback weeks **within** ceiling (173 total): 2021-01-30  2025-08-30

### region_05

- Primary ceiling: **2026-01-17**
- Fallback weeks **beyond** ceiling (3): 2026-02-14, 2026-03-14, 2026-04-11
- Fallback weeks **within** ceiling (178 total): 2021-01-30  2026-01-17

### region_06

- Primary ceiling: **2026-02-14**
- Fallback weeks **beyond** ceiling (2): 2026-03-14, 2026-04-11
- Fallback weeks **within** ceiling (179 total): 2021-01-30  2026-02-14

### region_07

- Primary ceiling: **2026-03-14**
- Fallback weeks **beyond** ceiling (1): 2026-04-11
- Fallback weeks **within** ceiling (180 total): 2021-01-30  2026-03-14

### region_08

- Primary ceiling: **2026-01-17**
- Fallback weeks **beyond** ceiling (3): 2026-02-14, 2026-03-14, 2026-04-11
- Fallback weeks **within** ceiling (178 total): 2021-01-30  2026-01-17

### region_09

- Primary ceiling: **2026-03-14**
- Fallback weeks **beyond** ceiling (1): 2026-04-11
- Fallback weeks **within** ceiling (180 total): 2021-01-30  2026-03-14

### region_10

- Primary ceiling: **2025-10-25**
- Fallback weeks **beyond** ceiling (5): 2025-11-22, 2025-12-20, 2026-01-17, 2026-02-14, 2026-03-14
- Fallback weeks **within** ceiling (175 total): 2021-01-30  2025-10-25
