# Flu Subtype Weekly Share Forecast Workflow (Season 2026)

Forecasts **weekly** shares of A(H1), A(H3), and B across the entire 2026
flu season (2026 wk40 → 2027 wk39, 52 weeks) via a Markov transition model
on dominant subtype + Monte Carlo simulation of weekly trajectories within
each dominant scenario.

Script: `forecast_2026.py` (in this folder, runnable as-is).
Inputs: `ICL_NREVSS_1997_2026_season_summary.csv`, `ICL_NREVSS_1997_2026_clean_absolute1.csv`.
Outputs: `flu_2026_H1_weekly_share_simulations.csv`,
`flu_2026_H3_weekly_share_simulations.csv`,
`flu_2026_B_weekly_share_simulations.csv`.

## 1. Data inputs

- **`ICL_NREVSS_1997_2026_season_summary.csv`** — one row per historical
  season (1997–2025, 29 seasons), with `SEASON`, `START_YEAR/WEEK`,
  `END_YEAR/WEEK`, `DOMINANT_SUBTYPE_LINEAGE` (`A (H1)`, `A (H3)`, or `B`),
  `DOMINANT_COUNT`, `TOTAL_ALLOCATED`.
- **`ICL_NREVSS_1997_2026_clean_absolute1.csv`** — weekly national counts by
  subtype (`A (H1)`, `A (H3)`, `H3N2v`, `A (H5)`, `B`) and `TOTAL_ALLOCATED`.

Historical dominant-subtype sequence (1997–2025): 10 seasons H1-dominant, 18
H3-dominant, **1 season B-dominant (2002 only)**. The current season (2025,
data available only through 2026 wk32 — 46 of its ~52 weeks) is
H3-dominant; this is the conditioning variable for the 2026 forecast.

**Season length note**: seasons are normally 52 weeks (MMWR wk40 →
wk39 of the next year). A small number of seasons run 53 weeks because
their start year has an extra MMWR week: 1997, 2003, 2008, 2014, 2020 (and
2025 will too, once complete). These follow a ~5–6 year cycle, so **2026 is
a normal 52-week season** (2026 wk40 → 2027 wk39).

## 2. Season-level aggregate share (used only to fit the Gaussians in §6)

```
share(season, X) = sum(count_X over that season's weeks) / sum(TOTAL_ALLOCATED over that season's weeks)
```

Verified to exactly reproduce `DOMINANT_COUNT / TOTAL_ALLOCATED` from
`season_summary.csv` for each season's dominant subtype; the same method is
used to back out the two non-dominant subtypes' season shares.

## 3. Transition matrix (dominant subtype → next season's dominant subtype)

Built from the 28 consecutive season pairs (1997→1998, …, 2024→2025). Row
needed here (previous = 2025's dominant, H3):

| prev dominant | → H1 | → H3 | → B | n |
|---|---|---|---|---|
| H3 | 7 | 9 | 1 | 17 |

## 4. Predicted P(dominant subtype in 2026) given 2025 = H3

MLE, no smoothing needed (no zero cells): P(H1) = 7/17 ≈ **41.2%**,
P(H3) = 9/17 ≈ **52.9%**, P(B) = 1/17 ≈ **5.9%**.

## 5. Allocating 300 simulations across dominant cases

`n_X = round(300 × P(X))`, rounding remainder added to/subtracted from the
**largest** group so the total is exactly 300.

Result: **H1 = 124, H3 = 158, B = 18** simulations.

## 6. Gaussian fit for each dominant subtype's own season-aggregate share

Pooling `share(season, X)` (§2) across every historical season where *X*
was dominant:

| dominant | n | mean | std | std source |
|---|---|---|---|---|
| H1 | 10 | 0.621 | 0.173 | empirical |
| H3 | 18 | 0.740 | 0.169 | empirical |
| B | 1 | 0.415 | 0.105 | **borrowed** (see below) |

**B has only one historical observation (2002)**, so its standard deviation
can't be estimated empirically. The coefficient of variation (std/mean) is
averaged across the H1 and H3 groups and applied to B's single-point mean:
`std_B = mean(CV_H1, CV_H3) × mean_B`. Treat B-dominant simulation spread
with extra caution — it's a modeling assumption, not an empirical estimate.

This Gaussian only supplies each simulation's **target season-aggregate
share for its dominant subtype** — the actual output is a full weekly curve
(§7–8), not this scalar.

## 7. Weekly-shape template library

To generate realistic within-season dynamics (rise/peak/fall timing), each
simulation borrows an entire historical season's **weekly shape**, not just
a single aggregate number:

- For every historical season *except* the current incomplete one (2025)
  and trimmed to its **first 52 weeks** (any 53-week season simply drops
  its trailing 53rd week — same rule agreed for handling irregular-length
  seasons), build a week-by-week `(H1_share, H3_share, B_share)` sequence,
  renormalized each week to sum to 1 (dropping the negligible `H3N2v`/`A
  (H5)` categories).
- The current season (2025) is excluded from this template library only
  (it's missing its final ~6 weeks and can't supply a full 52-week shape),
  but it's still used normally in the transition matrix (§3) and its own
  season-aggregate share is still used in the H3 Gaussian (§6).
- Also record each template's own weekly `TOTAL_ALLOCATED`, normalized to
  sum to 1 across the 52 weeks, as weights — used to compute a
  volume-weighted aggregate from the weekly curve (so it matches how the
  real season aggregate is actually computed, not a naive average).

Library sizes: H1 = 10 seasons, H3 = 17 seasons (18 minus the excluded
2025), B = 1 season (2002).

## 8. Simulating each scenario's weekly trajectory

For each of the 300 simulations (in its assigned dominant group `X`):

1. Draw `target_share_X ~ Normal(mean_X, std_X)` (§6), clipped to `(0, 1)`.
2. Bootstrap (sample with replacement) **one historical season's whole
   weekly template** from group `X`'s library (§7) — this one template
   supplies both the shape and the weekly volume weights.
3. Compute that template's own volume-weighted aggregate dominant share,
   `template_agg_X`, and a scale factor `k = target_share_X / template_agg_X`.
4. For every week `i`, scale the dominant subtype's weekly share by `k`
   (clipped to `(0,1)`), then split the remainder `1 - new_share_X` between
   the two non-dominant subtypes **in that week's original historical
   ratio**:
   ```
   new_share_dom_i = clip(hist_share_dom_i * k)
   remainder_i      = 1 - new_share_dom_i
   ratio_i          = hist_share_A_i / (hist_share_A_i + hist_share_B_i)
   new_share_A_i    = remainder_i * ratio_i
   new_share_B_i    = remainder_i * (1 - ratio_i)
   ```

This scales the borrowed season's real epidemic curve up/down to hit the
Gaussian-drawn season-aggregate target (by construction, the volume-weighted
average of the new weekly dominant curve equals `target_share_X` exactly),
while preserving its historical week-to-week shape and the realistic
co-movement between the two non-dominant subtypes. All three shares sum to
exactly 1 in every week of every simulation.

## 9. Output

Three CSVs, each with **300 simulations × 52 weeks = 15,600 rows**:

```
simulation_id, dominant_case, source_season, year, week, share
```

- `flu_2026_H1_weekly_share_simulations.csv`
- `flu_2026_H3_weekly_share_simulations.csv`
- `flu_2026_B_weekly_share_simulations.csv`

`dominant_case` records which of the three scenarios produced that
simulation; `source_season` records which historical season's shape was
bootstrapped for it, so results can be traced back or filtered.

## 10. Caveats

- Transition matrix uses only 28 season-pairs total; the H3 row (n=17) used
  here is reasonably sized, but the model would be fragile if conditioned on
  a different previous-season dominant subtype (e.g. the B row has n=1).
- B-dominant share spread is a borrowed-variance assumption (§6), and its
  weekly shape library has only one season (2002) to bootstrap from — every
  B-dominant simulation shares the same underlying shape, just rescaled.
- The 5 trimmed 53-week season templates lose their true final week (week
  39) and instead retain their spurious extra week (week 53) mid-sequence;
  a minor approximation accepted for template alignment simplicity.
- `random.seed(42)` is fixed in the script for reproducibility; remove/change
  it to get different draws across runs.
