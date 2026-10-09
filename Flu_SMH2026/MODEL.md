# Three-Strain Influenza Model — Structure, Data Sources, and Parameters

This documents the current state of the pipeline in `epi_model/flu/`:
`three_strain_model.py` (core model), `data_prep.py` (all data loaders),
`fit_texas_hosp_2021.py`/`fit_texas_chain.py` (Texas, 2021-22 season and
the chain that fits forward through 2025-26 via the real Algorithm 10
remap -- see §5), and `fit_national_hosp_2021.py`/`fit_national_chain.py`
(the same methodology run against US national data instead -- see §6;
both geographies are maintained side by side, neither replaces the
other). It reproduces `hill_three_strain_influenza_algorithm.md`
Algorithms 1-11, extended with Hospitalization (H) and Death (D)
compartments downstream of Infectious (I) -- as of §5.5, `I` and `H` are
each split into two branches (`I_R`/`I_H` and `H_R`/`H_D`) with their own
progression rates rather than one shared rate split by probability; see
§1 for the current structure and §5.5 for why.

An earlier calibration attempt fit the same model against Texas ILI data for
the 2019-20 season (`fit_texas_2019.py`); that script no longer exists in
the working tree, but the ILI-based method it used is noted where relevant
for context, since `data_prep.py` still carries the loaders it needed.

---

## 1. Model Structure

### 1.1 Population bookkeeping

Every individual is cross-classified by two axes:

- **Previous-season exposure history** `h` — 8 groups:
  `N` (naive), `H1`, `H3`, `B` (infected with that strain last season,
  unvaccinated), `V` (vaccinated only), `H1V`, `H3V`, `BV` (infected +
  vaccinated combinations).
- **Current-season vaccination status** `X` — `N` (unvaccinated) or `V`
  (vaccinated).

Susceptibles `S^{X,h}` are cross-tabulated by both axes (shape `(2, 8)`).
Disease-progression compartments (`E`, `I_R`, `I_H`, `H_R`, `H_D`, `R`, `D`)
are tracked per strain `m ∈ {H1, H3, B}` and per current-season status `X`,
shape `(2, 3)` each — **not** further split by `h`, since prior-season
history only matters through susceptibility, not infection dynamics. Total
state size: `16 + 6×7 = 58`. (`I` and `H` are each split into two branches
— see §1.4/§5.5 — rather than the single `I`/`H` compartments this model
started with; total state size was `16 + 6×5 = 46` before that split.)

There is **one shared susceptible pool** per `(X, h)` group exposed to all
three strain-specific forces of infection — not three separate
sub-populations — so H1/H3/B compete for the same susceptibles rather than
each having an independent copy of the population.

**Units — proportions vs. headcounts.** The module defaults to
`N_TOTAL=1.0` (every compartment a *fraction* of the population, matching
the algorithm doc's "prefer population proportions"). `set_population(n)`
switches `N_TOTAL` to a real headcount instead (e.g. `fit_texas_hosp_2021.py`
calls `set_population(TX_POP)` so compartments are literal numbers of
people, directly comparable to raw hospitalization counts). This is safe
everywhere in the code because `I0` (Algorithm 11's seed) is scaled by
`N_TOTAL` in `seed_infections`, and the force of infection is normalized by
`N_TOTAL` (§1.3) — both proportional to population scale, so results are
identical up to that scale factor either way.

### 1.2 Susceptibility matrix f(h, m)

`f(h, m)` = relative susceptibility to current strain `m` given
previous-season history `h` (`build_f_matrix` in `three_strain_model.py`):

| h \ m | H1 | H3 | B |
|---|---:|---:|---:|
| N | 1 | 1 | 1 |
| H1 | a | 1 | 1 |
| H3 | 1 | a | 1 |
| B | 1 | 1 | a |
| V | c_H1 | c_H3 | c_B |
| H1V | min(a,c_H1) | c_H3 | c_B |
| H3V | c_H1 | min(a,c_H3) | c_B |
| BV | c_H1 | c_H3 | min(a,c_B) |

where `c_{m} = 1 - ξ·α_m` (residual susceptibility from last season's
vaccine). Same-strain prior infection gives the strongest protection (`a`);
different-strain history gives none (`f=1`); prior vaccination gives weak
carryover protection (`c_m` close to 1, since `ξ` is tiny).

Since `α_m` became season-specific (§5.6), `α_m` here uses the *current*
season's own value, not the previous season's, even though `c_m`
conceptually describes residual protection from *last* season's vaccine —
a deliberate simplification, not an oversight: because `ξ = 0.0051` is so
small, `c_m` only ranges from 0.9949 (α=1) to 1.0 (α=0) regardless of which
season's α is plugged in, so which season's value is used here is
numerically negligible (unlike `α_m`'s other role in §1.3, where it enters
undamped).

### 1.3 Force of infection and infection flow

```
I_m^{tot}(t) = I_m^{R,N}(t) + I_m^{R,V}(t) + I_m^{H,N}(t) + I_m^{H,V}(t)   [total currently-infectious]
λ_m(t) = β_m · I_m^{tot}(t) / N_TOTAL                          [per strain]

F_{m,h}^N(t) = f(h,m) · S^{N,h}(t) · λ_m(t)                  [unvaccinated]
F_{m,h}^V(t) = f(h,m) · S^{V,h}(t) · (1-α_m) · λ_m(t)        [vaccinated: leaky protection]
```
(`compute_flows` in `three_strain_model.py`.) `I_m^{tot}` sums *both*
recovery-bound (`I_R`) and hospitalization-bound (`I_H`) infectious people
— they're equally infectious, only their eventual outcome differs (§1.4).
The `/N_TOTAL` is Algorithm 4's counts-based form ("if counts rather than
proportions are used, divide the infectious term by N"); with the default
`N_TOTAL=1` it's a no-op, so this is the same formula either way `N_TOTAL`
is set.

### 1.4 Compartment flows (per strain m, per status X)

`I` is split into `I_R` (recovery-bound) and `I_H` (hospitalization-bound)
*at the moment of leaving E*, and `H` is split into `H_R` (recovery-bound)
and `H_D` (death-bound) *at the moment of leaving `I_H`* — each branch has
its own progression rate (`γ2`/`γ_ih` for I_R/I_H, `γ3`/`γ_hd` for H_R/H_D)
rather than one shared exit rate split post hoc by `p_H`/`p_D`. The
original shared-rate design (`I --γ2·(1-p_H)--> R`, `I --γ2·p_H--> H`, same
for `H`) implicitly forced hospitalization-bound illness and
direct-recovery illness to have the *same* duration distribution, which
isn't necessarily realistic; branching at entry lets each pathway run on
its own independently-calibrated duration (§5.5).

```
dS^{N,h}/dt = -Σ_m F_{m,h}^N - μ(t)·S^{N,h}
dS^{V,h}/dt = -Σ_m F_{m,h}^V + μ(t)·S^{N,h}

dE_m^N/dt = F_m^N - γ1_m·E_m^N - μ(t)·E_m^N            dE_m^V/dt = F_m^V - γ1_m·E_m^V + μ(t)·E_m^N

dI_{R,m}^N/dt = γ1_m·(1-p_H)·E_m^N - γ2·I_{R,m}^N - μ(t)·I_{R,m}^N      [+ symmetric V terms, all rows]
dI_{H,m}^N/dt = γ1_m·p_H·E_m^N - γ_ih·I_{H,m}^N - μ(t)·I_{H,m}^N

dH_{R,m}^N/dt = γ_ih·(1-p_D)·I_{H,m}^N - γ3·H_{R,m}^N - μ(t)·H_{R,m}^N
dH_{D,m}^N/dt = γ_ih·p_D·I_{H,m}^N - γ_hd·H_{D,m}^N - μ(t)·H_{D,m}^N

dR_m^N/dt = γ2·I_{R,m}^N + γ3·H_{R,m}^N - μ(t)·R_m^N
dD_m^N/dt = γ_hd·H_{D,m}^N
```
`μ(t)` is the current-season vaccination hazard rate (0 by default; a
pluggable, generally data-derived, function of continuous time — see §3.3).
`D` is a permanent sink: it has no outflow and no `μ`-driven transfer.

Integrated with `scipy.integrate.solve_ivp` (adaptive RK45, `max_step=1`
day, `rtol=1e-8`, `atol=1e-10`) rather than fixed-step Euler, because
`γ1_B ≈ 1.67 day⁻¹` is fast enough that a 1-day Euler step can drive `I_B`
negative.

### 1.5 Between-season transition (Algorithms 10-11)

At each season boundary (`end_of_season_remap`), this season's end state is
relabeled into next season's history groups, and current-season vaccination
status resets to `X=N`:

```
new_S[N=uninfected,unvaccinated]  = Σ_h S^{N,h}(T)
new_S[V=uninfected,vaccinated]    = Σ_h S^{V,h}(T)
new_S[m =infected-with-m, unvacc] = E_m^N(T) + I_{R,m}^N(T) + I_{H,m}^N(T) + H_{R,m}^N(T) + H_{D,m}^N(T) + R_m^N(T)
new_S[mV=infected-with-m, vacc]   = E_m^V(T) + I_{R,m}^V(T) + I_{H,m}^V(T) + H_{R,m}^V(T) + H_{D,m}^V(T) + R_m^V(T)
```
`D` is deliberately excluded — cumulative flu deaths do not re-enter the
living population carried into next season (population can shrink
season-over-season). A small infectious seed (`I0` per strain) is then
subtracted from the naive pool and injected into `I_R^N`/`I_H^N` (split by
`p_H`, same as any cohort leaving `E` — a seeded individual's eventual
outcome is just as undetermined as anyone else's), at the start of *every*
season including season 1 (`seed_infections`).

Population conservation is checked *within* one season's trajectory
(start-of-season total == end-of-season total including `D`), not against
a fixed constant, since the constant itself legitimately decreases
season-over-season.

### 1.6 Outputs

- **Infection incidence** (Algorithm 8): `C_m(t) = F_m^N(t) + F_m^V(t)`,
  aggregated to weekly `Z_{m,w}` (`compute_daily_incidence` /
  `aggregate_weekly`). Used by the (now-removed) ILI-based fit.
- **Hospitalization incidence**: `γ_ih·(I_{H,m}^N(t)+I_{H,m}^V(t))`
  (`compute_daily_hosp_incidence`) — the inflow from `I_H` into `H`. No
  `p_H` multiplication here (unlike before the §1.4 split): that
  probability split already happened upstream, at `E`→`I_H`. Used by the
  current hospitalization fit.

---

## 2. Data Sources

All under `/Users/boyapeng/Desktop/Dissertation/Aim2/Data/Flu` unless noted.

| File | Provides | Used for |
|---|---|---|
| `FluViewPhase2Data/ILINet.csv` | Weekly `%UNWEIGHTED ILI` by state (state-level `%WEIGHTED ILI` is unreported) | (Former ILI-based fit only) |
| `FluViewPhase2Data/ICL_NREVSS_Clinical_Labs.csv` | Weekly `%A`/`%B` positivity by state, 2015-2026 | (Former ILI-based fit only) |
| `FluViewPhase2Data/ICL_NREVSS_Public_Health_Labs.csv` | **Season-level** (not weekly) H1/H3/B/BVic/BYam subtype counts by state | Subtype-share allocation (`load_subtype_shares`), both fits |
| `FluViewPhase2Data/ICL_NREVSS_Combined_prior_to_2015_16.csv` | Weekly subtype counts by state, but **only seasons through 2014-15** | Not usable for 2018-19 onward; superseded by combining the two files above |
| `vaccination/cdc_judz-8etw_flu_vaccination.csv` | Weekly cumulative %-vaccinated, **children 6mo-17y**, statewide, 2019-20 onward | Children series in the population-weighted `μ(t)` blend |
| `vaccination/cdc_sw5n-wg2p_flu_vaccination.csv` | Weekly cumulative %-vaccinated, **adults 18+**, statewide, **2021-22 onward** (real weekly cadence only from **2023-24 onward** — 2021-22/2022-23 have just ~9 real monthly points within the weekly schema, rest blank/interpolated) | Adult series in the population-weighted `μ(t)` blend (current, 2021-22 fit) |
| `vaccination/CDC_FluVaxView_2019-20_AdultsLocalAreas.xlsx` | Monthly adult coverage, Bexar County + City of Houston only, 2019-20 | (Former ILI-based fit only — adult proxy before sw5n-wg2p covered that season) |
| `epi_model/flu/time-series.csv` | Weekly incident flu hospitalizations by state (MIDAS flu-scenario-modeling-hub target-data schema) | Hospitalization fitting target and 2020-21 initial-state source |
| `Data/pop_risk.csv` | Per-state population by age band (`0_17_All`, `18_49_All`, `50_64_All`, `65+_All`, `pop_total`), 52 rows (50 states + DC + US) | Per-state child (0-17)/adult (18+) population shares (`data_prep.get_pop_shares`), replacing the old Texas-only 27.3%/72.7% constant — generalizes `build_overall_vax_daily` to any state |
| `vaccination/state_weekly_vaccination_2023_2026.csv` | Derived: population-weighted whole-population weekly cumulative %-vaccinated, all 51 states, 2023-24/2024-25/2025-26 (2025-26 partial, ongoing) | Reference/QA output — not yet wired into any fit script; built by `vaccination/build_state_weekly_vax.py` |
| `vaccination/RD1_2022_23_Sc_A_B_C_D.csv`, `RD2_2022_23_Sc_A_B_C_D.csv`, `RD4_2023_24_Sc_A_B_C_D_E_F.csv`, `RD5_2024_25_Sc_A_B_C_D_E_F.csv`, `RD6_2025_26_Sc_A_B.csv` | MIDAS flu-scenario-modeling-hub **vaccination scenario** files — each round's scenario columns are a real historical season's coverage curve times a known constant (state × 6 age bins × 44 weeks/season) | Source for reconstructing real 2020-21/2021-22/2022-23 coverage that no direct CDC series otherwise covers at state/age resolution (see below) |
| `vaccination/flu-scenario-model-hub_2020-2024.csv` | Derived: RD1/RD2/RD5/RD6's scenario scaling inverted back out to real coverage, re-dated onto the true historical calendar (MMWR-week-preserving, not naive year-shift), all age bins kept unaggregated, for 2020-21 through 2023-24 | Intermediate — see reconstruction method below |
| `vaccination/state_weekly_vaccination_2020_2026.csv` | Derived: `state_weekly_vaccination_2023_2026.csv`'s real (`cdc_direct`) 2023-24/2024-25/2025-26 rows, merged with the scenario-hub file's 2020-21/2021-22/2022-23 rows aggregated to child/adult (never both for the same season — 2023-24 stays `cdc_direct` only) | The full 2020-21→2025-26 whole-population weekly vaccination series; not yet wired into any fit script |
| `vaccination/influenza_subtype_VE_by_season_with_children.xlsx` | CDC outpatient vaccine-effectiveness (VE) estimates by season/subtype/age group (2018-19 through 2025-26), already resolved to one number per cell (interim-vs-final, multi-network averaging already decided — see its own "Model-use notes" sheet) | Season-specific `α_m` (current-season VE), replacing the fixed Hill-derived defaults (`data_prep.build_season_alpha`, §5.6) |

### Reconstructing real 2020-21/2021-22/2022-23 vaccination from the scenario-hub files

No direct CDC series gives state-level, age-resolved weekly vaccination
coverage before 2023-24 (adults) — but the MIDAS flu-scenario-modeling-hub's
own vaccination-scenario files (RD1/RD2/RD4/RD5/RD6, built for *future*
scenario projections) each construct their scenarios by multiplying a real
historical season's coverage curve by a known constant, so the real curve
is exactly recoverable by inverting that constant:

| Round (labeled season) | Scenario columns | Real base season recovered | Inversion |
|---|---|---|---|
| RD1 (2022-23) | `sc_A_B` (110%), `sc_C_D` (90%) | **2020-21** | `base = sc_A_B/1.10` (verified exactly equal to `sc_C_D/0.90`, every state/age/week checked) |
| RD2 (2022-23) | one unscaled column | **2021-22** | direct (primary source) |
| RD4 (2023-24) | `sc_A_B` (120%), `sc_C_D` (100%), `sc_E_F` (80%) | **2021-22** | `sc_C_D` — cross-check only, matches RD2 to <0.5% (residual is interpolation-grid rounding); not included in the output to avoid duplicate rows |
| RD5 (2024-25) | `sc_A_B` (120%), `sc_C_D` (100%), `sc_E_F` (80%) | **2022-23** | `sc_C_D` directly |
| RD6 (2025-26) | `sc_A` (100%), `sc_B` (65%, <65y only) | **2023-24** | `sc_A` directly — secondary source only; real `cdc_direct` data already covers 2023-24 |

Each file's `Week_Ending_Sat` column is a real date, but on the *labeled*
round's calendar, not the true source season's — re-dating uses the
`epiweeks` package to convert to (MMWR year, week), shift the year by the
season gap while keeping the same MMWR week number, and take that week's
Saturday end-date. This correctly rolls over the Dec/Jan season boundary
and 53-week years (a naive "subtract N calendar years" drifts off Saturday
because of leap days).

**Validated**: the reconstructed 2021-22 Texas curve (aggregated
population-weighted across its 6 age bins) was checked directly against
the real `cdc_judz`/`cdc_sw5n` series at matching dates — child values
track within ~1-5 points throughout the season, and adult values match
closely at every real sparse observation (e.g. 21.6% real vs. 21.52%
reconstructed on 2021-10-30). **2020-21 (RD1) has no independent real
series to check against** (the adult CDC series doesn't start until
2021-22) — treat it as the lowest-confidence segment, sourced solely from
this reconstruction.

**Important distinction:** `Data/time-series.csv` (a *different* file, at the
top-level `Data/` directory) is **COVID-19** hospitalization data, not flu —
confirmed by its Texas series starting exactly at the Aug 2020 COVID wave
tail, zero `inc_death` throughout the pre-COVID 2019-20 flu season, and an
"2021-22 season" spike matching the Omicron wave, not a flu curve. The
correct flu-specific file is `epi_model/flu/time-series.csv`, sourced from
the [MIDAS flu-scenario-modeling-hub target-data repo](https://github.com/midas-network/flu-scenario-modeling-hub/tree/main/target-data).

**Texas hospitalization reporting gap:** `epi_model/flu/time-series.csv` has
a real gap for Texas from Aug 2021 to Jan 2022 (no rows, not zero-value
rows) before resuming Feb 2022 — coincident with, but not fully explained
by, that season's well-documented unusually late H3N2-dominated wave. Only
the 34 weeks actually reported (Feb-Sep 2022) are used as the 2021-22
fitting target; the gap is not backfilled with zeros.

### Subtype-share allocation method (`load_subtype_shares`)

From `ICL_NREVSS_Public_Health_Labs.csv`'s season-total counts:
```
H1c = A(2009 H1N1)                     H3c = A(H3)
U   = A(Subtyping not Performed)        B_alloc = B + BVic + BYam
H1_alloc = H1c + U·H1c/(H1c+H3c)       H3_alloc = H3c + U·H3c/(H1c+H3c)
H1_share = H1_alloc / (H1_alloc+H3_alloc+B_alloc)   [denominator = all positives, not just A]
H3_share, B_share analogously (all three sum to 1)
```
`H3N2v`/`A(H5)` excluded (negligible). These are **season-level constants**
(not weekly), applied multiplicatively to whatever weekly total series
(ILI×positivity, or hospitalization count) is being subtype-split.

Real edge case: Texas's 2020-21 season had `H1c=0, H3c=3, U=3, B_alloc=6` →
`H1_share=0, H3_share=B_share=0.5` — genuinely near-zero flu circulation
that season (pandemic-era NPI suppression), not a data or formula bug.

---

## 3. Current Fitting Exercise: Texas, 2021-22 Season (Hospitalizations)

`fit_texas_hosp_2021.py`. Initial state built from the 2020-21 season;
current season fit against 2021-22 hospitalizations.

### 3.1 Initial state (start of 2021-22 season)

`N_TOTAL = TX_POP` for this run (`tsm.set_population(TX_POP)`), so every
quantity below is a headcount, not a fraction. Empirical, data-driven
analogue of the Algorithm 10 remap (`build_initial_state_from_2020_21`):
```
cumulative_hosp_m = Σ_weeks (2020-21 weekly hosp count × subtype_share_m)      [raw count]
AR_m = cumulative_hosp_m / (epsilon_H · p_H)        [p_H converts hosp→infections;
                                                       epsilon_H converts reported→true count]
uninfected = TX_POP - Σ_m AR_m
S[h=N]  = uninfected·(1-V_2020_21)      S[h=V]  = uninfected·V_2020_21
S[h=m]  = AR_m·(1-V_2020_21)            S[h=mV] = AR_m·V_2020_21
```
then Algorithm 11 reseeding. `V_2020_21 = 0.467` (fixed, user-supplied
end-of-2020-21-season coverage). Infection/vaccination status independence
is assumed (no individual-level joint data available). If `AR_m` is negative
or `ΣAR_m ≥ TX_POP` (i.e. `epsilon_H` implies more people infected in
2020-21 than exist — happens for any `epsilon_H ≲ 0.014` given this
season's real hospitalization counts, comfortably inside the optimizer's
bounded search range), `build_initial_state_from_2020_21` now raises
`RuntimeError`, caught by `residuals()`'s existing large-penalty path — the
same mechanism used for other infeasible trial points (`check_positivity`,
`check_population_conservation`). An earlier version of this code instead
silently rescaled `AR` down to fit under a population cap; that flattened
`AR`'s dependence on `epsilon_H` across a wide, plausible part of the
search range instead of rejecting those points, risking an artificial
attractor for `least_squares`. Refitting after the change reproduced the
identical optimum (same cost, same parameters), confirming the true fit
was always safely inside the feasible region and this was a latent risk,
not an active bug in the reported result.

**Caveat found while implementing this:** at the fitted `epsilon_H≈0.040`,
this formula implies an H3 (and B) attack rate of **~34% combined (~17%
each) of Texas's population** for the 2020-21 season — implausibly high for
a season widely documented as having *extraordinarily low* flu activity (pandemic-era
NPI suppression). This reveals a real limitation: the same `epsilon_H`,
calibrated against the *active* 2021-22 season's hospitalization scale, may
not be an appropriate ascertainment fraction for the very different,
near-flu-free 2020-21 season — reporting completeness plausibly differed
between the two periods. Not corrected here; flagged for follow-up (e.g. a
separate `epsilon` for the initial-state season, or an independent source
for 2020-21's true attack rate).

### 3.2 Current-season vaccination μ(t)

Population-weighted (per-state Census age shares from `Data/pop_risk.csv` —
`data_prep.get_pop_shares(region)`, generalized from the old Texas-only
27.3%/72.7% constant so `build_overall_vax_daily` now takes a `region`
argument and works for any state) blend of two **real, weekly,
region-specific** series for 2021-22 (`build_overall_vax_daily`):
- Children 6mo-17y: `cdc_judz-8etw...csv`
- Adults 18+: `cdc_sw5n-wg2p...csv` (sparser — roughly monthly real values
  even within the weekly-schema file, rest blank; deduplicated, blanks
  dropped, linearly interpolated)

The combined cumulative curve is converted to a daily hazard via
`mu(t) = -ln(1 - ΔV/(1-V(t)))` (`hazard_from_daily_cum`), then
linearly interpolated to continuous time for the ODE solver.

### 3.3 Observation model and fitting

```
Y_hat_m(week) = epsilon_H · [γ_ih·(I_{H,m}^N+I_{H,m}^V) integrated over that week]
```
aligned to the 34 actually-reported week-ending dates via
`aggregate_at_dates` (handles the Aug2021-Jan2022 gap; not a fixed
contiguous 7-day binning from day 0). Compared against
`observed_hosp(week) × subtype_share_m` (season-level share, § above).

Calibrated via `scipy.optimize.least_squares` (bounded, `x_scale` set per
parameter's natural magnitude) minimizing `(Y_hat - target)` across all
strains/weeks.

**Why epsilon_H is a calibrated parameter, not fixed:** tested fixing
`epsilon_H=1` (i.e. assuming full ascertainment) — this produced a ~24x
overshoot of observed counts. `epsilon_H` absorbs real, unavoidable
under-ascertainment in a partial hospital-reporting network (this MIDAS hub
series, like FluSurv-NET/COVID-NET, is not full statewide capture).
`epsilon_H` is directly the ascertainment fraction here (no separate
division needed, since the model is already in headcount units) — the
fitted value means this network captures ~3.7% of Texas's true statewide
flu hospitalizations.

---

## 4. Parameters

### 4.1 Fixed (never calibrated)

| Parameter | Value | Source |
|---|---:|---|
| `β_{H1}, β_{H3}, β_B` (initial guess only — calibrated in fitting runs) | 0.3913, 0.3917, 0.35815 day⁻¹ | Hill et al. posterior medians (initial/demo values; superseded by calibration) |
| `a` | 0.7883 | Hill et al. estimate — residual susceptibility, same-strain reinfection |
| `ξ` | 0.0051 | Hill et al. estimate — fraction of previous-season VE retained |
| `α_{H1}, α_{H3}, α_B` (initial guess only) | 0.5525, 0.3045, 0.5425 | **No longer fixed** — was the Hill-derived default; every season now uses a real, season-specific, population-weighted CDC VE estimate instead (`tsm.set_alpha`, §5.6). These values now only matter as the fixed Hill-defaults candidate in multi-start search. |
| `γ1_{H1}, γ1_{H3}` | 1/1.4 day⁻¹ | Latency-loss rate |
| `γ1_B` | 1/0.6 day⁻¹ | Latency-loss rate |
| `γ2` | 1/3.8 day⁻¹ | `I_R`→`R` progression rate (recovery-bound infectious duration) |
| `γ_ih` | 1/5.0 day⁻¹ | `I_H`→`H` progression rate (time from infectious onset to hospitalization, hospitalization-bound cases) — new, §5.5 |
| `p_H` (`P_H_VEC`) | H1=0.0119, H3=0.0190, B=0.0112 | P(hospitalized\|infected), **now subtype-specific** (was a single shared 0.0111, CDC 2019-20 national burden estimate) — the `E`→`I_H` vs. `E`→`I_R` branch probability (§5.5), not a post hoc split of a shared `I` exit rate. Ordering H3>H1>B is directionally consistent with a Netherlands cohort study's relative-severity finding (H3N2>B>H1N1pdm09), though these exact values are a different, user-supplied source. |
| `p_D` | 0.0641 | P(died\|hospitalized) = 25,000/390,000, CDC 2019-20 national burden estimate — now the `I_H`→`H_D` vs. `I_H`→`H_R` branch probability (§5.5). Not subtype-specific. |
| `γ3` | 1/4 day⁻¹ | `H_R`→`R` progression rate (~4-day median length of stay, CDC/prospective surveillance literature) |
| `γ_hd` | 1/5.2 day⁻¹ | `H_D`→`D` progression rate (time from hospitalization to death, death-bound cases) — new, §5.5 |
| `I0` | 2.5×10⁻⁶ | Per-subtype reseed fraction (Algorithm 11), split into `I_R`/`I_H` by `p_H` |
| `Δt` (reporting) | 1 day | Solver step is adaptive; this is the `t_eval` grid |
| Season length | 365 days | |
| `V_2020_21` | 0.467 | Fixed, user-supplied 2020-21 end-of-season Texas vaccination coverage |
| `N_TOTAL` (`TX_POP`) | 29,527,941 | Census Bureau Vintage 2021 Texas population estimate; set via `tsm.set_population(TX_POP)` so all compartments are headcounts, not fractions (§1.1) |
| Child/adult population share (Texas) | 0.2494 / 0.7506 | Per-state Census age shares from `Data/pop_risk.csv` (`data_prep.get_pop_shares`, `0_17_All`/`pop_total`), weights the vaccination-curve blend — supersedes the earlier 0.273/0.727 ACS-2019 estimate |

`p_D`/`γ3` remain **not subtype-specific** — no clean U.S.
subtype-decomposed fatality data was found. `p_H` became subtype-specific
(above) after this was identified as arguably the least-justified
remaining shared-across-strains assumption in the model, given the
project had already replaced fixed Hill-derived `β`/`α` with real
season-specific data (§5.6) — see §4.2/§5.4/§6.4 for the resulting refits.

### 4.2 Estimated (calibrated by `fit_texas_hosp_2021.py`)

| Parameter | Role | Fitted value (Texas, 2021-22) |
|---|---|---:|
| `β_{H1}` | H1N1 transmission rate | 0.3704 day⁻¹ |
| `β_{H3}` | H3N2 transmission rate | 0.4105 day⁻¹ |
| `β_B` | Influenza B transmission rate | 0.3786 day⁻¹ |
| `epsilon_H` | Hospitalization ascertainment fraction | 0.0249 (≈2.5% of true hospitalizations captured) |

Calibrated via bounded nonlinear least squares (`scipy.optimize.least_squares`)
against the 34 reported weeks of Texas 2021-22 subtype-split hospitalizations.
Refitted several times since the first version of this section:
- After generalizing `data_prep.py`'s population-share logic (per-state
  Census shares instead of the old Texas-only 27.3%/72.7% constant) and
  fixing a blank-estimate handling bug in `load_children_vax_weekly`.
- After switching `μ(t)` to `data_prep.build_overall_vax_daily_from_master`
  (§5.1) instead of `build_overall_vax_daily` — the raw CDC files' first
  real Texas observation for 2021-22 is after the season's Oct-1 anchor,
  so the old loader clamped the early ramp-up to an inflated flat value;
  cost actually *improved* with the corrected curve (1.20e5→1.00e5).
- After the `I_R`/`I_H`/`H_R`/`H_D` compartment restructuring and
  season-specific VE (§5.5-5.6): cost initially got much *worse*
  (1.00e5→7.94e5) purely because `fit()` still only tried one starting
  point — the same multi-start fix already validated in the chain (§5.3)
  recovered most of the gap (7.94e5→2.22e5).
- After making `p_H` subtype-specific (H1=0.0119, H3=0.0190, B=0.0112,
  replacing the shared 0.0111 — §4.1): `epsilon_H` dropped
  (0.0422→0.0249) because H3's IHR rose relative to the old shared value,
  so the model now predicts more true hospitalizations per H3 infection
  and needs a smaller ascertainment fraction to match the same observed
  counts; current values above.

Fit quality: H3 peak timing is close (modeled peak mid-March vs. observed
early-April); peak magnitude is somewhat undershot; a secondary observed
bump (~week 12) and late-season uptick (~week 34) are not reproduced —
plausible limits of a single-peak SEIR-style curve fit to a season with
more complex real-world dynamics. `H1` is fitted slightly above zero
(small nonzero bump) despite observed H1 being ~0 all season — a minor
residual discrepancy.

---

## 5. Multi-Season Chain: Texas, 2021-22 through 2025-26 (`fit_texas_chain.py`)

Extends the single-season 2021-22 fit forward through 2025-26 by chaining
seasons with the real Algorithm 10 remap (`tsm.end_of_season_remap`)
instead of an empirical AR-based bootstrap at every transition. 2021-22
remains the chain's sole data-anchored season (§3.1's AR bootstrap from
real 2020-21 hospitalizations) — no real Texas hospitalization data exists
for 2019-20 to bootstrap 2020-21 itself the same way (`time-series.csv`'s
earliest row, any location, is 2020-08-08), so the chain starts one season
later than 2020-21 rather than reaching back further. Every later
transition (2021-22→2022-23→...→2025-26) uses the model's own simulated
end-of-season E/I/H/R directly.

### 5.1 What's different from the single-season fit

- **Independent per-season refit**: β_H1, β_H3, β_B, and ε_H are all
  refit separately for each season rather than reused, since there's no
  reason to assume a shared hospitalization-ascertainment fraction across
  seasons with potentially different reporting-network completeness.
- **μ(t) from one unified source**: `data_prep.build_overall_vax_daily_from_master`
  reads `vaccination/state_weekly_vaccination_2020_2026.csv` directly for
  every season, rather than re-deriving the child/adult blend per source
  file — this is what also triggered the 2021-22 refit above.
- **Population growth**: `end_of_season_remap` conserves total population
  exactly (it only relabels compartments), so real Texas year-over-year
  population growth is injected manually at each season boundary as new
  naive/unvaccinated (`h=N`) individuals, using deltas from the Census
  Bureau's Texas population series (FRED series TXPOP):

  | Year (July 1) | TX population |
  |---|---:|
  | 2020 | 29,237,895 |
  | 2021 | 29,572,672 |
  | 2022 | 30,118,002 |
  | 2023 | 30,719,247 |
  | 2024 | 31,318,578 |
  | 2025 | 31,709,821 |
  | 2026 | 31,709,821 (placeholder — Vintage 2026 not yet released) |

  Deltas are applied on top of `fit_texas_hosp_2021.TX_POP` (29,527,941,
  the Vintage-2021 estimate already validated in §3.1) rather than
  switching to FRED's absolute levels mid-chain, since FRED's revised 2021
  value differs from that estimate by <0.2% — switching bases would
  introduce a spurious one-time jump at the first boundary instead of a
  smooth year-over-year change. `tsm.set_population` is called at every
  boundary to keep `N_TOTAL` (the force-of-infection denominator) in sync.

### 5.2 A 5th calibrated parameter: t0 (seeding offset)

**Problem found**: every season previously reseeded at a hardcoded t=0
with a fixed `I0` (Algorithm 11) — no actual infectious individuals ever
survive `end_of_season_remap` (it converts end-of-season E/I/H/R into
*susceptibility* for the next season, not leftover infections), so each
season's entire observed rise-to-peak has to come from pure exponential
growth off that fixed seed. With only β free, that couples peak *timing*
to peak *magnitude* (both governed by growth rate) — real seasons whose
onset ran earlier or later than usual can't be fit on both axes at once.
Diagnosed directly: initial fits of 2023-24 and 2025-26 peaked 35 days
early and 70 days late respectively, in *opposite* directions (ruling out
a single systematic bug), while the initial susceptibility pools, subtype
shares, and vaccination hazard curves for those seasons looked normal —
and for 2025-26, multi-start explicitly confirmed that trying *larger* β
made the fit worse (higher cost from overshooting magnitude), not better.

**Fix**: `t0`, the calendar day (relative to the season's Oct-1 anchor) at
which the Algorithm-11 seed is actually injected, is now a 5th calibrated
parameter per season, bounded to `[-30, 30]` days. A negative `t0` gives
the epidemic a head start before the nominal season boundary (no
vaccination pressure applies before t=0, i.e. `μ_eff(t) = 0` for `t < 0`,
then the real `μ(t)` from t=0 onward); a positive `t0` delays it. This is
*not* equivalent to simply rescaling the seed size (`I0`): because `μ(t)`
is a fixed function of calendar time, not of time-since-seeding, a truly
delayed epidemic is exposed to whatever vaccination-driven depletion has
already happened in the real world by the time it gets going — a bigger
seed at t=0 does not capture that interaction, while a genuinely delayed
seeding time does. Implemented as a single continuous `solve_ivp`
integration over `[t0, 365]` (via `tsm.run_season`'s new `t_start`
parameter) rather than two separate integrations.

### 5.3 Optimizer robustness: multi-start least_squares

A single warm-started `least_squares` call routinely got stuck: e.g.
2022-23's fit initially terminated (`xtol`) after 21 function evaluations
having barely moved from the 2021-22-derived starting point, because that
starting point was locally flat relative to the enormous gap to the true
optimum (2022-23 was the unusually early, explosive ~4x-larger-than-2021-22
H3 season). Each season is now fit via multi-start: a cross product of 6
β-scale variants (the previous season's own fit, the Hill et al. defaults,
and ×1.5/×2.0/×0.6/×0.4 scalings of the previous fit) × 4 `t0` offsets
(the warm-started value, 0, +20, -20) = 24 starting points, keeping
whichever converges to the lowest cost. Varying β-scale and `t0`
independently (rather than crossed) was tried first and left a real
regression on the table: none of the *scale-only* variants (all at t0=0)
nor the *t0-only* variants (at the previous season's exact β) happened to
land near 2024-25's own good optimum, so its cost got noticeably worse
(9.97e6 → 1.86e7) purely from search coverage, not because a free `t0`
makes the achievable fit worse — the full cross product recovers it
(8.69e6, better than before).

### 5.4 Season-by-season fitted parameters

Values below are from the **current model** (post-§5.5 compartment
restructuring, post-§5.6 season-specific VE, post-§4.1 subtype-specific
`p_H`). Earlier versions of this table are preserved in git history /
conversation history; the qualitative finding that `t0` fixes
2023-24/2025-26's peak-timing errors holds under every version of the
model.

| Season | β_H1 | β_H3 | β_B | ε_H | t0 (days) | cost |
|---|---:|---:|---:|---:|---:|---:|
| 2021-22 | 0.3704 | 0.4105 | 0.3786 | 0.0249 | +0.0 (fixed) | 3.59e5 |
| 2022-23 | 0.6295 | 0.6286 | 0.4998 | 0.0299 | −0.1 | 1.56e6 |
| 2023-24 | 0.6180 | 0.5599 | 0.5438 | 0.0350 | +20.0 | 2.72e6 |
| 2024-25 | 0.4382 | 0.4008 | 0.0032 | 0.1509 | −18.2 | 1.44e6 |
| 2025-26 | 0.4365 | 0.4446 | 0.3905 | 0.0662 | −17.3 | 1.74e6 |

Fit quality remains good to excellent across all five seasons (visual
shape match is comparable to the pre-subtype-specific-`p_H` version); the
main effect of the `p_H` change was a systematic drop in `ε_H` (see §4.2)
and a widening of β's season-to-season spread (§5.7).

**2024-25's `β_B ≈ 0`**: not a bug — confirmed by inspecting the fit plot.
Real Texas B hospitalizations that season were negligible (peaking
~150-200/week vs. thousands for H1/H3), so `β_B` is very weakly identified
there; near-zero values fit the negligible observed B curve about as well
as any small positive value would. Excluded as an outlier when fitting
Texas's β_B distribution (§5.7).

**Caveats** (carried over, still apply):
- `t0` is a single shared value per season, not per-strain — H1/H3/B could
  plausibly have genuinely different real introduction timing within the
  same season; this hasn't been tested.
- ε_H varies substantially across seasons, which is a real finding, not an
  assumption — see the §3.3 discussion on why a single shared
  ascertainment fraction was already known to be questionable across very
  different seasons (2020-21 vs. 2021-22).
- The chain cannot currently be extended backward past 2021-22 without
  either the ILI-based proxy method (adds an unanchored ascertainment
  parameter, §2's "earlier ILI-based fit") or accepting a fully-naive
  2020-21 start (loses the AR-bootstrap's real-data anchor) — both
  considered and set aside during this exercise in favor of keeping
  2021-22 as the sole anchor.
- 2026's Texas population (used for the last population-growth injection)
  is a placeholder (2025's value held flat) pending Vintage 2026's release.

### 5.5 Compartment restructuring: I_R/I_H and H_R/H_D

**Motivation**: the original model had a single `I` compartment with one
shared exit rate `γ2`, split *post hoc* by `p_H` into `H` vs. `R` (same
for `H`'s shared `γ3` split by `p_D` into `D` vs. `R`). That implicitly
forces hospitalization-bound illness and direct-recovery illness (or
death-bound and recovery-bound hospital stays) to have the *same*
duration distribution — a modeling simplification worth removing once the
data supports it.

**Change**: `I` now splits into `I_R` (recovery-bound) and `I_H`
(hospitalization-bound) *at the moment of leaving E*, using `p_H`; `H`
splits into `H_R` (recovery-bound) and `H_D` (death-bound) *at the moment
of leaving `I_H`*, using `p_D`. Each branch has its own progression rate:
`γ2` (`I_R`→`R`, unchanged value), `γ_ih=1/5.0` day⁻¹ (`I_H`→`H`, new),
`γ3` (`H_R`→`R`, unchanged value), `γ_hd=1/5.2` day⁻¹ (`H_D`→`D`, new).
See §1.4 for the full flow equations. State size grew from 46 to 58
(§1.1). Total infectiousness driving the force of infection is `I_R+I_H`
combined (§1.3) — both branches transmit identically, only their eventual
outcome differs. The Algorithm-11 seed is now split into `I_R`/`I_H` by
`p_H` too (§1.5) — a seeded individual's eventual outcome is just as
undetermined as anyone else's leaving `E`.

**Validated**: plugging the *old* (pre-restructuring) fitted 2021-22
parameters into the *new* model structure produces a sensible
growth-then-peak curve (confirmed by direct simulation), ruling out a
basic implementation bug. The initial post-restructuring fit did get much
worse (cost 1.00e5→7.94e5) with a visibly wrong shape (immediate decay,
no rise) — but this traced to `fit_texas_hosp_2021.py`'s `fit()` still
using only one starting point, not a structural problem: adding the same
multi-start approach already used in the chain (§5.3) recovered most of
the gap (7.94e5→2.22e5, current value, §4.2). All five chained seasons
refit cleanly afterward (§5.4), several fitting *better* than before the
restructuring.

### 5.6 Season-specific vaccine effectiveness (α_m)

**Motivation**: `α_m` (current-season VE) was a single fixed Hill-derived
triple (`{H1: 0.5525, H3: 0.3045, B: 0.5425}`) used identically for every
season. Real CDC VE estimates swing substantially season to season (e.g.
H3N2 ranged from −11% to 47% across 2018-19 through 2025-26) and by
subtype, exactly the kind of parameter this chain has otherwise made
season-specific (β, ε_H, t0) rather than fixed.

**Source**: `vaccination/influenza_subtype_VE_by_season_with_children.xlsx`
— a compiled workbook of CDC outpatient VE estimates (its own "Model-use
notes" sheet documents outpatient-over-inpatient preference and how
interim-vs-final/multi-network conflicts were already resolved per cell).
Outpatient VE is the mechanistically correct choice: `α_m` discounts the
*force of infection* for vaccinated people (protection against becoming
infected at all), which is what outpatient VE measures — hospitalization
VE would conflate that with vaccine-driven severity reduction, which this
model doesn't separately represent (`p_H`/`p_D` are fixed, not
vaccine-modified).

**Method** (`data_prep.build_season_alpha`, `tsm.set_alpha`): the
workbook gives separate Adult and Children VE per subtype/season; since
the model isn't age-stratified (one shared `X∈{N,V}` axis), these are
population-weighted into one `α_m` per season using the same child/adult
shares already used for vaccination coverage (`get_pop_shares`). One
negative point estimate survived this blend by design rather than needing
special handling: children's 2023-24 H3N2 VE is −5% (CI −90% to 43%, not
significant) but adults (75% of the TX population weight) show 30%, so
the population-weighted result (≈21%) comes out positive on its own.

**Current-vs-previous-season simplification**: `α_m` conceptually plays
two roles — current-season leaky protection (§1.3, undamped) and
previous-season residual carryover (§1.2's `c_m`, damped by `ξ=0.0051`).
Rather than tracking two different seasons' `α_m` values, one value is
used for both roles: the carryover term is numerically insensitive to
which season's `α_m` is used (`c_m` only spans 0.9949-1.0 across the
*entire* possible range of `α_m`), so the simplification costs essentially
nothing while avoiding the complexity of threading two seasons' VE through
the same calculation.

**2021-22 real vs. old-default `α_m`**: population-weighted real VE turned
out substantially different from the old fixed defaults — H1 dropped
0.5525→0.165, B dropped 0.5425→0.380, H3 stayed close (0.3045→0.312).
Lower `α_m` means vaccinated people are more susceptible than the old
defaults assumed, which changes the multi-strain competition for the
shared susceptible pool — part of why the 2021-22 refit (§4.2, §5.5)
needed multi-start to find a good optimum under the new dynamics.

---

## 6. National Fitting (`fit_national_hosp_2021.py`, `fit_national_chain.py`)

Runs the identical model and methodology as §3-§5 (same AR bootstrap →
multi-start `least_squares` → `t0` → real-remap chaining) against **US
national** data instead of Texas, alongside the Texas pipeline (both are
kept and maintained, not one replacing the other).

**Motivation**: the goal of estimating season-level β's to forecast
2026-27 doesn't inherently need state-level granularity, and national
aggregation sidesteps several Texas-specific data problems this project
spent real effort working around (the 2021-22 reporting gap, sparse
state-level adult vaccination, the `t0` extremes needed to fit
Texas-specific timing quirks). It also removes a real geography mismatch
that was already present: the CDC VE estimates used for `α_m` (§5.6) are
*national* surveillance-network numbers (US Flu VE Network, NVSN, VISION
are never state-specific) that were being applied to a Texas-specific
model the whole time.

### 6.1 Data availability audit

Before implementing, every Texas-pipeline input was checked for a national
equivalent (full audit preserved in conversation history):

| Input | Finding |
|---|---|
| Hospitalizations (`time-series.csv`, `location="US"`) | **More complete than Texas**: no gap in 2021-22 (vs. Texas's severe Aug 2021-Jan 2022 hole — national's only gap that season is 35 days, Oct 9-Nov 13), full season in 2020-21. 2022-23/2023-24/2024-25 rows are byte-identical to Texas's own (same hub, national reported as its own row, not a sum of states). |
| Subtype shares | Texas's source (`FluViewPhase2Data/ICL_NREVSS_Public_Health_Labs.csv`) has **zero National rows**. National data lives in a **separate file pair** (`ICL_NREVSS_1997_2026_clean_absolute1.csv` / `..._season_summary.csv`, already used by `forecast_2026.py`) that's actually **simpler** to parse: `TOTAL_ALLOCATED` already equals `A(H1)+A(H3)+H3N2v+A(H5)+B` exactly, no "Subtyping not Performed" bucket to reallocate like the state-level method needs. |
| Vaccination — children | National rows exist, same filters as `load_children_vax_weekly`. **100% real weekly data every season 2019-20 through 2025-26** — better than any single state (no suppression-driven gaps at national scale). |
| Vaccination — adults | National rows exist from 2021-22 onward, same filters as `load_adult_vax_weekly`. Same cadence pattern as Texas: sparse/monthly in 2021-22/2022-23, fully weekly from 2023-24. |
| Population shares | `pop_risk.csv`'s national row (`"United States "`) gives child_share=0.2174, adult_share=0.7826 — but `get_pop_shares("United States")` raised `KeyError` (a real bug, `load_state_pop_shares` hardcoded to skip this row; fixed, §6.2). |
| μ(t) master file | `state_weekly_vaccination_2020_2026.csv` already had "United States" rows for 2020-21/2021-22/2022-23 (the scenario-hub-reconstructed portion computed its own weights independently of the buggy `get_pop_shares`) but **zero** for 2023-24 onward (`build_state_weekly_vax.py` never treated the nation as one of its 51 "states"); fixed, §6.2. |
| VE (`α_m`) | Already national — no change needed. |

**National season-level subtype shares** (computed via the new national
loader, §6.2):

| Season | H1 | H3 | B |
|---|---:|---:|---:|
| 2020-21 | 0.173 | 0.513 | 0.314 |
| 2021-22 | 0.005 | **0.988** | 0.007 |
| 2022-23 | 0.288 | 0.663 | 0.049 |
| 2023-24 | 0.500 | 0.275 | 0.226 |
| 2024-25 | 0.500 | 0.441 | 0.059 |
| 2025-26 (partial) | 0.112 | 0.725 | 0.163 |

Nationally, 2021-22 was **98.8% H3-dominant** — even more extreme than
Texas's own mix that season, illustrating real state-vs-national
divergence in subtype composition.

One naming trap worth documenting explicitly: the raw vaccination files
(`cdc_judz-8etw`, `cdc_sw5n-wg2p`) use `geographic_name="National"`, while
`pop_risk.csv`, the master vax file, and the VE workbook all use
`"United States"` — two different strings for the same entity, easy to
silently mismatch (it caused one bug during implementation, §6.3).

### 6.2 Code changes

- **`data_prep.load_state_pop_shares`**: removed the hardcoded
  `if state == "United States": continue` — `get_pop_shares("United
  States")` now returns `(0.2174, 0.7826)` instead of raising `KeyError`.
- **`data_prep.load_national_subtype_shares`** (new): reads
  `ICL_NREVSS_1997_2026_clean_absolute1.csv` directly — no unsubtyped
  reallocation step needed (see §6.1). `build_weekly_hosp_target` gained a
  `shares_fn` parameter (default unchanged, so Texas is unaffected) to
  select it.
- **`vaccination/build_state_weekly_vax.py`**: same `"United States"`
  exclusion removed from its own local copy of `load_state_pop_shares`,
  plus a `VAX_REGION_OVERRIDE = {"United States": "National"}` mapping to
  bridge the naming mismatch above when calling the raw vaccination
  loaders. Re-run to regenerate `state_weekly_vaccination_2023_2026.csv`
  (52 states now, was 51) and the merged
  `state_weekly_vaccination_2020_2026.csv` (now has national rows for all
  6 seasons, not just 2020-21/2021-22/2022-23).

### 6.3 National single-season fit: 2021-22 (`fit_national_hosp_2021.py`)

Structurally identical to `fit_texas_hosp_2021.py` (same AR bootstrap,
same multi-start `least_squares`), with every input swapped for its
national equivalent: `NATIONAL_POP = 332,454,000` (FRED series `POPTHM`,
July 2021 — the national analogue of `TX_POP`'s Census Vintage 2021
estimate), national hospitalization target (`hosp_location="US"`),
national subtype shares (`load_national_subtype_shares`), national μ(t)
(`region="United States"`), and national VE
(`build_season_alpha("United States")`).

**`V_2020_21` (previous-season vaccination coverage for the AR bootstrap)**:
population-weighted blend of a *real* national children's end-of-season
2020-21 coverage (58.38%, from `load_children_vax_weekly(2020,
region="National")` — unlike Texas, which had no real 2020-21 series
either way) and a fixed CDC national adult estimate (**52.1%**,
user-supplied) — the adult weekly series doesn't start until 2021-22
nationally either, same gap as Texas, just no weekly fallback for this one
season:
```
V_2020_21 = 0.2174 × 58.38% + 0.7826 × 52.1% = 53.47%
```
(One implementation bug hit and fixed here: the children/adult loaders
need `region="National"`, not `"United States"` — see §6.1's naming trap.)

**Result**: β_H1=0.3593, β_H3=0.4102, β_B=0.3215, ε_H=0.0328, cost=6.46e7
(not comparable to Texas's cost in raw units — national counts are ~10x
larger, so squared residuals are mechanically bigger; only the fit *shape*
is comparable). Implied national 2020-21 attack rate at this ε_H: 16.6%
combined (H1 2.9%, H3 8.5%, B 5.2%) — still high for a season with
well-documented NPI-suppressed circulation, but noticeably less extreme
than Texas's own version of this same check (~34% combined H3+B, §3.1).

**Fit quality**: nationally, 2021-22 was **bimodal** — a January H3 spike,
a dip, then a larger April-May peak — a genuinely different shape from
Texas's own smoother single-peak 2021-22 curve. The single-peak SEIR model
matches the second, dominant peak's timing/magnitude reasonably but
completely misses the January spike, the same kind of limitation already
documented for other multi-wave seasons (§4.2, §5.4).

### 6.4 National multi-season chain (`fit_national_chain.py`)

Structurally identical to `fit_texas_chain.py` — same 5-parameter fit
(β_H1, β_H3, β_B, ε_H, `t0`), same beta-scale × `t0`-offset multi-start
cross product, same population-growth-injection mechanism at each season
boundary. Population growth uses real US national year-over-year deltas
(FRED series `POPTHM`):

| Year (July 1) | US population |
|---|---:|
| 2020 | 331,862,000 |
| 2021 | 332,454,000 |
| 2022 | 334,370,000 |
| 2023 | 337,147,000 |
| 2024 | 340,335,000 |
| 2025 | 342,076,000 |
| 2026 | 342,909,000 |

(All 7 years already published as of this writing — no placeholder
needed, unlike Texas's 2026 TXPOP value.)

**Season-by-season fitted parameters** (current model, post-subtype-specific
`p_H`, §4.1):

| Season | β_H1 | β_H3 | β_B | ε_H | t0 (days) | cost |
|---|---:|---:|---:|---:|---:|---:|
| 2021-22 | 0.3728 | 0.4049 | 0.3918 | 0.0205 | +0.0 (fixed) | 5.75e7 |
| 2022-23 | 0.4964 | 0.4990 | 0.4325 | 0.0601 | −19.8 | 8.03e7 |
| 2023-24 | 0.4324 | 0.4132 | 0.4217 | 0.0976 | −19.1 | 9.73e7 |
| 2024-25 | 0.4728 | 0.4223 | 0.3901 | 0.1864 | +0.0 | 3.60e8 |
| 2025-26 | 0.4386 | 0.4346 | 0.3925 | 0.0916 | −20.6 | 2.48e8 |

**Fit quality**: 2022-23, 2024-25, and 2025-26 match observed peak timing
and magnitude closely (visually near-perfect for 2022-23 and 2024-25);
2023-24 lags the real peak by roughly 4-5 weeks — notably, this is the
*same season* that gave the Texas chain the most trouble too (§5.4),
suggesting something genuinely unusual about how the 2023-24 season's
epidemic curve shape interacts with this model's fixed-seed/`t0`
mechanism, not a Texas- or national-specific artifact. As with Texas,
`ε_H` dropped across the board after `p_H` became subtype-specific (same
mechanism: H3's IHR rose relative to the old shared value, so less
ascertainment is needed to match the same observed counts) — unlike
Texas, national β_B never became unidentifiable (national B counts stayed
non-negligible every season, §6.1's subtype-share table).

### 6.5 Comparison with Texas

- **β values are broadly consistent** between the two independent fits
  (different data, same model) — both cluster in the 0.32-0.53 range, a
  reassuring cross-check rather than a contradiction.
- **ε_H is systematically higher nationally** at the same season (e.g.
  2024-25: 26.2% national vs. 12.7% Texas) — plausible, since aggregating
  the whole national reporting network likely captures a larger share of
  true hospitalizations than any single state's local piece of it.
- **Costs are not comparable in raw units** (national counts are ~10x
  Texas's, so squared residuals are mechanically larger); only fit *shape*
  comparisons are meaningful.
- **2023-24's timing lag appears in both fits** (§6.4) — evidence this is
  a real property of that season's epidemic curve interacting with the
  model, not a geography-specific data issue.
- **2021-22 is bimodal nationally but single-peaked in Texas** — a real
  finding about state-vs-national heterogeneity, not a fitting artifact.

### 6.6 Final fitted β distributions (Texas vs. National)

Normal distributions fit to each subtype's β across the 5 chained seasons
(sample mean/std), using the current model (post-subtype-specific `p_H`,
§4.1/§5.4/§6.4):

| Subtype | Texas | National |
|---|---|---|
| β_H1 (n=5) | Normal(0.4985, 0.1176) | Normal(0.4426, 0.0469) |
| β_H3 (n=5) | Normal(0.4889, 0.1005) | Normal(0.4348, 0.0375) |
| β_B (Texas n=4, National n=5) | Normal(0.4532, 0.0814) | Normal(0.4057, 0.0199) |

**Texas β_B excludes 2024-25** (β_B≈0.0032, §5.4's identifiability note) —
including it gives Normal(0.3632, 0.2132), which puts real probability
mass on negative β and isn't usable; the n=4 version above is the
defensible one.

**Texas's spread widened substantially** after `p_H` became
subtype-specific — e.g. β_H1's CV went from 10.7% (shared `p_H`) to 23.6%
(subtype-specific `p_H`), driven by 2022-23/2023-24 both landing
unusually high (β≈0.55-0.63) relative to the other three seasons. National
stayed comparatively tight (CVs 5-11%, barely changed). This reinforces
§6.5's point: Texas's smaller, noisier hospitalization series produces
markedly less stable per-season β estimates than the national aggregate —
not evidence that Texas's underlying transmission dynamics genuinely vary
more season to season. **For anything downstream that needs one stable
distribution** (e.g. a 2026-27 Monte Carlo forecast prior), the national
distributions are the safer default; use Texas's only where
state-specific output is actually required.

As before, these β-triples are correlated within a season (an unusually
transmissive season tends to show up in all three strains together), so
independent per-strain sampling from these Normals would lose that
correlation structure if used for Monte Carlo scenario generation.
