# Modeling Steps

## 1. Prepare the data

1. Load hospitalization data from [`time-series.csv`](/Users/boyapeng/Desktop/Dissertation/Aim2/epi_model/time-series.csv).
2. Filter Texas all-age hospitalization data using:
   - `location == "48"`
   - `age_group == "0-130"`
3. Define the fitting window:
   - start: `2024-11-09`
   - end: `2025-04-26`
4. Define the longer observed-plus-projection window:
   - start: `2024-11-09`
   - end: `2025-12-13`
5. Load vaccination data from [`COVID_RD19_Vaccination_curves.csv`](/Users/boyapeng/Desktop/Dissertation/Aim2/Data/COVID_RD19_Vaccination_curves.csv).
6. Aggregate vaccination counts to Texas overall weekly new vaccination counts.
7. Load lineage marker data from [`variant_all_marker.csv`](/Users/boyapeng/Desktop/Dissertation/Aim2/epi_model/variant_all_marker.csv).

## 2. Define transmission epochs

The model now uses explicit transmission epochs instead of one generic rolling dominant-sequence rule.

1. Fitting epoch 1:
   - `2024-11-09` to `2024-12-20`
   - `A = KP.3.1.1`
   - `B = XEC`
   - `beta_multiplier_B = 1.0`
   - `immune_escape_history_B_on_A = 0.20`
   - `immune_escape_vaccine_mismatch = 0.20`
2. Fitting epoch 2:
   - `2024-12-21` to `2025-04-26`
   - `A = XEC`
   - `B = LP.8.1`
   - `beta_multiplier_B = 1.0`
   - `immune_escape_history_B_on_A = 0.20`
   - `immune_escape_vaccine_mismatch = 0.20`
3. Forecast epoch:
   - `2025-04-27` onward
   - `A = LP.8.1`
   - `B = XFG`
   - `beta_multiplier_B = 1.25`
   - `immune_escape_history_B_on_A = 0.45`
   - `immune_escape_vaccine_mismatch = 0.45`

Within each epoch, the generic B share is built as:

```text
B_share / (A_share + B_share)
```

This keeps the fit and forecast aligned with the marker file:
- `KP.3.1.1 -> XEC` first
- then `XEC -> LP.8.1`
- then `LP.8.1 -> XFG` after the fitting period

The forecast epoch gives `XFG` both an additional transmission advantage and a larger immune-escape assumption than the fitting epochs. This is needed because immune escape at the same moderate level used for `XEC` and `LP.8.1` may not create a fast enough post-fit rebound after the winter `LP.8.1` peak if the epidemic is already declining.

## 3. Define vaccine-target epochs

Vaccination specificity is handled separately from the transmission A/B pair.

1. `2024-10-01` to `2025-09-30`
   - vaccine target lineage = `KP.3.1.1`
2. `2025-10-01` onward
   - vaccine target lineage = `XFG`

Interpretation:
- during fitting, vaccination is treated as `KP.3.1.1`-specific
- all non-`KP.3.1.1` variants are assumed to have the same vaccine escape penalty
- after October 2025, the new vaccination is treated as `XFG`-specific

## 4. Build the variant share path

1. Use `build_variant_share_series(...)` from [`variant_model.py`](/Users/boyapeng/Desktop/Dissertation/Aim2/epi_model/variant_model.py).
2. Pass the explicit `transmission_epochs` schedule from the notebook configuration.
3. Build a daily B-share path over the simulation horizon.
4. Store epoch switch dates so the simulator can relabel the generic A/B compartments at epoch boundaries.

## 5. Build the vaccine mismatch logic

1. Keep the vaccinated compartment `V` as the vaccinated population.
2. For each simulation day, resolve:
   - current transmission epoch
   - current vaccine-target epoch
3. Apply matched vaccine protection when the infecting lineage equals the vaccine target.
4. Apply escaped vaccine protection when the infecting lineage differs from the vaccine target.
5. Allow the active transmission epoch to override both:
   - history escape from `A` immunity to `B`
   - vaccine mismatch escape for `V`

During fitting:
- `KP.3.1.1` infections use matched vaccine protection
- `XEC` and `LP.8.1` infections use the same vaccine-escape penalty relative to the `KP.3.1.1`-specific vaccine

During late forecast after `2025-10-01`:
- `XFG` infections use matched vaccine protection
- other variants use the same vaccine-escape penalty relative to the `XFG`-specific vaccine

Before `2025-10-01`, `XFG` is still mismatched to the `2024-2025` KP.3.1.1-focused vaccine, so the model gives it both:
- vaccine mismatch escape
- larger `LP.8.1 -> XFG` immune escape than in the fitting epochs
- forecast-epoch transmission advantage through `beta_multiplier_B = 1.25`

## 6. Fit the model

1. Fit only on hospitalization observations from `2024-11-09` through `2025-04-26`.
2. Use `curve_fit(...)` on `sim_fit(...)`.
3. Estimate the transmission parameter `beta`.
4. Save the main fitted value as:

```python
beta_fit_main = float(popt[0])
```

5. Generate the deterministic fitted hospitalization curve with:

```python
weekly_hosp_fit = sim_det(0, beta_fit_main)
```

## 7. Plot the fit

1. Plot the fitted line only over the fitting period.
2. Use the actual observed weekly dates from `GC_`.
3. Overlay observed hospitalization points for the same period.
4. Save the fitting figure as `fit1.png`.

## 8. Run the stochastic forecast

1. Keep the main fitted parameter `beta_fit_main`.
2. Use the longer simulation horizon starting from `2024-11-09`.
3. Continue variant replacement after the fit using the explicit forecast epoch:
   - `A = LP.8.1`
   - `B = XFG`
   - `beta_multiplier_B = 1.25`
4. Run stochastic simulations with:
   - fitted `beta_fit_main`
   - stochastic volatility parameter `std`
   - mean reversion parameter `kappa`
   - seeds from `seed.csv` if available
5. Compute percentile bands across simulations.

## 9. Plot the forecast

1. Keep the fitted line visible through `2025-04-26`.
2. Start the forecast median and 95% CI only after `2025-04-26`.
3. Use observed dates from `GC_1` for the observed portion.
4. Extend weekly forecast dates beyond the last observed date for future predictions.
5. Save the stochastic forecast figure as `sto1.png`.

## 10. Files involved

- [`explore.ipynb`](/Users/boyapeng/Desktop/Dissertation/Aim2/epi_model/explore.ipynb)
- [`variant_model.py`](/Users/boyapeng/Desktop/Dissertation/Aim2/epi_model/variant_model.py)
- [`variant_all_marker.csv`](/Users/boyapeng/Desktop/Dissertation/Aim2/epi_model/variant_all_marker.csv)
- [`time-series.csv`](/Users/boyapeng/Desktop/Dissertation/Aim2/epi_model/time-series.csv)
- [`COVID_RD19_Vaccination_curves.csv`](/Users/boyapeng/Desktop/Dissertation/Aim2/Data/COVID_RD19_Vaccination_curves.csv)
