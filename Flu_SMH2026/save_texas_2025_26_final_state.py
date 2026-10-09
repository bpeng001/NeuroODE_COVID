"""
Rebuild and persist Texas's 2025-26 end-of-season compartment state, using
the already-fitted parameters from the current fit_texas_chain.py run (no
re-optimization) -- confirmed the underlying 2025-26 hospitalization and
vaccination data haven't changed since that fit was last run (same 45
reported weeks through 2026-08-08 for hospitalizations, same real
vaccination coverage through the dates already used), so replaying the
known-good parameters reproduces the identical fit without repeating the
expensive multi-start search.

This is the checkpoint the 2026-27 forecast chains forward from --
Algorithm 10's between-season remap only needs the immediately-prior
season's end-of-season compartments (not the full 2021-22-2025-26
history: previous-season history only tracks one season back by design,
see MODEL.md SS1.1/1.5), so there's no need to keep re-simulating the
whole chain from scratch every time a new season's forecast is wanted.
"""
import numpy as np

import three_strain_model as tsm
import fit_texas_hosp_2021 as fit2021
import fit_texas_chain as chain

OUT_PATH = "texas_2025_26_final_state.npz"

# Already-fitted parameters from fit_texas_chain.py's most recent full run
# (post subtype-specific p_H -- see MODEL.md SS5.4). fit2021 uses only the
# first 4 (no t0 -- 2021-22 keeps t0 fixed at 0, it's the chain's sole
# data-anchored season).
KNOWN_PARAMS = {
    2021: [0.3704, 0.4105, 0.3786, 0.0249],
    2022: [0.6295, 0.6286, 0.4998, 0.0299, -0.1],
    2023: [0.6180, 0.5599, 0.5438, 0.0350, 20.0],
    2024: [0.4382, 0.4008, 0.0032, 0.1509, -18.2],
    2025: [0.4365, 0.4446, 0.3905, 0.0662, -17.3],
}


def main():
    Y_hat, final_state = fit2021.run_full(KNOWN_PARAMS[2021])
    print(f"2021-22: replayed, Y_hat total={Y_hat.sum():.1f}, N_TOTAL={tsm.N_TOTAL:,.0f}")

    prev_year = fit2021.FIT_SEASON_START_YEAR
    for season_start_year in chain.SEASON_START_YEARS:
        remapped = tsm.end_of_season_remap(*final_state, reseed=False)
        S0, E0, I_R0, I_H0, H_R0, H_D0, R0, D0 = remapped
        S0 = chain.add_population_growth(S0, prev_year, season_start_year)
        initial_state_unseeded = (S0, E0, I_R0, I_H0, H_R0, H_D0, R0, D0)

        params = KNOWN_PARAMS[season_start_year]
        Y_hat, final_state, dates, target = chain.run_season_with_params(
            season_start_year, initial_state_unseeded, params)
        print(f"{season_start_year}: replayed, Y_hat total={Y_hat.sum():.1f}, "
              f"target total={target.sum():.1f}, N_TOTAL={tsm.N_TOTAL:,.0f}")
        prev_year = season_start_year

    S, E, I_R, I_H, H_R, H_D, R, D = final_state
    total_pop = (S.sum() + E.sum() + I_R.sum() + I_H.sum()
                 + H_R.sum() + H_D.sum() + R.sum() + D.sum())
    np.savez(OUT_PATH, S=S, E=E, I_R=I_R, I_H=I_H, H_R=H_R, H_D=H_D, R=R, D=D,
             N_TOTAL=tsm.N_TOTAL)
    print(f"\nSaved {OUT_PATH}")
    print(f"2025-26 end-of-season total (living+dead) = {total_pop:,.0f} "
          f"(N_TOTAL={tsm.N_TOTAL:,.0f})")


if __name__ == "__main__":
    main()
