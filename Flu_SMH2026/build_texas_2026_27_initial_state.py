"""
Step 2 of the Texas 2026-27 forecast: remap the persisted 2025-26
end-of-season state (texas_2025_26_final_state.npz, from
save_texas_2025_26_final_state.py) into 2026-27's unseeded initial state
via Algorithm 10 (tsm.end_of_season_remap), then inject Texas's
2025->2026 population-growth delta.

Population growth: per plan, TX_POP for the 2026-27 boundary holds flat
at the existing 2025 value (fit_texas_chain.TX_POP_BY_YEAR already has
2026 held flat at 2025's 31,709,821 as a placeholder, from when it was
built for the *2025-26* season's own start -- reusing it here for the
2026-27 transition means add_population_growth(S, 2025, 2026) correctly
injects zero net growth, consistent with "assume same as Texas" for this
input).

The resulting initial state is NOT yet seeded (Algorithm 11's infectious
seed is injected later, at the calibrated/sampled t0 for each ensemble
draw -- see fit_texas_chain.fit_one_season/run_season_with_params for why
seeding happens per-parameter-draw, not once here).
"""
import numpy as np

import three_strain_model as tsm
import fit_texas_chain as chain

IN_PATH = "texas_2025_26_final_state.npz"
OUT_PATH = "texas_2026_27_initial_state_unseeded.npz"

FROM_YEAR = 2025   # 2025-26 season
TO_YEAR = 2026     # 2026-27 season


def main():
    data = np.load(IN_PATH)
    final_state_2025_26 = (data["S"], data["E"], data["I_R"], data["I_H"],
                            data["H_R"], data["H_D"], data["R"], data["D"])
    tsm.set_population(float(data["N_TOTAL"]))
    print(f"Loaded 2025-26 final state, N_TOTAL={tsm.N_TOTAL:,.0f}")

    remapped = tsm.end_of_season_remap(*final_state_2025_26, reseed=False)
    S0, E0, I_R0, I_H0, H_R0, H_D0, R0, D0 = remapped

    S0 = chain.add_population_growth(S0, FROM_YEAR, TO_YEAR)
    print(f"Population growth {FROM_YEAR}->{TO_YEAR}: "
          f"{chain.TX_POP_BY_YEAR[FROM_YEAR]:,} -> {chain.TX_POP_BY_YEAR[TO_YEAR]:,} "
          f"(delta={chain.TX_POP_BY_YEAR[TO_YEAR] - chain.TX_POP_BY_YEAR[FROM_YEAR]:,}), "
          f"N_TOTAL now={tsm.N_TOTAL:,.0f}")

    idx = {h: i for i, h in enumerate(tsm.H_GROUPS)}
    print("\n2026-27 unseeded initial state -- susceptible pool by history group (h):")
    for h in tsm.H_GROUPS:
        print(f"  h={h:<4} unvacc={S0[0, idx[h]]:,.0f}  vacc={S0[1, idx[h]]:,.0f}")
    print(f"  Total S = {S0.sum():,.0f}  (E={E0.sum():,.0f}, R={R0.sum():,.0f}, "
          f"D excluded, all I/H compartments should be 0 pre-seeding: "
          f"I_R={I_R0.sum()}, I_H={I_H0.sum()}, H_R={H_R0.sum()}, H_D={H_D0.sum()})")

    np.savez(OUT_PATH, S=S0, E=E0, I_R=I_R0, I_H=I_H0, H_R=H_R0, H_D=H_D0, R=R0, D=D0,
             N_TOTAL=tsm.N_TOTAL)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
