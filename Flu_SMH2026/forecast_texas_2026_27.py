"""
Step 3 of the Texas 2026-27 forecast: Monte Carlo ensemble of weekly
per-subtype hospitalization trajectories, chained from the persisted,
already-remapped+growth-adjusted 2026-27 initial state
(texas_2026_27_initial_state_unseeded.npz, from
build_texas_2026_27_initial_state.py).

Sampled per draw (national fitting, MODEL.md SS6.6 for beta, this
conversation for epsilon_H/t0):
    beta_H1 ~ Normal(0.4426, 0.0469)
    beta_H3 ~ Normal(0.4348, 0.0375)
    beta_B  ~ Normal(0.4057, 0.0199)
    epsilon_H ~ Normal(0.0912, 0.0614)          [n=5, all national seasons]
    t0 ~ Normal(-14.875, 9.9356) days           [n=4, excl. 2021-22's fixed t0=0]
All clipped to the same physical bounds used during fitting
(beta/epsilon_H > 0, |t0| <= 30) before use.

Fixed (not sampled) assumptions, per "assume same as Texas":
    - alpha_m (VE): Texas's own 2025-26 population-weighted value
      (data_prep.build_season_alpha(2025, region="Texas")), held flat.
    - mu(t) (vaccination hazard): Texas's own real 2025-26 cumulative
      coverage curve shape, reused day-for-day as the 2026-27 projection.
    - N_TOTAL: already fixed via the persisted initial state (Texas 2025
      population, no further 2026-27 growth injected, per the plan).

No real 2026-27 target data exists yet (the season starts Oct 1, 2026) --
this is a genuine forecast, not a fit. See MODEL.md / conversation history
for the fuller reconciliation-with-flu_2026-share-simulations plan (step 4,
not yet implemented here).
"""
import datetime as dt

import numpy as np

import data_prep as dp
import three_strain_model as tsm

IN_PATH = "texas_2026_27_initial_state_unseeded.npz"
OUT_PATH = "texas_2026_27_forecast_ensemble.npz"

SEASON_START_YEAR = 2026
REGION = "Texas"
ANCHOR = dt.date(SEASON_START_YEAR, 10, 1)
N_DRAWS = 300   # matches flu_2026_*_weekly_share_simulations.csv's ensemble size

# National fitting (MODEL.md SS6.6 for beta; conversation for t0).
# epsilon_H is now a FIXED point estimate (2025-26's fitted national value),
# not sampled: investigation found epsilon_H alone explained ~90% of the
# ensemble's output variance (Spearman rho=0.95 with season-total
# hospitalizations, vs <=0.24 for every beta) because (a) its historical
# CV is huge (56-67%, vs 5-10% for beta) and (b) it enters Y_hat as a
# direct linear multiplier, unlike beta which only affects outbreak size
# through nonlinear epidemic dynamics. Checked and ruled out: this isn't
# one outlier season (dropping 2024-25's 0.1864 only cuts the CV from 67%
# to 52%) and isn't a beta/epsilon_H fitting degeneracy that would justify
# joint sampling (their correlation across the 5 seasons is weak and
# inconsistently signed: +0.53 with beta_H1, -0.13 with beta_H3, -0.25
# with beta_B -- not reliable evidence with n=5 either way). The real
# issue is conceptual: epsilon_H reflects how complete *this specific*
# hospital-reporting network is, not an epidemiological quantity that
# should randomly differ every season the way transmission rates do --
# sampling it from 5 historical seasons' spread bakes in reporting-network
# history that may not apply next season at all. A fixed point estimate
# (the most recent season) is the more defensible choice; beta/t0 stay
# sampled since those represent genuine season-to-season epidemiological
# uncertainty.
BETA_DIST = {"H1": (0.4426, 0.0469), "H3": (0.4348, 0.0375), "B": (0.4057, 0.0199)}
EPSILON_H_FIXED = 0.0662   # Texas's OWN 2025-26 fitted value (most recent season) --
# epsilon_H reflects how complete *this state's* hospital-reporting network is, which
# is state-specific (confirmed by the CA/VT pilot: fitted epsilon_H varies by
# geography even under shared national beta), so it must come from Texas's own chain
# fit, not national's.
T0_DIST = (-14.875, 9.9356)
CLIP_BOUNDS = ([1e-3, 1e-3, 1e-3, 1e-4, -30.0], [3.0, 3.0, 3.0, 1.0, 30.0])


def sample_params(rng, max_tries=1000):
    """Rejection-resample (not clip) any draw outside CLIP_BOUNDS.
    Clipping would pile draws up exactly at the floor/ceiling -- for a
    sampled parameter with mass near zero, clipping to a tiny floor
    collapses Y_hat to near-zero, contaminating the ensemble's low end
    with a spurious "near-zero season" pile-up that isn't a real
    epidemiological scenario. Resampling instead preserves the
    distribution's actual shape within the valid range. (epsilon_H is now
    fixed, not sampled -- see the module-level comment above -- so this
    mainly guards beta/t0's tails now.)"""
    for _ in range(max_tries):
        beta_h1 = rng.normal(*BETA_DIST["H1"])
        beta_h3 = rng.normal(*BETA_DIST["H3"])
        beta_b = rng.normal(*BETA_DIST["B"])
        epsilon_H = EPSILON_H_FIXED
        t0 = rng.normal(*T0_DIST)
        params = np.array([beta_h1, beta_h3, beta_b, epsilon_H, t0])
        if np.all(params >= CLIP_BOUNDS[0]) and np.all(params <= CLIP_BOUNDS[1]):
            return params
    raise RuntimeError("sample_params: exceeded max_tries without a valid draw")


def build_weekly_dates(anchor, n_weeks=52):
    return [anchor + dt.timedelta(days=7 * k) for k in range(1, n_weeks + 1)]


def main():
    data = np.load(IN_PATH)
    initial_state_unseeded = (data["S"], data["E"], data["I_R"], data["I_H"],
                               data["H_R"], data["H_D"], data["R"], data["D"])
    tsm.set_population(float(data["N_TOTAL"]))
    print(f"Loaded 2026-27 unseeded initial state, N_TOTAL={tsm.N_TOTAL:,.0f}")

    # Fixed assumptions: TX's own 2025-26 VE and vaccination-curve shape,
    # held flat as the 2026-27 projection ("assume same as Texas").
    alpha_2026_27 = dp.build_season_alpha(2025, region=REGION)
    tsm.set_alpha(alpha_2026_27["H1"], alpha_2026_27["H3"], alpha_2026_27["B"])
    print(f"alpha_m (held from TX 2025-26): {alpha_2026_27}")

    day_grid = np.arange(0, tsm.SEASON_DAYS + 1)
    overall_cum_2025_26 = dp.build_overall_vax_daily_from_master(2025, day_grid, REGION)
    mu_daily = dp.hazard_from_daily_cum(overall_cum_2025_26)

    def mu_eff(t):
        return 0.0 if t < 0 else np.interp(t, day_grid, mu_daily)

    dates = build_weekly_dates(ANCHOR)

    rng = np.random.default_rng(seed=20262027)
    n_weeks = len(dates)
    n_strains = len(tsm.STRAINS)
    ensemble_Y = np.zeros((N_DRAWS, n_weeks, n_strains))
    ensemble_params = np.zeros((N_DRAWS, 5))
    n_failed = 0

    for i in range(N_DRAWS):
        params = sample_params(rng)
        beta_h1, beta_h3, beta_b, epsilon_H, t0 = params
        tsm.set_betas(beta_h1, beta_h3, beta_b)
        try:
            seeded_initial = tsm.seed_infections(*initial_state_unseeded)
            (t_grid, S_traj, E_traj, I_R_traj, I_H_traj,
             H_R_traj, H_D_traj, R_traj, D_traj) = tsm.run_season(
                initial=seeded_initial, mu_func=mu_eff, days=tsm.SEASON_DAYS, t_start=t0)
            tsm.check_positivity(S_traj, E_traj, I_R_traj, I_H_traj, H_R_traj, H_D_traj, R_traj, D_traj)
            daily_hosp = tsm.compute_daily_hosp_incidence(I_H_traj)
            weekly_hosp_headcount = dp.aggregate_at_dates(daily_hosp, t_grid, dates, ANCHOR)
            ensemble_Y[i] = epsilon_H * weekly_hosp_headcount
        except (AssertionError, RuntimeError) as e:
            n_failed += 1
            ensemble_Y[i] = np.nan
        ensemble_params[i] = params

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{N_DRAWS} draws done ({n_failed} failed so far)")

    print(f"\nDone: {N_DRAWS - n_failed}/{N_DRAWS} draws succeeded.")

    total_per_draw = np.nansum(ensemble_Y, axis=(1, 2))
    print(f"Season-total hospitalizations across draws: "
          f"median={np.nanmedian(total_per_draw):,.0f}, "
          f"[2.5%, 97.5%]=[{np.nanpercentile(total_per_draw, 2.5):,.0f}, "
          f"{np.nanpercentile(total_per_draw, 97.5):,.0f}]")

    peak_week_total = np.nanmax(ensemble_Y.sum(axis=2), axis=1)
    print(f"Peak week total hospitalizations across draws: "
          f"median={np.nanmedian(peak_week_total):,.0f}, "
          f"[2.5%, 97.5%]=[{np.nanpercentile(peak_week_total, 2.5):,.0f}, "
          f"{np.nanpercentile(peak_week_total, 97.5):,.0f}]")

    date_strs = np.array([d.isoformat() for d in dates])
    np.savez(OUT_PATH, ensemble_Y=ensemble_Y, ensemble_params=ensemble_params,
             dates=date_strs, strains=np.array(tsm.STRAINS), n_failed=n_failed)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
