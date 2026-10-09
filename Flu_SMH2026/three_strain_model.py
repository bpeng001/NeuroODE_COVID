"""
Three-strain Hill-style influenza SEIR model (H1N1 / H3N2 / B).

Reproduces hill_three_strain_influenza_algorithm.md, Algorithms 1-11,
run across one or more seasons with all parameters fixed at the values
in the Parameter Table, extended with Hospitalization (H) and Death (D)
compartments downstream of I:

    E --gamma1*(1-p_H_m)--> I_R --gamma2-->   R
    E --gamma1*p_H_m-->     I_H --gamma_ih--> [splits by p_D]
    I_H --gamma_ih*(1-p_D)--> H_R --gamma3--> R
    I_H --gamma_ih*p_D-->     H_D --gamma_hd--> D

I is split into I_R (recovery-bound) and I_H (hospitalization-bound) at
the moment of leaving E, and H is split into H_R (recovery-bound) and
H_D (death-bound) at the moment of leaving I_H -- each branch has its own
progression rate, rather than a single shared exit rate (gamma2 for I,
gamma3 for H) split post hoc by p_H/p_D. That earlier shared-rate
structure implicitly forced the time spent in I (or H) to have the same
distribution regardless of eventual outcome; branching at entry instead
lets hospitalization-bound illness and death-bound hospital stays run on
their own, independently calibrated, durations.

p_H (P_H_VEC) is subtype-specific -- H1=0.0119, H3=0.0190, B=0.0112 --
unlike the model's other severity parameters. p_D, gamma3, gamma_ih,
gamma_hd are fixed (not calibrated) and shared across strains, sourced
from CDC national burden estimates / hospital length-of-stay literature --
see the conversation-derived revision plan for exact values and sources.
D is a permanent sink: it is excluded when Algorithm 10 carries the living
population into the next season (so cross-season population can shrink
by cumulative flu deaths), but is included in the within-season
population-conservation check (S+E+I_R+I_H+H_R+H_D+R+D = N_TOTAL each
season). Total currently-infectious I_R+I_H (not just one branch) drives
the force of infection -- both branches are equally infectious, only
their eventual outcome differs.

Simplifying choices confirmed with user:
  - Season length = 365 days.
  - Season 1 starts naive and unvaccinated: previous-season history
    h = N for the whole population, current-season status X = N.
  - Current-season vaccination mu_y(t) defaults to 0 (all-naive demo
    runs) but is a pluggable function of t, passed into `run_season`/
    `run_multi_season`, so a real data-derived mu_y(t) (e.g. from
    data_prep.py) can drive the ODE for a specific fitting season.
  - Between seasons, Algorithm 10 remaps end-of-season E/I_R/I_H/H_R/H_D/R
    + leftover S into next season's previous-season-history groups h,
    current-season vaccination status resets to X=N, and Algorithm 11
    reseeds a small infectious fraction per subtype at the start of
    every season (including season 1), itself split into I_R/I_H by p_H.

Integration: scipy.integrate.solve_ivp (adaptive RK45), not fixed-step
Euler. gamma1_B = 1/0.6 ~= 1.67 day^-1 is fast enough that a 1-day
explicit-Euler step can drive I_B negative; an adaptive solver avoids
that. The full S/E/I_R/I_H/H_R/H_D/R/D trajectory is checked for
positivity after integration.

Output: weekly new-infection incidence Z_{m,w} (Algorithm 8), not
prevalence.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Fixed parameters (Parameter Table)
# ---------------------------------------------------------------------------
STRAINS = ["H1", "H3", "B"]

beta = {"H1": 0.3913, "H3": 0.3917, "B": 0.35815}          # day^-1
gamma1 = {"H1": 1 / 1.4, "H3": 1 / 1.4, "B": 1 / 0.6}       # day^-1
gamma2 = 1 / 3.8                                            # day^-1, I_R -> R only

a = 0.7883          # relative susceptibility to the same strain following infection in the previous season
xi = 0.0051         # fraction of previous-season VE retained

alpha = {"H1": 0.5525, "H3": 0.3045, "B": 0.5425}  # current-season VE

# Hospitalization/death parameters (fixed -- see revision-plan sources:
# CDC national burden estimates for p_D, hospital length-of-stay
# literature for gamma3/gamma_ih/gamma_hd). p_D/gamma3 are not
# subtype-specific: no clean U.S. subtype-decomposed fatality data was
# found. p_H IS subtype-specific (unlike the original Hill-derived shared
# value) -- ordering H3 > H1 > B matches the relative-severity pattern a
# Netherlands cohort study found (H3N2 > B > H1N1pdm09), though these
# exact values are a different source.
p_H = {"H1": 0.0119, "H3": 0.0190, "B": 0.0112}   # P(hospitalized | infected), by subtype
p_D = 0.0641        # P(died | hospitalized): 25,000/390,000, CDC 2019-20 national burden estimate
gamma3 = 1 / 4.0     # day^-1, H_R -> R (hospital removal rate, ~4-day median length of stay)
gamma_ih = 1 / 5.0   # day^-1, I_H -> H (time from infectious onset to hospitalization, hospitalization-bound cases)
gamma_hd = 1 / 5.2   # day^-1, H_D -> D (time from hospitalization to death, death-bound cases)

I0 = 2.5e-6         # initial/reseeded infectious fraction per subtype (Algorithm 11)
N_TOTAL = 1.0        # normalized population
DT = 1.0             # day (output/reporting resolution; solver step is adaptive)
SEASON_DAYS = 365
N_SEASONS = 3        # number of seasons to simulate back-to-back

# Previous-season exposure-history groups, in fixed order.
H_GROUPS = ["N", "H1", "H3", "B", "V", "H1V", "H3V", "BV"]
N_H, N_M = len(H_GROUPS), len(STRAINS)


def zero_vaccination(t):
    """Default mu_y(t): no current-season vaccination. Pass a data-derived
    callable (e.g. built from data_prep.hazard_from_daily_cum) to run_season/
    run_multi_season to override."""
    return 0.0


def build_f_matrix():
    """Susceptibility matrix f(h, m), Algorithm 2. Rows = H_GROUPS, cols = STRAINS."""
    # c_{m,y}: residual susceptibility from previous-season vaccination.
    # Uses previous-season VE alpha_{m,y-1}; since the non-N/V history
    # groups start at zero population in this single-season, all-naive
    # run, these values never actually get multiplied by a nonzero state.
    c = {m: 1 - xi * alpha[m] for m in STRAINS}

    f = np.ones((N_H, N_M))
    idx = {h: i for i, h in enumerate(H_GROUPS)}
    col = {m: j for j, m in enumerate(STRAINS)}

    for m in STRAINS:
        f[idx[m], col[m]] = a          # same-strain reinfection row (H1/H3/B)

    for h in ("V", "H1V", "H3V", "BV"):
        for m in STRAINS:
            f[idx[h], col[m]] = c[m]

    for h, m in (("H1V", "H1"), ("H3V", "H3"), ("BV", "B")):
        f[idx[h], col[m]] = min(a, c[m])

    return f


F_MATRIX = build_f_matrix()
ALPHA_VEC = np.array([alpha[m] for m in STRAINS])
GAMMA1_VEC = np.array([gamma1[m] for m in STRAINS])
P_H_VEC = np.array([p_H[m] for m in STRAINS])

# ---------------------------------------------------------------------------
# State packing: flat vector layout for solve_ivp.
#   [0:16)  S,   shape (2, 8)   -- X in {N,V}, h in H_GROUPS
#   [16:22) E,   shape (2, 3)   -- X in {N,V}, m in STRAINS
#   [22:28) I_R, shape (2, 3)   -- recovery-bound infectious
#   [28:34) I_H, shape (2, 3)   -- hospitalization-bound infectious
#   [34:40) H_R, shape (2, 3)   -- recovery-bound hospitalized
#   [40:46) H_D, shape (2, 3)   -- death-bound hospitalized
#   [46:52) R,   shape (2, 3)
#   [52:58) D,   shape (2, 3)
# ---------------------------------------------------------------------------
_S_SIZE = 2 * N_H
_M_SIZE = 2 * N_M
_S_END = _S_SIZE
_E_END = _S_END + _M_SIZE
_IR_END = _E_END + _M_SIZE
_IH_END = _IR_END + _M_SIZE
_HR_END = _IH_END + _M_SIZE
_HD_END = _HR_END + _M_SIZE
_R_END = _HD_END + _M_SIZE
_D_END = _R_END + _M_SIZE  # == total length


def pack(S, E, I_R, I_H, H_R, H_D, R, D):
    return np.concatenate([S.ravel(), E.ravel(), I_R.ravel(), I_H.ravel(),
                            H_R.ravel(), H_D.ravel(), R.ravel(), D.ravel()])


def unpack(y):
    S = y[0:_S_END].reshape(2, N_H)
    E = y[_S_END:_E_END].reshape(2, N_M)
    I_R = y[_E_END:_IR_END].reshape(2, N_M)
    I_H = y[_IR_END:_IH_END].reshape(2, N_M)
    H_R = y[_IH_END:_HR_END].reshape(2, N_M)
    H_D = y[_HR_END:_HD_END].reshape(2, N_M)
    R = y[_HD_END:_R_END].reshape(2, N_M)
    D = y[_R_END:_D_END].reshape(2, N_M)
    return S, E, I_R, I_H, H_R, H_D, R, D


def initial_state():
    """Season-1 start: all-naive (h=N, X=N) with a small infectious seed per
    subtype (Algorithm 11)."""
    S = np.zeros((2, N_H))
    E = np.zeros((2, N_M))
    I_R = np.zeros((2, N_M))
    I_H = np.zeros((2, N_M))
    H_R = np.zeros((2, N_M))
    H_D = np.zeros((2, N_M))
    R = np.zeros((2, N_M))
    D = np.zeros((2, N_M))

    h_N = H_GROUPS.index("N")
    S[0, h_N] = N_TOTAL
    return seed_infections(S, E, I_R, I_H, H_R, H_D, R, D)


def seed_infections(S, E, I_R, I_H, H_R, H_D, R, D):
    """Algorithm 11: introduce a small infectious seed per subtype at the start
    of a season, taken out of the naive-susceptible (h=N, X=N) pool so total
    population is conserved. I0 is a fraction of N_TOTAL (Hill et al.'s
    I_{0,m}=2.5e-6), so the actual seed is I0*N_TOTAL -- this matters once
    N_TOTAL is a real headcount (e.g. via set_population) rather than 1.
    The seed is split into I_R/I_H by p_H, same as any cohort leaving E,
    since these seeded individuals are already infectious and their
    eventual outcome (recover directly vs. hospitalize) is just as
    undetermined as anyone else's at the moment of becoming infectious."""
    S = S.copy()
    I_R = I_R.copy()
    I_H = I_H.copy()
    h_N = H_GROUPS.index("N")
    seed_each = I0 * N_TOTAL
    S[0, h_N] -= seed_each * N_M
    I_R[0, :] += seed_each * (1 - P_H_VEC)
    I_H[0, :] += seed_each * P_H_VEC
    return S, E, I_R, I_H, H_R, H_D, R, D


def end_of_season_remap(S, E, I_R, I_H, H_R, H_D, R, D, reseed=True):
    """Algorithm 10: build next season's previous-season-history distribution
    from this season's end-of-season state, then reseed (Algorithm 11).

    Everyone who stayed susceptible and unvaccinated -> h=N; susceptible and
    vaccinated -> h=V; infected with strain m (whether currently in E,
    I_R/I_H, H_R/H_D, or recovered to R) while unvaccinated -> h=m; same
    while vaccinated -> h=mV. Current-season vaccination status resets to
    X=N for the new season (Algorithm 10: "reset current-season
    vaccination status and retain only the previous-season exposure
    history"). D (deaths) is a permanent sink and is deliberately excluded
    here: cumulative flu deaths do not re-enter the living population
    carried into next season.

    `reseed=False` skips the Algorithm 11 reseed and returns the relabeled
    state with zero E/I_R/I_H/H_R/H_D instead -- for callers that need to
    seed infections themselves at a time other than t=0 (e.g. a calibrated
    per-season seeding offset t0, so the state returned here is what's
    correct for a solve_ivp initial condition at t=t0, not necessarily t=0).
    """
    idx = {h: i for i, h in enumerate(H_GROUPS)}
    new_S = np.zeros((2, N_H))
    new_E = np.zeros((2, N_M))
    new_I_R = np.zeros((2, N_M))
    new_I_H = np.zeros((2, N_M))
    new_H_R = np.zeros((2, N_M))
    new_H_D = np.zeros((2, N_M))
    new_R = np.zeros((2, N_M))
    new_D = np.zeros((2, N_M))

    new_S[0, idx["N"]] = S[0, :].sum()
    new_S[0, idx["V"]] = S[1, :].sum()
    for j, m in enumerate(STRAINS):
        new_S[0, idx[m]] = (E[0, j] + I_R[0, j] + I_H[0, j]
                             + H_R[0, j] + H_D[0, j] + R[0, j])
        new_S[0, idx[m + "V"]] = (E[1, j] + I_R[1, j] + I_H[1, j]
                                   + H_R[1, j] + H_D[1, j] + R[1, j])

    if not reseed:
        return new_S, new_E, new_I_R, new_I_H, new_H_R, new_H_D, new_R, new_D
    return seed_infections(new_S, new_E, new_I_R, new_I_H, new_H_R, new_H_D, new_R, new_D)


def compute_flows(S, I_total, t):
    """Algorithm 4-5: force of infection and infection flow F_{m,h}^{N/V}(t).

    lambda_m(t) = beta_m * (I_m^N+I_m^V) / N_TOTAL -- Algorithm 4's
    counts-based form ("if counts rather than proportions are used, divide
    the infectious term by N"). With N_TOTAL=1 (proportions) this reduces
    exactly to the proportions-based form used before `set_population` existed.

    `I_total` is TOTAL currently-infectious per strain/status (I_R + I_H
    combined) -- both branches transmit identically, only their eventual
    outcome (recover directly vs. hospitalize) differs.

    Returns (lam, F_N, F_V) where lam has shape (n_m,) and F_N/F_V have
    shape (n_h, n_m).
    """
    lam = beta_vec_cache * (I_total[0, :] + I_total[1, :]) / N_TOTAL   # (n_m,)

    F_N = F_MATRIX * S[0, :, None] * lam[None, :]                          # (h, m)
    F_V = F_MATRIX * S[1, :, None] * (1 - ALPHA_VEC)[None, :] * lam[None, :]
    return lam, F_N, F_V


beta_vec_cache = np.array([beta[m] for m in STRAINS])


def set_betas(beta_h1, beta_h3, beta_b):
    """Update the module-level beta dict/cache in place (for calibration
    scripts that sweep beta values across repeated `run_season` calls)."""
    global beta_vec_cache
    beta["H1"], beta["H3"], beta["B"] = beta_h1, beta_h3, beta_b
    beta_vec_cache = np.array([beta[m] for m in STRAINS])


def set_alpha(alpha_h1, alpha_h3, alpha_b):
    """Update the module-level alpha (current-season VE) dict and rebuild
    F_MATRIX in place -- for scripts that use a real, season-specific VE
    (e.g. data_prep.build_season_alpha) instead of the fixed Hill-derived
    defaults. F_MATRIX must be rebuilt here, not just ALPHA_VEC: it bakes
    in alpha via the c_m = 1-xi*alpha_m carryover term (build_f_matrix),
    so a stale F_MATRIX would silently keep using the old alpha there even
    after ALPHA_VEC itself was updated."""
    global ALPHA_VEC, F_MATRIX
    alpha["H1"], alpha["H3"], alpha["B"] = alpha_h1, alpha_h3, alpha_b
    ALPHA_VEC = np.array([alpha[m] for m in STRAINS])
    F_MATRIX = build_f_matrix()


def set_population(n_total):
    """Switch N_TOTAL from the default normalized 1.0 to a real headcount
    (e.g. a state's population), for scripts that want the model's
    compartments in raw counts rather than population fractions. Must be
    called before building any initial state or running the model --
    `compute_flows`'s N_TOTAL-normalized force of infection and
    `seed_infections`'s I0*N_TOTAL seed both depend on it."""
    global N_TOTAL
    N_TOTAL = n_total


def make_rhs(mu_func):
    """Build the ODE right-hand side for solve_ivp (Algorithms 3-7, plus
    E->I_R/I_H->H_R/H_D->R/D), closing over a time-varying vaccination-rate
    function `mu_func(t)`."""

    def rhs(t, y):
        mu = mu_func(t)
        S, E, I_R, I_H, H_R, H_D, R, D = unpack(y)

        I_total = I_R + I_H
        _, F_N, F_V = compute_flows(S, I_total, t)

        dS = np.zeros_like(S)
        dS[0, :] = -F_N.sum(axis=1) - mu * S[0, :]
        dS[1, :] = -F_V.sum(axis=1) + mu * S[0, :]

        F_m_N = F_N.sum(axis=0)   # (n_m,)
        F_m_V = F_V.sum(axis=0)

        dE = np.zeros_like(E)
        dI_R = np.zeros_like(I_R)
        dI_H = np.zeros_like(I_H)
        dH_R = np.zeros_like(H_R)
        dH_D = np.zeros_like(H_D)
        dR = np.zeros_like(R)
        dD = np.zeros_like(D)

        dE[0, :] = F_m_N - GAMMA1_VEC * E[0, :] - mu * E[0, :]
        dE[1, :] = F_m_V - GAMMA1_VEC * E[1, :] + mu * E[0, :]

        # E splits into I_R (recovery-bound) and I_H (hospitalization-bound)
        # by p_H at the moment of leaving E, each with its own exit rate --
        # not a single shared-rate I compartment split post hoc.
        dI_R[0, :] = GAMMA1_VEC * (1 - P_H_VEC) * E[0, :] - gamma2 * I_R[0, :] - mu * I_R[0, :]
        dI_R[1, :] = GAMMA1_VEC * (1 - P_H_VEC) * E[1, :] - gamma2 * I_R[1, :] + mu * I_R[0, :]

        dI_H[0, :] = GAMMA1_VEC * P_H_VEC * E[0, :] - gamma_ih * I_H[0, :] - mu * I_H[0, :]
        dI_H[1, :] = GAMMA1_VEC * P_H_VEC * E[1, :] - gamma_ih * I_H[1, :] + mu * I_H[0, :]

        # I_H splits into H_R (recovery-bound) and H_D (death-bound) by
        # p_D at the moment of hospitalization, each with its own exit rate.
        dH_R[0, :] = gamma_ih * (1 - p_D) * I_H[0, :] - gamma3 * H_R[0, :] - mu * H_R[0, :]
        dH_R[1, :] = gamma_ih * (1 - p_D) * I_H[1, :] - gamma3 * H_R[1, :] + mu * H_R[0, :]

        dH_D[0, :] = gamma_ih * p_D * I_H[0, :] - gamma_hd * H_D[0, :] - mu * H_D[0, :]
        dH_D[1, :] = gamma_ih * p_D * I_H[1, :] - gamma_hd * H_D[1, :] + mu * H_D[0, :]

        dR[0, :] = gamma2 * I_R[0, :] + gamma3 * H_R[0, :] - mu * R[0, :]
        dR[1, :] = gamma2 * I_R[1, :] + gamma3 * H_R[1, :] + mu * R[0, :]

        dD[0, :] = gamma_hd * H_D[0, :]
        dD[1, :] = gamma_hd * H_D[1, :]

        return pack(dS, dE, dI_R, dI_H, dH_R, dH_D, dR, dD)

    return rhs


def run_season(initial=None, mu_func=zero_vaccination, days=SEASON_DAYS, dt_report=DT, t_start=0.0):
    """Integrate one season starting from `initial` = (S, E, I_R, I_H, H_R,
    H_D, R, D) at time `t_start` (default 0, the season's nominal Oct-1
    anchor); defaults to the season-1 all-naive start (`initial_state()`).
    `mu_func(t)` is the current-season vaccination rate; defaults to no
    vaccination.

    `t_start` != 0 supports a calibrated per-season seeding offset t0 (see
    fit_texas_chain.py): pass a negative t_start to give the epidemic a
    head start before the nominal anchor, or positive to delay it -- the
    caller's `mu_func` is responsible for returning 0 for t < 0 in that
    case (no vaccination pressure before the season's nominal start)."""
    if initial is None:
        initial = initial_state()
    # Built via explicit step count (not np.arange(t_start, days+dt_report,
    # dt_report)) so the last point can never float-point-drift past `days`
    # when t_start is a non-round float -- happens when a calibrated t0 is
    # perturbed during least_squares' finite-difference Jacobian estimate,
    # and solve_ivp rejects any t_eval value outside t_span.
    n_steps = int(np.floor((days - t_start) / dt_report)) + 1
    t_eval = t_start + dt_report * np.arange(n_steps)
    y0 = pack(*initial)
    rhs = make_rhs(mu_func)

    sol = solve_ivp(
        rhs, [t_start, days], y0,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-8, atol=1e-10,
        max_step=1.0,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    n_t = len(sol.t)
    S_traj = np.zeros((n_t, 2, N_H))
    E_traj = np.zeros((n_t, 2, N_M))
    I_R_traj = np.zeros((n_t, 2, N_M))
    I_H_traj = np.zeros((n_t, 2, N_M))
    H_R_traj = np.zeros((n_t, 2, N_M))
    H_D_traj = np.zeros((n_t, 2, N_M))
    R_traj = np.zeros((n_t, 2, N_M))
    D_traj = np.zeros((n_t, 2, N_M))
    for k in range(n_t):
        (S_traj[k], E_traj[k], I_R_traj[k], I_H_traj[k],
         H_R_traj[k], H_D_traj[k], R_traj[k], D_traj[k]) = unpack(sol.y[:, k])

    return sol.t, S_traj, E_traj, I_R_traj, I_H_traj, H_R_traj, H_D_traj, R_traj, D_traj


def run_multi_season(n_seasons=N_SEASONS, days=SEASON_DAYS, dt_report=DT, mu_func=zero_vaccination):
    """Run `n_seasons` back-to-back, applying Algorithm 10's end-of-season
    remap (with Algorithm 11 reseeding) between seasons. `mu_func` may be a
    single callable (used every season) or a list of `n_seasons` callables.
    Returns a list of per-season (t_grid, S,E,I_R,I_H,H_R,H_D,R,D _traj) tuples."""
    mu_funcs = mu_func if isinstance(mu_func, (list, tuple)) else [mu_func] * n_seasons

    seasons = []
    initial = None  # season 1 uses initial_state() default inside run_season
    for y in range(n_seasons):
        (t_grid, S_traj, E_traj, I_R_traj, I_H_traj,
         H_R_traj, H_D_traj, R_traj, D_traj) = run_season(
            initial=initial, mu_func=mu_funcs[y], days=days, dt_report=dt_report)
        check_positivity(S_traj, E_traj, I_R_traj, I_H_traj, H_R_traj, H_D_traj, R_traj, D_traj)
        check_population_conservation(S_traj, E_traj, I_R_traj, I_H_traj, H_R_traj, H_D_traj, R_traj, D_traj)
        seasons.append((t_grid, S_traj, E_traj, I_R_traj, I_H_traj, H_R_traj, H_D_traj, R_traj, D_traj))

        initial = end_of_season_remap(
            S_traj[-1], E_traj[-1], I_R_traj[-1], I_H_traj[-1],
            H_R_traj[-1], H_D_traj[-1], R_traj[-1], D_traj[-1])

    return seasons


def check_positivity(S_traj, E_traj, I_R_traj, I_H_traj, H_R_traj, H_D_traj, R_traj, D_traj, tol=None):
    """tol defaults to 1e-8*N_TOTAL (relative to population scale) so this
    still works sensibly when N_TOTAL is a real headcount, not just 1."""
    if tol is None:
        tol = 1e-8 * N_TOTAL
    assert np.all(S_traj >= -tol), f"S went negative: min={S_traj.min()}"
    assert np.all(E_traj >= -tol), f"E went negative: min={E_traj.min()}"
    assert np.all(I_R_traj >= -tol), f"I_R went negative: min={I_R_traj.min()}"
    assert np.all(I_H_traj >= -tol), f"I_H went negative: min={I_H_traj.min()}"
    assert np.all(H_R_traj >= -tol), f"H_R went negative: min={H_R_traj.min()}"
    assert np.all(H_D_traj >= -tol), f"H_D went negative: min={H_D_traj.min()}"
    assert np.all(R_traj >= -tol), f"R went negative: min={R_traj.min()}"
    assert np.all(D_traj >= -tol), f"D went negative: min={D_traj.min()}"


def check_population_conservation(S_traj, E_traj, I_R_traj, I_H_traj, H_R_traj, H_D_traj, R_traj, D_traj, tol=None):
    """Checks total living+dead population is conserved *within* this season's
    trajectory (start-of-season total == end-of-season total). Not checked
    against the fixed N_TOTAL constant, since population legitimately shrinks
    season-over-season as end_of_season_remap excludes cumulative D. tol
    defaults to 1e-6*N_TOTAL (relative), so this still works sensibly when
    N_TOTAL is a real headcount rather than 1."""
    if tol is None:
        tol = 1e-6 * N_TOTAL
    def total_at(k):
        return (S_traj[k].sum() + E_traj[k].sum() + I_R_traj[k].sum() + I_H_traj[k].sum()
                + H_R_traj[k].sum() + H_D_traj[k].sum() + R_traj[k].sum() + D_traj[k].sum())
    start_total, end_total = total_at(0), total_at(-1)
    assert abs(end_total - start_total) < tol, (
        f"Population not conserved within season: start={start_total}, end={end_total}")


def compute_daily_hosp_incidence(I_H_traj):
    """New-hospitalization inflow gamma_ih*I_H(t) per day (fraction of
    population per day), summed over vaccination status X. Shape (n_t, n_m).
    I_H is already the hospitalization-bound branch (split off from E by
    p_H when it was formed), so unlike the old shared-rate structure this
    is NOT multiplied by p_H again here -- that split already happened
    upstream, at E->I_H."""
    return gamma_ih * (I_H_traj[:, 0, :] + I_H_traj[:, 1, :])


def compute_daily_incidence(t_grid, S_traj, I_R_traj, I_H_traj):
    """Algorithm 8: C_m(t) = F_m^N(t) + F_m^V(t), the new-infection flow into E."""
    n_t = len(t_grid)
    C = np.zeros((n_t, N_M))
    for k in range(n_t):
        I_total = I_R_traj[k] + I_H_traj[k]
        _, F_N, F_V = compute_flows(S_traj[k], I_total, t_grid[k])
        C[k, :] = F_N.sum(axis=0) + F_V.sum(axis=0)
    return C


def aggregate_weekly(t_grid, C, dt_report=DT):
    """Algorithm 8: Z_{m,w} = sum_{t in w} C_m(t) * dt, using one C(t) sample per
    reporting day (left endpoint of each day)."""
    n_days = len(t_grid) - 1   # t_grid has one extra endpoint (day 0 .. day 365)
    daily = C[:n_days, :] * dt_report   # (n_days, n_m)

    n_full_weeks = n_days // 7
    full = daily[: n_full_weeks * 7, :].reshape(n_full_weeks, 7, N_M).sum(axis=1)

    remainder = daily[n_full_weeks * 7 :, :]
    if remainder.shape[0] > 0:
        weekly = np.vstack([full, remainder.sum(axis=0, keepdims=True)])
    else:
        weekly = full

    week_grid = np.arange(1, weekly.shape[0] + 1)
    return week_grid, weekly


def incidence_across_seasons(seasons, dt_report=DT):
    """Compute weekly incidence per season, then concatenate onto one
    continuous week axis. Returns (week_grid, Z_weekly, season_boundaries),
    where season_boundaries lists the first week index of season 2, 3, ..."""
    week_chunks, Z_chunks = [], []
    week_offset = 0
    season_boundaries = []
    for season_idx, (t_grid, S_traj, E_traj, I_R_traj, I_H_traj,
                      H_R_traj, H_D_traj, R_traj, D_traj) in enumerate(seasons, start=1):
        C_daily = compute_daily_incidence(t_grid, S_traj, I_R_traj, I_H_traj)
        week_grid_local, Z_weekly_local = aggregate_weekly(t_grid, C_daily, dt_report=dt_report)

        week_chunks.append(week_grid_local + week_offset)
        Z_chunks.append(Z_weekly_local)
        week_offset += week_grid_local[-1]
        if season_idx < len(seasons):
            season_boundaries.append(week_offset + 1)

    return np.concatenate(week_chunks), np.vstack(Z_chunks), season_boundaries


def plot_incidence(week_grid, Z_weekly, season_boundaries=None, out_path="three_strain_incidence.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"H1": "#1f77b4", "H3": "#d62728", "B": "#2ca02c"}
    for j, m in enumerate(STRAINS):
        ax.plot(week_grid, Z_weekly[:, j], label=f"Influenza {m}",
                 color=colors[m], linewidth=1.8, marker="o", markersize=3)

    for b in (season_boundaries or []):
        ax.axvline(b - 0.5, color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("Week (continuous across seasons)")
    ax.set_ylabel("New infections per week (fraction of population)")
    n_seasons = len(season_boundaries or []) + 1
    title_suffix = "weekly incidence" if n_seasons == 1 else f"weekly incidence, {n_seasons} seasons"
    ax.set_title(f"Three-Strain Hill-Style Influenza Model\n{title_suffix}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    seasons = run_multi_season()
    week_grid, Z_weekly, season_boundaries = incidence_across_seasons(seasons)
    plot_incidence(week_grid, Z_weekly, season_boundaries)
