"""
Variant model v3: HIGH/LOW risk stratification on top of variant_model2.

Differences vs variant_model2.py:
  - Every compartment is doubled into _H and _L versions (42 state variables).
  - epsilon1 and IHR are per risk group (passed as model_params["epsilon1_H"],
    ["epsilon1_L"], ["IHR_H"], ["IHR_L"]). epsilon2 and HDR are shared.
  - N is partitioned as N_H + N_L; the force of infection uses the combined
    I_total over the full N (homogeneous mixing assumption -- both groups
    contact the same lambda).
  - Vaccinations are split between groups by frac_H / frac_L proportions
    (override via model_params["vacc_frac_H"] if available).
  - Outputs weekly_hosp_H, weekly_hosp_L, and weekly_hosp_total.

All disease-progression rates (gamma1, gamma2, delta, eta), waning rates,
HDR, and variant-related parameters (escape, beta_multiplier_b, humidity)
are unchanged from v2.

Reuses build_variant_share_series from variant_model2 verbatim.
"""

import numpy as np
import pandas as pd

from variant_model2 import (
    build_variant_share_series,
    humidity_factor,
    _resolve_schedule_entry,
)


def _weekly_sum(series, weekly_index):
    weekly_index = list(weekly_index)
    return np.array([np.sum(series[7 * i : 7 * (i + 1)]) for i in weekly_index], dtype=float)


def simulate_variant_model_HL(
    beta,
    horizon_days,
    weekly_index,
    vacc_series,
    variant_share_series,
    model_params,
    variant_cfg,
    stochastic_cfg=None,
    return_states=False,
):
    """Risk-stratified (HIGH / LOW) two-strain SEIR with humidity-driven beta.

    Required ``model_params`` (group-specific):
        N_H, N_L                  -- population in each risk group
        epsilon1_H, epsilon1_L    -- VE infection-block per group
        IHR_H, IHR_L              -- infection -> hospitalization per group
        vacc_frac_H               -- fraction of weekly vaccinations going to H
                                     (default = N_H / (N_H + N_L))

        For each compartment Xxx, provide Xxx0_H and Xxx0_L (initial counts).
        B-side initial values (suffix _B0) default to 0 when omitted.

    Shared ``model_params`` (same across groups):
        epsilon2, HDR, gamma1, gamma2, delta, eta, rep, re_pro, reinf, wan
    """
    # ---------- Bookkeeping ----------
    horizon_days = int(horizon_days)
    vacc_series = np.asarray(vacc_series, dtype=float)
    vacc_is_grouped = vacc_series.ndim == 2  # shape (n_weeks, 2): col0=H, col1=L
    p_b = np.asarray(pd.Series(variant_share_series).iloc[:horizon_days], dtype=float)
    if p_b.shape[0] != horizon_days:
        raise ValueError("variant_share_series length must match horizon_days")

    # ---------- Unpack shared progression parameters ----------
    epsilon2 = model_params["epsilon2"]
    HDR      = model_params["HDR"]
    gamma1   = model_params["gamma1"]
    gamma2   = model_params["gamma2"]
    delta    = model_params["delta"]
    eta      = model_params["eta"]
    rep      = model_params["rep"]
    re_pro   = model_params["re_pro"]
    rho_a    = model_params["reinf"]
    omega_a  = model_params["wan"]

    waning_multiplier_b = variant_cfg["waning_multiplier_B"]
    rho_b   = waning_multiplier_b * rho_a
    omega_b = waning_multiplier_b * omega_a

    # ---------- Group-specific scalars ----------
    N_H = float(model_params["N_H"])
    N_L = float(model_params["N_L"])
    N   = N_H + N_L
    e1_H = float(model_params["epsilon1_H"])
    e1_L = float(model_params["epsilon1_L"])
    IHR_H = float(model_params["IHR_H"])
    IHR_L = float(model_params["IHR_L"])
    vacc_frac_H = float(model_params.get("vacc_frac_H", N_H / N))
    vacc_frac_L = 1.0 - vacc_frac_H  # only used when vacc_series is 1D

    # ---------- Per-variant escape and transmission multiplier ----------
    a_escape_default = float(variant_cfg.get("a_escape", 0.0))
    b_escape_default = float(variant_cfg.get("b_escape", 0.0))
    beta_multiplier_b_default = variant_cfg["beta_multiplier_B"]
    transmission_epochs = variant_cfg.get("transmission_epochs", [])

    # ---------- Humidity-driven seasonality ----------
    ah_array             = variant_cfg.get("ah_array", None)
    ah_alpha             = float(variant_cfg.get("ah_alpha", 0.15))
    ah_baseline_mean_exp = float(variant_cfg.get("ah_baseline_mean_exp", 1.0))

    # ---------- Stochastic beta setup ----------
    beta_bar = float(np.squeeze(beta))
    beta_t = beta_bar
    if stochastic_cfg:
        seed = stochastic_cfg.get("seed")
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        std = stochastic_cfg["std"]
        kappa = stochastic_cfg["kappa"]
    else:
        rng = None
        std = 0.0
        kappa = 0.0

    # ---------- Initial compartment values (per group) ----------
    def _ic(name, group, default=0.0):
        return float(model_params.get(f"{name}_{group}", default))

    V_H, V_L = _ic("V0", "H"), _ic("V0", "L")
    R_A_H, R_A_L = _ic("R0", "H"), _ic("R0", "L")
    R_B_H, R_B_L = _ic("R_B0", "H"), _ic("R_B0", "L")
    S_A_prot_H, S_A_prot_L = _ic("SxS0", "H"), _ic("SxS0", "L")
    S_A_part_H, S_A_part_L = _ic("SxI0", "H"), _ic("SxI0", "L")
    S_B_prot_H, S_B_prot_L = _ic("S_B_prot0", "H"), _ic("S_B_prot0", "L")
    S_B_part_H, S_B_part_L = _ic("S_B_part0", "H"), _ic("S_B_part0", "L")

    Inh_A_H, Inh_A_L = _ic("Inh0", "H"), _ic("Inh0", "L")
    Inr_A_H, Inr_A_L = _ic("Inr0", "H"), _ic("Inr0", "L")
    Imh_A_H, Imh_A_L = _ic("Imh0", "H"), _ic("Imh0", "L")
    Imr_A_H, Imr_A_L = _ic("Imr0", "H"), _ic("Imr0", "L")

    Inh_B_H, Inh_B_L = _ic("Inh_B0", "H"), _ic("Inh_B0", "L")
    Inr_B_H, Inr_B_L = _ic("Inr_B0", "H"), _ic("Inr_B0", "L")
    Imh_B_H, Imh_B_L = _ic("Imh_B0", "H"), _ic("Imh_B0", "L")
    Imr_B_H, Imr_B_L = _ic("Imr_B0", "H"), _ic("Imr_B0", "L")

    Hd_A_H, Hd_A_L = _ic("Hd0", "H"), _ic("Hd0", "L")
    Hr_A_H, Hr_A_L = _ic("Hr0", "H"), _ic("Hr0", "L")
    Hd_B_H, Hd_B_L = _ic("Hd_B0", "H"), _ic("Hd_B0", "L")
    Hr_B_H, Hr_B_L = _ic("Hr_B0", "H"), _ic("Hr_B0", "L")

    D_H, D_L = _ic("D0", "H"), _ic("D0", "L")

    def _residual_S(N_g, V_g, R_g, R_B_g, SxS_g, SxI_g, SxBp_g, SxBpart_g,
                    InhA_g, InrA_g, ImhA_g, ImrA_g,
                    InhB_g, InrB_g, ImhB_g, ImrB_g,
                    HdA_g, HrA_g, HdB_g, HrB_g, D_g):
        return (
            N_g - V_g - R_g - R_B_g
            - SxS_g - SxI_g - SxBp_g - SxBpart_g
            - InhA_g - InrA_g - ImhA_g - ImrA_g
            - InhB_g - InrB_g - ImhB_g - ImrB_g
            - HdA_g - HrA_g - HdB_g - HrB_g
            - D_g
        )

    S_H = _residual_S(N_H, V_H, R_A_H, R_B_H, S_A_prot_H, S_A_part_H, S_B_prot_H, S_B_part_H,
                      Inh_A_H, Inr_A_H, Imh_A_H, Imr_A_H,
                      Inh_B_H, Inr_B_H, Imh_B_H, Imr_B_H,
                      Hd_A_H, Hr_A_H, Hd_B_H, Hr_B_H, D_H)
    S_L = _residual_S(N_L, V_L, R_A_L, R_B_L, S_A_prot_L, S_A_part_L, S_B_prot_L, S_B_part_L,
                      Inh_A_L, Inr_A_L, Imh_A_L, Imr_A_L,
                      Inh_B_L, Inr_B_L, Imh_B_L, Imr_B_L,
                      Hd_A_L, Hr_A_L, Hd_B_L, Hr_B_L, D_L)

    # ---------- Output buffers ----------
    cumcase = np.zeros(horizon_days)
    cumhosp = np.zeros(horizon_days)
    newcase = np.zeros(horizon_days)
    newhosp = np.zeros(horizon_days)
    newhosp_H = np.zeros(horizon_days)
    newhosp_L = np.zeros(horizon_days)
    newdeath = np.zeros(horizon_days)
    beta_path = np.zeros(horizon_days)

    if return_states:
        state_traces = []

    # ---------- Switch dates for epoch-boundary relabeling ----------
    switch_dates = set()
    variant_dates = None
    if isinstance(variant_share_series, pd.Series):
        switch_dates = {
            pd.Timestamp(d)
            for d in variant_share_series.attrs.get("switch_dates", [])
        }
        try:
            variant_dates = pd.DatetimeIndex(variant_share_series.index)
        except Exception:
            variant_dates = None

    # ===================== Main daily integration loop =====================
    for tt in range(horizon_days):
        # ---------- Step 1: resolve date & active epoch ----------
        current_date = None
        if variant_dates is not None:
            try:
                current_date = pd.Timestamp(variant_dates[tt])
            except Exception:
                current_date = None

        current_epoch = _resolve_schedule_entry(transmission_epochs, current_date) if current_date is not None else None
        a_escape = float(current_epoch.get("a_escape", a_escape_default)) if current_epoch else a_escape_default
        b_escape = float(current_epoch.get("b_escape", b_escape_default)) if current_epoch else b_escape_default
        beta_multiplier_b = current_epoch.get("beta_multiplier_b", beta_multiplier_b_default) if current_epoch else beta_multiplier_b_default

        # ---------- Step 2: epoch-boundary relabeling (both groups) ----------
        if current_date is not None and current_date in switch_dates:
            R_A_H, R_B_H = R_A_H + R_B_H, 0.0
            R_A_L, R_B_L = R_A_L + R_B_L, 0.0
            S_A_prot_H, S_B_prot_H = S_A_prot_H + S_B_prot_H, 0.0
            S_A_prot_L, S_B_prot_L = S_A_prot_L + S_B_prot_L, 0.0
            S_A_part_H, S_B_part_H = S_A_part_H + S_B_part_H, 0.0
            S_A_part_L, S_B_part_L = S_A_part_L + S_B_part_L, 0.0
            Inh_A_H, Inh_B_H = Inh_A_H + Inh_B_H, 0.0
            Inh_A_L, Inh_B_L = Inh_A_L + Inh_B_L, 0.0
            Inr_A_H, Inr_B_H = Inr_A_H + Inr_B_H, 0.0
            Inr_A_L, Inr_B_L = Inr_A_L + Inr_B_L, 0.0
            Imh_A_H, Imh_B_H = Imh_A_H + Imh_B_H, 0.0
            Imh_A_L, Imh_B_L = Imh_A_L + Imh_B_L, 0.0
            Imr_A_H, Imr_B_H = Imr_A_H + Imr_B_H, 0.0
            Imr_A_L, Imr_B_L = Imr_A_L + Imr_B_L, 0.0
            Hd_A_H, Hd_B_H = Hd_A_H + Hd_B_H, 0.0
            Hd_A_L, Hd_B_L = Hd_A_L + Hd_B_L, 0.0
            Hr_A_H, Hr_B_H = Hr_A_H + Hr_B_H, 0.0
            Hr_A_L, Hr_B_L = Hr_A_L + Hr_B_L, 0.0

        # ---------- Step 3: stochastic beta update (OU) ----------
        if stochastic_cfg:
            beta_t = beta_t + kappa * (beta_bar - beta_t) + std * rng.standard_normal()
            beta_t = float(np.clip(beta_t, 0.0, 5.0))
        else:
            beta_t = beta_bar
        beta_path[tt] = beta_t

        # ---------- Step 3b: humidity modulation ----------
        if ah_array is not None and tt < len(ah_array):
            ah_today = float(ah_array[tt])
            if not np.isnan(ah_today) and ah_baseline_mean_exp > 0:
                ah_factor = np.exp(-ah_alpha * ah_today) / ah_baseline_mean_exp
            else:
                ah_factor = 1.0
        else:
            ah_factor = 1.0
        beta_eff = beta_t * ah_factor

        # ---------- Step 4: force of infection (homogeneous mixing) ----------
        I_total = (Inh_A_H + Inr_A_H + Imh_A_H + Imr_A_H +
                   Inh_A_L + Inr_A_L + Imh_A_L + Imr_A_L +
                   Inh_B_H + Inr_B_H + Imh_B_H + Imr_B_H +
                   Inh_B_L + Inr_B_L + Imh_B_L + Imr_B_L)
        lambda_t = beta_eff * I_total / N
        lambda_a = (1.0 - p_b[tt]) * lambda_t
        lambda_b = p_b[tt] * beta_multiplier_b * lambda_t

        # ---------- Step 5: daily vaccination rate (split by group) ----------
        week_idx = min(tt // 7, len(vacc_series) - 1)
        if vacc_is_grouped:
            vac_d_H = vacc_series[week_idx, 0] / 7.0
            vac_d_L = vacc_series[week_idx, 1] / 7.0
        else:
            vac_total_d = vacc_series[week_idx] / 7.0
            vac_d_H = vac_total_d * vacc_frac_H
            vac_d_L = vac_total_d * vacc_frac_L

        # ---------- Step 6: susceptibility multipliers (per group) ----------
        # Homologous (same-variant) reinfection susceptibility -- same epsilon1 per group.
        sus_part_hom_H = 1.0 - e1_H
        sus_part_hom_L = 1.0 - e1_L
        # V vs A susceptibility (escape moves toward 1.0):
        sus_v_a_H = 1.0 - (1.0 - a_escape) * e1_H
        sus_v_a_L = 1.0 - (1.0 - a_escape) * e1_L
        sus_v_b_H = 1.0 - (1.0 - b_escape) * e1_H
        sus_v_b_L = 1.0 - (1.0 - b_escape) * e1_L
        # S_A_part vs B (Option A: use B's own escape):
        sus_a_part_b_H = sus_v_b_H
        sus_a_part_b_L = sus_v_b_L

        # ---------- Step 7: new infections (per group, per source) ----------
        # HIGH
        inf_s_a_H = lambda_a * S_H
        inf_s_b_H = lambda_b * S_H
        inf_v_a_H = lambda_a * V_H * sus_v_a_H
        inf_v_b_H = lambda_b * V_H * sus_v_b_H
        inf_sa_part_a_H = lambda_a * S_A_part_H * sus_part_hom_H
        inf_sa_part_b_H = lambda_b * S_A_part_H * sus_a_part_b_H
        inf_sb_part_a_H = lambda_a * S_B_part_H * sus_part_hom_H
        inf_sb_part_b_H = lambda_b * S_B_part_H * sus_part_hom_H

        # LOW
        inf_s_a_L = lambda_a * S_L
        inf_s_b_L = lambda_b * S_L
        inf_v_a_L = lambda_a * V_L * sus_v_a_L
        inf_v_b_L = lambda_b * V_L * sus_v_b_L
        inf_sa_part_a_L = lambda_a * S_A_part_L * sus_part_hom_L
        inf_sa_part_b_L = lambda_b * S_A_part_L * sus_a_part_b_L
        inf_sb_part_a_L = lambda_a * S_B_part_L * sus_part_hom_L
        inf_sb_part_b_L = lambda_b * S_B_part_L * sus_part_hom_L

        # ---------- Step 8: severity split per group (IHR_H vs IHR_L) ----------
        # HIGH
        imm_a_total_H = inf_v_a_H + inf_sa_part_a_H + inf_sb_part_a_H
        imm_b_total_H = inf_v_b_H + inf_sa_part_b_H + inf_sb_part_b_H
        severe_imm_a_H   = imm_a_total_H * IHR_H * (1.0 - epsilon2)
        mild_imm_a_H     = imm_a_total_H * (1.0 - IHR_H * (1.0 - epsilon2))
        severe_imm_b_H   = imm_b_total_H * IHR_H * (1.0 - epsilon2)
        mild_imm_b_H     = imm_b_total_H * (1.0 - IHR_H * (1.0 - epsilon2))
        severe_naive_a_H = inf_s_a_H * IHR_H
        mild_naive_a_H   = inf_s_a_H * (1.0 - IHR_H)
        severe_naive_b_H = inf_s_b_H * IHR_H
        mild_naive_b_H   = inf_s_b_H * (1.0 - IHR_H)

        # LOW
        imm_a_total_L = inf_v_a_L + inf_sa_part_a_L + inf_sb_part_a_L
        imm_b_total_L = inf_v_b_L + inf_sa_part_b_L + inf_sb_part_b_L
        severe_imm_a_L   = imm_a_total_L * IHR_L * (1.0 - epsilon2)
        mild_imm_a_L     = imm_a_total_L * (1.0 - IHR_L * (1.0 - epsilon2))
        severe_imm_b_L   = imm_b_total_L * IHR_L * (1.0 - epsilon2)
        mild_imm_b_L     = imm_b_total_L * (1.0 - IHR_L * (1.0 - epsilon2))
        severe_naive_a_L = inf_s_a_L * IHR_L
        mild_naive_a_L   = inf_s_a_L * (1.0 - IHR_L)
        severe_naive_b_L = inf_s_b_L * IHR_L
        mild_naive_b_L   = inf_s_b_L * (1.0 - IHR_L)

        # ---------- Step 9: progression flows ----------
        hosp_flow_a_H = delta * (Inh_A_H + Imh_A_H)
        hosp_flow_b_H = delta * (Inh_B_H + Imh_B_H)
        hosp_flow_a_L = delta * (Inh_A_L + Imh_A_L)
        hosp_flow_b_L = delta * (Inh_B_L + Imh_B_L)
        death_flow_a_H = eta * Hd_A_H
        death_flow_b_H = eta * Hd_B_H
        death_flow_a_L = eta * Hd_A_L
        death_flow_b_L = eta * Hd_B_L

        # ---------- Step 10: waning ----------
        move_ra_H = rho_a * R_A_H
        move_rb_H = rho_b * R_B_H
        move_ra_L = rho_a * R_A_L
        move_rb_L = rho_b * R_B_L

        # ---------- Step 11: compartment updates ----------
        # Susceptible / vaccinated
        S_H_new = S_H + omega_a * S_A_prot_H + omega_b * S_B_prot_H - vac_d_H - inf_s_a_H - inf_s_b_H
        S_L_new = S_L + omega_a * S_A_prot_L + omega_b * S_B_prot_L - vac_d_L - inf_s_a_L - inf_s_b_L
        V_H_new = V_H + vac_d_H - inf_v_a_H - inf_v_b_H
        V_L_new = V_L + vac_d_L - inf_v_a_L - inf_v_b_L

        # Recovered
        R_A_H_new = R_A_H + gamma1 * (Inr_A_H + Imr_A_H) + gamma2 * Hr_A_H - move_ra_H
        R_B_H_new = R_B_H + gamma1 * (Inr_B_H + Imr_B_H) + gamma2 * Hr_B_H - move_rb_H
        R_A_L_new = R_A_L + gamma1 * (Inr_A_L + Imr_A_L) + gamma2 * Hr_A_L - move_ra_L
        R_B_L_new = R_B_L + gamma1 * (Inr_B_L + Imr_B_L) + gamma2 * Hr_B_L - move_rb_L

        # Waned-protected / partially-susceptible
        S_A_prot_H_new = S_A_prot_H + move_ra_H * (1.0 - re_pro) - omega_a * S_A_prot_H
        S_A_part_H_new = S_A_part_H + move_ra_H * re_pro - inf_sa_part_a_H - inf_sa_part_b_H
        S_B_prot_H_new = S_B_prot_H + move_rb_H * (1.0 - re_pro) - omega_b * S_B_prot_H
        S_B_part_H_new = S_B_part_H + move_rb_H * re_pro - inf_sb_part_a_H - inf_sb_part_b_H
        S_A_prot_L_new = S_A_prot_L + move_ra_L * (1.0 - re_pro) - omega_a * S_A_prot_L
        S_A_part_L_new = S_A_part_L + move_ra_L * re_pro - inf_sa_part_a_L - inf_sa_part_b_L
        S_B_prot_L_new = S_B_prot_L + move_rb_L * (1.0 - re_pro) - omega_b * S_B_prot_L
        S_B_part_L_new = S_B_part_L + move_rb_L * re_pro - inf_sb_part_a_L - inf_sb_part_b_L

        # Infectious - A
        Imh_A_H_new = Imh_A_H + severe_imm_a_H   - delta * Imh_A_H
        Imr_A_H_new = Imr_A_H + mild_imm_a_H     - gamma1 * Imr_A_H
        Inh_A_H_new = Inh_A_H + severe_naive_a_H - delta * Inh_A_H
        Inr_A_H_new = Inr_A_H + mild_naive_a_H   - gamma1 * Inr_A_H
        Imh_A_L_new = Imh_A_L + severe_imm_a_L   - delta * Imh_A_L
        Imr_A_L_new = Imr_A_L + mild_imm_a_L     - gamma1 * Imr_A_L
        Inh_A_L_new = Inh_A_L + severe_naive_a_L - delta * Inh_A_L
        Inr_A_L_new = Inr_A_L + mild_naive_a_L   - gamma1 * Inr_A_L

        # Infectious - B
        Imh_B_H_new = Imh_B_H + severe_imm_b_H   - delta * Imh_B_H
        Imr_B_H_new = Imr_B_H + mild_imm_b_H     - gamma1 * Imr_B_H
        Inh_B_H_new = Inh_B_H + severe_naive_b_H - delta * Inh_B_H
        Inr_B_H_new = Inr_B_H + mild_naive_b_H   - gamma1 * Inr_B_H
        Imh_B_L_new = Imh_B_L + severe_imm_b_L   - delta * Imh_B_L
        Imr_B_L_new = Imr_B_L + mild_imm_b_L     - gamma1 * Imr_B_L
        Inh_B_L_new = Inh_B_L + severe_naive_b_L - delta * Inh_B_L
        Inr_B_L_new = Inr_B_L + mild_naive_b_L   - gamma1 * Inr_B_L

        # Hospitalized and deaths
        Hd_A_H_new = Hd_A_H + hosp_flow_a_H * HDR       - death_flow_a_H
        Hr_A_H_new = Hr_A_H + hosp_flow_a_H * (1.0 - HDR) - gamma2 * Hr_A_H
        Hd_B_H_new = Hd_B_H + hosp_flow_b_H * HDR       - death_flow_b_H
        Hr_B_H_new = Hr_B_H + hosp_flow_b_H * (1.0 - HDR) - gamma2 * Hr_B_H
        Hd_A_L_new = Hd_A_L + hosp_flow_a_L * HDR       - death_flow_a_L
        Hr_A_L_new = Hr_A_L + hosp_flow_a_L * (1.0 - HDR) - gamma2 * Hr_A_L
        Hd_B_L_new = Hd_B_L + hosp_flow_b_L * HDR       - death_flow_b_L
        Hr_B_L_new = Hr_B_L + hosp_flow_b_L * (1.0 - HDR) - gamma2 * Hr_B_L
        D_H_new = D_H + death_flow_a_H + death_flow_b_H
        D_L_new = D_L + death_flow_a_L + death_flow_b_L

        # ---------- Step 12: accumulate output ----------
        total_new_cases = (inf_s_a_H + inf_s_b_H + imm_a_total_H + imm_b_total_H +
                           inf_s_a_L + inf_s_b_L + imm_a_total_L + imm_b_total_L)
        total_new_hosp_H = hosp_flow_a_H + hosp_flow_b_H
        total_new_hosp_L = hosp_flow_a_L + hosp_flow_b_L
        total_new_hosp   = total_new_hosp_H + total_new_hosp_L
        total_new_death  = (death_flow_a_H + death_flow_b_H +
                            death_flow_a_L + death_flow_b_L)

        cumcase[tt] = (cumcase[tt - 1] if tt > 0 else 0.0) + total_new_cases
        cumhosp[tt] = (cumhosp[tt - 1] if tt > 0 else 0.0) + total_new_hosp
        newcase[tt] = total_new_cases
        newhosp[tt] = total_new_hosp
        newhosp_H[tt] = total_new_hosp_H
        newhosp_L[tt] = total_new_hosp_L
        newdeath[tt] = total_new_death

        if return_states:
            state_traces.append({
                "S_H": S_H, "S_L": S_L, "V_H": V_H, "V_L": V_L,
                "R_A_H": R_A_H, "R_A_L": R_A_L, "R_B_H": R_B_H, "R_B_L": R_B_L,
                "I_total": I_total, "p_B": p_b[tt], "beta_t": beta_t,
                "a_escape": a_escape, "b_escape": b_escape,
                "ah_factor": ah_factor,
            })

        # ---------- Step 13: advance ----------
        S_H, S_L = S_H_new, S_L_new
        V_H, V_L = V_H_new, V_L_new
        R_A_H, R_A_L = R_A_H_new, R_A_L_new
        R_B_H, R_B_L = R_B_H_new, R_B_L_new
        S_A_prot_H, S_A_prot_L = S_A_prot_H_new, S_A_prot_L_new
        S_A_part_H, S_A_part_L = S_A_part_H_new, S_A_part_L_new
        S_B_prot_H, S_B_prot_L = S_B_prot_H_new, S_B_prot_L_new
        S_B_part_H, S_B_part_L = S_B_part_H_new, S_B_part_L_new
        Inh_A_H, Inh_A_L = Inh_A_H_new, Inh_A_L_new
        Inr_A_H, Inr_A_L = Inr_A_H_new, Inr_A_L_new
        Imh_A_H, Imh_A_L = Imh_A_H_new, Imh_A_L_new
        Imr_A_H, Imr_A_L = Imr_A_H_new, Imr_A_L_new
        Inh_B_H, Inh_B_L = Inh_B_H_new, Inh_B_L_new
        Inr_B_H, Inr_B_L = Inr_B_H_new, Inr_B_L_new
        Imh_B_H, Imh_B_L = Imh_B_H_new, Imh_B_L_new
        Imr_B_H, Imr_B_L = Imr_B_H_new, Imr_B_L_new
        Hd_A_H, Hd_A_L = Hd_A_H_new, Hd_A_L_new
        Hr_A_H, Hr_A_L = Hr_A_H_new, Hr_A_L_new
        Hd_B_H, Hd_B_L = Hd_B_H_new, Hd_B_L_new
        Hr_B_H, Hr_B_L = Hr_B_H_new, Hr_B_L_new
        D_H, D_L = D_H_new, D_L_new

    # ===================== Outputs =====================
    # Same reporting discount as hosp by default (rep=0.2): the "true" internal
    # hosp/death flows are coupled (death derives from the same unscaled hosp
    # flow via HDR), so reporting deaths at full ascertainment (rep_death=1.0)
    # while hosp is discounted 5x would inflate deaths relative to hosp with no
    # independent evidence for that asymmetry.
    rep_death     = float(model_params.get("rep_death", rep))
    weekly_hosp   = _weekly_sum(newhosp,   weekly_index) * rep
    weekly_hosp_H = _weekly_sum(newhosp_H, weekly_index) * rep
    weekly_hosp_L = _weekly_sum(newhosp_L, weekly_index) * rep
    weekly_death  = _weekly_sum(newdeath,  weekly_index) * rep_death

    result = {
        "weekly_hosp":   weekly_hosp,    # total observed-scale weekly hosp (fit target)
        "weekly_hosp_H": weekly_hosp_H,
        "weekly_hosp_L": weekly_hosp_L,
        "weekly_death":  weekly_death,   # incident weekly deaths (rep_death defaults to rep=0.2)
        "cumcase": cumcase,
        "cumhosp": cumhosp,
        "newcase": newcase,
        "newhosp": newhosp,
        "newhosp_H": newhosp_H,
        "newhosp_L": newhosp_L,
        "newdeath": newdeath,
        "beta_path": beta_path,
        "variant_share": p_b,
    }
    if return_states:
        result["state_traces"] = pd.DataFrame(state_traces)
    return result
