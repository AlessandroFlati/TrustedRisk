"""SCALE-3 -- Sequential decision modeling via finite-horizon MDP.

Models the 90-day post-discharge trajectory as a Markov chain over four
states {home_well, home_sick, readmitted, deceased} with three 30-day
stages (t30, t60, t90). Per discharge action, the per-stage transition
matrix is parameterized by:
  - baseline 30-day readmission probability (from compute_readmission_risk)
  - action-conditional risk reduction (RRR -- literature-derived for SNF/HWC)
  - mortality given readmission (Krumholz JAMA 2013 ~5%)
  - background morbidity (home_well -> home_sick) ~3% per stage

Backward induction (Bellman) computes the value V(state) at each stage,
yielding the optimal action at t0 plus per-action expected values.

This complements compute_decision_utility (single-stage QALY) by adding
the temporal cascade -- a high-risk patient discharged_home today may have
a large negative expected value at t90 even when the t30 score looks
acceptable.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np

from shared.schemas import (
    MDPActionValue,
    MDPDecisionReport,
    RiskEstimate,
)


# ─────────────────────── State + reward priors ───────────────────────

_STATE_NAMES: list[str] = [
    "home_well", "home_sick", "readmitted", "deceased",
]
_S_HOME_WELL = 0
_S_HOME_SICK = 1
_S_READMIT = 2
_S_DECEASED = 3

# Per-state per-stage rewards (utility units, NOT $). 30 days each stage.
# Calibrated against rough QALY-equivalents: a healthy 30-day stretch ≈
# 0.083 QALY (1/12 of a year); morbidity -> 0.7 weight; readmission -> -0.5;
# death -> terminal.
_STAGE_REWARD: list[float] = [0.083, 0.058, -0.50, -2.0]
_GAMMA_PER_STAGE = 0.97


# Action priors:
#   rrr            -- relative risk reduction in readmission probability
#   morbidity_mult -- multiplier on background morbidity (home_well->home_sick)
#   utility_cost   -- one-time disutility deducted at t0, expressed in QALY-
#                    equivalent units. Captures the resource cost of more
#                    aggressive dispositions (HWC home-visits, SNF days,
#                    continued hospital days) using a $100 000/QALY WTP
#                    threshold to translate $ -> QALY equivalents.
_ACTION_PRIORS: dict[str, dict[str, float]] = {
    "discharge_home":      {"rrr": 0.00, "morbidity_mult": 1.00,
                              "utility_cost": 0.00},
    "home_with_care":      {"rrr": 0.25, "morbidity_mult": 0.85,
                              "utility_cost": 0.003},
    "snf":                 {"rrr": 0.40, "morbidity_mult": 0.70,
                              "utility_cost": 0.05},
    "continued_admission": {"rrr": 0.55, "morbidity_mult": 0.55,
                              "utility_cost": 0.30},
}


# Background priors -- these come from the readmission-mortality literature
# (Krumholz JAMA 2013) and are constant across actions in v1; institutions
# can re-fit them via override.
_P_DEATH_GIVEN_READMIT = 0.05
_P_BACKGROUND_MORBIDITY = 0.03   # home_well -> home_sick per stage
_P_RECOVERY_FROM_SICK = 0.40     # home_sick -> home_well per stage
_P_REHOSP_FROM_SICK_MULT = 1.5   # readmission risk multiplier when home_sick


# ─────────────────────── Transition matrix per action ───────────────────────

def _transition_matrix(
    baseline_p_readmit: float,
    action: str,
    *,
    death_given_readmit: float = _P_DEATH_GIVEN_READMIT,
    p_morbidity: float = _P_BACKGROUND_MORBIDITY,
    p_recovery: float = _P_RECOVERY_FROM_SICK,
    p_rehosp_mult: float = _P_REHOSP_FROM_SICK_MULT,
) -> np.ndarray:
    """Return a 4×4 row-stochastic matrix T[s, s'] for the chosen action.

    Rows: from-state (home_well, home_sick, readmitted, deceased).
    Cols: to-state (same order).
    """
    priors = _ACTION_PRIORS[action]
    rrr = float(priors["rrr"])
    morb_mult = float(priors["morbidity_mult"])

    p_readmit_well = max(0.0, baseline_p_readmit * (1.0 - rrr))
    p_readmit_sick = max(0.0, min(1.0,
                                       baseline_p_readmit * (1.0 - rrr) * p_rehosp_mult))
    p_morb = p_morbidity * morb_mult

    T = np.zeros((4, 4), dtype=np.float64)

    # From home_well
    T[_S_HOME_WELL, _S_READMIT] = p_readmit_well
    T[_S_HOME_WELL, _S_HOME_SICK] = max(0.0, p_morb - p_readmit_well * 0.0)
    T[_S_HOME_WELL, _S_HOME_WELL] = max(
        0.0, 1.0 - T[_S_HOME_WELL, _S_READMIT] - T[_S_HOME_WELL, _S_HOME_SICK])

    # From home_sick
    T[_S_HOME_SICK, _S_READMIT] = p_readmit_sick
    T[_S_HOME_SICK, _S_HOME_WELL] = p_recovery * (1.0 - p_readmit_sick)
    T[_S_HOME_SICK, _S_HOME_SICK] = max(
        0.0, 1.0 - T[_S_HOME_SICK, _S_READMIT] - T[_S_HOME_SICK, _S_HOME_WELL])

    # From readmitted: small mortality, otherwise back to sick recovery path
    T[_S_READMIT, _S_DECEASED] = death_given_readmit
    T[_S_READMIT, _S_HOME_SICK] = (1.0 - death_given_readmit) * 0.7
    T[_S_READMIT, _S_HOME_WELL] = (1.0 - death_given_readmit) * 0.30
    T[_S_READMIT, _S_READMIT] = max(
        0.0, 1.0 - T[_S_READMIT, _S_DECEASED]
        - T[_S_READMIT, _S_HOME_SICK]
        - T[_S_READMIT, _S_HOME_WELL])

    # Deceased is absorbing
    T[_S_DECEASED, _S_DECEASED] = 1.0

    # Defensive renormalization: clip negatives + renormalize rows
    T = np.clip(T, 0.0, 1.0)
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    T = T / row_sums
    return T


# ─────────────────────── Backward induction ───────────────────────

def _value_for_action(
    T: np.ndarray, n_stages: int, gamma: float,
    initial_state: int = _S_HOME_WELL,
) -> dict[str, float]:
    """Run backward induction; return summary stats of running this action."""
    R = np.array(_STAGE_REWARD, dtype=np.float64)
    V = np.zeros(4, dtype=np.float64)  # terminal V = 0

    # Backward induction: V_t(s) = R(s) + gamma * sum_s' T(s,s') V_{t+1}(s')
    for _ in range(n_stages):
        V = R + gamma * (T @ V)

    expected_value = float(V[initial_state])

    # Propagate state distribution forward to compute outcome probabilities
    p = np.zeros(4); p[initial_state] = 1.0
    for _ in range(n_stages):
        p = p @ T
    p_alive = float(1.0 - p[_S_DECEASED])
    p_readmit_anywhere = float(_marginal_readmit(T, n_stages, initial_state))

    # QALY in window: sum of stage rewards weighted by occupancy probabilities.
    p_state = np.zeros(4); p_state[initial_state] = 1.0
    qaly_total = 0.0
    for _ in range(n_stages):
        p_state = p_state @ T
        # Convert utility to QALY contribution (30-day window): R is already
        # a 30-day reward in QALY-like units; we sum across stages and clip
        # so deceased / negative contributions don't drive QALY < 0.
        for state_idx, weight in enumerate(p_state):
            stage_q = max(0.0, _STAGE_REWARD[state_idx]) * weight
            qaly_total += stage_q

    return {
        "expected_value": expected_value,
        "p_alive": p_alive,
        "p_readmit": p_readmit_anywhere,
        "qaly": qaly_total,
    }


def _marginal_readmit(T: np.ndarray, n_stages: int,
                          initial_state: int) -> float:
    """Probability the patient is in the readmitted state at any stage in
    [1, n_stages]."""
    p = np.zeros(4); p[initial_state] = 1.0
    p_ever_readmit = 0.0
    p_not_readmit = 1.0
    for _ in range(n_stages):
        p = p @ T
        # Probability of being in readmitted state this stage
        p_this_stage_readmit = float(p[_S_READMIT])
        # Approx -- assume independence; clip to [0,1]
        p_ever_readmit = min(1.0, p_ever_readmit + p_this_stage_readmit
                                * p_not_readmit)
        p_not_readmit *= max(0.0, 1.0 - p_this_stage_readmit)
    return p_ever_readmit


# ─────────────────────── Public API ───────────────────────

def compute_sequential_mdp_value(
    baseline_readmission_30d_prob: float | None = None,
    risk_estimate: RiskEstimate | dict | None = None,
    horizon_days: int = 90,
    patient_id: str | None = None,
    initial_state: Literal["home_well", "home_sick"] = "home_well",
    discount_factor: float = _GAMMA_PER_STAGE,
    actions: Iterable[str] | None = None,
) -> MDPDecisionReport:
    """Run a finite-horizon MDP over the 4-state post-discharge model.

    Args:
        baseline_readmission_30d_prob: required if `risk_estimate` is None.
            Per-30-day probability of readmission for an UNTREATED patient.
        risk_estimate: alternative input -- pulls `probability_mean`.
        horizon_days: simulation horizon (1-365). 90 days is the canonical
            CMS quality measure window.
        patient_id: optional -- propagated into the report.
        initial_state: "home_well" by default; "home_sick" for a patient
            already showing morbidity at discharge.
        discount_factor: per-stage gamma. 0.97 ≈ 1.5%/month implicit
            discount for healthcare CEA.
        actions: subset of the 4 canonical actions; defaults to all 4.

    Returns:
        MDPDecisionReport with per-action expected values + the optimal
        action under Bellman backward induction.
    """
    if risk_estimate is not None:
        if isinstance(risk_estimate, dict):
            baseline = float(risk_estimate.get("probability_mean", 0.0))
        else:
            baseline = float(risk_estimate.probability_mean)
    elif baseline_readmission_30d_prob is not None:
        baseline = float(baseline_readmission_30d_prob)
    else:
        raise ValueError(
            "Either baseline_readmission_30d_prob or risk_estimate "
            "must be provided.")

    if not (0.0 <= baseline <= 1.0):
        raise ValueError(
            f"baseline must be in [0, 1], got {baseline}")
    if horizon_days < 1 or horizon_days > 365:
        raise ValueError("horizon_days must be in [1, 365].")
    if not (0.0 <= discount_factor <= 1.0):
        raise ValueError("discount_factor must be in [0, 1].")

    n_stages = max(1, horizon_days // 30)
    initial_idx = _S_HOME_WELL if initial_state == "home_well" \
        else _S_HOME_SICK

    if actions is None:
        actions = list(_ACTION_PRIORS.keys())
    else:
        actions = [a for a in actions if a in _ACTION_PRIORS]
        if not actions:
            raise ValueError(
                "actions must overlap the supported set "
                f"{list(_ACTION_PRIORS.keys())}.")

    action_values: list[MDPActionValue] = []
    for action in actions:
        T = _transition_matrix(baseline, action)
        stats = _value_for_action(T, n_stages, discount_factor, initial_idx)
        utility_cost = float(_ACTION_PRIORS[action]["utility_cost"])
        ev = float(stats["expected_value"]) - utility_cost
        action_values.append(MDPActionValue(
            action=action,                      # type: ignore[arg-type]
            expected_value=ev,
            probability_alive_at_horizon=max(0.0, min(1.0,
                                                            float(stats["p_alive"]))),
            probability_readmitted_in_window=max(0.0, min(1.0,
                                                                float(stats["p_readmit"]))),
            expected_qaly_in_window=float(stats["qaly"]),
        ))

    optimal = max(action_values, key=lambda a: a.expected_value)

    rationale = (
        f"Backward induction over {n_stages} 30-day stage(s) "
        f"(γ={discount_factor:.2f}) starting from {initial_state}. "
        f"Baseline 30d readmission={baseline:.3f}. "
        f"Optimal action: {optimal.action} "
        f"(expected_value={optimal.expected_value:.3f}, "
        f"p_alive={optimal.probability_alive_at_horizon:.3f})."
    )

    return MDPDecisionReport(
        patient_id=patient_id,
        horizon_days=horizon_days,
        n_stages=n_stages,
        discount_factor=discount_factor,
        baseline_readmission_30d_prob=baseline,
        action_values=action_values,
        optimal_action=optimal.action,    # type: ignore[arg-type]
        optimal_action_expected_value=optimal.expected_value,
        rationale=rationale,
    )
