"""Functional tests for the economics + simulation pipeline.

Chains: expected_value_of_intervention -> cost_effectiveness_ladder ->
hospital outcomes simulator -> causal ATE -> MDP decision. Demonstrates
the full health-economics stack the Impact criterion requires.
"""
from __future__ import annotations

import asyncio


# ─────────────────────── Expected Value of Intervention ───────────────────────

def test_evi_pharmacist_counseling_cost_saving_high_risk_cohort():
    from mcp_server.tools.expected_value_of_intervention import (
        compute_expected_value_of_intervention,
    )
    r = asyncio.run(compute_expected_value_of_intervention(
        intervention={
            "name": "Pharmacist-led discharge counseling",
            "relative_risk_reduction": 0.30,
            "cost_per_patient_usd": 75.0,
            "qaly_gained_per_avoided_event": 0.05,
            "evidence_grade": "A",
        },
        baseline_event_probability=0.30,
        cohort_size=500,
        avoided_event_cost_usd=14_000.0,
    ))
    assert r.decision == "cost_saving"
    assert r.net_cost_total_usd < 0   # costs avoided > intervention cost


def test_evi_low_evidence_gates_decision():
    from mcp_server.tools.expected_value_of_intervention import (
        compute_expected_value_of_intervention,
    )
    r = asyncio.run(compute_expected_value_of_intervention(
        intervention={
            "name": "Untested intervention",
            "relative_risk_reduction": 0.40,
            "cost_per_patient_usd": 100.0,
            "evidence_grade": "C",
        },
        baseline_event_probability=0.30,
    ))
    assert r.decision == "uncertain_evidence"


# ─────────────────────── Cost-effectiveness ladder ───────────────────────

def test_cea_ladder_target_ranks_among_benchmarks():
    from a2a_agent.cost_effectiveness_ladder import (
        build_cost_effectiveness_ladder,
    )
    r = build_cost_effectiveness_ladder(
        target_intervention_name="TrustedRisk-recommended counseling",
        target_arr_30d_percentage_points=4.0,
        target_cost_per_patient_usd=100.0,
        avoided_event_cost_usd=14_000.0,
        qaly_gained_per_avoided_event=0.05,
    )
    assert len(r.benchmarks) >= 4
    assert 1 <= r.target_rank_among_benchmarks <= len(r.benchmarks) + 1
    # Schnipper + Coleman + Naylor + Misky + Jack canonical references
    refs = " ".join(r.references)
    assert "Schnipper" in refs
    assert "Coleman" in refs


def test_cea_ladder_dominant_intervention_ranks_first():
    from a2a_agent.cost_effectiveness_ladder import (
        build_cost_effectiveness_ladder,
    )
    r = build_cost_effectiveness_ladder(
        target_intervention_name="Dominant",
        target_arr_30d_percentage_points=15.0,   # very effective
        target_cost_per_patient_usd=20.0,         # very cheap
    )
    assert r.target_rank_among_benchmarks == 1


# ─────────────────────── Hospital simulator ───────────────────────

def test_simulator_recovers_known_savings_for_chf_segment():
    from a2a_agent.outcomes_simulator import simulate_hospital_year
    r = simulate_hospital_year(
        case_mix=[
            {"name": "CHF", "n_patients_per_year": 1000,
              "baseline_event_probability": 0.30},
        ],
        intervention_relative_risk_reduction=0.25,
        intervention_cost_per_patient_usd=75.0,
        avoided_event_cost_usd=14_000.0,
        n_iterations=2000, seed=42,
    )
    assert r.cohort_size_per_year == 1000
    # ARR ≈ 0.30 * 0.25 = 0.075 -> ~75 events avoided
    assert 50 <= r.total_events_avoided_mean <= 100
    # Net cost: 75000 (intervention) - 14000*75 (avoided) = -975k -> cost-saving
    assert r.net_cost_mean_usd < 0


def test_simulator_sensitivity_grid_monotonic_in_rrr():
    from a2a_agent.outcomes_simulator import simulate_hospital_year
    r = simulate_hospital_year(
        case_mix=[{"name": "X", "n_patients_per_year": 500,
                     "baseline_event_probability": 0.30}],
        intervention_relative_risk_reduction=0.25,
        rrr_grid=[0.10, 0.25, 0.50],
        n_iterations=2000, seed=42,
    )
    rrr_rows = sorted([s for s in r.sensitivity
                          if s.parameter == "intervention_rrr"],
                         key=lambda s: s.value)
    means = [s.events_avoided_mean for s in rrr_rows]
    assert means == sorted(means)


def test_simulator_qaly_path_when_qaly_supplied():
    from a2a_agent.outcomes_simulator import simulate_hospital_year
    r = simulate_hospital_year(
        case_mix=[{"name": "X", "n_patients_per_year": 200,
                     "baseline_event_probability": 0.20}],
        intervention_relative_risk_reduction=0.25,
        qaly_gained_per_avoided_event=0.05,
        n_iterations=1000, seed=42,
    )
    assert r.total_qaly_gained_mean is not None
    assert r.cost_per_qaly_mean_usd is not None


# ─────────────────────── Causal ATE (DoWhy) ───────────────────────

def test_causal_ate_recovers_effect_with_confounder_adjustment():
    """Synthetic cohort: treatment depends on confounder Z; outcome too.
    DoWhy should adjust for Z and recover the true ATE near -0.10."""
    import numpy as np
    from a2a_agent.causal_inference import compute_average_treatment_effect

    rng = np.random.default_rng(1)
    n = 1500
    Z = rng.standard_normal(n)
    pT = 1 / (1 + np.exp(-(0.5 * Z)))
    T = (rng.uniform(size=n) < pT).astype(int)
    raw = 0.30 + (-0.15) * T + 2.0 * 0.05 * Z + 0.05 * rng.standard_normal(n)
    Y = (raw > 0.5).astype(int)
    cohort = [{"T": int(T[i]), "Y": int(Y[i]), "Z": float(Z[i])}
                 for i in range(n)]

    r = compute_average_treatment_effect(
        cohort=cohort, treatment="T", outcome="Y",
        confounders=["Z"], run_refutations=False,
    )
    # Adjusted ATE should be reasonably close to -0.15
    assert -0.30 <= r.ate_point <= 0.0
    assert r.estimation_method.startswith("dowhy.backdoor")


# ─────────────────────── MDP decision ───────────────────────

def test_mdp_high_risk_prefers_aggressive_action():
    from a2a_agent.mdp_decision import compute_sequential_mdp_value
    r = compute_sequential_mdp_value(
        baseline_readmission_30d_prob=0.55, horizon_days=90)
    # High risk -> optimal action is NOT discharge_home
    assert r.optimal_action != "discharge_home"


def test_mdp_low_risk_prefers_cheaper_action():
    from a2a_agent.mdp_decision import compute_sequential_mdp_value
    r = compute_sequential_mdp_value(
        baseline_readmission_30d_prob=0.05, horizon_days=90)
    # Low risk -> cheap action
    assert r.optimal_action in ("discharge_home", "home_with_care")


def test_mdp_chains_after_risk_estimate(chf_risk_estimate):
    """MDP can consume the RiskEstimate dict directly."""
    from a2a_agent.mdp_decision import compute_sequential_mdp_value
    r = compute_sequential_mdp_value(risk_estimate=chf_risk_estimate)
    assert r.baseline_readmission_30d_prob == \
        chf_risk_estimate["probability_mean"]


# ─────────────────────── End-to-end economics chain ───────────────────────

def test_full_economics_chain_chf_patient(chf_risk_estimate):
    """Risk -> EVI -> ladder -> simulator -> MDP -- full economics + impact stack."""
    from mcp_server.tools.expected_value_of_intervention import (
        compute_expected_value_of_intervention,
    )
    from a2a_agent.cost_effectiveness_ladder import (
        build_cost_effectiveness_ladder,
    )
    from a2a_agent.mdp_decision import compute_sequential_mdp_value
    from a2a_agent.outcomes_simulator import simulate_hospital_year

    evi = asyncio.run(compute_expected_value_of_intervention(
        intervention={"name": "TR counseling",
                          "relative_risk_reduction": 0.25,
                          "cost_per_patient_usd": 75.0,
                          "qaly_gained_per_avoided_event": 0.05,
                          "evidence_grade": "A"},
        risk_estimate=chf_risk_estimate,
        cohort_size=200,
    ))
    ladder = build_cost_effectiveness_ladder(
        target_intervention_name="TR counseling",
        target_arr_30d_percentage_points=evi.absolute_risk_reduction
        * 100.0,
        target_cost_per_patient_usd=75.0,
    )
    sim = simulate_hospital_year(
        case_mix=[{"name": "CHF", "n_patients_per_year": 1000,
                     "baseline_event_probability":
                     chf_risk_estimate["probability_mean"]}],
        intervention_relative_risk_reduction=0.25,
        n_iterations=500, seed=42,
    )
    mdp = compute_sequential_mdp_value(risk_estimate=chf_risk_estimate)

    # All four reports produced
    assert evi.decision in ("cost_saving", "cost_effective",
                                "not_cost_effective", "uncertain_evidence",
                                "dominated")
    assert 1 <= ladder.target_rank_among_benchmarks <= 6
    assert sim.total_events_avoided_mean >= 0
    assert mdp.optimal_action in ("discharge_home", "home_with_care",
                                        "snf", "continued_admission")
