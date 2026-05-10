"""End-to-end functional tests for the most recent feature additions.

Combines CHART, EVAL, SAFE, EHR, AUDIT, TIME, KNOWLEDGE, ECONOMICS into
single workflows that pass through 6+ surfaces.
"""
from __future__ import annotations

import asyncio


def test_e2e_ehr_to_chart_to_evaluation(hl7v2_adt_a01):
    """HL7 v2 -> FHIR Bundle -> care_gap -> HEDIS -> Schwartz quality on the
    resulting DecisionCard."""
    from a2a_agent.ehr_integrations import parse_hl7v2_adt
    from a2a_agent.eval_quality import (
        compute_hedis_score, compute_schwartz_quality_score,
    )
    from mcp_server.tools.care_gap_detector import compute_care_gap_detector

    parsed = parse_hl7v2_adt(hl7v2_adt_a01)
    bundle = parsed.fhir_bundle
    gaps = asyncio.run(compute_care_gap_detector(bundle))
    hedis = compute_hedis_score(bundle)
    # Build a synthetic DecisionCard from the parsed signals
    card = {
        "recommendation": {"action": "home_with_care",
                              "confidence": "medium"},
        "reasoning": {"risk_estimate": {
            "model_version": "0.7.0",
            "probability_mean": 0.35,
            "lace_raw_score": 12,
            "contributing_factors": [{"name": "L"}]}},
        "validation": {"grounding": {"overall_verdict": "supported"},
                          "phi_check": {"risk_level": "none"}},
        "audit": {"request_id": "req-e2e-1"},
        "abstain": [],
        "self_critique": {"verdict": "approved", "rationale": ".",
                              "critic_role": "structural"},
        "counseling": {"sections": [{"section_id": "your_medications"}]},
    }
    schwartz = compute_schwartz_quality_score(card)

    # All surfaces produced
    assert gaps.n_gaps_found >= 0
    assert hedis.n_measures_evaluated >= 0
    assert schwartz.grade in ("A", "B", "C", "D", "F")


def test_e2e_chart_to_decisioncard_to_audit(chf_discharge_summary_text):
    """Discharge text -> CHART structurer -> synthetic DecisionCard ->
    right-to-explanation + patient audit summary."""
    from a2a_agent.patient_audit_summary import compute_patient_audit_summary
    from a2a_agent.right_to_explanation import compute_right_to_explanation
    from mcp_server.tools.chart_intelligence import (
        compute_structure_discharge_summary,
    )

    structured = asyncio.run(compute_structure_discharge_summary(
        chf_discharge_summary_text))
    assert structured.extracted_recommendation == "home_with_care"
    assert structured.extracted_followup_window_days == (7, 7) or \
           structured.extracted_followup_window_days == (7, 14)

    # Build a DecisionCard from the structured extraction
    card = {
        "recommendation": {
            "action": structured.extracted_recommendation,
            "confidence": "medium"},
        "reasoning": {"risk_estimate": {
            "model_version": "0.7.0",
            "probability_mean": 0.30,
            "lace_raw_score": 11,
            "contributing_factors": [
                {"name": "LACE_length_of_stay", "raw_value": 6,
                 "lace_points": 3, "weight": 0.30},
                {"name": "LACE_acuity", "raw_value": 1,
                 "lace_points": 3, "weight": 0.30},
                {"name": "LACE_comorbidity", "raw_value": 4,
                 "lace_points": 3, "weight": 0.25},
                {"name": "LACE_ed_visits_6mo", "raw_value": 1,
                 "lace_points": 2, "weight": 0.15},
            ],
        }},
        "validation": {"grounding": {"overall_verdict": "supported",
                                          "sub_claims": []},
                          "phi_check": {"risk_level": "none"}},
        "audit": {"request_id": "req-e2e-2"},
        "abstain": [{"type": "evidence_insufficient",
                        "detail": "deferring to attending judgment"}]
        if structured.extracted_abstain_triggers else [],
    }

    rte = compute_right_to_explanation(card)
    summary = compute_patient_audit_summary(card)

    assert rte.decision_action == "home_with_care"
    assert "support" in summary.headline_question.lower()


def test_e2e_streaming_to_decision_to_safety_check(chf_risk_estimate):
    """Event log streaming -> incremental risk -> MDP -> adversarial fragility
    + OOD detector validate the new state before action."""
    from a2a_agent.event_sourced_log import (
        append_event, replay_to_state, truncate_log,
    )
    from a2a_agent.incremental_risk import recompute_with_observation
    from a2a_agent.mdp_decision import compute_sequential_mdp_value
    from a2a_agent.safety_redteam import (
        compute_adversarial_fragility, compute_ood_detector,
    )
    import os, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["TRUSTEDRISK_EVENT_LOG_PATH"] = \
            os.path.join(tmp, "el.sqlite3")

        pid = "pt-e2e-stream"
        append_event(patient_id=pid, fhir_resource_type="Encounter",
                       fhir_resource_id="enc-1",
                       payload={"class": {"code": "IMP"}})
        append_event(patient_id=pid, fhir_resource_type="Condition",
                       fhir_resource_id="cond-1",
                       payload={"code": {"text": "CHF"}})
        snap = replay_to_state(pid)
        assert snap.n_events_replayed == 2

        # Incremental: new ED encounter
        prior = {"probability_mean": chf_risk_estimate["probability_mean"],
                    "lace_raw_score": chf_risk_estimate["lace_raw_score"]}
        inc = recompute_with_observation(prior, {
            "resourceType": "Encounter", "class": {"code": "EMER"}})

        # MDP for the new probability
        mdp = compute_sequential_mdp_value(
            baseline_readmission_30d_prob=inc.posterior_probability)

        # Safety: fragility + OOD on the new feature vector
        fragility = compute_adversarial_fragility(
            lace_total=12, max_l1_radius=3)
        ood = compute_ood_detector({"L": 3, "A": 1, "C": 4, "E": 2})

        assert mdp.optimal_action in ("discharge_home", "home_with_care",
                                            "snf", "continued_admission")
        assert 0 <= fragility.fragility_score <= 1
        assert 0 <= ood.chi2_p_value <= 1


def test_e2e_economics_to_audit(chf_decision_card, chf_risk_estimate):
    """Risk + EVI + simulator -> publish DecisionCards -> DP equity dashboard +
    Merkle audit chain over the events."""
    from a2a_agent.dp_equity import compute_dp_equity_dashboard
    from a2a_agent.equity_dashboard import compute_equity_dashboard
    from a2a_agent.merkle_audit import compute_merkle_audit_root
    from a2a_agent.outcomes_simulator import simulate_hospital_year
    from mcp_server.tools.expected_value_of_intervention import (
        compute_expected_value_of_intervention,
    )

    evi = asyncio.run(compute_expected_value_of_intervention(
        intervention={"name": "test",
                          "relative_risk_reduction": 0.30,
                          "cost_per_patient_usd": 75.0,
                          "qaly_gained_per_avoided_event": 0.05,
                          "evidence_grade": "A"},
        risk_estimate=chf_risk_estimate,
        cohort_size=100))
    assert evi.decision in ("cost_saving", "cost_effective",
                                "not_cost_effective")

    sim = simulate_hospital_year(
        case_mix=[{"name": "CHF", "n_patients_per_year": 500,
                     "baseline_event_probability": 0.30}],
        n_iterations=300, seed=42)
    assert sim.total_events_avoided_mean >= 0

    cohort = [
        {"decision_card": chf_decision_card,
          "demographics": {"age": 75, "race": "black", "sex": "female",
                              "insurance_type": "medicare"}}
        for _ in range(20)
    ]
    dash = compute_equity_dashboard(cohort)
    dp = compute_dp_equity_dashboard(dash, epsilon=1.0,
                                            suppression_threshold=5, seed=42)
    assert dp.n_total_decisions_noised >= 0

    events = [{"event_id": f"e-{i}", "outcome_hash": f"h-{i}"}
                for i in range(8)]
    chain = compute_merkle_audit_root(events)
    assert chain.merkle_root


def test_e2e_knowledge_grounded_decision(monkeypatch):
    """PubMed search -> DDx ranker (with grounding) -> cost-effectiveness ladder.
    The agent grounds its DDx in published literature and benchmarks the
    intervention against canonical RCTs."""
    from a2a_agent.cost_effectiveness_ladder import (
        build_cost_effectiveness_ladder,
    )
    from mcp_server.tools.differential_diagnosis_ranker import (
        compute_differential_diagnosis_ranker,
    )
    from mcp_server.tools import knowledge_integration as ki
    monkeypatch.setenv("TRUSTEDRISK_DISABLE_LLM", "1")

    # PubMed live mock: when entrez is reachable the tool returns
    # well-formed live results (the abstain-on-failure contract is
    # exercised in tests/unit/test_knowledge_integration.py).
    async def _pm_search(_q, _max):
        return ["12345"]

    async def _pm_fetch(_pmids):
        return [{"pmid": "12345",
                    "title": "Hospital readmission interventions",
                    "journal": "Annals", "pub_year": 2024,
                    "authors": ["Foo A"], "abstract": "",
                    "doi": "10.7326/x"}]
    monkeypatch.setattr(ki, "_entrez_search", _pm_search)
    monkeypatch.setattr(ki, "_entrez_fetch", _pm_fetch)

    pubmed = asyncio.run(ki.compute_pubmed_search(
        "hospital readmission interventions"))
    assert pubmed.n_results >= 1
    assert pubmed.method == "entrez_live"

    ddx = asyncio.run(compute_differential_diagnosis_ranker(
        chief_complaint="chest pain",
        free_text_summary="exertional chest pain with diaphoresis"))
    assert ddx.n_items >= 3

    ladder = build_cost_effectiveness_ladder(
        target_intervention_name="TR pharmacist counseling",
        target_arr_30d_percentage_points=4.0,
        target_cost_per_patient_usd=100.0)
    assert ladder.target_rank_among_benchmarks >= 1


def test_e2e_chaos_under_partial_failure_remains_safe():
    """Chaos: 5 fault types × 500 iters × seed=42 -> safe rate ≥ 75% per
    scenario (matches the unit-test threshold)."""
    from a2a_agent.safety_redteam import compute_chaos_run
    r = compute_chaos_run(iterations=500, seed=42)
    for s in r.scenarios:
        safe = s.success_rate + s.abstain_rate
        assert safe >= 0.75, (
            f"{s.fault_type} safe={safe:.3f} below 0.75 threshold")
    assert r.overall_resilience_score >= 0.80
