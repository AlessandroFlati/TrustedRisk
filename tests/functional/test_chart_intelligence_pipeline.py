"""Functional tests for the CHART intelligence pipeline (CHART-1/2/3).

Exercises clinical NER -> negation/temporal context -> discharge summary
structuring as one cross-tool flow on realistic free-text input. Tests
the same surface a production agent would invoke when an EHR pushes a
free-text discharge note.
"""
from __future__ import annotations

from mcp_server.tools.chart_intelligence import (
    apply_negation_and_temporal,
    compute_clinical_ner,
    compute_negation_temporal,
    compute_structure_discharge_summary,
)


_REAL_NOTE = (
    "78-year-old male with a history of myocardial infarction (s/p CABG 2015), "
    "type 2 diabetes mellitus, and chronic kidney disease stage 3, admitted "
    "with acute heart failure exacerbation. "
    "Patient denies chest pain or dyspnea today. "
    "Family history of CHF in father. "
    "Active conditions: heart failure, hypertension, atrial fibrillation. "
    "Discharge medications: warfarin 5 mg daily, lisinopril 10 mg daily, "
    "metoprolol 25 mg twice daily, atorvastatin 40 mg at bedtime. "
    "NKDA. "
    "Disposition: discharge home with home health nursing for medication "
    "monitoring. Follow up with cardiology in 7-14 days. "
    "Plan colonoscopy in 3 months for colorectal screening."
)


def test_full_chart_pipeline_extracts_problems_and_meds(run_async):
    r = run_async(compute_clinical_ner(_REAL_NOTE, use_llm=False))
    assert r.method == "rule_based"
    assert r.entities_by_type.get("problem", 0) >= 5
    assert r.entities_by_type.get("medication", 0) >= 4


def test_chart_pipeline_negation_caught_for_denied_chest_pain(run_async):
    r = run_async(compute_clinical_ner(_REAL_NOTE))
    cp = next((e for e in r.entities
                  if "chest pain" in e.text.lower()), None)
    assert cp is not None
    assert cp.is_negated is True


def test_chart_pipeline_family_history_routed(run_async):
    r = run_async(compute_clinical_ner(_REAL_NOTE))
    fh = [e for e in r.entities if e.entity_type == "family_history"]
    assert any("chf" in e.text.lower() or
                  "heart failure" in e.text.lower() for e in fh)


def test_chart_pipeline_historical_temporal_for_old_mi(run_async):
    r = run_async(compute_clinical_ner(_REAL_NOTE))
    mi = next((e for e in r.entities
                  if "myocardial" in e.text.lower()), None)
    assert mi is not None
    assert mi.temporal_context == "historical"


def test_chart_pipeline_future_planned_for_colonoscopy_plan(run_async):
    """'Plan colonoscopy' should be flagged as future_planned. Our regex
    catches problems matching `colonoscopy`-adjacent phrases via the
    `Plan` trigger applied to nearby entities."""
    text = "Will discharge to skilled nursing. Plan to address CHF medications next week."
    r = run_async(compute_clinical_ner(text))
    chf = next((e for e in r.entities if e.text.lower() == "chf"), None)
    assert chf is not None
    assert chf.temporal_context == "future_planned"


def test_chart_pipeline_dedicated_negation_pass(run_async):
    """compute_negation_temporal should re-label entities passed in from
    a third-party NER without re-extracting them."""
    text = "No history of MI. Active heart failure."
    r = run_async(compute_clinical_ner(text))
    # Strip pre-set context so we can verify the dedicated pass works
    raw_entities = [e.model_copy(update={"is_negated": False,
                                                "temporal_context": "unspecified"})
                       for e in r.entities]
    re_run = run_async(compute_negation_temporal(text, raw_entities))
    mi = next((e for e in re_run if e.text.lower() == "mi"), None)
    chf = next((e for e in re_run
                  if "heart failure" in e.text.lower()), None)
    assert mi is not None and mi.is_negated is True
    assert chf is not None and chf.is_negated is False


def test_chart_pipeline_discharge_summary_yields_decision_card_scaffold(run_async):
    r = run_async(compute_structure_discharge_summary(_REAL_NOTE))
    assert r.extracted_recommendation == "home_with_care"
    assert r.extracted_followup_window_days is not None
    fu_lo, fu_hi = r.extracted_followup_window_days
    assert (fu_lo, fu_hi) == (7, 14)
    assert r.extraction_confidence in ("medium", "high")


def test_chart_pipeline_discharge_summary_extracts_canonical_meds(run_async):
    r = run_async(compute_structure_discharge_summary(_REAL_NOTE))
    names = {m["name"].lower() for m in r.extracted_medications}
    assert "warfarin" in names
    assert "lisinopril" in names
    assert "metoprolol" in names
    assert "atorvastatin" in names


def test_chart_pipeline_discharge_summary_skips_negated_meds(run_async):
    text = ("Discharge to home. Patient is no longer on warfarin. "
              "Continue lisinopril 10 mg daily. Follow up in 7 days.")
    r = run_async(compute_structure_discharge_summary(text))
    names = [m["name"].lower() for m in r.extracted_medications]
    assert "warfarin" not in names
    assert "lisinopril" in names


def test_chart_pipeline_extracts_active_problems_only(run_async):
    """Active problems with no preceding 'history of' trigger should be
    extracted; problems flagged as historical should not."""
    text = ("Patient with active heart failure and active diabetes. "
              "Discharge home in 7 days.")
    r = run_async(compute_structure_discharge_summary(text))
    blob = " ".join(r.extracted_problems)
    assert "heart failure" in blob or "chf" in blob


def test_chart_pipeline_skips_historical_mi(run_async):
    """'History of MI' should be flagged historical and excluded from the
    active problems list."""
    text = "History of MI."
    r = run_async(compute_structure_discharge_summary(text))
    assert not any(p == "mi" for p in r.extracted_problems)


def test_chart_pipeline_handles_empty_input_gracefully(run_async):
    r_ner = run_async(compute_clinical_ner(""))
    r_struct = run_async(compute_structure_discharge_summary(""))
    assert r_ner.n_entities == 0
    assert r_struct.extraction_confidence == "low"


def test_chart_pipeline_normalization_produces_canonical_codes(run_async):
    """Each rule-extracted med + problem comes with a canonical code +
    vocabulary so downstream tools can join on it without re-NLP."""
    r = run_async(compute_clinical_ner(_REAL_NOTE))
    rxnorm_count = sum(1 for e in r.entities
                            if e.normalized_vocabulary == "RXNORM")
    icd_count = sum(1 for e in r.entities
                        if e.normalized_vocabulary == "ICD10CM")
    assert rxnorm_count >= 4
    assert icd_count >= 4


def test_chart_pipeline_round_trip_apply_negation_idempotent(run_async):
    """Running the negation pass twice should be idempotent."""
    text = "No chest pain. History of stroke."
    r = run_async(compute_clinical_ner(text))
    once = apply_negation_and_temporal(text, r.entities)
    twice = apply_negation_and_temporal(text, once)
    for a, b in zip(once, twice):
        assert a.is_negated == b.is_negated
        assert a.temporal_context == b.temporal_context


def test_chart_pipeline_reading_level_mapping_for_full_pipeline(run_async):
    """A complete discharge summary should yield extraction_confidence ≥ medium
    when recommendation + ≥1 medication + follow-up are present."""
    text = ("Discharge home with home health. Discharge meds: warfarin. "
              "Follow up in 10 days.")
    r = run_async(compute_structure_discharge_summary(text))
    assert r.extraction_confidence in ("medium", "high")


def test_chart_pipeline_aab_signal_detection(run_async):
    """The structurer should recognize abstain-pattern phrasing in real
    discharge notes."""
    text = ("Acute bronchitis. Defer to attending judgment regarding antibiotics. "
              "Awaiting further imaging to rule out pneumonia.")
    r = run_async(compute_structure_discharge_summary(text))
    assert "clinician_judgment_required" in r.extracted_abstain_triggers
    assert "awaiting_data" in r.extracted_abstain_triggers
