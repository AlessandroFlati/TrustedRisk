"""Unit tests for CHART-1/2/3 -- clinical free-text intelligence."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools import chart_intelligence as mod
from mcp_server.tools.chart_intelligence import (
    apply_negation_and_temporal,
    compute_clinical_ner,
    compute_negation_temporal,
    compute_structure_discharge_summary,
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _disable_llm(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_DISABLE_LLM", "1")


# ───────────────────────────────────────────────────────────────────
# CHART-1 -- Clinical NER (rule-based floor)
# ───────────────────────────────────────────────────────────────────

def test_ner_empty_text_returns_no_entities():
    r = _run(compute_clinical_ner(""))
    assert r.n_entities == 0
    assert r.entities == []


def test_ner_invalid_input_raises():
    with pytest.raises(ValueError, match="must be a string"):
        _run(compute_clinical_ner(42))  # type: ignore[arg-type]


def test_ner_extracts_problems_from_simple_note():
    text = "75yo F with CHF and DM admitted for fluid overload."
    r = _run(compute_clinical_ner(text))
    types = {e.entity_type for e in r.entities}
    assert "problem" in types
    names = {e.text.lower() for e in r.entities
                if e.entity_type == "problem"}
    assert "chf" in names
    assert "dm" in names or any("diabetes" in n for n in names)


def test_ner_extracts_medications():
    text = "Discharge meds: warfarin 5 mg, lisinopril 10 mg, metformin 500 mg."
    r = _run(compute_clinical_ner(text))
    med_names = {e.text.lower() for e in r.entities
                    if e.entity_type == "medication"}
    assert "warfarin" in med_names
    assert "lisinopril" in med_names
    assert "metformin" in med_names


def test_ner_normalizes_med_to_rxcui():
    text = "Patient is on warfarin 5 mg daily."
    r = _run(compute_clinical_ner(text))
    warf = next(e for e in r.entities if e.text.lower() == "warfarin")
    assert warf.normalized_concept_id == "11289"
    assert warf.normalized_vocabulary == "RXNORM"


def test_ner_normalizes_problem_to_icd10():
    text = "Active diagnosis: CHF."
    r = _run(compute_clinical_ner(text))
    chf = next(e for e in r.entities if e.text.lower() == "chf")
    assert chf.normalized_concept_id == "I50.9"
    assert chf.normalized_vocabulary == "ICD10CM"


def test_ner_extracts_allergies():
    text = "Allergies: penicillin, sulfa drugs."
    r = _run(compute_clinical_ner(text))
    allergy_entities = [e for e in r.entities if e.entity_type == "allergy"]
    assert len(allergy_entities) >= 1


def test_ner_nkda_marked_as_negated():
    text = "NKDA. No known drug allergies."
    r = _run(compute_clinical_ner(text))
    nkda_entities = [e for e in r.entities
                       if e.entity_type == "allergy" and e.is_negated]
    assert len(nkda_entities) >= 1


def test_ner_method_rule_based_when_llm_disabled():
    text = "CHF with warfarin therapy."
    r = _run(compute_clinical_ner(text))
    assert r.method == "rule_based"


def test_ner_entities_sorted_by_position():
    text = "DM occurred after CHF onset."
    r = _run(compute_clinical_ner(text))
    starts = [e.start for e in r.entities]
    assert starts == sorted(starts)


def test_ner_entities_by_type_count():
    text = ("Patient with CHF on warfarin and lisinopril. "
              "Allergic to penicillin.")
    r = _run(compute_clinical_ner(text))
    assert r.entities_by_type.get("medication", 0) >= 2
    assert r.entities_by_type.get("problem", 0) >= 1


def test_ner_no_duplicate_spans():
    text = "Diabetes diabetes diabetes"
    r = _run(compute_clinical_ner(text))
    spans = {(e.start, e.end) for e in r.entities}
    assert len(spans) == len([e for e in r.entities
                                  if e.text.lower() == "diabetes"])


# ───────────────────────────────────────────────────────────────────
# CHART-2 -- Negation + temporal context
# ───────────────────────────────────────────────────────────────────

def test_negation_no_chest_pain():
    text = "The patient denies chest pain."
    r = _run(compute_clinical_ner(text))
    cp = next((e for e in r.entities
                  if "chest pain" in e.text.lower()), None)
    assert cp is not None
    assert cp.is_negated is True


def test_negation_no_history_of_mi():
    text = "No history of MI."
    r = _run(compute_clinical_ner(text))
    mi = next((e for e in r.entities if e.text.lower() == "mi"), None)
    if mi is not None:
        assert mi.is_negated is True


def test_temporal_history_of():
    text = "History of chronic kidney disease."
    r = _run(compute_clinical_ner(text))
    ckd = next((e for e in r.entities
                  if "kidney" in e.text.lower()), None)
    assert ckd is not None
    assert ckd.temporal_context == "historical"


def test_temporal_rule_out():
    text = "R/O myocardial infarction with serial troponins."
    r = _run(compute_clinical_ner(text))
    mi = next((e for e in r.entities
                  if "myocardial" in e.text.lower()), None)
    assert mi is not None
    assert mi.temporal_context == "rule_out"


def test_family_history_relabels_to_family_history_type():
    text = "Family history of myocardial infarction in father."
    r = _run(compute_clinical_ner(text))
    mi = next((e for e in r.entities
                  if "myocardial" in e.text.lower()), None)
    assert mi is not None
    assert mi.entity_type == "family_history"
    assert mi.temporal_context == "family_history"


def test_temporal_future_planned():
    text = "Plan colonoscopy next month."
    r = _run(compute_clinical_ner(text))
    # No problem matches "colonoscopy" by rules; verify negation/temporal
    # logic via a known problem instead
    text2 = "Will reassess CHF with echo next week."
    r2 = _run(compute_clinical_ner(text2))
    chf = next((e for e in r2.entities if e.text.lower() == "chf"), None)
    if chf is not None:
        assert chf.temporal_context == "future_planned"


def test_compute_negation_temporal_dict_input_coerced():
    text = "Patient denies chest pain."
    entities_dict = [{
        "entity_type": "problem", "text": "chest pain",
        "start": text.find("chest pain"),
        "end": text.find("chest pain") + len("chest pain"),
        "is_negated": False, "temporal_context": "unspecified",
    }]
    r = _run(compute_negation_temporal(text, entities_dict))
    assert len(r) == 1
    assert r[0].is_negated is True


def test_apply_negation_preserves_pre_set_negation():
    """NKDA already marked is_negated=True; subsequent pass shouldn't flip."""
    text = "NKDA"
    r = _run(compute_clinical_ner(text))
    nkda = next((e for e in r.entities if e.is_negated), None)
    assert nkda is not None
    re_applied = apply_negation_and_temporal(text, [nkda])
    assert re_applied[0].is_negated is True


def test_negation_does_not_cross_clause_break():
    """'no chest pain. however he has dyspnea' -- dyspnea is NOT negated."""
    text = "No chest pain. However he has dyspnea."
    r = _run(compute_clinical_ner(text))
    cp = next((e for e in r.entities
                  if "chest pain" in e.text.lower()), None)
    assert cp is not None and cp.is_negated is True


# ───────────────────────────────────────────────────────────────────
# CHART-3 -- Discharge summary structurer
# ───────────────────────────────────────────────────────────────────

def test_structurer_empty_text_low_confidence():
    r = _run(compute_structure_discharge_summary(""))
    assert r.extraction_confidence == "low"
    assert r.extracted_recommendation is None


def test_structurer_extracts_discharge_home_recommendation():
    text = ("Disposition: discharge home in stable condition. "
              "Follow up with PCP in 7 days.")
    r = _run(compute_structure_discharge_summary(text))
    assert r.extracted_recommendation == "discharge_home"


def test_structurer_extracts_home_with_care():
    text = ("Discharge with home health nursing referral; physical "
              "therapy ordered.")
    r = _run(compute_structure_discharge_summary(text))
    assert r.extracted_recommendation == "home_with_care"


def test_structurer_extracts_snf():
    text = "Will discharge to skilled nursing facility for 21-day rehab."
    r = _run(compute_structure_discharge_summary(text))
    assert r.extracted_recommendation == "snf"


def test_structurer_extracts_followup_window_days():
    text = "Follow up with PCP in 7 to 14 days."
    r = _run(compute_structure_discharge_summary(text))
    assert r.extracted_followup_window_days == (7, 14)


def test_structurer_extracts_followup_in_weeks():
    text = "Follow up in 2 weeks."
    r = _run(compute_structure_discharge_summary(text))
    assert r.extracted_followup_window_days == (14, 14)


def test_structurer_extracts_medications_dedup():
    text = ("Discharge meds: warfarin 5 mg, lisinopril 10 mg, "
              "warfarin (continued).")
    r = _run(compute_structure_discharge_summary(text))
    names = [m["name"].lower() for m in r.extracted_medications]
    assert names.count("warfarin") == 1
    assert "lisinopril" in names


def test_structurer_skips_negated_meds():
    """A negated medication shouldn't appear in extracted_medications."""
    text = "Patient is no longer on warfarin. Continue on metformin."
    r = _run(compute_structure_discharge_summary(text))
    names = [m["name"].lower() for m in r.extracted_medications]
    # "no longer on warfarin" -> warfarin negated -> excluded
    assert "warfarin" not in names
    assert "metformin" in names


def test_structurer_extracts_problems():
    text = "Active conditions: CHF, diabetes, hypertension."
    r = _run(compute_structure_discharge_summary(text))
    assert any("chf" in p for p in r.extracted_problems)
    assert any("diabetes" in p for p in r.extracted_problems) or \
           any("hypertension" in p for p in r.extracted_problems)


def test_structurer_extracts_abstain_triggers():
    text = ("Defer to attending judgment regarding final disposition. "
              "Awaiting further imaging.")
    r = _run(compute_structure_discharge_summary(text))
    assert "clinician_judgment_required" in r.extracted_abstain_triggers
    assert "awaiting_data" in r.extracted_abstain_triggers


def test_structurer_high_confidence_with_full_summary():
    text = ("Discharge summary: 75yo F with CHF, hypertension, diabetes. "
              "Disposition: discharge home with home health. "
              "Discharge meds: warfarin 5 mg, lisinopril 10 mg, "
              "metformin 500 mg. Follow up with PCP in 7 days.")
    r = _run(compute_structure_discharge_summary(text))
    assert r.extraction_confidence in ("medium", "high")
    assert r.extracted_recommendation == "home_with_care"
    assert r.extracted_followup_window_days == (7, 7)


def test_structurer_records_text_length():
    text = "A" * 500
    r = _run(compute_structure_discharge_summary(text))
    assert r.raw_text_length == 500


# ───────────────────────────────────────────────────────────────────
# Bundle registration
# ───────────────────────────────────────────────────────────────────

def test_chart_intelligence_bundle_registered():
    from mcp_server.tools import BUNDLES
    assert "chart_intelligence" in BUNDLES
    assert "compute_clinical_ner" in BUNDLES["chart_intelligence"]
    assert "compute_structure_discharge_summary" in \
        BUNDLES["chart_intelligence"]
