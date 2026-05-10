"""Unit tests for the deterministic refusal classifier.

These tests pin the contract for what the classifier accepts/rejects.
The full eval suite (tests/eval/eval_queries.json) is also runnable from
pytest via test_eval_suite.py -- this file isolates each rule for fine-grained
regression detection.
"""
from __future__ import annotations

import pytest

from a2a_agent.refusal_classifier import classify_query


# ─────────────────────── Decide path ───────────────────────

@pytest.mark.parametrize("text", [
    "Can patient X42 be discharged home safely?",
    "Review the discharge plan for patient Z789.",
    "What is the 30-day readmission risk for encounter A123?",
    "Should this CHF patient go home or to SNF?",
    "Adult (67yo) post-MI, ready for review.",
])
def test_decide_path(text):
    assert classify_query(text).category == "decide"


# ─────────────────────── Validate-only path ───────────────────────

@pytest.mark.parametrize("text", [
    "Scan this clinical note for PHI.",
    "Does this discharge summary contain any protected information?",
    "Verify the claim that patient is stable for discharge.",
    "Check this note for protected health information.",
])
def test_validate_only_path(text):
    assert classify_query(text).category == "validate_only"


# ─────────────────────── Pediatric refusal ───────────────────────

@pytest.mark.parametrize("text", [
    "Can this 12-year-old patient be discharged?",
    "9-year-old post-tonsillectomy.",
    "17-year-old with chest pain.",
    "Patient is 17 years and 11 months old.",
    "Pediatric patient ready for discharge.",
])
def test_pediatric_refusal(text):
    assert classify_query(text).category == "refuse_pediatric"


def test_pediatric_does_not_match_adult_age():
    # 18+ should NOT trigger pediatric
    assert classify_query("Patient is 18 years old.").category != "refuse_pediatric"
    assert classify_query("65-year-old patient.").category != "refuse_pediatric"


def test_pediatric_keyword_with_adult_context_no_match():
    # "pediatric surgeon" with explicit adult age -> must not refuse
    result = classify_query("Adult patient (52yo), pediatric surgeon by profession.")
    assert result.category == "decide"


def test_pediatric_keyword_alone_triggers():
    # Bare "pediatric patient" without adult-affirming context -> refuse
    assert classify_query("Pediatric patient ready for discharge.").category == "refuse_pediatric"


# ─────────────────────── Oncology refusal ───────────────────────

@pytest.mark.parametrize("text", [
    "Stage IV lung cancer patient on chemotherapy.",
    "Active metastatic breast cancer, finishing chemo.",
    "Adult with chronic lymphoma on maintenance therapy.",
    "Patient has carcinoma of the colon.",
    "Recent leukemia diagnosis, adult patient.",
])
def test_oncology_refusal(text):
    assert classify_query(text).category == "refuse_oncology"


# ─────────────────────── EOL refusal ───────────────────────

@pytest.mark.parametrize("text", [
    "Patient on hospice care, discharge plan?",
    "Comfort-care-only patient.",
    "End-of-life decisions for this patient.",
    "Palliative care discharge.",
    "Family meeting about goals of care tomorrow.",
])
def test_eol_refusal(text):
    assert classify_query(text).category == "refuse_eol"


def test_dnr_alone_does_not_trigger_eol():
    """DNR status alone is not EOL -- patient may still want active treatment."""
    assert classify_query("Patient is 70 with DNR status, considering rehab placement.").category != "refuse_eol"


# ─────────────────────── Emergency refusal ───────────────────────

@pytest.mark.parametrize("text", [
    "Patient is coding right now, what's the readmission risk?",
    "STAT -- emergency disposition decision needed.",
    "Need urgent disposition guidance.",
    "Code blue patient, future risk?",
    "Patient is dying, need decision in 5 minutes.",
])
def test_emergency_refusal(text):
    cat = classify_query(text).category
    # Note: "patient is dying" may also match EOL -- both refusals are correct
    assert cat in ("refuse_emergency", "refuse_eol")


# ─────────────────────── Non-clinical refusal ───────────────────────

@pytest.mark.parametrize("text", [
    "What would this patient's premium be if classified high-risk?",
    "Should we deny insurance coverage for this admission?",
    "Use this risk score in the upcoming malpractice case.",
    "Is this employee fit to return to work?",
    "Help me prepare a coverage denial letter.",
])
def test_non_clinical_refusal(text):
    assert classify_query(text).category == "refuse_non_clinical"


def test_clinical_decision_with_insurance_framing():
    """Mixed signal -- clinical risk request framed for insurance use should refuse."""
    text = "What's the readmission risk for patient X42 to support our insurance coverage appeal?"
    assert classify_query(text).category == "refuse_non_clinical"


# ─────────────────────── Edge cases ───────────────────────

def test_empty_input_default_decide():
    assert classify_query("").category == "decide"


def test_whitespace_only_default_decide():
    assert classify_query("   \n   ").category == "decide"


def test_non_string_raises():
    with pytest.raises(TypeError):
        classify_query(12345)  # type: ignore[arg-type]


def test_no_keywords_defaults_to_decide():
    assert classify_query("Hello world.").category == "decide"


# ─────────────────────── Result fields ───────────────────────

def test_classification_result_carries_pattern_and_rationale():
    result = classify_query("Patient is 8 years old.")
    assert result.category == "refuse_pediatric"
    assert result.matched_pattern  # truthy
    assert result.rationale  # truthy
    assert "8" in result.rationale or "below" in result.rationale.lower()


def test_decide_path_has_rationale():
    result = classify_query("Adult (67) discharge review.")
    assert result.category == "decide"
    assert result.rationale  # truthy even when matched_pattern may be None
