"""Phase 17.Z - Stigma + reading-level linter tests."""

from __future__ import annotations

import pytest

from a2a_agent.stigma_linter import (
    PatientFacingTextAudit, StigmaLintReport, ReadabilityReport,
    assess_readability, audit_patient_facing_text, lint_stigma,
)


# ─────────────────────────────────────────────────────────────────────
# Readability
# ─────────────────────────────────────────────────────────────────────

def test_readability_simple_text_passes_grade_8():
    text = (
        "Take your pill once a day. Drink water with it. "
        "Call us if you feel sick. We are here to help you."
    )
    rep = assess_readability(text, target_grade_max=8)
    assert rep.meets_red_toolkit_target is True
    assert rep.flesch_kincaid_grade < 8


def test_readability_complex_text_fails_grade_8():
    text = (
        "Postoperative anticoagulation prophylaxis necessitates "
        "individualised pharmacokinetic considerations relative "
        "to renal clearance, hepatic metabolism, concurrent "
        "antiplatelet therapy, and the patient's underlying "
        "thrombophilia diathesis."
    )
    rep = assess_readability(text, target_grade_max=8)
    assert rep.meets_red_toolkit_target is False
    assert rep.flesch_kincaid_grade > 12


def test_readability_rejects_empty_text():
    with pytest.raises(ValueError):
        assess_readability("")


def test_readability_round_trip_through_pydantic():
    rep = assess_readability("This is a sentence. This is another.")
    payload = rep.model_dump(mode="json")
    rebuilt = ReadabilityReport.model_validate(payload)
    assert rebuilt.n_words == rep.n_words


def test_readability_counts_words_and_sentences():
    text = "Hello world. This is fine."
    rep = assess_readability(text)
    assert rep.n_sentences == 2
    assert rep.n_words == 5


# ─────────────────────────────────────────────────────────────────────
# Stigma flagger
# ─────────────────────────────────────────────────────────────────────

def test_stigma_clean_text():
    rep = lint_stigma(
        "The patient with diabetes will continue lisinopril."
    )
    assert rep.overall_grade == "clean"
    assert rep.n_findings == 0


def test_stigma_flags_diabetic_patient():
    rep = lint_stigma("The diabetic patient was discharged home.")
    assert rep.n_findings >= 1
    assert any(
        "patient with diabetes" in f.suggested_rewrite
        for f in rep.findings
    )


def test_stigma_high_severity_addict_blocks():
    rep = lint_stigma("The addict refused detox.")
    assert rep.overall_grade == "block"
    assert rep.n_high >= 1


def test_stigma_committed_suicide_blocks():
    rep = lint_stigma(
        "The patient committed suicide last December."
    )
    assert rep.overall_grade == "block"
    assert any(
        "died by suicide" in f.suggested_rewrite for f in rep.findings
    )


def test_stigma_non_compliant_warns():
    rep = lint_stigma(
        "The patient was non-compliant with the medication regimen."
    )
    assert rep.overall_grade == "warn"
    assert rep.n_medium >= 1


def test_stigma_clinical_diabetic_retinopathy_not_flagged():
    """'diabetic' should not flag when followed by clinical "
    "descriptors like 'retinopathy'."""
    rep = lint_stigma(
        "Proliferative diabetic retinopathy with macular edema."
    )
    flagged_diabetic = [
        f for f in rep.findings
        if "diabetes" in f.suggested_rewrite
    ]
    assert flagged_diabetic == []


def test_stigma_round_trip_through_pydantic():
    rep = lint_stigma("The addict refused detox.")
    payload = rep.model_dump(mode="json")
    rebuilt = StigmaLintReport.model_validate(payload)
    assert rebuilt.n_findings == rep.n_findings


def test_stigma_findings_carry_offset_and_citation():
    rep = lint_stigma(
        "The non-compliant patient was a frequent flyer in the ED."
    )
    for f in rep.findings:
        assert f.char_offset >= 0
        assert f.citation.strip() != ""


# ─────────────────────────────────────────────────────────────────────
# Combined audit
# ─────────────────────────────────────────────────────────────────────

def test_combined_audit_passes_clean_simple_text():
    audit = audit_patient_facing_text(
        "Take your pill at noon. Drink water. Call us if you feel ill."
    )
    assert audit.overall_pass is True


def test_combined_audit_fails_on_high_stigma():
    audit = audit_patient_facing_text(
        "The addict was discharged home today."
    )
    assert audit.overall_pass is False
    assert audit.stigma.overall_grade == "block"


def test_combined_audit_fails_on_high_grade():
    text = (
        "Postoperative anticoagulation prophylaxis necessitates "
        "individualised pharmacokinetic considerations relative "
        "to renal clearance and hepatic metabolism."
    )
    audit = audit_patient_facing_text(text, target_grade_max=8)
    assert audit.overall_pass is False
    assert audit.readability.meets_red_toolkit_target is False


def test_combined_audit_round_trip_through_pydantic():
    audit = audit_patient_facing_text(
        "Take your pill at noon every day."
    )
    payload = audit.model_dump(mode="json")
    rebuilt = PatientFacingTextAudit.model_validate(payload)
    assert rebuilt.overall_pass == audit.overall_pass
