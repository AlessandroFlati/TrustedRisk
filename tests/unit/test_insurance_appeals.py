"""Phase 10.3 -- insurance appeals specialist tests (APPEALS-1/2/3)."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.insurance_appeals import (
    compute_appeal_escalation_path,
    compute_appeal_letter_draft,
    compute_denial_letter_parse,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# APPEALS-1 -- compute_denial_letter_parse
# ─────────────────────────────────────────────────────────────────────

_UHC_LETTER = """
Dear Provider,

This letter is dated 2026-04-15 from UnitedHealthcare regarding
claim id: ABC-12345-X.

We have reviewed your claim and the requested service is not
medically necessary based on our coverage policy. The contested
amount is $12,450.00.

Appeal deadline: 2026-10-12. Please submit any supporting evidence.
""".strip()

_AETNA_STEP_THERAPY = """
This denial is from Aetna. Claim # AET-554433.
The requested medication has been denied because step therapy
requirements have not been met. Patient must trial a preferred
alternative first. Date: 2026-04-20. Appeal deadline: 2026-10-17.
""".strip()


def test_parse_detects_united_payer():
    out = _run(compute_denial_letter_parse(_UHC_LETTER))
    assert out.payer == "UnitedHealthcare"


def test_parse_extracts_claim_id():
    out = _run(compute_denial_letter_parse(_UHC_LETTER))
    assert out.claim_id == "ABC-12345-X"


def test_parse_extracts_appeal_deadline():
    out = _run(compute_denial_letter_parse(_UHC_LETTER))
    assert out.appeal_deadline_iso == "2026-10-12"


def test_parse_extracts_dollar_amount():
    out = _run(compute_denial_letter_parse(_UHC_LETTER))
    assert out.contested_dollar_amount == 12450.0


def test_parse_categorises_medical_necessity():
    out = _run(compute_denial_letter_parse(_UHC_LETTER))
    cats = {r.category for r in out.reasons}
    assert "medical_necessity" in cats


def test_parse_categorises_step_therapy():
    out = _run(compute_denial_letter_parse(_AETNA_STEP_THERAPY))
    cats = {r.category for r in out.reasons}
    assert "step_therapy_not_met" in cats


def test_parse_unknown_payer_yields_unknown():
    out = _run(compute_denial_letter_parse(
        "Generic denial -- service not medically necessary.",
    ))
    assert out.payer == "Unknown"


def test_parse_no_pattern_match_yields_other_reason():
    out = _run(compute_denial_letter_parse("Generic header text."))
    assert out.n_reasons == 1
    assert out.reasons[0].category == "other"


def test_parse_partial_denial_flag():
    text = (
        "Aetna partially denied this claim. The procedure code 12345 "
        "is not medically necessary."
    )
    out = _run(compute_denial_letter_parse(text))
    assert out.is_partial_denial is True


def test_parse_expected_payer_overrides_unknown():
    out = _run(compute_denial_letter_parse(
        "Generic denial -- service not medically necessary.",
        expected_payer="Cigna",
    ))
    assert out.payer == "Cigna"


# ─────────────────────────────────────────────────────────────────────
# APPEALS-2 -- compute_appeal_letter_draft
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def parsed_uhc_denial():
    return _run(compute_denial_letter_parse(_UHC_LETTER))


def test_draft_contains_required_sections(parsed_uhc_denial):
    out = _run(compute_appeal_letter_draft(
        parsed_uhc_denial,
        patient_summary="65y M with stage IV NSCLC, EGFR exon 19 deletion.",
        medical_necessity_argument="Targeted therapy is the standard of care.",
    ))
    sections = {p.section for p in out.paragraphs}
    assert {"header", "patient_summary", "denial_summary",
            "medical_necessity_argument", "evidence_supporting",
            "policy_counter_argument", "closing"} <= sections


def test_draft_full_text_concatenates_paragraphs(parsed_uhc_denial):
    out = _run(compute_appeal_letter_draft(
        parsed_uhc_denial,
        patient_summary="A.",
        medical_necessity_argument="B.",
    ))
    for p in out.paragraphs:
        assert p.text in out.full_text


def test_draft_cite_back_count_propagates(parsed_uhc_denial):
    out = _run(compute_appeal_letter_draft(
        parsed_uhc_denial,
        patient_summary="A.", medical_necessity_argument="B.",
        cited_evidence_ids=["fhir:Condition/c-1", "PMID:12345678"],
    ))
    assert out.n_cite_backs >= 2


def test_draft_alternative_section_only_when_provided(parsed_uhc_denial):
    no_alt = _run(compute_appeal_letter_draft(
        parsed_uhc_denial,
        patient_summary="A.", medical_necessity_argument="B.",
    ))
    with_alt = _run(compute_appeal_letter_draft(
        parsed_uhc_denial,
        patient_summary="A.", medical_necessity_argument="B.",
        alternative_proposed="Alternative: pembrolizumab.",
    ))
    assert (any(p.section == "alternative_proposed" for p in with_alt.paragraphs)
            and not any(p.section == "alternative_proposed" for p in no_alt.paragraphs))


def test_draft_llm_polish_flag(parsed_uhc_denial):
    out = _run(compute_appeal_letter_draft(
        parsed_uhc_denial,
        patient_summary="A.", medical_necessity_argument="B.",
        enable_llm_polish=True, llm_model_id="gpt-4o-2024",
    ))
    assert out.contains_llm_polish is True
    assert out.llm_model_id == "gpt-4o-2024"


def test_draft_accepts_dict_input(parsed_uhc_denial):
    out = _run(compute_appeal_letter_draft(
        parsed_uhc_denial.model_dump(),
        patient_summary="A.", medical_necessity_argument="B.",
    ))
    assert out.payer == parsed_uhc_denial.payer


def test_draft_appeal_level_chosen_propagates(parsed_uhc_denial):
    out = _run(compute_appeal_letter_draft(
        parsed_uhc_denial,
        patient_summary="A.", medical_necessity_argument="B.",
        appeal_level="external_independent_review",
    ))
    assert out.appeal_level == "external_independent_review"
    # Header text references ACA external review
    header = next(p for p in out.paragraphs if p.section == "header")
    assert "ACA" in header.text or "external" in header.text.lower()


# ─────────────────────────────────────────────────────────────────────
# APPEALS-3 -- compute_appeal_escalation_path
# ─────────────────────────────────────────────────────────────────────

def test_escalation_starts_at_first_level_by_default():
    out = _run(compute_appeal_escalation_path(payer="Aetna"))
    assert out.steps[0].level == "internal_first_level"


def test_escalation_truncates_to_starting_level():
    out = _run(compute_appeal_escalation_path(
        payer="Aetna", starting_level="external_independent_review",
    ))
    levels = [s.level for s in out.steps]
    assert "internal_first_level" not in levels
    assert "internal_second_level" not in levels
    assert levels[0] == "external_independent_review"


def test_escalation_cumulative_days_positive():
    out = _run(compute_appeal_escalation_path(payer="Aetna"))
    assert out.cumulative_max_days > 0


def test_escalation_unknown_starting_level_rejected():
    with pytest.raises(ValueError):
        _run(compute_appeal_escalation_path(
            payer="Aetna", starting_level="bogus_level",
        ))


def test_escalation_external_review_success_around_40pct():
    """KFF prior -- IRO overturns ~ 40 % of denials."""
    out = _run(compute_appeal_escalation_path(payer="Aetna"))
    iro = next(s for s in out.steps
                  if s.level == "external_independent_review")
    assert 0.30 <= iro.estimated_success_probability <= 0.50


# ─────────────────────────────────────────────────────────────────────
# Bundle wiring
# ─────────────────────────────────────────────────────────────────────

def test_insurance_appeals_bundle_registered():
    from mcp_server.tools import BUNDLES
    assert "insurance_appeals" in BUNDLES
    assert set(BUNDLES["insurance_appeals"]) == {
        "compute_denial_letter_parse",
        "compute_appeal_letter_draft",
        "compute_appeal_escalation_path",
    }


def test_insurance_appeals_scopes_declared():
    from mcp_server.scopes import BUNDLE_SCOPES
    assert "insurance_appeals" in BUNDLE_SCOPES
    assert "patient/Coverage.rs" in BUNDLE_SCOPES["insurance_appeals"]
