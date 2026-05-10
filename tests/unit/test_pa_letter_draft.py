"""Unit tests for compute_pa_letter_draft (Phase 2.1 PA-4)."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.pa_evidence_pack import compute_pa_evidence_pack
from mcp_server.tools.pa_letter_draft import compute_pa_letter_draft
from mcp_server.tools.pa_payer_rules_match import compute_pa_payer_rules_match
from shared.schemas import PARequestedService


def _run(coro):
    return asyncio.run(coro)


def _bundle() -> dict:
    return {"resourceType": "Bundle", "type": "collection", "entry": [
        {"resource": {
            "resourceType": "Condition",
            "id": "cond-back-pain",
            "code": {
                "coding": [{
                    "code": "M54.5",
                    "system": "http://hl7.org/fhir/sid/icd-10-cm",
                    "display": "Low back pain",
                }],
                "text": "Low back pain",
            },
            "recordedDate": "2026-04-01",
        }},
        {"resource": {
            "resourceType": "Observation",
            "id": "obs-cmp",
            "code": {"coding": [{"code": "1742-6", "display": "ALT"}]},
            "valueQuantity": {"value": 35, "unit": "U/L"},
            "effectiveDateTime": "2026-04-15T00:00:00Z",
        }},
        {"resource": {
            "resourceType": "MedicationRequest",
            "id": "med-ibuprofen",
            "medicationCodeableConcept": {"text": "ibuprofen 600 mg"},
            "authoredOn": "2026-03-01",
        }},
        {"resource": {
            "resourceType": "Procedure",
            "id": "proc-xray",
            "code": {"text": "Plain-film X-ray lumbar spine"},
            "performedDateTime": "2026-02-15",
        }},
    ]}


async def _full_pipeline(payer: str):
    pack = await compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=PARequestedService(
            service_type="imaging_advanced",
            cpt_codes=["72148"],
            description="MRI lumbar spine without contrast",
        ),
        payer=payer,
        fhir_bundle=_bundle(),
        chart_excerpts=[
            {"text": "Patient seen by orthopedic specialist; "
                       "no improvement on conservative therapy",
             "fhir_resource_id": "chart-ortho-note"},
        ],
    )
    rules = await compute_pa_payer_rules_match(pack)
    draft = await compute_pa_letter_draft(pack, rules)
    return pack, rules, draft


# ─────────────────────── Behaviour ───────────────────────

def test_letter_draft_returns_paragraphs():
    _, _, draft = _run(_full_pipeline("unitedhealth"))
    assert draft.paragraphs
    sections = [p.section for p in draft.paragraphs]
    assert "header" in sections
    assert "diagnosis" in sections
    assert "medical_necessity" in sections
    assert "closing" in sections


def test_letter_draft_full_text_concatenates_paragraphs():
    _, _, draft = _run(_full_pipeline("unitedhealth"))
    for p in draft.paragraphs:
        assert p.text in draft.full_text


def test_letter_draft_carries_cite_backs():
    _, _, draft = _run(_full_pipeline("unitedhealth"))
    assert draft.n_cite_backs > 0
    # diagnosis paragraph cites the condition resource
    diag_p = next(p for p in draft.paragraphs if p.section == "diagnosis")
    assert "cond-back-pain" in diag_p.cited_evidence_ids


def test_letter_draft_coverage_pct_in_unit_range():
    _, _, draft = _run(_full_pipeline("unitedhealth"))
    assert 0.0 <= draft.coverage_pct <= 1.0


def test_letter_draft_fixed_payer_label_in_header():
    _, _, draft_uhc = _run(_full_pipeline("unitedhealth"))
    assert "UnitedHealthcare" in draft_uhc.full_text
    _, _, draft_aetna = _run(_full_pipeline("aetna"))
    assert "Aetna" in draft_aetna.full_text


def test_letter_draft_no_llm_polish_by_default(monkeypatch):
    """Without TRUSTEDRISK_PA_LLM_POLISH=1, the draft is purely
    deterministic."""
    monkeypatch.delenv("TRUSTEDRISK_PA_LLM_POLISH", raising=False)
    _, _, draft = _run(_full_pipeline("unitedhealth"))
    assert draft.contains_llm_polish is False
    assert draft.llm_model_id is None


def test_letter_draft_no_rules_match_still_works():
    """If rules_match is omitted, the medical_necessity + policy_match
    paragraphs are absent but the draft still carries the deterministic
    spine."""
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=PARequestedService(
            service_type="imaging_advanced",
            description="MRI lumbar",
        ),
        payer="unitedhealth",
        fhir_bundle=_bundle(),
        chart_excerpts=[
            {"text": "specialist consult",
             "fhir_resource_id": "chart-1"},
        ],
    ))
    draft = _run(compute_pa_letter_draft(pack, None))
    sections = [p.section for p in draft.paragraphs]
    assert "header" in sections
    assert "diagnosis" in sections
    assert "medical_necessity" not in sections
    assert "closing" in sections


def test_letter_draft_handles_dict_inputs():
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=PARequestedService(
            service_type="imaging_advanced",
            description="MRI",
        ),
        payer="unitedhealth", fhir_bundle=_bundle(),
        chart_excerpts=[{"text": "x", "fhir_resource_id": "c1"}],
    ))
    rules = _run(compute_pa_payer_rules_match(pack))
    draft = _run(compute_pa_letter_draft(
        pack.model_dump(), rules.model_dump(),
    ))
    assert draft.paragraphs


def test_letter_draft_lacks_invented_clinical_facts():
    """The deterministic floor only references evidence items present
    in the pack. Specifically: the diagnosis section cites only the
    Condition resources we put in."""
    _, _, draft = _run(_full_pipeline("unitedhealth"))
    diag_p = next(p for p in draft.paragraphs if p.section == "diagnosis")
    # No mentions of conditions we did NOT add
    assert "diabetes" not in diag_p.text.lower()
    assert "hypertension" not in diag_p.text.lower()
    # The one we DID add appears
    assert "low back pain" in diag_p.text.lower()


def test_letter_draft_carries_references():
    _, _, draft = _run(_full_pipeline("unitedhealth"))
    assert any("AMA" in r for r in draft.references)
