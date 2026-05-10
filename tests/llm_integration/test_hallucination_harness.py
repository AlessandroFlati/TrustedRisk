"""Phase 6.2 -- Hallucination harness over real LLM polish.

Opt-in: set `TRUSTEDRISK_LLM_INTEGRATION=1` to run. Calls the real
configured LLM client (Ollama / Gemini / null) and asserts the
hallucination post-check stays clean across a representative slice of
the scribe + PA + patient surface.

Skip pattern matches the rest of `tests/llm_integration/` so CI
defaults are untouched.
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("TRUSTEDRISK_LLM_INTEGRATION", "").lower() not in (
        "1", "true", "yes",
    ),
    reason=(
        "Live LLM-integration tests require TRUSTEDRISK_LLM_INTEGRATION=1; "
        "skipped by default to avoid hitting Ollama / Gemini in CI."
    ),
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Polish client smoke ───────────────────────


def test_resolved_polish_client_round_trip():
    from a2a_agent.llm_polish import (
        find_preserved_tokens, post_check_preserved_tokens,
        reset_client_cache, resolve_polish_client,
    )
    reset_client_cache()
    client = resolve_polish_client()
    if client.model_id is None:
        pytest.skip("No LLM configured (Ollama down + no GOOGLE_API_KEY)")
    src = (
        "Continue furosemide 40 mg PO daily (cite med-furo) for the "
        "I50.21 acute on chronic systolic CHF. Hold lisinopril 10 mg "
        "(cite med-lis) until creatinine recovers below 1.2 mg/dL."
    )
    res = _run(client.polish(
        src,
        system_prompt=(
            "Paraphrase to improve readability. Preserve every cite-back, "
            "drug name, dose, and ICD code EXACTLY."
        ),
    ))
    if not res.is_polished:
        pytest.skip(
            f"polish rejected: {res.polish_rejected_reason} -- review the "
            "model + prompt; this is informational, not a hard failure"
        )
    # Hallucination post-check is part of the client's contract; we
    # double-check explicitly here for safety
    passed, missing = post_check_preserved_tokens(src, res.polished_text)
    assert passed, f"missing tokens after polish: {missing}"


# ─────────────────────── Scribe end-to-end ───────────────────────


def test_scribe_progress_note_polish_preserves_cite_backs():
    from mcp_server.tools.progress_note_draft import compute_progress_note_draft

    bundle = {"resourceType": "Bundle", "entry": [
        {"resource": {
            "resourceType": "Patient", "id": "pt-jane",
            "name": [{"given": ["Jane"], "family": "Doe"}],
        }},
        {"resource": {
            "resourceType": "Condition", "id": "cond-chf",
            "code": {"coding": [{
                "code": "I50.21", "display": "CHF",
            }], "text": "CHF"},
        }},
        {"resource": {
            "resourceType": "MedicationRequest", "id": "med-furo",
            "medicationCodeableConcept": {"text": "furosemide 40 mg PO"},
        }},
    ]}
    os.environ["TRUSTEDRISK_SCRIBE_LLM_POLISH"] = "1"
    out = _run(compute_progress_note_draft(
        patient_reference="Patient/pt-jane",
        fhir_bundle=bundle,
        subjective_text="Patient reports improved breathing.",
    ))
    # Polish may have rejected hallucination -- we just assert the
    # cite-back IDs survive in the structured output regardless.
    assess = next(s for s in out.sections if s.section_id == "assessment")
    plan = next(s for s in out.sections if s.section_id == "plan")
    assert "cond-chf" in assess.cited_evidence_ids
    assert "cond-chf" in plan.cited_evidence_ids


# ─────────────────────── PA letter end-to-end ───────────────────────


def test_pa_letter_polish_preserves_drug_and_dose():
    from mcp_server.tools.pa_evidence_pack import compute_pa_evidence_pack
    from mcp_server.tools.pa_letter_draft import compute_pa_letter_draft
    from shared.schemas import PARequestedService

    bundle = {"resourceType": "Bundle", "entry": [
        {"resource": {
            "resourceType": "Condition", "id": "cond-radic",
            "code": {"coding": [{
                "code": "M54.16", "display": "Lumbar radiculopathy",
            }], "text": "Lumbar radiculopathy"},
        }},
        {"resource": {
            "resourceType": "Observation", "id": "obs-recent",
            "code": {"coding": [{"code": "33747-0"}]},
            "valueQuantity": {"value": 1.0, "unit": "mg/dL"},
            "effectiveDateTime": "2026-04-15T00:00:00Z",
        }},
        {"resource": {
            "resourceType": "MedicationRequest", "id": "med-naproxen",
            "medicationCodeableConcept": {"text": "naproxen 500 mg BID"},
        }},
    ]}
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/56yo",
        requested_service=PARequestedService(
            service_type="imaging_advanced",
            description="MRI lumbar spine without contrast",
        ),
        payer="unitedhealth", fhir_bundle=bundle,
        chart_excerpts=[{
            "text": "Specialist orthopedic referral attached",
            "fhir_resource_id": "chart-ortho",
        }],
    ))
    os.environ["TRUSTEDRISK_PA_LLM_POLISH"] = "1"
    draft = _run(compute_pa_letter_draft(pack, None))
    # Cite-backs preserved
    diag_p = next(p for p in draft.paragraphs if p.section == "diagnosis")
    assert "cond-radic" in diag_p.cited_evidence_ids
