"""Unit tests for compute_pa_payer_rules_match (Phase 2.1 PA-2)."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.pa_evidence_pack import compute_pa_evidence_pack
from mcp_server.tools.pa_payer_rules_match import compute_pa_payer_rules_match
from shared.schemas import PARequestedService


def _run(coro):
    return asyncio.run(coro)


def _bundle_with(*, with_meds: int = 0, with_xray: bool = False,
                    with_obs: bool = True) -> dict:
    entries: list[dict] = [
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
    ]
    if with_obs:
        entries.append({"resource": {
            "resourceType": "Observation",
            "id": "obs-cmp",
            "code": {"coding": [{"code": "1742-6", "display": "ALT"}]},
            "valueQuantity": {"value": 35, "unit": "U/L"},
            "effectiveDateTime": "2026-04-15T00:00:00Z",
        }})
    for i in range(with_meds):
        entries.append({"resource": {
            "resourceType": "MedicationRequest",
            "id": f"med-{i}",
            "medicationCodeableConcept": {"text": f"med-{i}"},
            "authoredOn": "2026-03-01",
        }})
    if with_xray:
        entries.append({"resource": {
            "resourceType": "Procedure",
            "id": "proc-xray",
            "code": {"text": "Plain-film X-ray lumbar spine"},
            "performedDateTime": "2026-02-15",
        }})
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def _imaging_pack(payer: str, **kw):
    return _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=PARequestedService(
            service_type="imaging_advanced",
            cpt_codes=["72148"],
            description="MRI lumbar spine",
        ),
        payer=payer,
        fhir_bundle=_bundle_with(**kw),
    ))


# ─────────────────────── Behaviour ───────────────────────

def test_uhc_imaging_xray_first_rule_met_when_xray_present():
    pack = _imaging_pack("unitedhealth", with_xray=True)
    out = _run(compute_pa_payer_rules_match(pack))
    rule = next(r for r in out.rules
                    if r.rule_id == "uhc.img.exhaust_xray")
    assert rule.status == "met"
    assert rule.evidence_ids == ["proc-xray"]


def test_uhc_imaging_xray_first_unmet_without_xray():
    pack = _imaging_pack("unitedhealth", with_xray=False)
    out = _run(compute_pa_payer_rules_match(pack))
    rule = next(r for r in out.rules
                    if r.rule_id == "uhc.img.exhaust_xray")
    assert rule.status == "unmet"
    assert "X-ray" in (rule.gap_text or "")


def test_alignment_tier_changes_with_more_met_rules():
    pack_no_xray = _imaging_pack("unitedhealth", with_xray=False)
    pack_xray = _imaging_pack("unitedhealth", with_xray=True)
    out_no = _run(compute_pa_payer_rules_match(pack_no_xray))
    out_yes = _run(compute_pa_payer_rules_match(pack_xray))
    assert out_yes.n_met >= out_no.n_met


def test_next_steps_populated_with_unmet_gaps():
    pack = _imaging_pack("unitedhealth", with_xray=False)
    out = _run(compute_pa_payer_rules_match(pack))
    assert out.next_steps
    assert any("X-ray" in step for step in out.next_steps)


def test_anthem_bcbs_two_prior_trials_rule_partial():
    """Anthem requires 2+ prior trials. With only 1 med it should be unmet."""
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=PARequestedService(
            service_type="specialty_drug",
            description="adalimumab",
        ),
        payer="anthem_bcbs",
        fhir_bundle=_bundle_with(with_meds=1),
    ))
    out = _run(compute_pa_payer_rules_match(pack))
    rule = next(r for r in out.rules
                    if r.rule_id == "bcbs.drug.formulary_alternatives")
    assert rule.status == "unmet"
    assert "two prior" in (rule.gap_text or "").lower()


def test_anthem_bcbs_two_prior_trials_rule_met():
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=PARequestedService(
            service_type="specialty_drug",
            description="adalimumab",
        ),
        payer="anthem_bcbs",
        fhir_bundle=_bundle_with(with_meds=3),
    ))
    out = _run(compute_pa_payer_rules_match(pack))
    rule = next(r for r in out.rules
                    if r.rule_id == "bcbs.drug.formulary_alternatives")
    assert rule.status == "met"
    assert len(rule.evidence_ids) == 2


def test_overall_alignment_misaligned_for_minimal_evidence():
    pack = _imaging_pack("aetna", with_xray=False, with_obs=False)
    out = _run(compute_pa_payer_rules_match(pack))
    assert out.overall_alignment in (
        "misaligned", "weakly_aligned", "partially_aligned",
    )


def test_overall_alignment_majority_or_better_with_full_pack():
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=PARequestedService(
            service_type="imaging_advanced",
            description="MRI lumbar",
        ),
        payer="unitedhealth",
        fhir_bundle=_bundle_with(with_meds=2, with_xray=True),
        chart_excerpts=[
            {"text": "Patient saw orthopedic specialist for evaluation",
             "fhir_resource_id": "chart-1"},
            {"text": "Conservative therapy 6 weeks without improvement",
             "fhir_resource_id": "chart-2"},
        ],
    ))
    out = _run(compute_pa_payer_rules_match(pack))
    assert out.n_met >= 2
    assert out.overall_alignment in (
        "fully_aligned", "majority_aligned", "partially_aligned",
    )


def test_generic_rules_apply_to_unknown_payer_combo():
    """A payer/service combination without specific rules still gets
    the 3 generic rules."""
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=PARequestedService(
            service_type="dme",
            description="walker, standard adult",
        ),
        payer="generic",
        fhir_bundle=_bundle_with(with_meds=1),
        chart_excerpts=[
            {"text": "Patient unable to ambulate without assistance",
             "fhir_resource_id": "chart-1"},
        ],
    ))
    out = _run(compute_pa_payer_rules_match(pack))
    rule_ids = {r.rule_id for r in out.rules}
    assert "generic.dx_specificity" in rule_ids
    assert "generic.evidence_recency" in rule_ids
    assert "generic.medical_necessity_narrative" in rule_ids


def test_rules_match_carries_references():
    pack = _imaging_pack("unitedhealth")
    out = _run(compute_pa_payer_rules_match(pack))
    assert any("AMA" in r or "Coverage" in r for r in out.references)
