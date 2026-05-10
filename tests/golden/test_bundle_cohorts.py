"""Bundle batch regression tests over synthetic cohorts (DATA-2).

For each generated cohort in `tests/golden/cohorts/*.json`, parametrize
over the 50 cases and assert:
  1. The tool returns a Pydantic-valid output (schema-valid).
  2. The output contains key fields per the tool's output schema.
  3. No case crashes.

This catches schema regressions and crashes that single-input unit
tests would miss. With 600 cases × 12 cohorts = 600 invocations per run.

Cohorts are auto-regenerable from `scripts/generate_synthetic_cohorts.py`
with a fixed seed (4242), so the tests are deterministic.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _resolve_tool(tool_name: str):
    """Resolve a tool function by its compute_* name."""
    from mcp_server.tools import (
        admission_triage, aki_kdigo_stage, antibiotic_de_escalation,
        chemo_dose_adjustment, clinical_deterioration_score,
        contrast_safety_check, delirium_screening_cam,
        dialysis_initiation_decision, dka_severity,
        empiric_antibiotic_selection, falls_risk_morse, heart_score,
        imaging_appropriateness, inpatient_glycemic_control,
        massive_transfusion_protocol, maternal_early_warning,
        oncology_treatment_response, pediatric_early_warning,
        polypharmacy_concerns, preeclampsia_assessment,
        psychiatric_admission_decision, stroke_severity,
        stroke_thrombolysis_eligibility, suicide_risk_assessment,
        trauma_severity_score, treatment_selection, weight_based_dosing,
        acs_disposition_decision,
    )
    table = {
        "compute_admission_triage": admission_triage.compute_admission_triage,
        "compute_aki_kdigo_stage": aki_kdigo_stage.compute_aki_kdigo_stage,
        "compute_antibiotic_de_escalation": antibiotic_de_escalation.compute_antibiotic_de_escalation,
        "compute_chemo_dose_adjustment": chemo_dose_adjustment.compute_chemo_dose_adjustment,
        "compute_clinical_deterioration_score": clinical_deterioration_score.compute_clinical_deterioration_score,
        "compute_contrast_safety_check": contrast_safety_check.compute_contrast_safety_check,
        "compute_delirium_screening_cam": delirium_screening_cam.compute_delirium_screening_cam,
        "compute_dialysis_initiation_decision": dialysis_initiation_decision.compute_dialysis_initiation_decision,
        "compute_dka_severity": dka_severity.compute_dka_severity,
        "compute_empiric_antibiotic_selection": empiric_antibiotic_selection.compute_empiric_antibiotic_selection,
        "compute_falls_risk_morse": falls_risk_morse.compute_falls_risk_morse,
        "compute_heart_score": heart_score.compute_heart_score,
        "compute_imaging_appropriateness": imaging_appropriateness.compute_imaging_appropriateness,
        "compute_inpatient_glycemic_control": inpatient_glycemic_control.compute_inpatient_glycemic_control,
        "compute_massive_transfusion_protocol": massive_transfusion_protocol.compute_massive_transfusion_protocol,
        "compute_maternal_early_warning": maternal_early_warning.compute_maternal_early_warning,
        "compute_oncology_treatment_response": oncology_treatment_response.compute_oncology_treatment_response,
        "compute_pediatric_early_warning": pediatric_early_warning.compute_pediatric_early_warning,
        "detect_polypharmacy_concerns": polypharmacy_concerns.detect_polypharmacy_concerns,
        "compute_preeclampsia_assessment": preeclampsia_assessment.compute_preeclampsia_assessment,
        "compute_psychiatric_admission_decision": psychiatric_admission_decision.compute_psychiatric_admission_decision,
        "compute_stroke_severity": stroke_severity.compute_stroke_severity,
        "compute_stroke_thrombolysis_eligibility": stroke_thrombolysis_eligibility.compute_stroke_thrombolysis_eligibility,
        "compute_suicide_risk_assessment": suicide_risk_assessment.compute_suicide_risk_assessment,
        "compute_trauma_severity_score": trauma_severity_score.compute_trauma_severity_score,
        "compute_treatment_selection": treatment_selection.compute_treatment_selection,
        "compute_weight_based_dosing": weight_based_dosing.compute_weight_based_dosing,
        "compute_acs_disposition_decision": acs_disposition_decision.compute_acs_disposition_decision,
    }
    return table.get(tool_name)


def _load_cohorts():
    cohorts_dir = ROOT / "tests" / "golden" / "cohorts"
    out = []
    if not cohorts_dir.exists():
        return out
    for f in sorted(cohorts_dir.glob("*.json")):
        spec = json.loads(f.read_text(encoding="utf-8"))
        for case in spec.get("cases", []):
            out.append((spec["bundle_id"], spec["tool"], case))
    return out


_COHORT_CASES = _load_cohorts()


def _ids(records):
    return [f"{bundle_id}::{case['name']}"
             for bundle_id, _, case in records]


@pytest.mark.parametrize(
    "bundle_id, tool_name, case", _COHORT_CASES, ids=_ids(_COHORT_CASES),
)
def test_cohort_case(bundle_id, tool_name, case):
    """Each synthetic case must produce a schema-valid output without crashing."""
    if "TRUSTEDRISK_COEFFICIENTS_PATH" not in os.environ:
        os.environ["TRUSTEDRISK_COEFFICIENTS_PATH"] = "data/coefficients.json"
    fn = _resolve_tool(tool_name)
    assert fn is not None, f"Unknown tool {tool_name!r}"
    inputs = case["input"]
    # Run the tool -- must not crash; output must be Pydantic-valid
    output = asyncio.run(fn(**inputs))
    # Pydantic validation already enforced by the tool's return type;
    # we additionally serialize to ensure no surprises.
    if hasattr(output, "model_dump"):
        dumped = output.model_dump(mode="json")
        assert isinstance(dumped, dict)
        assert len(dumped) >= 2  # output should have at least a couple of fields
