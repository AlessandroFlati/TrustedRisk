"""Data-driven regression suite -- golden cases for clinical tools.

For each `tests/golden/cases/*.json` file, parametrize over the cases and
assert that the tool's output contains the expected_subset fields exactly.

This catches drift when:
  - someone changes a threshold and forgets to update the test
  - a refactor accidentally swaps a tool's output schema
  - a new dependency changes calibration coefficients

The expected_subset is a SUBSET match (additive new fields are ignored),
so the suite tolerates schema additions but flags changes to existing
field semantics.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest


# Tool name -> resolver function
def _resolve_tool(tool_name: str):
    """Map a tool function name (e.g. 'compute_stroke_severity') to the
    callable. Lazy import to avoid loading every tool at collection time."""
    from mcp_server.tools import (
        admission_triage,
        aki_kdigo_stage,
        antibiotic_de_escalation,
        chemo_dose_adjustment,
        clinical_deterioration_score,
        contrast_safety_check,
        delirium_screening_cam,
        dialysis_initiation_decision,
        dka_severity,
        empiric_antibiotic_selection,
        falls_risk_morse,
        heart_score,
        imaging_appropriateness,
        inpatient_glycemic_control,
        massive_transfusion_protocol,
        maternal_early_warning,
        oncology_treatment_response,
        pediatric_early_warning,
        preeclampsia_assessment,
        psychiatric_admission_decision,
        stroke_severity,
        stroke_thrombolysis_eligibility,
        suicide_risk_assessment,
        trauma_severity_score,
        treatment_selection,
        weight_based_dosing,
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
    fn = table.get(tool_name)
    if fn is None:
        raise KeyError(f"golden case references unknown tool {tool_name!r}")
    return fn


def _load_cases() -> list[tuple[str, str, str, dict[str, Any]]]:
    """Return list of (file_id, case_name, tool_name, case_dict)."""
    here = Path(__file__).parent / "cases"
    out: list[tuple[str, str, str, dict[str, Any]]] = []
    for f in sorted(here.glob("*.json")):
        spec = json.loads(f.read_text(encoding="utf-8"))
        default_tool = spec.get("tool", "multi_tool")
        for case in spec.get("cases", []):
            tool_name = case.get("tool") or default_tool
            out.append((f.stem, case["name"], tool_name, case))
    return out


def _ids(records):
    return [f"{r[0]}::{r[1]}" for r in records]


_CASES = _load_cases()


def _assert_subset_match(output: Any, expected: dict[str, Any], path: str = "") -> None:
    """Recursively assert that every key in `expected` is present in `output`
    with an equal value. Non-listed keys are ignored."""
    if not isinstance(output, dict):
        # Try Pydantic dump
        if hasattr(output, "model_dump"):
            output = output.model_dump(mode="json")
    for k, v in expected.items():
        full_path = f"{path}.{k}" if path else k
        assert k in output, f"Missing field {full_path!r} in output"
        actual = output[k]
        if isinstance(v, dict) and isinstance(actual, dict):
            _assert_subset_match(actual, v, full_path)
        elif isinstance(v, float):
            assert abs(actual - v) < 1e-3, (
                f"Field {full_path!r}: expected ≈{v}, got {actual}"
            )
        else:
            assert actual == v, (
                f"Field {full_path!r}: expected {v!r}, got {actual!r}"
            )


@pytest.mark.parametrize("file_id, case_name, tool_name, case",
                            _CASES, ids=_ids(_CASES))
def test_golden_case(file_id, case_name, tool_name, case, monkeypatch):
    """Run one golden case against its tool and assert expected_subset."""
    # Ensure the calibrated coefficients are reachable
    if "TRUSTEDRISK_COEFFICIENTS_PATH" not in os.environ:
        os.environ["TRUSTEDRISK_COEFFICIENTS_PATH"] = "data/coefficients.json"

    fn = _resolve_tool(tool_name)
    inputs = case.get("input", {})
    expected_subset = case.get("expected_subset")
    expected_top_pick_substring = case.get("expected_top_pick_contains")

    output = asyncio.run(fn(**inputs))

    if expected_subset:
        _assert_subset_match(output, expected_subset)
    if expected_top_pick_substring is not None:
        # Used by imaging_appropriateness
        if hasattr(output, "model_dump"):
            d = output.model_dump(mode="json")
        else:
            d = output
        top = d.get("top_pick_modality") or d.get("top_pick_id") or ""
        assert expected_top_pick_substring.lower() in str(top).lower(), (
            f"top pick {top!r} does not contain {expected_top_pick_substring!r}"
        )
