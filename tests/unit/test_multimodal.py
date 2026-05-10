"""Phase 11.4 -- Multi-modal bundle (ECG QT + DICOM SR ingest)."""

from __future__ import annotations

import asyncio
import math

import pytest

from mcp_server.tools.multimodal import (
    compute_dicom_sr_ingest,
    compute_ecg_qt_analyzer,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# ECG QT analyzer -- golden HR/QT/QTc
# ─────────────────────────────────────────────────────────────────────

def _r_peaks_for_hr(hr_bpm: float, sample_rate_hz: float,
                            n_beats: int = 6) -> list[int]:
    """Generate evenly-spaced R-peak indices that match a given heart rate."""
    rr_seconds = 60.0 / hr_bpm
    rr_samples = int(round(rr_seconds * sample_rate_hz))
    return [i * rr_samples for i in range(n_beats)]


def _qt_indices(rate: int, qt_ms: float, n_beats: int):
    """Generate per-beat QRS-onset / T-end indices for a fixed QT."""
    qt_samples = int(round(qt_ms * rate / 1000.0))
    qrs = [i * 1000 for i in range(n_beats)]
    t_end = [s + qt_samples for s in qrs]
    return qrs, t_end


def test_ecg_qt_classifies_normal_male():
    sample_rate = 500
    rs = _r_peaks_for_hr(60, sample_rate, 6)
    qrs, t_end = _qt_indices(sample_rate, qt_ms=400, n_beats=6)
    out = _run(compute_ecg_qt_analyzer(
        sample_rate_hz=sample_rate,
        r_peak_indices=rs,
        qrs_onset_indices=qrs,
        t_end_indices=t_end,
        sex="male",
    ))
    # HR ~ 60 bpm -> R-R = 1.0 s -> Bazett QTc = 400 / sqrt(1.0) = 400 ms (normal male)
    assert out.qtc_classification == "normal"
    assert 55 <= out.heart_rate_bpm <= 65
    assert 390 <= out.qtc_bazett_ms <= 410


def test_ecg_qt_classifies_severe_prolonged_when_qt_long():
    sample_rate = 500
    # HR 90 -> R-R ~0.667 s -> QT 460 ms -> Bazett ~ 563 ms (severe_prolonged)
    rs = _r_peaks_for_hr(90, sample_rate, 6)
    qrs, t_end = _qt_indices(sample_rate, qt_ms=460, n_beats=6)
    out = _run(compute_ecg_qt_analyzer(
        sample_rate_hz=sample_rate, r_peak_indices=rs,
        qrs_onset_indices=qrs, t_end_indices=t_end, sex="female",
    ))
    assert out.qtc_classification in ("prolonged", "severe_prolonged")


def test_ecg_qt_bazett_and_fridericia_diverge_when_hr_off_60():
    """Bazett over-corrects at high HR; Fridericia is closer to the
    true QTc. Sanity-check that the two values diverge."""
    sample_rate = 500
    rs = _r_peaks_for_hr(120, sample_rate, 6)   # HR 120
    qrs, t_end = _qt_indices(sample_rate, qt_ms=380, n_beats=6)
    out = _run(compute_ecg_qt_analyzer(
        sample_rate_hz=sample_rate, r_peak_indices=rs,
        qrs_onset_indices=qrs, t_end_indices=t_end,
    ))
    # Bazett > Fridericia at HR > 60 (Bazett over-corrects upwards)
    assert out.qtc_bazett_ms > out.qtc_fridericia_ms


def test_ecg_qt_flags_qt_prolonging_drug_in_history():
    sample_rate = 500
    rs = _r_peaks_for_hr(60, sample_rate, 6)
    qrs, t_end = _qt_indices(sample_rate, qt_ms=400, n_beats=6)
    out = _run(compute_ecg_qt_analyzer(
        sample_rate_hz=sample_rate, r_peak_indices=rs,
        qrs_onset_indices=qrs, t_end_indices=t_end,
        history_text=(
            "Patient on methadone 80 mg daily for chronic pain, plus "
            "recent ondansetron for nausea."
        ),
    ))
    factors = " ".join(out.prolongation_risk_factors).lower()
    assert "methadone" in factors
    assert "ondansetron" in factors


def test_ecg_qt_abstains_on_too_few_r_peaks():
    """Sparse-input case (no waveform usable for R-R estimation)
    returns a structured abstain record, not a ValueError. The
    BYO LLM consumes the abstain_reason verbatim; the surrounding
    schema (rationale, references) stays populated so the report
    is renderable end-to-end.
    """
    report = _run(compute_ecg_qt_analyzer(
        sample_rate_hz=500, r_peak_indices=[100],
        qrs_onset_indices=[100], t_end_indices=[300],
    ))
    assert report.abstain_recommended is True
    assert report.abstain_reason == "missing_ecg_observation"
    assert report.rr_interval_ms is None
    assert report.qtc_classification is None
    assert report.references  # provenance still attached


def test_ecg_qt_rejects_mismatched_qrs_t_lengths():
    with pytest.raises(ValueError):
        _run(compute_ecg_qt_analyzer(
            sample_rate_hz=500, r_peak_indices=[0, 500, 1000],
            qrs_onset_indices=[100, 600],
            t_end_indices=[300, 800, 1300],
        ))


def test_ecg_qt_rejects_zero_sample_rate():
    with pytest.raises(ValueError):
        _run(compute_ecg_qt_analyzer(
            sample_rate_hz=0, r_peak_indices=[0, 500],
            qrs_onset_indices=[100], t_end_indices=[300],
        ))


# ─────────────────────────────────────────────────────────────────────
# DICOM SR ingest
# ─────────────────────────────────────────────────────────────────────

def test_dicom_sr_normalises_modality_and_extracts_findings():
    sr = {
        "id": "diag-001",
        "Modality": "CT",
        "BodyPartExamined": "ABDOMEN",
        "Findings": [
            {"CodeValue": "16531000",
             "CodingSchemeDesignator": "SCT",
             "CodeMeaning": "Acute appendicitis"},
            {"CodeValue": "65389002",
             "CodingSchemeDesignator": "SCT",
             "CodeMeaning": "Trace free fluid"},
        ],
        "Impression": "Acute appendicitis, surgical consult.",
    }
    out = _run(compute_dicom_sr_ingest(sr))
    assert out.modality == "CT"
    assert out.body_part == "ABDOMEN"
    assert out.n_findings == 2
    sevs = {f.severity for f in out.findings}
    # "acute" -> high severity; "trace" -> low severity
    assert "high" in sevs
    assert "low" in sevs


def test_dicom_sr_supports_fhir_diagnosticreport_shape():
    fhir_dr = {
        "id": "DR-XR-CHEST",
        "resourceType": "DiagnosticReport",
        "category": [{"coding": [{"code": "XR"}]}],
        "bodySite": {"text": "Chest"},
        "result": [
            {"code": {"coding": [{"system": "http://loinc.org",
                                       "code": "12345-6",
                                       "display": "Pneumothorax, severe"}]}},
        ],
        "conclusion": "Severe pneumothorax -- chest tube indicated.",
    }
    out = _run(compute_dicom_sr_ingest(fhir_dr))
    assert out.modality == "XR"
    assert out.body_part == "Chest"
    assert out.findings[0].severity == "high"
    assert out.conclusion is not None
    assert "DR-XR-CHEST" in out.cited_resource_ids


def test_dicom_sr_normalises_alias_modality():
    sr = {"modality": "X-Ray", "Findings": []}
    out = _run(compute_dicom_sr_ingest(sr))
    assert out.modality == "XR"


def test_dicom_sr_unknown_modality_falls_back_to_OT():
    out = _run(compute_dicom_sr_ingest({"modality": "weird-thing"}))
    assert out.modality == "OT"


def test_dicom_sr_rejects_empty_input():
    with pytest.raises(ValueError):
        _run(compute_dicom_sr_ingest({}))


def test_dicom_sr_handles_free_text_findings():
    """Findings can be plain strings -- the heuristic severity must
    still fire."""
    out = _run(compute_dicom_sr_ingest({
        "Modality": "MR",
        "Findings": [
            "No acute intracranial abnormality.",
            "Severe white-matter changes consistent with chronic ischaemia.",
        ],
    }))
    sevs = {f.severity for f in out.findings}
    assert "informational" in sevs
    assert "high" in sevs


# ─────────────────────────────────────────────────────────────────────
# Bundle wiring
# ─────────────────────────────────────────────────────────────────────

def test_multimodal_bundle_registered():
    from mcp_server.tools import BUNDLES
    assert "multimodal" in BUNDLES
    assert set(BUNDLES["multimodal"]) == {
        "compute_ecg_qt_analyzer",
        "compute_dicom_sr_ingest",
    }


def test_multimodal_scopes_declared():
    from mcp_server.scopes import BUNDLE_SCOPES
    assert "multimodal" in BUNDLE_SCOPES
    assert "patient/DiagnosticReport.rs" in BUNDLE_SCOPES["multimodal"]
