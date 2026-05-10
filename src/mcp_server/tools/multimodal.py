"""healthcare.compute_ecg_qt_analyzer / compute_dicom_sr_ingest
-- Phase 11.4 Multi-modal bundle.

First multi-modal surface in TrustedRisk. Two pure-deterministic
tools that consume signal / structured-report inputs:

  - **compute_ecg_qt_analyzer**: takes a 12-lead ECG strip dict
    (synthetic structured form: leads, sample_rate_hz, R-peak indices,
    QRS-onset / T-end indices). Computes mean R-R, heart rate,
    QT interval, **QTc (Bazett + Fridericia)**, and the prolongation
    classification (normal / borderline / prolonged / severe_prolonged)
    using AHA/ACC/HRS 2009 cut-offs, sex-aware.

  - **compute_dicom_sr_ingest**: takes a DICOM Structured Report
    serialised as a FHIR DiagnosticReport-shaped dict (or DICOM SR
    JSON), surfaces the modality, body part, coded findings, and
    the conclusion text. The SR can carry CT/MR/US/XR/ECG/etc.; the
    tool normalises to a `DICOMSRIngestReport` for downstream agents.

References:
- Rautaharju PM et al. AHA/ACCF/HRS Recommendations for the
  Standardization and Interpretation of the ECG, Part IV: The ST
  Segment, T and U Waves, and the QT Interval. Circulation 2009.
- Bazett HC. An analysis of the time-relations of electrocardiograms.
  Heart 1920;7:353-370.
- Fridericia LS. The duration of systole in an electrocardiogram in
  normal humans and in patients with heart disease. Acta Med Scand
  1920;53:469-486.
- DICOM Standard PS3.16 -- Content Mapping Resource (Coded Concepts).
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from shared.schemas import (
    DICOMSRFinding,
    DICOMSRIngestReport,
    ECGQTReport,
)


# ─────────────────────────────────────────────────────────────────────
# ECG QT analyzer (compute_ecg_qt_analyzer)
# ─────────────────────────────────────────────────────────────────────

# AHA/ACC/HRS 2009 QTc cut-offs (ms)
#                          normal   borderline   prolonged   severe
#   Adult male:            ≤ 430    431-450      451-499     ≥ 500
#   Adult female:          ≤ 450    451-470      471-499     ≥ 500
#   Unspecified default uses the more conservative female set.

_QTC_CUTS = {
    "male":        (430, 450, 499),
    "female":      (450, 470, 499),
    "unspecified": (450, 470, 499),
}


def _classify_qtc(qtc_ms: float, sex: str) -> str:
    cuts = _QTC_CUTS.get(sex, _QTC_CUTS["unspecified"])
    if qtc_ms <= cuts[0]:
        return "normal"
    if qtc_ms <= cuts[1]:
        return "borderline"
    if qtc_ms <= cuts[2]:
        return "prolonged"
    return "severe_prolonged"


# Common QT-prolonging risk factors flagged on the report
_RISK_KEYWORDS: list[tuple[str, str]] = [
    ("hypokalemia", "Hypokalaemia"),
    ("hypomagnesemia", "Hypomagnesaemia"),
    ("hypocalcemia", "Hypocalcaemia"),
    ("methadone", "Methadone (QT-prolonging)"),
    ("amiodarone", "Amiodarone (QT-prolonging class III)"),
    ("sotalol", "Sotalol (QT-prolonging class III)"),
    ("citalopram", "Citalopram (QT-prolonging SSRI)"),
    ("fluoroquinolone", "Fluoroquinolone (QT-prolonging)"),
    ("ondansetron", "Ondansetron (QT-prolonging)"),
    ("haloperidol", "Haloperidol (QT-prolonging antipsychotic)"),
    ("bradycardia", "Bradycardia"),
    ("congenital long qt", "Congenital long-QT syndrome"),
]


def _scan_risk_factors(history_text: str | None) -> list[str]:
    if not history_text:
        return []
    lo = history_text.lower()
    return [label for kw, label in _RISK_KEYWORDS if kw in lo]


def _seconds_per_sample(sample_rate_hz: float) -> float:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    return 1.0 / sample_rate_hz


async def compute_ecg_qt_analyzer(
    sample_rate_hz: float,
    r_peak_indices: list[int],
    qrs_onset_indices: list[int],
    t_end_indices: list[int],
    *,
    sex: str = "unspecified",
    history_text: str | None = None,
) -> ECGQTReport:
    """Compute QT/QTc on a structured ECG strip.

    Args:
        sample_rate_hz: ECG sampling frequency (Hz).
        r_peak_indices: sample indices of R peaks across the strip.
            Must contain at least 2 points so the R-R interval can be
            estimated.
        qrs_onset_indices: per-beat QRS-onset sample index. Length must
            equal len(t_end_indices).
        t_end_indices: per-beat T-end sample index.
        sex: 'male' / 'female' / 'unspecified' for the cut-off table.
        history_text: optional free-text history for QT-prolongation
            risk-factor mining (electrolytes, QT-prolonging drugs,
            congenital LQTS).

    Returns:
        ECGQTReport with QT, QTc Bazett + Fridericia, classification,
        risk factors, and rationale.
    """
    # Length-mismatch is a contract bug, not a sparse-input case -- raise.
    if len(qrs_onset_indices) != len(t_end_indices):
        raise ValueError(
            "qrs_onset_indices and t_end_indices must have equal length"
        )

    # Sparse-input cases return a structured abstain record so the BYO
    # LLM can surface the missing-data reason verbatim instead of either
    # fabricating a QTc or seeing a hard exception.
    if len(r_peak_indices) < 2:
        return ECGQTReport(
            sex=sex if sex in ("male", "female") else "unspecified",  # type: ignore[arg-type]
            rationale=(
                "Cannot compute R-R interval: fewer than 2 R peaks "
                "supplied. ECG QT analysis abstained."
            ),
            references=[
                "Rautaharju PM et al. AHA/ACCF/HRS recommendations for "
                "ECG interpretation, Part IV -- QT interval. Circulation 2009.",
            ],
            abstain_recommended=True,
            abstain_reason="missing_ecg_observation",
        )
    if not qrs_onset_indices:
        return ECGQTReport(
            sex=sex if sex in ("male", "female") else "unspecified",  # type: ignore[arg-type]
            rationale=(
                "No QRS-onset / T-end annotations supplied. Cannot "
                "compute QT interval. ECG QT analysis abstained."
            ),
            references=[
                "Rautaharju PM et al. AHA/ACCF/HRS recommendations for "
                "ECG interpretation, Part IV -- QT interval. Circulation 2009.",
            ],
            abstain_recommended=True,
            abstain_reason="no_qt_annotations",
        )

    sps = _seconds_per_sample(sample_rate_hz)

    # Mean R-R (ms): consecutive deltas
    rr_samples = [
        r_peak_indices[i + 1] - r_peak_indices[i]
        for i in range(len(r_peak_indices) - 1)
    ]
    rr_seconds = statistics.mean(rr_samples) * sps
    if rr_seconds <= 0:
        raise ValueError("R-R interval non-positive -- bad input")
    rr_ms = rr_seconds * 1000.0
    hr_bpm = 60.0 / rr_seconds

    # Mean QT (ms) across beats
    qt_samples = [
        max(0, t_end - qrs_onset)
        for qrs_onset, t_end in zip(qrs_onset_indices, t_end_indices)
    ]
    qt_seconds = statistics.mean(qt_samples) * sps
    qt_ms = qt_seconds * 1000.0

    # Bazett: QTc = QT / sqrt(R-R[s])
    qtc_b_ms = qt_ms / math.sqrt(rr_seconds)
    # Fridericia: QTc = QT / cuberoot(R-R[s])
    qtc_f_ms = qt_ms / (rr_seconds ** (1.0 / 3.0))

    sex_norm = sex if sex in ("male", "female") else "unspecified"
    qtc_class = _classify_qtc(qtc_b_ms, sex_norm)
    risk_factors = _scan_risk_factors(history_text)

    rationale = (
        f"R-R {rr_ms:.0f} ms (HR {hr_bpm:.1f} bpm); QT {qt_ms:.0f} ms; "
        f"QTc Bazett {qtc_b_ms:.0f} ms / Fridericia {qtc_f_ms:.0f} ms; "
        f"classification = {qtc_class} (sex={sex_norm}). "
        f"Risk factors detected: {len(risk_factors)}."
    )

    return ECGQTReport(
        rr_interval_ms=round(rr_ms, 1),
        heart_rate_bpm=round(hr_bpm, 1),
        qt_interval_ms=round(qt_ms, 1),
        qtc_bazett_ms=round(qtc_b_ms, 1),
        qtc_fridericia_ms=round(qtc_f_ms, 1),
        qtc_classification=qtc_class,             # type: ignore[arg-type]
        sex=sex_norm,                              # type: ignore[arg-type]
        prolongation_risk_factors=risk_factors,
        rationale=rationale,
        references=[
            "Rautaharju PM et al. AHA/ACCF/HRS recommendations for "
            "ECG interpretation, Part IV -- QT interval. Circulation 2009.",
            "Bazett HC. Heart 1920;7:353-370.",
            "Fridericia LS. Acta Med Scand 1920;53:469-486.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# DICOM SR ingest (compute_dicom_sr_ingest)
# ─────────────────────────────────────────────────────────────────────


_VALID_MODALITIES = {
    "CT", "MR", "US", "XR", "CR", "ECG", "PET", "NM", "OT",
}


_SEVERITY_HINT_KEYWORDS: list[tuple[str, str]] = [
    # Negation / stability patterns must come BEFORE the positive cues
    # they would otherwise mask (e.g. "no acute" before "acute").
    ("no acute", "informational"),
    ("no significant", "informational"),
    ("unremarkable", "informational"),
    ("stable", "informational"),
    ("ruptured", "critical"),
    ("dissection", "critical"),
    ("massive", "critical"),
    ("active extravasation", "critical"),
    ("perforation", "high"),
    ("acute", "high"),
    ("severe", "high"),
    ("malignant", "high"),
    ("moderate", "moderate"),
    ("mild", "low"),
    ("trace", "low"),
]


def _heuristic_severity(text: str) -> str:
    lo = text.lower()
    for kw, sev in _SEVERITY_HINT_KEYWORDS:
        if kw in lo:
            return sev
    return "informational"


def _normalise_modality(value: str) -> str:
    v = (value or "").strip().upper()
    if v in _VALID_MODALITIES:
        return v
    # Common aliases
    if v in ("X-RAY", "RX", "RADIOGRAPH"):
        return "XR"
    if v in ("ULTRASOUND",):
        return "US"
    if v in ("MAGNETIC RESONANCE", "MRI"):
        return "MR"
    if v in ("COMPUTED TOMOGRAPHY",):
        return "CT"
    return "OT"


async def compute_dicom_sr_ingest(
    sr_document: dict[str, Any],
    *,
    cite_resource_id_path: str | None = None,
) -> DICOMSRIngestReport:
    """Surface a DICOM Structured Report (or FHIR DiagnosticReport)
    into a structured `DICOMSRIngestReport`.

    Accepted shapes:
        - FHIR R4 DiagnosticReport with `code`, `category`, `result`,
          `conclusion`, `presentedForm` etc.
        - DICOM SR JSON with `Modality`, `BodyPartExamined`,
          `Findings`, `Impression`.

    Args:
        sr_document: dict-shaped report.
        cite_resource_id_path: optional explicit `id` to record under
            `cited_resource_ids` (otherwise we extract from the doc).

    Returns:
        DICOMSRIngestReport.
    """
    if not isinstance(sr_document, dict) or not sr_document:
        raise ValueError("sr_document must be a non-empty dict")

    # Modality + body part
    modality_raw = (
        sr_document.get("Modality")
        or sr_document.get("modality")
        or (
            (sr_document.get("category") or [{}])[0].get(
                "coding", [{}])[0].get("code", "")
            if isinstance(sr_document.get("category"), list) else ""
        )
        or "OT"
    )
    modality = _normalise_modality(str(modality_raw))

    body_part = (
        sr_document.get("BodyPartExamined")
        or sr_document.get("body_part_examined")
        or sr_document.get("bodySite")
        or None
    )
    if isinstance(body_part, dict):
        body_part = (
            body_part.get("text")
            or (body_part.get("coding") or [{}])[0].get("display")
        )

    findings_raw: list[Any] = []
    for key in ("Findings", "findings", "result", "results"):
        v = sr_document.get(key)
        if isinstance(v, list) and v:
            findings_raw = v
            break

    findings: list[DICOMSRFinding] = []
    for f in findings_raw:
        if isinstance(f, dict):
            code_block = f.get("code") or {}
            coding = (code_block.get("coding") or [{}])[0]
            code_system = (
                coding.get("system")
                or f.get("CodingSchemeDesignator")
                or "http://snomed.info/sct"
            )
            code = (
                coding.get("code")
                or f.get("CodeValue")
                or "UNKNOWN"
            )
            display = (
                coding.get("display")
                or code_block.get("text")
                or f.get("CodeMeaning")
                or f.get("text")
                or "(no display)"
            )
            severity = (
                f.get("severity")
                or _heuristic_severity(str(display))
            )
            findings.append(DICOMSRFinding(
                code_system=str(code_system),
                code=str(code),
                display=str(display),
                severity=severity,                     # type: ignore[arg-type]
            ))
        elif isinstance(f, str):
            findings.append(DICOMSRFinding(
                code_system="text",
                code="FREE_TEXT",
                display=f,
                severity=_heuristic_severity(f),       # type: ignore[arg-type]
            ))

    # Impressions
    impressions_raw = (
        sr_document.get("Impression")
        or sr_document.get("impression")
        or sr_document.get("impressions")
        or []
    )
    if isinstance(impressions_raw, str):
        impressions = [impressions_raw]
    elif isinstance(impressions_raw, list):
        impressions = [str(x) for x in impressions_raw if x]
    else:
        impressions = []

    conclusion = (
        sr_document.get("conclusion")
        or sr_document.get("Conclusion")
        or None
    )
    if isinstance(conclusion, list):
        conclusion = " ".join(str(x) for x in conclusion if x) or None

    cite_ids: list[str] = []
    if cite_resource_id_path:
        cite_ids.append(cite_resource_id_path)
    elif sr_document.get("id"):
        cite_ids.append(str(sr_document["id"]))

    rationale = (
        f"Modality {modality}; body part {body_part or '<none>'}; "
        f"{len(findings)} finding(s) ingested. "
        f"Severity tiers: "
        + ", ".join(
            f"{sev}={sum(1 for f in findings if f.severity == sev)}"
            for sev in ("informational", "low", "moderate", "high", "critical")
            if any(f.severity == sev for f in findings)
        )
    )

    return DICOMSRIngestReport(
        modality=modality,                            # type: ignore[arg-type]
        body_part=str(body_part) if body_part else None,
        n_findings=len(findings),
        findings=findings,
        impressions=impressions,
        conclusion=str(conclusion) if conclusion else None,
        cited_resource_ids=cite_ids,
        rationale=rationale,
        references=[
            "DICOM Standard PS3.16 -- Content Mapping Resource.",
            "FHIR R4 DiagnosticReport profile -- hl7.org/fhir/diagnosticreport.html",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_ecg_qt_analyzer)
    mcp.tool()(compute_dicom_sr_ingest)
