"""healthcare.compute_admission_triage -- ED ESI triage + disposition.

Hybrid implementation of the AHRQ Emergency Severity Index v4 (2020) plus
chief-complaint pattern matching from the Manchester Triage System (Wiley
2014). Produces an ESI level (1=immediate, 5=non-urgent), a disposition
recommendation (resus/admit/observe/ED-workup/discharge), and a recommended
unit when admission is warranted.

The tool is deliberately conservative: any red-flag finding pulls the
triage upward (toward more urgent), and any decision that would normally be
"discharge from ED" is gated behind the absence of red flags. When inputs
are ambiguous (e.g. unknown vital signs), the tool sets `abstain_recommended`
so the EM physician makes the call.
"""

from __future__ import annotations

import re
from typing import Any

from shared.schemas import AdmissionTriage

from ._chart_inputs import harden_clinical_inputs


# ─────────────────────────────────────────────────────────────────────
# Chief-complaint -> red flag rules (Manchester triage flow charts)
# ─────────────────────────────────────────────────────────────────────
# Each entry maps a chief-complaint keyword family to a list of:
#   (red_flag_pattern, esi_floor, finding_text)
# A patient's chief complaint can match multiple categories; the highest
# severity (lowest ESI level) wins.

_RED_FLAG_RULES: list[tuple[re.Pattern[str], int, str]] = [
    # Cardiac / chest
    (re.compile(r"chest\s+pain", re.I), 2,
     "Chest pain in adult -- exclude ACS / PE / aortic syndrome before discharge"),
    (re.compile(r"radiating\s+(arm|jaw|back)|crushing\s+chest", re.I), 1,
     "Chest pain radiating to arm/jaw/back -- concern for ACS"),

    # Neurological
    (re.compile(r"slurred\s+speech|facial\s+droop|sudden\s+weakness|"
                r"hemiparesis|aphasia|stroke", re.I), 1,
     "Acute focal neurological deficit -- stroke pathway"),
    (re.compile(r"sudden\s+severe\s+headache|thunderclap", re.I), 1,
     "Thunderclap headache -- concern for SAH"),
    (re.compile(r"seizure", re.I), 2,
     "Seizure -- first-time or breakthrough requires workup"),
    (re.compile(r"altered\s+mental\s+status|confusion|delirium", re.I), 2,
     "Altered mental status"),

    # Respiratory
    (re.compile(r"shortness\s+of\s+breath|dyspnea|trouble\s+breathing", re.I), 2,
     "Dyspnea -- assess SpO2 + work of breathing"),
    (re.compile(r"stridor|airway", re.I), 1,
     "Stridor / airway compromise"),

    # Sepsis / infectious
    (re.compile(r"fever\s+with\s+rigor|sepsis|septic", re.I), 2,
     "Possible sepsis -- qSOFA / lactate workup"),

    # GI bleeding
    (re.compile(r"hematemesis|melena|coffee[-\s]ground|massive\s+gi\s+bleed", re.I), 2,
     "Active GI bleeding"),

    # Trauma
    (re.compile(r"major\s+trauma|polytrauma|head\s+injury\s+with\s+loss", re.I), 1,
     "Major trauma / head injury with LOC"),

    # Pediatric red flags
    (re.compile(r"infant.*lethargic|neonatal\s+fever", re.I), 1,
     "Pediatric red flag (lethargic infant / neonatal fever)"),

    # Pregnancy
    (re.compile(r"vaginal\s+bleed.*pregnan|ectopic", re.I), 2,
     "Vaginal bleeding in pregnancy / ectopic concern"),

    # Mental health
    (re.compile(r"suicidal|self[-\s]harm|overdose", re.I), 2,
     "Mental-health red flag (suicidal / overdose)"),
]


# Vitals that automatically push ESI down (more urgent)
def _vitals_red_flags(vitals: dict[str, float | str | None]) -> list[tuple[int, str]]:
    """Return list of (esi_floor, finding_text) triggered by abnormal vitals."""
    flags: list[tuple[int, str]] = []
    sbp = _as_float(vitals.get("systolic_bp"))
    if sbp is not None:
        if sbp < 90:
            flags.append((1, f"Systolic BP {sbp} mmHg (shock range)"))
        elif sbp < 100:
            flags.append((2, f"Systolic BP {sbp} mmHg (borderline hypotension)"))
        elif sbp >= 220:
            flags.append((1, f"Systolic BP {sbp} mmHg (hypertensive emergency)"))

    hr = _as_float(vitals.get("heart_rate"))
    if hr is not None:
        if hr >= 130:
            flags.append((2, f"Heart rate {hr}/min (severe tachycardia)"))
        elif hr <= 40:
            flags.append((2, f"Heart rate {hr}/min (severe bradycardia)"))

    rr = _as_float(vitals.get("respiratory_rate"))
    if rr is not None:
        if rr >= 30:
            flags.append((1, f"Respiratory rate {rr}/min (severe tachypnea)"))
        elif rr >= 25:
            flags.append((2, f"Respiratory rate {rr}/min (tachypnea)"))
        elif rr <= 8:
            flags.append((1, f"Respiratory rate {rr}/min (severe bradypnea)"))

    spo2 = _as_float(vitals.get("spo2"))
    if spo2 is not None:
        if spo2 < 88:
            flags.append((1, f"SpO₂ {spo2}% (severe hypoxemia)"))
        elif spo2 < 92:
            flags.append((2, f"SpO₂ {spo2}% (hypoxemia)"))

    temp = _as_float(vitals.get("temperature"))
    if temp is not None:
        if temp >= 39.5:
            flags.append((2, f"Temperature {temp}°C (high fever)"))
        elif temp <= 35.0:
            flags.append((2, f"Temperature {temp}°C (hypothermia -- sepsis concern)"))

    consciousness = vitals.get("consciousness")
    if isinstance(consciousness, str) and consciousness.strip().upper()[:1] not in ("A", ""):
        flags.append((1, f"Consciousness = {consciousness} (not Alert)"))

    return flags


# ─────────────────────────────────────────────────────────────────────
# ESI level -> priority + disposition mapping
# ─────────────────────────────────────────────────────────────────────

_ESI_PRIORITY = {
    1: ("immediate", "resuscitation_room", "icu"),
    2: ("emergent", "admit_inpatient", "stepdown"),
    3: ("urgent", "ed_workup_then_dispose", "ed_treatment"),
    4: ("less_urgent", "discharge_from_ed", "outpatient_followup"),
    5: ("non_urgent", "discharge_from_ed", "outpatient_followup"),
}


def _esi_to_decision(esi: int, age: int | None) -> tuple[str, str, str | None]:
    """Map ESI level to (priority, disposition, recommended_unit). Pediatric
    patients (<18) at ESI 2-3 default to admit_observation rather than ED-only
    workup, reflecting lower threshold for pediatric ward admission."""
    priority, disposition, unit = _ESI_PRIORITY[esi]
    if age is not None and age < 18 and esi in (2, 3):
        if disposition == "ed_workup_then_dispose":
            disposition = "admit_observation"
            unit = "observation_unit"
    if age is not None and age >= 75 and esi == 3:
        # Older adults benefit from observation rather than rapid discharge
        disposition = "admit_observation"
        unit = "observation_unit"
    return priority, disposition, unit


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_admission_triage(
    chief_complaint: str,
    vital_signs: dict[str, Any] | list[dict[str, Any]] | None = None,
    age: int | None = None,
    comorbidity_summary: str | None = None,
    patient_id: str | None = None,
) -> AdmissionTriage:
    """ED triage: ESI level + disposition + recommended unit.

    Args:
        chief_complaint: free-text chief complaint string. Matched against
            the Manchester Triage red-flag patterns to determine ESI floor.
        vital_signs: optional dict with keys among {systolic_bp, heart_rate,
            respiratory_rate, spo2, temperature, consciousness}. Abnormal
            values further push the ESI floor down (more urgent).
        age: patient age in years; affects pediatric / geriatric defaults.
        comorbidity_summary: free-text summary of relevant comorbidities
            (used for rationale, not scoring).
        patient_id: opaque ID for audit (does not affect the decision).

    Returns:
        AdmissionTriage with esi_level, priority, disposition,
        recommended_unit, red_flag_findings, rationale.
    """
    _resolved, _sharp_bound, _ = await harden_clinical_inputs(
        {"age": age}, chart_derivable={"age"},
    )
    if _sharp_bound:
        age = _resolved.get("age")
        # vital_signs: when SHARP bound, prefer chart-derived dict over
        # caller-supplied values (which the agent may have fabricated).
        from ._chart_inputs import (
            extract_chart_vitals as _vc, fetch_patient_bundle as _fb,
            resolve_patient_id as _rp,
        )
        try:
            _b = await _fb(await _rp(None))
            _chart_vs = _vc(_b)
            if _chart_vs:
                # Tool's internal logic accepts dict[str, value]
                vital_signs = {
                    k: v for k, v in _chart_vs.items()
                    if k in {"systolic_bp", "diastolic_bp", "heart_rate",
                              "respiratory_rate", "spo2", "temperature_c",
                              "glasgow_coma_scale"}
                }
        except Exception:
            pass

    if not chief_complaint or not isinstance(chief_complaint, str):
        return AdmissionTriage(
            patient_id=patient_id,
            chief_complaint=str(chief_complaint or ""),
            esi_level=3,
            priority="urgent",
            disposition="ed_workup_then_dispose",
            recommended_unit="ed_treatment",
            red_flag_findings=[],
            rationale="Chief complaint missing or invalid -- defaulting to "
                       "urgent (ESI 3) for clinician review.",
            abstain_recommended=True,
            abstain_reason="missing_chief_complaint",
        )

    red_flags: list[str] = []
    esi_floor: int = 5  # least urgent by default

    for pat, floor, finding in _RED_FLAG_RULES:
        if pat.search(chief_complaint):
            red_flags.append(finding)
            if floor < esi_floor:
                esi_floor = floor

    # Accept both shapes: dict (legacy / direct caller) and list of
    # `{type, value, ...}` dicts (the FHIR-overlay shape used by NEWS2).
    # Internally everything below works on the dict-shape, so the list
    # form is collapsed by taking the most recent value per type.
    if isinstance(vital_signs, list):
        vitals: dict[str, Any] = {}
        for item in vital_signs:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            v = item.get("value")
            if isinstance(t, str) and v is not None:
                # Later entries overwrite earlier ones; the dispatcher
                # ships the most-recent vital-set, so this is safe.
                vitals[t] = v
    elif isinstance(vital_signs, dict):
        vitals = vital_signs
    else:
        vitals = {}
    for floor, finding in _vitals_red_flags(vitals):
        red_flags.append(finding)
        if floor < esi_floor:
            esi_floor = floor

    # Without any red flag and without abnormal vitals we still default to
    # ESI 4 unless the chief complaint clearly screams "non-urgent" -- but
    # better to bias slightly toward urgent than to send people home.
    if esi_floor == 5 and not red_flags:
        esi_floor = 4

    priority, disposition, unit = _esi_to_decision(esi_floor, age)

    abstain = False
    abstain_reason: str | None = None
    if esi_floor in (1, 2) and not vital_signs:
        abstain = True
        abstain_reason = "no_vitals_for_high_acuity_complaint"

    rationale = _build_rationale(
        chief_complaint, esi_floor, priority, disposition, red_flags,
        age, comorbidity_summary,
    )

    return AdmissionTriage(
        patient_id=patient_id,
        chief_complaint=chief_complaint,
        esi_level=esi_floor,
        priority=priority,  # type: ignore[arg-type]
        disposition=disposition,  # type: ignore[arg-type]
        recommended_unit=unit,  # type: ignore[arg-type]
        red_flag_findings=red_flags,
        rationale=rationale,
        abstain_recommended=abstain,
        abstain_reason=abstain_reason,
    )


def _build_rationale(complaint: str, esi: int, priority: str, disposition: str,
                      flags: list[str], age: int | None,
                      comorbidities: str | None) -> str:
    parts = [f"ESI level {esi} ({priority}). Disposition: {disposition}."]
    if flags:
        parts.append("Red-flag findings: " + "; ".join(flags) + ".")
    else:
        parts.append("No red-flag findings on chief complaint or vitals.")
    if age is not None:
        if age < 18:
            parts.append(f"Pediatric patient ({age}y) -- lower threshold "
                          f"for observation/admission.")
        elif age >= 75:
            parts.append(f"Older adult ({age}y) -- observation is preferred "
                          f"over rapid discharge at ESI 3.")
    if comorbidities:
        parts.append(f"Comorbidity context: {comorbidities}")
    return " ".join(parts)


def _as_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip().split()[0])
        except (ValueError, IndexError):
            return None
    return None


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_admission_triage)
