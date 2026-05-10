"""healthcare.compute_medication_reconciliation -- discharge med safety check.

Per AHRQ MATCH (Medications At Transitions and Clinical Handoffs) toolkit,
plus the synthesized clinical reference at workflows/grounding-corpus-build/
data/sources/medication_reconciliation.md (in the W3 corpus).

The tool answers two distinct questions:

  1. **What changed?** -- diff admission med-list vs discharge med-list,
     surfacing additions, removals, and dose changes.
  2. **Is the discharge plan safe?** -- for each high-risk drug class on the
     discharge list, check that the matching monitoring observation exists
     within `monitoring_window_hours` (default 48).

Sources of truth:
  - If `admission_meds` and `discharge_meds` are provided explicitly, the
    tool uses those (stateless mode -- useful for in-app testing).
  - Otherwise the tool fetches the patient bundle from FHIR and partitions
    its `MedicationRequest` resources by encounter timing.

The result is a structured `MedReconReport` that downstream agents (or the
A2A self-critique stage) can fold into the DecisionCard's validation block.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable

from shared.schemas import (
    Medication,
    MedicationConcern,
    MedReconReport,
)

from ..fhir.client import fetch_patient_bundle, resolve_patient_id


# ─────────────────────────────────────────────────────────────────────
# Drug-class taxonomy + monitoring requirements
# ─────────────────────────────────────────────────────────────────────
# Maps a name keyword (case-insensitive substring) to a drug class. The
# taxonomy is intentionally small -- extending it is part of the future
# calibration workflow.

_NAME_TO_CLASS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"warfarin", re.IGNORECASE), "anticoagulant_vka"),
    (re.compile(r"apixaban|rivaroxaban|dabigatran|edoxaban|eliquis|xarelto|pradaxa|savaysa",
                re.IGNORECASE), "anticoagulant_doac"),
    (re.compile(r"\bheparin\b|enoxaparin|lovenox", re.IGNORECASE), "anticoagulant_heparin"),
    (re.compile(r"insulin\b|lispro|aspart|glargine|detemir|degludec|humalog|novolog|lantus",
                re.IGNORECASE), "insulin"),
    (re.compile(r"empagliflozin|dapagliflozin|canagliflozin|jardiance|farxiga|invokana",
                re.IGNORECASE), "sglt2_inhibitor"),
    (re.compile(r"metformin|glucophage", re.IGNORECASE), "biguanide"),
    (re.compile(r"lisinopril|enalapril|ramipril|captopril", re.IGNORECASE), "ace_inhibitor"),
    (re.compile(r"losartan|valsartan|irbesartan|candesartan|olmesartan", re.IGNORECASE),
     "arb"),
    (re.compile(r"metoprolol|carvedilol|bisoprolol|atenolol|propranolol", re.IGNORECASE),
     "beta_blocker"),
    (re.compile(r"spironolactone|eplerenone", re.IGNORECASE), "mra"),
    (re.compile(r"furosemide|lasix|torsemide|bumetanide", re.IGNORECASE), "loop_diuretic"),
    (re.compile(r"tacrolimus|cyclosporine|sirolimus|everolimus|mycophenolate",
                re.IGNORECASE), "immunosuppressant"),
    (re.compile(r"oxycodone|hydrocodone|morphine|fentanyl|tramadol|codeine|hydromorphone",
                re.IGNORECASE), "opioid"),
]

# Class -> monitoring requirements. Each entry lists observation-code substrings
# (case-insensitive) that satisfy the monitoring contract. Severity escalates
# the concern when monitoring is absent.
_CLASS_MONITORING: dict[str, dict[str, Any]] = {
    "anticoagulant_vka": {
        "monitoring_substrings": ["inr", "pt"],
        "severity_if_absent": "high",
        "rationale": (
            "Warfarin requires INR within target range at discharge. Without "
            "a recent INR, the dose at discharge may be sub- or supra-therapeutic."
        ),
    },
    "anticoagulant_doac": {
        "monitoring_substrings": ["creatinine", "egfr", "renal", "anti-xa", "anti xa"],
        "severity_if_absent": "high",
        "rationale": (
            "Direct oral anticoagulants are renally cleared; eGFR / creatinine "
            "must be current at discharge. Acute kidney injury during admission "
            "shifts dosing."
        ),
    },
    "anticoagulant_heparin": {
        "monitoring_substrings": ["aptt", "anti-xa", "anti xa", "platelet"],
        "severity_if_absent": "high",
        "rationale": (
            "Heparin requires aPTT or anti-Xa monitoring; LMWH transition "
            "requires platelet count to rule out HIT."
        ),
    },
    "insulin": {
        "monitoring_substrings": ["glucose", "fingerstick", "hba1c", "a1c"],
        "severity_if_absent": "high",
        "rationale": (
            "Insulin regimens (especially basal/bolus) require recent glucose "
            "monitoring. Hypoglycemia at home discharge is a known readmission cause."
        ),
    },
    "sglt2_inhibitor": {
        "monitoring_substrings": ["creatinine", "egfr", "glucose", "ketone"],
        "severity_if_absent": "medium",
        "rationale": (
            "SGLT2-inhibitors require recent renal function. Euglycemic DKA "
            "awareness counseling assumed but unverifiable from bundle."
        ),
    },
    "ace_inhibitor": {
        "monitoring_substrings": ["potassium", "creatinine", "k+", "egfr"],
        "severity_if_absent": "medium",
        "rationale": (
            "ACE-I require recent K+ and renal function -- hyperkalemia and "
            "AKI are common readmission triggers."
        ),
    },
    "arb": {
        "monitoring_substrings": ["potassium", "creatinine", "k+", "egfr"],
        "severity_if_absent": "medium",
        "rationale": (
            "ARBs require recent K+ and renal function (same risk profile as ACE-I)."
        ),
    },
    "beta_blocker": {
        "monitoring_substrings": ["heart rate", "hr ", "ecg", "ekg"],
        "severity_if_absent": "low",
        "rationale": (
            "Beta-blockers in heart failure require recent heart rate / ECG "
            "evidence the dose is tolerated."
        ),
    },
    "mra": {
        "monitoring_substrings": ["potassium", "creatinine", "k+", "egfr"],
        "severity_if_absent": "medium",
        "rationale": (
            "MRAs (spironolactone / eplerenone) require recent K+ and renal "
            "function -- hyperkalemia is the leading discontinuation cause."
        ),
    },
    "loop_diuretic": {
        "monitoring_substrings": ["weight", "potassium", "k+", "creatinine", "egfr"],
        "severity_if_absent": "medium",
        "rationale": (
            "Loop diuretics in heart failure require recent weight + electrolytes "
            "to titrate at home."
        ),
    },
    "immunosuppressant": {
        "monitoring_substrings": ["tacrolimus level", "cyclosporine level",
                                   "sirolimus level", "everolimus level", "mpa", "drug level"],
        "severity_if_absent": "high",
        "rationale": (
            "Immunosuppressants require therapeutic drug monitoring. Missed "
            "first follow-up is a known readmission trigger for transplant patients."
        ),
    },
    "opioid": {
        "monitoring_substrings": [],  # no observation can satisfy; we flag separately
        "severity_if_absent": "medium",
        "rationale": (
            "Opioid prescribing at discharge requires PDMP query + naloxone "
            "co-prescription where state law allows. Cannot verify from FHIR alone."
        ),
        "always_flag": True,
    },
}


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_medication_reconciliation(
    patient_id: str | None = None,
    admission_meds: list[dict[str, Any]] | None = None,
    discharge_meds: list[dict[str, Any]] | None = None,
    monitoring_window_hours: int = 48,
) -> MedReconReport:
    """Diff admission vs discharge med-list and audit the discharge contract.

    Args:
        patient_id: FHIR Patient ID. If None, uses the X-Patient-ID header.
        admission_meds: optional explicit list, bypasses FHIR fetch.
        discharge_meds: optional explicit list, bypasses FHIR fetch.
        monitoring_window_hours: how recent an observation must be to count
            as satisfying a class's monitoring contract. Default 48h.

    Returns:
        MedReconReport with added/removed/dose_changed sets, plus concerns
        for any high-risk class lacking required monitoring.
    """
    bundle: dict[str, Any] | None = None
    if admission_meds is None or discharge_meds is None:
        pid = await resolve_patient_id(patient_id)
        bundle = await fetch_patient_bundle(pid)

    if admission_meds is None:
        admission_meds = _extract_meds_from_bundle(bundle, phase="admission")
    if discharge_meds is None:
        discharge_meds = _extract_meds_from_bundle(bundle, phase="discharge")

    # Normalize to Medication objects with class tags
    adm = [_normalize_med(m) for m in admission_meds]
    dis = [_normalize_med(m) for m in discharge_meds]

    # noqa: ABSTAIN-GUARD -- fail-fast when both med lists are empty after fetch.
    # If neither the caller supplied medications nor the FHIR bundle contained
    # any MedicationRequest resources, the reconciliation diff is meaningless:
    # added/removed/dose_changed would all be empty and no monitoring concerns
    # could be raised.  A downstream orchestrator must not treat an empty
    # MedReconReport as a clean bill of health.
    if not adm and not dis:
        return MedReconReport(
            added=[],
            removed=[],
            dose_changed=[],
            concerns=[],
            severity_counts={"high": 0, "medium": 0, "low": 0},
            discharge_contract_satisfied=False,
            n_admission_meds=0,
            n_discharge_meds=0,
            monitoring_window_hours=monitoring_window_hours,
            abstain_recommended=True,
            abstain_reason=(
                "missing_admission_meds_and_discharge_meds: "
                "compute_medication_reconciliation requires at least one "
                "medication in either the admission or discharge list. "
                "No demo fallback in live mode."
            ),
        )

    added = _diff_by_name(dis, adm)
    removed = _diff_by_name(adm, dis)
    dose_changed = _dose_changes(adm, dis)

    # Audit each discharge med for monitoring contract
    obs_index = _build_observation_index(bundle, monitoring_window_hours)
    concerns = _audit_monitoring(dis, obs_index)

    # Polypharmacy concern (≥5 high-risk classes on discharge list)
    high_risk_count = sum(1 for m in dis if m.drug_class and _is_high_risk_class(m.drug_class))
    if high_risk_count >= 5:
        concerns.append(MedicationConcern(
            medication_name="(polypharmacy)",
            drug_class="multiple_high_risk",
            severity="medium",
            concern_type="polypharmacy_high_risk",
            detail=(
                f"{high_risk_count} high-risk medications at discharge. AHRQ "
                "RED toolkit flags >5 high-risk meds as a discharge-disposition "
                "trigger (consider home_with_care over home, see corpus §3.2)."
            ),
        ))

    severity_counts = {
        "high": sum(1 for c in concerns if c.severity == "high"),
        "medium": sum(1 for c in concerns if c.severity == "medium"),
        "low": sum(1 for c in concerns if c.severity == "low"),
    }
    contract_ok = severity_counts["high"] == 0 and severity_counts["medium"] <= 1

    return MedReconReport(
        added=added,
        removed=removed,
        dose_changed=dose_changed,
        concerns=concerns,
        severity_counts=severity_counts,
        discharge_contract_satisfied=contract_ok,
        n_admission_meds=len(adm),
        n_discharge_meds=len(dis),
        monitoring_window_hours=monitoring_window_hours,
    )


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

_HIGH_RISK_CLASSES = {
    "anticoagulant_vka", "anticoagulant_doac", "anticoagulant_heparin",
    "insulin", "sglt2_inhibitor", "immunosuppressant", "opioid",
    "ace_inhibitor", "arb", "mra",
}


def _is_high_risk_class(cls: str) -> bool:
    return cls in _HIGH_RISK_CLASSES


def _classify_drug(name: str) -> str | None:
    for pattern, cls in _NAME_TO_CLASS:
        if pattern.search(name):
            return cls
    return None


def _normalize_med(m: dict[str, Any]) -> Medication:
    """Coerce a FHIR-ish or simple dict into a Medication, attaching a drug_class."""
    if not isinstance(m, dict):
        m = {"name": str(m)}
    name = str(m.get("name") or m.get("medication") or m.get("display") or "").strip()
    cls = m.get("drug_class") or _classify_drug(name)
    return Medication(
        name=name,
        dose=str(m["dose"]) if m.get("dose") is not None else None,
        route=str(m["route"]) if m.get("route") is not None else None,
        rxnorm_code=str(m["rxnorm_code"]) if m.get("rxnorm_code") else None,
        drug_class=cls,
        status=str(m.get("status") or "active"),  # type: ignore[arg-type]
    )


def _extract_meds_from_bundle(bundle: dict[str, Any] | None, *, phase: str) -> list[dict[str, Any]]:
    """Pull MedicationRequest entries from a FHIR Bundle.

    `phase` is "admission" or "discharge"; Synthea-style bundles don't always
    distinguish these explicitly, so we use a heuristic on Encounter.period
    boundaries when available, else split by status (admission = stopped or
    completed, discharge = active).
    """
    if not isinstance(bundle, dict):
        return []
    entries = bundle.get("entry") or []
    meds: list[dict[str, Any]] = []
    for e in entries:
        res = e.get("resource") if isinstance(e, dict) else None
        if not isinstance(res, dict) or res.get("resourceType") != "MedicationRequest":
            continue
        med = _med_request_to_dict(res)
        status = (res.get("status") or "active").lower()
        if phase == "admission":
            # Meds the patient was on at admission appear in FHIR as either
            # already-stopped (transient inpatient meds) or as active
            # prescriptions that pre-date this encounter. The heuristic is
            # not perfect -- production should anchor against Encounter.period
            # and MedicationRequest.authoredOn -- but it works for stub fixtures
            # and the W2 demo bundles.
            if status in ("stopped", "completed"):
                meds.append(med)
        else:  # discharge
            if status == "active":
                meds.append(med)
    return meds


def _med_request_to_dict(res: dict[str, Any]) -> dict[str, Any]:
    """Pull display/code/dose from a FHIR MedicationRequest resource.

    Three R4 shapes are handled, in priority order:

      1. inline `medicationCodeableConcept.text` /
         `medicationCodeableConcept.coding[].display`
      2. inline `medication.text` (legacy DSTU2/STU3 shape)
      3. external `medicationReference.reference` -- servers that do
         not expose `patient/Medication.rs` in the OAuth scope (e.g.
         Prompt Opinion's workspace FHIR) leave the human name
         unresolvable. We surface the reference itself as the name so
         the medication is not silently dropped from the diff; downstream
         tools can detect the unresolved shape (name starts with
         "Medication/") and either skip or render an honest placeholder.
    """
    name = ""
    code_obj = res.get("medicationCodeableConcept") or res.get("medication") or {}
    if isinstance(code_obj, dict):
        name = (code_obj.get("text") or "").strip()
        if not name:
            for c in code_obj.get("coding", []) or []:
                if isinstance(c, dict):
                    disp = c.get("display") or ""
                    if disp:
                        name = disp.strip()
                        break
    if not name:
        ref_obj = res.get("medicationReference") or {}
        if isinstance(ref_obj, dict):
            ref = (ref_obj.get("reference") or "").strip()
            if ref:
                name = ref  # e.g. "Medication/9ee3a897-..."
    rxnorm = None
    if isinstance(code_obj, dict):
        for c in code_obj.get("coding", []) or []:
            if isinstance(c, dict) and "rxnorm" in (c.get("system") or "").lower():
                rxnorm = c.get("code")
                break

    dose = None
    di = res.get("dosageInstruction") or []
    if di and isinstance(di, list) and isinstance(di[0], dict):
        dose = di[0].get("text") or None
        if not dose:
            doseAndRate = di[0].get("doseAndRate") or []
            if doseAndRate and isinstance(doseAndRate[0], dict):
                doseQuantity = doseAndRate[0].get("doseQuantity") or {}
                if doseQuantity.get("value") is not None:
                    dose = f"{doseQuantity.get('value')} {doseQuantity.get('unit', '')}".strip()

    return {
        "name": name,
        "dose": dose,
        "rxnorm_code": rxnorm,
        "status": res.get("status") or "active",
    }


def _diff_by_name(left: Iterable[Medication], right: Iterable[Medication]) -> list[Medication]:
    """Return meds in `left` whose normalized name is not in `right`."""
    right_names = {m.name.lower() for m in right if m.name}
    return [m for m in left if m.name and m.name.lower() not in right_names]


def _dose_changes(adm: list[Medication], dis: list[Medication]) -> list[Medication]:
    """Return discharge meds whose dose changed vs admission (same drug name)."""
    adm_doses: dict[str, str | None] = {m.name.lower(): m.dose for m in adm if m.name}
    out: list[Medication] = []
    for m in dis:
        if not m.name:
            continue
        prev_dose = adm_doses.get(m.name.lower())
        if prev_dose is not None and m.dose is not None and prev_dose != m.dose:
            out.append(m)
    return out


def _build_observation_index(
    bundle: dict[str, Any] | None,
    window_hours: int,
) -> list[tuple[str, datetime | None]]:
    """Return (text_or_code, effectiveDate) tuples for Observations within the
    monitoring window. Used to check whether a class's monitoring contract is
    satisfied at discharge time.
    """
    if not isinstance(bundle, dict):
        return []
    out: list[tuple[str, datetime | None]] = []
    cutoff: datetime | None = None
    if window_hours > 0:
        # We'll filter later -- when the bundle has no "now" anchor, accept all
        cutoff = datetime.now(timezone.utc)
    for e in bundle.get("entry") or []:
        res = e.get("resource") if isinstance(e, dict) else None
        if not isinstance(res, dict) or res.get("resourceType") != "Observation":
            continue
        text_parts: list[str] = []
        code = res.get("code", {}) or {}
        if isinstance(code, dict):
            if code.get("text"):
                text_parts.append(str(code["text"]))
            for c in code.get("coding", []) or []:
                if isinstance(c, dict):
                    if c.get("display"):
                        text_parts.append(str(c["display"]))
                    if c.get("code"):
                        text_parts.append(str(c["code"]))
        text = " | ".join(text_parts).lower()
        eff = _parse_iso_datetime(res.get("effectiveDateTime") or res.get("issued") or "")
        if cutoff is not None and eff is not None:
            age_hours = (cutoff - eff).total_seconds() / 3600.0
            if age_hours > window_hours:
                continue
        out.append((text, eff))
    return out


def _parse_iso_datetime(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _audit_monitoring(
    discharge_meds: list[Medication],
    observations: list[tuple[str, datetime | None]],
) -> list[MedicationConcern]:
    """For each discharge med with a known class, check that the class's
    monitoring contract is satisfied. Emit a MedicationConcern if not."""
    concerns: list[MedicationConcern] = []
    obs_corpus = " ".join(text for text, _ in observations)

    for med in discharge_meds:
        if not med.drug_class:
            continue
        rule = _CLASS_MONITORING.get(med.drug_class)
        if not rule:
            continue

        substrings = rule.get("monitoring_substrings", [])
        always_flag = rule.get("always_flag", False)
        observed: list[str] = []
        if substrings:
            for sub in substrings:
                if sub.lower() in obs_corpus:
                    observed.append(sub)

        # Decide whether to flag
        if always_flag or (substrings and not observed):
            concerns.append(MedicationConcern(
                medication_name=med.name,
                drug_class=med.drug_class,
                severity=rule.get("severity_if_absent", "medium"),  # type: ignore[arg-type]
                concern_type="missing_monitoring",
                detail=str(rule.get("rationale", "")),
                monitoring_required=list(substrings) if substrings else [],
                monitoring_observed=observed,
            ))

    return concerns


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    """Register this tool on the shared FastMCP instance."""
    mcp.tool()(compute_medication_reconciliation)
