"""healthcare.compute_care_gap_detector -- LIB-3.

Scans a FHIR Patient Bundle against USPSTF / CDC ACIP / ADA / ACC/AHA
schedules to surface care gaps (overdue screenings, missing vaccinations,
chronic-disease monitoring lapses) ranked by clinical priority.

The schedules are encoded as `_GAP_DEFINITIONS` below -- a curated subset
of the highest-prevalence preventive items. Each entry traces to its
authoritative recommendation source.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from shared.schemas import (
    CareGap,
    CareGapReport,
)

_HL7_UNKNOWN_SENTINELS: frozenset[str] = frozenset({"unknown", "u", "unk", ""})


# ─────────────────────── Gap definitions ───────────────────────
#
# Each entry: gap_id, title, domain, age_range, sex_specific,
# recommended_interval_days, ICD-10 / SNOMED / LOINC / CVX hints,
# priority, recommendation, source URL.

_GAP_DEFINITIONS: list[dict[str, Any]] = [
    # ─── USPSTF cancer screenings ───
    {
        "gap_id": "uspstf_breast_cancer_screening",
        "title": "Mammography",
        "domain": "cancer_screening",
        "age_range": (40, 74),
        "sex_specific": "female",
        "interval_days": 730,
        "matchers": {"loinc": ["24606-6"],
                       "snomed": ["241055006"],
                       "name_keywords": ["mammogram", "mammography",
                                              "breast screening"]},
        "priority": "high",
        "recommendation": "Biennial mammography for women 40-74 (USPSTF "
                              "2024 Grade B).",
        "reference": "U.S. Preventive Services Task Force 2024 -- "
                        "Breast Cancer Screening Recommendation.",
    },
    {
        "gap_id": "uspstf_colon_cancer_screening",
        "title": "Colorectal cancer screening (colonoscopy / FIT / Cologuard)",
        "domain": "cancer_screening",
        "age_range": (45, 75),
        "sex_specific": "any",
        "interval_days": 3650,  # colonoscopy 10y or FIT yearly
        "matchers": {"loinc": ["29771-3"],
                       "snomed": ["73761001", "446745002"],
                       "name_keywords": ["colonoscopy", "fit test",
                                              "fecal immunochemical"]},
        "priority": "high",
        "recommendation": "Colorectal cancer screening every 10y "
                              "(colonoscopy) or every 1y (FIT) for adults 45-75.",
        "reference": "USPSTF 2021 -- Colorectal Cancer Screening Grade A.",
    },
    {
        "gap_id": "uspstf_cervical_cancer_screening",
        "title": "Cervical cancer screening (Pap / HPV co-test)",
        "domain": "cancer_screening",
        "age_range": (21, 65),
        "sex_specific": "female",
        "interval_days": 1095,  # Pap every 3y
        "matchers": {"loinc": ["10524-7"],
                       "name_keywords": ["pap smear", "pap test",
                                              "cervical cytology", "hpv test"]},
        "priority": "moderate",
        "recommendation": "Cervical cancer screening every 3y (Pap) or "
                              "every 5y (HPV co-test) for women 21-65.",
        "reference": "USPSTF 2018 -- Cervical Cancer Screening Grade A.",
    },
    {
        "gap_id": "uspstf_lung_cancer_screening",
        "title": "Low-dose CT chest (lung cancer screening)",
        "domain": "cancer_screening",
        "age_range": (50, 80),
        "sex_specific": "any",
        "interval_days": 365,
        "matchers": {"loinc": ["24632-2"],
                       "name_keywords": ["low-dose ct chest", "ldct chest",
                                              "lung cancer screening"]},
        "priority": "moderate",
        "recommendation": "Annual LDCT in adults 50-80 with ≥ 20 pack-year "
                              "history who currently smoke or quit < 15y ago.",
        "reference": "USPSTF 2021 -- Lung Cancer Screening Grade B.",
    },
    # ─── CDC vaccinations ───
    {
        "gap_id": "cdc_influenza_vaccine",
        "title": "Annual influenza vaccination",
        "domain": "vaccination",
        "age_range": (1, 130),
        "sex_specific": "any",
        "interval_days": 365,
        "matchers": {"cvx": ["140", "141", "150", "155", "158", "168",
                              "171", "166", "186"],
                       "name_keywords": ["influenza", "flu vaccine"]},
        "priority": "high",
        "recommendation": "Annual influenza vaccine for everyone ≥ 6 months.",
        "reference": "CDC ACIP -- Influenza Vaccination Recommendations.",
    },
    {
        "gap_id": "cdc_pneumococcal_vaccine",
        "title": "Pneumococcal vaccination (PCV20 / PPSV23)",
        "domain": "vaccination",
        "age_range": (65, 130),
        "sex_specific": "any",
        "interval_days": 3650,
        "matchers": {"cvx": ["33", "100", "133", "152", "215", "216"],
                       "name_keywords": ["pneumococcal", "pneumovax",
                                              "prevnar"]},
        "priority": "high",
        "recommendation": "PCV20 once for adults ≥ 65 (or PCV15 + PPSV23 ≥ "
                              "1 year later).",
        "reference": "CDC ACIP 2024 -- Pneumococcal Vaccine Recommendations.",
    },
    {
        "gap_id": "cdc_shingrix_vaccine",
        "title": "Recombinant zoster vaccine (Shingrix, 2-dose)",
        "domain": "vaccination",
        "age_range": (50, 130),
        "sex_specific": "any",
        "interval_days": 36500,  # essentially once-in-a-lifetime
        "matchers": {"cvx": ["187"],
                       "name_keywords": ["shingrix", "zoster", "shingles "
                                              "vaccine"]},
        "priority": "moderate",
        "recommendation": "RZV (Shingrix) 2-dose series for adults ≥ 50.",
        "reference": "CDC ACIP -- Shingles (Herpes Zoster) Recommendations.",
    },
    {
        "gap_id": "cdc_tdap_booster",
        "title": "Tdap booster (every 10y)",
        "domain": "vaccination",
        "age_range": (19, 130),
        "sex_specific": "any",
        "interval_days": 3650,
        "matchers": {"cvx": ["115", "138", "139"],
                       "name_keywords": ["tdap", "tetanus", "pertussis"]},
        "priority": "moderate",
        "recommendation": "Tdap booster every 10 years.",
        "reference": "CDC ACIP -- Tdap booster recommendation.",
    },
    {
        "gap_id": "cdc_covid19_vaccine",
        "title": "COVID-19 vaccine (current annual formulation)",
        "domain": "vaccination",
        "age_range": (1, 130),
        "sex_specific": "any",
        "interval_days": 365,
        "matchers": {"cvx": ["207", "208", "210", "212", "217", "218",
                              "219", "227", "229", "300", "308", "309"],
                       "name_keywords": ["covid", "sars-cov-2"]},
        "priority": "moderate",
        "recommendation": "Annual COVID-19 vaccine per CDC current "
                              "season recommendation.",
        "reference": "CDC ACIP -- COVID-19 Vaccination Recommendations.",
    },
    # ─── ADA / chronic disease monitoring ───
    {
        "gap_id": "ada_a1c_quarterly",
        "title": "Hemoglobin A1c (every 3 months in active diabetes)",
        "domain": "metabolic_monitoring",
        "age_range": (1, 130),
        "sex_specific": "any",
        "interval_days": 90,
        "matchers": {"loinc": ["4548-4"],
                       "name_keywords": ["hba1c", "hemoglobin a1c"]},
        "priority": "high",
        "recommendation": "HbA1c every 3 months when not at goal; every 6 "
                              "months when stable at goal.",
        "reference": "American Diabetes Association Standards of Care "
                        "2024 -- Glycemic targets.",
        "applies_only_with_condition_prefix": ["E10", "E11", "E13"],
    },
    {
        "gap_id": "lipid_panel_5y",
        "title": "Lipid panel (every 4-6 years)",
        "domain": "cardiovascular_screening",
        "age_range": (40, 75),
        "sex_specific": "any",
        "interval_days": 1825,
        "matchers": {"loinc": ["57698-3", "2093-3"],
                       "name_keywords": ["lipid panel", "cholesterol panel"]},
        "priority": "moderate",
        "recommendation": "Lipid panel every 4-6 years for ASCVD risk "
                              "estimation.",
        "reference": "ACC/AHA 2018 -- Cholesterol Management Guideline.",
    },
    {
        "gap_id": "uspstf_aaa_screening",
        "title": "Abdominal aortic aneurysm screening (one-time US)",
        "domain": "preventive_screening",
        "age_range": (65, 75),
        "sex_specific": "male",
        "interval_days": 36500,
        "matchers": {"loinc": ["55409-6"],
                       "name_keywords": ["abdominal aortic", "aaa screening",
                                              "abdominal ultrasound aorta"]},
        "priority": "moderate",
        "recommendation": "One-time AAA ultrasound for men 65-75 who have "
                              "ever smoked.",
        "reference": "USPSTF 2019 -- AAA Screening Grade B.",
    },
]


# ─────────────────────── FHIR Bundle parsing helpers ───────────────────────

def _entries(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    return [e.get("resource") or {} for e in bundle.get("entry", []) or []
            if isinstance(e, dict)]


def _patient_demographics(bundle: dict[str, Any]
                              ) -> tuple[int | None, str | None]:
    for r in _entries(bundle):
        if (r.get("resourceType") or "").lower() == "patient":
            age = None
            sex = None
            bd = r.get("birthDate")
            if isinstance(bd, str) and len(bd) >= 4:
                try:
                    dob = datetime.fromisoformat(bd[:10])
                    age = (datetime.now(timezone.utc).date()
                              - dob.date()).days // 365
                except Exception:
                    pass
            g = r.get("gender")
            if isinstance(g, str):
                sex = g.lower()
            return age, sex
    return None, None


def _icd10_codes_in_bundle(bundle: dict[str, Any]) -> set[str]:
    codes: set[str] = set()
    for r in _entries(bundle):
        if (r.get("resourceType") or "").lower() != "condition":
            continue
        cc = r.get("code") or {}
        for coding in cc.get("coding", []) or []:
            sys = (coding.get("system") or "").lower()
            if "icd-10" in sys or "icd10" in sys:
                code = (coding.get("code") or "").strip().upper()
                if code:
                    codes.add(code.replace(".", ""))
    return codes


def _matches_observation(res: dict[str, Any],
                            matchers: dict[str, Any]) -> bool:
    if (res.get("resourceType") or "").lower() != "observation":
        return False
    cc = res.get("code") or {}
    text = (cc.get("text") or "").lower()
    keywords = matchers.get("name_keywords", [])
    if any(k in text for k in keywords):
        return True
    for coding in cc.get("coding", []) or []:
        code = (coding.get("code") or "").strip()
        sys = (coding.get("system") or "").lower()
        if "loinc" in sys and code in matchers.get("loinc", []):
            return True
        if "snomed" in sys and code in matchers.get("snomed", []):
            return True
        disp = (coding.get("display") or "").lower()
        if any(k in disp for k in keywords):
            return True
    return False


def _matches_immunization(res: dict[str, Any],
                              matchers: dict[str, Any]) -> bool:
    if (res.get("resourceType") or "").lower() != "immunization":
        return False
    vc = res.get("vaccineCode") or {}
    text = (vc.get("text") or "").lower()
    keywords = matchers.get("name_keywords", [])
    if any(k in text for k in keywords):
        return True
    for coding in vc.get("coding", []) or []:
        code = (coding.get("code") or "").strip()
        sys = (coding.get("system") or "").lower()
        if "cvx" in sys and code in matchers.get("cvx", []):
            return True
        disp = (coding.get("display") or "").lower()
        if any(k in disp for k in keywords):
            return True
    return False


def _matches_procedure(res: dict[str, Any],
                          matchers: dict[str, Any]) -> bool:
    if (res.get("resourceType") or "").lower() != "procedure":
        return False
    cc = res.get("code") or {}
    text = (cc.get("text") or "").lower()
    keywords = matchers.get("name_keywords", [])
    if any(k in text for k in keywords):
        return True
    for coding in cc.get("coding", []) or []:
        code = (coding.get("code") or "").strip()
        if code in matchers.get("loinc", []):
            return True
        if code in matchers.get("snomed", []):
            return True
        disp = (coding.get("display") or "").lower()
        if any(k in disp for k in keywords):
            return True
    return False


def _resource_date(res: dict[str, Any]) -> str | None:
    for key in ("effectiveDateTime", "occurrenceDateTime", "performedDateTime"):
        v = res.get(key)
        if isinstance(v, str):
            return v
    period = res.get("period") or {}
    return period.get("start") if isinstance(period, dict) else None


def _last_seen_for_gap(bundle: dict[str, Any],
                          matchers: dict[str, Any]
                          ) -> str | None:
    last: str | None = None
    for r in _entries(bundle):
        is_match = (
            _matches_observation(r, matchers)
            or _matches_immunization(r, matchers)
            or _matches_procedure(r, matchers)
        )
        if not is_match:
            continue
        d = _resource_date(r)
        if d is None:
            continue
        if last is None or d > last:
            last = d
    return last


def _overdue_days(last_seen_iso: str | None,
                     interval_days: int) -> int | None:
    if last_seen_iso is None:
        return None
    try:
        s = last_seen_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s[:25] if len(s) > 25 else s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None
    delta = (datetime.now(timezone.utc) - dt).days
    return max(0, delta - interval_days)


# ─────────────────────── Public API ───────────────────────

def _is_hl7_unknown(value: Any) -> bool:
    """Return True when value carries the HL7 'unknown' sentinel.

    Recognises 'U', 'UNK', 'unknown', and empty string. None means absent
    (not unknown), so None returns False.
    """
    if value is None:
        return False
    return str(value).strip().lower() in _HL7_UNKNOWN_SENTINELS


async def compute_care_gap_detector(
    fhir_bundle: dict[str, Any],
    patient_age: int | None = None,
    patient_sex: str | None = None,
) -> CareGapReport:
    """Detect care gaps in a patient FHIR Bundle.

    Args:
        fhir_bundle: FHIR R4 Bundle dict (collection or transaction).
        patient_age: optional override (otherwise inferred from Patient).
            Pass None to use the FHIR bundle's birthDate. Passing an integer
            value of 0, or a string sentinel ('unknown', 'U', 'unk') triggers
            abstain because age is critical for every gap-age-range check.
        patient_sex: optional override (male / female / any). Passing an HL7
            unknown sentinel triggers abstain for the sex-specific gap checks.

    Returns:
        CareGapReport with gaps grouped by priority. When a critical
        demographic field carries an HL7 unknown sentinel,
        abstain_recommended=True and all gap lists are empty.
    """
    if not isinstance(fhir_bundle, dict):
        raise ValueError("fhir_bundle must be a dict.")

    # Detect HL7 unknown sentinels on explicit overrides before FHIR inference.
    # These checks fire only when the caller explicitly passes an unknown-valued
    # override; absent parameters (None) fall through to FHIR bundle inference.
    unknown_fields: list[str] = []
    if patient_age is not None and (
        _is_hl7_unknown(patient_age) or int(patient_age) == 0
    ):
        unknown_fields.append("patient_age")
    if patient_sex is not None and _is_hl7_unknown(patient_sex):
        unknown_fields.append("patient_sex")

    if unknown_fields:
        reason = (
            "missing_critical_demographic: "
            + ", ".join(
                f"{f} required for compute_care_gap_detector but source value "
                "is 'unknown' or absent. The HL7 'unknown' sentinel cannot be "
                "substituted with a population mean."
                for f in unknown_fields
            )
        )
        return CareGapReport(
            patient_age=None,
            patient_sex=None,
            n_gaps_found=0,
            high_priority_gaps=[],
            moderate_priority_gaps=[],
            low_priority_gaps=[],
            rationale=(
                f"Care gap detection aborted: {reason} "
                "Age and sex are required to apply USPSTF/CDC age-range and "
                "sex-specific gap eligibility criteria."
            ),
            abstain_recommended=True,
            abstain_reason=reason,
        )

    age, sex = _patient_demographics(fhir_bundle)
    if patient_age is not None:
        age = patient_age
    if patient_sex is not None:
        sex = patient_sex

    icd_codes = _icd10_codes_in_bundle(fhir_bundle)

    high: list[CareGap] = []
    moderate: list[CareGap] = []
    low: list[CareGap] = []

    for definition in _GAP_DEFINITIONS:
        # Age check
        if age is None:
            continue
        a_lo, a_hi = definition["age_range"]
        if not (a_lo <= age <= a_hi):
            continue

        # Sex check
        ss = definition.get("sex_specific", "any")
        if ss != "any" and sex != ss:
            continue

        # Condition gating (e.g. A1c only applies if patient has diabetes)
        cond_prefixes = definition.get("applies_only_with_condition_prefix")
        if cond_prefixes:
            if not any(any(c.startswith(p) for c in icd_codes)
                          for p in cond_prefixes):
                continue

        last_seen = _last_seen_for_gap(fhir_bundle, definition["matchers"])
        overdue = _overdue_days(last_seen, definition["interval_days"])

        # Gap fires if: never seen, OR overdue
        if last_seen is None or overdue is not None and overdue > 0:
            gap = CareGap(
                gap_id=definition["gap_id"],
                title=definition["title"],
                domain=definition["domain"],          # type: ignore[arg-type]
                age_range=definition["age_range"],
                sex_specific=ss,                       # type: ignore[arg-type]
                last_seen_iso=last_seen,
                overdue_days=overdue,
                priority=definition["priority"],       # type: ignore[arg-type]
                recommendation=definition["recommendation"],
                references=[definition["reference"]],
            )
            if gap.priority == "high":
                high.append(gap)
            elif gap.priority == "moderate":
                moderate.append(gap)
            else:
                low.append(gap)

    n_total = len(high) + len(moderate) + len(low)
    rationale = (
        f"Patient age {age}, sex {sex}, {len(icd_codes)} ICD-10 codes. "
        f"Found {n_total} gap(s): high={len(high)}, "
        f"moderate={len(moderate)}, low={len(low)}."
    )

    return CareGapReport(
        patient_age=age,
        patient_sex=sex,
        n_gaps_found=n_total,
        high_priority_gaps=high,
        moderate_priority_gaps=moderate,
        low_priority_gaps=low,
        rationale=rationale,
    )


def register(mcp) -> None:
    mcp.tool()(compute_care_gap_detector)
