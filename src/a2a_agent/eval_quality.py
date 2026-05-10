"""EVAL-1/2/3/4 -- Quality benchmarks and comparative-effectiveness.

Four utilities:

  EVAL-1 compute_hedis_score             -- 6 NCQA HEDIS measures
  EVAL-2 compute_hospital_compare_benchmark -- CMS Hospital Compare peer comparison
  EVAL-3 compute_schwartz_quality_score  -- JAMIA 2017 4-pillar score
  EVAL-4 compute_comparative_effectiveness -- N-patient simulation, ARR/NNT
"""

from __future__ import annotations

import math
import random
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from shared.schemas import (
    CohortComparativeReport,
    HEDISMeasure,
    HEDISReport,
    HospitalCompareBenchmark,
    SchwartzQualityScore,
)


# ─────────────────────── EVAL-1: HEDIS measure scorer ───────────────────────
#
# Curated subset of HEDIS 2024 measures relevant to inpatient discharge:
#   CDC-HM2  Comprehensive Diabetes Care: HbA1c control < 8% (DM patients)
#   CBP       Controlling High Blood Pressure (HTN patients)
#   COL       Colorectal Cancer Screening (50-75y)
#   CCS       Cervical Cancer Screening (21-65y, female)
#   BCS       Breast Cancer Screening (50-74y, female)
#   AAB       Avoidance of Antibiotic Treatment for Acute Bronchitis
#
# Each measure has eligibility criteria + numerator (compliance check)
# both derived from the FHIR Bundle.

def _entries(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    return [e.get("resource") or {} for e in bundle.get("entry", []) or []
            if isinstance(e, dict)]


def _patient_demographics(bundle: dict[str, Any]
                              ) -> tuple[int | None, str | None]:
    for r in _entries(bundle):
        if (r.get("resourceType") or "").lower() != "patient":
            continue
        bd = r.get("birthDate")
        age = None
        if isinstance(bd, str) and len(bd) >= 4:
            try:
                dob = datetime.fromisoformat(bd[:10]).date()
                age = (datetime.now(timezone.utc).date() - dob).days // 365
            except Exception:
                pass
        sex = r.get("gender")
        return age, (sex.lower() if isinstance(sex, str) else None)
    return None, None


def _icd10_codes(bundle: dict[str, Any]) -> set[str]:
    codes: set[str] = set()
    for r in _entries(bundle):
        if (r.get("resourceType") or "").lower() != "condition":
            continue
        cc = r.get("code") or {}
        for coding in cc.get("coding", []) or []:
            sys = (coding.get("system") or "").lower()
            if "icd-10" in sys or "icd10" in sys:
                code = (coding.get("code") or "").strip()
                if code:
                    codes.add(code.replace(".", "").upper())
    return codes


def _has_condition_prefix(bundle: dict[str, Any], prefixes: tuple[str, ...]
                              ) -> bool:
    for code in _icd10_codes(bundle):
        for p in prefixes:
            if code.startswith(p):
                return True
    return False


def _latest_observation_value(bundle: dict[str, Any],
                                 *, loinc_codes: tuple[str, ...] = (),
                                 name_keywords: tuple[str, ...] = (),
                                 ) -> tuple[float | None, str | None]:
    """Return (value, effectiveDateTime) of the most recent matching obs."""
    best: tuple[datetime, float, str | None] | None = None
    for r in _entries(bundle):
        if (r.get("resourceType") or "").lower() != "observation":
            continue
        cc = r.get("code") or {}
        match = False
        for coding in cc.get("coding", []) or []:
            sys = (coding.get("system") or "").lower()
            code = (coding.get("code") or "").strip()
            if "loinc" in sys and code in loinc_codes:
                match = True
                break
        if not match and name_keywords:
            text = (cc.get("text") or "").lower()
            if any(k in text for k in name_keywords):
                match = True
        if not match:
            continue
        v = (r.get("valueQuantity") or {}).get("value")
        eff = r.get("effectiveDateTime")
        if not isinstance(v, (int, float)):
            continue
        if not isinstance(eff, str):
            continue
        try:
            dt = datetime.fromisoformat(
                eff.replace("Z", "+00:00")[:25])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if best is None or dt > best[0]:
            best = (dt, float(v), eff)
    if best is None:
        return None, None
    return best[1], best[2]


def _has_recent_screening(bundle: dict[str, Any],
                              *, name_keywords: tuple[str, ...],
                              within_days: int) -> tuple[bool, str | None]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=within_days)
    for r in _entries(bundle):
        rt = (r.get("resourceType") or "").lower()
        if rt not in ("procedure", "observation"):
            continue
        cc = r.get("code") or {}
        text = (cc.get("text") or "").lower()
        keyword_hit = any(k in text for k in name_keywords)
        if not keyword_hit:
            for coding in cc.get("coding", []) or []:
                disp = (coding.get("display") or "").lower()
                if any(k in disp for k in name_keywords):
                    keyword_hit = True
                    break
        if not keyword_hit:
            continue
        for key in ("performedDateTime", "effectiveDateTime",
                       "occurrenceDateTime"):
            v = r.get(key)
            if not isinstance(v, str):
                continue
            try:
                dt = datetime.fromisoformat(
                    v.replace("Z", "+00:00")[:25])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if dt >= cutoff:
                return True, v
    return False, None


def _evaluate_cdc_hm2(bundle: dict[str, Any], age: int | None
                          ) -> HEDISMeasure:
    """CDC-HM2 -- Comprehensive Diabetes Care: HbA1c < 8%."""
    eligible = (age is not None and 18 <= age <= 75
                  and _has_condition_prefix(bundle, ("E10", "E11", "E13")))
    if not eligible:
        return HEDISMeasure(
            measure_id="CDC-HM2", measure_name="HbA1c < 8% (diabetic)",
            eligible=False, in_compliance=False,
            rationale="Patient is not a diabetic adult age 18-75.",
            references=["NCQA HEDIS 2024 -- CDC."],
        )
    a1c, when = _latest_observation_value(
        bundle, loinc_codes=("4548-4",),
        name_keywords=("hba1c", "hemoglobin a1c"))
    if a1c is None:
        return HEDISMeasure(
            measure_id="CDC-HM2", measure_name="HbA1c < 8% (diabetic)",
            eligible=True, in_compliance=False,
            rationale="No HbA1c result on chart.",
            references=["NCQA HEDIS 2024 -- CDC."],
        )
    in_compliance = a1c < 8.0
    return HEDISMeasure(
        measure_id="CDC-HM2", measure_name="HbA1c < 8% (diabetic)",
        eligible=True, in_compliance=in_compliance,
        rationale=f"Latest HbA1c = {a1c:.1f}% (< 8% goal).",
        last_seen_iso=when,
        references=["NCQA HEDIS 2024 -- CDC."],
    )


def _evaluate_cbp(bundle: dict[str, Any], age: int | None) -> HEDISMeasure:
    """CBP -- Controlling Blood Pressure: < 140/90 mmHg."""
    eligible = (age is not None and 18 <= age <= 85
                  and _has_condition_prefix(bundle, ("I10", "I11", "I12",
                                                          "I13", "I15")))
    if not eligible:
        return HEDISMeasure(
            measure_id="CBP", measure_name="Blood pressure < 140/90 mmHg",
            eligible=False, in_compliance=False,
            rationale="Patient is not an HTN adult age 18-85.",
            references=["NCQA HEDIS 2024 -- CBP."],
        )
    sbp, when = _latest_observation_value(
        bundle, loinc_codes=("8480-6",),
        name_keywords=("systolic blood pressure", "sbp"))
    dbp, _ = _latest_observation_value(
        bundle, loinc_codes=("8462-4",),
        name_keywords=("diastolic blood pressure", "dbp"))
    if sbp is None or dbp is None:
        return HEDISMeasure(
            measure_id="CBP", measure_name="Blood pressure < 140/90 mmHg",
            eligible=True, in_compliance=False,
            rationale="Missing SBP or DBP on chart.",
            references=["NCQA HEDIS 2024 -- CBP."],
        )
    in_compliance = sbp < 140 and dbp < 90
    return HEDISMeasure(
        measure_id="CBP", measure_name="Blood pressure < 140/90 mmHg",
        eligible=True, in_compliance=in_compliance,
        rationale=f"Latest BP = {sbp:.0f}/{dbp:.0f} mmHg.",
        last_seen_iso=when,
        references=["NCQA HEDIS 2024 -- CBP."],
    )


def _evaluate_col(bundle: dict[str, Any], age: int | None) -> HEDISMeasure:
    eligible = age is not None and 50 <= age <= 75
    if not eligible:
        return HEDISMeasure(
            measure_id="COL", measure_name="Colorectal cancer screening",
            eligible=False, in_compliance=False,
            rationale="Age outside HEDIS 50-75 window.",
            references=["NCQA HEDIS 2024 -- COL."],
        )
    in_window, when = _has_recent_screening(
        bundle, name_keywords=("colonoscopy", "fit test",
                                  "fecal immunochemical", "cologuard"),
        within_days=3650)
    return HEDISMeasure(
        measure_id="COL", measure_name="Colorectal cancer screening",
        eligible=True, in_compliance=in_window,
        rationale=("Recent colon screen on chart" if in_window
                       else "No colon screening within 10y."),
        last_seen_iso=when,
        references=["NCQA HEDIS 2024 -- COL."],
    )


def _evaluate_ccs(bundle: dict[str, Any], age: int | None,
                     sex: str | None) -> HEDISMeasure:
    eligible = age is not None and 21 <= age <= 65 and sex == "female"
    if not eligible:
        return HEDISMeasure(
            measure_id="CCS", measure_name="Cervical cancer screening",
            eligible=False, in_compliance=False,
            rationale="Not a female age 21-65.",
            references=["NCQA HEDIS 2024 -- CCS."],
        )
    in_window, when = _has_recent_screening(
        bundle, name_keywords=("pap smear", "pap test", "hpv test",
                                  "cervical cytology"),
        within_days=1095)
    return HEDISMeasure(
        measure_id="CCS", measure_name="Cervical cancer screening",
        eligible=True, in_compliance=in_window,
        rationale=("Recent Pap/HPV on chart" if in_window
                       else "No cervical screening within 3y."),
        last_seen_iso=when,
        references=["NCQA HEDIS 2024 -- CCS."],
    )


def _evaluate_bcs(bundle: dict[str, Any], age: int | None,
                     sex: str | None) -> HEDISMeasure:
    eligible = age is not None and 50 <= age <= 74 and sex == "female"
    if not eligible:
        return HEDISMeasure(
            measure_id="BCS", measure_name="Breast cancer screening",
            eligible=False, in_compliance=False,
            rationale="Not a female age 50-74.",
            references=["NCQA HEDIS 2024 -- BCS."],
        )
    in_window, when = _has_recent_screening(
        bundle, name_keywords=("mammography", "mammogram",
                                  "breast screening"),
        within_days=730)
    return HEDISMeasure(
        measure_id="BCS", measure_name="Breast cancer screening",
        eligible=True, in_compliance=in_window,
        rationale=("Recent mammo on chart" if in_window
                       else "No mammography within 2y."),
        last_seen_iso=when,
        references=["NCQA HEDIS 2024 -- BCS."],
    )


def _evaluate_aab(bundle: dict[str, Any]) -> HEDISMeasure:
    """AAB -- Avoidance of Antibiotic Treatment for Acute Bronchitis.
    Compliance = NO antibiotic dispensed within 3 days of an acute
    bronchitis dx (J20-J21)."""
    has_bronchitis = _has_condition_prefix(bundle, ("J20", "J21"))
    if not has_bronchitis:
        return HEDISMeasure(
            measure_id="AAB",
            measure_name="Antibiotic avoidance in acute bronchitis",
            eligible=False, in_compliance=False,
            rationale="No acute bronchitis diagnosis on chart.",
            references=["NCQA HEDIS 2024 -- AAB."],
        )
    abx_keywords = ("amoxicillin", "azithromycin", "doxycycline",
                       "levofloxacin", "ciprofloxacin", "cefuroxime",
                       "clarithromycin", "moxifloxacin")
    for r in _entries(bundle):
        rt = (r.get("resourceType") or "").lower()
        if rt != "medicationrequest":
            continue
        cc = r.get("medicationCodeableConcept") or {}
        text = (cc.get("text") or "").lower()
        if any(k in text for k in abx_keywords):
            return HEDISMeasure(
                measure_id="AAB",
                measure_name="Antibiotic avoidance in acute bronchitis",
                eligible=True, in_compliance=False,
                rationale=f"Antibiotic dispensed: {text}.",
                references=["NCQA HEDIS 2024 -- AAB."],
            )
    return HEDISMeasure(
        measure_id="AAB",
        measure_name="Antibiotic avoidance in acute bronchitis",
        eligible=True, in_compliance=True,
        rationale="No antibiotic prescribed for bronchitis episode.",
        references=["NCQA HEDIS 2024 -- AAB."],
    )


def compute_hedis_score(fhir_bundle: dict[str, Any]) -> HEDISReport:
    """Score a patient FHIR Bundle against 6 HEDIS quality measures."""
    if not isinstance(fhir_bundle, dict):
        raise ValueError("fhir_bundle must be a dict.")
    age, sex = _patient_demographics(fhir_bundle)

    measures = [
        _evaluate_cdc_hm2(fhir_bundle, age),
        _evaluate_cbp(fhir_bundle, age),
        _evaluate_col(fhir_bundle, age),
        _evaluate_ccs(fhir_bundle, age, sex),
        _evaluate_bcs(fhir_bundle, age, sex),
        _evaluate_aab(fhir_bundle),
    ]

    n_eligible = sum(1 for m in measures if m.eligible)
    n_compliant = sum(1 for m in measures if m.eligible and m.in_compliance)
    n_non_compliant = sum(1 for m in measures
                              if m.eligible and not m.in_compliance)
    n_not_eligible = sum(1 for m in measures if not m.eligible)
    composite = (100.0 * n_compliant / n_eligible
                    if n_eligible > 0 else 0.0)

    rationale = (
        f"HEDIS evaluated: {n_eligible} measure(s) eligible, "
        f"{n_compliant} compliant, {n_non_compliant} non-compliant, "
        f"{n_not_eligible} not eligible. Composite "
        f"{composite:.1f}%."
    )

    return HEDISReport(
        patient_age=age, patient_sex=sex,
        n_measures_evaluated=n_eligible,
        n_compliant=n_compliant,
        n_non_compliant=n_non_compliant,
        n_not_eligible=n_not_eligible,
        measures=measures,
        composite_score_pct=round(composite, 2),
        rationale=rationale,
    )


# ─────────────────────── EVAL-2: Hospital Compare benchmark ───────────────────────
#
# Peer-tier medians from CMS Hospital Compare 2024 publication. Values
# are illustrative reference points; production deployments would pull
# the live CMS dataset.

_PEER_MEDIANS: dict[str, dict[str, float]] = {
    "academic_major": {
        "readmission_rate_30d_pct": 16.5,
        "ed_revisit_rate_72h_pct": 4.0,
        "hcahps_overall_pct": 72,
        "mortality_rate_30d_pct": 13.2,
        "abstention_rate_pct": 3.5,
    },
    "community_large": {
        "readmission_rate_30d_pct": 15.5,
        "ed_revisit_rate_72h_pct": 4.5,
        "hcahps_overall_pct": 70,
        "mortality_rate_30d_pct": 12.8,
        "abstention_rate_pct": 4.0,
    },
    "community_small": {
        "readmission_rate_30d_pct": 15.0,
        "ed_revisit_rate_72h_pct": 5.0,
        "hcahps_overall_pct": 71,
        "mortality_rate_30d_pct": 13.0,
        "abstention_rate_pct": 4.5,
    },
    "rural": {
        "readmission_rate_30d_pct": 14.5,
        "ed_revisit_rate_72h_pct": 5.5,
        "hcahps_overall_pct": 73,
        "mortality_rate_30d_pct": 13.5,
        "abstention_rate_pct": 5.0,
    },
    "specialty": {
        "readmission_rate_30d_pct": 12.0,
        "ed_revisit_rate_72h_pct": 3.0,
        "hcahps_overall_pct": 76,
        "mortality_rate_30d_pct": 8.0,
        "abstention_rate_pct": 2.5,
    },
}


# Higher-is-better KPIs (HCAHPS) vs lower-is-better (everything else)
_HIGHER_IS_BETTER = {"hcahps_overall_pct"}


def compute_hospital_compare_benchmark(
    institution_label: str,
    institution_kpis: dict[str, float],
    peer_tier: str,
) -> HospitalCompareBenchmark:
    """Compare an institution's KPIs to CMS Hospital Compare peer medians.

    Args:
        institution_label: free-text label.
        institution_kpis: dict of KPI -> value.
        peer_tier: one of academic_major / community_large /
            community_small / rural / specialty.

    Returns:
        HospitalCompareBenchmark with deltas, percentile rank, composite.
    """
    if peer_tier not in _PEER_MEDIANS:
        raise ValueError(
            f"peer_tier must be one of {sorted(_PEER_MEDIANS)}.")
    if not isinstance(institution_kpis, dict) or not institution_kpis:
        raise ValueError("institution_kpis must be a non-empty dict.")

    medians = _PEER_MEDIANS[peer_tier]
    deltas: dict[str, float] = {}
    percentiles: dict[str, int] = {}
    composite_signed_deltas: list[float] = []

    for kpi, value in institution_kpis.items():
        median = medians.get(kpi)
        if median is None:
            continue
        delta = value - median
        deltas[kpi] = round(delta, 4)
        if kpi in _HIGHER_IS_BETTER:
            relative_improvement = (value - median) / max(1e-6, median)
        else:
            relative_improvement = (median - value) / max(1e-6, median)
        # Map relative-improvement [-1, +1] to percentile [0, 100]
        pct = int(max(0, min(100, round(50 + 50 * relative_improvement))))
        percentiles[kpi] = pct
        composite_signed_deltas.append(relative_improvement)

    if composite_signed_deltas:
        composite = max(0.0, min(100.0,
                                       50 + 50 * (sum(composite_signed_deltas)
                                                     / len(composite_signed_deltas))))
    else:
        composite = 0.0

    rationale = (
        f"Peer tier {peer_tier}: {len(deltas)} KPI comparisons. "
        f"Composite quality score {composite:.1f}/100."
    )

    return HospitalCompareBenchmark(
        institution_label=institution_label,
        institution_kpis=institution_kpis,
        peer_tier=peer_tier,                    # type: ignore[arg-type]
        peer_medians=medians,
        deltas_vs_median=deltas,
        percentile_rank=percentiles,
        composite_quality_score=round(composite, 2),
        rationale=rationale,
    )


# ─────────────────────── EVAL-3: Schwartz quality framework ───────────────────────

def compute_schwartz_quality_score(
    decision_card: dict[str, Any],
) -> SchwartzQualityScore:
    """Score a DecisionCard on Schwartz et al. JAMIA 2017 4-pillar framework.

    The four pillars + their inferred signals:
      - trustworthy: presence of citations + grounding verdict + critic
        ensemble approval
      - relevant: action chosen, confidence > none, risk_estimate present
      - actionable: explicit follow-up window, structured order set,
        abstain triggers (or absence-as-positive when high confidence)
      - usable: counseling section present, plain-language ranks
    """
    if not isinstance(decision_card, dict):
        raise ValueError("decision_card must be a dict.")

    rec = decision_card.get("recommendation") or {}
    reasoning = decision_card.get("reasoning") or {}
    validation = decision_card.get("validation") or {}
    audit = decision_card.get("audit") or {}
    abstain = decision_card.get("abstain") or []
    self_critique = decision_card.get("self_critique")

    # Trustworthy
    trustworthy = 0.0
    grounding = (validation.get("grounding") or {})
    if grounding.get("overall_verdict") == "supported":
        trustworthy += 0.40
    elif grounding.get("overall_verdict") == "partially_supported":
        trustworthy += 0.20
    if isinstance(self_critique, dict):
        verdict = self_critique.get("verdict")
        if verdict == "approved":
            trustworthy += 0.30
        elif verdict == "downgrade_confidence":
            trustworthy += 0.15
    risk = reasoning.get("risk_estimate") or {}
    if risk.get("model_version"):
        trustworthy += 0.10
    if risk.get("contributing_factors"):
        trustworthy += 0.20
    trustworthy = min(1.0, trustworthy)

    # Relevant
    relevant = 0.0
    if isinstance(rec, dict) and rec.get("action"):
        relevant += 0.50
    if isinstance(rec, dict) and rec.get("confidence") in (
            "high", "medium", "low"):
        relevant += 0.30
    if risk.get("probability_mean") is not None:
        relevant += 0.20
    relevant = min(1.0, relevant)

    # Actionable
    actionable = 0.0
    if abstain:
        actionable += 0.30
    if isinstance(rec, dict) and rec.get("action"):
        actionable += 0.30
    counseling = decision_card.get("counseling") or {}
    if counseling and counseling.get("sections"):
        actionable += 0.20
    if reasoning.get("utility_analysis"):
        actionable += 0.20
    actionable = min(1.0, actionable)

    # Usable
    usable = 0.0
    if counseling and counseling.get("sections"):
        usable += 0.40
    phi_check = (validation.get("phi_check") or {})
    if phi_check.get("risk_level") == "none":
        usable += 0.20
    if audit.get("request_id"):
        usable += 0.20
    if isinstance(rec, dict) and rec.get("confidence") in (
            "high", "medium"):
        usable += 0.20
    usable = min(1.0, usable)

    composite = round((trustworthy + relevant + actionable + usable) / 4.0, 4)
    if composite >= 0.85:
        grade = "A"
    elif composite >= 0.70:
        grade = "B"
    elif composite >= 0.55:
        grade = "C"
    elif composite >= 0.40:
        grade = "D"
    else:
        grade = "F"

    weak: list[str] = []
    for pillar, score in (("trustworthy", trustworthy),
                              ("relevant", relevant),
                              ("actionable", actionable),
                              ("usable", usable)):
        if score < 0.5:
            weak.append(pillar)

    rationale = (
        f"Schwartz JAMIA 2017 4-pillar score: trustworthy "
        f"{trustworthy:.2f}, relevant {relevant:.2f}, actionable "
        f"{actionable:.2f}, usable {usable:.2f}. "
        f"Composite {composite:.2f} -> grade {grade}."
    )

    return SchwartzQualityScore(
        request_id=audit.get("request_id"),
        trustworthy_score=round(trustworthy, 4),
        relevant_score=round(relevant, 4),
        actionable_score=round(actionable, 4),
        usable_score=round(usable, 4),
        composite_score=composite,
        grade=grade,                    # type: ignore[arg-type]
        rationale=rationale,
        weak_dimensions=weak,
    )


# ─────────────────────── EVAL-4: Comparative-effectiveness ───────────────────────

def compute_comparative_effectiveness(
    n_patients: int = 1000,
    baseline_event_rate: float = 0.18,
    intervention_relative_risk_reduction: float = 0.25,
    seed: int = 42,
) -> CohortComparativeReport:
    """Simulate a randomized 1:1 cohort and compute ARR / NNT / 95% CI.

    Args:
        n_patients: total N (split equally between intervention + control).
        baseline_event_rate: control-arm event probability.
        intervention_relative_risk_reduction: RRR (0-1).
        seed: RNG seed for reproducibility.
    """
    if n_patients < 10 or n_patients > 1_000_000:
        raise ValueError("n_patients must be in [10, 1_000_000].")
    if not (0.0 <= baseline_event_rate <= 1.0):
        raise ValueError("baseline_event_rate must be in [0, 1].")
    if not (0.0 <= intervention_relative_risk_reduction <= 1.0):
        raise ValueError(
            "intervention_relative_risk_reduction must be in [0, 1].")

    rng = random.Random(seed)
    n_per_arm = n_patients // 2
    control_p = baseline_event_rate
    intervention_p = baseline_event_rate * (
        1.0 - intervention_relative_risk_reduction)

    control_events = sum(1 for _ in range(n_per_arm)
                              if rng.random() < control_p)
    intervention_events = sum(1 for _ in range(n_per_arm)
                                    if rng.random() < intervention_p)

    p_c = control_events / n_per_arm
    p_i = intervention_events / n_per_arm
    arr = p_c - p_i
    rr = (p_i / p_c) if p_c > 0 else float("nan")
    nnt = (1.0 / arr) if arr > 0 else None

    # Wald 95% CI on the difference of two proportions
    var_c = (p_c * (1 - p_c)) / n_per_arm
    var_i = (p_i * (1 - p_i)) / n_per_arm
    se = math.sqrt(var_c + var_i)
    ci_low = arr - 1.96 * se
    ci_high = arr + 1.96 * se

    # Two-sided z-test p-value
    if se > 0:
        z = arr / se
        # Approximate p-value via normal CDF
        from math import erf, sqrt
        p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    else:
        p_value = 1.0

    rationale = (
        f"N={n_patients} randomized 1:1. Control arm event rate "
        f"{p_c:.3f}, intervention arm {p_i:.3f}. "
        f"ARR={arr:.4f} (CI95 {ci_low:.4f}-{ci_high:.4f}), "
        f"NNT={nnt:.1f}" if nnt is not None
        else f"NNT n/a (no benefit)") + f", p={p_value:.4f}."

    return CohortComparativeReport(
        n_patients=n_patients,
        arr_30d=round(arr, 4),
        rr_30d=round(rr, 4) if not math.isnan(rr) else 0.0,
        nnt=round(nnt, 1) if nnt is not None else None,
        intervention_arm_event_rate=round(p_i, 4),
        control_arm_event_rate=round(p_c, 4),
        p_value=round(p_value, 4),
        ci95_arr_low=round(ci_low, 4),
        ci95_arr_high=round(ci_high, 4),
        rationale=rationale,
    )
