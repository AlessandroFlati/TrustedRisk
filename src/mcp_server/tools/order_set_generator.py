"""healthcare.compute_order_set -- LIB-1.

Composes a structured discharge order set from:

  - the discharge action (Action enum: discharge_home / home_with_care /
    snf / continued_admission)
  - the calibrated risk estimate (drives monitoring intensity)
  - the medication list (drives med-specific monitoring orders, e.g.
    warfarin -> INR weekly)
  - the chief complaint (drives specialty consults + dietary)

Pure-template generator -- no LLM. Each order carries a rationale + a
reference chain so an EHR ingestion layer can import directly without
human re-keying.
"""

from __future__ import annotations

from typing import Any, Iterable

from shared.schemas import (
    OrderSet,
    OrderSetEntry,
    RiskEstimate,
)


# ─────────────────────── Risk tier mapping ───────────────────────

def _risk_tier(probability_mean: float | None,
                  confidence: str | None) -> str:
    p = probability_mean or 0.0
    if p >= 0.30:
        return "high"
    if p >= 0.15:
        return "moderate"
    if confidence == "low":
        return "moderate"
    return "low"


_NEXT_VISIT_DAYS_BY_TIER: dict[str, tuple[int, int]] = {
    "high":     (3, 7),
    "moderate": (7, 14),
    "low":      (14, 30),
}


# ─────────────────────── Drug-class -> monitoring orders ───────────────────────
#
# Per AHRQ MATCH program + Lexicomp drug-class monitoring contracts.

_MED_MONITORING: dict[str, list[tuple[str, str, str]]] = {
    "anticoagulant_vka": [
        ("INR (Prothrombin time)",
         "Goal INR 2-3 for atrial fibrillation, 2.5-3.5 for mechanical valves",
         "every 1-2 weeks until stable, then every 4 weeks"),
    ],
    "anticoagulant_doac": [
        ("CBC + Renal function panel",
         "Monitor for occult bleeding + renal-driven dose adjustment",
         "every 6 months"),
    ],
    "ace_inhibitor": [
        ("Basic metabolic panel (K+, Cr)",
         "Hyperkalemia + AKI risk especially with concurrent MRA",
         "within 7 days, then every 1-3 months"),
    ],
    "mra": [
        ("Basic metabolic panel (K+)",
         "Hyperkalemia risk amplified by ACE-I / ARB",
         "within 7 days, then monthly while titrating"),
    ],
    "loop_diuretic": [
        ("Basic metabolic panel + renal function",
         "Volume + electrolyte management; eGFR monitoring",
         "weekly until euvolemic, then monthly"),
    ],
    "biguanide": [
        ("HbA1c, eGFR",
         "Discontinue if eGFR < 30 (lactic acidosis risk)",
         "every 3 months for HbA1c, every 6 months eGFR"),
    ],
    "beta_blocker": [
        ("Heart rate + blood pressure log",
         "Goal HR 60-80 in HF; avoid abrupt cessation",
         "daily by patient, review at 2 weeks"),
    ],
    "statin": [
        ("Liver function panel",
         "ALT / AST baseline + repeat if symptomatic",
         "baseline + at 3 months if elevated"),
    ],
}


# ─────────────────────── Per-action backbone orders ───────────────────────

def _backbone_orders(action: str, tier: str) -> list[OrderSetEntry]:
    visit_lo, visit_hi = _NEXT_VISIT_DAYS_BY_TIER[tier]
    orders: list[OrderSetEntry] = []

    if action in ("discharge_home", "home_with_care"):
        orders.append(OrderSetEntry(
            category="followup_appointment",
            order_text=f"PCP follow-up within {visit_lo}-{visit_hi} days",
            rationale=("AHRQ RED bundle + Hansen NEJM 2011 -- "
                          f"{tier}-risk discharges benefit from short-interval "
                          "primary-care touchpoint."),
            timing=f"{visit_lo}-{visit_hi} days post-discharge",
            priority=("urgent" if tier == "high" else "routine"),
        ))

    if action == "home_with_care":
        orders.append(OrderSetEntry(
            category="consult",
            order_text="Home health nursing referral (skilled visit + "
                          "medication reconciliation)",
            rationale="home_with_care disposition; supports adherence + "
                          "vital-sign monitoring.",
            timing="within 48 hours of discharge",
            priority="routine",
        ))

    if action == "snf":
        orders.append(OrderSetEntry(
            category="followup_appointment",
            order_text="SNF intake assessment + medication reconciliation "
                          "with facility pharmacist",
            rationale="SNF transition requires reconciliation handshake to "
                          "prevent dose drift.",
            timing="day of admission to SNF",
            priority="urgent",
        ))

    if action == "continued_admission":
        orders.append(OrderSetEntry(
            category="vital_monitoring",
            order_text="Vital signs every 4 hours",
            rationale="Continued inpatient monitoring; deterioration risk.",
            timing="ongoing",
            priority="routine",
        ))

    if tier == "high":
        orders.append(OrderSetEntry(
            category="patient_education",
            order_text="Teach-back education: warning signs + when to call "
                          "911 vs PCP",
            rationale=("High-risk discharge: AHRQ teach-back reduces "
                          "readmission ~ 12%."),
            timing="before discharge",
            priority="routine",
        ))

    return orders


# ─────────────────────── Per-medication orders ───────────────────────

def _medication_orders(medications: Iterable[Any] | None
                          ) -> list[OrderSetEntry]:
    if not medications:
        return []
    orders: list[OrderSetEntry] = []
    for med in medications:
        if isinstance(med, dict):
            name = med.get("name", "")
            drug_class = med.get("drug_class")
        else:
            name = getattr(med, "name", "")
            drug_class = getattr(med, "drug_class", None)
        if not name:
            continue
        orders.append(OrderSetEntry(
            category="medication",
            order_text=f"Continue {name} at discharge dose",
            rationale="Medication reconciliation against admission list.",
            timing="ongoing",
            priority="routine",
        ))
        if drug_class and drug_class in _MED_MONITORING:
            for label, rationale, timing in _MED_MONITORING[drug_class]:
                orders.append(OrderSetEntry(
                    category="lab_monitoring",
                    order_text=label,
                    rationale=f"{name} ({drug_class}) -- {rationale}",
                    timing=timing,
                    priority="routine",
                ))
    return orders


# ─────────────────────── Chief-complaint specialty orders ───────────────────────

_SPECIALTY_ORDERS: list[tuple[str, str, str, str]] = [
    ("chf|heart failure|pulmonary edema",
     "Cardiology follow-up within 7-14 days",
     "consult",
     "ACC/AHA HF guideline -- early follow-up reduces readmission."),
    ("aki|acute kidney injury",
     "Nephrology follow-up within 7-14 days",
     "consult",
     "KDIGO 2012 -- close monitoring of renal recovery + medication review."),
    ("dka|diabetic ketoacidosis",
     "Endocrinology follow-up within 7 days",
     "consult",
     "Post-DKA endocrine review for insulin titration."),
    ("stroke|cva",
     "Neurology + speech/PT/OT therapy referral",
     "consult",
     "AHA stroke guideline -- multidisciplinary follow-up reduces "
     "secondary events."),
    ("acs|chest pain|nstemi|stemi",
     "Cardiology follow-up within 7 days; cardiac rehab referral",
     "consult",
     "ACC/AHA ACS guideline -- cardiac rehab Class I."),
    ("preeclampsia",
     "Postpartum BP check at 3-7 days; obstetric follow-up at 1-2 weeks",
     "followup_appointment",
     "ACOG guideline 222 -- postpartum HTN persists 6-12 weeks."),
    ("suicid",
     "Mental health follow-up within 72 hours; safety plan reviewed",
     "consult",
     "Joint Commission NPSG 15 -- short-interval mental health contact."),
]


def _specialty_orders(chief_complaint: str | None) -> list[OrderSetEntry]:
    if not chief_complaint:
        return []
    cc = chief_complaint.lower()
    out: list[OrderSetEntry] = []
    import re
    for pattern, order_text, category, rationale in _SPECIALTY_ORDERS:
        if re.search(pattern, cc):
            out.append(OrderSetEntry(
                category=category,                  # type: ignore[arg-type]
                order_text=order_text,
                rationale=rationale,
                timing="per text",
                priority="routine",
            ))
    return out


# ─────────────────────── Public API ───────────────────────

async def compute_order_set(
    discharge_action: str,
    risk_estimate: RiskEstimate | dict | None = None,
    medications: list[Any] | None = None,
    chief_complaint: str | None = None,
) -> OrderSet:
    """Generate a structured discharge order set.

    Args:
        discharge_action: one of discharge_home / home_with_care / snf /
            continued_admission.
        risk_estimate: RiskEstimate (or dict) -- drives monitoring intensity.
        medications: list of Medication-shaped dicts/objects with at least
            `name` and (optionally) `drug_class`.
        chief_complaint: free text -- drives specialty consult orders.

    Returns:
        OrderSet with categorized orders + per-order rationale + references.
    """
    valid_actions = {
        "discharge_home", "home_with_care", "snf", "continued_admission",
    }
    if discharge_action not in valid_actions:
        raise ValueError(
            f"discharge_action must be one of {sorted(valid_actions)}; "
            f"got {discharge_action!r}.")

    if risk_estimate is None:
        prob = None
        confidence = None
    elif isinstance(risk_estimate, dict):
        prob = risk_estimate.get("probability_mean")
        confidence = risk_estimate.get("confidence")
    else:
        prob = risk_estimate.probability_mean
        confidence = getattr(risk_estimate, "confidence", None)

    tier = _risk_tier(prob, confidence)
    visit_window = _NEXT_VISIT_DAYS_BY_TIER[tier]

    orders: list[OrderSetEntry] = []
    orders.extend(_backbone_orders(discharge_action, tier))
    orders.extend(_medication_orders(medications))
    orders.extend(_specialty_orders(chief_complaint))

    return OrderSet(
        discharge_action=discharge_action,
        risk_tier=tier,                    # type: ignore[arg-type]
        next_visit_window_days=visit_window,
        orders=orders,
        n_orders=len(orders),
    )


def register(mcp) -> None:
    mcp.tool()(compute_order_set)
