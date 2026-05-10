"""AUDIT-4 -- Patient-facing audit summary.

Companion to AUDIT-3 (right-to-explanation). The right-to-explanation
output is a NARRATIVE -- the audit summary is a STRUCTURED bill-of-
materials: what features were considered, what weight each carried, what
sources were cited, and what abstain triggers fired. The patient (or
their advocate) can use this to file a request for clinician review
under GDPR Art. 22.

PHI safety: this report carries NO patient identifiers -- only feature
names and aggregate weights. The discharge plan paper itself contains
the patient-identifiable information.
"""

from __future__ import annotations

from typing import Any

from shared.schemas import (
    DecisionCard,
    PatientAuditSummary,
)


def _features_table(card: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    reasoning = card.get("reasoning") or {}
    risk = reasoning.get("risk_estimate") or {}
    for f in risk.get("contributing_factors", []) or []:
        if not isinstance(f, dict):
            continue
        out.append({
            "feature_name": str(f.get("name", "")),
            "raw_value": f.get("raw_value"),
            "lace_points": f.get("lace_points"),
            "weight": float(f.get("weight", 0.0)),
            "explanation": _factor_explanation(f.get("name", "")),
        })
    return out


_FACTOR_EXPLANATIONS: dict[str, str] = {
    "LACE_length_of_stay":
        "How long you stayed in the hospital. Longer stays usually mean "
        "more complex illness, which raises the chance of needing to "
        "come back.",
    "LACE_acuity":
        "How urgently you were admitted. An emergency admission "
        "carries more risk than a planned admission.",
    "LACE_comorbidity":
        "How many other long-term conditions you have. More conditions "
        "means more moving parts to monitor after discharge.",
    "LACE_ed_visits_6mo":
        "How often you have been in the emergency room in the last 6 "
        "months. More visits is a known signal for higher readmission risk.",
}


def _factor_explanation(name: str) -> str:
    return _FACTOR_EXPLANATIONS.get(
        name,
        f"This factor ({name}) was used by the model. Ask your care team "
        "for a clinical interpretation."
    )


def _citations_table(card: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    validation = card.get("validation") or {}
    grounding = validation.get("grounding") or {}
    for sc in grounding.get("sub_claims", []) or []:
        if not isinstance(sc, dict):
            continue
        for ev in sc.get("evidence_sources", []) or []:
            if not isinstance(ev, dict):
                continue
            out.append({
                "source_id": str(ev.get("source_id", "")),
                "source_type": str(ev.get("source_type", "")),
                "excerpt": str(ev.get("excerpt", ""))[:300],
            })
    # Dedup
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for c in out:
        key = c["source_id"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def _abstain_triggers_present(card: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for t in card.get("abstain", []) or []:
        if isinstance(t, dict):
            out.append(str(t.get("type") or t.get("trigger") or "unknown"))
    return out


def _headline_question(card: dict[str, Any]) -> str:
    rec = card.get("recommendation") or {}
    action = rec.get("action") if isinstance(rec, dict) else None
    if action == "discharge_home":
        return "Why does the system recommend that I go home?"
    if action == "home_with_care":
        return ("Why does the system recommend that I go home with "
                "extra support?")
    if action == "snf":
        return ("Why does the system recommend a skilled nursing "
                "facility instead of home?")
    if action == "continued_admission":
        return "Why does the system recommend that I stay in the hospital?"
    return ("Why was no automatic recommendation made -- what did the "
            "system consider?")


def _plain_answer(card: dict[str, Any],
                     features: list[dict[str, Any]]) -> str:
    rec = card.get("recommendation") or {}
    action = rec.get("action") if isinstance(rec, dict) else None
    confidence = rec.get("confidence") if isinstance(rec, dict) else None
    parts: list[str] = []
    if action:
        parts.append(
            f"The system recommended {action.replace('_', ' ')} with "
            f"{confidence or 'unspecified'} confidence."
        )
    else:
        parts.append(
            "The system did NOT make a final recommendation. Your "
            "clinician makes the call here."
        )
    if features:
        top = sorted(features,
                        key=lambda f: -abs(f.get("weight", 0.0)))[:3]
        parts.append(
            "Top factors used: "
            + "; ".join(f["feature_name"] for f in top)
            + "."
        )
    parts.append(
        "Below you'll find the full list of factors and the literature "
        "the system cited. If anything looks wrong or you disagree, you "
        "have the right to ask a human clinician to review."
    )
    return " ".join(parts)


def compute_patient_audit_summary(
    decision_card: DecisionCard | dict[str, Any],
) -> PatientAuditSummary:
    """Build a PHI-safe patient-facing audit summary."""
    if isinstance(decision_card, DecisionCard):
        card = decision_card.model_dump(mode="json")
    elif isinstance(decision_card, dict):
        card = decision_card
    else:
        raise ValueError("decision_card must be DecisionCard or dict.")

    features = _features_table(card)
    citations = _citations_table(card)
    abstain = _abstain_triggers_present(card)

    rec = card.get("recommendation") or {}
    confidence = rec.get("confidence") if isinstance(rec, dict) else None

    audit = card.get("audit") or {}
    request_id = audit.get("request_id")

    return PatientAuditSummary(
        request_id=request_id,
        headline_question=_headline_question(card),
        plain_language_answer=_plain_answer(card, features),
        features_considered=features,
        citations_used=citations,
        abstain_triggers_present=abstain,
        confidence=confidence,
    )
