"""Phase 16.J1 - Patient-advocate A2A specialist.

A second-opinion agent that runs in parallel with the clinician
specialist. Reads the same FHIR context and the candidate
DecisionCard, then evaluates whether the recommendation is in the
*patient's* interest along five axes:

  - **fairness**: is the patient in a documented under-prediction
    subgroup without a fresh fairness audit?
  - **autonomy**: would the recommendation reduce shared-decision-
    making surface (e.g. abrupt forced admission for a stable
    patient)?
  - **accessibility**: does the recommendation rely on home-care
    that requires resources the patient may not have (transport,
    pharmacy refills, English-language support)?
  - **financial_burden**: does the recommendation push high
    out-of-pocket cost (uninsured + chronic-med-heavy regimen)?
  - **language**: is the patient's preferred language served by
    available counseling artefacts?

The output is a ``SecondOpinionCard`` with per-axis verdict + an
overall recommendation: ``concur`` / ``challenge`` / ``escalate``.

Pure-deterministic floor. No LLM in the floor.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


_AxisVerdict = Literal["ok", "concern", "blocker"]
_OverallVerdict = Literal["concur", "challenge", "escalate"]


_FLAGGED_RACE = frozenset({
    "black", "indigenous", "native_american", "african_american",
})
_FLAGGED_INSURANCE = frozenset({
    "medicaid", "uninsured",
})
_SUPPORTED_LANGUAGES = frozenset({
    "english", "spanish", "chinese", "vietnamese", "arabic",
})


# ─────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────


class PatientAdvocateInput(BaseModel):
    patient_age: int = Field(ge=0)
    patient_race: str | None = None
    patient_insurance: str | None = None
    patient_language: str | None = None
    n_chronic_medications: int = Field(default=0, ge=0)
    has_home_caregiver_available: bool = True
    has_transportation: bool = True
    fairness_audit_present: bool = False
    recommended_action: str
    risk_point_estimate: float = Field(ge=0.0, le=1.0)


class AxisFinding(BaseModel):
    axis: str
    verdict: _AxisVerdict
    rationale: str


class SecondOpinionCard(BaseModel):
    overall_verdict: _OverallVerdict
    findings: list[AxisFinding]
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Per-axis evaluators
# ─────────────────────────────────────────────────────────────────────


def _axis_fairness(p: PatientAdvocateInput) -> AxisFinding:
    race = (p.patient_race or "").strip().lower().replace(" ", "_")
    insurance = (p.patient_insurance or "").strip().lower()
    flagged = race in _FLAGGED_RACE or insurance in _FLAGGED_INSURANCE
    if flagged and not p.fairness_audit_present:
        return AxisFinding(
            axis="fairness", verdict="blocker",
            rationale=(
                f"Patient is in flagged subgroup "
                f"(race={race or '-'}, insurance={insurance or '-'}) "
                "and no fairness-audit artefact is supplied; the "
                "advocate cannot endorse the recommendation."
            ),
        )
    if flagged:
        return AxisFinding(
            axis="fairness", verdict="concern",
            rationale=(
                "Patient is in a documented under-prediction "
                "subgroup; fairness audit is present but the case "
                "warrants a clinician sanity-check."
            ),
        )
    return AxisFinding(
        axis="fairness", verdict="ok",
        rationale="No flagged subgroup or audit confirms parity.",
    )


def _axis_autonomy(p: PatientAdvocateInput) -> AxisFinding:
    if (
        p.recommended_action == "continued_admission"
        and p.risk_point_estimate < 0.20
    ):
        return AxisFinding(
            axis="autonomy", verdict="concern",
            rationale=(
                f"Continued admission proposed at low risk "
                f"({p.risk_point_estimate*100:.1f}%); shared-"
                "decision-making must be offered before locking the "
                "patient in."
            ),
        )
    if p.recommended_action == "abstain":
        return AxisFinding(
            axis="autonomy", verdict="ok",
            rationale=(
                "Abstain preserves clinician + patient agency."
            ),
        )
    return AxisFinding(
        axis="autonomy", verdict="ok",
        rationale="Action proportional to risk; autonomy preserved.",
    )


def _axis_accessibility(p: PatientAdvocateInput) -> AxisFinding:
    needs_home_care = (
        p.recommended_action in (
            "discharge_home", "discharge_with_homecare",
        )
    )
    if not needs_home_care:
        return AxisFinding(
            axis="accessibility", verdict="ok",
            rationale="Inpatient setting; home support not required.",
        )
    blocking_factors: list[str] = []
    if not p.has_home_caregiver_available:
        blocking_factors.append("no caregiver at home")
    if not p.has_transportation:
        blocking_factors.append("no reliable transportation")
    if blocking_factors:
        return AxisFinding(
            axis="accessibility", verdict="blocker",
            rationale=(
                f"Discharge plan relies on resources the patient "
                f"does not have: {', '.join(blocking_factors)}."
            ),
        )
    return AxisFinding(
        axis="accessibility", verdict="ok",
        rationale="Home-care resources confirmed.",
    )


def _axis_financial_burden(
    p: PatientAdvocateInput,
) -> AxisFinding:
    insurance = (p.patient_insurance or "").lower()
    if (
        insurance in ("uninsured", "")
        and p.n_chronic_medications >= 4
    ):
        return AxisFinding(
            axis="financial_burden", verdict="concern",
            rationale=(
                f"Uninsured patient with {p.n_chronic_medications} "
                "chronic medications - referral to a 340B / "
                "patient-assistance programme should accompany the "
                "discharge."
            ),
        )
    return AxisFinding(
        axis="financial_burden", verdict="ok",
        rationale=(
            "No salient financial-burden flag identified."
        ),
    )


def _axis_language(p: PatientAdvocateInput) -> AxisFinding:
    lang = (p.patient_language or "english").strip().lower()
    if lang not in _SUPPORTED_LANGUAGES:
        return AxisFinding(
            axis="language", verdict="concern",
            rationale=(
                f"Patient prefers `{lang}` which is outside the "
                "currently supported counseling languages "
                "(en/es/zh/vi/ar). Request human interpreter."
            ),
        )
    return AxisFinding(
        axis="language", verdict="ok",
        rationale=(
            f"Counseling artefact available in `{lang}`."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────


def _aggregate(
    findings: list[AxisFinding],
) -> _OverallVerdict:
    has_blocker = any(f.verdict == "blocker" for f in findings)
    has_concern = any(f.verdict == "concern" for f in findings)
    if has_blocker:
        return "escalate"
    if has_concern:
        return "challenge"
    return "concur"


def evaluate_patient_advocate(
    payload: PatientAdvocateInput,
) -> SecondOpinionCard:
    """Run the 5-axis evaluation and return a SecondOpinionCard."""
    findings = [
        _axis_fairness(payload),
        _axis_autonomy(payload),
        _axis_accessibility(payload),
        _axis_financial_burden(payload),
        _axis_language(payload),
    ]
    overall = _aggregate(findings)
    rationale = {
        "concur": (
            "All five advocate axes are satisfied; the recommendation "
            "is endorsed."
        ),
        "challenge": (
            "One or more axes raised concerns; the advocate requests "
            "the clinician revisit before enacting."
        ),
        "escalate": (
            "An axis is a hard blocker; the advocate requests "
            "escalation to a patient-rights officer or social worker "
            "before the recommendation is enacted."
        ),
    }[overall]
    return SecondOpinionCard(
        overall_verdict=overall,
        findings=findings,
        rationale=rationale,
        references=[
            "TrustedRisk Phase 16.J - patient-advocate A2A specialist.",
            "AHRQ patient-centred care framework (2018).",
            "Charter on Medical Professionalism (2002) "
            "- patient autonomy + social justice principles.",
        ],
    )
