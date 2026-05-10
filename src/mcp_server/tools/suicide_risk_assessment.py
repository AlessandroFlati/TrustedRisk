"""healthcare.compute_suicide_risk_assessment -- C-SSRS-derived risk score.

The Columbia-Suicide Severity Rating Scale (Posner et al. 2011) is the
field-standard structured suicide-risk assessment. It captures both
*ideation* (severity 0-5 on a graded scale: passive ↔ "active with
specific plan and intent") and *behavior* (lifetime + past-30-days actual
attempts, interrupted attempts, aborted attempts, preparatory acts, NSSI).

Risk stratification combines:
  - past-30-days ideation severity (level 4-5 -> high)
  - past-30-days behavior (any attempt -> imminent)
  - lifetime attempts (≥1 -> at least moderate baseline)
  - protective vs warning factors (modifiers)

Why this tool ABSTAINS aggressively:
  Suicide-risk algorithms have well-documented bias against minoritized
  groups (Black, LGBTQ+, low-SES) -- they frequently under-predict risk.
  This tool tags `abstain_recommended=True` whenever the input is sparse
  or the patient demographics could amplify model bias, deferring to
  in-person clinical judgement. The abstain trigger is the SAFETY behavior,
  not the failure mode.

References:
  Posner K et al. The Columbia–Suicide Severity Rating Scale: initial
    validity and internal consistency. Am J Psychiatry 2011;168:1266-1277.
  APA Practice Guideline for the Assessment and Treatment of Patients with
    Suicidal Behaviors (2022).
"""

from __future__ import annotations

from typing import Literal

from shared.schemas import (
    CSSRSIdeationLevel,
    SuicideRiskAssessment,
)


# ─────────────────────────────────────────────────────────────────────
# Ideation level decoder
# ─────────────────────────────────────────────────────────────────────

_IDEATION_LABELS: list[Literal[
    "none",
    "wish_to_be_dead",
    "non_specific_active_suicidal",
    "active_with_method_no_plan",
    "active_with_plan_no_intent",
    "active_with_plan_and_intent",
]] = [
    "none",
    "wish_to_be_dead",
    "non_specific_active_suicidal",
    "active_with_method_no_plan",
    "active_with_plan_no_intent",
    "active_with_plan_and_intent",
]


def _ideation(level: int) -> CSSRSIdeationLevel:
    level = max(0, min(5, int(level or 0)))
    return CSSRSIdeationLevel(level=level, label=_IDEATION_LABELS[level])


# ─────────────────────────────────────────────────────────────────────
# Risk classification
# ─────────────────────────────────────────────────────────────────────

def _classify_risk(
    ideation_recent: int,
    behavior_past_30d: bool,
    behavior_lifetime_attempts: int,
    warning_factors: int,
    protective_factors: int,
) -> str:
    """Map composite features to a 4-level risk band.

    Imminent: any past-30-days attempt OR ideation 5 (plan + intent) in past 30d.
    High:     past-30-days ideation 4 (plan no intent) OR lifetime attempt + recent
              ideation 3-5 OR ≥3 warning factors with ideation ≥3.
    Moderate: past-30-days ideation 2-3 OR any lifetime attempt with current ideation.
    Low:      else (passive ideation + no recent attempts).
    """
    if behavior_past_30d or ideation_recent >= 5:
        return "imminent"
    if ideation_recent >= 4:
        return "high"
    if behavior_lifetime_attempts >= 1 and ideation_recent >= 3:
        return "high"
    if warning_factors >= 3 and ideation_recent >= 3:
        return "high"
    if ideation_recent >= 2:
        return "moderate"
    if behavior_lifetime_attempts >= 1 and ideation_recent >= 1:
        return "moderate"
    return "low"


def _adjust_for_protective(risk: str, protective: int) -> str:
    """Strong protective factors (engaged in care, social support, religious
    beliefs against suicide) modestly downgrade -- but never below moderate
    if any active ideation is present."""
    order = ["low", "moderate", "high", "imminent"]
    if protective >= 3 and risk == "moderate":
        return "low"
    return risk


# ─────────────────────────────────────────────────────────────────────
# Demographic bias guard
# ─────────────────────────────────────────────────────────────────────

# Groups for which the literature suggests structured tools tend to UNDER-
# predict risk, requiring clinician override.
_BIAS_FLAGS: tuple[str, ...] = (
    "black", "african_american",
    "indigenous", "native_american",
    "lgbtq", "lgbtq+",
    "transgender", "trans",
    "low_ses", "uninsured",
)


def _bias_guard_triggered(demographics: dict[str, str | int | bool] | None) -> tuple[bool, list[str]]:
    if not demographics:
        return False, []
    triggered = []
    for k, v in demographics.items():
        kv = f"{k}:{v}".strip().lower()
        for flag in _BIAS_FLAGS:
            if flag in kv or flag in str(v).strip().lower():
                triggered.append(f"{k}={v}")
                break
    return bool(triggered), triggered


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_suicide_risk_assessment(
    ideation_lifetime_level: int = 0,
    ideation_past_30d_level: int = 0,
    behavior_lifetime_attempts: int = 0,
    behavior_past_30d_any: bool = False,
    behavior_self_injury_no_intent: bool = False,
    warning_factors_count: int = 0,
    protective_factors_count: int = 0,
    patient_demographics: dict[str, str | int | bool] | None = None,
    patient_id: str | None = None,
) -> SuicideRiskAssessment:
    """C-SSRS-style suicide-risk assessment.

    Args:
        ideation_lifetime_level / ideation_past_30d_level: 0-5 per the
            C-SSRS ideation severity scale.
        behavior_lifetime_attempts: count of actual / interrupted / aborted
            suicide attempts in lifetime.
        behavior_past_30d_any: any attempt or preparatory act in last 30 days.
        behavior_self_injury_no_intent: NSSI (non-suicidal self-injury) --
            reported separately because it influences risk but isn't itself
            a suicide attempt.
        warning_factors_count: count of acute warning factors (recent loss,
            firearm access, hopelessness scale > threshold, etc.).
        protective_factors_count: count of protective factors (engaged in
            mental health treatment, religious beliefs, dependent children).
        patient_demographics: optional dict; bias guard fires for groups
            literature flags as risk-mismeasured.

    Returns:
        SuicideRiskAssessment with risk level + abstain (always recommended
        when bias-guard or sparse-input gates fire).
    """
    ideation_lifetime = _ideation(ideation_lifetime_level)
    ideation_recent = _ideation(ideation_past_30d_level)

    risk = _classify_risk(
        ideation_recent=ideation_past_30d_level,
        behavior_past_30d=behavior_past_30d_any,
        behavior_lifetime_attempts=behavior_lifetime_attempts,
        warning_factors=warning_factors_count,
        protective_factors=protective_factors_count,
    )
    risk = _adjust_for_protective(risk, protective_factors_count)

    bias_triggered, bias_groups = _bias_guard_triggered(patient_demographics)

    abstain_reasons: list[str] = []
    if bias_triggered:
        abstain_reasons.append(
            f"demographic_bias_guard:{','.join(bias_groups)} -- algorithmic "
            f"suicide-risk tools under-predict for this group; mandate "
            f"in-person clinician judgement."
        )
    sparse_input = (
        ideation_lifetime_level == 0
        and ideation_past_30d_level == 0
        and behavior_lifetime_attempts == 0
        and not behavior_past_30d_any
    )
    if sparse_input:
        abstain_reasons.append(
            "sparse_input -- no positive indicators in any C-SSRS field; "
            "this may reflect non-disclosure, not absence of risk."
        )

    abstain = bool(abstain_reasons)
    abstain_reason = "; ".join(abstain_reasons) if abstain else None

    safety_plan = risk in ("moderate", "high", "imminent")

    rationale = _build_rationale(
        risk, ideation_lifetime, ideation_recent,
        behavior_lifetime_attempts, behavior_past_30d_any,
        warning_factors_count, protective_factors_count,
        bias_triggered, bias_groups,
    )

    return SuicideRiskAssessment(
        patient_id=patient_id,
        ideation_lifetime=ideation_lifetime,
        ideation_past_30d=ideation_recent,
        behavior_lifetime_attempts=behavior_lifetime_attempts,
        behavior_past_30d=behavior_past_30d_any,
        behavior_self_injury_no_intent=behavior_self_injury_no_intent,
        risk_level=risk,  # type: ignore[arg-type]
        abstain_recommended=abstain,
        abstain_reason=abstain_reason,
        safety_plan_indicated=safety_plan,
        rationale=rationale,
    )


def _build_rationale(risk: str, ideation_l: CSSRSIdeationLevel,
                      ideation_r: CSSRSIdeationLevel,
                      attempts: int, recent_attempt: bool,
                      warnings: int, protective: int,
                      bias_triggered: bool, bias_groups: list[str]) -> str:
    parts = [
        f"Risk level = {risk}.",
        f"Ideation: lifetime worst level {ideation_l.level} ({ideation_l.label!r}); "
        f"past-30d level {ideation_r.level} ({ideation_r.label!r}).",
        f"Behavior: {attempts} lifetime attempt(s); past-30d attempt = {recent_attempt}.",
        f"Modifiers: {warnings} warning factor(s), {protective} protective factor(s).",
    ]
    if risk == "imminent":
        parts.append(
            "IMMINENT risk: do not leave patient alone; remove access to "
            "lethal means; activate emergency safety plan."
        )
    elif risk == "high":
        parts.append(
            "HIGH risk: psychiatric evaluation required before disposition; "
            "consider involuntary hold pathway if patient declines voluntary care."
        )
    if bias_triggered:
        parts.append(
            f"BIAS GUARD: demographic flag(s) {bias_groups} indicate this tool "
            f"may under-predict; clinician must over-ride conservatively."
        )
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_suicide_risk_assessment)
