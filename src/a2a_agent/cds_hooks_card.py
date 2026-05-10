"""Phase 16.H3 - CDS Hooks v1.1 Card builder.

Turns a TrustedRisk DecisionCard into a CDS Hooks v1.1 ``cards``
response payload with:
  - summary + detail + indicator
  - source (label + URL + icon)
  - **suggestions** - one per recommended action with `actions[]` of
    type create / update / delete
  - **overrideReasons** - clinician-facing list when the recommendation
    is rejected
  - **links** - SMART app launch URL pointing back to TrustedRisk's
    explanation surface

Pure-data, deterministic. Used by `apps/cds_hooks/server.py` to build
its response and by tests to verify card shape conformance.

References:
- HL7 CDS Hooks Specification v1.1 (2024-08).
- AHRQ CDS Connect Card Catalogue.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


_Indicator = Literal["info", "warning", "critical"]


class CDSHooksAction(BaseModel):
    type: Literal["create", "update", "delete"]
    description: str
    resource: dict[str, Any] | None = None
    resourceId: str | None = None


class CDSHooksSuggestion(BaseModel):
    label: str
    uuid: str
    isRecommended: bool = False
    actions: list[CDSHooksAction] = Field(default_factory=list)


class CDSHooksOverrideReason(BaseModel):
    code: dict[str, str]
    display: str


class CDSHooksLink(BaseModel):
    label: str
    url: str
    type: Literal["absolute", "smart"]
    appContext: str | None = None


class CDSHooksCard(BaseModel):
    summary: str
    detail: str
    indicator: _Indicator
    source: dict[str, str]
    suggestions: list[CDSHooksSuggestion] = Field(default_factory=list)
    overrideReasons: list[CDSHooksOverrideReason] = Field(default_factory=list)
    links: list[CDSHooksLink] = Field(default_factory=list)


class CDSHooksResponse(BaseModel):
    cards: list[CDSHooksCard]
    systemActions: list[dict[str, Any]] = Field(default_factory=list)


_DEFAULT_OVERRIDE_REASONS: list[CDSHooksOverrideReason] = [
    CDSHooksOverrideReason(
        code={"code": "patient-preference",
              "system": "https://trustedrisk.local/cds/override"},
        display="Patient preference / shared decision-making",
    ),
    CDSHooksOverrideReason(
        code={"code": "clinician-judgment",
              "system": "https://trustedrisk.local/cds/override"},
        display="Clinician judgment based on context not in chart",
    ),
    CDSHooksOverrideReason(
        code={"code": "false-positive",
              "system": "https://trustedrisk.local/cds/override"},
        display="System flagged false positive",
    ),
    CDSHooksOverrideReason(
        code={"code": "social-determinants",
              "system": "https://trustedrisk.local/cds/override"},
        display="Social determinants override",
    ),
]


def _indicator_for(action: str, risk: float) -> _Indicator:
    if action == "abstain" or risk >= 0.30:
        return "warning"
    if risk >= 0.20:
        return "warning"
    return "info"


def _summary_for(action: str, risk: float) -> str:
    return (
        f"TrustedRisk: 30-day readmission risk "
        f"{risk*100:.1f}% - recommended: {action}"
    )


def build_decision_card(
    *,
    patient_id: str,
    encounter_id: str | None,
    recommended_action: str,
    risk_point_estimate: float,
    risk_ci95: tuple[float, float] | None = None,
    rationale: str = "",
    explanation_url: str = (
        "https://trustedrisk.local/explain"
    ),
) -> CDSHooksCard:
    """Compose the CDS Hooks card for a single discharge decision."""
    if not 0.0 <= risk_point_estimate <= 1.0:
        raise ValueError("risk_point_estimate must be in [0,1]")
    detail_lines = [
        f"**Recommended action**: `{recommended_action}`.",
        f"**Risk point estimate**: {risk_point_estimate*100:.2f}%.",
    ]
    if risk_ci95 is not None:
        detail_lines.append(
            f"**95% CI**: ({risk_ci95[0]*100:.2f}%, "
            f"{risk_ci95[1]*100:.2f}%)."
        )
    if rationale:
        detail_lines.append(f"**Rationale**: {rationale}")
    detail = "\n".join(detail_lines)

    enc_part = f"&encounter={encounter_id}" if encounter_id else ""
    smart_link = (
        f"{explanation_url}?patient={patient_id}{enc_part}"
    )

    actions: list[CDSHooksAction] = []
    if recommended_action != "abstain":
        actions.append(CDSHooksAction(
            type="update",
            description=(
                f"Update Encounter status to reflect "
                f"{recommended_action}."
            ),
            resourceId=(
                f"Encounter/{encounter_id}" if encounter_id else None
            ),
        ))
    suggestion = CDSHooksSuggestion(
        label=f"Apply {recommended_action}",
        uuid=str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"trustedrisk:{patient_id}:{recommended_action}",
        )),
        isRecommended=True, actions=actions,
    )

    return CDSHooksCard(
        summary=_summary_for(recommended_action, risk_point_estimate),
        detail=detail,
        indicator=_indicator_for(
            recommended_action, risk_point_estimate),
        source={
            "label": "TrustedRisk - calibrated discharge support",
            "url": "https://trustedrisk.local",
            "icon": "https://trustedrisk.local/icon.png",
        },
        suggestions=[suggestion],
        overrideReasons=list(_DEFAULT_OVERRIDE_REASONS),
        links=[
            CDSHooksLink(
                label="Explain (SMART app)",
                url=smart_link, type="smart",
                appContext=(
                    f"trustedrisk:{patient_id}"
                ),
            ),
            CDSHooksLink(
                label="Audit trail",
                url=(
                    f"https://trustedrisk.local/audit?patient="
                    f"{patient_id}"
                ),
                type="absolute",
            ),
        ],
    )


def build_response(
    cards: list[CDSHooksCard],
) -> CDSHooksResponse:
    return CDSHooksResponse(cards=cards)
