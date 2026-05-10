"""healthcare.compute_massive_transfusion_protocol -- ABC score + 1:1:1 plan.

ABC score (Nunez 2009): 4 dichotomous items, ≥2 -> activate MTP:
  1. Penetrating injury mechanism
  2. SBP ≤ 90 mmHg in the field or arrival
  3. HR ≥ 120 bpm
  4. Positive FAST exam (free intra-abdominal fluid)

PROPPR trial (Holcomb 2015) established 1:1:1 plasma:platelets:RBC ratio
(approximately) as superior to 1:1:2 in survival to 24h in trauma. Initial
"first round" typically: 6 RBC + 6 FFP + 1 apheresis platelet pack.

TXA (CRASH-2 2010): tranexamic acid 1 g IV bolus + 1 g infusion over 8h
reduces mortality if given within 3h of injury; MUST NOT be given >3h
post-injury (signal of harm in CRASH-3 subgroup).
"""

from __future__ import annotations

from typing import Any, Literal

from shared.schemas import MTPDecision


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_massive_transfusion_protocol(
    penetrating_mechanism: bool = False,
    field_or_arrival_sbp_le_90: bool = False,
    heart_rate_ge_120: bool = False,
    positive_fast_exam: bool = False,
    estimated_blood_loss_ml: int | None = None,
    minutes_since_injury: int | None = None,
    pediatric_weight_kg: float | None = None,
    patient_id: str | None = None,
) -> MTPDecision:
    """Apply the ABC score for massive transfusion + plan initial blood product
    request + decide TXA eligibility.

    Args:
        penetrating_mechanism / field_or_arrival_sbp_le_90 / heart_rate_ge_120 /
        positive_fast_exam: 4 ABC items.
        estimated_blood_loss_ml: optional -- drives the size of the initial
            request when MTP is activated.
        minutes_since_injury: gates TXA eligibility (must be ≤180 min).
        pediatric_weight_kg: if provided, the initial request is sized as
            10 mL/kg per cycle of each component (pediatric trauma).

    Returns:
        MTPDecision with abc_score, mtp_activated, target_ratio,
        estimated_initial_request, txa_indicated, additional_recommendations.
    """
    components: list[str] = []
    if penetrating_mechanism:
        components.append("penetrating mechanism")
    if field_or_arrival_sbp_le_90:
        components.append("SBP ≤ 90 mmHg")
    if heart_rate_ge_120:
        components.append("HR ≥ 120 bpm")
    if positive_fast_exam:
        components.append("positive FAST exam")
    abc = len(components)
    activate = abc >= 2

    target = "1:1:1 plasma:platelets:RBC (PROPPR 2015)"

    if pediatric_weight_kg is not None and pediatric_weight_kg > 0:
        # Pediatric: 10 mL/kg per component cycle
        units = max(1, round(pediatric_weight_kg / 30))  # rough scaling
        request = {
            "rbc_units": units * 2,
            "ffp_units": units * 2,
            "platelet_packs": units,
            "cryoprecipitate_units": 0,
        }
    else:
        if not activate:
            request = {"rbc_units": 0, "ffp_units": 0,
                        "platelet_packs": 0, "cryoprecipitate_units": 0}
        elif (estimated_blood_loss_ml or 0) >= 2000:
            # Large hemorrhage -- second-round resources
            request = {"rbc_units": 6, "ffp_units": 6,
                        "platelet_packs": 1, "cryoprecipitate_units": 10}
        else:
            request = {"rbc_units": 4, "ffp_units": 4,
                        "platelet_packs": 1, "cryoprecipitate_units": 10}

    txa_window_open = (
        minutes_since_injury is not None
        and 0 <= minutes_since_injury <= 180
    )
    txa_indicated = activate and txa_window_open

    additional: list[str] = []
    if activate:
        additional.append(
            "Activate institutional MTP -- call blood bank now; serial labs "
            "(CBC, fibrinogen, ionized Ca, lactate) every 30 min."
        )
        additional.append(
            "Permissive hypotension (SBP target 80-90 mmHg) until hemorrhage "
            "controlled, EXCEPT in TBI / spinal cord injury where MAP > 80 "
            "is the target."
        )
        additional.append(
            "Calcium replacement: each unit of citrated PRBC chelates calcium; "
            "monitor ionized Ca and replace empirically (1 g calcium chloride "
            "after every 4-6 units PRBC)."
        )
        additional.append(
            "Fibrinogen surveillance: maintain > 150 mg/dL with cryoprecipitate "
            "10 units or fibrinogen concentrate."
        )
    if txa_indicated:
        additional.append(
            "Tranexamic acid 1 g IV bolus over 10 min, then 1 g infusion over 8h "
            "(CRASH-2). Window: ≤180 min from injury."
        )
    elif activate and minutes_since_injury is not None and minutes_since_injury > 180:
        additional.append(
            "TXA NOT recommended: > 180 min from injury (CRASH-2 / CRASH-3 "
            "signal of harm in late administration)."
        )

    rationale = _build_rationale(abc, components, activate, txa_indicated,
                                   minutes_since_injury, request)

    return MTPDecision(
        patient_id=patient_id,
        abc_score=abc,
        abc_components=components,
        mtp_activated=activate,
        target_ratio=target,
        estimated_initial_request=request,
        txa_indicated=txa_indicated,
        txa_window_open=txa_window_open,
        additional_recommendations=additional,
        rationale=rationale,
    )


def _build_rationale(abc: int, components: list[str], active: bool,
                       txa: bool, mins: int | None,
                       request: dict[str, int]) -> str:
    parts = [
        f"ABC score = {abc}/4 ({', '.join(components) if components else 'no items met'}).",
        f"MTP {'ACTIVATED' if active else 'not activated'}.",
    ]
    if active:
        parts.append(
            f"Initial round request: {request['rbc_units']} PRBC, "
            f"{request['ffp_units']} FFP, {request['platelet_packs']} apheresis "
            f"platelet pack(s), cryoprecipitate {request['cryoprecipitate_units']} units."
        )
    if mins is not None:
        parts.append(
            f"TXA window: {mins} min from injury -- "
            f"{'IN window (≤180 min)' if (mins <= 180) else 'OUT of window (>180 min)'}."
        )
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_massive_transfusion_protocol)
