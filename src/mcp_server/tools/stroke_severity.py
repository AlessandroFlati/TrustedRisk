"""healthcare.compute_stroke_severity -- NIH Stroke Scale (0-42).

The NIHSS is the field-standard validated stroke severity score used in
every randomized stroke trial since 1995 (NINDS tPA). 15 items, each scored
0-2/3/4 depending on the item. Total 0-42 with empirical bands:
    0-4    minor stroke
    5-15   moderate
    16-20  moderate-severe
    21+    severe

LVO indicator: NIHSS ≥6 with cortical signs (gaze, language, neglect)
strongly suggests proximal MCA / ICA / basilar occlusion -> EVT candidate.

References:
  Brott T et al. Measurements of acute cerebral infarction: a clinical
    examination scale. Stroke 1989;20:864-870.
  AHA/ASA 2019 Guidelines for the Early Management of AIS.
"""

from __future__ import annotations

from typing import Any, Literal

from shared.schemas import NIHSSItem, NIHSSReport


# ─────────────────────────────────────────────────────────────────────
# Item-name -> max-points mapping (for input validation)
# ─────────────────────────────────────────────────────────────────────

_ITEM_MAX: dict[str, int] = {
    "loc_responsiveness": 3,
    "loc_questions": 2,
    "loc_commands": 2,
    "best_gaze": 2,
    "visual_fields": 3,
    "facial_palsy": 3,
    "motor_arm_left": 4,
    "motor_arm_right": 4,
    "motor_leg_left": 4,
    "motor_leg_right": 4,
    "limb_ataxia": 2,
    "sensory": 2,
    "best_language": 3,
    "dysarthria": 2,
    "extinction_inattention": 2,
}

# "Cortical signs" -- items whose presence at score ≥1 strongly suggests
# anterior-circulation LVO when combined with NIHSS ≥6.
_CORTICAL_ITEMS = {"best_gaze", "best_language", "extinction_inattention",
                    "visual_fields"}


# ─────────────────────────────────────────────────────────────────────
# Severity classification
# ─────────────────────────────────────────────────────────────────────

def _classify_severity(total: int) -> str:
    if total >= 21:
        return "severe"
    if total >= 16:
        return "moderate_severe"
    if total >= 5:
        return "moderate"
    return "minor"


def _recommended_response(total: int, lvo: bool, last_known_well_min: int | None
                           ) -> str:
    """Map severity + LVO + time-since-LKW to a disposition recommendation.
    Time gates the type of reperfusion (tPA vs EVT) but here we surface the
    'evaluation' tier; the dedicated thrombolysis tool decides eligibility."""
    if total < 4:
        return "outpatient_workup"
    if last_known_well_min is None or last_known_well_min > 1440:
        # Out of any reperfusion window
        return "stroke_unit_admission"
    if lvo and last_known_well_min <= 1440:
        return "endovascular_thrombectomy_evaluation"
    if last_known_well_min <= 270:  # 4.5h
        return "thrombolysis_evaluation"
    return "stroke_unit_admission"


def _build_item_rationale(item: str, points: int) -> str:
    if points == 0:
        return f"{item}: 0 (normal/no deficit)"
    return f"{item}: {points} (deficit present)"


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_stroke_severity(
    item_scores: dict[str, int],
    last_known_well_minutes_ago: int | None = None,
    patient_id: str | None = None,
) -> NIHSSReport:
    """Compute NIHSS total + severity tier + LVO indicator + disposition.

    Args:
        item_scores: dict mapping the 15 NIHSS items to their points (0-4).
            Missing items are treated as 0 (no deficit) for totaling, but
            the rationale flags untested items separately.
        last_known_well_minutes_ago: minutes since last-known-well -- used to
            shape the recommended_response (does not affect totals).
        patient_id: opaque ID for audit.

    Returns:
        NIHSSReport with score_total, severity_tier, lvo_suspected,
        recommended_response, per-item breakdown.
    """
    items: list[NIHSSItem] = []
    total = 0
    cortical_positive = False

    for item_name, max_pts in _ITEM_MAX.items():
        raw = item_scores.get(item_name, 0)
        try:
            pts = int(raw)
        except (TypeError, ValueError):
            pts = 0
        pts = max(0, min(max_pts, pts))
        total += pts
        items.append(NIHSSItem(
            item=item_name,  # type: ignore[arg-type]
            points=pts,
            rationale=_build_item_rationale(item_name, pts),
        ))
        if item_name in _CORTICAL_ITEMS and pts >= 1:
            cortical_positive = True

    severity = _classify_severity(total)
    lvo = (total >= 6 and cortical_positive)
    response = _recommended_response(total, lvo, last_known_well_minutes_ago)

    rationale = _build_overall_rationale(total, severity, lvo, response,
                                            last_known_well_minutes_ago, items)

    return NIHSSReport(
        patient_id=patient_id,
        score_total=total,
        items=items,
        severity_tier=severity,  # type: ignore[arg-type]
        lvo_suspected=lvo,
        recommended_response=response,  # type: ignore[arg-type]
        rationale=rationale,
    )


def _build_overall_rationale(total: int, severity: str, lvo: bool,
                                response: str, lkw_min: int | None,
                                items: list[NIHSSItem]) -> str:
    parts = [f"NIHSS total = {total} ({severity} tier)."]
    if lvo:
        parts.append(
            "LVO suspected: NIHSS ≥6 with cortical signs (gaze / language / "
            "neglect / visual field cuts) -- anterior-circulation large-vessel "
            "occlusion is a high pre-test probability."
        )
    nonzero = [it for it in items if it.points > 0]
    if nonzero:
        top = sorted(nonzero, key=lambda c: c.points, reverse=True)[:5]
        items_str = ", ".join(f"{it.item}=+{it.points}" for it in top)
        parts.append(f"Top contributing items: {items_str}.")
    if lkw_min is not None:
        parts.append(f"Last-known-well: {lkw_min} min ago.")
    parts.append(f"Recommended response: {response}.")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_stroke_severity)
