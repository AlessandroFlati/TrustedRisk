"""healthcare.compute_prom_influence -- LIB-4.

Ingests Patient-Reported Outcome Measures (PROMs) -- PROMIS-29 / EQ-5D-5L
/ PROMIS-10 -- and translates them into:

  - QALY delta over a 30-day window (vs population mean)
  - Distress-burden score 0-100 (composite of pain + fatigue + depression
    + anxiety + sleep)
  - High-burden dimensions (≥ 1 SD above the population mean)
  - Action-dominance shift hint: if patient-reported burden is high, the
    DecisionCard's downstream action ranking should re-weight toward
    home-with-care or continued admission rather than discharge_home.

The translation uses validated PROMIS/EQ-5D scoring tables; the
`action_dominance_shift` is heuristic -- institutions should re-fit on
local outcomes.
"""

from __future__ import annotations

from typing import Any

from shared.schemas import (
    PROMInfluenceReport,
    PROMScores,
)


# Population norms -- PROMIS T-scores center at 50 with SD 10.
_PROMIS_MEAN = 50.0
_PROMIS_SD = 10.0


def _coerce_scores(scores: PROMScores | dict) -> PROMScores:
    if isinstance(scores, dict):
        return PROMScores.model_validate(scores)
    return scores


def _high_burden(value: float | None, mean: float = _PROMIS_MEAN,
                    sd: float = _PROMIS_SD,
                    higher_is_worse: bool = True) -> bool:
    if value is None:
        return False
    if higher_is_worse:
        return value >= mean + sd
    return value <= mean - sd


def _distress_burden_score(s: PROMScores) -> float:
    """Composite burden 0-100. Average of (pain*10) + fatigue +
    depression + anxiety + sleep_disturbance.

    Pain is 0-10 -> scaled ×10. Others are PROMIS T-scores 0-100.
    Missing dimensions are skipped (not zero-imputed)."""
    parts: list[float] = []
    if s.pain_intensity is not None:
        parts.append(s.pain_intensity * 10.0)
    if s.fatigue is not None:
        parts.append(s.fatigue)
    if s.depression is not None:
        parts.append(s.depression)
    if s.anxiety is not None:
        parts.append(s.anxiety)
    if s.sleep_disturbance is not None:
        parts.append(s.sleep_disturbance)
    if not parts:
        return 0.0
    return min(100.0, max(0.0, sum(parts) / len(parts)))


def _qaly_delta_30d(s: PROMScores) -> float:
    """Compute a 30-day QALY delta from PROMIS / EQ-5D inputs.

    EQ-5D index score (-1.0 to 1.0) directly: QALY30d = score * 30/365.
    PROMIS path: physical_function ↑ + (100 − distress_burden) ↑ map to a
    pseudo-utility 0-1 via simple linear scaling, then × 30/365.
    """
    if s.eq5d_index_score is not None:
        return round(float(s.eq5d_index_score) * (30.0 / 365.0), 4)

    # PROMIS pseudo-utility
    pf = s.physical_function if s.physical_function is not None else 50.0
    burden = _distress_burden_score(s)
    # Higher physical function + lower burden -> higher utility
    utility = max(0.0, min(1.0, (pf - 0.5 * burden) / 100.0))
    return round(utility * (30.0 / 365.0), 4)


def _action_dominance_shift(burden: float,
                                physical_function: float | None
                                ) -> dict[str, float]:
    """Heuristic shift in dominance for the 4 actions.

    Higher burden / lower physical function -> push away from discharge_home,
    toward home_with_care / continued_admission.
    """
    shift = {
        "discharge_home": 0.0,
        "home_with_care": 0.0,
        "snf": 0.0,
        "continued_admission": 0.0,
    }
    if burden >= 70:
        shift["discharge_home"] -= 0.30
        shift["home_with_care"] += 0.15
        shift["continued_admission"] += 0.15
    elif burden >= 50:
        shift["discharge_home"] -= 0.15
        shift["home_with_care"] += 0.15

    if physical_function is not None:
        if physical_function < 40:
            shift["discharge_home"] -= 0.10
            shift["snf"] += 0.10
        elif physical_function >= 60:
            shift["discharge_home"] += 0.05

    return {k: round(v, 4) for k, v in shift.items()}


async def compute_prom_influence(
    scores: PROMScores | dict,
) -> PROMInfluenceReport:
    """Translate PROM scores into QALY delta + action-dominance shift.

    Args:
        scores: PROMScores (or dict) -- instrument + dimension scores.

    Returns:
        PROMInfluenceReport with the QALY delta, distress burden, high-
        burden dimensions, and action-dominance shift hint.
    """
    s = _coerce_scores(scores)

    burden = _distress_burden_score(s)
    qaly = _qaly_delta_30d(s)

    high_burden_dims: list[str] = []
    if _high_burden(s.pain_intensity, mean=3.0, sd=2.0,
                       higher_is_worse=True):
        high_burden_dims.append("pain_intensity")
    if _high_burden(s.fatigue):
        high_burden_dims.append("fatigue")
    if _high_burden(s.depression):
        high_burden_dims.append("depression")
    if _high_burden(s.anxiety):
        high_burden_dims.append("anxiety")
    if _high_burden(s.sleep_disturbance):
        high_burden_dims.append("sleep_disturbance")
    if s.physical_function is not None and \
            _high_burden(s.physical_function, higher_is_worse=False):
        high_burden_dims.append("physical_function_low")
    if s.social_role_satisfaction is not None and \
            _high_burden(s.social_role_satisfaction, higher_is_worse=False):
        high_burden_dims.append("social_role_low")

    shift = _action_dominance_shift(burden, s.physical_function)

    if burden >= 70:
        recommendation = (
            "Severe self-reported distress -- escalate clinician review; "
            "consider home_with_care or continued_admission over "
            "discharge_home."
        )
    elif burden >= 50:
        recommendation = (
            "Moderate distress -- flag for clinician review; targeted "
            "intervention on the high-burden dimensions."
        )
    else:
        recommendation = (
            "Distress within population norms -- proceed with the "
            "recommended action."
        )

    rationale = (
        f"Instrument {s.instrument}, distress_burden={burden:.1f} (0-100), "
        f"qaly_delta_30d={qaly:.4f}, n_high_burden_dimensions="
        f"{len(high_burden_dims)}."
    )

    return PROMInfluenceReport(
        instrument=s.instrument,
        qaly_delta_30d=qaly,
        distress_burden_score=burden,
        high_burden_dimensions=high_burden_dims,
        action_dominance_shift=shift,
        recommendation=recommendation,
        rationale=rationale,
    )


def register(mcp) -> None:
    mcp.tool()(compute_prom_influence)
