"""IMPACT-2 -- aggregate impact KPIs across a cohort of DecisionCards.

Combines counts (action distribution, abstention rate, critic flags) with a
cost-effectiveness rollup that applies a fixed RRR assumption to every
non-`discharge_home` recommendation. The result feeds the playground
`/api/impact/cumulative` endpoint and can also be driven directly by the A2A
agent or batch consumers.

The per-decision cost-effectiveness math here mirrors what
`compute_expected_value_of_intervention` does for a single intervention; this
module is the cohort-wide rollup, not a replacement.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from shared.schemas import ImpactKPIs


# ─────────────────────── Pull-from-archive helper ───────────────────────

def _decision_card_from_outputs(outputs: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the DecisionCard slice from an archived `outputs_json` payload.

    Archive callers may store either the bare DecisionCard or a wrapper that
    embeds the card under `decision_card`/`card`. We try both, then fall back
    to assuming the dict itself is a DecisionCard.
    """
    if not isinstance(outputs, dict):
        return None
    for key in ("decision_card", "card"):
        candidate = outputs.get(key)
        if isinstance(candidate, dict):
            return candidate
    if "recommendation" in outputs or "audit" in outputs:
        return outputs
    return None


def load_archived_decisions(limit: int = 500) -> list[dict[str, Any]]:
    """Read DecisionCard-shaped dicts from the audit archive."""
    from .audit import fetch_archived_decision, list_archived_decisions

    rows = list_archived_decisions(limit=limit)
    cards: list[dict[str, Any]] = []
    for r in rows:
        full = fetch_archived_decision(r["request_id"])
        if full is None:
            continue
        card = _decision_card_from_outputs(full.get("outputs", {}))
        if card is not None:
            cards.append(card)
    return cards


# ─────────────────────── Aggregation ───────────────────────

_INTERVENTION_ACTIONS = {"home_with_care", "snf", "continued_admission"}


def _baseline_risk(card: dict[str, Any]) -> float | None:
    """Extract the calibrated readmission probability from a DecisionCard.

    Looks at `reasoning.risk_estimate.probability_mean` first (the canonical
    location), then falls back to `audit.tools[*].outputs.probability_mean`
    for older payload shapes.
    """
    reasoning = card.get("reasoning") or {}
    risk = (reasoning.get("risk_estimate") or {})
    mean = risk.get("probability_mean")
    if isinstance(mean, (int, float)):
        return float(mean)

    audit = card.get("audit") or {}
    for tool in audit.get("tools", []):
        if tool.get("tool") == "compute_readmission_risk":
            outputs = tool.get("outputs") or {}
            mean = outputs.get("probability_mean")
            if isinstance(mean, (int, float)):
                return float(mean)
    return None


def _action(card: dict[str, Any]) -> str | None:
    rec = card.get("recommendation")
    if not isinstance(rec, dict):
        return None
    a = rec.get("action")
    if isinstance(a, str):
        return a
    return None


def _confidence(card: dict[str, Any]) -> str | None:
    rec = card.get("recommendation")
    if not isinstance(rec, dict):
        return None
    c = rec.get("confidence")
    if isinstance(c, str):
        return c
    return None


def aggregate_impact_kpis(
    decisions: Iterable[dict[str, Any]] | None = None,
    *,
    intervention_relative_risk_reduction: float = 0.25,
    avoided_event_cost_usd: float = 14_000.0,
    pull_from_archive: bool = False,
    archive_limit: int = 500,
) -> ImpactKPIs:
    """Aggregate impact KPIs across a list of DecisionCard-shaped dicts.

    Args:
        decisions: explicit list of DecisionCard dicts. When None and
            `pull_from_archive` is True, reads from the audit archive.
        intervention_relative_risk_reduction: the RRR assumed for each
            non-`discharge_home` recommendation. Default 0.25 -- conservative
            midpoint of pharmacist-led counseling and transitional-care RCTs
            (Schnipper 2006 0.30; Coleman 2006 0.50; Naylor 1999 0.41).
        avoided_event_cost_usd: cost per avoided readmission. Default 14000 ≈
            HCUP 2018 average direct cost of a 30-day Medicare readmission.
        pull_from_archive: when True, also reads archived decisions from the
            audit DB (after filtering the explicit list). Useful for the
            playground dashboard which has no other state.
        archive_limit: cap for `list_archived_decisions` when pulling.

    Returns:
        ImpactKPIs with action distribution, abstention rates, average
        baseline + post-intervention risk, expected events avoided, and
        estimated cost avoided.
    """
    if not (0.0 <= intervention_relative_risk_reduction <= 1.0):
        raise ValueError(
            "intervention_relative_risk_reduction must be in [0, 1], got "
            f"{intervention_relative_risk_reduction}"
        )
    if avoided_event_cost_usd < 0:
        raise ValueError("avoided_event_cost_usd must be non-negative.")

    cards: list[dict[str, Any]] = list(decisions) if decisions is not None else []
    if pull_from_archive:
        cards.extend(load_archived_decisions(limit=archive_limit))

    n = len(cards)
    action_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    n_with_rec = 0
    n_abstained = 0
    risks: list[float] = []
    risk_for_intervention: list[float] = []  # baseline risks where action implies intervention
    n_force_abstain = 0
    n_downgrade = 0

    for card in cards:
        action = _action(card)
        if action is None:
            n_abstained += 1
        else:
            n_with_rec += 1
            action_counts[action] = action_counts.get(action, 0) + 1
            conf = _confidence(card)
            if conf is not None:
                confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

        # Critic ensemble flags
        critique = card.get("self_critique") or {}
        verdict = critique.get("verdict") if isinstance(critique, dict) else None
        if verdict == "force_abstain":
            n_force_abstain += 1
        elif verdict == "downgrade_confidence":
            n_downgrade += 1

        # Risk
        baseline = _baseline_risk(card)
        if baseline is not None:
            risks.append(baseline)
            if action in _INTERVENTION_ACTIONS:
                risk_for_intervention.append(baseline)

    avg_baseline = sum(risks) / len(risks) if risks else 0.0

    rrr = float(intervention_relative_risk_reduction)
    arr_per_intervention_patient = [r * rrr for r in risk_for_intervention]
    estimated_events_avoided = sum(arr_per_intervention_patient)

    if risks:
        # post-intervention risk: subtract the per-decision ARR from the baseline,
        # zero ARR for non-intervention actions
        adjusted = []
        for card in cards:
            b = _baseline_risk(card)
            if b is None:
                continue
            a = _action(card)
            if a in _INTERVENTION_ACTIONS:
                adjusted.append(max(0.0, b * (1.0 - rrr)))
            else:
                adjusted.append(b)
        avg_post = sum(adjusted) / len(adjusted) if adjusted else 0.0
    else:
        avg_post = 0.0

    estimated_cost_avoided = estimated_events_avoided * float(avoided_event_cost_usd)

    return ImpactKPIs(
        n_decisions=n,
        n_with_recommendation=n_with_rec,
        n_abstained=n_abstained,
        action_counts=action_counts,
        confidence_counts=confidence_counts,
        avg_baseline_risk=avg_baseline,
        avg_post_intervention_risk=avg_post,
        intervention_relative_risk_reduction=rrr,
        estimated_events_avoided=estimated_events_avoided,
        avoided_event_cost_per_event_usd=float(avoided_event_cost_usd),
        estimated_cost_avoided_usd=estimated_cost_avoided,
        n_critic_force_abstain=n_force_abstain,
        n_critic_downgrade=n_downgrade,
        generated_at_iso=datetime.now(timezone.utc).isoformat(),
    )
