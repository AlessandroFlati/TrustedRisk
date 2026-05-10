"""SIM-2 -- Population equity dashboard.

Cohort-wide equity rollup that complements the per-patient fairness_audit
tool. Slices a list of (DecisionCard, demographics) pairs by:
  - age_band  (0-17, 18-44, 45-64, 65-74, 75-84, 85+)
  - race      (whatever literal strings the caller supplies)
  - sex       (male / female / other)
  - insurance_type

For each slice it computes intervention rate (non-discharge_home / total),
abstention rate, average baseline risk, and high-confidence rate. Then
surfaces segments with disparity above a tunable threshold.

Note: descriptive metrics only -- disparity ≠ bias. The dashboard is a
screening surface; algorithmic-bias root-cause analysis lives elsewhere
(per the included Obermeyer 2019 reference).
"""

from __future__ import annotations

from typing import Any, Iterable

from shared.schemas import EquityDashboard, EquitySegment


_AGE_BANDS: list[tuple[int, int, str]] = [
    (0, 17, "0-17"),
    (18, 44, "18-44"),
    (45, 64, "45-64"),
    (65, 74, "65-74"),
    (75, 84, "75-84"),
    (85, 130, "85+"),
]


def _age_band(age: int | None) -> str | None:
    if age is None:
        return None
    for lo, hi, lbl in _AGE_BANDS:
        if lo <= age <= hi:
            return lbl
    return None


def _action_of(card: dict[str, Any]) -> str | None:
    rec = card.get("recommendation")
    if isinstance(rec, dict):
        a = rec.get("action")
        if isinstance(a, str):
            return a
    return None


def _confidence_of(card: dict[str, Any]) -> str | None:
    rec = card.get("recommendation")
    if isinstance(rec, dict):
        c = rec.get("confidence")
        if isinstance(c, str):
            return c
    return None


def _risk_of(card: dict[str, Any]) -> float | None:
    reasoning = card.get("reasoning") or {}
    risk = reasoning.get("risk_estimate") or {}
    v = risk.get("probability_mean")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _bucket_value(dim: str, demo: dict[str, Any]) -> str | None:
    if dim == "age_band":
        a = demo.get("age")
        return _age_band(int(a)) if isinstance(a, (int, float)) else None
    if dim == "race":
        v = demo.get("race")
        return str(v).strip().lower() if isinstance(v, str) and v.strip() \
            else None
    if dim == "sex":
        v = demo.get("sex") or demo.get("gender")
        if isinstance(v, str) and v.strip():
            v = v.strip().lower()
            return v if v in {"male", "female", "other"} else None
        return None
    if dim == "insurance_type":
        v = demo.get("insurance_type") or demo.get("insurance")
        return str(v).strip().lower() if isinstance(v, str) and v.strip() \
            else None
    return None


_INTERVENTION_ACTIONS = {"home_with_care", "snf", "continued_admission"}


def compute_equity_dashboard(
    cohort: Iterable[dict[str, Any]],
    *,
    disparity_threshold_pct: float = 0.15,
) -> EquityDashboard:
    """Aggregate per-segment equity metrics across a cohort.

    Args:
        cohort: list of `{"decision_card": <card>, "demographics": <dict>}`
            entries. Cards without a recommendation count as abstained.
        disparity_threshold_pct: a segment is flagged when its intervention
            or abstention rate diverges by more than this absolute fraction
            from the cohort-wide mean. Default 0.15 (= 15 percentage points).

    Returns:
        EquityDashboard with per-segment slices, disparity max-values, and
        a flagged_segments list.
    """
    if not (0.0 <= disparity_threshold_pct <= 1.0):
        raise ValueError("disparity_threshold_pct must be in [0, 1].")

    items = list(cohort)
    n_total = len(items)

    # Tally cohort-wide intervention/abstention rates for the disparity baseline
    n_intervention = 0
    n_abstain_total = 0
    for entry in items:
        card = entry.get("decision_card") or {}
        action = _action_of(card)
        if action is None:
            n_abstain_total += 1
        elif action in _INTERVENTION_ACTIONS:
            n_intervention += 1
    cohort_intervention_rate = (
        n_intervention / n_total if n_total > 0 else 0.0)
    cohort_abstention_rate = (
        n_abstain_total / n_total if n_total > 0 else 0.0)

    segments: list[EquitySegment] = []
    flagged: list[str] = []

    # We build per-(dim, value) buckets and one EquitySegment per bucket
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in items:
        demo = entry.get("demographics") or {}
        for dim in ("age_band", "race", "sex", "insurance_type"):
            value = _bucket_value(dim, demo)
            if value is None:
                continue
            buckets.setdefault((dim, value), []).append(entry)

    for (dim, value), bucket in sorted(buckets.items()):
        n = len(bucket)
        action_counts: dict[str, int] = {}
        n_abstain = 0
        n_intervention_local = 0
        n_high_conf = 0
        risks: list[float] = []
        for entry in bucket:
            card = entry.get("decision_card") or {}
            action = _action_of(card)
            if action is None:
                n_abstain += 1
            else:
                action_counts[action] = action_counts.get(action, 0) + 1
                if action in _INTERVENTION_ACTIONS:
                    n_intervention_local += 1
                if _confidence_of(card) == "high":
                    n_high_conf += 1
            r = _risk_of(card)
            if r is not None:
                risks.append(r)

        # avg_risk is None when no card in this segment carried a risk
        # estimate. Reporting 0.0 would let a downstream consumer
        # interpret a data-poor segment as a "low risk" cohort.
        avg_risk = (sum(risks) / len(risks)) if risks else None
        intervention_rate = n_intervention_local / n if n > 0 else 0.0
        abstention_rate = n_abstain / n if n > 0 else 0.0
        high_conf_rate = n_high_conf / n if n > 0 else 0.0

        seg = EquitySegment(
            subgroup_dimension=dim,    # type: ignore[arg-type]
            subgroup_value=value,
            n_decisions=n,
            action_counts=action_counts,
            n_abstained=n_abstain,
            n_with_risk=len(risks),
            avg_risk=avg_risk,
            intervention_rate=intervention_rate,
            abstention_rate=abstention_rate,
            confidence_high_rate=high_conf_rate,
        )
        segments.append(seg)

        if abs(intervention_rate - cohort_intervention_rate) \
                > disparity_threshold_pct:
            flagged.append(f"{dim}={value}|intervention_rate")
        if abs(abstention_rate - cohort_abstention_rate) \
                > disparity_threshold_pct:
            flagged.append(f"{dim}={value}|abstention_rate")

    # Disparity ranges across segments
    if segments:
        max_int_disparity = max(
            abs(s.intervention_rate - cohort_intervention_rate)
            for s in segments)
        max_abs_disparity = max(
            abs(s.abstention_rate - cohort_abstention_rate)
            for s in segments)
        # Exclude segments with avg_risk=None from the disparity range:
        # they have no risk data, comparing them as 0.0 would manufacture
        # a false disparity signal.
        risks_per_seg = [s.avg_risk for s in segments
                            if s.n_decisions > 0 and s.avg_risk is not None]
        max_risk_disparity = (max(risks_per_seg) - min(risks_per_seg)) \
            if risks_per_seg else 0.0
    else:
        max_int_disparity = 0.0
        max_abs_disparity = 0.0
        max_risk_disparity = 0.0

    rationale = (
        f"Cohort intervention rate {cohort_intervention_rate:.2f} "
        f"(threshold ±{disparity_threshold_pct:.2f}). "
        f"{len(flagged)} segment-metric pairs flagged across "
        f"{len(segments)} segments."
    )

    return EquityDashboard(
        n_total_decisions=n_total,
        segments=segments,
        max_intervention_rate_disparity=float(max_int_disparity),
        max_abstention_rate_disparity=float(max_abs_disparity),
        max_avg_risk_disparity=float(max_risk_disparity),
        rationale=rationale,
        flagged_segments=flagged,
    )
