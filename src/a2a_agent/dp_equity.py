"""AUDIT-2 -- Differential privacy on the population equity dashboard.

Adds calibrated Laplace noise to the integer count fields of an
`EquityDashboard` to prevent re-identification of small subgroups.
Segments below a configurable suppression threshold (default 5) are
suppressed entirely.

Privacy budget:
  - per-segment ε = epsilon (default 1.0)
  - sensitivity = 1 (count) -- adding/removing one record changes any
    count by at most 1
  - Laplace noise scale = sensitivity / ε

The output preserves the EquityDashboard.segments structure but replaces
counts with noised values (rounded to integers, clipped to ≥ 0). Rates
(intervention_rate, abstention_rate) are recomputed from noised counts so
they remain consistent.
"""

from __future__ import annotations

import os
import random
from typing import Any

from shared.schemas import (
    DPNoisedEquityDashboard,
    EquityDashboard,
    EquitySegment,
)


def _laplace_noise(scale: float, rng: random.Random) -> float:
    """Draw a Laplace(0, scale) variate via inverse-CDF."""
    u = rng.random() - 0.5
    sign = 1.0 if u >= 0 else -1.0
    return -scale * sign * (math_log(1 - 2 * abs(u)))


def math_log(x: float) -> float:
    import math
    return math.log(x) if x > 0 else 0.0


def _noised_count(value: int, scale: float,
                    rng: random.Random) -> int:
    return max(0, int(round(value + _laplace_noise(scale, rng))))


def _coerce_dashboard(d: EquityDashboard | dict[str, Any]
                          ) -> EquityDashboard:
    if isinstance(d, dict):
        return EquityDashboard.model_validate(d)
    return d


def compute_dp_equity_dashboard(
    dashboard: EquityDashboard | dict[str, Any],
    *,
    epsilon: float = 1.0,
    suppression_threshold: int = 5,
    seed: int | None = None,
) -> DPNoisedEquityDashboard:
    """Apply Laplace-mechanism DP to an EquityDashboard.

    Args:
        dashboard: EquityDashboard (or dict).
        epsilon: per-segment privacy budget. Lower = more noise.
        suppression_threshold: segments with raw n_decisions <
            this value are dropped.
        seed: optional RNG seed for reproducible tests.

    Returns:
        DPNoisedEquityDashboard with noised counts + suppressed segments.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")
    if suppression_threshold < 0:
        raise ValueError("suppression_threshold must be >= 0.")

    d = _coerce_dashboard(dashboard)
    sensitivity = 1.0
    scale = sensitivity / epsilon

    rng = random.Random(seed) if seed is not None else random.Random()

    out_segments: list[dict[str, Any]] = []
    suppressed: list[str] = []
    for seg in d.segments:
        if seg.n_decisions < suppression_threshold:
            suppressed.append(f"{seg.subgroup_dimension}={seg.subgroup_value}")
            continue
        n_noised = _noised_count(seg.n_decisions, scale, rng)
        n_abst = _noised_count(seg.n_abstained, scale, rng)
        # Recompute rates from noised counts; clip to [0, 1]
        intervention_rate = (
            min(1.0, max(0.0,
                            seg.intervention_rate + _laplace_noise(
                                scale / max(seg.n_decisions, 1), rng)))
        )
        abstention_rate = n_abst / max(1, n_noised)
        out_segments.append({
            "subgroup_dimension": seg.subgroup_dimension,
            "subgroup_value": seg.subgroup_value,
            "n_decisions_noised": n_noised,
            "n_abstained_noised": n_abst,
            "intervention_rate_noised": round(intervention_rate, 4),
            "abstention_rate_noised": round(abstention_rate, 4),
            "avg_risk": seg.avg_risk,
        })

    n_total_noised = _noised_count(d.n_total_decisions, scale, rng)

    rationale = (
        f"DP Laplace mechanism, ε={epsilon}, sensitivity={sensitivity}. "
        f"Suppression threshold {suppression_threshold} (segments with "
        f"raw n < threshold are dropped). "
        f"Suppressed {len(suppressed)} segments; emitted "
        f"{len(out_segments)} segments."
    )

    return DPNoisedEquityDashboard(
        epsilon=epsilon,
        sensitivity=sensitivity,
        n_total_decisions_noised=n_total_noised,
        segments=out_segments,
        suppressed_segments=suppressed,
        suppression_threshold=suppression_threshold,
        rationale=rationale,
    )
