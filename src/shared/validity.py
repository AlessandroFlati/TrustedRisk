"""Temporal validity windows for DecisionCards.

A discharge decision is a snapshot in time. As soon as a clinical state
changes -- new lab result, new vitals, new symptom -- a previously-valid
recommendation may become unsafe. TrustedRisk encodes this by attaching
`valid_until` to every RiskEstimate, with the window length scaled by
clinical-risk severity:

  - High risk (LACE >= 12):  4 hours.   Sicker patients change fast.
  - Moderate (LACE 6-11):    12 hours.  Default review-on-rounds cadence.
  - Low risk (LACE 0-5):     24 hours.  Stable patients tolerate longer windows.

The runtime applies this in two places:

  1. **At emission** -- `compute_readmission_risk` populates `valid_until` on
     the RiskEstimate it returns.
  2. **At memory hit** -- when the memory layer (`a2a_agent.memory`) finds a
     prior DecisionCard for the patient, it must check `is_card_valid()`
     before reusing it. Expired cards force a re-call.

These rules are deterministic and unit-testable -- the LLM is not in the loop.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import DecisionCard, RiskEstimate


# ─────────────────────── Default windows ───────────────────────

# (lace_min, lace_max_inclusive, minutes)
_VALIDITY_WINDOWS_BY_LACE: list[tuple[int, int, int]] = [
    (0, 5,   24 * 60),  # 24 hours
    (6, 11,  12 * 60),  # 12 hours
    (12, 19, 4 * 60),   # 4 hours
]


def compute_validity_window_minutes(lace_total: int) -> int:
    """Return the temporal-validity window length in minutes for a LACE total."""
    if lace_total < 0:
        lace_total = 0
    if lace_total > 19:
        lace_total = 19
    for lo, hi, minutes in _VALIDITY_WINDOWS_BY_LACE:
        if lo <= lace_total <= hi:
            return minutes
    # Fallback (should be unreachable given the table covers [0, 19])
    return 12 * 60


def compute_valid_until(computed_at: datetime, lace_total: int) -> datetime:
    """Compute the `valid_until` timestamp from a base time + LACE total."""
    minutes = compute_validity_window_minutes(lace_total)
    if computed_at.tzinfo is None:
        computed_at = computed_at.replace(tzinfo=timezone.utc)
    return computed_at + timedelta(minutes=minutes)


# ─────────────────────── Validity check ───────────────────────

def is_risk_estimate_valid(
    risk: "RiskEstimate",
    now: datetime | None = None,
) -> bool:
    """Return True if the RiskEstimate is still within its temporal-validity
    window at `now` (default: current UTC). When `valid_until` is None
    (legacy artifact without the field), default to True (compat)."""
    if risk.valid_until is None:
        return True
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    valid_until = risk.valid_until
    if valid_until.tzinfo is None:
        valid_until = valid_until.replace(tzinfo=timezone.utc)
    return now < valid_until


def is_card_valid(
    card: "DecisionCard",
    now: datetime | None = None,
) -> bool:
    """Return True if the DecisionCard is still safe to surface at `now`.
    A card is considered valid iff its inner RiskEstimate is valid."""
    if card.reasoning is None or card.reasoning.risk_estimate is None:
        return False
    return is_risk_estimate_valid(card.reasoning.risk_estimate, now=now)


def card_validity_remaining_minutes(
    card: "DecisionCard",
    now: datetime | None = None,
) -> int:
    """Return remaining validity in minutes (0 if expired or no risk estimate)."""
    if card.reasoning is None or card.reasoning.risk_estimate is None:
        return 0
    risk = card.reasoning.risk_estimate
    if risk.valid_until is None:
        return risk.valid_for_minutes
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    valid_until = risk.valid_until
    if valid_until.tzinfo is None:
        valid_until = valid_until.replace(tzinfo=timezone.utc)
    delta = valid_until - now
    return max(0, int(delta.total_seconds() // 60))
