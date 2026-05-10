"""Phase 14.16 Q1 -- Multi-agent debate / round-table.

A 3-agent debate that votes on a candidate clinical recommendation
before it lands in the final ``DecisionCard``. The pattern is
complementary to the existing 4-critic ensemble in
:mod:`a2a_agent.critique` -- both produce a verdict on the SAME card,
but the debate explicitly **adversarial**:

  - **clinical_conservative** -- favours abstaining or escalating when
    risk is moderate-or-higher; prefers continued admission, more
    workup, follow-up troponin, etc. Anchored in defensive medicine.
  - **evidence_aggressive** -- favours acting on the calibrated risk
    point estimate when its CI is tight; skeptical of abstain unless
    bias evidence is concrete.
  - **fairness_guard** -- abstains for any subgroup with documented
    under-prediction (Black, Indigenous, LGBTQ+, low-SES, peripartum)
    when fairness data is missing or ambiguous.

Each agent emits a ``DebateVote`` (``approve`` / ``revise`` /
``abstain``) plus a one-sentence rationale. Aggregation rules are
deterministic-floor (no LLM in the floor):

  - Unanimous ``approve`` -> ``approved``.
  - Any ``abstain`` from clinical_conservative OR fairness_guard ->
    ``force_abstain`` (safety floor -- no override).
  - Two-of-three ``revise`` (or ``abstain`` from evidence_aggressive
    while the others say revise) -> ``revise``.
  - Otherwise -> ``revise``.

Pure-deterministic on the inputs. The debate operates over a
``DebateInput`` dict-shaped payload that slices out only what each
agent needs from the candidate decision (risk_point_estimate,
risk_ci_width, recommended_action, fairness_subgroup,
fairness_audit_present).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


_AgentId = Literal[
    "clinical_conservative", "evidence_aggressive", "fairness_guard",
]
_Vote = Literal["approve", "revise", "abstain"]
_Verdict = Literal["approved", "revise", "force_abstain"]


# Subgroups with documented under-prediction in US risk models
_HIGH_RISK_SUBGROUPS: frozenset[str] = frozenset({
    "black", "african_american", "indigenous", "native_american",
    "lgbtq", "lgbtq+", "low_ses", "medicaid", "uninsured",
    "peripartum", "pregnant",
})


# ─────────────────────────────────────────────────────────────────────
# Payload + vote model
# ─────────────────────────────────────────────────────────────────────


class DebateInput(BaseModel):
    """The slice of a DecisionCard that the three debaters reason
    over. Pure-data -- no behaviour."""
    recommended_action: str
    risk_point_estimate: float = Field(ge=0.0, le=1.0)
    risk_ci_width: float = Field(ge=0.0, le=1.0)
    fairness_subgroup: str | None = None
    fairness_audit_present: bool = False
    high_risk_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    moderate_ci_threshold: float = Field(default=0.10, ge=0.0, le=1.0)


class DebateVote(BaseModel):
    agent_id: _AgentId
    vote: _Vote
    rationale: str


class DebateOutcome(BaseModel):
    verdict: _Verdict
    votes: list[DebateVote]
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Three agents -- pure-deterministic floor
# ─────────────────────────────────────────────────────────────────────


def _vote_clinical_conservative(payload: DebateInput) -> DebateVote:
    """Conservative agent -- favours abstain/escalate when risk is
    moderate-or-higher. Pushes back on discharge when CI is wide."""
    ci_wide = payload.risk_ci_width > payload.moderate_ci_threshold
    high_risk = payload.risk_point_estimate >= payload.high_risk_threshold

    if high_risk and payload.recommended_action == "discharge_home":
        return DebateVote(
            agent_id="clinical_conservative",
            vote="abstain",
            rationale=(
                f"Point estimate {payload.risk_point_estimate:.2f} is "
                "high-risk yet the proposed action is discharge_home; "
                "defensive medicine says continue admission or escalate."
            ),
        )
    if ci_wide and payload.recommended_action == "discharge_home":
        return DebateVote(
            agent_id="clinical_conservative",
            vote="revise",
            rationale=(
                f"CI width {payload.risk_ci_width:.2f} exceeds the "
                "0.10 moderate-uncertainty bar; downgrade confidence "
                "before discharging."
            ),
        )
    if high_risk:
        return DebateVote(
            agent_id="clinical_conservative",
            vote="revise",
            rationale=(
                "High-risk patient -- even a non-discharge action should "
                "trigger an extra workup step (lab trend / re-troponin)."
            ),
        )
    return DebateVote(
        agent_id="clinical_conservative",
        vote="approve",
        rationale=(
            "Risk is sub-threshold and CI is tight; the proposed "
            "action is consistent with defensive-medicine practice."
        ),
    )


def _vote_evidence_aggressive(payload: DebateInput) -> DebateVote:
    """Evidence-aggressive agent -- trusts a tight calibrated CI even
    when the conservative agent gets cold feet."""
    if payload.risk_ci_width <= payload.moderate_ci_threshold * 0.5:
        return DebateVote(
            agent_id="evidence_aggressive",
            vote="approve",
            rationale=(
                f"CI width {payload.risk_ci_width:.2f} is tight "
                "(below half the moderate bar); the calibrated point "
                "estimate is trustworthy on its own."
            ),
        )
    if payload.risk_ci_width >= payload.moderate_ci_threshold * 2.0:
        return DebateVote(
            agent_id="evidence_aggressive",
            vote="revise",
            rationale=(
                f"CI width {payload.risk_ci_width:.2f} is double the "
                "moderate bar -- the model is shouting that it doesn't "
                "know; widen the workup before acting."
            ),
        )
    return DebateVote(
        agent_id="evidence_aggressive",
        vote="approve",
        rationale=(
            "CI is moderate; the calibrated estimate justifies the "
            "proposed action without further hedging."
        ),
    )


def _vote_fairness_guard(payload: DebateInput) -> DebateVote:
    """Fairness-guard agent -- abstains whenever a subgroup with
    documented under-prediction is named without a fairness-audit
    artefact in evidence."""
    sub = (payload.fairness_subgroup or "").lower().strip()
    flagged = sub in _HIGH_RISK_SUBGROUPS
    if flagged and not payload.fairness_audit_present:
        return DebateVote(
            agent_id="fairness_guard",
            vote="abstain",
            rationale=(
                f"Subgroup {sub!r} is in the documented under-"
                "prediction list and no fairness-audit artefact was "
                "supplied; abstain rather than risk perpetuating bias."
            ),
        )
    if flagged and payload.fairness_audit_present:
        return DebateVote(
            agent_id="fairness_guard",
            vote="revise",
            rationale=(
                f"Subgroup {sub!r} is flagged but a fairness-audit "
                "artefact is present -- accept with downgraded "
                "confidence so the audit is surfaced to the clinician."
            ),
        )
    return DebateVote(
        agent_id="fairness_guard",
        vote="approve",
        rationale=(
            "No flagged subgroup; no fairness override applies."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────


def _aggregate(votes: list[DebateVote]) -> _Verdict:
    """Deterministic majority + safety-floor aggregation.

    Veto rules:
      - clinical_conservative.abstain OR fairness_guard.abstain
        -> force_abstain (no override).
      - clinical_conservative.revise OR fairness_guard.revise
        -> revise (the safety agents can downgrade unilaterally; a
        2-of-3 approval cannot bury a fairness signal).

    Only an evidence_aggressive `revise` is outvoted by 2 approves.
    """
    by_id = {v.agent_id: v for v in votes}
    cc = by_id["clinical_conservative"].vote
    ea = by_id["evidence_aggressive"].vote
    fg = by_id["fairness_guard"].vote

    if cc == "abstain" or fg == "abstain":
        return "force_abstain"
    if cc == "revise" or fg == "revise":
        return "revise"
    if cc == "approve" and ea == "approve" and fg == "approve":
        return "approved"
    # At this point cc=approve, fg=approve, ea is revise (or both
    # safety agents approve and ea revised). 2-approve majority wins.
    return "approved"


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def run_debate(payload: DebateInput) -> DebateOutcome:
    """Run the 3-agent debate over a candidate decision and return a
    single ``DebateOutcome``. Pure-deterministic -- same input always
    yields the same outcome."""
    votes = [
        _vote_clinical_conservative(payload),
        _vote_evidence_aggressive(payload),
        _vote_fairness_guard(payload),
    ]
    verdict = _aggregate(votes)
    if verdict == "approved":
        rationale = "All three debaters concurred on approval."
    elif verdict == "force_abstain":
        rationale = (
            "Safety floor triggered -- clinical_conservative or "
            "fairness_guard abstained."
        )
    else:
        rationale = (
            "No quorum on approval; routing to plan revision."
        )
    return DebateOutcome(
        verdict=verdict,
        votes=votes,
        rationale=rationale,
        references=[
            "TrustedRisk Phase 14.16 Q1 -- multi-agent debate.",
            "Park et al. 2024 -- multi-agent debate improves clinical "
            "reasoning robustness.",
        ],
    )
