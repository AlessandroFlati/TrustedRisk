"""Plan revision orchestrator (AMB-5.2).

Wraps the standard pipeline (compose DecisionCard -> critic vote -> apply
verdict) in a bounded retry loop. When the critic ensemble emits
`request_replay`, the orchestrator re-runs the pipeline with the critic
rationale appended as a "caveat" the second pass should attend to.

Design constraints:
  - Bounded retries: max 2 replays (3 attempts total) per request -- beyond
    that, collapse to `force_abstain` (safest).
  - Attempt 0 (initial): standard pipeline.
  - Attempt 1+ (replay): same pipeline, with the previous attempt's
    critique rationale prepended to the input -- the upstream composer is
    expected to read the caveat and adjust (e.g., re-invoke a missing tool).
  - Audit trail: every attempt's CritiqueDecision is preserved in
    `attempts` list of the final RevisionTrace.

This module is decoupled from the ADK LlmAgent -- it works against any
async `compose_fn(input, caveats) -> DecisionCard` and any async
`critic_ensemble_fn(card) -> CriticEnsembleVerdict`. The Streamlit
playground (AMB-2) uses this module directly without the LLM layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from shared.schemas import (
    CriticEnsembleVerdict,
    CritiqueDecision,
    DecisionCard,
)

from .critique import apply_critique, collapse_ensemble_to_single_critique


@dataclass
class RevisionAttempt:
    """One attempt in a plan-revision sequence."""
    attempt_index: int             # 0 = initial, ≥1 = replay
    candidate_card: DecisionCard
    ensemble: CriticEnsembleVerdict
    final_card_after_apply: DecisionCard
    caveats_passed_in: list[str] = field(default_factory=list)


@dataclass
class RevisionTrace:
    """Outcome of a plan-revision orchestration."""
    attempts: list[RevisionAttempt]
    final_card: DecisionCard
    revisions_used: int
    revisions_budget: int
    terminated_reason: str         # "approved" / "force_abstain" / "budget_exhausted"


ComposeFn = Callable[[Any, list[str]], Awaitable[DecisionCard]]
CriticEnsembleFn = Callable[[DecisionCard], Awaitable[CriticEnsembleVerdict]]


async def run_with_plan_revision(
    compose_fn: ComposeFn,
    critic_ensemble_fn: CriticEnsembleFn,
    initial_input: Any,
    *,
    max_replays: int = 2,
) -> RevisionTrace:
    """Run a compose-then-critique pipeline with bounded plan revision.

    Args:
        compose_fn: async (input, caveats: list[str]) -> DecisionCard.
            Caveats from a prior critic vote are passed back in for the
            next attempt.
        critic_ensemble_fn: async DecisionCard -> CriticEnsembleVerdict.
            Returns 3-critic vote.
        initial_input: opaque input handed to compose_fn for attempt 0.
        max_replays: max number of replays after the initial attempt
            (default 2 -> 3 total attempts).

    Returns:
        RevisionTrace with all attempts + final_card + termination reason.
    """
    attempts: list[RevisionAttempt] = []
    caveats: list[str] = []
    final_card: DecisionCard | None = None
    terminated_reason: str = "approved"

    for attempt_index in range(max_replays + 1):
        retries_remaining = max_replays - attempt_index

        candidate = await compose_fn(initial_input, list(caveats))

        ensemble = await critic_ensemble_fn(candidate)
        # Override retries_remaining post-hoc since the critic_ensemble_fn
        # may not know the budget -- the orchestrator owns this.
        ensemble = CriticEnsembleVerdict.model_validate({
            **ensemble.model_dump(mode="json"),
            "retries_remaining": retries_remaining,
            # Re-apply the budget-aware applied_verdict logic here:
            "applied_verdict": (
                "force_abstain"
                if (ensemble.aggregated_verdict == "request_replay"
                      and retries_remaining <= 0)
                else ensemble.aggregated_verdict
            ),
        })

        single = collapse_ensemble_to_single_critique(ensemble)
        applied_card = apply_critique(candidate, single)

        attempts.append(RevisionAttempt(
            attempt_index=attempt_index,
            candidate_card=candidate,
            ensemble=ensemble,
            final_card_after_apply=applied_card,
            caveats_passed_in=list(caveats),
        ))

        final_card = applied_card

        if ensemble.applied_verdict == "request_replay":
            # Will retry -- append caveats from this attempt's rationale
            caveat = (
                f"[attempt {attempt_index} request_replay] "
                f"{ensemble.rationale[:300]}"
            )
            caveats.append(caveat)
            continue

        if ensemble.applied_verdict == "approved":
            terminated_reason = "approved"
        elif ensemble.applied_verdict == "downgrade_confidence":
            terminated_reason = "downgrade_confidence"
        else:  # force_abstain
            terminated_reason = "force_abstain"
        break
    else:
        # Exhausted all attempts via replay -- final attempt's
        # applied_verdict will already be force_abstain (budget rule)
        terminated_reason = "budget_exhausted"

    assert final_card is not None
    revisions_used = max(0, len(attempts) - 1)
    return RevisionTrace(
        attempts=attempts,
        final_card=final_card,
        revisions_used=revisions_used,
        revisions_budget=max_replays,
        terminated_reason=terminated_reason,
    )
