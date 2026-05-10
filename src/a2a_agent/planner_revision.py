"""Phase 14.15 Q2 -- Planner revision feedback loop.

Wraps :func:`a2a_agent.planner.plan_tool_use` in a bounded critic->revise
loop that mirrors the pattern in :mod:`a2a_agent.plan_revision` (which
revises DecisionCards) but operates on ``ToolUsePlan`` instead.

Why a planner-level loop?

  - The DecisionCard revision loop in ``plan_revision.py`` runs *after*
    the agent has already executed every tool. By that point the
    expensive work is sunk cost.
  - A planner-level loop catches mistakes (missing safety step, redundant
    duplicate tool, bundle drift across steps) *before* any tool is
    invoked. Cheap, deterministic, and bounded.

Components
----------

  - **Critic** -- pure-deterministic ``critique_plan(plan)`` that emits
    a ``PlannerCritique`` enumerating issues + a single verdict
    (``approved`` / ``revise`` / ``force_abstain``).
  - **Revisor** -- ``revise_plan(plan, critique)`` produces a revised
    ``ToolUsePlan`` by applying the critique's suggested edits.
  - **Loop** -- ``run_with_planner_revision(plan_fn, query, max_iterations)``
    runs ``plan_fn`` then iterates critique->revise until ``approved`` or
    the iteration budget is exhausted.

Pure-deterministic. The deterministic floor catches three concrete
issue types: missing-PHI-scrub, duplicate-tool, and orphan-bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal

from pydantic import BaseModel, Field

from shared.schemas import ToolUsePlan, ToolUsePlanStep


# ─────────────────────────────────────────────────────────────────────
# Critique data model
# ─────────────────────────────────────────────────────────────────────


_IssueCode = Literal[
    "missing_phi_scrub_for_phi_query",
    "duplicate_tool_in_plan",
    "orphan_bundle_for_specialist",
    "no_steps_planned",
    "mandatory_step_dropped",
    "redundant_optional_step_after_intent",
]


class PlannerIssue(BaseModel):
    code: _IssueCode
    message: str
    affected_step_ids: list[str] = Field(default_factory=list)
    suggested_action: str


class PlannerCritique(BaseModel):
    verdict: Literal["approved", "revise", "force_abstain"]
    issues: list[PlannerIssue] = Field(default_factory=list)
    rationale: str


# ─────────────────────────────────────────────────────────────────────
# Critic -- pure deterministic
# ─────────────────────────────────────────────────────────────────────


def _detect_issues(plan: ToolUsePlan) -> list[PlannerIssue]:
    issues: list[PlannerIssue] = []

    # 1. No steps at all
    if plan.n_steps == 0:
        issues.append(PlannerIssue(
            code="no_steps_planned",
            message=(
                "The planner produced no steps. The user query may need "
                "TF-IDF retrieval to surface candidates."
            ),
            affected_step_ids=[],
            suggested_action=(
                "Re-run the planner with enable_tfidf_retrieval=True "
                "or expand the curated intent rules."
            ),
        ))

    # 2. PHI markers in the query but no detect_phi step
    import re
    phi_marker = re.compile(
        r"\b(mrn|ssn|dob|phone|email|address|zip|patient\s+name|"
        r"social[\s-]?security|medical\s+record\s+number)\b",
        re.IGNORECASE,
    )
    has_phi_marker = phi_marker.search(plan.user_query) is not None
    has_phi_scrub = any(s.tool == "detect_phi" for s in plan.steps)
    if has_phi_marker and not has_phi_scrub:
        issues.append(PlannerIssue(
            code="missing_phi_scrub_for_phi_query",
            message=(
                "The user query contains PHI markers but no detect_phi "
                "step was emitted. The mandatory safety floor must "
                "always include PHI scrubbing on free-text PHI input."
            ),
            affected_step_ids=[],
            suggested_action=(
                "Inject a mandatory detect_phi step at position 0."
            ),
        ))

    # 3. Duplicate tools
    seen: dict[str, str] = {}
    duplicate_step_ids: list[str] = []
    for s in plan.steps:
        if s.tool in seen:
            duplicate_step_ids.append(s.step_id)
        else:
            seen[s.tool] = s.step_id
    if duplicate_step_ids:
        issues.append(PlannerIssue(
            code="duplicate_tool_in_plan",
            message=(
                f"{len(duplicate_step_ids)} step(s) re-invoke a tool that "
                "another step already covers. Drop the duplicates."
            ),
            affected_step_ids=duplicate_step_ids,
            suggested_action=(
                "Keep the first occurrence (or the mandatory one) and "
                "drop subsequent duplicates."
            ),
        ))

    # 4. Orphan bundle: bundle name not in BUNDLES
    try:
        from mcp_server.tools import BUNDLES
        bundle_keys = set(BUNDLES.keys())
        orphan_step_ids = [
            s.step_id for s in plan.steps if s.bundle not in bundle_keys
        ]
        if orphan_step_ids:
            issues.append(PlannerIssue(
                code="orphan_bundle_for_specialist",
                message=(
                    f"{len(orphan_step_ids)} step(s) reference a bundle "
                    "id that isn't registered in mcp_server.tools.BUNDLES."
                ),
                affected_step_ids=orphan_step_ids,
                suggested_action=(
                    "Re-route the affected step to its tool's first "
                    "registered bundle, or drop the step."
                ),
            ))
    except ImportError:
        pass

    # 5. Redundant TF-IDF advisory steps after a clinical intent matched
    has_intent = any(s.step_id.startswith("floor-int-") for s in plan.steps)
    redundant_tfidf_ids = [
        s.step_id for s in plan.steps
        if s.step_id.startswith("floor-tfidf-")
    ]
    if has_intent and redundant_tfidf_ids:
        issues.append(PlannerIssue(
            code="redundant_optional_step_after_intent",
            message=(
                "TF-IDF advisory steps surfaced even though a curated "
                "intent rule already matched. The advisory steps are "
                "noise and should be dropped."
            ),
            affected_step_ids=redundant_tfidf_ids,
            suggested_action="Drop the floor-tfidf-* steps.",
        ))

    return issues


def critique_plan(plan: ToolUsePlan) -> PlannerCritique:
    """Pure-deterministic plan critic. Emits ``approved`` when the plan
    has no issues, ``revise`` for fixable issues, and ``force_abstain``
    only for the ``no_steps_planned`` case where there's nothing to
    revise into."""
    issues = _detect_issues(plan)
    if not issues:
        return PlannerCritique(
            verdict="approved",
            issues=[],
            rationale="No deterministic-floor issues detected.",
        )
    if any(i.code == "no_steps_planned" for i in issues):
        return PlannerCritique(
            verdict="force_abstain",
            issues=issues,
            rationale=(
                "Planner produced zero steps and no recovery route is "
                "in scope; abstain rather than guess."
            ),
        )
    return PlannerCritique(
        verdict="revise",
        issues=issues,
        rationale=(
            f"{len(issues)} fixable issue(s) detected; revising before "
            "tool execution to avoid wasted work."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Revisor
# ─────────────────────────────────────────────────────────────────────


def revise_plan(plan: ToolUsePlan, critique: PlannerCritique) -> ToolUsePlan:
    """Apply the critique's suggested actions to the plan and return a
    new ``ToolUsePlan``. Pure-deterministic -- same inputs, same output."""
    if critique.verdict == "approved":
        return plan

    steps: list[ToolUsePlanStep] = list(plan.steps)
    drop_ids: set[str] = set()

    for issue in critique.issues:
        if issue.code == "missing_phi_scrub_for_phi_query":
            steps.insert(0, ToolUsePlanStep(
                step_id="revised-phi-001",
                specialist="trustedrisk-discharge",
                bundle="core_discharge",
                tool="detect_phi",
                args={"text": plan.user_query},
                mandatory=True,
                rationale=(
                    "Injected by planner-revision (Phase 14.15 Q2) to "
                    "satisfy the missing PHI-scrub safety floor."
                ),
            ))
        elif issue.code == "duplicate_tool_in_plan":
            drop_ids.update(issue.affected_step_ids)
        elif issue.code == "orphan_bundle_for_specialist":
            drop_ids.update(issue.affected_step_ids)
        elif issue.code == "redundant_optional_step_after_intent":
            drop_ids.update(issue.affected_step_ids)

    if drop_ids:
        steps = [s for s in steps if s.step_id not in drop_ids]

    rationale_suffix = (
        f" Revised by Q2 loop: applied {len(critique.issues)} critique "
        "issue(s)."
    )
    return ToolUsePlan(
        user_query=plan.user_query,
        steps=steps,
        n_steps=len(steps),
        confidence=plan.confidence,
        safety_floor_engaged=(
            plan.safety_floor_engaged or any(s.mandatory for s in steps)
        ),
        llm_router_used=plan.llm_router_used,
        llm_model_id=plan.llm_model_id,
        rationale=plan.rationale + rationale_suffix,
        references=list(plan.references) + [
            "TrustedRisk Phase 14.15 Q2 -- planner-revision feedback loop.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Bounded loop
# ─────────────────────────────────────────────────────────────────────


@dataclass
class PlannerRevisionAttempt:
    iteration: int
    plan: ToolUsePlan
    critique: PlannerCritique


@dataclass
class PlannerRevisionTrace:
    attempts: list[PlannerRevisionAttempt]
    final_plan: ToolUsePlan
    iterations_used: int
    iterations_budget: int
    terminated_reason: str   # "approved" / "force_abstain" / "budget_exhausted"


PlanFn = Callable[[str], Awaitable[ToolUsePlan]]


async def run_with_planner_revision(
    plan_fn: PlanFn,
    user_query: str,
    *,
    max_iterations: int = 2,
) -> PlannerRevisionTrace:
    """Run ``plan_fn`` then iterate critique->revise up to
    ``max_iterations`` times.

    The first iteration always calls ``plan_fn``. Subsequent iterations
    run :func:`revise_plan` against the latest critique. The loop
    terminates on ``approved`` / ``force_abstain`` or when the iteration
    budget is exhausted.
    """
    if max_iterations < 0:
        raise ValueError("max_iterations must be >= 0")

    attempts: list[PlannerRevisionAttempt] = []
    current_plan = await plan_fn(user_query)
    critique = critique_plan(current_plan)
    attempts.append(PlannerRevisionAttempt(
        iteration=0, plan=current_plan, critique=critique,
    ))

    iterations_used = 0
    terminated_reason = critique.verdict
    for i in range(1, max_iterations + 1):
        if critique.verdict in ("approved", "force_abstain"):
            break
        revised = revise_plan(current_plan, critique)
        critique = critique_plan(revised)
        attempts.append(PlannerRevisionAttempt(
            iteration=i, plan=revised, critique=critique,
        ))
        current_plan = revised
        iterations_used = i
        terminated_reason = critique.verdict
        if critique.verdict in ("approved", "force_abstain"):
            break
    else:
        if critique.verdict == "revise":
            terminated_reason = "budget_exhausted"

    if critique.verdict == "approved":
        terminated_reason = "approved"
    elif critique.verdict == "force_abstain":
        terminated_reason = "force_abstain"
    elif critique.verdict == "revise":
        terminated_reason = "budget_exhausted"

    return PlannerRevisionTrace(
        attempts=attempts,
        final_plan=current_plan,
        iterations_used=iterations_used,
        iterations_budget=max_iterations,
        terminated_reason=terminated_reason,
    )
