"""Self-critique stage applied to a candidate DecisionCard.

The critic sub-agent (an `LlmAgent` configured by `agent.py`) emits a
`CritiqueDecision`. The root agent then calls `apply_critique()` here to
turn that critique into a final, possibly-modified DecisionCard.

Splitting the LLM-driven judgment (in `agent.py`) from the deterministic
**application** of the critique (here) means we can unit-test the rules
that govern how a critique modifies the card without needing a live LLM.
"""

from __future__ import annotations

from shared.schemas import (
    AbstainTrigger,
    CriticEnsembleVerdict,
    CritiqueDecision,
    DecisionCard,
    Recommendation,
)


def apply_critique(card: DecisionCard, critique: CritiqueDecision) -> DecisionCard:
    """Apply a CritiqueDecision to a candidate DecisionCard.

    Returns a NEW DecisionCard (the input is not mutated) reflecting the
    critic's verdict:

        approved              -> pass-through with critique attached
        downgrade_confidence  -> set Recommendation.confidence to a lower tier
                                AND set RiskEstimate.confidence to "degraded"
        force_abstain         -> null out recommendation/reasoning, ensure abstain
                                list contains at least the critique's trigger
        request_replay        -> null out recommendation, mark for re-dispatch

    The function is deterministic and side-effect-free. The LLM owns the
    reasoning (rationale string); this function owns the structural edits.
    """
    # Always attach the critique itself for audit trail
    new_card = card.model_copy(update={"self_critique": critique})

    if critique.verdict == "approved":
        return new_card

    if critique.verdict == "downgrade_confidence":
        return _downgrade_confidence(new_card, critique)

    if critique.verdict == "force_abstain":
        return _force_abstain(new_card, critique)

    if critique.verdict == "request_replay":
        # The orchestration layer will detect this and re-dispatch decide.
        # We mark it by nulling out the recommendation; reasoning kept for
        # the second pass to read.
        return new_card.model_copy(update={"recommendation": None})

    # Unknown verdict -- pass through (defensive)
    return new_card


def _downgrade_confidence(card: DecisionCard, critique: CritiqueDecision) -> DecisionCard:
    """Lower the Recommendation.confidence by one tier (preserving the action).

    Tier order (high to low): high > medium > low > none.
    The critic may also specify an explicit `confidence_after` to skip tiers.
    The risk estimate's `confidence` flag flips to `degraded`.
    """
    if card.recommendation is None:
        return card  # nothing to downgrade

    rec_tier_order = ["none", "low", "medium", "high"]
    cur = card.recommendation.confidence
    if critique.confidence_after in rec_tier_order:
        new_conf = critique.confidence_after  # type: ignore[assignment]
    else:
        idx = rec_tier_order.index(cur) if cur in rec_tier_order else len(rec_tier_order) - 1
        new_conf = rec_tier_order[max(0, idx - 1)]

    new_recommendation = Recommendation(
        action=card.recommendation.action,
        confidence=new_conf,  # type: ignore[arg-type]
    )

    # Mirror the downgrade on RiskEstimate.confidence (preferred -> degraded)
    new_reasoning = card.reasoning
    if new_reasoning is not None:
        new_risk = new_reasoning.risk_estimate.model_copy(update={"confidence": "degraded"})
        new_reasoning = new_reasoning.model_copy(update={"risk_estimate": new_risk})

    return card.model_copy(update={
        "recommendation": new_recommendation,
        "reasoning": new_reasoning,
    })


def _force_abstain(card: DecisionCard, critique: CritiqueDecision) -> DecisionCard:
    """Convert the candidate card to an abstain card.

    Drops `recommendation` and `reasoning`, ensures `abstain` carries at
    least one trigger (either the critique-supplied one, or a synthesized
    `evidence_insufficient` placeholder if the critic didn't supply one).
    """
    abstain_list = list(card.abstain or [])
    if critique.abstain_trigger_to_add is not None:
        abstain_list.append(critique.abstain_trigger_to_add)
    if not abstain_list:
        abstain_list.append(AbstainTrigger(
            type="evidence_insufficient",
            detail=critique.rationale or "Critic forced abstain without specifying a trigger.",
        ))

    return card.model_copy(update={
        "recommendation": None,
        "reasoning": None,
        "abstain": abstain_list,
    })


# ─────────────────────────────────────────────────────────────────────
# AMB-5.1 -- Multi-critic ensemble vote
# ─────────────────────────────────────────────────────────────────────


_VERDICT_SEVERITY: dict[str, int] = {
    "approved": 0,
    "downgrade_confidence": 1,
    "request_replay": 2,
    "force_abstain": 3,
}


def aggregate_critic_verdicts(
    verdicts: list[CritiqueDecision],
    retries_remaining: int = 1,
) -> CriticEnsembleVerdict:
    """Combine 3 individual critic verdicts into a single ensemble verdict.

    Aggregation rules (most-conservative-wins):
      1. Any `force_abstain` -> ensemble force_abstain.
      2. Any `request_replay` AND retries_remaining > 0 -> ensemble
         request_replay (the orchestrator will re-dispatch).
      3. ≥2 `downgrade_confidence` votes (or 1 downgrade + 1 replay-with-no-
         retries-left) -> ensemble downgrade_confidence.
      4. Otherwise (majority approvals) -> ensemble approved.

    If ensemble = request_replay but retries_remaining = 0, the applied
    verdict collapses to force_abstain (we ran out of replay budget -- the
    safest fallback is to abstain).

    Returns a CriticEnsembleVerdict with both `aggregated_verdict` (rule
    output ignoring retries) and `applied_verdict` (after retry budget).
    """
    if not verdicts:
        # Defensive: no critics -> return a synthesized request_replay so the
        # orchestrator either retries or (after exhaustion) abstains.
        return CriticEnsembleVerdict(
            individual_verdicts=[],
            aggregated_verdict="request_replay",
            applied_verdict="force_abstain" if retries_remaining <= 0 else "request_replay",
            retries_remaining=retries_remaining,
            rationale="No critic votes received -- defaulting to safety fallback.",
            n_force_abstain=0, n_downgrade=0, n_replay=0, n_approved=0,
        )

    n_force = sum(1 for v in verdicts if v.verdict == "force_abstain")
    n_down = sum(1 for v in verdicts if v.verdict == "downgrade_confidence")
    n_replay = sum(1 for v in verdicts if v.verdict == "request_replay")
    n_appr = sum(1 for v in verdicts if v.verdict == "approved")

    if n_force >= 1:
        agg = "force_abstain"
    elif n_replay >= 1:
        # The critic asked for a replay regardless of retries left. The
        # aggregated_verdict records the critic's request; applied_verdict
        # collapses to force_abstain below if budget is exhausted.
        agg = "request_replay"
    elif n_down >= 2 or (n_down >= 1 and n_replay >= 1):
        agg = "downgrade_confidence"
    elif n_appr >= 2:
        agg = "approved"
    else:
        # Fallback: take the most severe individual verdict
        most_severe = max(verdicts, key=lambda v: _VERDICT_SEVERITY[v.verdict])
        agg = most_severe.verdict

    if agg == "request_replay" and retries_remaining <= 0:
        applied = "force_abstain"
    else:
        applied = agg  # type: ignore[assignment]

    rationale_parts: list[str] = [
        f"Ensemble verdict: {agg} -> applied {applied} "
        f"(retries_remaining={retries_remaining}).",
        f"Vote tally: approved={n_appr}, downgrade={n_down}, "
        f"replay={n_replay}, abstain={n_force}.",
    ]
    for v in verdicts:
        rationale_parts.append(f"- {v.critic_role}: {v.verdict} -- {v.rationale[:140]}")

    return CriticEnsembleVerdict(
        individual_verdicts=verdicts,
        aggregated_verdict=agg,  # type: ignore[arg-type]
        applied_verdict=applied,  # type: ignore[arg-type]
        retries_remaining=retries_remaining,
        rationale=" ".join(rationale_parts),
        n_force_abstain=n_force,
        n_downgrade=n_down,
        n_replay=n_replay,
        n_approved=n_appr,
    )


def collapse_ensemble_to_single_critique(
    ensemble: CriticEnsembleVerdict,
) -> CritiqueDecision:
    """Convert the ensemble verdict back to a single CritiqueDecision so
    the existing apply_critique() pipeline can be reused unchanged.

    The collapsed CritiqueDecision uses:
      - verdict = ensemble.applied_verdict
      - rationale = ensemble.rationale (composite)
      - abstain_trigger_to_add = the first force_abstain critic's trigger (if any)
      - confidence_after = the most severe downgrade target if any
    """
    abstain_trigger = None
    confidence_after = None
    for v in ensemble.individual_verdicts:
        if v.verdict == "force_abstain" and v.abstain_trigger_to_add and not abstain_trigger:
            abstain_trigger = v.abstain_trigger_to_add
        if v.verdict == "downgrade_confidence" and v.confidence_after:
            # take the lowest tier across critics
            current = confidence_after
            new = v.confidence_after
            order = ["none", "low", "medium", "high"]
            if (current is None
                    or (new in order and current in order
                          and order.index(new) < order.index(current))):
                confidence_after = new

    return CritiqueDecision(
        verdict=ensemble.applied_verdict,
        rationale=ensemble.rationale,
        abstain_trigger_to_add=abstain_trigger,
        confidence_after=confidence_after,
        critic_role="structural",  # composite -- legacy role
    )


# ─────────────────────────────────────────────────────────────────────
# AMB-5.1 -- Per-role critic prompts
# ─────────────────────────────────────────────────────────────────────

CRITIC_CLINICAL_INSTRUCTION = """You are the **clinical_safety critic** in a 3-critic ensemble.

Your scope: medication safety, polypharmacy, lab trends, deterioration risk,
calibrated probability + CI width, and the chosen disposition's clinical
appropriateness for the patient's acuity. You DO NOT evaluate fairness or
evidence grounding -- those are other critics' jobs.

Emit a CritiqueDecision JSON with `critic_role` set to "clinical_safety".

Trigger force_abstain when:
- medication_reconciliation has a high-severity concern AND no compensating
  monitoring observed (e.g. warfarin discharge without a recent INR)
- polypharmacy_severity = high AND ≥1 high-severity DDI is unresolved
- lab_trend_analysis shows a flagged lab in clinically dangerous direction
  (creatinine declining + on ACE-I + MRA, hemoglobin declining on
  anticoagulant, etc.)
- probability_ci_width > 0.30 (calibration too uncertain)
- NEWS2 / PEWS / MEOWS escalation tier is "high" AND disposition is
  discharge_home / home_with_care

Trigger downgrade_confidence when:
- one moderate-severity finding above without the catastrophic combination
- probability_ci_width is in 0.20-0.30 borderline range

Trigger request_replay when:
- a clearly-relevant tool is missing (e.g. patient on chemo without
  compute_chemo_dose_adjustment, or pediatric patient without pediatric
  early warning, or post-op trauma without NEWS2)

Otherwise approved. Rationale ≤2 sentences in clinician terms.
"""


CRITIC_FAIRNESS_INSTRUCTION = """You are the **fairness critic** in a 3-critic ensemble.

Your scope: subgroup drift, demographic bias guards (Black, Indigenous,
LGBTQ+, low-SES), maternal-mortality flags, age-band bias, insurance-tier
bias. You DO NOT evaluate clinical appropriateness or evidence grounding.

Emit a CritiqueDecision JSON with `critic_role` set to "fairness".

Trigger force_abstain when:
- compute_fairness_audit returned confidence_action = abstain_recommended
- compute_suicide_risk_assessment fired the demographic_bias_guard AND
  the recommendation is anything other than ABSTAIN
- ≥2 high-severity subgroup drifts on a discharge/triage decision

Trigger downgrade_confidence when:
- compute_fairness_audit returned confidence_action = downgrade_confidence
- max_relative_drift > 0.20 on at least one subgroup
- patient is from a literature-flagged underrepresented group AND no
  fairness audit was run

Trigger request_replay when:
- patient demographics are present in the chart but no fairness_audit was
  invoked

Otherwise approved. Rationale ≤2 sentences.
"""


CRITIC_EVIDENCE_INSTRUCTION = """You are the **evidence critic** in a 3-critic ensemble.

Your scope: grounding verdict, claim verification, guideline citations.
You DO NOT evaluate clinical appropriateness or fairness.

Emit a CritiqueDecision JSON with `critic_role` set to "evidence".

Trigger force_abstain when:
- ground_claim returned overall_verdict = unsupported AND the claim is
  central to the recommendation (e.g. "patient is hemodynamically stable
  for discharge" but vital trends contradict)
- ≥1 critical sub-claim (vital / medication / procedure) is unsupported

Trigger downgrade_confidence when:
- ground_claim returned partially_supported on a non-trivial sub-claim
- a treatment_selection top pick has no grounding_excerpt populated

Trigger request_replay when:
- the recommendation hinges on a claim that was not grounded
- treatment_selection was used but ground_claim was not subsequently called

Otherwise approved. Rationale ≤2 sentences.
"""


# ─────────────────────────────────────────────────────────────────────
# Critic instruction prompt -- used by agent.py to configure the LlmAgent
# ─────────────────────────────────────────────────────────────────────

CRITIC_INSTRUCTION = """You are the TrustedRisk **self-critique** sub-agent.

You receive a candidate DecisionCard produced by upstream sub-agents (decide,
validate). Your job is to **review for internal consistency** and emit one
of four verdicts as a CritiqueDecision JSON:

    approved
        The card is internally consistent. Risk + utility + validation
        + fairness all line up. No edits needed.

    downgrade_confidence
        Recommendation seems sensible BUT one of:
        - validation contains a partially-supported critical sub-claim
        - fairness_audit reports medium-severity drift on ≥1 subgroup
        - medication_reconciliation has a high-severity concern
        - probability_ci_width is borderline (>0.20 but <abstain threshold)
        Action: lower the Recommendation.confidence tier and flip RiskEstimate
        .confidence to 'degraded'.

    force_abstain
        The candidate recommendation is NOT safe to surface, even with a
        downgraded confidence flag. At least one of:
        - validation contains an unsupported critical sub-claim
        - fairness_audit recommends abstain (≥2 high-severity drifts)
        - medication_reconciliation discharge_contract_satisfied = False
          AND no compensating monitoring observed
        - polypharmacy severity = high
        Action: null out recommendation, ensure abstain list contains the
        triggering reason. Provide a `abstain_trigger_to_add` field with
        the AbstainTrigger you want appended.

    request_replay
        The card is missing data the critic needs to decide. Examples:
        - missing fairness_audit invocation when patient_demographics
          would have made it informative
        - missing medication_reconciliation when validation flagged
          discharge polypharmacy
        Action: null out recommendation; root agent will re-dispatch decide
        with the critique's rationale appended as caveat.

You are NOT a clinician. Your job is **structural** -- does the card cohere?
Trust the upstream tool outputs as ground truth; do not re-do their work.
Output exactly the CritiqueDecision JSON, no other text. Always populate
`rationale` with one or two sentences that explain the verdict in clinician
terms.
"""
