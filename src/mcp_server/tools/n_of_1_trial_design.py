"""healthcare.compute_n_of_1_trial_design -- Phase 9.5 N-of-1 designer.

Single-patient experimental design: structured randomisation of
A (baseline / placebo) vs B (intervention) blocks across an extended
period. Returns the randomisation sequence, expected carryover, the
minimum detectable effect at 80% power, and an analysis-plan template.

Pure-deterministic -- same inputs + seed -> same sequence. Useful for
patient-driven trials of lifestyle / pain / GI / sleep interventions
where a population RCT is impractical and the patient wants to be
their own control.

References:
- Vohra S et al. CONSORT extension for N-of-1 trials (CENT). BMJ 2015;350:h1738.
- Kravitz RL, Duan N (eds.). Design and Implementation of N-of-1
  Trials: A User's Guide. AHRQ Publication No. 14-EHC122-EF (2014).
"""

from __future__ import annotations

import math
import random
from typing import Any

from shared.schemas import (
    NofOneAnalysisStep,
    NofOneDesignType,
    NofOneTrialDesignReport,
)


_DESIGN_BLOCK_COUNTS: dict[str, int] = {
    "AB":            2,
    "ABA":           3,
    "ABAB":          4,
    "ABABAB":        6,
    "randomized_block": 6,    # default 6 random blocks
}


def _generate_sequence(
    design: str, *, n_blocks: int, seed: int,
) -> list[str]:
    if design == "randomized_block":
        rng = random.Random(seed)
        # Blocked randomisation: pairs of (A, B) shuffled
        if n_blocks % 2 != 0:
            n_blocks += 1   # ensure even
        seq: list[str] = []
        for _ in range(n_blocks // 2):
            pair = ["A", "B"]
            rng.shuffle(pair)
            seq.extend(pair)
        return seq
    # Pre-defined alternating designs
    pattern = list(design)
    return pattern[:n_blocks]


def _carryover_estimate(design: str, intervention_half_life_days: int,
                            block_days: int) -> int:
    """Crude estimate of how many blocks the residual effect spans
    after a B -> A transition. ~ 5 half-lives = practical wash-out."""
    if intervention_half_life_days <= 0 or block_days <= 0:
        return 0
    ko_blocks = math.ceil((5 * intervention_half_life_days) / block_days)
    return min(ko_blocks, _DESIGN_BLOCK_COUNTS.get(design, 6) // 2)


def _minimum_detectable_effect(
    n_blocks: int, n_observations_per_block: int,
) -> float:
    """Compute the minimum detectable effect (Cohen's d) at α=0.05,
    1-β=0.80, paired comparison.

    Using the simplified formula d_min ≈ (z_α + z_β) / sqrt(n_pairs):
    z_α = 1.96 (two-sided 0.05); z_β = 0.84 (power 0.80) -> 2.80.
    """
    n_pairs = max(1, (n_blocks // 2) * n_observations_per_block)
    return round(2.80 / math.sqrt(n_pairs), 4)


_DEFAULT_ANALYSIS_PLAN: list[NofOneAnalysisStep] = [
    NofOneAnalysisStep(
        step_id="exclude_carryover",
        description=(
            "Exclude observations within the carryover window after a "
            "B -> A transition."
        ),
        statistical_method="visual + protocol-defined exclusion window",
    ),
    NofOneAnalysisStep(
        step_id="paired_t_test",
        description="Compare per-block means using a paired t-test.",
        statistical_method="scipy.stats.ttest_rel (paired t)",
    ),
    NofOneAnalysisStep(
        step_id="bayesian_credible_interval",
        description=(
            "Bayesian posterior on the standardised mean difference "
            "with weakly-informative N(0,1) prior."
        ),
        statistical_method="conjugate posterior or PyMC",
    ),
    NofOneAnalysisStep(
        step_id="time_series_diagnostic",
        description=(
            "Plot the per-day outcome with the block boundaries; "
            "inspect for trend / autocorrelation that violates the "
            "iid assumption."
        ),
        statistical_method="visual inspection + Durbin-Watson",
    ),
    NofOneAnalysisStep(
        step_id="patient_subjective_summary",
        description=(
            "Patient writes a one-paragraph summary of perceived "
            "difference between A and B blocks; complement to the "
            "quantitative analysis."
        ),
        statistical_method="qualitative",
    ),
]


async def compute_n_of_1_trial_design(
    intervention_label: str,
    outcome_label: str,
    *,
    design_type: NofOneDesignType = "ABAB",
    block_duration_days: int = 14,
    n_observations_per_block: int = 14,
    intervention_half_life_days: int = 1,
    n_blocks: int | None = None,
    seed: int = 42,
    patient_reference: str | None = None,
) -> NofOneTrialDesignReport:
    """Design an N-of-1 trial with structured randomisation +
    analysis plan.

    Args:
        intervention_label: free-text intervention name (e.g. "PT for
            chronic low back pain").
        outcome_label: outcome (e.g. "self-reported pain 0-10").
        design_type: one of {AB, ABA, ABAB, ABABAB, randomized_block}.
            ABAB is the recommended minimum for inference.
        block_duration_days: days per block.
        n_observations_per_block: observations per block (drives MDE).
        intervention_half_life_days: pharmacokinetic / behavioural
            half-life for the intervention; drives the carryover
            window.
        n_blocks: override the default block count for the design.
        seed: RNG seed for `randomized_block`.
        patient_reference: optional FHIR Patient reference.

    Returns:
        NofOneTrialDesignReport.
    """
    if block_duration_days < 1 or n_observations_per_block < 1:
        raise ValueError(
            "block_duration_days and n_observations_per_block must be ≥ 1"
        )
    nb = n_blocks if n_blocks is not None else _DESIGN_BLOCK_COUNTS.get(
        design_type, 4)
    if nb < 2:
        raise ValueError("n_blocks must be ≥ 2")

    seq = _generate_sequence(design_type, n_blocks=nb, seed=seed)
    carryover = _carryover_estimate(
        design_type, intervention_half_life_days, block_duration_days,
    )
    mde = _minimum_detectable_effect(nb, n_observations_per_block)
    total_days = nb * block_duration_days

    rationale = (
        f"{design_type} design: {nb} blocks × {block_duration_days} days "
        f"= {total_days} days total. Sequence "
        f"{''.join(seq)}. Carryover ≈ {carryover} block(s). "
        f"Minimum detectable effect d ≈ {mde:.3f} at α=0.05, 1-β=0.80."
    )

    return NofOneTrialDesignReport(
        patient_reference=patient_reference,
        intervention_label=intervention_label,
        outcome_label=outcome_label,
        design_type=design_type,
        block_duration_days=block_duration_days,
        n_blocks=nb,
        total_duration_days=total_days,
        randomization_sequence=seq,
        expected_carryover_periods=carryover,
        minimum_detectable_effect=mde,
        analysis_plan=list(_DEFAULT_ANALYSIS_PLAN),
        rationale=rationale,
        references=[
            "Vohra S et al. CONSORT extension for N-of-1 trials (CENT). "
            "BMJ 2015;350:h1738.",
            "Kravitz RL, Duan N. Design and Implementation of N-of-1 "
            "Trials: A User's Guide. AHRQ Publication No. 14-EHC122-EF.",
            "Cohen J. Statistical Power Analysis for the Behavioral "
            "Sciences. 2nd ed. 1988.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_n_of_1_trial_design)
