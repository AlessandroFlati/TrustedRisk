"""Abstain policy logic -- 3 triggers composed into DecisionCard.abstain.

Per design doc §3.5 Three abstain triggers:
  1. confidence_interval_too_wide: RiskEstimate.probability_ci_width > threshold
  2. evidence_insufficient: ClaimGrounding.overall_verdict == "unsupported"
     (or a critical sub-claim unsupported)
  3. out_of_distribution: distance to cohort exceeds threshold (topological) OR
     lace_score exceeds threshold (lace_percentile fallback)
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schemas import AbstainTrigger, ClaimGrounding, RiskEstimate


# ─────────────────────────────────────────────────────────────────────
# Trigger 1 -- CI width
# ─────────────────────────────────────────────────────────────────────

def check_ci_width(risk: RiskEstimate, threshold: float | None = None) -> AbstainTrigger | None:
    """Return AbstainTrigger if probability_ci_width exceeds threshold, else None."""
    if threshold is None:
        threshold = float(os.environ.get("TRUSTEDRISK_CI_WIDTH_THRESHOLD", "0.30"))
    if risk.probability_ci_width > threshold:
        return AbstainTrigger(
            type="confidence_interval_too_wide",
            detail=(
                f"Readmission probability CI width {risk.probability_ci_width:.3f} "
                f"exceeds threshold {threshold:.3f}."
            ),
            threshold_exceeded={
                "ci_width": risk.probability_ci_width,
                "threshold": threshold,
            },
        )
    return None


# ─────────────────────────────────────────────────────────────────────
# Trigger 2 -- evidence insufficient
# ─────────────────────────────────────────────────────────────────────

def check_evidence(
    grounding: ClaimGrounding,
    critical_atomic_kinds: tuple[str, ...] = ("vital", "medication", "procedure"),
) -> AbstainTrigger | None:
    """Return AbstainTrigger if grounding.overall_verdict is 'unsupported' OR any
    critical sub-claim (per atomic_kind) is 'unsupported'.
    """
    if grounding.overall_verdict == "unsupported":
        return AbstainTrigger(
            type="evidence_insufficient",
            detail="Grounding check failed: overall_verdict = 'unsupported'.",
            threshold_exceeded={},
        )

    for sc in grounding.sub_claims:
        if sc.verdict == "unsupported" and sc.atomic_kind in critical_atomic_kinds:
            return AbstainTrigger(
                type="evidence_insufficient",
                detail=(
                    f"Grounding check failed: critical sub-claim '{sc.text}' "
                    f"(atomic_kind={sc.atomic_kind}) is unsupported. "
                    f"Reason: {sc.reason_if_unsupported or 'not specified'}."
                ),
                threshold_exceeded={},
            )

    return None


# ─────────────────────────────────────────────────────────────────────
# Trigger 3 -- out of distribution
# ─────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_ood_policy() -> dict[str, Any] | None:
    """Load the promoted OOD policy at module-import-time, cached."""
    policy_path = os.environ.get("TRUSTEDRISK_OOD_POLICY_PATH")
    if not policy_path:
        return None
    fp = Path(policy_path)
    if not fp.exists():
        return None
    try:
        with fp.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def check_ood(
    lace_score: int,
    encounter_embedding: list[float] | None = None,
) -> AbstainTrigger | None:
    """Return AbstainTrigger if the encounter is out-of-distribution.

    Dispatches on TRUSTEDRISK_OOD_POLICY_TYPE env var:
      - "topological": use cluster-distance threshold + encounter_embedding.
      - "lace_percentile": use lace_score threshold (simpler fallback).
    """
    policy = _load_ood_policy()
    if policy is None:
        return None

    policy_type = os.environ.get("TRUSTEDRISK_OOD_POLICY_TYPE", "").strip().lower()

    if policy_type == "lace_percentile":
        threshold = int(policy.get("threshold_lace", 99))
        if lace_score >= threshold:
            return AbstainTrigger(
                type="out_of_distribution",
                detail=(
                    f"Patient LACE score {lace_score} exceeds fallback OOD "
                    f"threshold {threshold} (top {100 - policy.get('method', {}).get('cohort_percentile_cutoff', 90)}%)."
                ),
                threshold_exceeded={
                    "lace_score": lace_score,
                    "threshold_lace": threshold,
                },
            )
        return None

    if policy_type == "topological":
        if encounter_embedding is None:
            # No embedding available -- skip topological check
            return None
        threshold = float(policy.get("ood_distance_threshold", 1.0))
        distance = _min_distance_to_centroids(encounter_embedding)
        if distance is None:
            return None
        if distance > threshold:
            return AbstainTrigger(
                type="out_of_distribution",
                detail=(
                    f"Encounter distance to nearest cluster centroid "
                    f"{distance:.3f} exceeds OOD threshold {threshold:.3f}."
                ),
                threshold_exceeded={
                    "distance": distance,
                    "threshold": threshold,
                },
            )
        return None

    # Unknown policy type
    return None


def _min_distance_to_centroids(embedding: list[float]) -> float | None:
    """Compute min distance between embedding and cohort cluster centroids.

    Cohort centroids are loaded lazily from cohort_embeddings.npz.
    Returns None if centroids are unavailable.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    centroids = _load_cluster_centroids()
    if centroids is None or len(centroids) == 0:
        return None

    query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    if query.shape[1] != centroids.shape[1]:
        # Dimension mismatch -- likely wrong embedder at runtime
        return None
    # Simple euclidean distance (centroids and query already L2-normalized)
    diffs = centroids - query
    dists = np.linalg.norm(diffs, axis=1)
    return float(dists.min())


@lru_cache(maxsize=1)
def _load_cluster_centroids():
    """Load cluster_centroids from cohort_embeddings.npz, cached."""
    try:
        import numpy as np
    except ImportError:
        return None
    npz_path = os.environ.get("TRUSTEDRISK_COHORT_EMBEDDINGS_PATH")
    if not npz_path:
        return None
    fp = Path(npz_path)
    if not fp.exists():
        return None
    try:
        data = np.load(fp)
        if "cluster_centroids" in data:
            return np.asarray(data["cluster_centroids"], dtype=np.float32)
    except (OSError, ValueError):
        return None
    return None


# ─────────────────────────────────────────────────────────────────────
# Compose all 3 triggers
# ─────────────────────────────────────────────────────────────────────

def compose_abstain_triggers(
    risk: RiskEstimate,
    grounding: ClaimGrounding,
    lace_score: int,
    encounter_embedding: list[float] | None = None,
) -> list[AbstainTrigger]:
    """Run all 3 trigger checks, return the list of fired triggers (possibly empty)."""
    triggers: list[AbstainTrigger] = []
    for check_fn in (
        lambda: check_ci_width(risk),
        lambda: check_evidence(grounding),
        lambda: check_ood(lace_score, encounter_embedding),
    ):
        t = check_fn()
        if t is not None:
            triggers.append(t)
    return triggers
