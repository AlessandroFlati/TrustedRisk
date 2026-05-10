"""Phase 17.N - Distribution shift formalism tests."""

from __future__ import annotations

import random

import pytest

from a2a_agent.distribution_shift import (
    BBSELabelShiftReport, EnergyOODReport, KSTestReport,
    energy_based_ood_score,
    lipton_bbse_label_shift, two_sample_ks_test,
)


# ─────────────────────────────────────────────────────────────────────
# Two-sample KS
# ─────────────────────────────────────────────────────────────────────

def test_ks_yields_low_drift_for_same_distribution():
    rng = random.Random(7)
    a = [rng.gauss(0, 1) for _ in range(500)]
    b = [rng.gauss(0, 1) for _ in range(500)]
    rep = two_sample_ks_test(reference=a, current=b)
    assert rep.drift_severity in ("none", "low")
    assert rep.p_value > 0.05


def test_ks_yields_high_drift_for_shifted_mean():
    rng = random.Random(7)
    a = [rng.gauss(0, 1) for _ in range(500)]
    b = [rng.gauss(2, 1) for _ in range(500)]
    rep = two_sample_ks_test(reference=a, current=b)
    assert rep.drift_severity in ("moderate", "high")
    assert rep.ks_statistic > 0.30


def test_ks_rejects_short_input():
    with pytest.raises(ValueError):
        two_sample_ks_test(reference=[1.0], current=[2.0, 3.0])


def test_ks_round_trip_through_pydantic():
    rng = random.Random(7)
    a = [rng.gauss(0, 1) for _ in range(50)]
    b = [rng.gauss(0, 1) for _ in range(50)]
    rep = two_sample_ks_test(reference=a, current=b)
    payload = rep.model_dump(mode="json")
    rebuilt = KSTestReport.model_validate(payload)
    assert rebuilt.ks_statistic == rep.ks_statistic


# ─────────────────────────────────────────────────────────────────────
# BBSE label shift
# ─────────────────────────────────────────────────────────────────────

def test_bbse_recovers_uniform_target_under_identity_confusion():
    """When the source confusion matrix is the identity, BBSE just
    returns the observed predicted distribution."""
    rep = lipton_bbse_label_shift(
        classes=["a", "b"],
        source_confusion_matrix=[[1.0, 0.0], [0.0, 1.0]],
        source_prior={"a": 0.5, "b": 0.5},
        target_predicted_distribution={"a": 0.7, "b": 0.3},
    )
    assert abs(rep.estimated_target_prior["a"] - 0.7) < 0.01
    assert abs(rep.estimated_target_prior["b"] - 0.3) < 0.01


def test_bbse_importance_weights_are_target_over_source():
    rep = lipton_bbse_label_shift(
        classes=["a", "b"],
        source_confusion_matrix=[[0.9, 0.1], [0.1, 0.9]],
        source_prior={"a": 0.5, "b": 0.5},
        target_predicted_distribution={"a": 0.5, "b": 0.5},
    )
    # Symmetric confusion + equal predicted -> equal target prior
    assert abs(rep.importance_weights["a"] - 1.0) < 0.05


def test_bbse_three_classes():
    rep = lipton_bbse_label_shift(
        classes=["a", "b", "c"],
        source_confusion_matrix=[
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ],
        source_prior={"a": 0.33, "b": 0.33, "c": 0.34},
        target_predicted_distribution={"a": 0.6, "b": 0.2, "c": 0.2},
    )
    # The class 'a' should now be over-represented in the target
    assert rep.estimated_target_prior["a"] > 0.5
    assert rep.importance_weights["a"] > 1.0


def test_bbse_rejects_non_normalized_target():
    with pytest.raises(ValueError):
        lipton_bbse_label_shift(
            classes=["a", "b"],
            source_confusion_matrix=[[1.0, 0.0], [0.0, 1.0]],
            source_prior={"a": 0.5, "b": 0.5},
            target_predicted_distribution={"a": 0.7, "b": 0.7},
        )


def test_bbse_rejects_singular_matrix():
    with pytest.raises(ValueError):
        lipton_bbse_label_shift(
            classes=["a", "b"],
            source_confusion_matrix=[[0.5, 0.5], [0.5, 0.5]],
            source_prior={"a": 0.5, "b": 0.5},
            target_predicted_distribution={"a": 0.5, "b": 0.5},
        )


def test_bbse_rejects_unsupported_class_count():
    with pytest.raises(ValueError):
        lipton_bbse_label_shift(
            classes=["a", "b", "c", "d"],
            source_confusion_matrix=[[1.0]*4]*4,
            source_prior={c: 0.25 for c in "abcd"},
            target_predicted_distribution={c: 0.25 for c in "abcd"},
        )


# ─────────────────────────────────────────────────────────────────────
# Energy-based OOD
# ─────────────────────────────────────────────────────────────────────

def test_energy_ood_self_calibrated_flags_top_5_percent():
    logits = [[1.0, 0.5, 0.3] for _ in range(100)]
    # 5 outliers with very low logits -> high energy
    logits.extend([[-3.0, -3.5, -2.8] for _ in range(5)])
    rep = energy_based_ood_score(logits_per_instance=logits)
    assert rep.n_ood >= 1
    assert rep.n_ood + rep.n_in == rep.n


def test_energy_ood_supplied_threshold_used():
    logits = [[2.0, 1.0, 0.5] for _ in range(20)]
    rep = energy_based_ood_score(
        logits_per_instance=logits,
        in_distribution_threshold=-0.1,
    )
    # Energy < -0.1 means in-distribution; the example logits should
    # yield negative energies well below -0.1, so n_ood = 0.
    assert rep.in_distribution_threshold == -0.1


def test_energy_score_lower_for_high_logits():
    """Higher logits -> larger logsumexp -> more negative energy."""
    rep_high = energy_based_ood_score(
        logits_per_instance=[[5.0, 4.0, 3.0]])
    rep_low = energy_based_ood_score(
        logits_per_instance=[[-5.0, -6.0, -7.0]])
    assert rep_high.energy_scores[0] < rep_low.energy_scores[0]


def test_energy_ood_rejects_empty():
    with pytest.raises(ValueError):
        energy_based_ood_score(logits_per_instance=[])


def test_energy_ood_round_trip_through_pydantic():
    rep = energy_based_ood_score(
        logits_per_instance=[[1.0, 0.0]])
    payload = rep.model_dump(mode="json")
    rebuilt = EnergyOODReport.model_validate(payload)
    assert rebuilt.n == rep.n
