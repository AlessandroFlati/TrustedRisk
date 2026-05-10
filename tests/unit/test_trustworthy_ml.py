"""Phase 16.F2 - Trustworthy-ML formalisms tests."""

from __future__ import annotations

import pytest

from a2a_agent.trustworthy_ml import (
    per_subgroup_calibration_tension,
    selective_classification_curve,
    split_conformal_multiclass,
)


# ─────────────────────────────────────────────────────────────────────
# Conformal multi-class
# ─────────────────────────────────────────────────────────────────────

def _three_class_probs(top: str, top_p: float = 0.7,
                        rest_p: float = 0.15):
    out = {"discharge_home": rest_p, "discharge_with_homecare": rest_p,
           "continued_admission": rest_p}
    out[top] = top_p
    return out


def test_conformal_quantile_threshold_in_unit_interval():
    cal_probs = [_three_class_probs("discharge_home", 0.85, 0.075)
                 for _ in range(50)]
    cal_y = ["discharge_home"] * 50
    test_probs = [_three_class_probs("discharge_home", 0.6, 0.2)
                  for _ in range(20)]
    res = split_conformal_multiclass(
        cal_probabilities=cal_probs, cal_true_labels=cal_y,
        test_probabilities=test_probs,
    )
    assert 0.0 <= res.quantile_threshold <= 1.0
    assert res.n_calibration == 50
    assert res.n_test == 20


def test_conformal_marginal_coverage_meets_target_on_clean_split():
    """When the calibration and test sets share the same generative
    distribution, empirical coverage must meet the target."""
    import random
    rng = random.Random(11)
    classes = ["discharge_home", "discharge_with_homecare",
               "continued_admission"]
    cal_probs, cal_y, test_probs, test_y = [], [], [], []
    for _ in range(1000):
        true = rng.choice(classes)
        prob_true = 0.55 + 0.4 * rng.random()
        rest = (1.0 - prob_true) / 2
        probs = {c: rest for c in classes}
        probs[true] = prob_true
        cal_probs.append(probs)
        cal_y.append(true)
    for _ in range(500):
        true = rng.choice(classes)
        prob_true = 0.55 + 0.4 * rng.random()
        rest = (1.0 - prob_true) / 2
        probs = {c: rest for c in classes}
        probs[true] = prob_true
        test_probs.append(probs)
        test_y.append(true)
    res = split_conformal_multiclass(
        cal_probabilities=cal_probs, cal_true_labels=cal_y,
        test_probabilities=test_probs,
        test_true_labels=test_y, target_coverage=0.90,
    )
    # Conformal guarantees marginal coverage >= target - 1/(n+1).
    # Coverage may overshoot on synthetic well-calibrated data, so we
    # only enforce the lower bound.
    assert res.empirical_coverage_test >= 0.85
    assert 1.0 <= res.average_set_size <= 3.0


def test_conformal_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        split_conformal_multiclass(
            cal_probabilities=[{"a": 1.0}],
            cal_true_labels=[],
            test_probabilities=[],
        )


def test_conformal_target_coverage_must_be_in_open_unit():
    with pytest.raises(ValueError):
        split_conformal_multiclass(
            cal_probabilities=[{"a": 1.0}],
            cal_true_labels=["a"],
            test_probabilities=[{"a": 1.0}],
            target_coverage=1.5,
        )


def test_conformal_falls_back_to_argmax_for_empty_set():
    """When the quantile is so tight that no class qualifies for an
    instance, we must still return at least the argmax label."""
    cal_probs = [_three_class_probs("discharge_home", 0.99, 0.005)
                 for _ in range(20)]
    cal_y = ["discharge_home"] * 20
    test_probs = [_three_class_probs("continued_admission", 0.9, 0.05)]
    res = split_conformal_multiclass(
        cal_probabilities=cal_probs, cal_true_labels=cal_y,
        test_probabilities=test_probs, target_coverage=0.5,
    )
    assert len(res.prediction_sets[0]) >= 1


# ─────────────────────────────────────────────────────────────────────
# Per-subgroup calibration tension
# ─────────────────────────────────────────────────────────────────────

def test_calibration_tension_returns_per_subgroup_metrics():
    preds = [0.1, 0.2, 0.7, 0.8, 0.4, 0.6]
    labels = [0, 0, 1, 1, 1, 0]
    sg = ["a", "a", "a", "b", "b", "b"]
    res = per_subgroup_calibration_tension(
        predictions=preds, labels=labels, subgroup_labels=sg,
    )
    assert set(res.subgroup_calibration_error) == {"a", "b"}
    assert set(res.subgroup_generalised_fpr) == {"a", "b"}


def test_calibration_tension_witnesses_impossibility_with_disparate_base_rates():
    """Two subgroups with very different base rates should witness the
    Pleiss 2017 impossibility."""
    import random
    rng = random.Random(42)
    preds, labels, sg = [], [], []
    # Group A: base rate 10%
    for _ in range(500):
        y = 1 if rng.random() < 0.1 else 0
        p = 0.1 + 0.05 * rng.random()
        if y:
            p = min(1.0, p + 0.1)
        preds.append(p)
        labels.append(y)
        sg.append("a")
    # Group B: base rate 50%
    for _ in range(500):
        y = 1 if rng.random() < 0.5 else 0
        p = 0.5 + 0.05 * rng.random()
        if y:
            p = min(1.0, p + 0.05)
        preds.append(p)
        labels.append(y)
        sg.append("b")
    res = per_subgroup_calibration_tension(
        predictions=preds, labels=labels, subgroup_labels=sg,
    )
    assert res.impossibility_witnessed is True


def test_calibration_tension_does_not_witness_when_only_one_subgroup():
    preds = [0.1, 0.7]
    labels = [0, 1]
    sg = ["a", "a"]
    res = per_subgroup_calibration_tension(
        predictions=preds, labels=labels, subgroup_labels=sg,
    )
    assert res.impossibility_witnessed is False


def test_calibration_tension_rejects_misaligned():
    with pytest.raises(ValueError):
        per_subgroup_calibration_tension(
            predictions=[0.5], labels=[0, 1],
            subgroup_labels=["a"],
        )


# ─────────────────────────────────────────────────────────────────────
# Selective classification
# ─────────────────────────────────────────────────────────────────────

def test_selective_classification_emits_n_thresholds_points():
    preds = [0.1, 0.4, 0.6, 0.9] * 25
    confs = [0.5, 0.6, 0.8, 0.95] * 25
    labels = [0, 0, 1, 1] * 25
    res = selective_classification_curve(
        predictions=preds, confidences=confs, labels=labels,
        n_thresholds=11,
    )
    assert res.n_thresholds == 11
    assert len(res.points) == 11


def test_selective_classification_coverage_is_monotonically_decreasing():
    preds = [0.1, 0.4, 0.6, 0.9] * 25
    confs = list(range(100))
    labels = [0, 0, 1, 1] * 25
    res = selective_classification_curve(
        predictions=preds,
        confidences=[c / 100 for c in confs],
        labels=labels, n_thresholds=10,
    )
    coverages = [pt.coverage for pt in res.points]
    for a, b in zip(coverages, coverages[1:]):
        assert a >= b - 1e-9


def test_selective_classification_optimal_inside_range():
    preds = [0.05, 0.45, 0.55, 0.95] * 25
    confs = [0.4, 0.5, 0.7, 0.9] * 25
    labels = [0, 0, 1, 1] * 25
    res = selective_classification_curve(
        predictions=preds, confidences=confs, labels=labels,
        n_thresholds=15,
    )
    min_c = min(c for c in confs)
    max_c = max(c for c in confs)
    assert min_c <= res.optimal_threshold <= max_c


def test_selective_classification_rejects_empty():
    with pytest.raises(ValueError):
        selective_classification_curve(
            predictions=[], confidences=[], labels=[],
        )


def test_selective_classification_rejects_misaligned():
    with pytest.raises(ValueError):
        selective_classification_curve(
            predictions=[0.5], confidences=[0.5, 0.5], labels=[1],
        )
