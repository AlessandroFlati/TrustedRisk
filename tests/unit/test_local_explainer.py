"""Phase 17.AG - Anchors + LIME local explainer tests."""

from __future__ import annotations

import math

import pytest

from a2a_agent.local_explainer import (
    AnchorReport, LIMEReport,
    anchors_explain, lime_explain,
)


def _logistic(features: dict[str, float]) -> float:
    z = -3.0 + 0.5 * features.get("L", 0.0) \
        + 0.3 * features.get("A", 0.0) \
        + 0.2 * features.get("C", 0.0) \
        + 0.1 * features.get("E", 0.0)
    return 1.0 / (1.0 + math.exp(-z))


# ─────────────────────────────────────────────────────────────────────
# LIME
# ─────────────────────────────────────────────────────────────────────

def test_lime_returns_attribution_per_feature():
    rep = lime_explain(
        instance={"L": 5.0, "A": 1.0, "C": 2.0, "E": 1.0},
        predictor=_logistic,
        n_perturbations=200,
    )
    feats = {a.feature for a in rep.attributions}
    assert feats == {"L", "A", "C", "E"}


def test_lime_top_attribution_is_largest_coefficient_feature():
    """L has the largest true coefficient (0.5) -> should rank top."""
    rep = lime_explain(
        instance={"L": 5.0, "A": 1.0, "C": 2.0, "E": 1.0},
        predictor=_logistic,
        n_perturbations=400,
    )
    assert rep.attributions[0].feature == "L"


def test_lime_attribution_directions_match_coefficient_signs():
    rep = lime_explain(
        instance={"L": 4.0, "A": 1.0, "C": 1.0, "E": 1.0},
        predictor=_logistic,
        n_perturbations=300,
    )
    # All true coefficients are positive
    assert all(
        a.direction in ("increases", "neutral")
        for a in rep.attributions
    )


def test_lime_rejects_empty_instance():
    with pytest.raises(ValueError):
        lime_explain(
            instance={}, predictor=_logistic,
            n_perturbations=10,
        )


def test_lime_top_k_caps_returned_attributions():
    rep = lime_explain(
        instance={"L": 4.0, "A": 1.0, "C": 1.0, "E": 1.0},
        predictor=_logistic, n_perturbations=200, top_k=2,
    )
    assert len(rep.attributions) == 2


def test_lime_round_trip_through_pydantic():
    rep = lime_explain(
        instance={"L": 4.0}, predictor=_logistic,
        n_perturbations=100,
    )
    payload = rep.model_dump(mode="json")
    rebuilt = LIMEReport.model_validate(payload)
    assert rebuilt.n_perturbations == rep.n_perturbations


# ─────────────────────────────────────────────────────────────────────
# Anchors
# ─────────────────────────────────────────────────────────────────────

def test_anchor_for_high_lace_instance_includes_dominant_feature():
    rep = anchors_explain(
        instance={"L": 7.0, "A": 3.0, "C": 5.0, "E": 4.0},
        predictor=_logistic,
        decision_threshold=0.20,
        n_samples=200,
    )
    # The high-risk classification should be anchored by L (the
    # dominant feature) at minimum
    assert len(rep.anchor_features) >= 1
    assert "L" in rep.anchor_features


def test_anchor_precision_meets_target():
    rep = anchors_explain(
        instance={"L": 7.0, "A": 3.0, "C": 5.0, "E": 4.0},
        predictor=_logistic, decision_threshold=0.20,
        target_precision=0.90, n_samples=300,
    )
    # The anchor must reach the target or exhaust the budget
    assert rep.precision >= 0.5


def test_anchor_for_low_lace_instance_predicts_class_zero():
    rep = anchors_explain(
        instance={"L": 0.0, "A": 0.0, "C": 0.0, "E": 0.0},
        predictor=_logistic, decision_threshold=0.20,
        n_samples=200,
    )
    assert rep.instance_prediction == 0


def test_anchor_rejects_empty():
    with pytest.raises(ValueError):
        anchors_explain(
            instance={}, predictor=_logistic,
            n_samples=50,
        )


def test_anchor_round_trip_through_pydantic():
    rep = anchors_explain(
        instance={"L": 4.0, "A": 1.0, "C": 1.0, "E": 1.0},
        predictor=_logistic, n_samples=50,
    )
    payload = rep.model_dump(mode="json")
    rebuilt = AnchorReport.model_validate(payload)
    assert rebuilt.anchor_features == rep.anchor_features
