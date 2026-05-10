"""Unit tests for compute_counterfactual_explanation."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from mcp_server.tools.counterfactual_explanation import (
    _modifiability_rank,
    _tier,
    _tier_label,
    compute_counterfactual_explanation,
)
from shared.schemas import Factor, RiskEstimate


def _run(coro):
    return asyncio.run(coro)


def _risk(*, lace=10, prob=0.183) -> RiskEstimate:
    half = 0.05
    return RiskEstimate(
        model_name="lace-plus-bayesian-v1",
        model_version="test",
        horizon_days=30,
        lace_raw_score=lace,
        probability_mean=prob,
        probability_ci95=(max(0.0, prob - half), min(1.0, prob + half)),
        probability_ci_width=2 * half,
        contributing_factors=[
            Factor(name="LACE_length_of_stay",  raw_value=5, lace_points=4, weight=0.40),
            Factor(name="LACE_acuity",          raw_value=1, lace_points=3, weight=0.30),
            Factor(name="LACE_comorbidity",     raw_value=4, lace_points=2, weight=0.20),
            Factor(name="LACE_ed_visits_6mo",   raw_value=1, lace_points=1, weight=0.10),
        ],
        computed_at=datetime.now(timezone.utc),
    )


# ─────────────────────── helpers ───────────────────────

def test_tier_low():
    assert _tier(0.05) == "low"
    assert _tier(0.10) == "low"


def test_tier_moderate():
    assert _tier(0.15) == "moderate"
    assert _tier(0.20) == "moderate"


def test_tier_high():
    assert _tier(0.25) == "high"
    assert _tier(0.50) == "high"


def test_tier_label():
    assert "low" in _tier_label("low")
    assert "moderate" in _tier_label("moderate")
    assert "high" in _tier_label("high")


def test_modifiability_rank_los_lower_than_acuity():
    assert _modifiability_rank("LACE_length_of_stay") < _modifiability_rank("LACE_acuity")


def test_modifiability_rank_unknown_factor_is_fixed():
    assert _modifiability_rank("LACE_unknown_factor") == 2


# ─────────────────────── End-to-end ───────────────────────

def test_counterfactual_basic_shape():
    rep = _run(compute_counterfactual_explanation(risk=_risk(lace=10, prob=0.183)))
    assert rep.current_lace_total == 10
    assert rep.current_prob_mean == 0.183
    assert len(rep.factors) == 4
    assert rep.most_influential_factor in {
        "LACE_length_of_stay", "LACE_acuity",
        "LACE_comorbidity", "LACE_ed_visits_6mo",
    }
    # At least one of the per-factor sweeps must be non-zero
    assert any(abs(f.delta_prob_if_zero) > 1e-6 or abs(f.delta_prob_if_max) > 1e-6
               for f in rep.factors)


def test_counterfactual_factor_zero_lowers_lace():
    rep = _run(compute_counterfactual_explanation(risk=_risk(lace=10, prob=0.183)))
    los = next(f for f in rep.factors if f.factor_name == "LACE_length_of_stay")
    # los had 4 points -> if zero, LACE total = 10 - 4 = 6
    assert los.if_zero_lace_total == 6


def test_counterfactual_factor_max_raises_lace():
    rep = _run(compute_counterfactual_explanation(risk=_risk(lace=10, prob=0.183)))
    los = next(f for f in rep.factors if f.factor_name == "LACE_length_of_stay")
    # los current 4 pts, max 7 -> if max, LACE = 10 + (7-4) = 13 (capped at 19)
    assert los.if_max_lace_total == 13


def test_counterfactual_modifiability_tags():
    rep = _run(compute_counterfactual_explanation(risk=_risk()))
    by_name = {f.factor_name: f for f in rep.factors}
    # Factor metadata: LOS partial; A, C, E fixed
    assert by_name["LACE_length_of_stay"].modifiability == "partial"
    assert by_name["LACE_acuity"].modifiability == "fixed"
    assert by_name["LACE_comorbidity"].modifiability == "fixed"
    assert by_name["LACE_ed_visits_6mo"].modifiability == "fixed"


def test_counterfactual_high_risk_patient_has_safer_flip_path():
    """LACE 14 -> prob ~0.28. Reducing factors should flip to moderate/low tier."""
    rep = _run(compute_counterfactual_explanation(
        risk=_risk(lace=14, prob=0.282),
    ))
    assert rep.flip_path_safer is not None
    assert rep.flip_path_safer.target_lace_max < 14
    # Some path should be discoverable in the LACE 0-19 space
    assert rep.flip_path_safer.cumulative_delta_points >= 1


def test_counterfactual_low_risk_patient_has_riskier_flip_path():
    """LACE 5 -> prob ~0.10. Increasing factors should flip to moderate/high tier."""
    rep = _run(compute_counterfactual_explanation(
        risk=_risk(lace=5, prob=0.103),
    ))
    assert rep.flip_path_riskier is not None
    assert rep.flip_path_riskier.target_lace_max > 5
    assert rep.flip_path_riskier.achievable is False  # riskier paths are illustrative only


def test_counterfactual_low_risk_no_safer_flip_path():
    """If already in low-risk tier, no safer flip is possible."""
    rep = _run(compute_counterfactual_explanation(
        risk=_risk(lace=2, prob=0.05),
    ))
    # With only 2 pts to give up, we may already be at the lowest tier
    assert rep.flip_path_safer is None or rep.flip_path_safer.cumulative_delta_points <= 2


def test_counterfactual_dict_input():
    """Tool accepts dict form of RiskEstimate."""
    risk_dict = _risk().model_dump()
    rep = _run(compute_counterfactual_explanation(risk=risk_dict))
    assert rep.current_lace_total == 10


def test_counterfactual_rejects_empty_factors():
    risk = _risk()
    risk_no_factors = risk.model_copy(update={"contributing_factors": []})
    with pytest.raises(ValueError, match="contributing_factors"):
        _run(compute_counterfactual_explanation(risk=risk_no_factors))


def test_counterfactual_rationale_mentions_lace():
    rep = _run(compute_counterfactual_explanation(risk=_risk(lace=14, prob=0.282)))
    assert "LACE" in rep.rationale or "lace" in rep.rationale.lower()


def test_counterfactual_most_influential_picked_consistently():
    """The factor with the largest absolute swing (zero or max) wins."""
    rep = _run(compute_counterfactual_explanation(risk=_risk()))
    # Compute swings per factor manually
    swings = {f.factor_name: max(abs(f.delta_prob_if_zero), abs(f.delta_prob_if_max))
              for f in rep.factors}
    expected = max(swings, key=swings.get)
    assert rep.most_influential_factor == expected
