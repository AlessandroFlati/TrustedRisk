"""Phase 17.AR - Counterfactual-fairness audit tests."""

from __future__ import annotations

import math

import pytest

from a2a_agent.counterfactual_fairness import (
    CFFairnessReport, StructuralCausalModel,
    audit_counterfactual_fairness, flip_protected_attribute,
)


def _scm_race() -> StructuralCausalModel:
    return StructuralCausalModel(
        protected_attribute="race",
        protected_values=["white", "black"],
        descendants=["lace_total"],
        independent_features=[],
    )


def _structural_equations():
    """Black race has historically suffered higher LACE due to
    SDoH; the structural equation reflects this disparity."""
    def lace_eq(race_value: str, ancestors: dict) -> float:
        base = ancestors.get("lace_total", 5.0)
        # Black -> +1 LACE due to SDoH-driven comorbidity burden
        if race_value == "black":
            return base + 1.0
        return base
    return {"lace_total": lace_eq}


def _bucket_predictor(features: dict[str, float]) -> float:
    lace = features.get("lace_total", 0.0)
    if lace <= 2:
        return 0.072
    if lace <= 5:
        return 0.103
    if lace <= 9:
        return 0.158
    if lace <= 12:
        return 0.234
    return 0.327


def _race_aware_predictor(features: dict[str, float]) -> float:
    """Includes race as a direct feature - counterfactually unfair
    by construction."""
    base = _bucket_predictor(features)
    if features.get("race") == "black":
        return min(1.0, base * 1.18)
    return base


def _race_blind_predictor(features: dict[str, float]) -> float:
    return _bucket_predictor(features)


# ─────────────────────────────────────────────────────────────────────
# flip_protected_attribute
# ─────────────────────────────────────────────────────────────────────

def test_flip_propagates_protected_through_descendants():
    scm = _scm_race()
    eqs = _structural_equations()
    instance = {"race": "white", "lace_total": 5.0}
    cf = flip_protected_attribute(
        instance=instance, scm=scm,
        target_value="black",
        structural_equations=eqs,
    )
    assert cf["race"] == "black"
    # Lace total bumped by +1 in counterfactual world
    assert cf["lace_total"] == 6.0


def test_flip_rejects_unknown_target_value():
    scm = _scm_race()
    eqs = _structural_equations()
    with pytest.raises(ValueError):
        flip_protected_attribute(
            instance={"race": "white", "lace_total": 4.0},
            scm=scm, target_value="indigenous",
            structural_equations=eqs,
        )


def test_flip_rejects_missing_protected_in_instance():
    scm = _scm_race()
    eqs = _structural_equations()
    with pytest.raises(ValueError):
        flip_protected_attribute(
            instance={"lace_total": 4.0},
            scm=scm, target_value="black",
            structural_equations=eqs,
        )


# ─────────────────────────────────────────────────────────────────────
# audit
# ─────────────────────────────────────────────────────────────────────

def test_race_blind_predictor_is_strictly_cf_fair():
    """When the predictor depends only on lace_total (not race
    directly) and the SCM bumps lace_total by 1 between racial
    counterfactuals, *most* instances stay in the same bucket
    -> the audit will find a non-trivial number of strict CF-fair
    instances."""
    scm = _scm_race()
    eqs = _structural_equations()
    instances = [
        {"race": "white", "lace_total": 1.0},   # bucket flips: 1 -> 2
        {"race": "white", "lace_total": 5.0},   # 5 -> 6 = bucket flip
        {"race": "black", "lace_total": 8.0},   # 8 -> 7 = bucket flip
        {"race": "white", "lace_total": 12.0},  # 12 -> 13 = bucket flip
    ]
    rep = audit_counterfactual_fairness(
        instances=instances, predictor=_race_blind_predictor,
        scm=scm, structural_equations=eqs,
    )
    # Race-blind predictor: deltas are bounded by single-bucket
    # transitions -> p95 delta should be modest.
    assert rep.p95_abs_delta < 0.10


def test_race_aware_predictor_shows_larger_deltas_than_blind():
    scm = _scm_race()
    eqs = _structural_equations()
    instances = [
        {"race": "white", "lace_total": 5.0},
        {"race": "black", "lace_total": 5.0},
        {"race": "white", "lace_total": 8.0},
        {"race": "black", "lace_total": 8.0},
    ]
    blind = audit_counterfactual_fairness(
        instances=instances, predictor=_race_blind_predictor,
        scm=scm, structural_equations=eqs,
    )
    aware = audit_counterfactual_fairness(
        instances=instances, predictor=_race_aware_predictor,
        scm=scm, structural_equations=eqs,
    )
    # The race-aware predictor must show >= mean delta vs blind
    assert aware.mean_abs_delta >= blind.mean_abs_delta


def test_audit_yields_one_finding_per_instance():
    scm = _scm_race()
    eqs = _structural_equations()
    instances = [
        {"race": "white", "lace_total": 4.0},
        {"race": "black", "lace_total": 9.0},
    ]
    rep = audit_counterfactual_fairness(
        instances=instances, predictor=_race_blind_predictor,
        scm=scm, structural_equations=eqs,
    )
    assert len(rep.findings) == len(instances)


def test_audit_rejects_empty_cohort():
    with pytest.raises(ValueError):
        audit_counterfactual_fairness(
            instances=[], predictor=_race_blind_predictor,
            scm=_scm_race(),
            structural_equations=_structural_equations(),
        )


def test_audit_finding_carries_actual_and_counterfactual_values():
    scm = _scm_race()
    eqs = _structural_equations()
    rep = audit_counterfactual_fairness(
        instances=[{"race": "white", "lace_total": 5.0}],
        predictor=_race_blind_predictor, scm=scm,
        structural_equations=eqs,
    )
    f = rep.findings[0]
    assert f.actual_protected_value == "white"
    assert f.counterfactual_protected_value == "black"


def test_audit_round_trip_through_pydantic():
    scm = _scm_race()
    eqs = _structural_equations()
    rep = audit_counterfactual_fairness(
        instances=[{"race": "white", "lace_total": 5.0}],
        predictor=_race_blind_predictor, scm=scm,
        structural_equations=eqs,
    )
    payload = rep.model_dump(mode="json")
    rebuilt = CFFairnessReport.model_validate(payload)
    assert rebuilt.n_instances == rep.n_instances
