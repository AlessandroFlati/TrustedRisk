"""Phase 17.AL - Bayesian network inference tests."""

from __future__ import annotations

import pytest

from a2a_agent.bayes_net import (
    BNNode, BayesianNetwork, build_network,
    build_readmission_network, infer,
)


def _two_node_network() -> BayesianNetwork:
    rain = BNNode(
        name="Rain", states=["yes", "no"],
        cpt={"": {"yes": 0.3, "no": 0.7}},
    )
    grass = BNNode(
        name="GrassWet", states=["yes", "no"],
        parents=["Rain"],
        cpt={
            "yes": {"yes": 0.95, "no": 0.05},
            "no":  {"yes": 0.30, "no": 0.70},
        },
    )
    return build_network([rain, grass])


# ─────────────────────────────────────────────────────────────────────
# Network construction
# ─────────────────────────────────────────────────────────────────────

def test_build_rejects_unknown_parent():
    bad = BNNode(name="Y", states=["a"], parents=["X"],
                 cpt={"a": {"a": 1.0}})
    with pytest.raises(ValueError):
        build_network([bad])


def test_build_rejects_non_normalized_cpt_row():
    bad = BNNode(name="X", states=["a", "b"],
                 cpt={"": {"a": 0.5, "b": 0.6}})
    with pytest.raises(ValueError):
        build_network([bad])


def test_build_rejects_duplicate_node_names():
    a = BNNode(name="X", states=["a"],
               cpt={"": {"a": 1.0}})
    b = BNNode(name="X", states=["a"],
               cpt={"": {"a": 1.0}})
    with pytest.raises(ValueError):
        build_network([a, b])


def test_build_rejects_cycle():
    a = BNNode(name="A", states=["x"], parents=["B"],
               cpt={"x": {"x": 1.0}})
    b = BNNode(name="B", states=["x"], parents=["A"],
               cpt={"x": {"x": 1.0}})
    with pytest.raises(ValueError):
        build_network([a, b])


# ─────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────

def test_marginal_query_matches_root_prior():
    net = _two_node_network()
    res = infer(network=net, query="Rain")
    assert abs(res.distribution["yes"] - 0.3) < 1e-6
    assert abs(res.distribution["no"] - 0.7) < 1e-6


def test_query_with_evidence_propagates_correctly():
    """If we observe wet grass, P(Rain | GrassWet=yes) should rise."""
    net = _two_node_network()
    no_ev = infer(network=net, query="Rain")
    with_ev = infer(
        network=net, query="Rain",
        evidence={"GrassWet": "yes"},
    )
    assert with_ev.distribution["yes"] > no_ev.distribution["yes"]


def test_inference_rejects_unknown_query():
    net = _two_node_network()
    with pytest.raises(ValueError):
        infer(network=net, query="NonExistent")


def test_inference_rejects_unknown_evidence_node():
    net = _two_node_network()
    with pytest.raises(ValueError):
        infer(network=net, query="Rain",
              evidence={"DoesNotExist": "yes"})


def test_inference_rejects_unknown_evidence_state():
    net = _two_node_network()
    with pytest.raises(ValueError):
        infer(network=net, query="Rain",
              evidence={"GrassWet": "maybe"})


def test_distribution_normalises_to_one():
    net = _two_node_network()
    res = infer(network=net, query="GrassWet")
    assert abs(sum(res.distribution.values()) - 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────
# Clinical readmission network
# ─────────────────────────────────────────────────────────────────────

def test_clinical_network_severe_lace_increases_readmit_risk():
    net = build_readmission_network()
    low = infer(network=net, query="Outcome",
                evidence={"LACE_band": "low"})
    high = infer(network=net, query="Outcome",
                  evidence={"LACE_band": "severe"})
    assert (
        high.distribution["readmit"] > low.distribution["readmit"]
    )


def test_clinical_network_inadequate_sdoh_increases_readmit():
    net = build_readmission_network()
    adequate = infer(
        network=net, query="Outcome",
        evidence={"LACE_band": "high", "SDoH": "adequate"},
    )
    inadequate = infer(
        network=net, query="Outcome",
        evidence={"LACE_band": "high", "SDoH": "inadequate"},
    )
    assert (
        inadequate.distribution["readmit"]
        > adequate.distribution["readmit"]
    )


def test_clinical_network_round_trip_through_pydantic():
    net = build_readmission_network()
    payload = net.model_dump(mode="json")
    rebuilt = BayesianNetwork.model_validate(payload)
    assert set(rebuilt.nodes.keys()) == set(net.nodes.keys())


def test_clinical_network_marginal_outcome_is_well_defined():
    net = build_readmission_network()
    res = infer(network=net, query="Outcome")
    assert abs(sum(res.distribution.values()) - 1.0) < 1e-6
    assert 0.0 <= res.distribution["readmit"] <= 1.0
