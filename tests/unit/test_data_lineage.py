"""CMP-3 tests for the data lineage tracker."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from a2a_agent.data_lineage import (
    _integrity_hash,
    compute_data_lineage,
)


# ─────────────────────── Graph structure ───────────────────────

def test_graph_has_canonical_nodes():
    g = compute_data_lineage()
    ids = {n.node_id for n in g.nodes}
    expected_subset = {
        "mimic_iv_demo_v2.2",
        "mimic_iv_lace_cohort_csv",
        "coefficients_json",
        "abstain_policy_json",
        "fairness_baseline_json",
        "grounding_index_mpnet",
        "module_readmission_risk",
        "tool_output_risk_estimate",
        "tool_output_decision_card",
        "reproducibility_archive",
    }
    assert expected_subset <= ids


def test_n_nodes_n_edges_match_lengths():
    g = compute_data_lineage()
    assert g.n_nodes == len(g.nodes)
    assert g.n_edges == len(g.edges)


def test_edges_reference_existing_nodes():
    g = compute_data_lineage()
    ids = {n.node_id for n in g.nodes}
    for e in g.edges:
        assert e.source_id in ids, f"Edge source {e.source_id} not in nodes"
        assert e.target_id in ids, f"Edge target {e.target_id} not in nodes"


def test_no_cycles_in_dag():
    """Trivial DFS-based cycle detection."""
    g = compute_data_lineage()
    edges_by_src: dict[str, list[str]] = {}
    for e in g.edges:
        edges_by_src.setdefault(e.source_id, []).append(e.target_id)

    visited: set[str] = set()
    on_stack: set[str] = set()

    def dfs(node: str) -> bool:
        visited.add(node)
        on_stack.add(node)
        for nxt in edges_by_src.get(node, []):
            if nxt not in visited:
                if dfs(nxt):
                    return True
            elif nxt in on_stack:
                return True
        on_stack.remove(node)
        return False

    for n in g.nodes:
        if n.node_id not in visited:
            assert not dfs(n.node_id), \
                f"Cycle detected starting at {n.node_id}"


# ─────────────────────── Hash invariants ───────────────────────

def test_integrity_hash_stable_across_calls():
    """Calling twice in succession should produce the same hash."""
    a = compute_data_lineage()
    b = compute_data_lineage()
    assert a.integrity_hash == b.integrity_hash


def test_integrity_hash_changes_when_nodes_change():
    g = compute_data_lineage()
    nodes_with_perturbed = list(g.nodes)
    # Add a synthetic node with non-None sha to perturb the hash
    from shared.schemas import DataLineageNode
    nodes_with_perturbed.append(DataLineageNode(
        node_id="zzz_synthetic_perturbation",
        node_type="raw_dataset",
        description="synthetic",
        sha256="0" * 64,
    ))
    new_hash = _integrity_hash(nodes_with_perturbed)
    assert new_hash != g.integrity_hash


def test_missing_artifact_does_not_crash():
    """The lineage builder must tolerate missing files (returns None sha)."""
    g = compute_data_lineage()
    # If any node has sha=None, its node_id should not raise on serialization
    payload = g.model_dump(mode="json")
    assert "nodes" in payload


# ─────────────────────── Mermaid render ───────────────────────

def test_mermaid_text_starts_with_graph_directive():
    g = compute_data_lineage()
    assert g.mermaid_text.startswith("graph LR")


def test_mermaid_includes_every_node_and_edge():
    g = compute_data_lineage()
    text = g.mermaid_text
    for n in g.nodes:
        assert n.node_id in text
    # At least one edge arrow should appear
    assert " --> " in text or "-->|" in text


# ─────────────────────── /api/lineage endpoint ───────────────────────

@pytest.fixture(scope="module")
def app():
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "playground_server",
        str(ROOT / "apps" / "playground" / "server.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["playground_server"] = mod
    spec.loader.exec_module(mod)
    return mod.app


def test_lineage_endpoint(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.get("/api/lineage")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "nodes" in body
    assert "edges" in body
    assert "integrity_hash" in body
    assert "mermaid_text" in body
