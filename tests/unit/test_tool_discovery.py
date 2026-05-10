"""GENAI-4 unit tests for the semantic tool discovery layer."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from a2a_agent import tool_discovery as td
from a2a_agent.tool_discovery import semantic_search_tools


@pytest.fixture(autouse=True)
def _reset_index():
    td._reset_index()
    yield
    td._reset_index()


@pytest.fixture
def keyword_only(monkeypatch):
    """Force the keyword fallback path (no sentence-transformers loaded)."""
    monkeypatch.setattr(td, "_load_embedder", lambda: None)


# ─────────────────────── Index enumeration ───────────────────────

def test_enumerate_tools_finds_canonical_set():
    records = td._enumerate_tools()
    names = {r["name"] for r in records}
    expected_subset = {
        "compute_readmission_risk",
        "compute_dka_severity",
        "compute_resolve_patient_from_query",
        "compute_differential_diagnosis_ranker",
        "ground_claim",
        "detect_phi",
    }
    assert expected_subset <= names


def test_enumerate_tools_records_bundle_memberships():
    records = td._enumerate_tools()
    by_name = {r["name"]: r for r in records}
    # ground_claim is in multiple bundles
    bundles = set(by_name["ground_claim"]["bundle_memberships"])
    assert len(bundles) >= 2


def test_enumerate_tools_skips_helpers_without_docstrings():
    """Internal helpers + private functions must NOT appear in the index."""
    records = td._enumerate_tools()
    names = {r["name"] for r in records}
    assert "_compute_lace_components" not in names
    assert "_score_los" not in names


# ─────────────────────── Validation ───────────────────────

def test_empty_query_returns_empty_result(keyword_only):
    r = semantic_search_tools("")
    assert r.n_returned == 0
    assert r.hits == []
    assert r.embedder_model == "empty_query"


def test_invalid_top_k_raises(keyword_only):
    with pytest.raises(ValueError, match="top_k"):
        semantic_search_tools("chest pain", top_k=0)
    td._reset_index()
    with pytest.raises(ValueError, match="top_k"):
        semantic_search_tools("chest pain", top_k=51)


# ─────────────────────── Keyword fallback path ───────────────────────

def test_keyword_fallback_finds_obvious_matches(keyword_only):
    r = semantic_search_tools("dka diabetic ketoacidosis severity", top_k=3)
    top_names = {h.tool_name for h in r.hits}
    assert "compute_dka_severity" in top_names


def test_keyword_fallback_finds_readmission_terms(keyword_only):
    r = semantic_search_tools("30-day readmission lace risk", top_k=3)
    top_names = {h.tool_name for h in r.hits}
    assert "compute_readmission_risk" in top_names


def test_keyword_fallback_finds_phi_detection(keyword_only):
    r = semantic_search_tools("scrub PHI from clinical notes", top_k=3)
    top_names = {h.tool_name for h in r.hits}
    assert "detect_phi" in top_names


def test_keyword_fallback_skips_zero_score(keyword_only):
    """Tools with no keyword overlap must NOT appear in results."""
    r = semantic_search_tools("xyzzyzyz nonsense query", top_k=10)
    for hit in r.hits:
        assert hit.score > 0.0


# ─────────────────────── Hit shape + summary ───────────────────────

def test_hits_have_required_fields(keyword_only):
    r = semantic_search_tools("readmission risk", top_k=3)
    assert r.hits
    for h in r.hits:
        assert h.tool_name
        assert h.summary
        assert h.docstring_excerpt
        assert -1.0 <= h.score <= 1.0


def test_hits_ordered_by_descending_score(keyword_only):
    r = semantic_search_tools("medication reconciliation discharge", top_k=10)
    scores = [h.score for h in r.hits]
    assert scores == sorted(scores, reverse=True)


def test_top_k_caps_results(keyword_only):
    r = semantic_search_tools("risk", top_k=2)
    assert len(r.hits) <= 2


def test_n_tools_indexed_matches_enumeration(keyword_only):
    r = semantic_search_tools("risk", top_k=1)
    expected = len(td._enumerate_tools())
    assert r.n_tools_indexed == expected


# ─────────────────────── Embedder path (mocked) ───────────────────────

def test_embedder_path_used_when_available(monkeypatch):
    """When the embedder + numpy are available, score is cosine-based."""
    import numpy as np

    class _MockEmbedder:
        def encode(self, texts, show_progress_bar=False):
            # Return distinct unit-norm vectors per text -- deterministic by hash.
            out = []
            for t in texts:
                seed = abs(hash(t)) % (2**32)
                rng = np.random.default_rng(seed)
                v = rng.standard_normal(8)
                out.append(v / np.linalg.norm(v))
            return np.asarray(out, dtype=np.float32)

    monkeypatch.setattr(td, "_load_embedder", lambda: _MockEmbedder())

    r = semantic_search_tools("compute LACE-based readmission risk", top_k=3)
    assert r.embedder_model.startswith("sentence-transformers/")
    # The mock embedding is deterministic -- top hits should still be valid
    assert r.n_returned >= 1
    for h in r.hits:
        assert -1.0 <= h.score <= 1.0


def test_index_cached_across_calls(monkeypatch):
    """The expensive build runs once."""
    call_count = {"n": 0}
    real_build = td._build_index

    def _spy_build():
        call_count["n"] += 1
        return real_build()

    monkeypatch.setattr(td, "_build_index", _spy_build)
    monkeypatch.setattr(td, "_load_embedder", lambda: None)

    semantic_search_tools("dka", top_k=2)
    semantic_search_tools("readmission", top_k=2)
    semantic_search_tools("phi", top_k=2)
    assert call_count["n"] == 1


# ─────────────────────── /api/tools/search endpoint ───────────────────────

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


def test_endpoint_returns_results(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.get("/api/tools/search?q=dka%20severity&top_k=3")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "hits" in body
    assert body["query"] == "dka severity"


def test_endpoint_rejects_missing_q(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.get("/api/tools/search")
    # FastAPI handles required-arg validation -> 422
    assert r.status_code in (400, 422)


def test_endpoint_rejects_invalid_top_k(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.get("/api/tools/search?q=risk&top_k=0")
    assert r.status_code == 400
