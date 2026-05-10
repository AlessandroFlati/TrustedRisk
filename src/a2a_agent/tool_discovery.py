"""GENAI-4 -- Semantic tool discovery across the 40 MCP tools.

Builds a dense-vector index over (tool_name + docstring + bundle tags) and
ranks tools by cosine similarity to a natural-language query. The index
is built lazily on first call + cached in-process. Falls back to keyword
matching when sentence-transformers is unavailable, so production usage
never crashes -- only ranking quality degrades.

Designed to feed:
  - Agent runtime: "the agent doesn't know which tool fits this case" -> it
    queries this index instead of hard-coding bundle picks.
  - Operator dashboard: "which tools cover X" -- endpoint /api/tools/search.
"""

from __future__ import annotations

import inspect
import re
import threading
from typing import Any

from shared.schemas import ToolSearchHit, ToolSearchResult


# ─────────────────────── Index state (cached) ───────────────────────

_INDEX_LOCK = threading.Lock()
_INDEX: dict[str, Any] | None = None


_DEFAULT_EMBEDDER = "sentence-transformers/all-mpnet-base-v2"


def _reset_index() -> None:
    """Test helper -- discard the cached index."""
    global _INDEX
    with _INDEX_LOCK:
        _INDEX = None


# ─────────────────────── Tool inventory ───────────────────────

def _enumerate_tools() -> list[dict[str, Any]]:
    """Walk the MCP tool registry and produce one record per tool function.

    Each record has:
      - name (the registered tool function name)
      - module (importable path)
      - docstring (first 1000 chars)
      - bundle_memberships (list of bundle ids this tool appears in)
    """
    from mcp_server import tools as tools_pkg

    bundles_map = tools_pkg.BUNDLES
    bundles_for_tool: dict[str, list[str]] = {}
    for bundle_id, names in bundles_map.items():
        for name in names:
            bundles_for_tool.setdefault(name, []).append(bundle_id)

    records: list[dict[str, Any]] = []
    for module_name in tools_pkg.__all__:
        module = getattr(tools_pkg, module_name, None)
        if module is None:
            continue
        for attr_name, attr in inspect.getmembers(module):
            if not callable(attr):
                continue
            # Tool functions are async + take a registered name. Look for the
            # canonical pattern: function whose name starts with `compute_`,
            # `detect_`, `ground_`, or one of the standalone tool names.
            if not (attr_name.startswith(("compute_", "detect_", "ground_"))):
                continue
            doc = (inspect.getdoc(attr) or "").strip()
            if not doc:
                continue
            records.append({
                "name": attr_name,
                "module": module.__name__,
                "docstring": doc[:1000],
                "bundle_memberships": bundles_for_tool.get(attr_name, []),
            })
    # Dedup by name (some modules export helpers; keep first occurrence)
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for r in records:
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        unique.append(r)
    return unique


# ─────────────────────── Embedder loading (lazy) ───────────────────────

def _load_embedder():
    """Load sentence-transformers -- same model the grounding corpus uses."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        return SentenceTransformer(_DEFAULT_EMBEDDER)
    except Exception:
        return None


def _embed_texts(embedder, texts: list[str]):
    import numpy as np
    vecs = embedder.encode(texts, show_progress_bar=False)
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return arr / norms


# ─────────────────────── Build index ───────────────────────

def _build_index() -> dict[str, Any]:
    records = _enumerate_tools()
    embedder = _load_embedder()

    if embedder is None or not records:
        return {
            "records": records,
            "vectors": None,
            "embedder_model": "keyword_fallback",
        }

    texts = [
        f"{r['name']}. {r['docstring']} bundles: "
        f"{' '.join(r['bundle_memberships'])}"
        for r in records
    ]
    vectors = _embed_texts(embedder, texts)
    return {
        "records": records,
        "vectors": vectors,
        "embedder_model": _DEFAULT_EMBEDDER,
    }


def _ensure_index() -> dict[str, Any]:
    global _INDEX
    with _INDEX_LOCK:
        if _INDEX is None:
            _INDEX = _build_index()
        return _INDEX


# ─────────────────────── Search ───────────────────────

_KEYWORD_BOOST = 0.05


def _keyword_score(query: str, record: dict[str, Any]) -> float:
    """Light keyword overlap fallback when no embedder."""
    q_tokens = {t for t in re.findall(r"[a-z0-9]+", query.lower())
                  if len(t) >= 3}
    if not q_tokens:
        return 0.0
    target = (record["name"] + " " + record["docstring"] + " "
                + " ".join(record["bundle_memberships"])).lower()
    target_tokens = set(re.findall(r"[a-z0-9]+", target))
    overlap = q_tokens & target_tokens
    return len(overlap) / max(1, len(q_tokens))


def _rank_with_embedder(query: str, top_k: int,
                          index: dict[str, Any]) -> list[tuple[int, float]]:
    import numpy as np
    embedder = _load_embedder()
    if embedder is None or index["vectors"] is None:
        return []
    q_vec = _embed_texts(embedder, [query])
    sims = (index["vectors"] @ q_vec.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in order]


def _rank_with_keywords(query: str, top_k: int,
                          index: dict[str, Any]) -> list[tuple[int, float]]:
    scored = [
        (i, _keyword_score(query, r))
        for i, r in enumerate(index["records"])
    ]
    scored.sort(key=lambda t: -t[1])
    return scored[:top_k]


def semantic_search_tools(
    query: str,
    top_k: int = 5,
) -> ToolSearchResult:
    """Rank MCP tools by relevance to a natural-language query.

    Uses the cached embedder index when available; falls back to keyword
    overlap when sentence-transformers can't load. Either way, results
    are ranked descending by `score`.
    """
    if not isinstance(query, str) or not query.strip():
        return ToolSearchResult(
            query=query or "",
            n_tools_indexed=0,
            n_returned=0,
            embedder_model="empty_query",
            hits=[],
        )
    if top_k < 1 or top_k > 50:
        raise ValueError("top_k must be in [1, 50]")

    index = _ensure_index()
    n_indexed = len(index["records"])

    if index["vectors"] is not None:
        ranked = _rank_with_embedder(query, top_k, index)
        embedder_model = index["embedder_model"]
    else:
        ranked = _rank_with_keywords(query, top_k, index)
        embedder_model = index["embedder_model"]

    hits: list[ToolSearchHit] = []
    for idx, score in ranked:
        if score <= 0.0:
            continue
        rec = index["records"][idx]
        # Summary = first sentence of the docstring
        first_sentence = re.split(r"(?<=[.!?])\s", rec["docstring"], maxsplit=1)
        summary = first_sentence[0] if first_sentence else rec["docstring"][:120]
        hits.append(ToolSearchHit(
            tool_name=rec["name"],
            bundle_memberships=list(rec["bundle_memberships"]),
            score=float(round(score, 4)),
            summary=summary[:200],
            docstring_excerpt=rec["docstring"][:400],
        ))

    return ToolSearchResult(
        query=query,
        n_tools_indexed=n_indexed,
        n_returned=len(hits),
        embedder_model=embedder_model,
        hits=hits,
    )
