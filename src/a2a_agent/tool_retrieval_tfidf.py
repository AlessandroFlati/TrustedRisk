"""Phase 14.14 Q3 -- Pure-Python TF-IDF tool retrieval.

A deterministic-floor companion to ``tool_discovery.py`` (which uses
sentence-transformers when available). Q3's purpose is to give the
planner a hard floor that always works in CI, on cloud images without
sentence-transformers, and on networks without model-download access.

The index is built once over the registered MCP tool surface
(name + docstring + bundle membership) and cached in-process. Lookups
are pure cosine similarity with deterministic top-k tie-breaking on the
tool name. No external dependencies -- only the Python standard library.

Used by:
  - :mod:`a2a_agent.planner` -- when no hand-curated intent rule matches
    the query, we surface the top-k tools by TF-IDF similarity as
    advisory plan steps.
  - Operator dashboards that need a deterministic "find me a tool" surface
    for the 145-tool catalog without paying the embedder cost.
"""

from __future__ import annotations

import inspect
import math
import re
import threading
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# Tokenisation + stop-word list
# ─────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "has", "have", "if", "in", "into", "is", "it", "its",
    "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "which", "with", "would", "could", "should", "any", "all", "not",
    "no", "do", "does", "did", "so", "than", "then", "there", "those",
    "these", "their", "they", "them", "we", "you", "your", "i", "me",
    "my", "our", "us", "he", "she", "his", "her", "him", "what",
    "when", "who", "how", "why", "where", "can", "may", "might",
    "shall", "will", "via", "such", "also", "between", "across", "per",
    "use", "uses", "using", "used", "based", "given", "via", "etc",
    "more", "most", "least", "less", "very", "much", "many",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, ``[a-z0-9_]+`` tokens, plus underscore-split sub-tokens
    so ``compute_readmission_risk`` matches both the full identifier and
    the bare words ``readmission`` / ``risk``. Stop-words and length<2
    are dropped."""
    if not text:
        return []
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw not in _STOP_WORDS and len(raw) > 1:
            out.append(raw)
        if "_" in raw:
            for sub in raw.split("_"):
                if sub and sub not in _STOP_WORDS and len(sub) > 1:
                    out.append(sub)
    return out


# ─────────────────────────────────────────────────────────────────────
# Catalogue
# ─────────────────────────────────────────────────────────────────────


def _enumerate_tool_records() -> list[dict[str, Any]]:
    """One record per registered MCP tool function: name + docstring +
    bundle memberships."""
    from mcp_server import tools as tools_pkg

    bundles_for_tool: dict[str, list[str]] = {}
    for bundle_id, names in tools_pkg.BUNDLES.items():
        for name in names:
            bundles_for_tool.setdefault(name, []).append(bundle_id)

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for module_name in tools_pkg.__all__:
        module = getattr(tools_pkg, module_name, None)
        if module is None:
            continue
        for attr_name, attr in inspect.getmembers(module):
            if not callable(attr):
                continue
            if not attr_name.startswith(("compute_", "detect_", "ground_")):
                continue
            if attr_name in seen:
                continue
            doc = (inspect.getdoc(attr) or "").strip()
            if not doc:
                continue
            seen.add(attr_name)
            records.append({
                "name": attr_name,
                "module": module.__name__,
                "docstring": doc[:1000],
                "bundle_memberships": bundles_for_tool.get(attr_name, []),
            })
    records.sort(key=lambda r: r["name"])
    return records


# ─────────────────────────────────────────────────────────────────────
# TF-IDF index
# ─────────────────────────────────────────────────────────────────────


class _TfIdfIndex:
    """Pure-Python TF-IDF cosine index. Vectors are sparse dicts."""

    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        self.n_docs = len(records)
        self.df: Counter[str] = Counter()
        self.tokens_per_doc: list[list[str]] = []
        for r in records:
            text = (
                f"{r['name']} {r['docstring']} "
                f"{' '.join(r['bundle_memberships'])}"
            )
            toks = _tokenize(text)
            self.tokens_per_doc.append(toks)
            self.df.update(set(toks))

        self.idf: dict[str, float] = {
            t: math.log((self.n_docs + 1) / (df_t + 1)) + 1.0
            for t, df_t in self.df.items()
        }

        self.doc_vectors: list[dict[str, float]] = [
            self._vectorise(toks) for toks in self.tokens_per_doc
        ]
        self.doc_norms: list[float] = [
            math.sqrt(sum(v * v for v in vec.values())) or 1.0
            for vec in self.doc_vectors
        ]

    def _vectorise(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        max_tf = max(tf.values())
        out: dict[str, float] = {}
        for tok, count in tf.items():
            idf = self.idf.get(tok)
            if idf is None:
                continue
            out[tok] = (0.5 + 0.5 * count / max_tf) * idf
        return out

    def query(self, query_text: str, top_k: int) -> list[tuple[int, float]]:
        toks = _tokenize(query_text)
        if not toks:
            return []
        q_vec = self._vectorise(toks)
        if not q_vec:
            return []
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        scores: list[tuple[int, float]] = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            if not doc_vec:
                continue
            common = q_vec.keys() & doc_vec.keys()
            if not common:
                continue
            dot = sum(q_vec[t] * doc_vec[t] for t in common)
            sim = dot / (q_norm * self.doc_norms[idx])
            if sim > 0.0:
                scores.append((idx, sim))
        # Deterministic ordering: similarity desc, then tool name asc
        scores.sort(
            key=lambda x: (-x[1], self.records[x[0]]["name"])
        )
        return scores[:top_k]


# ─────────────────────────────────────────────────────────────────────
# Cached singleton + reset (test helper)
# ─────────────────────────────────────────────────────────────────────


_INDEX_LOCK = threading.Lock()
_INDEX: _TfIdfIndex | None = None


def _get_index() -> _TfIdfIndex:
    global _INDEX
    with _INDEX_LOCK:
        if _INDEX is None:
            _INDEX = _TfIdfIndex(_enumerate_tool_records())
        return _INDEX


def _reset_index() -> None:
    """Test helper -- discard the cached index."""
    global _INDEX
    with _INDEX_LOCK:
        _INDEX = None


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


class ToolRetrievalHit(BaseModel):
    tool_name: str
    score: float = Field(ge=0.0, le=1.0)
    bundle_memberships: list[str] = Field(default_factory=list)
    docstring_excerpt: str


class ToolRetrievalResult(BaseModel):
    query: str
    n_indexed: int
    top_k: int
    hits: list[ToolRetrievalHit]
    method: str = "tfidf"
    rationale: str


def retrieve_tools_by_query(
    query: str, *, top_k: int = 5,
) -> ToolRetrievalResult:
    """Return up to ``top_k`` tools ranked by TF-IDF cosine similarity.

    Pure-deterministic -- same query always yields the same ranking, the
    index is built once per process, and ties are broken on tool name.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    index = _get_index()
    hits_raw = index.query(query, top_k)
    hits = [
        ToolRetrievalHit(
            tool_name=index.records[i]["name"],
            score=round(score, 6),
            bundle_memberships=list(
                index.records[i]["bundle_memberships"]
            ),
            docstring_excerpt=index.records[i]["docstring"][:200],
        )
        for i, score in hits_raw
    ]
    return ToolRetrievalResult(
        query=query, n_indexed=index.n_docs, top_k=top_k,
        hits=hits, method="tfidf",
        rationale=(
            f"TF-IDF cosine ranking over {index.n_docs} indexed tools; "
            f"returned {len(hits)} hit(s). Pure-Python deterministic "
            f"floor -- no embedder loaded."
        ),
    )
