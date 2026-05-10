"""healthcare.ground_claim -- evidence grounding for clinical claims.

Per design doc §3.3.3:
  1. Decompose claim_text into atomic sub-claims via simple NER + regex.
  2. For each sub-claim, retrieve supporting passages:
       - From FHIR bundle of the patient (Observations, Conditions).
       - From the pre-indexed grounding corpus (faiss + sentence-transformers).
  3. Classify verdict: supported | partially_supported | unsupported.
  4. Aggregate into overall_verdict.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from shared.schemas import (
    ClaimGrounding,
    EvidenceSource,
    SubClaim,
)

from ..fhir.client import fetch_patient_bundle, resolve_patient_id


# ─────────────────────────────────────────────────────────────────────
# Grounding index loading (lazy, cached)
# ─────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_grounding_index() -> dict[str, Any] | None:
    """Load the faiss index + chunks from the pickle produced by W3."""
    path = os.environ.get(
        "TRUSTEDRISK_GROUNDING_INDEX_PATH", "data/grounding_index_mpnet.pkl"
    )
    fp = Path(path)
    if not fp.exists():
        return None  # Stub mode -- no grounding corpus available
    try:
        with fp.open("rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, OSError):
        return None


@lru_cache(maxsize=1)
def _load_embedder():
    """Lazy-load the sentence-transformer matching the grounding index embedder."""
    index_pkg = _load_grounding_index()
    if index_pkg is None:
        return None
    model_name = (index_pkg.get("embedder") or {}).get("name")
    if not model_name:
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer(model_name)
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def ground_claim(
    claim_text: str,
    patient_id: str | None = None,
    sources: list[Literal["fhir", "guidelines"]] | None = None,
) -> ClaimGrounding:
    """Ground a clinical claim against FHIR patient data + guideline corpus.

    Args:
        claim_text: prose claim to verify (e.g., "patient is stable for discharge").
        patient_id: FHIR Patient ID for FHIR evidence retrieval.
        sources: subset of ["fhir", "guidelines"] to consult (default both).

    Returns:
        ClaimGrounding with per-sub-claim verdicts + overall_verdict.
    """
    sources_list = sources or ["fhir", "guidelines"]
    pid = await resolve_patient_id(patient_id) if "fhir" in sources_list else ""

    # Step 1 -- decompose claim into sub-claims
    sub_claim_texts = _decompose_claim(claim_text)

    # Step 2 -- gather evidence for each sub-claim
    fhir_bundle = await fetch_patient_bundle(pid) if pid else None

    sub_claims: list[SubClaim] = []
    for sc_text, atomic_kind in sub_claim_texts:
        evidence: list[EvidenceSource] = []

        if "fhir" in sources_list and fhir_bundle:
            evidence.extend(_retrieve_fhir_evidence(fhir_bundle, sc_text, atomic_kind))

        if "guidelines" in sources_list:
            evidence.extend(_retrieve_corpus_evidence(sc_text))

        verdict, confidence, reason = _classify(sc_text, atomic_kind, evidence)
        sub_claims.append(SubClaim(
            text=sc_text,
            atomic_kind=atomic_kind,  # type: ignore
            verdict=verdict,
            confidence=confidence,
            evidence_sources=evidence,
            reason_if_unsupported=reason,
        ))

    # Step 3 -- aggregate overall verdict
    overall = _aggregate_overall(sub_claims)

    # Context fingerprint
    fp_input = f"{pid}||{claim_text}||{datetime.now(timezone.utc).isoformat()}"
    fingerprint = "sha256:" + hashlib.sha256(fp_input.encode("utf-8")).hexdigest()

    return ClaimGrounding(
        claim_text=claim_text,
        sub_claims=sub_claims,
        overall_verdict=overall,
        context_fingerprint=fingerprint,
        grounded_at=datetime.now(timezone.utc),
    )


# ─────────────────────────────────────────────────────────────────────
# Sub-claim decomposition (regex-based, v1 -- production: clinical NER)
# ─────────────────────────────────────────────────────────────────────

_VITAL_KEYWORDS = re.compile(
    r"\b(stable|vital|blood pressure|heart rate|oxygen|saturation|"
    r"respiratory|temperature|pulse)\b", re.IGNORECASE,
)
_SYMPTOM_KEYWORDS = re.compile(
    r"\b(pain|dyspnea|nausea|dizzy|fatigue|asymptomatic)\b", re.IGNORECASE,
)
_MEDICATION_KEYWORDS = re.compile(
    r"\b(medication|prescrib|anticoagulant|antibiotic|insulin|dialysis)\b",
    re.IGNORECASE,
)
_PROCEDURE_KEYWORDS = re.compile(
    r"\b(surgery|procedure|post-op|CABG|catheter|biopsy)\b", re.IGNORECASE,
)


def _decompose_claim(claim_text: str) -> list[tuple[str, str]]:
    """Split claim into atomic sub-claims with atomic_kind labels."""
    # Simple split on conjunctions and sentence boundaries
    parts = re.split(r"[.;]|\band\b|\bbut\b", claim_text)
    sub_claims: list[tuple[str, str]] = []
    for p in parts:
        p = p.strip().rstrip(",. ")
        if not p or len(p) < 5:
            continue
        if _VITAL_KEYWORDS.search(p):
            sub_claims.append((p, "vital"))
        elif _SYMPTOM_KEYWORDS.search(p):
            sub_claims.append((p, "symptom"))
        elif _MEDICATION_KEYWORDS.search(p):
            sub_claims.append((p, "medication"))
        elif _PROCEDURE_KEYWORDS.search(p):
            sub_claims.append((p, "procedure"))
        else:
            sub_claims.append((p, "general"))
    if not sub_claims:
        sub_claims.append((claim_text, "general"))
    return sub_claims


# ─────────────────────────────────────────────────────────────────────
# FHIR evidence retrieval
# ─────────────────────────────────────────────────────────────────────

def _retrieve_fhir_evidence(
    bundle: dict[str, Any],
    sub_claim: str,
    atomic_kind: str,
) -> list[EvidenceSource]:
    """Find FHIR resources that might support the sub-claim."""
    evidence: list[EvidenceSource] = []
    entries = bundle.get("entry") or []
    target_types = _target_types_for_kind(atomic_kind)
    now = datetime.now(timezone.utc)

    for entry in entries:
        res = entry.get("resource") if isinstance(entry, dict) else None
        if not isinstance(res, dict):
            continue
        rtype = res.get("resourceType")
        if rtype not in target_types:
            continue
        source_type = _res_to_source_type(rtype)
        if source_type is None:
            continue
        excerpt = _excerpt_from_resource(res)
        recency = _recency_days(res, now)
        evidence.append(EvidenceSource(
            source_type=source_type,  # type: ignore
            source_id=str(res.get("id", "")),
            excerpt=excerpt,
            relevance_score=_keyword_relevance(excerpt, sub_claim),
            recency_days=recency,
        ))
        if len(evidence) >= 3:
            break

    return sorted(evidence, key=lambda e: e.relevance_score, reverse=True)


def _target_types_for_kind(atomic_kind: str) -> set[str]:
    if atomic_kind == "vital":
        return {"Observation"}
    if atomic_kind == "symptom":
        return {"Observation", "Condition"}
    if atomic_kind == "medication":
        return {"MedicationRequest"}
    if atomic_kind == "procedure":
        return {"Procedure"}
    return {"Observation", "Condition"}


def _res_to_source_type(rtype: str) -> str | None:
    if rtype == "Observation":
        return "fhir_observation"
    if rtype == "Condition":
        return "fhir_condition"
    return None


def _excerpt_from_resource(res: dict[str, Any]) -> str:
    code = res.get("code") or {}
    coding = (code.get("coding") or [{}])[0]
    label = coding.get("display") or code.get("text") or ""
    val = ""
    qty = res.get("valueQuantity")
    if isinstance(qty, dict) and "value" in qty:
        val = f" = {qty['value']} {qty.get('unit', '')}".strip()
    return f"{label}{val}"[:200]


def _recency_days(res: dict[str, Any], now: datetime) -> int | None:
    """How many days ago was this resource recorded."""
    dt_str = (
        res.get("effectiveDateTime")
        or res.get("issued")
        or (res.get("period") or {}).get("start")
        or ""
    )
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(str(dt_str).replace("Z", "+00:00"))
        return max(0, (now - dt).days)
    except (ValueError, TypeError):
        return None


def _keyword_relevance(excerpt: str, sub_claim: str) -> float:
    """Simple keyword overlap between excerpt and sub-claim -> [0, 1]."""
    excerpt_words = set(re.findall(r"\w+", excerpt.lower()))
    claim_words = set(re.findall(r"\w+", sub_claim.lower()))
    if not claim_words:
        return 0.0
    common = excerpt_words & claim_words
    return min(1.0, len(common) / len(claim_words))


# ─────────────────────────────────────────────────────────────────────
# Corpus evidence retrieval (faiss + sentence-transformers)
# ─────────────────────────────────────────────────────────────────────

def _retrieve_corpus_evidence(sub_claim: str, top_k: int = 2) -> list[EvidenceSource]:
    """Retrieve top-K relevant chunks from the grounding corpus."""
    index_pkg = _load_grounding_index()
    if index_pkg is None:
        return []
    embedder = _load_embedder()
    if embedder is None:
        return []

    try:
        query_vec = embedder.encode([sub_claim], show_progress_bar=False)
    except Exception:
        return []

    import numpy as np
    query_arr = np.asarray(query_vec, dtype=np.float32)
    norms = np.linalg.norm(query_arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    query_arr = query_arr / norms

    faiss_index = index_pkg.get("faiss_index")
    chunks = index_pkg.get("chunks") or []
    if faiss_index is None or not chunks:
        return []

    # Handle both real faiss.IndexFlatIP and mock dict
    if isinstance(faiss_index, dict) and faiss_index.get("_mock"):
        vectors = np.array(faiss_index.get("vectors", []), dtype=np.float32)
        if vectors.size == 0:
            return []
        similarities = (vectors @ query_arr.T).flatten()
        top_idx = np.argsort(-similarities)[:top_k]
        scores = similarities[top_idx]
    else:
        try:
            scores, top_idx = faiss_index.search(query_arr, top_k)
            scores = scores[0]
            top_idx = top_idx[0]
        except Exception:
            return []

    evidence: list[EvidenceSource] = []
    for score, idx in zip(scores, top_idx, strict=False):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        # Faiss IP scores on un-normalized embeddings (e.g., multi-qa-mpnet
        # dot-product variant) can exceed 1.0 -- schema requires [0, 1], so
        # clip and document. Cosine-similarity indices stay in [-1, 1] and we
        # also clip negatives to 0.
        clipped = max(0.0, min(1.0, float(score)))
        evidence.append(EvidenceSource(
            source_type="guideline_passage",
            source_id=str(chunk.get("chunk_id", "")),
            excerpt=str(chunk.get("text", ""))[:200],
            relevance_score=clipped,
            recency_days=None,
        ))
    return evidence


# ─────────────────────────────────────────────────────────────────────
# Verdict classification + aggregation
# ─────────────────────────────────────────────────────────────────────

def _classify(
    sub_claim: str,
    atomic_kind: str,
    evidence: list[EvidenceSource],
) -> tuple[Literal["supported", "partially_supported", "unsupported"], float, str | None]:
    """Map evidence -> verdict for a sub-claim."""
    if not evidence:
        return (
            "unsupported", 0.0,
            "No FHIR evidence or guideline passage found for this sub-claim.",
        )

    max_relevance = max(e.relevance_score for e in evidence)
    # Penalize stale FHIR evidence for vital / medication claims
    if atomic_kind in ("vital", "medication"):
        fresh_evidence = [
            e for e in evidence
            if e.recency_days is None or e.recency_days <= 48
        ]
        if not fresh_evidence:
            recency = min(
                (e.recency_days for e in evidence if e.recency_days is not None),
                default=None,
            )
            rec_str = f"{recency}h+" if recency is not None else "unknown"
            return (
                "unsupported", 0.3,
                f"Evidence exists but is stale (oldest recency_days={rec_str}; "
                f"vital/medication claims require evidence within 48h).",
            )

    if max_relevance >= 0.5:
        return "supported", max_relevance, None
    if max_relevance >= 0.2:
        return "partially_supported", max_relevance, (
            "Evidence is weakly related to the sub-claim."
        )
    return "unsupported", max_relevance, "Evidence relevance below threshold."


def _aggregate_overall(
    sub_claims: list[SubClaim],
) -> Literal["supported", "partially_supported", "unsupported"]:
    if not sub_claims:
        return "unsupported"
    critical_kinds = {"vital", "medication", "procedure"}
    for sc in sub_claims:
        if sc.atomic_kind in critical_kinds and sc.verdict == "unsupported":
            return "unsupported"
    verdicts = {sc.verdict for sc in sub_claims}
    if verdicts == {"supported"}:
        return "supported"
    if "unsupported" in verdicts:
        return "unsupported" if len(verdicts) == 1 else "partially_supported"
    return "partially_supported"


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(ground_claim)
