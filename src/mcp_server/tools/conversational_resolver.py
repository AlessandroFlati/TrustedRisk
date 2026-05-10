"""healthcare.compute_resolve_patient_from_query -- LLM-3.

Conversational SHARP context resolver. Maps a natural-language patient
reference (e.g. "the patient I just admitted with chest pain") to a ranked
list of FHIR Patient candidates by combining:
  1. Deterministic feature extraction (regex over a small clinical taxonomy).
  2. Optional LLM-enhanced extraction (Ollama, when available -- same pattern
     as discharge_counseling._polish_with_llm). Always falls back to (1).
  3. FHIR search via the SHARP context.
  4. Score = overlap of matched_features over a fixed weight schedule.

The output's `resolved_patient_id` field is set ONLY when there is a single
high-confidence match (top score >= 0.6 AND second-best margin >= 0.2).
Anything ambiguous -> abstain_recommended=True with a multi-candidate list.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from shared.schemas import (
    PatientCandidate,
    PatientFeatureExtraction,
    PatientResolutionResult,
)

from ..fhir.client import get_fhir_client


# ─────────────────────── Clinical lexicon ───────────────────────

_COMPLAINT_PATTERNS: list[tuple[str, str]] = [
    (r"chest pain|substernal|crushing chest", "chest pain"),
    (r"shortness of breath|dyspnea|sob\b", "shortness of breath"),
    (r"abdominal pain|belly pain|epigastric", "abdominal pain"),
    (r"headache|cephalgia", "headache"),
    (r"dizziness|vertigo|syncope|fainting", "syncope"),
    (r"fever|febrile|pyrexia", "fever"),
    (r"stroke|cva|facial droop|hemiparesis|slurred speech", "stroke"),
    (r"sepsis|septic shock|bacteremia", "sepsis"),
    (r"trauma|polytrauma|mvc|fall from height", "trauma"),
    (r"dka|diabetic ketoacidosis|ketoacidosis", "dka"),
    (r"aki|acute kidney injury|renal failure", "aki"),
    (r"chf|heart failure|pulmonary edema", "heart failure"),
    (r"pneumonia|consolidation", "pneumonia"),
    (r"copd|exacerbation", "copd"),
    (r"suicid", "suicide ideation"),
    (r"preeclampsia|eclampsia", "preeclampsia"),
    (r"chemo|chemotherapy|carboplatin|cisplatin", "chemotherapy"),
]

_SEX_PATTERNS: list[tuple[str, str]] = [
    # Two forms: full word (\bmale\b) or post-age abbreviation (\d+yo m).
    # The post-age form is the common ED chart shorthand "58yo M".
    (r"\b(?:male|man|gentleman|guy)\b|\d+\s*(?:y[oa]?|yr|year[\s-]?old)\s+m\b",
     "male"),
    (r"\b(?:female|woman|lady|girl)\b|\d+\s*(?:y[oa]?|yr|year[\s-]?old)\s+f\b",
     "female"),
]

_TIME_ANCHOR_PATTERNS: list[tuple[str, int]] = [
    (r"just admitted|currently admitting|moments ago|right now", 6),
    (r"this (morning|afternoon|shift)", 12),
    (r"today|today's|earlier today", 24),
    (r"yesterday|last night", 48),
    (r"this week|past few days", 168),
    (r"last week", 336),
]

_CARE_ROLE_PATTERNS: list[tuple[str, str]] = [
    (r"\bmy patient\b|\bi (just )?admitted\b|\badmitting\b", "admitting"),
    (r"\battending\b|\brounding on\b", "attending"),
    (r"\bconsult(ation)?\b|\bsigning off\b", "consultant"),
]


# ─────────────────────── Deterministic extraction ───────────────────────

def _extract_age_range(text: str) -> tuple[int | None, int | None]:
    """Find an age token (e.g. '75yo', 'age 82', '50-year-old') and return
    a (min, max) range with ±5y bracketing."""
    m = re.search(r"\b(\d{1,3})[\s-]*(?:y|yr|yo|year[\s-]?old|year)s?\b",
                  text, re.I)
    if not m:
        return None, None
    age = int(m.group(1))
    if age > 130:
        return None, None
    return max(0, age - 5), min(130, age + 5)


def _extract_features_regex(query: str) -> PatientFeatureExtraction:
    """Build a PatientFeatureExtraction from the query string deterministically."""
    q = query.lower().strip()

    complaints = [tag for pat, tag in _COMPLAINT_PATTERNS
                    if re.search(pat, q)]
    age_min, age_max = _extract_age_range(q)

    sex: str | None = None
    for pat, tag in _SEX_PATTERNS:
        if re.search(pat, q):
            sex = tag
            break

    time_hours = 24
    for pat, hours in _TIME_ANCHOR_PATTERNS:
        if re.search(pat, q):
            time_hours = hours
            break

    role = "any"
    for pat, tag in _CARE_ROLE_PATTERNS:
        if re.search(pat, q):
            role = tag
            break

    # Confidence -- high if at least 2 dimensions hit
    n_hits = (len(complaints) > 0) + (age_min is not None) + (sex is not None) \
             + (role != "any")
    if n_hits >= 3:
        conf = "high"
    elif n_hits >= 2:
        conf = "medium"
    else:
        conf = "low"

    return PatientFeatureExtraction(
        chief_complaint_terms=complaints,
        age_min_years=age_min,
        age_max_years=age_max,
        sex=sex,                  # type: ignore[arg-type]
        time_anchor_hours=time_hours,
        care_role=role,           # type: ignore[arg-type]
        confidence=conf,          # type: ignore[arg-type]
        extraction_method="regex",
    )


# ─────────────────────── Optional LLM enhancement ───────────────────────

def _maybe_llm_extract(
    query: str, base: PatientFeatureExtraction,
) -> PatientFeatureExtraction:
    """Best-effort LLM enhancement. Falls back to the regex extraction."""
    if os.environ.get("TRUSTEDRISK_DISABLE_LLM", "0") == "1":
        return base
    try:
        import ollama  # type: ignore
    except ImportError:
        return base

    model = os.environ.get(
        "TRUSTEDRISK_RESOLVER_LLM_MODEL", "llama3.1:8b")
    prompt = (
        "Extract clinical features from this patient query. Return JSON ONLY "
        "with these keys: chief_complaint_terms (list of strings), "
        "age_min_years (int or null), age_max_years (int or null), "
        "sex (\"male\"/\"female\"/\"other\"/null), "
        "time_anchor_hours (int 1-720), "
        "care_role (\"admitting\"/\"attending\"/\"consultant\"/\"any\"). "
        "Use medical taxonomy. NEVER invent patient identifiers.\n\n"
        f"Query: {query!r}\n\nJSON:"
    )
    try:
        resp = ollama.generate(model=model, prompt=prompt)
        raw = resp.get("response", "").strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return base
        import json as _json
        parsed = _json.loads(m.group(0))
        merged = base.model_dump()
        for key in ("chief_complaint_terms", "age_min_years",
                       "age_max_years", "sex", "time_anchor_hours",
                       "care_role"):
            if parsed.get(key) is not None:
                merged[key] = parsed[key]
        merged["extraction_method"] = "llm"
        # Tighten confidence one tier (LLM picked something up)
        if merged["confidence"] == "low":
            merged["confidence"] = "medium"
        return PatientFeatureExtraction.model_validate(merged)
    except Exception:
        return base


# ─────────────────────── FHIR search + scoring ───────────────────────

async def _search_candidates(
    feats: PatientFeatureExtraction, limit: int,
) -> list[dict[str, Any]]:
    """Search FHIR for recent Encounters; return raw FHIR dicts.

    The search uses _lastUpdated to honour the time anchor + an unfiltered
    Encounter limit, then we filter Patient demographics + complaints client-side.
    Real production deployments would push more filters server-side.
    """
    client = await get_fhir_client()

    cutoff = datetime.now(timezone.utc) - timedelta(hours=feats.time_anchor_hours)
    cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    encounters = await (
        client.resources("Encounter")
              .search(_lastUpdated=f"ge{cutoff_iso}")
              .limit(max(20, limit * 4))
              .fetch()
    )

    results: list[dict[str, Any]] = []
    seen_pids: set[str] = set()
    for enc in encounters:
        e = _to_dict(enc)
        # Encounter.subject.reference = "Patient/<pid>"
        subject = e.get("subject", {}) or {}
        ref = subject.get("reference", "")
        pid = ref.split("/")[-1] if "/" in ref else ref
        if not pid or pid in seen_pids:
            continue
        seen_pids.add(pid)
        # Fetch the Patient resource for demographics
        try:
            patients = await client.resources("Patient").search(_id=pid).fetch()
        except Exception:
            patients = []
        patient_dict = _to_dict(patients[0]) if patients else {}
        results.append({
            "patient_id": pid,
            "encounter": e,
            "patient": patient_dict,
        })
    return results


def _to_dict(resource: Any) -> dict[str, Any]:
    if hasattr(resource, "serialize"):
        return resource.serialize()
    if isinstance(resource, dict):
        return resource
    if hasattr(resource, "items"):
        return dict(resource)
    return {}


def _patient_age_years(p: dict[str, Any]) -> int | None:
    bd = p.get("birthDate")
    if not isinstance(bd, str):
        return None
    try:
        dob = datetime.fromisoformat(bd[:10])
    except ValueError:
        return None
    return (datetime.now(timezone.utc).date() - dob.date()).days // 365


def _score_candidate(
    candidate: dict[str, Any], feats: PatientFeatureExtraction,
) -> tuple[float, list[str], str | None, str | None]:
    """Return (score, matched_feature_labels, encounter_start, chief_complaint)."""
    matched: list[str] = []
    weights = 0.0
    total_weight = 0.0

    # Time recency -- always counts
    enc = candidate.get("encounter") or {}
    period = enc.get("period") or {}
    start_iso = period.get("start")
    total_weight += 0.20
    if start_iso:
        matched.append(f"recent_encounter ({start_iso[:10]})")
        weights += 0.20

    # Chief complaint: search Encounter.reasonCode + Encounter.type
    enc_text_blob = ""
    for field in ("reasonCode", "type"):
        for code in enc.get(field, []) or []:
            if isinstance(code, dict):
                enc_text_blob += " " + (
                    code.get("text") or "")
                for cc in code.get("coding", []) or []:
                    enc_text_blob += " " + (cc.get("display") or "")
    enc_text_blob = enc_text_blob.lower()
    chief_complaint: str | None = None
    if feats.chief_complaint_terms:
        cc_weight = 0.40
        total_weight += cc_weight
        hits = [t for t in feats.chief_complaint_terms
                  if any(w in enc_text_blob for w in t.split())]
        if hits:
            matched.append(f"chief_complaint:{hits[0]}")
            chief_complaint = hits[0]
            weights += cc_weight * (len(hits) / len(feats.chief_complaint_terms))

    # Age range
    if feats.age_min_years is not None and feats.age_max_years is not None:
        age_weight = 0.20
        total_weight += age_weight
        age = _patient_age_years(candidate.get("patient") or {})
        if age is not None and feats.age_min_years <= age <= feats.age_max_years:
            matched.append(f"age:{age}")
            weights += age_weight

    # Sex
    if feats.sex is not None:
        sex_weight = 0.10
        total_weight += sex_weight
        if (candidate.get("patient") or {}).get("gender", "").lower() \
                == feats.sex:
            matched.append(f"sex:{feats.sex}")
            weights += sex_weight

    # Care role: encoded in Encounter.participant -- best-effort check
    if feats.care_role != "any":
        role_weight = 0.10
        total_weight += role_weight
        for part in enc.get("participant", []) or []:
            for code in (part.get("type") or []):
                for coding in code.get("coding", []) or []:
                    if feats.care_role in (
                            (coding.get("display") or "").lower()):
                        matched.append(f"care_role:{feats.care_role}")
                        weights += role_weight
                        break

    score = weights / total_weight if total_weight > 0 else 0.0
    return score, matched, start_iso, chief_complaint


async def compute_resolve_patient_from_query(
    query: str,
    limit: int = 5,
    time_window_hours: int | None = None,
    include_extraction_method: str = "auto",  # "auto" / "regex" / "llm_only"
) -> PatientResolutionResult:
    """Resolve a natural-language patient query to ranked Patient candidates.

    Args:
        query: free-text reference (e.g. "the patient I just admitted with chest pain").
        limit: max number of candidates returned.
        time_window_hours: if provided, overrides the LLM/regex-derived anchor.
        include_extraction_method: "auto" (LLM if available, else regex),
            "regex" (skip LLM), "llm_only" (require LLM, abstain otherwise).

    Returns:
        PatientResolutionResult. `resolved_patient_id` is set only when
        top score ≥ 0.6 AND margin to second-best ≥ 0.2.
    """
    if not isinstance(query, str) or not query.strip():
        return PatientResolutionResult(
            query_text=query or "",
            extracted_features=PatientFeatureExtraction(),
            candidates=[],
            n_candidates=0,
            resolved_patient_id=None,
            abstain_recommended=True,
            abstain_reason="empty_query",
        )

    method_pref = (include_extraction_method or "auto").lower()
    if method_pref not in {"auto", "regex", "llm_only"}:
        raise ValueError(
            "include_extraction_method must be one of auto/regex/llm_only.")

    feats = _extract_features_regex(query)
    if method_pref == "auto":
        feats = _maybe_llm_extract(query, feats)
    elif method_pref == "llm_only":
        enhanced = _maybe_llm_extract(query, feats)
        if enhanced.extraction_method != "llm":
            return PatientResolutionResult(
                query_text=query,
                extracted_features=feats,
                candidates=[],
                n_candidates=0,
                resolved_patient_id=None,
                abstain_recommended=True,
                abstain_reason="llm_unavailable_in_llm_only_mode",
            )
        feats = enhanced

    if time_window_hours is not None:
        feats = feats.model_copy(
            update={"time_anchor_hours": int(time_window_hours)})

    raw = await _search_candidates(feats, limit=limit)
    scored: list[PatientCandidate] = []
    for cand in raw:
        score, matched, start_iso, cc = _score_candidate(cand, feats)
        scored.append(PatientCandidate(
            patient_id=cand["patient_id"],
            match_score=round(score, 4),
            matched_features=matched,
            encounter_start=start_iso,
            chief_complaint=cc,
            rationale=f"matched {len(matched)} feature(s) over the past "
                        f"{feats.time_anchor_hours}h",
        ))
    scored.sort(key=lambda c: c.match_score, reverse=True)
    scored = scored[:limit]

    resolved: str | None = None
    abstain = False
    abstain_reason: str | None = None
    task_state = "completed"
    clarification = None

    if not scored:
        abstain = True
        abstain_reason = "no_candidates_in_time_window"
    elif scored[0].match_score >= 0.6 and (
            len(scored) == 1 or
            scored[0].match_score - scored[1].match_score >= 0.2):
        resolved = scored[0].patient_id
    else:
        # Ambiguous top match -- promote to A2A INPUT_REQUIRED so the
        # BYO orchestrator can ask the user to pick from candidates
        # instead of giving up.
        from shared.schemas import ClarificationRequest
        abstain = True
        abstain_reason = "ambiguous_top_match" if scored else "no_match"
        if scored:
            task_state = "input_required"
            clarification = ClarificationRequest(
                question=(
                    f"Multiple patients match {query!r}. Please specify "
                    f"which patient you mean by selecting one of the "
                    f"candidates below."
                ),
                expected_answer_kind="patient_id",
                candidates=[c.patient_id for c in scored[:5]],
                cite_back_section="candidates",
            )

    return PatientResolutionResult(
        query_text=query,
        extracted_features=feats,
        candidates=scored,
        n_candidates=len(scored),
        resolved_patient_id=resolved,
        abstain_recommended=abstain,
        abstain_reason=abstain_reason,
        task_state=task_state,                          # type: ignore[arg-type]
        clarification_request=clarification,
    )


# ─────────────────────── MCP registration ───────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_resolve_patient_from_query)
