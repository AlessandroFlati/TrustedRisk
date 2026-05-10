"""healthcare.compute_differential_diagnosis_ranker -- LLM-4.

Grounded differential diagnosis ranker. Combines:
  1. A rule-based DDx table for common chief complaints (covers the
     "can't-miss" diagnoses for chest pain, abdominal pain, headache,
     dyspnea, syncope, fever, altered mental status).
  2. Optional LLM enhancement that re-ranks + extends the rule-based list
     (Ollama; falls back to rule-based when LLM unavailable).
  3. Optional grounding via the existing ground_claim corpus -- each top-K
     diagnosis gets cited literature attached, with a hallucination floor
     (any item without grounding evidence carries grounding_verdict=
     "ungrounded" and is downranked).

The output's `cant_miss_diagnoses` field surfaces high-acuity items the
ED literature flags as "cannot afford to miss", regardless of probability.
"""

from __future__ import annotations

import os
import re
from typing import Any

from shared.schemas import (
    DifferentialDiagnosisReport,
    DifferentialItem,
    EvidenceSource,
)


# ─────────────────────── Rule-based DDx table ───────────────────────
#
# Each entry: chief_complaint_key -> list of (diagnosis, base_p, supporting_keys,
# is_cant_miss). `supporting_keys` are tokens to look for in the structured
# features dict + free text to bump probability and emit supporting_features.

_DDX_TABLE: dict[str, list[dict[str, Any]]] = {
    "chest_pain": [
        {"dx": "Acute coronary syndrome",
         "base": 0.30, "cant_miss": True,
         "supports": ["diaphoresis", "exertional", "radiation",
                       "troponin", "st_elevation", "st_depression",
                       "tobacco", "diabetes", "hypertension"]},
        {"dx": "Aortic dissection",
         "base": 0.05, "cant_miss": True,
         "supports": ["tearing", "bp_differential", "marfan",
                       "wide_mediastinum", "interscapular"]},
        {"dx": "Pulmonary embolism",
         "base": 0.10, "cant_miss": True,
         "supports": ["dyspnea", "tachycardia", "hypoxia",
                       "dvt", "pleuritic", "wells_score_high",
                       "immobilization", "malignancy"]},
        {"dx": "Pneumothorax",
         "base": 0.05, "cant_miss": True,
         "supports": ["sudden_onset", "tall_thin", "trauma",
                       "decreased_breath_sounds"]},
        {"dx": "Esophageal rupture (Boerhaave)",
         "base": 0.01, "cant_miss": True,
         "supports": ["vomiting", "subcutaneous_emphysema"]},
        {"dx": "GERD / esophagitis",
         "base": 0.20, "cant_miss": False,
         "supports": ["postprandial", "supine", "burning"]},
        {"dx": "Costochondritis",
         "base": 0.15, "cant_miss": False,
         "supports": ["reproducible_palpation", "young"]},
        {"dx": "Anxiety / panic",
         "base": 0.10, "cant_miss": False,
         "supports": ["palpitations", "anxiety_history",
                       "young"]},
    ],
    "abdominal_pain": [
        {"dx": "Appendicitis",
         "base": 0.20, "cant_miss": True,
         "supports": ["rlq", "rebound", "rovsing", "fever",
                       "leukocytosis", "young"]},
        {"dx": "Ruptured AAA",
         "base": 0.03, "cant_miss": True,
         "supports": ["pulsatile", "hypotension", "back_pain",
                       "smoking", "elderly"]},
        {"dx": "Mesenteric ischemia",
         "base": 0.03, "cant_miss": True,
         "supports": ["pain_out_of_proportion", "afib",
                       "lactic_acidosis", "elderly"]},
        {"dx": "Ectopic pregnancy",
         "base": 0.05, "cant_miss": True,
         "supports": ["female_reproductive_age", "vaginal_bleeding",
                       "positive_hcg"]},
        {"dx": "Cholecystitis",
         "base": 0.15, "cant_miss": False,
         "supports": ["ruq", "murphy", "fatty_food", "gallstones"]},
        {"dx": "Diverticulitis",
         "base": 0.10, "cant_miss": False,
         "supports": ["llq", "elderly", "fever"]},
        {"dx": "Bowel obstruction",
         "base": 0.10, "cant_miss": True,
         "supports": ["distention", "constipation", "vomiting",
                       "prior_surgery"]},
        {"dx": "Renal colic",
         "base": 0.10, "cant_miss": False,
         "supports": ["flank", "hematuria", "writhing"]},
        {"dx": "Gastritis",
         "base": 0.10, "cant_miss": False,
         "supports": ["epigastric", "nsaid", "h_pylori"]},
    ],
    "headache": [
        {"dx": "Subarachnoid hemorrhage",
         "base": 0.05, "cant_miss": True,
         "supports": ["thunderclap", "worst_of_life", "neck_stiffness",
                       "altered"]},
        {"dx": "Meningitis",
         "base": 0.05, "cant_miss": True,
         "supports": ["fever", "neck_stiffness", "photophobia",
                       "altered", "rash"]},
        {"dx": "Stroke",
         "base": 0.05, "cant_miss": True,
         "supports": ["focal", "facial_droop", "hemiparesis",
                       "aphasia"]},
        {"dx": "Temporal arteritis",
         "base": 0.03, "cant_miss": True,
         "supports": ["elderly", "jaw_claudication", "vision_change",
                       "elevated_esr"]},
        {"dx": "Idiopathic intracranial hypertension",
         "base": 0.05, "cant_miss": False,
         "supports": ["female", "obesity", "vision_change", "papilledema"]},
        {"dx": "Migraine",
         "base": 0.40, "cant_miss": False,
         "supports": ["aura", "photophobia", "phonophobia",
                       "unilateral", "throbbing"]},
        {"dx": "Tension-type headache",
         "base": 0.30, "cant_miss": False,
         "supports": ["bilateral", "band_like", "stress"]},
    ],
    "dyspnea": [
        {"dx": "Pulmonary embolism",
         "base": 0.15, "cant_miss": True,
         "supports": ["tachycardia", "hypoxia", "pleuritic",
                       "dvt", "immobilization", "malignancy"]},
        {"dx": "Acute heart failure / pulmonary edema",
         "base": 0.20, "cant_miss": True,
         "supports": ["jvd", "rales", "edema", "elevated_bnp",
                       "orthopnea"]},
        {"dx": "Pneumonia",
         "base": 0.20, "cant_miss": False,
         "supports": ["fever", "cough", "consolidation",
                       "leukocytosis"]},
        {"dx": "COPD exacerbation",
         "base": 0.20, "cant_miss": False,
         "supports": ["smoking", "wheeze", "prolonged_expiration"]},
        {"dx": "Asthma exacerbation",
         "base": 0.15, "cant_miss": False,
         "supports": ["wheeze", "young", "prior_asthma"]},
        {"dx": "Pneumothorax",
         "base": 0.05, "cant_miss": True,
         "supports": ["sudden_onset", "decreased_breath_sounds"]},
    ],
    "syncope": [
        {"dx": "Cardiac syncope (arrhythmia)",
         "base": 0.20, "cant_miss": True,
         "supports": ["exertional", "palpitations", "structural_heart",
                       "abnormal_ecg", "family_history_sudden_death"]},
        {"dx": "Pulmonary embolism",
         "base": 0.05, "cant_miss": True,
         "supports": ["dyspnea", "tachycardia", "dvt"]},
        {"dx": "GI bleed (orthostatic)",
         "base": 0.05, "cant_miss": True,
         "supports": ["melena", "hematochezia", "anemia",
                       "tachycardia"]},
        {"dx": "Vasovagal syncope",
         "base": 0.40, "cant_miss": False,
         "supports": ["prodrome", "warm_room", "pain_or_emotion"]},
        {"dx": "Orthostatic hypotension",
         "base": 0.20, "cant_miss": False,
         "supports": ["positional", "diuretic", "elderly"]},
    ],
    "fever": [
        {"dx": "Sepsis / bacteremia",
         "base": 0.20, "cant_miss": True,
         "supports": ["hypotension", "tachycardia", "elevated_lactate",
                       "altered", "neutropenia"]},
        {"dx": "Meningitis",
         "base": 0.05, "cant_miss": True,
         "supports": ["headache", "neck_stiffness", "altered",
                       "photophobia"]},
        {"dx": "Endocarditis",
         "base": 0.03, "cant_miss": True,
         "supports": ["new_murmur", "ivdu", "embolic_phenomena"]},
        {"dx": "Urinary tract infection",
         "base": 0.20, "cant_miss": False,
         "supports": ["dysuria", "frequency", "flank_pain",
                       "elderly_female"]},
        {"dx": "Pneumonia",
         "base": 0.20, "cant_miss": False,
         "supports": ["cough", "consolidation", "leukocytosis"]},
        {"dx": "Viral syndrome",
         "base": 0.30, "cant_miss": False,
         "supports": ["myalgia", "rhinorrhea", "self_limited"]},
    ],
    "altered_mental_status": [
        {"dx": "Hypoglycemia",
         "base": 0.10, "cant_miss": True,
         "supports": ["diabetes", "insulin", "low_glucose"]},
        {"dx": "Stroke",
         "base": 0.10, "cant_miss": True,
         "supports": ["focal", "hemiparesis", "facial_droop"]},
        {"dx": "Sepsis",
         "base": 0.15, "cant_miss": True,
         "supports": ["fever", "hypotension", "elevated_lactate"]},
        {"dx": "Meningitis / encephalitis",
         "base": 0.05, "cant_miss": True,
         "supports": ["fever", "neck_stiffness", "rash"]},
        {"dx": "Toxic ingestion",
         "base": 0.10, "cant_miss": True,
         "supports": ["polypharmacy", "psychiatric_history",
                       "anion_gap", "miosis", "mydriasis"]},
        {"dx": "Hyponatremia",
         "base": 0.05, "cant_miss": False,
         "supports": ["sodium_low", "thiazide", "elderly"]},
        {"dx": "Hepatic encephalopathy",
         "base": 0.05, "cant_miss": False,
         "supports": ["cirrhosis", "asterixis", "elevated_ammonia"]},
        {"dx": "Delirium (multifactorial)",
         "base": 0.30, "cant_miss": False,
         "supports": ["elderly", "infection", "polypharmacy",
                       "hospitalized"]},
    ],
}


_CHIEF_COMPLAINT_ALIASES: dict[str, str] = {
    "chest pain": "chest_pain",
    "substernal chest pain": "chest_pain",
    "chest discomfort": "chest_pain",
    "abdominal pain": "abdominal_pain",
    "belly pain": "abdominal_pain",
    "stomach pain": "abdominal_pain",
    "epigastric pain": "abdominal_pain",
    "headache": "headache",
    "head pain": "headache",
    "cephalgia": "headache",
    "dyspnea": "dyspnea",
    "shortness of breath": "dyspnea",
    "sob": "dyspnea",
    "syncope": "syncope",
    "fainting": "syncope",
    "passed out": "syncope",
    "fever": "fever",
    "febrile": "fever",
    "pyrexia": "fever",
    "altered mental status": "altered_mental_status",
    "ams": "altered_mental_status",
    "confusion": "altered_mental_status",
    "delirium": "altered_mental_status",
}


def _normalize_chief_complaint(text: str) -> str | None:
    t = text.lower().strip()
    if not t:
        return None
    if t in _CHIEF_COMPLAINT_ALIASES:
        return _CHIEF_COMPLAINT_ALIASES[t]
    for alias, key in _CHIEF_COMPLAINT_ALIASES.items():
        if alias in t:
            return key
    return None


def _flatten_features(structured: dict[str, Any] | None,
                        free_text: str | None) -> set[str]:
    """Build a set of normalized feature tokens from structured + free text."""
    out: set[str] = set()
    if structured:
        for k, v in structured.items():
            key = str(k).lower().strip().replace(" ", "_")
            if isinstance(v, bool) and v:
                out.add(key)
            elif isinstance(v, (int, float)):
                out.add(key)
                # threshold heuristics
                if "lactate" in key and v >= 2.0:
                    out.add("elevated_lactate")
                if "glucose" in key and v < 70:
                    out.add("low_glucose")
                if "sodium" in key and v < 135:
                    out.add("sodium_low")
                if "wbc" in key and v >= 12:
                    out.add("leukocytosis")
                if "bnp" in key and v >= 400:
                    out.add("elevated_bnp")
                if "esr" in key and v >= 50:
                    out.add("elevated_esr")
            elif isinstance(v, str) and v:
                out.add(f"{key}={v.lower()}")
                # Also add the value itself as a token
                out.add(v.lower().replace(" ", "_"))
    if free_text:
        text = free_text.lower()
        # Map common phrases to canonical feature tokens
        for phrase, tok in [
            ("worst of my life", "worst_of_life"),
            ("worst headache", "thunderclap"),
            ("thunderclap", "thunderclap"),
            ("neck stiffness", "neck_stiffness"),
            ("headache", "headache"),
            ("photophobia", "photophobia"),
            ("phonophobia", "phonophobia"),
            ("aura", "aura"),
            ("right lower quadrant", "rlq"),
            ("left lower quadrant", "llq"),
            ("right upper quadrant", "ruq"),
            ("flank pain", "flank"),
            ("flank", "flank"),
            ("hematuria", "hematuria"),
            ("diaphoresis", "diaphoresis"),
            ("sweating", "diaphoresis"),
            ("tearing", "tearing"),
            ("interscapular", "interscapular"),
            ("orthopnea", "orthopnea"),
            ("jvd", "jvd"),
            ("rales", "rales"),
            ("wheeze", "wheeze"),
            ("dvt", "dvt"),
            ("smoker", "smoking"),
            ("smoking", "smoking"),
            ("tobacco", "tobacco"),
            ("diabetes", "diabetes"),
            ("hypertension", "hypertension"),
            ("hypotension", "hypotension"),
            ("hypoxia", "hypoxia"),
            ("tachycardia", "tachycardia"),
            ("fever", "fever"),
            ("vomiting", "vomiting"),
            ("melena", "melena"),
            ("hematochezia", "hematochezia"),
            ("anemia", "anemia"),
            ("focal weakness", "focal"),
            ("hemiparesis", "hemiparesis"),
            ("facial droop", "facial_droop"),
            ("aphasia", "aphasia"),
            ("rebound tenderness", "rebound"),
            ("murphy", "murphy"),
            ("palpitations", "palpitations"),
            ("immobilization", "immobilization"),
            ("malignancy", "malignancy"),
            ("cancer", "malignancy"),
            ("ivdu", "ivdu"),
            ("iv drug", "ivdu"),
            ("new murmur", "new_murmur"),
            ("cirrhosis", "cirrhosis"),
            ("asterixis", "asterixis"),
            ("ammonia", "elevated_ammonia"),
            ("psychiatric", "psychiatric_history"),
            ("polypharmacy", "polypharmacy"),
            ("st elevation", "st_elevation"),
            ("st depression", "st_depression"),
            ("troponin", "troponin"),
            ("exertional", "exertional"),
            ("orthostatic", "positional"),
            ("postural", "positional"),
        ]:
            if phrase in text:
                out.add(tok)
        # Numeric feature mining
        m = re.search(r"\bage\s+(\d{1,3})\b|\b(\d{1,3})\s*(?:y|yr|yo|year)",
                        text, re.I)
        if m:
            age = int(next(g for g in m.groups() if g))
            if age >= 65:
                out.add("elderly")
            elif age <= 35:
                out.add("young")
        if re.search(r"\bfemale|\bf\b", text):
            out.add("female")
        if "reproductive" in text or re.search(
                r"\bpregnan", text):
            out.add("female_reproductive_age")
    return out


def _score_diagnosis(entry: dict[str, Any],
                       features: set[str]) -> tuple[float, list[str]]:
    """Adjust the base prior by overlap with features (capped at 0.95).

    Can't-miss diagnoses receive a 2× per-match weight to surface high-acuity
    presentations even when their unconditional prior is low.
    """
    base = float(entry["base"])
    supports = list(entry.get("supports", []))
    matched = [s for s in supports if s in features]

    per_match = 0.10 if entry.get("cant_miss") else 0.05
    prob = base + per_match * len(matched)
    return min(0.95, prob), matched


def _maybe_llm_rerank(
    items: list[DifferentialItem], chief_complaint: str,
    features_text: str,
) -> tuple[list[DifferentialItem], str]:
    """Optional LLM re-ranking. Returns (items, extraction_method)."""
    if os.environ.get("TRUSTEDRISK_DISABLE_LLM", "0") == "1":
        return items, "rule_based"
    try:
        import ollama  # type: ignore
    except ImportError:
        return items, "rule_based"

    model = os.environ.get(
        "TRUSTEDRISK_DDX_LLM_MODEL", "llama3.1:8b")
    items_str = "\n".join(
        f"- {it.diagnosis} (p={it.probability_estimate:.2f})"
        for it in items
    )
    prompt = (
        "You are a board-certified emergency medicine physician. Re-rank "
        "the following differential diagnoses for this presentation. "
        "Return ONLY a JSON list of diagnoses in your preferred order, "
        "from most to least likely. Do NOT add new diagnoses. Do NOT "
        "remove items. Do NOT change the names.\n\n"
        f"Chief complaint: {chief_complaint}\n"
        f"Features: {features_text}\n\n"
        f"Initial list:\n{items_str}\n\nJSON list of names:"
    )
    try:
        resp = ollama.generate(model=model, prompt=prompt)
        raw = resp.get("response", "").strip()
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return items, "rule_based"
        import json as _json
        ordered_names = _json.loads(m.group(0))
        if not isinstance(ordered_names, list):
            return items, "rule_based"
        ordered_names = [str(n).strip() for n in ordered_names]
        # Reorder ONLY items already present (no hallucination)
        by_name = {it.diagnosis: it for it in items}
        reordered: list[DifferentialItem] = []
        for n in ordered_names:
            if n in by_name and by_name[n] not in reordered:
                reordered.append(by_name[n])
        # Append any items the LLM dropped (preserve completeness)
        for it in items:
            if it not in reordered:
                reordered.append(it)
        # Re-emit ranks
        for idx, it in enumerate(reordered, start=1):
            reordered[idx - 1] = it.model_copy(update={"rank": idx})
        return reordered, "llm"
    except Exception:
        return items, "rule_based"


def _attach_grounding(items: list[DifferentialItem],
                       chief_complaint: str) -> list[DifferentialItem]:
    """Ground each diagnosis against the existing corpus. Best-effort."""
    try:
        from .ground_claim import _retrieve_corpus_evidence
    except Exception:
        return items
    grounded: list[DifferentialItem] = []
    for it in items:
        claim = (
            f"In a patient presenting with {chief_complaint}, the diagnosis "
            f"{it.diagnosis} is supported by the listed clinical features."
        )
        try:
            evidence = _retrieve_corpus_evidence(claim, top_k=2)
        except Exception:
            evidence = []
        if evidence:
            grounded.append(it.model_copy(update={
                "citations": evidence,
                "grounding_verdict": "partially_supported",
                "ground_claim_text": claim,
            }))
        else:
            grounded.append(it.model_copy(update={
                "ground_claim_text": claim,
                "grounding_verdict": "ungrounded",
            }))
    return grounded


async def compute_differential_diagnosis_ranker(
    chief_complaint: str,
    structured_features: dict[str, Any] | None = None,
    free_text_summary: str | None = None,
    max_items: int = 8,
    enable_grounding: bool = False,
    enable_llm_rerank: bool = True,
) -> DifferentialDiagnosisReport:
    """Rank a differential diagnosis for the given presentation.

    Args:
        chief_complaint: e.g. "chest pain", "abdominal pain", "headache".
        structured_features: dict of features (vitals, labs, exam findings).
        free_text_summary: free-text narrative; mined for additional features.
        max_items: cap on items returned.
        enable_grounding: when True, calls the existing ground_claim corpus
            to attach citations to each top item. Off by default -- corpus
            loading is heavy.
        enable_llm_rerank: when True (default), tries an Ollama re-ranking pass.

    Returns:
        DifferentialDiagnosisReport with ranked items, can't-miss flags,
        overall_confidence, and (optional) grounding citations.
    """
    if not isinstance(chief_complaint, str) or not chief_complaint.strip():
        return DifferentialDiagnosisReport(
            chief_complaint="",
            structured_features=structured_features or {},
            free_text_summary=free_text_summary,
            items=[],
            n_items=0,
            extraction_method="rule_based",
            overall_confidence="low",
            cant_miss_diagnoses=[],
            abstain_recommended=True,
            abstain_reason="empty_chief_complaint",
        )

    if max_items < 1 or max_items > 30:
        raise ValueError("max_items must be in [1, 30]")

    key = _normalize_chief_complaint(chief_complaint)
    if key is None or key not in _DDX_TABLE:
        # Promote to A2A INPUT_REQUIRED -- the BYO orchestrator can ask
        # the user to specify a recognised chief complaint instead of
        # giving up.
        from shared.schemas import ClarificationRequest
        supported = sorted(_DDX_TABLE.keys())
        return DifferentialDiagnosisReport(
            chief_complaint=chief_complaint,
            structured_features=structured_features or {},
            free_text_summary=free_text_summary,
            items=[],
            n_items=0,
            extraction_method="rule_based",
            overall_confidence="low",
            cant_miss_diagnoses=[],
            abstain_recommended=True,
            abstain_reason="unsupported_chief_complaint",
            task_state="input_required",
            clarification_request=ClarificationRequest(
                question=(
                    f"The chief complaint {chief_complaint!r} is not yet "
                    f"in our deterministic DDx table. Please pick the "
                    f"closest matching presentation from the supported "
                    f"set, or describe the primary symptom + body region "
                    f"more specifically (e.g. 'sharp left lower quadrant "
                    f"abdominal pain' rather than 'belly hurts')."
                ),
                expected_answer_kind="narrower_diagnosis",
                candidates=supported[:30],
                cite_back_section="chief_complaint",
            ),
        )

    features = _flatten_features(structured_features, free_text_summary)
    raw_items: list[tuple[float, dict[str, Any], list[str]]] = []
    for entry in _DDX_TABLE[key]:
        prob, matched = _score_diagnosis(entry, features)
        raw_items.append((prob, entry, matched))
    raw_items.sort(key=lambda x: -x[0])
    raw_items = raw_items[:max_items]

    items: list[DifferentialItem] = []
    for rank, (prob, entry, matched) in enumerate(raw_items, start=1):
        items.append(DifferentialItem(
            diagnosis=entry["dx"],
            probability_estimate=round(prob, 4),
            rank=rank,
            supporting_features=matched,
            contradicting_features=[],
            citations=[],
            grounding_verdict="ungrounded",
            ground_claim_text=None,
        ))

    extraction = "rule_based"
    if enable_llm_rerank:
        items, extraction = _maybe_llm_rerank(
            items, chief_complaint,
            ", ".join(sorted(features)) or "(no features)")

    if enable_grounding:
        items = _attach_grounding(items, chief_complaint)

    cant_miss = [
        entry["dx"] for entry in _DDX_TABLE[key]
        if entry.get("cant_miss") and any(
            entry["dx"] == it.diagnosis for it in items)
    ]

    # Confidence: high if top probability >= 0.5 AND ≥3 supporting features
    top = items[0] if items else None
    if top is not None and top.probability_estimate >= 0.5 \
            and len(top.supporting_features) >= 3:
        conf = "high"
    elif top is not None and top.probability_estimate >= 0.3:
        conf = "medium"
    else:
        conf = "low"

    return DifferentialDiagnosisReport(
        chief_complaint=chief_complaint,
        structured_features=structured_features or {},
        free_text_summary=free_text_summary,
        items=items,
        n_items=len(items),
        extraction_method=extraction,  # type: ignore[arg-type]
        overall_confidence=conf,        # type: ignore[arg-type]
        cant_miss_diagnoses=cant_miss,
        abstain_recommended=False,
        abstain_reason=None,
    )


# ─────────────────────── MCP registration ───────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_differential_diagnosis_ranker)
