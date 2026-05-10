"""healthcare.compute_clinical_ner / negation / discharge_structurer -- CHART-1/2/3.

Three tools that pull structured information out of clinical free text:

  - CHART-1 compute_clinical_ner -- entity extraction (problems / meds /
    allergies / family-history) with rule-based floor + optional LLM
    enhancement.
  - CHART-2 compute_negation_temporal -- adds NegEx-style negation +
    temporal context (current / historical / family-history / rule-out)
    to extracted entities.
  - CHART-3 compute_structure_discharge_summary -- high-level extractor
    that produces a `StructuredDischargeSummary` scaffold suitable for
    downstream DecisionCard composition.

All three honor `TRUSTEDRISK_DISABLE_LLM=1` and fall back to deterministic
rule-based extraction. The rule-based floor covers the highest-prevalence
conditions, medications, and allergens -- the LLM enhancement is the
generalizer.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from shared.schemas import (
    ClinicalEntity,
    ClinicalNERReport,
    StructuredDischargeSummary,
)


# ─────────────────────── Lexicons (rule-based floor) ───────────────────────

_PROBLEM_PATTERNS: list[tuple[re.Pattern[str], str | None, str]] = [
    (re.compile(r"\b(heart\s+failure|chf|congestive\s+heart\s+failure)\b",
                  re.I), "I50.9", "ICD10CM"),
    (re.compile(r"\bmyocardial\s+infarction\b|\bmi\b|\bnstemi\b|\bstemi\b",
                  re.I), "I21.9", "ICD10CM"),
    (re.compile(r"\bdiabetes(?:\s+(mellitus|type\s*2|t2dm))?\b|\bdm\b|\bt2dm\b",
                  re.I), "E11.9", "ICD10CM"),
    (re.compile(r"\bdiabetes\s+type\s*1\b|\bt1dm\b", re.I),
     "E10.9", "ICD10CM"),
    (re.compile(r"\bhypertension\b|\bhtn\b|\bhigh\s+blood\s+pressure\b",
                  re.I), "I10", "ICD10CM"),
    (re.compile(r"\bcopd\b|\bchronic\s+obstructive\s+pulmonary\s+disease\b",
                  re.I), "J44.9", "ICD10CM"),
    (re.compile(r"\basthma\b", re.I), "J45.909", "ICD10CM"),
    (re.compile(r"\bpneumonia\b", re.I), "J18.9", "ICD10CM"),
    (re.compile(r"\bstroke\b|\bcva\b|\bcerebrovascular\s+accident\b",
                  re.I), "I63.9", "ICD10CM"),
    (re.compile(r"\bckd\b|\bchronic\s+kidney\s+disease\b", re.I),
     "N18.9", "ICD10CM"),
    (re.compile(r"\baki\b|\bacute\s+kidney\s+injury\b", re.I),
     "N17.9", "ICD10CM"),
    (re.compile(r"\bcirrhosis\b", re.I), "K74.60", "ICD10CM"),
    (re.compile(r"\batrial\s+fibrillation\b|\bafib\b|\ba\.\s*fib\b",
                  re.I), "I48.91", "ICD10CM"),
    (re.compile(r"\bdementia\b|\balzheimer", re.I), "F03.90", "ICD10CM"),
    (re.compile(r"\bdepression\b", re.I), "F32.9", "ICD10CM"),
    (re.compile(r"\bcancer\b|\bcarcinoma\b|\bneoplasm\b|\btumor\b",
                  re.I), None, "none"),
    (re.compile(r"\bsepsis\b|\bbacteremia\b", re.I), "A41.9", "ICD10CM"),
    (re.compile(r"\bdka\b|\bdiabetic\s+ketoacidosis\b", re.I),
     "E11.10", "ICD10CM"),
    (re.compile(r"\bobesity\b", re.I), "E66.9", "ICD10CM"),
    (re.compile(r"\bdvt\b|\bdeep\s+vein\s+thrombosis\b", re.I),
     "I82.40", "ICD10CM"),
    (re.compile(r"\bpulmonary\s+embolism\b|\bpe\b", re.I),
     "I26.99", "ICD10CM"),
    (re.compile(r"\bchest\s+pain\b", re.I), "R07.9", "ICD10CM"),
]


_MED_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\bwarfarin\b|\bcoumadin\b", re.I), "11289", "RXNORM"),
    (re.compile(r"\bapixaban\b|\beliquis\b", re.I), "1364430", "RXNORM"),
    (re.compile(r"\brivaroxaban\b|\bxarelto\b", re.I), "1037045", "RXNORM"),
    (re.compile(r"\baspirin\b|\basa\b", re.I), "1191", "RXNORM"),
    (re.compile(r"\bclopidogrel\b|\bplavix\b", re.I), "32968", "RXNORM"),
    (re.compile(r"\blisinopril\b", re.I), "29046", "RXNORM"),
    (re.compile(r"\benalapril\b", re.I), "3827", "RXNORM"),
    (re.compile(r"\blosartan\b", re.I), "52175", "RXNORM"),
    (re.compile(r"\bspironolactone\b|\baldactone\b", re.I), "9997", "RXNORM"),
    (re.compile(r"\bfurosemide\b|\blasix\b", re.I), "4603", "RXNORM"),
    (re.compile(r"\bmetoprolol\b", re.I), "6918", "RXNORM"),
    (re.compile(r"\bcarvedilol\b|\bcoreg\b", re.I), "20352", "RXNORM"),
    (re.compile(r"\bsimvastatin\b|\bzocor\b", re.I), "36567", "RXNORM"),
    (re.compile(r"\batorvastatin\b|\blipitor\b", re.I), "83367", "RXNORM"),
    (re.compile(r"\brosuvastatin\b|\bcrestor\b", re.I), "301542", "RXNORM"),
    (re.compile(r"\bmetformin\b|\bglucophage\b", re.I), "6809", "RXNORM"),
    (re.compile(r"\binsulin\s+glargine\b|\blantus\b", re.I),
     "274783", "RXNORM"),
    (re.compile(r"\bempagliflozin\b|\bjardiance\b", re.I),
     "1545653", "RXNORM"),
    (re.compile(r"\bsacubitril.*valsartan\b|\bentresto\b", re.I),
     "1656339", "RXNORM"),
    (re.compile(r"\bomeprazole\b|\bprilosec\b", re.I), "7646", "RXNORM"),
    (re.compile(r"\bpantoprazole\b|\bprotonix\b", re.I), "40790", "RXNORM"),
    (re.compile(r"\bibuprofen\b|\bmotrin\b|\badvil\b", re.I),
     "5640", "RXNORM"),
    (re.compile(r"\bacetaminophen\b|\btylenol\b|\bparacetamol\b", re.I),
     "161", "RXNORM"),
    (re.compile(r"\bamiodarone\b", re.I), "703", "RXNORM"),
    (re.compile(r"\blevothyroxine\b|\bsynthroid\b", re.I), "10582", "RXNORM"),
]


_ALLERGY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(allergic\s+to|allergy\s+to|allergies?:?\s+)"
                  r"([a-z0-9\-,\s]{2,40})\b", re.I),
    re.compile(r"\b(NKDA|no\s+known\s+drug\s+allergies)\b", re.I),
]


_FAMILY_HISTORY_TRIGGERS: list[re.Pattern[str]] = [
    re.compile(r"\bfamily\s+history\s+of\b", re.I),
    re.compile(r"\bfather'?s?\s+(?:history\s+of\s+)?", re.I),
    re.compile(r"\bmother'?s?\s+(?:history\s+of\s+)?", re.I),
    re.compile(r"\bbrother'?s?\s+", re.I),
    re.compile(r"\bsister'?s?\s+", re.I),
    re.compile(r"\b(maternal|paternal)\s+(?:grand|aunt|uncle)", re.I),
]


# ─────────────────────── Negation + temporal triggers (CHART-2) ───────────────────────

_NEGATION_TRIGGERS: list[re.Pattern[str]] = [
    re.compile(r"\bno\s+", re.I),
    re.compile(r"\bnot\s+", re.I),
    re.compile(r"\bdenies\s+", re.I),
    re.compile(r"\bwithout\s+", re.I),
    re.compile(r"\bnegative\s+for\s+", re.I),
    re.compile(r"\babsent\s+(of\s+)?", re.I),
    re.compile(r"\bruled?\s+out\s+", re.I),
    re.compile(r"\bfree\s+of\s+", re.I),
    re.compile(r"\bnon[-\s]", re.I),
]

_HISTORICAL_TRIGGERS: list[re.Pattern[str]] = [
    re.compile(r"\bhistory\s+of\b", re.I),
    re.compile(r"\bhx\s+of\b", re.I),
    re.compile(r"\bpast\s+medical\s+history\b", re.I),
    re.compile(r"\bs/p\b|\bstatus\s+post\b", re.I),
    re.compile(r"\bprior\s+", re.I),
    re.compile(r"\bformer\s+", re.I),
    re.compile(r"\bprevious(ly)?\b", re.I),
    re.compile(r"\bresolved\b", re.I),
]

_RULEOUT_TRIGGERS: list[re.Pattern[str]] = [
    re.compile(r"\brule\s+out\s+", re.I),
    re.compile(r"\br/o\s+", re.I),
    re.compile(r"\bsuspect(ed)?\b", re.I),
    re.compile(r"\bpossible\b", re.I),
    re.compile(r"\bquestionable\b", re.I),
]

_FUTURE_TRIGGERS: list[re.Pattern[str]] = [
    re.compile(r"\bplan(ned)?\s+", re.I),
    re.compile(r"\bschedule(d)?\s+", re.I),
    re.compile(r"\bwill\s+", re.I),
    re.compile(r"\bto\s+be\s+", re.I),
    re.compile(r"\bawaiting\s+", re.I),
]


# ─────────────────────── Helpers ───────────────────────

def _safe_window(text: str, start: int, before_chars: int = 60,
                    after_chars: int = 30) -> tuple[str, str]:
    """Return (text_before_entity, text_after_entity_start)."""
    pre_start = max(0, start - before_chars)
    pre = text[pre_start:start]
    post = text[start:start + after_chars]
    return pre, post


def _detect_negation(pre_text: str) -> bool:
    """Look for negation triggers within the last ~60 chars before the entity.

    Anchored: triggers within 60 chars and not followed by a clause break.
    """
    for trigger in _NEGATION_TRIGGERS:
        for m in trigger.finditer(pre_text):
            # Check there's no clause-break between trigger and end of pre
            tail = pre_text[m.end():]
            if not re.search(r"\.|;|,\s+(?:but|however|although)\b", tail):
                return True
    return False


def _detect_temporal(pre_text: str) -> str:
    for tr in _HISTORICAL_TRIGGERS:
        if tr.search(pre_text):
            return "historical"
    for tr in _RULEOUT_TRIGGERS:
        if tr.search(pre_text):
            return "rule_out"
    for tr in _FUTURE_TRIGGERS:
        if tr.search(pre_text):
            return "future_planned"
    return "current"


def _detect_family_history(pre_text: str) -> bool:
    for tr in _FAMILY_HISTORY_TRIGGERS:
        if tr.search(pre_text):
            return True
    return False


# ─────────────────────── Rule-based extraction (CHART-1 floor) ───────────────────────

def _extract_problems_rules(text: str) -> list[ClinicalEntity]:
    out: list[ClinicalEntity] = []
    seen: set[tuple[int, int]] = set()
    for pattern, code, vocab in _PROBLEM_PATTERNS:
        for m in pattern.finditer(text):
            span = (m.start(), m.end())
            if span in seen:
                continue
            seen.add(span)
            entity_type = "problem"
            pre, _ = _safe_window(text, m.start())
            if _detect_family_history(pre):
                entity_type = "family_history"
            out.append(ClinicalEntity(
                entity_type=entity_type,             # type: ignore[arg-type]
                text=m.group(0),
                start=m.start(), end=m.end(),
                normalized_concept_id=code,
                normalized_vocabulary=vocab,        # type: ignore[arg-type]
                is_negated=False,
                temporal_context="unspecified",     # type: ignore[arg-type]
                confidence=0.65,
            ))
    return out


def _extract_medications_rules(text: str) -> list[ClinicalEntity]:
    out: list[ClinicalEntity] = []
    seen: set[tuple[int, int]] = set()
    for pattern, rxcui, vocab in _MED_PATTERNS:
        for m in pattern.finditer(text):
            span = (m.start(), m.end())
            if span in seen:
                continue
            seen.add(span)
            out.append(ClinicalEntity(
                entity_type="medication",
                text=m.group(0),
                start=m.start(), end=m.end(),
                normalized_concept_id=rxcui,
                normalized_vocabulary=vocab,         # type: ignore[arg-type]
                is_negated=False,
                temporal_context="unspecified",      # type: ignore[arg-type]
                confidence=0.70,
            ))
    return out


def _extract_allergies_rules(text: str) -> list[ClinicalEntity]:
    out: list[ClinicalEntity] = []
    for pattern in _ALLERGY_PATTERNS:
        for m in pattern.finditer(text):
            tag = m.group(0)
            if "nkda" in tag.lower() or \
                    "no known drug" in tag.lower():
                out.append(ClinicalEntity(
                    entity_type="allergy",
                    text=tag,
                    start=m.start(), end=m.end(),
                    is_negated=True,
                    temporal_context="current",     # type: ignore[arg-type]
                    confidence=0.90,
                ))
                continue
            # Otherwise the second group holds the allergen text
            if m.lastindex and m.lastindex >= 2:
                allergen = m.group(2)
                a_start = m.start(2)
                a_end = m.end(2)
                out.append(ClinicalEntity(
                    entity_type="allergy",
                    text=allergen.strip(),
                    start=a_start, end=a_end,
                    is_negated=False,
                    temporal_context="current",     # type: ignore[arg-type]
                    confidence=0.65,
                ))
    return out


# ─────────────────────── LLM extraction (optional) ───────────────────────

_DEFAULT_NER_MODEL = os.environ.get(
    "TRUSTEDRISK_NER_LLM_MODEL", "llama3.1:8b")


def _call_ollama(prompt: str) -> str | None:
    if os.environ.get("TRUSTEDRISK_DISABLE_LLM", "0") == "1":
        return None
    try:
        import ollama  # type: ignore
    except ImportError:
        return None
    try:
        resp = ollama.generate(model=_DEFAULT_NER_MODEL, prompt=prompt,
                                  options={"temperature": 0.0})
        return str(resp.get("response", "")).strip()
    except Exception:
        return None


_NER_PROMPT = """Extract clinical entities from this clinical note. \
Return ONLY a JSON object with key "entities", whose value is a list of \
items. Each item must have: entity_type (one of: problem, medication, \
allergy, family_history, procedure, lab_result, vital_sign), text \
(verbatim from the note), start (int, character index), end (int, \
character index).

DO NOT invent entities not present in the note. Use the source character \
indices exactly. Limit to ≤ 50 entities.

Note:
{text}

JSON:"""


def _llm_extract(text: str) -> list[ClinicalEntity] | None:
    raw = _call_ollama(_NER_PROMPT.format(text=text[:5000]))
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        payload = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    items = payload.get("entities") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None

    valid_types = {"problem", "medication", "allergy", "family_history",
                      "procedure", "lab_result", "vital_sign"}
    out: list[ClinicalEntity] = []
    for it in items[:50]:
        if not isinstance(it, dict):
            continue
        et = it.get("entity_type")
        if et not in valid_types:
            continue
        try:
            start = int(it.get("start", 0))
            end = int(it.get("end", 0))
            if start < 0 or end <= start or end > len(text):
                continue
            entity_text = text[start:end]
        except Exception:
            continue
        out.append(ClinicalEntity(
            entity_type=et,                          # type: ignore[arg-type]
            text=entity_text,
            start=start, end=end,
            normalized_concept_id=None,
            normalized_vocabulary="none",
            is_negated=False,
            temporal_context="unspecified",          # type: ignore[arg-type]
            confidence=0.55,
        ))
    return out


# ─────────────────────── CHART-2: Apply negation + temporal context ───────────────────────

def apply_negation_and_temporal(
    text: str,
    entities: list[ClinicalEntity],
) -> list[ClinicalEntity]:
    """Update each entity's is_negated + temporal_context based on its
    surrounding text. Returns a new list (entities are immutable)."""
    out: list[ClinicalEntity] = []
    for e in entities:
        pre, _ = _safe_window(text, e.start)
        is_negated = _detect_negation(pre)
        if e.is_negated:
            is_negated = e.is_negated  # respect prior negation (e.g. NKDA)
        is_family = _detect_family_history(pre)
        temporal = "family_history" if is_family else _detect_temporal(pre)
        new_type = "family_history" if (is_family
                                                and e.entity_type == "problem") \
            else e.entity_type
        out.append(e.model_copy(update={
            "is_negated": is_negated,
            "temporal_context": temporal,
            "entity_type": new_type,
        }))
    return out


# ─────────────────────── Public API ───────────────────────

async def compute_clinical_ner(
    text: str,
    use_llm: bool = True,
) -> ClinicalNERReport:
    """Extract clinical entities from free-text clinical notes.

    Args:
        text: free-text input (max ~5000 chars sent to LLM if used).
        use_llm: when True, supplements rule-based extraction with LLM
            output (deduplicated). Off -> rule-based only.

    Returns:
        ClinicalNERReport with entities + per-type counts.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string.")
    if not text.strip():
        return ClinicalNERReport(
            source_text_length=0, n_entities=0, entities=[],
            method="rule_based",
        )

    rules_out = (_extract_problems_rules(text)
                    + _extract_medications_rules(text)
                    + _extract_allergies_rules(text))

    method = "rule_based"
    safety: list[str] = []
    if use_llm:
        llm_out = _llm_extract(text)
        if llm_out is not None:
            # Merge by span -- rules win on overlap (more reliable IDs)
            spans = {(e.start, e.end) for e in rules_out}
            llm_only = [e for e in llm_out
                            if (e.start, e.end) not in spans]
            rules_out.extend(llm_only)
            method = "hybrid"
        else:
            safety.append("llm_unavailable_rule_based_only")

    entities = apply_negation_and_temporal(text, rules_out)
    entities.sort(key=lambda e: e.start)

    by_type: dict[str, int] = {}
    for e in entities:
        by_type[e.entity_type] = by_type.get(e.entity_type, 0) + 1

    return ClinicalNERReport(
        source_text_length=len(text),
        n_entities=len(entities),
        entities_by_type=by_type,
        entities=entities,
        method=method,                          # type: ignore[arg-type]
        safety_warnings=safety,
    )


async def compute_negation_temporal(
    text: str,
    entities: list[ClinicalEntity | dict],
) -> list[ClinicalEntity]:
    """Apply CHART-2 negation + temporal context to a pre-extracted
    entity list. Useful when entities come from a different NER source."""
    if not isinstance(text, str):
        raise ValueError("text must be a string.")
    coerced: list[ClinicalEntity] = []
    for e in entities or []:
        if isinstance(e, dict):
            coerced.append(ClinicalEntity.model_validate(e))
        else:
            coerced.append(e)
    return apply_negation_and_temporal(text, coerced)


# ─────────────────────── CHART-3: Discharge summary structurer ───────────────────────

_RECOMMENDATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # More specific patterns FIRST (home_with_care, snf, continued_admission)
    # so they win over the generic "discharge home" pattern.
    (re.compile(r"\b(skilled\s+nursing|snf|rehabilitation\s+facility|rehab\s+placement)\b",
                  re.I), "snf"),
    (re.compile(r"\b(continued?\s+admission|inpatient\s+observation|stay\s+admitted)\b",
                  re.I), "continued_admission"),
    (re.compile(
        r"\bhome\s+(?:with\s+)?(?:home\s+)?(?:health|care|nursing)"
        r"(?:\s+(?:nursing|referral|aide|services))?\b",
        re.I), "home_with_care"),
    (re.compile(r"\b(discharge\s+(to\s+)?home|disposition\s*[:=]?\s*home)\b",
                  re.I), "discharge_home"),
]


_FOLLOWUP_WINDOW_PATTERNS: list[tuple[re.Pattern[str], tuple[int, int]]] = [
    (re.compile(r"\bfollow[\s\-]?up\s+(?:in\s+|within\s+)?"
                   r"(\d+)\s*(?:to\s+|-)?\s*(\d+)?\s*(?:days|d|weeks|wks)\b",
                   re.I), (0, 0)),
]


def _extract_followup_window(text: str) -> tuple[int, int] | None:
    # Allow up to ~30 chars between "follow up" and the number (e.g.
    # "follow up with PCP in 7 days" -- "with PCP in " between them).
    m = re.search(
        r"follow[\s\-]?up\b[^.;\n]{0,40}?"
        r"(\d+)\s*(?:to\s+|-\s*)?(\d+)?\s*"
        r"(days|d|weeks|wks)\b",
        text, re.I,
    )
    if not m:
        return None
    n1 = int(m.group(1))
    n2 = int(m.group(2)) if m.group(2) else n1
    unit = m.group(3).lower()
    if unit in ("weeks", "wks"):
        n1 *= 7
        n2 *= 7
    return (min(n1, n2), max(n1, n2))


def _extract_recommendation(text: str) -> str | None:
    for pattern, action in _RECOMMENDATION_PATTERNS:
        if pattern.search(text):
            return action
    return None


_ABSTAIN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "defer to <X> judgment" -- allow words between "to" and "judgment"
    (re.compile(r"\bdefer\s+to\s+(?:\w+\s+){0,3}(?:judgment|judgement)\b",
                  re.I), "clinician_judgment_required"),
    (re.compile(r"\battending\s+to\s+(?:make\s+the\s+)?call\b", re.I),
     "clinician_judgment_required"),
    (re.compile(r"\b(?:high\s+)?uncertainty\s+(?:re|regarding)\b", re.I),
     "high_uncertainty"),
    (re.compile(r"\bawait(?:ing)?\s+further\s+(?:tests|imaging|labs|results)\b",
                  re.I), "awaiting_data"),
]


def _extract_abstain_triggers(text: str) -> list[str]:
    triggers: list[str] = []
    for pattern, label in _ABSTAIN_PATTERNS:
        if pattern.search(text):
            if label not in triggers:
                triggers.append(label)
    return triggers


async def compute_structure_discharge_summary(
    text: str,
    use_llm: bool = True,
) -> StructuredDischargeSummary:
    """Extract a DecisionCard scaffold from a free-text discharge summary.

    Args:
        text: the discharge summary (free text).
        use_llm: when True, supplements rule-based with LLM extraction.

    Returns:
        StructuredDischargeSummary with extracted recommendation,
        medications, problems, allergies, follow-up, abstain triggers.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string.")
    if not text.strip():
        return StructuredDischargeSummary(
            raw_text_length=0,
            method="rule_based",
            extraction_confidence="low",
        )

    ner = await compute_clinical_ner(text, use_llm=use_llm)
    rec = _extract_recommendation(text)

    meds: list[dict[str, str]] = []
    seen_meds: set[str] = set()
    for e in ner.entities:
        if e.entity_type != "medication" or e.is_negated:
            continue
        key = e.text.lower()
        if key in seen_meds:
            continue
        seen_meds.add(key)
        meds.append({
            "name": e.text,
            "rxcui": e.normalized_concept_id or "",
        })

    problems = sorted({
        e.text.lower() for e in ner.entities
        if e.entity_type == "problem" and not e.is_negated
        and e.temporal_context in ("current", "unspecified")
    })
    allergies = sorted({
        e.text.lower() for e in ner.entities
        if e.entity_type == "allergy" and not e.is_negated
    })
    followup = _extract_followup_window(text)
    abstain = _extract_abstain_triggers(text)

    n_signals = sum(1 for v in (rec, meds, problems, followup) if v)
    if n_signals >= 3:
        confidence: str = "high"
    elif n_signals >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    return StructuredDischargeSummary(
        raw_text_length=len(text),
        extracted_recommendation=rec,
        extracted_confidence=None,
        extracted_medications=meds,
        extracted_problems=list(problems),
        extracted_allergies=list(allergies),
        extracted_followup_window_days=followup,
        extracted_abstain_triggers=abstain,
        extraction_confidence=confidence,       # type: ignore[arg-type]
        method=ner.method,                       # type: ignore[arg-type]
        safety_warnings=ner.safety_warnings,
    )


# ─────────────────────── MCP registration ───────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_clinical_ner)
    mcp.tool()(compute_negation_temporal)
    mcp.tool()(compute_structure_discharge_summary)
