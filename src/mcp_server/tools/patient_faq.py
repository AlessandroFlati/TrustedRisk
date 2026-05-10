"""healthcare.compute_patient_faq -- PATIENT-2.

LLM-grounded patient FAQ answerer. The patient asks a question (free
text); the tool:

  1. Classifies scope (in-scope general-care, out-of-scope clinical
     decision).
  2. For out-of-scope questions -> refuses + redirects to the clinician
     ("Please call your nurse or doctor"). Refusal is the safe default.
  3. For in-scope questions -> grounded LLM answer with citations from the
     existing guideline corpus (ground_claim's `_retrieve_corpus_evidence`).
  4. When LLM unavailable -> deterministic template response with citations
     still attached.

Out-of-scope categories (refuse on detection):
  - dose changes ("should I take more / less / stop")
  - new medication requests
  - lab interpretation ("what does my INR of 4 mean")
  - prognosis ("how long do I have")
  - urgent-symptom decisions ("am I having a heart attack")
"""

from __future__ import annotations

import os
import re
from typing import Any

from shared.schemas import (
    DecisionCard,
    DischargeCounseling,
    EvidenceSource,
    PatientFAQAnswer,
)


# ─────────────────────── Scope classification ───────────────────────

_OUT_OF_SCOPE_PATTERNS: list[tuple[str, str]] = [
    # Dose-change requires a quantity modifier -- "should i take more / less /
    # another / extra / two / double" -- not a bare "should i take my pills".
    (r"should i (take|have|swallow) (more|less|another|extra|two|double|"
     r"a higher|a lower|fewer|additional)\b",
     "dose_change_request"),
    (r"should i (stop|skip|halve|cut|reduce|increase|double|triple)\b",
     "dose_change_request"),
    (r"\b(can i (stop|start|change))",
     "medication_change_request"),
    (r"what (does|is) my (inr|glucose|sodium|potassium|wbc|hgb|creatinine)\b",
     "lab_interpretation"),
    (r"\b(diagnose|diagnosis) (me|my)",
     "diagnosis_request"),
    (r"\b(prognosis|how long (do i have|will i live))",
     "prognosis_request"),
    (r"\b(am i (having|going to have) (a |an )?"
     r"(heart attack|stroke|seizure))",
     "acute_symptom_decision"),
    (r"\b(should i (go|come) to (the )?(er|emergency room))",
     "triage_decision"),
]

# In-scope patterns (positive matches) for the deterministic fallback to use
_IN_SCOPE_PATTERNS: list[tuple[str, str]] = [
    (r"\bwhy (am i|do i)\b", "purpose_inquiry"),
    (r"\bwhat (is|are) (this|these|the).*(for|do)\b", "purpose_inquiry"),
    (r"\bwhen (should|do) i (take|use)\b", "schedule_inquiry"),
    (r"\bwhat (happens|side effect)\b", "side_effect_inquiry"),
    (r"\b(can i|may i) (eat|drink|drive|exercise|fly|work)\b",
     "lifestyle_inquiry"),
    (r"\bwhen (should|do) i (call|see|visit)\b", "follow_up_inquiry"),
    (r"\b(what|which) (foods?|drinks?) (should|can|to) (avoid|eat)\b",
     "dietary_inquiry"),
]


def _classify_scope(question: str) -> tuple[bool, str | None, str]:
    """Return (in_scope, refusal_reason, category)."""
    q = (question or "").lower().strip()
    for pat, reason in _OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, q):
            return False, reason, reason
    for pat, category in _IN_SCOPE_PATTERNS:
        if re.search(pat, q):
            return True, None, category
    # Default: cautiously in-scope but flag as generic
    return True, None, "generic_inquiry"


# ─────────────────────── Refusal templates ───────────────────────

_REFUSAL_TEMPLATES: dict[str, str] = {
    "dose_change_request": (
        "I cannot help with changing your medication dose. Please call "
        "your prescribing doctor or pharmacist. They have your full "
        "history and can make a safe change for your specific situation."
    ),
    "medication_change_request": (
        "Adding, stopping, or changing medications is a clinical decision "
        "that requires your doctor's review. Please call them before "
        "making any change."
    ),
    "lab_interpretation": (
        "Lab results need to be interpreted by your care team in the "
        "context of your full chart. Please call your nurse or doctor "
        "with your specific value -- they can tell you what it means and "
        "whether any action is needed."
    ),
    "diagnosis_request": (
        "I cannot diagnose. Please contact your doctor; they can examine "
        "you, review your tests, and give you a clinical answer."
    ),
    "prognosis_request": (
        "Questions about how your condition will progress are best "
        "answered by your treating doctor. They know your full chart "
        "and can give you a realistic, individualized answer."
    ),
    "acute_symptom_decision": (
        "If you think you may be having a heart attack, stroke, or "
        "seizure, call 911 (or your local emergency number) RIGHT NOW. "
        "Do not wait for a chat answer."
    ),
    "triage_decision": (
        "Whether to go to the ER depends on your specific symptoms and "
        "your full medical history. Please call your doctor's office (or "
        "911 if symptoms are severe). I cannot make that decision for you."
    ),
}


# ─────────────────────── Deterministic in-scope template ───────────────────────

_IN_SCOPE_TEMPLATES: dict[str, str] = {
    "purpose_inquiry": (
        "Your discharge instructions explain why each medication and "
        "follow-up step matters. Look at the 'Your medications' and "
        "'Follow up' sections of your discharge summary; if anything "
        "is unclear, call your nurse or doctor."
    ),
    "schedule_inquiry": (
        "The exact schedule for each medication or follow-up is on your "
        "discharge paperwork -- see the 'Your medications' and "
        "'Follow up' sections. If the timing isn't clear, call your "
        "nurse for clarification."
    ),
    "side_effect_inquiry": (
        "Watch for the warning signs listed in the 'Warning signs' "
        "section of your discharge summary. If you notice any of them, "
        "follow the instructions there (most say to call your doctor or "
        "go to the ER)."
    ),
    "lifestyle_inquiry": (
        "Activity, diet, and other lifestyle questions are answered in "
        "the 'Activities and self-care' section of your discharge "
        "summary. If your specific situation isn't covered, ask your "
        "nurse before you act."
    ),
    "follow_up_inquiry": (
        "Your follow-up appointment timing is in the 'Follow up' section "
        "of your discharge summary. If you cannot reach that office, "
        "call your primary care doctor."
    ),
    "dietary_inquiry": (
        "Specific food/drink restrictions are listed in the 'Your "
        "medications' section (some medications interact with food). "
        "If unsure, ask your pharmacist."
    ),
    "generic_inquiry": (
        "I can answer general questions about your discharge plan, but "
        "specific medical decisions need your care team. Look at your "
        "discharge summary first; if your question isn't answered, call "
        "your nurse or doctor."
    ),
}


# ─────────────────────── LLM call ───────────────────────

_DEFAULT_MODEL = os.environ.get(
    "TRUSTEDRISK_FAQ_LLM_MODEL", "llama3.1:8b")


def _call_ollama(prompt: str) -> str | None:
    if os.environ.get("TRUSTEDRISK_DISABLE_LLM", "0") == "1":
        return None
    try:
        import ollama  # type: ignore
    except ImportError:
        return None
    try:
        resp = ollama.generate(
            model=_DEFAULT_MODEL, prompt=prompt,
            options={"temperature": 0.1})
        return str(resp.get("response", "")).strip()
    except Exception:
        return None


_LLM_PROMPT = """You are a patient-education assistant. The patient has \
asked a question. Answer using ONLY the structured discharge summary + \
the cited guideline excerpts below. Follow these rules strictly:

  1. Answer in plain 6th-grade English, ≤ 120 words.
  2. NEVER add new medications, doses, lab interpretations, or diagnoses.
  3. NEVER tell the patient to take more or less of any drug.
  4. If the answer requires information you don't have, redirect to the \
clinician -- say "I don't know -- please call your nurse or doctor."
  5. End with one short sentence reminding the patient that this does \
NOT replace their official discharge paper.

QUESTION: {question}

DISCHARGE SUMMARY EXCERPT (JSON):
{summary_json}

CITED GUIDELINE EXCERPTS:
{citations_text}

ANSWER:"""


# ─────────────────────── Citation retrieval ───────────────────────

def _retrieve_citations(question: str,
                          k: int = 2) -> list[EvidenceSource]:
    try:
        from .ground_claim import _retrieve_corpus_evidence
    except Exception:
        return []
    try:
        return _retrieve_corpus_evidence(question, top_k=k)
    except Exception:
        return []


# ─────────────────────── Tool body ───────────────────────

def _summarize_for_prompt(card: DecisionCard | dict | None,
                            counseling: DischargeCounseling | dict | None,
                            ) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(card, DecisionCard):
        card = card.model_dump(mode="json")
    if isinstance(counseling, DischargeCounseling):
        counseling = counseling.model_dump(mode="json")
    if isinstance(card, dict):
        rec = card.get("recommendation")
        if isinstance(rec, dict):
            out["recommendation_action"] = rec.get("action")
            out["recommendation_confidence"] = rec.get("confidence")
        reasoning = card.get("reasoning") or {}
        risk = reasoning.get("risk_estimate") or {}
        if "probability_mean" in risk:
            out["risk_30d_readmission_probability"] = risk["probability_mean"]
    if isinstance(counseling, dict):
        out["counseling_sections"] = [
            {"section_id": s["section_id"],
             "title": s["title"],
             "plain_text": s["plain_text"][:300],
             "bullets": list(s.get("bullets", []))[:6]}
            for s in counseling.get("sections", [])[:6]
        ]
    return out


async def compute_patient_faq(
    question: str | None = None,
    decision_card: DecisionCard | dict | None = None,
    counseling: DischargeCounseling | dict | None = None,
) -> PatientFAQAnswer:
    """Answer a patient FAQ grounded in the discharge plan + guideline corpus.

    Args:
        question: free-text patient question. The dispatcher does not
            inject a default; if the caller did not supply a clinician-
            or patient-authored question, the tool abstains rather than
            inventing one.
        decision_card: optional -- provides risk + recommendation context.
        counseling: optional -- provides the structured discharge sections.

    Returns:
        PatientFAQAnswer. Out-of-scope questions return in_scope=False
        with a refusal_reason + a redirect-to-clinician answer_text.
        Empty question returns abstain_recommended=True so a downstream
        orchestrator does not present an empty answer as if it were
        a real patient-education response.
    """
    if not isinstance(question, str) or not question.strip():
        return PatientFAQAnswer(
            question="",
            in_scope=False,
            answer_text="",
            citations=[],
            refusal_reason="empty_question",
            method="deterministic_template",
            abstain_recommended=True,
            abstain_reason=(
                "missing_question: compute_patient_faq requires a "
                "specific patient or clinician-authored question (e.g. "
                "'When can I drive again?', 'Why is my INR being "
                "checked weekly?'). The tool refuses to fabricate a "
                "question and answer pair from the discharge plan in "
                "the absence of an explicit input."
            ),
        )

    in_scope, refusal_reason, category = _classify_scope(question)

    if not in_scope and refusal_reason is not None:
        return PatientFAQAnswer(
            question=question,
            in_scope=False,
            answer_text=_REFUSAL_TEMPLATES[refusal_reason],
            citations=[],
            refusal_reason=refusal_reason,
            method="deterministic_template",
        )

    citations = _retrieve_citations(question, k=2)

    summary = _summarize_for_prompt(decision_card, counseling)
    citations_text = "\n".join(
        f"- [{c.source_id}] {c.excerpt[:200]}"
        for c in citations
    ) or "(no guideline excerpts available)"

    import json as _json
    prompt = _LLM_PROMPT.format(
        question=question,
        summary_json=_json.dumps(summary, indent=2, default=str),
        citations_text=citations_text,
    )
    raw = _call_ollama(prompt)

    safety_warnings: list[str] = []
    method = "llm"
    if raw is None or not raw.strip():
        method = "deterministic_template"
        answer = _IN_SCOPE_TEMPLATES.get(category,
                                              _IN_SCOPE_TEMPLATES["generic_inquiry"])
        safety_warnings.append("llm_unavailable_template_used")
    else:
        # Validation: refuse if the LLM tried to inject dose-change advice
        bad_phrases = ("take more", "take less", "double the dose",
                          "stop taking", "skip the dose", "increase your")
        low = raw.lower()
        if any(b in low for b in bad_phrases):
            method = "deterministic_template"
            answer = _IN_SCOPE_TEMPLATES.get(
                category, _IN_SCOPE_TEMPLATES["generic_inquiry"])
            safety_warnings.append("llm_dose_change_phrase_blocked")
        else:
            answer = raw[:1500]

    return PatientFAQAnswer(
        question=question,
        in_scope=True,
        answer_text=answer,
        citations=citations,
        refusal_reason=None,
        method=method,                  # type: ignore[arg-type]
        safety_warnings=safety_warnings,
    )


def register(mcp) -> None:
    mcp.tool()(compute_patient_faq)
