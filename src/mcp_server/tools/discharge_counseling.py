"""healthcare.compute_discharge_counseling -- patient-language discharge instructions.

Why this tool exists
====================
Discharge counseling failures are a top-3 cause of preventable 30-day
readmissions (AHRQ 2018; Kripalani et al. NEJM 2014). Patients leave the
hospital with a paper packet they can't read at a 6th-grade level, no clear
sense of which medications are new vs continued, and no list of symptoms
that should send them back to the ER. This tool generates a structured,
patient-language summary that complements the official discharge paperwork.

Design decisions
================
1. **Deterministic template assembly first.** Every statement binds to a
   structured input (drug class, LACE score, recommendation). Hallucinated
   dosages or invented red-flags at the discharge handoff are a safety
   hazard, not a UX paper-cut.
2. **Optional LLM polish.** When TRUSTEDRISK_COUNSELING_LLM_POLISH=1, a
   thin wrapper paraphrases the template output without changing facts.
   This is opt-in and disabled in tests.
3. **Drug-class mapping reused** from medication_reconciliation. If the
   class taxonomy changes, both tools update together.
4. **Reading level**: short sentences, common words, no jargon without a
   parenthetical translation. Targeted at ~6th grade Flesch-Kincaid.
5. **No PHI**: the output never mentions the patient's name, dates, or
   provider names. Only medical content.
"""

from __future__ import annotations

import os
import re
from typing import Any

from shared.schemas import (
    CounselingSection,
    DischargeCounseling,
    Medication,
)

from ._chart_inputs import harden_clinical_inputs
from .medication_reconciliation import _classify_drug


# ─────────────────────────────────────────────────────────────────────
# Drug class -> patient-language explanation + red flags
# ─────────────────────────────────────────────────────────────────────

_CLASS_PATIENT_LANGUAGE: dict[str, dict[str, Any]] = {
    "anticoagulant_vka": {
        "label": "blood thinner (warfarin)",
        "purpose": "to prevent dangerous blood clots.",
        "monitoring": "You will need a blood test (called INR) about every 1 to 4 weeks.",
        "red_flags": [
            "Unusual bruising or bleeding that won't stop.",
            "Pink, red, or dark brown urine.",
            "Black or bloody stools.",
            "Severe headache or sudden confusion.",
        ],
    },
    "anticoagulant_doac": {
        "label": "blood thinner (newer type)",
        "purpose": "to prevent dangerous blood clots.",
        "monitoring": "Your kidney function may be checked at follow-up.",
        "red_flags": [
            "Unusual bruising or bleeding that won't stop.",
            "Pink, red, or dark brown urine.",
            "Black or bloody stools.",
            "Severe headache.",
        ],
    },
    "anticoagulant_heparin": {
        "label": "blood thinner injection",
        "purpose": "to prevent dangerous blood clots short-term.",
        "monitoring": "Watch the injection sites. Bruising near the site is normal.",
        "red_flags": [
            "Bleeding that won't stop.",
            "Severe pain or color change at injection sites.",
        ],
    },
    "insulin": {
        "label": "insulin",
        "purpose": "to control your blood sugar.",
        "monitoring": "Check your blood sugar before meals as instructed.",
        "red_flags": [
            "Shakiness, sweating, or sudden confusion (low blood sugar) -- eat or drink something sweet right away.",
            "Very high blood sugar above 300, or feeling very thirsty and tired.",
        ],
    },
    "biguanide": {
        "label": "diabetes pill (metformin)",
        "purpose": "to control your blood sugar.",
        "monitoring": "Take with food to reduce stomach upset.",
        "red_flags": [
            "Severe muscle pain or weakness, slow heartbeat, or feeling cold (rare but serious).",
        ],
    },
    "sglt2_inhibitor": {
        "label": "diabetes pill (SGLT2 inhibitor)",
        "purpose": "to control your blood sugar and protect your heart and kidneys.",
        "monitoring": "Drink enough water. Stop the pill if you become very sick or dehydrated and call your doctor.",
        "red_flags": [
            "Stomach pain, nausea, or unusual tiredness even if your blood sugar looks normal.",
            "Genital pain, swelling, or unusual discharge.",
        ],
    },
    "ace_inhibitor": {
        "label": "blood pressure medicine (ACE inhibitor)",
        "purpose": "to lower your blood pressure and protect your heart and kidneys.",
        "monitoring": "Your potassium and kidney blood test will be checked at follow-up.",
        "red_flags": [
            "Swelling of the face, lips, or tongue -- go to the ER right away.",
            "A persistent dry cough that does not go away.",
            "Lightheadedness when standing up.",
        ],
    },
    "arb": {
        "label": "blood pressure medicine (ARB)",
        "purpose": "to lower your blood pressure and protect your heart and kidneys.",
        "monitoring": "Your potassium and kidney blood test will be checked at follow-up.",
        "red_flags": [
            "Swelling of the face, lips, or tongue -- go to the ER right away.",
            "Lightheadedness when standing up.",
        ],
    },
    "beta_blocker": {
        "label": "heart rate medicine (beta blocker)",
        "purpose": "to slow your heart rate and lower your blood pressure.",
        "monitoring": "Check your pulse. If it is slower than 50 beats per minute, call your doctor before taking the next dose.",
        "red_flags": [
            "Pulse below 50, fainting, or severe tiredness.",
            "Trouble breathing or new wheezing.",
        ],
    },
    "mra": {
        "label": "water pill (potassium-sparing)",
        "purpose": "to remove extra fluid and protect your heart.",
        "monitoring": "Your potassium will be checked at follow-up. Avoid salt substitutes that contain potassium.",
        "red_flags": [
            "Muscle weakness or an irregular heartbeat (high potassium).",
        ],
    },
    "loop_diuretic": {
        "label": "water pill (diuretic)",
        "purpose": "to remove extra fluid from your body.",
        "monitoring": "Weigh yourself every morning. Call your doctor if you gain more than 2 pounds (about 1 kg) in 1 day or 5 pounds in 1 week.",
        "red_flags": [
            "Severe thirst or dry mouth, very little urine, or leg cramps.",
            "Sudden weight gain or worsening swelling.",
        ],
    },
    "immunosuppressant": {
        "label": "immune-suppressing medicine",
        "purpose": "to keep your transplanted organ working or to control an autoimmune condition.",
        "monitoring": "Drug level blood tests are required regularly -- do NOT miss your follow-up appointment.",
        "red_flags": [
            "Fever, chills, or any sign of infection.",
            "Mouth sores or unusual tiredness.",
        ],
    },
    "opioid": {
        "label": "strong pain medicine (opioid)",
        "purpose": "to control severe pain -- use the smallest amount that works.",
        "monitoring": "Take only as prescribed. Do not drink alcohol or take sleeping pills with this.",
        "red_flags": [
            "Very slow breathing, severe drowsiness, or confusion -- call 911.",
            "If naloxone (Narcan) was prescribed, make sure a family member knows where it is and how to use it.",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────
# Follow-up timing -- scaled by LACE score
# ─────────────────────────────────────────────────────────────────────
#
# Higher LACE -> earlier follow-up. Windows are conservative and align with
# the readmission prevention literature: a 7-day post-discharge phone call +
# 14-day in-person visit halves 30-day readmission for high-risk discharges
# (AHRQ Re-Engineered Discharge / Project RED).

_LACE_TO_FOLLOWUP: list[tuple[int, tuple[int, int], str]] = [
    (10, (3, 7), "high-risk patients should see their doctor within a week"),
    (5, (7, 14), "moderate-risk patients should see their doctor within 1 to 2 weeks"),
    (0, (14, 30), "lower-risk patients should see their doctor within 2 to 4 weeks"),
]


def _follow_up_window(lace_score: int) -> tuple[tuple[int, int], str]:
    for threshold, window, rationale in _LACE_TO_FOLLOWUP:
        if lace_score >= threshold:
            return window, rationale
    # Defensive default (should never hit because last threshold is 0)
    return (14, 30), "follow-up within 2 to 4 weeks"


# ─────────────────────────────────────────────────────────────────────
# Section assembly
# ─────────────────────────────────────────────────────────────────────

def _section_medications(meds: list[Medication]) -> tuple[CounselingSection, int]:
    """Generate 'Your Medications' section. Returns (section, n_explained)."""
    bullets: list[str] = []
    n_explained = 0
    seen_classes: set[str] = set()
    for med in meds:
        if med.status != "active":
            continue
        cls = med.drug_class or _classify_drug(med.name)
        info = _CLASS_PATIENT_LANGUAGE.get(cls or "")
        if info is not None:
            seen_classes.add(cls)
            n_explained += 1
            bullet = (
                f"**{med.name}** is a {info['label']}. "
                f"It is {info['purpose']} {info['monitoring']}"
            )
            bullets.append(bullet)
        else:
            bullets.append(
                f"**{med.name}** -- keep taking this exactly as your discharge "
                f"paperwork describes. Ask your pharmacist if you are unsure why."
            )

    if not bullets:
        bullets.append("No active medications were listed in your discharge plan.")

    plain = (
        "Below is a plain-language summary of each medicine on your discharge list. "
        "Take each medicine exactly as written on the prescription label. "
        "If a medicine on this list is NOT on your label, ask your pharmacist before taking it."
    )
    return CounselingSection(
        section_id="your_medications",
        title="Your Medications",
        plain_text=plain,
        bullets=bullets,
    ), n_explained


def _section_follow_up(
    lace_score: int, recommendation_action: str | None,
) -> tuple[CounselingSection, tuple[int, int]]:
    window, rationale = _follow_up_window(lace_score)
    lo, hi = window
    bullets = [
        f"Make a follow-up appointment with your primary doctor in {lo} to {hi} days.",
        "Bring your medication list (or this summary) to that visit.",
        "Bring a list of any new symptoms or questions.",
    ]
    if recommendation_action == "snf":
        bullets.append(
            "Your team recommended a short stay at a skilled nursing facility "
            "for physical therapy and monitoring before going home."
        )
    elif recommendation_action == "home_with_care":
        bullets.append(
            "A home health nurse will visit you. Keep their phone number "
            "where you can find it easily."
        )
    elif recommendation_action == "continued_admission":
        bullets.append(
            "Your team recommended that you stay in the hospital longer. "
            "Discuss this with your doctor before you leave."
        )

    plain = (
        f"Going to your follow-up matters. Skipping it is one of the most common "
        f"reasons people end up back in the hospital. Based on your discharge plan, "
        f"{rationale}."
    )
    return CounselingSection(
        section_id="follow_up",
        title="Your Follow-up Appointments",
        plain_text=plain,
        bullets=bullets,
    ), window


def _section_warning_signs(
    meds: list[Medication], extra_red_flags: list[str] | None = None,
) -> tuple[CounselingSection, int]:
    seen_classes: set[str] = set()
    bullets: list[str] = []
    for med in meds:
        if med.status != "active":
            continue
        cls = med.drug_class or _classify_drug(med.name)
        if not cls or cls in seen_classes:
            continue
        info = _CLASS_PATIENT_LANGUAGE.get(cls)
        if info is None:
            continue
        seen_classes.add(cls)
        for flag in info["red_flags"]:
            bullets.append(flag)

    # Always include the universal red flags
    universal = [
        "Chest pain, pressure, or trouble breathing -- call 911.",
        "Sudden weakness, trouble speaking, or facial drooping -- call 911.",
        "A fever above 101°F (38.3°C) that does not go down.",
    ]
    for flag in universal:
        if flag not in bullets:
            bullets.append(flag)

    if extra_red_flags:
        for flag in extra_red_flags:
            if flag not in bullets:
                bullets.append(flag)

    plain = (
        "These are signs that need fast attention. If you have any of them, "
        "act on the instructions next to each one -- call 911, go to the ER, "
        "or call your doctor as written."
    )
    return CounselingSection(
        section_id="warning_signs",
        title="Warning Signs -- When to Call for Help",
        plain_text=plain,
        bullets=bullets,
    ), len(bullets)


def _section_activities() -> CounselingSection:
    bullets = [
        "Get up and walk a little every hour while you are awake. Short walks help your blood flow and your lungs.",
        "Drink water during the day unless your doctor told you to limit fluids.",
        "Eat balanced meals. Avoid extra salt if you have heart or blood pressure problems.",
        "Sleep at least 7 to 8 hours when you can. Rest helps you heal.",
        "Do NOT drive until your doctor says it is safe -- especially if you are taking pain medicine.",
    ]
    plain = (
        "These are general activity tips while you recover. Your discharge "
        "paperwork may have more specific limits -- follow those if they differ."
    )
    return CounselingSection(
        section_id="activities_self_care",
        title="Daily Activities and Self-Care",
        plain_text=plain,
        bullets=bullets,
    )


def _section_questions() -> CounselingSection:
    bullets = [
        "Which of my medicines are new, and which ones did I already take before?",
        "Are there any medicines I was taking before that I should now STOP?",
        "What signs mean I should come back to the ER?",
        "When is my next follow-up appointment, and what should I bring?",
        "Is there anything I should NOT eat or drink with these medicines?",
        "What number do I call if I have a question after I leave?",
    ]
    plain = (
        "Bring this list of questions to your next appointment, or ask them "
        "before you leave the hospital. There are no silly questions."
    )
    return CounselingSection(
        section_id="questions_to_ask",
        title="Questions to Ask Your Doctor",
        plain_text=plain,
        bullets=bullets,
    )


# ─────────────────────────────────────────────────────────────────────
# Optional LLM polish (paraphrase only, no fact creation)
# ─────────────────────────────────────────────────────────────────────

def _llm_polish_enabled() -> bool:
    return os.environ.get("TRUSTEDRISK_COUNSELING_LLM_POLISH", "").strip() in (
        "1", "true", "yes", "on",
    )


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

def _coerce_meds(meds: list[Any] | None) -> list[Medication]:
    if not meds:
        return []
    out: list[Medication] = []
    for m in meds:
        if isinstance(m, Medication):
            out.append(m)
        elif isinstance(m, dict):
            try:
                out.append(Medication.model_validate(m))
            except Exception:
                continue
        else:
            continue
    return out


async def compute_discharge_counseling(
    medications: list[dict[str, Any]] | None = None,
    lace_score: int = 0,
    recommendation_action: str | None = None,
    patient_id: str | None = None,
    extra_red_flags: list[str] | None = None,
    locale: str = "en",
) -> DischargeCounseling:
    """Generate a patient-language discharge summary.

    Args:
        medications: list of Medication dicts (or Pydantic instances). Only
            `status="active"` meds are included in the medication section.
        lace_score: the patient's LACE score, used to scale follow-up timing.
        recommendation_action: one of the Action enum values, or None. Used
            to add an action-specific bullet to the follow-up section.
        patient_id: for audit only -- never appears in the output text.
        extra_red_flags: caller-provided red flag strings (e.g. surgery-site
            specific symptoms). Appended after the drug-class red flags.
        locale: two-letter language code. Currently only "en" is implemented.

    Returns:
        DischargeCounseling with 5 sections + metadata.
    """
    if locale != "en":
        # Defensive fallback: never fail open in a different language.
        locale = "en"

    _r, _sb, _ = await harden_clinical_inputs(
        {"medications": medications},
        chart_derivable={"medications"},
    )
    if _sb and _r.get("medications"):
        # Chart returns flat list of name strings; counseling tool
        # accepts either list[str] or list[dict] / Medication shapes.
        medications = [{"name": n, "status": "active"} for n in _r["medications"]] \
            if all(isinstance(m, str) for m in _r["medications"]) else _r["medications"]

    # noqa: ABSTAIN-GUARD -- fail-fast when the medication list is None (not provided).
    # A None medications argument means the caller did not supply the discharge
    # med list at all.  Without any med list the drug-specific counseling and
    # red-flag warnings cannot be generated; the output would be a generic
    # template the caller may mistake for a personalised counseling document.
    # Note: an explicit empty list (medications=[]) is a valid clinical state
    # (patient discharged on no medications) and produces the standard 5-section
    # output with a "No active medications" note in the medications section.
    if medications is None:
        return DischargeCounseling(
            patient_id=patient_id,
            locale=locale,
            reading_level_grade=6,
            sections=[],
            follow_up_window_days=(14, 30),
            n_medications_explained=0,
            n_red_flags=0,
            abstain_recommended=True,
            abstain_reason=(
                "missing_medications: compute_discharge_counseling requires "
                "a non-empty medication list to generate drug-specific "
                "counseling and red-flag warnings. "
                "No demo fallback in live mode."
            ),
        )

    meds = _coerce_meds(medications)

    sec_meds, n_meds = _section_medications(meds)
    sec_follow, window = _section_follow_up(
        max(0, min(19, int(lace_score))), recommendation_action,
    )
    sec_warn, n_red = _section_warning_signs(meds, extra_red_flags=extra_red_flags)
    sec_activities = _section_activities()
    sec_questions = _section_questions()

    sections = [sec_meds, sec_follow, sec_warn, sec_activities, sec_questions]

    if _llm_polish_enabled():
        try:
            sections = await _polish_with_llm(sections)
        except Exception:
            # Polish is best-effort -- fall back to the deterministic output.
            pass

    return DischargeCounseling(
        patient_id=patient_id,
        locale=locale,
        reading_level_grade=6,
        sections=sections,
        follow_up_window_days=window,
        n_medications_explained=n_meds,
        n_red_flags=n_red,
    )


_POLISH_SYSTEM_PROMPT = (
    "You are a clinical-documentation editor. Rewrite the following "
    "discharge instructions for a patient reading at a 6th-grade level. "
    "Keep your output under 60 words. Do NOT add new medical advice, "
    "dosages, drug names, or symptoms. Preserve EVERY drug name, dose, "
    "ICD-10 code, and cite-back exactly as written. Output the polished "
    "prose only -- no headers, no commentary."
)


async def _polish_with_llm(
    sections: list[CounselingSection],
) -> list[CounselingSection]:
    """Paraphrase each section's plain_text through the audited polish
    client (Anthropic > Ollama > Gemini > null). Bullets unchanged.

    The shared client enforces a cite-back / preserved-token post-check:
    any polish output that drops a drug name, dose, ICD-10 code or
    cite-back is REJECTED and the deterministic floor is kept.
    """
    try:
        from a2a_agent.llm_polish import resolve_polish_client
    except Exception:
        return sections
    client = resolve_polish_client()
    if client.model_id is None:
        return sections

    polished: list[CounselingSection] = []
    for sec in sections:
        try:
            res = await client.polish(
                sec.plain_text,
                system_prompt=_POLISH_SYSTEM_PROMPT,
            )
        except Exception:
            polished.append(sec)
            continue
        if res.is_polished and res.polished_text.strip():
            new_text = re.sub(r"\s+", " ", res.polished_text).strip()
            polished.append(CounselingSection(
                section_id=sec.section_id, title=sec.title,
                plain_text=new_text, bullets=sec.bullets,
            ))
        else:
            polished.append(sec)
    return polished


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_discharge_counseling)
