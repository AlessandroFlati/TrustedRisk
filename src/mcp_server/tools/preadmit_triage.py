"""healthcare.compute_symptom_red_flag_check / when_to_seek_care /
symptom_followup_questions -- Phase 7.5 pre-arrival triage agent.

Patient-facing intake. Pure-deterministic regex-based red-flag
detection over plain-language symptom descriptions, plus a curated
mapping into care-level recommendations (self-care / telehealth / PCP /
urgent care / ED / 911) anchored on the AHRQ symptom-triage decision
trees + ACEP red-flag list.

Each tool runs deterministically and surfaces a clear care-level
verdict with rationale. The patient-facing rendering layer
(`patient_facing` bundle) is the natural caller; the
`pre-admit triage` specialist on port 8781 wraps these tools.

Sensitive area -- every "go to ER" / "stay home" boundary errs on the
side of escalation when the input is ambiguous. The follow-up-
questions tool reduces uncertainty before the care-level decision.

Sources:
- AHRQ symptom-triage protocols (Pediatric + Adult)
- ACEP "When to go to the ED" red-flag checklist
- Mayo Clinic First-Aid + Symptom Checker mapping
"""

from __future__ import annotations

import re

from shared.schemas import (
    FollowupQuestion,
    FollowupQuestionsReport,
    SymptomRedFlag,
    SymptomRedFlagReport,
    WhenToSeekCareReport,
)


# ─────────────────────────────────────────────────────────────────────
# Red-flag dictionary -- (regex, label, severity, rationale)
# ─────────────────────────────────────────────────────────────────────

_RED_FLAGS: list[dict[str, str]] = [
    # Critical -- call 911
    {
        "flag_id": "chest_pain_with_radiation",
        "regex": (r"chest pain.*(arm|jaw|back|neck)|"
                     r"(arm|jaw|back|neck).*chest pain"),
        "label": "chest pain with radiation to arm/jaw",
        "severity": "critical",
        "rationale": "Possible acute coronary syndrome (ACS).",
    },
    {
        "flag_id": "stroke_focal_deficit",
        "regex": (r"(slurr|trouble) speak|"
                     r"(facial|face) droop|"
                     r"weakness on one side|hemiparesis|"
                     r"sudden vision loss|sudden numbness on one side"),
        "label": "focal neurologic deficit (stroke red flag)",
        "severity": "critical",
        "rationale": "Possible acute stroke -- time-critical.",
    },
    {
        "flag_id": "thunderclap_headache",
        "regex": (r"thunderclap|"
                     r"worst headache (of my )?life|"
                     r"sudden severe headache"),
        "label": "thunderclap headache",
        "severity": "critical",
        "rationale": "Possible subarachnoid haemorrhage.",
    },
    {
        "flag_id": "respiratory_distress",
        "regex": (r"can't (breathe|breath)|"
                     r"struggling to breathe|cannot catch (my )?breath|"
                     r"blue (lips|fingers)|cyanosis"),
        "label": "respiratory distress / cyanosis",
        "severity": "critical",
        "rationale": "Imminent respiratory failure risk.",
    },
    {
        "flag_id": "anaphylaxis",
        "regex": (r"throat swell|"
                     r"throat closing|"
                     r"can't swallow|hives all over"),
        "label": "anaphylaxis red flag",
        "severity": "critical",
        "rationale": "Possible anaphylaxis -- time-critical.",
    },
    {
        "flag_id": "uncontrolled_bleeding",
        "regex": r"won'?t stop bleeding|severe bleeding|hemorrhage",
        "label": "uncontrolled bleeding",
        "severity": "critical",
        "rationale": "Hemorrhagic emergency.",
    },
    {
        "flag_id": "loss_of_consciousness",
        "regex": (r"passed out|fainted|unconscious|"
                     r"loss of consciousness"),
        "label": "loss of consciousness",
        "severity": "high",
        "rationale": "Syncope -- workup required, ED if recent + unexplained.",
    },
    # High -- go to ED today
    {
        "flag_id": "fever_immunocompromised",
        "regex": (r"fever.*chemo|chemo.*fever|"
                     r"fever.*immunocompromised"),
        "label": "fever in immunocompromised patient",
        "severity": "high",
        "rationale": "Neutropenic fever -- urgent.",
    },
    {
        "flag_id": "severe_abdominal_pain",
        "regex": (r"severe (abdom|stomach|belly) pain|"
                     r"can't move from belly pain"),
        "label": "severe abdominal pain",
        "severity": "high",
        "rationale": "Acute abdomen -- surgical workup.",
    },
    {
        "flag_id": "high_fever_baby",
        "regex": (r"baby.*fever (over |of )?(38\.5|39|40|41)|"
                     r"infant.*fever (over |of )?(38\.5|39|40|41)"),
        "label": "infant high fever",
        "severity": "high",
        "rationale": "Sepsis / serious bacterial infection risk in infants.",
    },
    {
        "flag_id": "suicidal_intent",
        "regex": (r"want to (die|kill myself|end (it|my life))|"
                     r"plan to (kill|hurt) myself"),
        "label": "active suicidal intent",
        "severity": "critical",
        "rationale": "Mental-health emergency.",
    },
    # Medium -- urgent care or PCP today
    {
        "flag_id": "pregnancy_bleeding",
        "regex": r"bleeding.*pregnan|pregnan.*bleeding",
        "label": "bleeding in pregnancy",
        "severity": "high",
        "rationale": "OB urgent evaluation.",
    },
    {
        "flag_id": "fever_persistent",
        "regex": (r"fever (for|over) (3|4|5|6|7) days|"
                     r"fever won'?t go away"),
        "label": "persistent fever ≥ 3 days",
        "severity": "medium",
        "rationale": "Persistent fever needs evaluation.",
    },
    {
        "flag_id": "rash_with_fever",
        "regex": r"rash.*fever|fever.*rash",
        "label": "rash with fever",
        "severity": "medium",
        "rationale": "Could be infectious; needs assessment.",
    },
    {
        "flag_id": "pain_score_high",
        "regex": r"pain.*(8|9|10).{0,10}(out of 10|/10)",
        "label": "self-reported pain ≥ 8/10",
        "severity": "medium",
        "rationale": "Severe pain -- clinical evaluation indicated.",
    },
]


_SEVERITY_RANK = {
    "none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4,
}


def _highest(severities: list[str]) -> str:
    if not severities:
        return "none"
    return max(severities, key=lambda s: _SEVERITY_RANK.get(s, 0))


# ─────────────────────────────────────────────────────────────────────
# 1. Red-flag check
# ─────────────────────────────────────────────────────────────────────


async def compute_symptom_red_flag_check(
    raw_input: str,
) -> SymptomRedFlagReport:
    """Detect red flags in patient-language symptom text.

    Returns the matched flags, the highest severity, and a coarse
    care-level recommendation tag.
    """
    text = (raw_input or "").lower()
    flags: list[SymptomRedFlag] = []
    for entry in _RED_FLAGS:
        m = re.search(entry["regex"], text, re.IGNORECASE)
        if not m:
            continue
        excerpt = raw_input[max(0, m.start() - 10):
                                min(len(raw_input), m.end() + 10)]
        flags.append(SymptomRedFlag(
            flag_id=entry["flag_id"],
            label=entry["label"],
            severity=entry["severity"],                 # type: ignore[arg-type]
            matched_text=excerpt.strip(),
            rationale=entry["rationale"],
        ))

    severities = [f.severity for f in flags]
    highest = _highest(severities)

    if highest == "critical":
        rec = "call_911"
    elif highest == "high":
        rec = "go_to_ed"
    elif highest == "medium":
        rec = "seek_urgent_care"
    elif highest == "low":
        rec = "seek_care_today"
    elif flags:
        rec = "monitor"
    else:
        rec = "no_red_flag"

    return SymptomRedFlagReport(
        raw_input=raw_input or "",
        flags=flags,
        n_flags=len(flags),
        highest_severity=highest,                          # type: ignore[arg-type]
        recommendation=rec,                                # type: ignore[arg-type]
        references=[
            "ACEP -- When to Go to the ED checklist.",
            "AHRQ Symptom Triage Protocols.",
            "Mayo Clinic First-Aid Symptom Checker.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 2. When to seek care
# ─────────────────────────────────────────────────────────────────────


_LEVEL_TIMELINE = {
    "self_care": "manage at home; reassess if worse over 24-48 h",
    "telehealth": "video / phone visit within 24 h",
    "primary_care_within_24h": "primary care within 24 h",
    "primary_care_within_7d": "primary care within 7 days",
    "urgent_care": "urgent care today",
    "ed_now": "emergency department now",
    "call_911": "call 911 immediately",
}


_GENERIC_WATCH = [
    "Worsening symptoms",
    "Difficulty breathing",
    "Loss of consciousness",
    "Severe / uncontrolled pain",
]


_GENERIC_AVOID = [
    "Driving yourself to the ED if you are short of breath or "
    "having chest pain",
    "Ignoring symptoms hoping they go away if they are worsening",
]


async def compute_when_to_seek_care(
    raw_input: str,
    duration_hours: int | None = None,
    severity_1_to_10: int | None = None,
) -> WhenToSeekCareReport:
    """Combine red-flag detection + duration + self-reported severity
    into a single care-level recommendation."""
    rf = await compute_symptom_red_flag_check(raw_input)

    if rf.recommendation == "call_911":
        level = "call_911"
        rationale = (
            f"Critical red flag detected: "
            + ", ".join(f.label for f in rf.flags
                              if f.severity == "critical")
            + ". Call 911 immediately."
        )
    elif rf.recommendation == "go_to_ed":
        level = "ed_now"
        rationale = (
            "High-severity red flag(s) detected. Go to the ED now."
        )
    elif rf.recommendation == "seek_urgent_care":
        level = "urgent_care"
        rationale = "Medium-severity red flag(s) detected."
    elif severity_1_to_10 and severity_1_to_10 >= 8:
        level = "urgent_care"
        rationale = (
            f"Self-reported severity {severity_1_to_10}/10 -- urgent "
            "care today."
        )
    elif duration_hours is not None and duration_hours >= 96 \
            and severity_1_to_10 is not None and severity_1_to_10 >= 5:
        level = "primary_care_within_24h"
        rationale = (
            f"Symptoms ≥ {duration_hours} h with moderate severity -- "
            "PCP within 24 h."
        )
    elif duration_hours is not None and duration_hours >= 168:
        level = "primary_care_within_7d"
        rationale = (
            f"Symptoms persisting > 7 days -- schedule PCP within "
            "the next week."
        )
    elif severity_1_to_10 is not None and severity_1_to_10 >= 4:
        level = "telehealth"
        rationale = (
            f"Moderate severity {severity_1_to_10}/10 -- telehealth "
            "consult is appropriate."
        )
    else:
        level = "self_care"
        rationale = (
            "No red flags; mild severity. Self-care + monitoring."
        )

    return WhenToSeekCareReport(
        recommended_level=level,                           # type: ignore[arg-type]
        rationale=rationale,
        timeline=_LEVEL_TIMELINE[level],
        things_to_avoid=list(_GENERIC_AVOID),
        things_to_watch_for=list(_GENERIC_WATCH),
        abstain_recommended=(not raw_input or not raw_input.strip()),
        abstain_reason=(
            "no_symptom_input"
            if (not raw_input or not raw_input.strip()) else None
        ),
        references=[
            "ACEP -- When to Go to the ED checklist.",
            "AHRQ Symptom Triage Protocols.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 3. Adaptive follow-up questions
# ─────────────────────────────────────────────────────────────────────


def _looks_like_chest_pain(t: str) -> bool:
    return bool(re.search(r"chest|heart", t, re.IGNORECASE))


def _looks_like_abdominal(t: str) -> bool:
    return bool(re.search(r"belly|abdom|stomach|gut", t, re.IGNORECASE))


def _looks_like_headache(t: str) -> bool:
    return bool(re.search(r"head ache|headache|migraine", t, re.IGNORECASE))


def _looks_like_respiratory(t: str) -> bool:
    return bool(re.search(r"breath|cough|wheez|short of breath",
                                t, re.IGNORECASE))


_GENERIC_QUESTIONS = [
    FollowupQuestion(
        question_id="duration",
        question="How long have you had this symptom?",
        expected_answer_kind="duration",
        rationale="Acute (<48 h) vs subacute (>1 week) changes triage.",
    ),
    FollowupQuestion(
        question_id="severity",
        question="On a scale of 1 to 10, how severe is the pain or discomfort right now?",
        expected_answer_kind="severity_1_to_10",
        rationale="Self-reported severity drives the care-level escalation.",
    ),
    FollowupQuestion(
        question_id="worse_with_movement",
        question="Does the symptom get worse with movement or activity?",
        expected_answer_kind="yes_no",
        rationale=(
            "Mechanical / inflammatory vs visceral discrimination."
        ),
    ),
]


async def compute_symptom_followup_questions(
    raw_input: str,
    max_questions: int = 5,
) -> FollowupQuestionsReport:
    """Adaptive follow-up question selector. The base set is 3 generic
    questions; symptom-domain triggers add 1-2 targeted questions."""
    text = (raw_input or "").strip()
    questions = list(_GENERIC_QUESTIONS)

    if _looks_like_chest_pain(text):
        questions.append(FollowupQuestion(
            question_id="chest_radiation",
            question=(
                "Does the chest pain spread to your arm, jaw, neck, or back?"
            ),
            expected_answer_kind="yes_no",
            rationale="Radiation pattern is a key ACS / MI signal.",
        ))
        questions.append(FollowupQuestion(
            question_id="chest_diaphoresis",
            question="Are you sweating with the chest pain?",
            expected_answer_kind="yes_no",
            rationale="Diaphoresis with chest pain raises ACS suspicion.",
        ))

    if _looks_like_abdominal(text):
        questions.append(FollowupQuestion(
            question_id="abd_location",
            question=(
                "Where exactly does it hurt -- upper right, upper left, "
                "lower right, lower left, or all over?"
            ),
            expected_answer_kind="free_text",
            rationale=(
                "Location maps to organs (RUQ -> biliary, RLQ -> "
                "appendicitis, etc.)."
            ),
        ))

    if _looks_like_headache(text):
        questions.append(FollowupQuestion(
            question_id="hd_thunderclap",
            question=(
                "Did the headache hit you suddenly like a thunderclap, "
                "or build up gradually?"
            ),
            expected_answer_kind="yes_no",
            rationale="Thunderclap onset -> possible subarachnoid haemorrhage.",
        ))

    if _looks_like_respiratory(text):
        questions.append(FollowupQuestion(
            question_id="resp_rest",
            question=(
                "Are you short of breath at rest, or only with activity?"
            ),
            expected_answer_kind="yes_no",
            rationale="Rest dyspnoea is more concerning than exertional.",
        ))

    questions = questions[:max_questions]

    return FollowupQuestionsReport(
        raw_input=text,
        questions=questions,
        n_questions=len(questions),
        references=[
            "AHRQ Symptom Triage Protocols.",
            "Mayo Clinic Symptom Checker.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_symptom_red_flag_check)
    mcp.tool()(compute_when_to_seek_care)
    mcp.tool()(compute_symptom_followup_questions)
