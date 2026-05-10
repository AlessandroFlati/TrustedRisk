"""Phase 17.Z - Stigma + reading-level linter.

Two pure-Python tools that audit any patient-facing text emitted by
TrustedRisk:

  1. **Flesch-Kincaid grade level** (Kincaid 1975) - higher = harder.
     AHRQ's Re-engineered Discharge (RED) toolkit recommends a
     6th-8th grade reading level for discharge counseling.

  2. **Stigma language flagger** - flags the AHRQ "Mental Health and
     Addiction Equity"-derived stigmatising-language patterns
     ("diabetic patient" -> "patient with diabetes", "addict" ->
     "patient with substance use disorder", "non-compliant" ->
     "did not adhere to the regimen", "frequent flyer", etc.). Emits
     a per-finding suggested rewrite.

Pure-stdlib, deterministic. The rules are deliberately conservative -
designed to *flag* rather than auto-rewrite (the rewrite step belongs
to a clinician + LLM polish layer, not the deterministic floor).

References:
- Kincaid 1975 - readability formulas (DOD).
- AHRQ Re-engineered Discharge Toolkit, Tool 4.
- Substance Abuse and Mental Health Services Administration (SAMHSA)
  guideline on non-stigmatising language (2017).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# Flesch-Kincaid grade level
# ─────────────────────────────────────────────────────────────────────


_VOWELS = set("aeiouy")


def _count_syllables(word: str) -> int:
    """Heuristic English syllable counter. Accurate enough for FKGL."""
    word = word.lower().strip(" .,;:!?\"'()[]{}")
    if not word:
        return 0
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in _VOWELS:
        count += 1
    return max(1, count)


_SENT_RE = re.compile(r"[.!?]+(?:\s+|$)")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


class ReadabilityReport(BaseModel):
    n_words: int = Field(ge=0)
    n_sentences: int = Field(ge=0)
    n_syllables: int = Field(ge=0)
    flesch_kincaid_grade: float
    flesch_reading_ease: float
    target_grade_max: int
    meets_red_toolkit_target: bool
    rationale: str


def assess_readability(
    text: str, *,
    target_grade_max: int = 8,
) -> ReadabilityReport:
    """Compute Flesch-Kincaid grade level + Flesch reading ease.

    Returns a report with `meets_red_toolkit_target=True` when the
    grade is at or below ``target_grade_max`` (default 8 per AHRQ
    RED toolkit Tool 4).
    """
    if not text or not text.strip():
        raise ValueError("text cannot be empty")
    sentences = [s for s in _SENT_RE.split(text) if s.strip()]
    n_sentences = max(1, len(sentences))
    words = _WORD_RE.findall(text)
    n_words = max(1, len(words))
    n_syllables = sum(_count_syllables(w) for w in words)

    fkgl = (
        0.39 * (n_words / n_sentences)
        + 11.8 * (n_syllables / n_words)
        - 15.59
    )
    fre = (
        206.835
        - 1.015 * (n_words / n_sentences)
        - 84.6 * (n_syllables / n_words)
    )
    return ReadabilityReport(
        n_words=n_words,
        n_sentences=n_sentences,
        n_syllables=n_syllables,
        flesch_kincaid_grade=round(fkgl, 2),
        flesch_reading_ease=round(fre, 2),
        target_grade_max=target_grade_max,
        meets_red_toolkit_target=fkgl <= target_grade_max,
        rationale=(
            f"FKGL={fkgl:.2f} ({n_words} words / {n_sentences} "
            f"sentences / {n_syllables} syllables); FRE="
            f"{fre:.2f}; "
            f"{'meets' if fkgl <= target_grade_max else 'exceeds'} "
            f"the AHRQ RED toolkit grade-{target_grade_max} target."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Stigma language flagger
# ─────────────────────────────────────────────────────────────────────


_StigmaSeverity = Literal["high", "medium", "low"]


# Each rule: (pattern, suggested_rewrite, severity, citation)
_STIGMA_RULES: list[
    tuple[re.Pattern, str, _StigmaSeverity, str]
] = [
    (re.compile(r"\bdiabetic\s+patient\w*\b", re.IGNORECASE),
     "patient with diabetes",
     "medium",
     "AHRQ + ADA 2017 - person-first language."),
    (re.compile(r"\bdiabetic\b(?!\s+(?:retin|nephro|neuro|keto|"
                r"foot|ulcer|coma|emergency|crisis))",
                re.IGNORECASE),
     "person with diabetes",
     "low",
     "ADA 2017 - person-first language; 'diabetic' is acceptable "
     "for descriptors of clinical entities (retinopathy, etc.)."),
    (re.compile(r"\baddict\b|\baddicts\b", re.IGNORECASE),
     "patient with substance use disorder",
     "high",
     "SAMHSA 2017 - non-stigmatising substance-use language."),
    (re.compile(r"\babuser\b|\babusers\b", re.IGNORECASE),
     "patient with substance use disorder",
     "high",
     "SAMHSA 2017."),
    (re.compile(r"\balcoholic\b(?!\s+(?:liver|hepatitis|"
                r"cirrhosis|cardiomyopathy))",
                re.IGNORECASE),
     "patient with alcohol use disorder",
     "medium",
     "SAMHSA 2017; 'alcoholic' is acceptable for clinical entities "
     "(alcoholic hepatitis)."),
    (re.compile(r"\bdrug\s+abuser\b", re.IGNORECASE),
     "patient with substance use disorder",
     "high",
     "SAMHSA 2017."),
    (re.compile(r"\bnon[-\s]?compliant\b|\bnoncompliant\b",
                re.IGNORECASE),
     "did not adhere to the regimen",
     "medium",
     "AHRQ patient-engagement guideline 2018."),
    (re.compile(r"\bdifficult\s+patient\b", re.IGNORECASE),
     "patient who is having difficulty engaging with care",
     "medium",
     "AHRQ patient-engagement guideline 2018."),
    (re.compile(r"\bfrequent\s+flyer\b", re.IGNORECASE),
     "patient with high health-care utilisation",
     "high",
     "AHRQ patient-engagement guideline 2018; ED slang."),
    (re.compile(r"\bweak\s+history\b", re.IGNORECASE),
     "limited collateral history",
     "low",
     "AHRQ NLP guideline."),
    (re.compile(r"\bcrazy\b", re.IGNORECASE),
     "patient with mental-health condition",
     "high",
     "APA + SAMHSA - non-stigmatising mental-health language."),
    (re.compile(r"\bschizophrenic\b(?!\s+(?:disorder|spectrum))",
                re.IGNORECASE),
     "person with schizophrenia",
     "medium",
     "APA 2013 DSM-5 person-first style."),
    (re.compile(r"\bpsychotic\s+patient\b", re.IGNORECASE),
     "patient experiencing psychosis",
     "medium",
     "APA 2013."),
    (re.compile(r"\bsuicidal\s+patient\b", re.IGNORECASE),
     "patient experiencing suicidal ideation",
     "medium",
     "AAS suicide-language recommendations."),
    (re.compile(r"\bcommitted\s+suicide\b", re.IGNORECASE),
     "died by suicide",
     "high",
     "Reporting on Suicide guidelines (AFSP 2020)."),
    (re.compile(r"\bobese\s+patient\w*\b", re.IGNORECASE),
     "patient with obesity",
     "medium",
     "AHRQ + ACP 2018 person-first language."),
    (re.compile(r"\bmorbidly\s+obese\b", re.IGNORECASE),
     "patient with severe obesity (BMI >= 40)",
     "medium",
     "ACP 2018 - prefer numeric class to value-laden adjective."),
    (re.compile(r"\bdrug\s+seeking\b", re.IGNORECASE),
     "patient with possible opioid use disorder; "
     "evaluate per IDSA + CDC pain guidelines",
     "high",
     "CDC opioid guidance 2022 + SAMHSA 2017."),
    (re.compile(r"\bhis\s+/\s*her\s+complaint\b|\bhis\s*/\s*her\b",
                re.IGNORECASE),
     "the patient's",
     "low",
     "AMA 2020 - inclusive language; prefer singular 'they' or "
     "'the patient's'."),
    (re.compile(r"\bdemented\b", re.IGNORECASE),
     "person with dementia",
     "high",
     "Alzheimer's Association language guidance 2021."),
]


class StigmaFinding(BaseModel):
    matched_text: str
    suggested_rewrite: str
    severity: _StigmaSeverity
    char_offset: int
    citation: str


class StigmaLintReport(BaseModel):
    n_findings: int = Field(ge=0)
    n_high: int = Field(ge=0)
    n_medium: int = Field(ge=0)
    n_low: int = Field(ge=0)
    findings: list[StigmaFinding]
    overall_grade: Literal["clean", "warn", "block"]
    rationale: str


def lint_stigma(text: str) -> StigmaLintReport:
    """Scan ``text`` for stigma language patterns. Returns a list of
    findings + an overall grade:

      - ``clean``: no findings.
      - ``warn``: only medium/low findings.
      - ``block``: at least one high-severity finding.
    """
    if not text:
        return StigmaLintReport(
            n_findings=0, n_high=0, n_medium=0, n_low=0,
            findings=[], overall_grade="clean",
            rationale="Empty input - no findings.",
        )
    findings: list[StigmaFinding] = []
    for pattern, rewrite, severity, citation in _STIGMA_RULES:
        for match in pattern.finditer(text):
            findings.append(StigmaFinding(
                matched_text=match.group(0),
                suggested_rewrite=rewrite,
                severity=severity,
                char_offset=match.start(),
                citation=citation,
            ))
    findings.sort(key=lambda f: f.char_offset)
    n_h = sum(1 for f in findings if f.severity == "high")
    n_m = sum(1 for f in findings if f.severity == "medium")
    n_l = sum(1 for f in findings if f.severity == "low")
    if n_h > 0:
        overall = "block"
    elif n_m + n_l > 0:
        overall = "warn"
    else:
        overall = "clean"
    return StigmaLintReport(
        n_findings=len(findings),
        n_high=n_h, n_medium=n_m, n_low=n_l,
        findings=findings,
        overall_grade=overall,
        rationale=(
            f"{len(findings)} findings ({n_h} high, {n_m} medium, "
            f"{n_l} low); overall `{overall}`."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Combined audit
# ─────────────────────────────────────────────────────────────────────


class PatientFacingTextAudit(BaseModel):
    readability: ReadabilityReport
    stigma: StigmaLintReport
    overall_pass: bool
    rationale: str


def audit_patient_facing_text(
    text: str, *, target_grade_max: int = 8,
) -> PatientFacingTextAudit:
    """One-call combined audit: readability + stigma. Pass = grade
    target met AND no high-severity stigma findings."""
    r = assess_readability(text, target_grade_max=target_grade_max)
    s = lint_stigma(text)
    overall_pass = (
        r.meets_red_toolkit_target and s.overall_grade != "block"
    )
    return PatientFacingTextAudit(
        readability=r, stigma=s,
        overall_pass=overall_pass,
        rationale=(
            f"Readability {r.flesch_kincaid_grade:.2f} grade "
            f"({'pass' if r.meets_red_toolkit_target else 'fail'}); "
            f"stigma overall `{s.overall_grade}`. "
            f"{'PASS' if overall_pass else 'FAIL'}."
        ),
    )
