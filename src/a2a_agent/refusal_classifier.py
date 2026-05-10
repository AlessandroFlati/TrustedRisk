"""Pure-Python refusal classifier mirroring the A2A root agent's safety policy.

This module encodes the same rules described prose-form in `instructions.md`,
exposed as deterministic regex/keyword matchers. Two motivations:

1. **Pre-LLM gate** -- when running the A2A agent, calling this classifier
   first lets us short-circuit obvious refuse-cases without paying for an LLM
   round-trip. The LLM still owns the routing for ambiguous cases.

2. **Eval and regression testing** -- the multi-prompt eval suite
   (tests/eval/) needs reproducible refusal classification independent of
   model availability or non-determinism.

Categories mirror the instructions.md refusal policy:
  - refuse_pediatric:      patient under 18
  - refuse_oncology:       active cancer with chemotherapy
  - refuse_eol:            end-of-life / palliative / hospice
  - refuse_emergency:      coding / STAT / emergency timescales
  - refuse_non_clinical:   pricing / coverage / employment / legal / malpractice
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

RefusalCategory = Literal[
    "decide",
    "validate_only",
    "refuse_pediatric",
    "refuse_oncology",
    "refuse_eol",
    "refuse_emergency",
    "refuse_non_clinical",
]


@dataclass
class ClassificationResult:
    category: RefusalCategory
    matched_pattern: str | None
    rationale: str


# ─────────────────────── Pattern definitions ───────────────────────

# Pediatric: explicit ages or pediatric markers
_PED_AGE_NUMERIC = re.compile(
    r"\b(?:age\s*[:=]?\s*|aged?\s+|"
    r"(?:is|am|she'?s|he'?s|they'?re|patient(?:\s+is)?)\s+(?:a\s+|the\s+)?)"
    r"(\d{1,2})[-\s]?(?:year[-\s]?old|y\.?o\.?|yrs?[-\s]?old)\b",
    re.IGNORECASE,
)
_PED_NUMERIC_ALT = re.compile(
    r"\b(\d{1,2})[-\s]?year[-\s]?old\b",
    re.IGNORECASE,
)
# Matches phrasings like "17 years and 11 months old" where the unit
# follows the number but the "old" lands later in the sentence. Requires
# the 'old' adjective within ~50 chars to avoid false positives like
# "patient was 17 years ago a smoker".
_PED_NUMERIC_VERBOSE = re.compile(
    r"\b(?:patient\s+is\s+|aged?\s+|age\s*[:=]?\s*)?(\d{1,2})\s*years?\b[^.]{0,50}?\bold\b",
    re.IGNORECASE,
)
_PED_KEYWORDS = re.compile(
    r"\b(?:pediatric|paediatric|infant|toddler|child|teenager|"
    r"adolescent|minor|underage|kid)\b",
    re.IGNORECASE,
)
_ADULT_AFFIRM_PATTERN = re.compile(
    r"\b(?:adult|adults|18\s*\+|over\s*18)\b",
    re.IGNORECASE,
)

# Oncology: active cancer + chemotherapy
_ONC_KEYWORDS = re.compile(
    r"\b(?:cancer|carcinoma|sarcoma|leukemia|leukaemia|lymphoma|melanoma|"
    r"oncology|tumor|tumour|metastatic|chemotherapy|chemo|radiotherapy)\b",
    re.IGNORECASE,
)

# End-of-life
_EOL_KEYWORDS = re.compile(
    r"\b(?:hospice|end[-\s]?of[-\s]?life|palliative|comfort[-\s]?care|"
    r"goals?\s+of\s+care|terminal|dying|withdraw\s+(?:care|support))\b",
    re.IGNORECASE,
)

# Emergency / STAT
_EMERG_KEYWORDS = re.compile(
    r"\b(?:coding|code\s+blue|stat\b|emergency(?:\s+(?:dispo|disposition|decision))?|"
    r"asap\b|urgent(?:ly)?|right\s+now|in\s+\d+\s+minutes)\b",
    re.IGNORECASE,
)

# Non-clinical (insurance / employment / legal)
_NON_CLINICAL_KEYWORDS = re.compile(
    r"\b(?:premium|insurance(?:\s+coverage|\s+denial|\s+approval)?|deny\s+(?:insurance|coverage|claim)|"
    r"coverage\s+(?:denial|decision)|malpractice|lawsuit|legal\s+case|legal\s+evidence|"
    r"employment|return\s+to\s+work|fit\s+(?:to|for)\s+work|"
    r"hiring|firing|disability\s+claim)\b",
    re.IGNORECASE,
)

# PHI-only intent
_PHI_INTENT_KEYWORDS = re.compile(
    r"\b(?:scan|check|detect|redact)\b.{0,40}\b(?:phi|protected|"
    r"identifiable\s+information|personal\s+info(?:rmation)?)\b",
    re.IGNORECASE,
)
_PHI_INTENT_ALT = re.compile(
    r"\b(?:phi(?:\s+leakage|\s+leak)?(?:\s+check)?|"
    r"protected\s+information|protected\s+health\s+information)\b",
    re.IGNORECASE,
)

# Clinical decision intent (used as positive signal)
_CLINICAL_INTENT = re.compile(
    r"\b(?:discharge|readmission|admit|disposition|risk(?:\s+(?:score|assessment))?|"
    r"snf|home\s+with\s+care|review\s+the\s+(?:plan|discharge)|"
    r"patient\s+(?:plan|disposition))\b",
    re.IGNORECASE,
)

# Verify-claim intent (validate only path)
_VERIFY_CLAIM = re.compile(
    r"\b(?:verify|check|validate|ground)\s+(?:the\s+|this\s+)?claim\b",
    re.IGNORECASE,
)


# ─────────────────────── Classifier ───────────────────────

def classify_query(text: str) -> ClassificationResult:
    """Classify a user query against the refusal policy + routing rules.

    Returns a `ClassificationResult` whose `category` is one of:
        - `decide` (clinical decision -> decide_then_validate)
        - `validate_only` (PHI-only check or claim verification -> validate)
        - `refuse_*` (one of 5 refusal categories)

    The order of checks is significant: refusal policies are evaluated first,
    then the affirmative intents (decide / validate_only). When a refusal
    pattern fires, no further checks are performed.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    raw = text.strip()
    if not raw:
        return ClassificationResult("decide", None, "Empty input -- defaulting to decide path")

    # 1. Pediatric -- match an age in [0, 17] OR pediatric keyword (without explicit "adult")
    ped = _check_pediatric(raw)
    if ped:
        return ped

    # 2. Oncology
    if _ONC_KEYWORDS.search(raw):
        m = _ONC_KEYWORDS.search(raw)
        return ClassificationResult(
            "refuse_oncology",
            m.group(0) if m else None,
            "Mentions oncology / chemotherapy -- LACE not validated for active oncology",
        )

    # 3. EOL
    if _EOL_KEYWORDS.search(raw):
        m = _EOL_KEYWORDS.search(raw)
        return ClassificationResult(
            "refuse_eol",
            m.group(0) if m else None,
            "End-of-life / palliative / hospice context -- ethics transcend risk calculus",
        )

    # 4. Emergency / STAT
    if _EMERG_KEYWORDS.search(raw):
        m = _EMERG_KEYWORDS.search(raw)
        return ClassificationResult(
            "refuse_emergency",
            m.group(0) if m else None,
            "Emergency timescale markers -- TrustedRisk not designed for emergency decisions",
        )

    # 5. Non-clinical (insurance/legal/employment)
    if _NON_CLINICAL_KEYWORDS.search(raw):
        m = _NON_CLINICAL_KEYWORDS.search(raw)
        return ClassificationResult(
            "refuse_non_clinical",
            m.group(0) if m else None,
            "Non-clinical decision area (pricing/coverage/legal/employment)",
        )

    # 6. Validate-only intent (PHI scan or claim verification, no full decision)
    if _PHI_INTENT_KEYWORDS.search(raw) or _PHI_INTENT_ALT.search(raw):
        m = _PHI_INTENT_KEYWORDS.search(raw) or _PHI_INTENT_ALT.search(raw)
        return ClassificationResult(
            "validate_only",
            m.group(0) if m else None,
            "Explicit PHI-detection intent",
        )
    if _VERIFY_CLAIM.search(raw):
        m = _VERIFY_CLAIM.search(raw)
        return ClassificationResult(
            "validate_only",
            m.group(0) if m else None,
            "Claim-verification intent without a full decision request",
        )

    # 7. Clinical decision intent
    if _CLINICAL_INTENT.search(raw):
        m = _CLINICAL_INTENT.search(raw)
        return ClassificationResult(
            "decide",
            m.group(0) if m else None,
            "Clinical decision-support intent",
        )

    # Fallback -- ambiguous, route to decide
    return ClassificationResult("decide", None, "No specific markers -- default routing to decide_then_validate")


def _check_pediatric(text: str) -> ClassificationResult | None:
    # Numeric age first -- only flag when the matched age is < 18
    for pattern in (_PED_AGE_NUMERIC, _PED_NUMERIC_ALT, _PED_NUMERIC_VERBOSE):
        for m in pattern.finditer(text):
            try:
                age = int(m.group(1))
            except (ValueError, IndexError):
                continue
            if 0 <= age < 18:
                return ClassificationResult(
                    "refuse_pediatric",
                    m.group(0),
                    f"Patient age {age} is below the validated adult cohort (≥18)",
                )

    # Pediatric keywords -- only when there's NO explicit adult-affirming context
    if _PED_KEYWORDS.search(text) and not _ADULT_AFFIRM_PATTERN.search(text):
        m = _PED_KEYWORDS.search(text)
        return ClassificationResult(
            "refuse_pediatric",
            m.group(0) if m else None,
            "Pediatric population marker without adult-affirming context",
        )

    return None
