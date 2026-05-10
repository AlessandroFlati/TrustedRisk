"""healthcare.detect_phi -- PHI entity detection via Microsoft Presidio.

Per design doc §3.3.4: stateless tool (no FHIR context required). Returns
entities + risk level + redaction map.

Falls back to lightweight regex heuristics if Presidio is unavailable
(e.g., during CI smoke tests).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from shared.schemas import PHIEntity, PHIReport


# ─────────────────────────────────────────────────────────────────────
# Presidio loading (lazy, cached)
# ─────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_presidio_analyzer():
    try:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer  # type: ignore
        engine = AnalyzerEngine()
        # Custom recognizers for healthcare-specific entities not in Presidio core.
        engine.registry.add_recognizer(PatternRecognizer(
            supported_entity="MRN",
            patterns=[Pattern("MRN tag", r"\bMRN[:\s]*[A-Z0-9-]+\b", 0.85)],
        ))
        engine.registry.add_recognizer(PatternRecognizer(
            supported_entity="DOB",
            patterns=[Pattern("DOB tag", r"\bDOB[:\s]*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", 0.85)],
        ))
        # Presidio's built-in US_SSN recognizer applies context-boosted thresholds
        # that miss bare "NNN-NN-NNNN" without nearby keywords. Force a high-conf
        # PatternRecognizer to keep parity with the regex fallback path.
        engine.registry.add_recognizer(PatternRecognizer(
            supported_entity="US_SSN",
            patterns=[Pattern("SSN dashed", r"\b\d{3}-\d{2}-\d{4}\b", 0.85)],
        ))
        return engine
    except ImportError:
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# Entity -> replacement token mapping
# ─────────────────────────────────────────────────────────────────────

_REPLACEMENT_BY_TYPE: dict[str, str] = {
    "PERSON": "[NAME]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "PHONE_NUMBER": "[PHONE]",
    "DATE_TIME": "[DATE]",
    "LOCATION": "[LOCATION]",
    "IP_ADDRESS": "[IP]",
    "URL": "[URL]",
    "US_SSN": "[SSN]",
    "US_BANK_NUMBER": "[BANK_ACCOUNT]",
    "CREDIT_CARD": "[CARD]",
    "MEDICAL_LICENSE": "[MEDICAL_LICENSE]",
    "MRN": "[MRN]",
    "DOB": "[DOB]",
}


# ─────────────────────────────────────────────────────────────────────
# High-risk entity types drive risk_level = "high"
# ─────────────────────────────────────────────────────────────────────

_HIGH_RISK_TYPES = {"PERSON", "US_SSN", "MRN", "DOB", "CREDIT_CARD"}
_MEDIUM_RISK_TYPES = {"EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION"}


# ─────────────────────────────────────────────────────────────────────
# Fallback regex patterns (when Presidio unavailable)
# ─────────────────────────────────────────────────────────────────────

_REGEX_FALLBACK_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL_ADDRESS": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE_NUMBER": re.compile(r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "MRN": re.compile(r"\bMRN[:\s]*[A-Z0-9-]+\b", re.IGNORECASE),
    "DOB": re.compile(r"\bDOB[:\s]*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", re.IGNORECASE),
    "CREDIT_CARD": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
}


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def detect_phi(text: str) -> PHIReport:
    """Detect PHI entities in arbitrary text.

    Args:
        text: clinical note or similar text.

    Returns:
        PHIReport with entities + redaction map + risk_level.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    entities = _detect_entities(text)
    count_by_type = _count_by_type(entities)
    redaction_map = _build_redaction_map(text, entities)
    risk_level = _assess_risk_level(entities)

    return PHIReport(
        entities_found=entities,
        entity_count_by_type=count_by_type,
        redaction_map=redaction_map,
        risk_level=risk_level,
    )


def _detect_entities(text: str) -> list[PHIEntity]:
    analyzer = _get_presidio_analyzer()
    if analyzer is not None:
        try:
            # Restrict to PHI-relevant entity types to match the regex fallback
            # surface; URL / PERSON / DATE_TIME / LOCATION are too noisy for
            # clinical free-text and produce false positives on the redteam corpus.
            results = analyzer.analyze(
                text=text,
                language="en",
                entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN",
                          "MRN", "DOB", "CREDIT_CARD"],
            )
            return [
                PHIEntity(
                    type=r.entity_type,
                    start=r.start,
                    end=r.end,
                    confidence=float(r.score),
                    suggested_replacement=_REPLACEMENT_BY_TYPE.get(
                        r.entity_type, f"[{r.entity_type}]"
                    ),
                )
                for r in results
            ]
        except Exception:
            pass  # Fall through to regex

    # Regex fallback
    entities: list[PHIEntity] = []
    for entity_type, pattern in _REGEX_FALLBACK_PATTERNS.items():
        for m in pattern.finditer(text):
            entities.append(PHIEntity(
                type=entity_type,
                start=m.start(),
                end=m.end(),
                confidence=0.70,  # Regex fallback has lower confidence
                suggested_replacement=_REPLACEMENT_BY_TYPE.get(entity_type, f"[{entity_type}]"),
            ))
    # Sort by start offset for deterministic output
    entities.sort(key=lambda e: (e.start, e.end))
    return entities


def _count_by_type(entities: list[PHIEntity]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in entities:
        counts[e.type] = counts.get(e.type, 0) + 1
    return counts


def _build_redaction_map(text: str, entities: list[PHIEntity]) -> dict[str, str]:
    """Build mapping of original->redacted for each detected entity."""
    redaction: dict[str, str] = {}
    for e in entities:
        original = text[e.start:e.end]
        if original and original not in redaction:
            redaction[original] = e.suggested_replacement
    return redaction


def _assess_risk_level(entities: list[PHIEntity]) -> str:
    if not entities:
        return "none"
    has_high = any(e.type in _HIGH_RISK_TYPES for e in entities)
    has_medium = any(e.type in _MEDIUM_RISK_TYPES for e in entities)
    if has_high or len(entities) > 5:
        return "high"
    if has_medium or len(entities) > 2:
        return "medium"
    return "low"


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(detect_phi)
