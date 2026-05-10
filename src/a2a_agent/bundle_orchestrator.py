"""Bundle composition orchestrator (SAFE-2).

Given a free-text patient summary + structured patient features, ranks
the available bundles by relevance and returns a recommended bundle
composition (primary + add-ons).

This is the auto-discovery half of the A2A capability negotiation: an
upstream client (or a federated agent) can call /api/bundles/suggest
without prior knowledge of which bundles exist; the orchestrator picks
them based on clinical context.

Three signals contribute to the per-bundle score:

  1. Keyword match in the chief complaint / free-text summary
     (e.g. "chest pain" -> stroke_acs +3.0)
  2. Structured patient feature match
     (age <18 -> pediatric +2.5; pregnant -> obstetric_geriatric +2.5;
      eGFR <30 -> nephrology +1.5)
  3. Cross-bundle add-on inference
     (trauma -> +ed_acute; chemo -> +nephrology if cisplatin; preeclampsia
      -> +imaging if severe headache)

The output ranks bundles 0.0-10.0 and returns the top primary +
add-on suggestions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────── Per-bundle keyword patterns ───────────────────────

_BUNDLE_KEYWORDS: dict[str, list[tuple[re.Pattern[str], float]]] = {
    "stroke_acs": [
        (re.compile(r"\bstroke\b|\bnihss\b|tpa\b|alteplase|thrombolysis|"
                     r"cerebral|cva\b", re.I), 3.5),
        (re.compile(r"facial droop|slurred speech|hemiparesis|"
                     r"aphasia|focal deficit", re.I), 3.0),
        (re.compile(r"chest pain|crushing|radiating to (left )?arm|"
                     r"angina|stemi|nstemi|troponin", re.I), 3.0),
        (re.compile(r"\bheart score\b", re.I), 2.0),
    ],
    "trauma_critical": [
        (re.compile(r"polytrauma|multi[- ]?cavity|mvc\b|motor vehicle|"
                     r"trauma activation", re.I), 4.0),
        (re.compile(r"fast (positive|exam)|hemorrhag|massive transfusion|"
                     r"\bmtp\b", re.I), 3.5),
        (re.compile(r"penetrating injury|gunshot|stab wound", re.I), 3.0),
    ],
    "pediatric": [
        (re.compile(r"\b(infant|child|newborn|neonate|pediatric|"
                     r"toddler)\b", re.I), 4.0),
        (re.compile(r"weight[- ]based|broselow|pews\b", re.I), 2.5),
    ],
    "obstetric_geriatric": [
        (re.compile(r"preeclampsia|eclampsia|hellp|gestational|pregnant|"
                     r"\bg\d+p\d+\b|postpartum|peripartum", re.I), 4.0),
        (re.compile(r"meows", re.I), 3.0),
        (re.compile(r"\bdelirium\b|\bcam\b|confused|fluctuating", re.I), 3.5),
        (re.compile(r"\bfalls?\b|fall risk|morse|fell|stumbling", re.I), 3.0),
        (re.compile(r"geriatric|elderly|85[- ]year|over 80", re.I), 2.0),
    ],
    "antimicrobial": [
        (re.compile(r"\bsepsis\b|septic shock|empiric antibiotics|"
                     r"culture", re.I), 3.5),
        (re.compile(r"\butis?\b|pneumonia|cellulitis|meningitis|"
                     r"complicated infection", re.I), 3.0),
        (re.compile(r"\bmrsa\b|\besbl\b|de[- ]escalation|stewardship",
                     re.I), 3.5),
    ],
    "oncology": [
        (re.compile(r"chemo|chemotherapy|cycle \d+|nsclc|breast cancer|"
                     r"oncology|recist", re.I), 4.0),
        (re.compile(r"neutropen|\banc\b|febrile neutropenia", re.I), 2.5),
        (re.compile(r"\bcarboplatin\b|cisplatin|paclitaxel|doxorubicin",
                     re.I), 3.0),
    ],
    "endocrine_acute": [
        (re.compile(r"\bdka\b|ketoacidosis|hyperglycemic|"
                     r"glucose 5\d{2}|glucose [4-9]\d{2}", re.I), 4.0),
        (re.compile(r"insulin|basal[- ]bolus|hypoglyc", re.I), 2.5),
    ],
    "imaging": [
        (re.compile(r"ct[- ]?p?a\b|ctpa|cta\b|mri|imaging|appropriateness|"
                     r"contrast safety", re.I), 3.0),
        (re.compile(r"acr criteria|alara|radiation dose", re.I), 2.0),
    ],
    "nephrology": [
        (re.compile(r"\baki\b|acute kidney injur|kdigo|dialysis|"
                     r"hemodial|crrt\b|nephrolog", re.I), 4.0),
        (re.compile(r"creatinine|\begfr\b|oliguric", re.I), 2.0),
    ],
    "mental_health": [
        (re.compile(r"suicid|self[- ]harm|ideation|psychiatric|"
                     r"\bmdd\b|major depression", re.I), 4.0),
        (re.compile(r"c[- ]?ssrs|hold|involuntary|baker act|5150|"
                     r"m[- ]?1 hold", re.I), 3.5),
    ],
    "ed_acute": [
        (re.compile(r"\bed\b|emergency department|triage|chief complaint|"
                     r"esi|news2|deterioration", re.I), 2.5),
        (re.compile(r"rapid response|early warning|inpatient deteriora",
                     re.I), 2.5),
    ],
    "core_discharge": [
        (re.compile(r"discharge|readmission|transition of care|"
                     r"medication reconciliation|polypharmacy", re.I), 2.5),
        (re.compile(r"safe to go home|home with services|outpatient followup",
                     re.I), 2.0),
    ],
}


# ─────────────────────── Structured-feature signals ───────────────────────

def _structured_score(features: dict[str, Any]) -> dict[str, float]:
    """Score bundles based on structured features (age, eGFR, GA, etc.)."""
    out: dict[str, float] = {}

    # Age-driven
    age = features.get("age")
    if isinstance(age, (int, float)):
        if age < 18:
            out["pediatric"] = out.get("pediatric", 0.0) + 2.5
        if age >= 75:
            out["obstetric_geriatric"] = (
                out.get("obstetric_geriatric", 0.0) + 1.5
            )

    # Pregnancy
    if features.get("pregnant") or features.get("gestational_age_weeks"):
        out["obstetric_geriatric"] = out.get("obstetric_geriatric", 0.0) + 2.5

    # Renal function
    egfr = features.get("egfr_ml_min")
    if isinstance(egfr, (int, float)):
        if egfr < 30:
            out["nephrology"] = out.get("nephrology", 0.0) + 1.5
        if egfr < 60 and features.get("contrast_imaging_planned"):
            out["imaging"] = out.get("imaging", 0.0) + 1.0

    # Cancer flag
    if features.get("active_cancer") or features.get("on_chemotherapy"):
        out["oncology"] = out.get("oncology", 0.0) + 2.0

    # Suicide risk fields
    if (features.get("suicide_ideation_present")
            or features.get("psychiatric_history")):
        out["mental_health"] = out.get("mental_health", 0.0) + 2.5

    # Trauma flags
    if features.get("trauma_activation") or features.get("massive_hemorrhage"):
        out["trauma_critical"] = out.get("trauma_critical", 0.0) + 3.0

    # Discharge planning
    if features.get("discharge_planning"):
        out["core_discharge"] = out.get("core_discharge", 0.0) + 2.0

    return out


# ─────────────────────── Cross-bundle inference ───────────────────────

_CROSS_BUNDLE_RULES: list[tuple[str, str, float, str]] = [
    ("trauma_critical", "ed_acute", 1.5,
     "Trauma activation -> NEWS2 + ESI on arrival also relevant."),
    ("oncology", "nephrology", 1.0,
     "Cisplatin / chemo regimens often gate on renal function."),
    ("obstetric_geriatric", "imaging", 1.0,
     "Severe preeclampsia features may warrant imaging workup."),
    ("antimicrobial", "ed_acute", 1.5,
     "Sepsis presentation triggers ED triage + NEWS2 monitoring."),
    ("antimicrobial", "nephrology", 1.0,
     "Renal-dose-adjust antibiotics for AKI patients."),
    ("nephrology", "imaging", 1.5,
     "AKI patients need contrast safety check before any CT."),
    ("endocrine_acute", "nephrology", 1.5,
     "DKA can precipitate AKI from osmotic diuresis."),
    ("stroke_acs", "imaging", 2.0,
     "Stroke activation requires non-contrast CT + CT angiography."),
    ("mental_health", "core_discharge", 1.0,
     "Discharge planning for psychiatric patients includes safety planning."),
]


# ─────────────────────── Output schema ───────────────────────

@dataclass
class BundleSuggestion:
    bundle_id: str
    score: float
    rationale: str
    role: str       # "primary" / "add_on"


@dataclass
class BundleSuggestionResult:
    primary_bundle: str
    add_on_bundles: list[str]
    ranked: list[BundleSuggestion]
    keyword_signals: dict[str, list[str]] = field(default_factory=dict)
    structured_signals: dict[str, float] = field(default_factory=dict)
    summary: str = ""


# ─────────────────────── Main API ───────────────────────

def suggest_bundles(
    chief_complaint: str = "",
    free_text_summary: str = "",
    structured_features: dict[str, Any] | None = None,
    add_on_score_cutoff: float = 1.0,
) -> BundleSuggestionResult:
    """Score and rank bundles for a clinical scenario.

    Args:
        chief_complaint: short ED-style chief complaint string.
        free_text_summary: longer free-text describing the patient.
        structured_features: dict with keys among {age, gestational_age_weeks,
            pregnant, egfr_ml_min, contrast_imaging_planned,
            active_cancer, on_chemotherapy, suicide_ideation_present,
            psychiatric_history, trauma_activation, massive_hemorrhage,
            discharge_planning}.
        add_on_score_cutoff: bundles below this score are not suggested
            as add-ons.

    Returns:
        BundleSuggestionResult with primary + add-ons + per-bundle
        rationale.
    """
    text = f"{chief_complaint}\n{free_text_summary}"
    features = structured_features or {}

    keyword_scores: dict[str, float] = {}
    keyword_signals: dict[str, list[str]] = {}

    for bundle, patterns in _BUNDLE_KEYWORDS.items():
        for pattern, weight in patterns:
            m = pattern.search(text)
            if m:
                keyword_scores[bundle] = keyword_scores.get(bundle, 0.0) + weight
                keyword_signals.setdefault(bundle, []).append(m.group(0))

    structured_scores = _structured_score(features)

    # Combine
    combined: dict[str, float] = {}
    for b in set(list(keyword_scores) + list(structured_scores)):
        combined[b] = (keyword_scores.get(b, 0.0)
                         + structured_scores.get(b, 0.0))

    if not combined:
        # No signal found -- fall back to core_discharge
        combined["core_discharge"] = 1.0

    # Cross-bundle add-on inference
    primary = max(combined.items(), key=lambda kv: kv[1])[0]
    cross_bonus_rationale: list[str] = []
    for src, dst, bonus, why in _CROSS_BUNDLE_RULES:
        if combined.get(src, 0.0) >= 2.0:
            combined[dst] = combined.get(dst, 0.0) + bonus
            cross_bonus_rationale.append(f"{src} -> {dst}: {why}")

    # Rank
    ranked_pairs = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
    primary = ranked_pairs[0][0]
    add_ons = [b for b, s in ranked_pairs[1:]
                if s >= add_on_score_cutoff and b != primary][:3]

    ranked: list[BundleSuggestion] = []
    for i, (b, s) in enumerate(ranked_pairs):
        role = "primary" if i == 0 else (
            "add_on" if b in add_ons else "candidate"
        )
        rationale_parts = []
        if b in keyword_signals:
            rationale_parts.append(
                f"keyword matches: {keyword_signals[b][:3]}"
            )
        if b in structured_scores:
            rationale_parts.append(
                f"structured signal +{structured_scores[b]}"
            )
        if not rationale_parts:
            rationale_parts.append("default fallback")
        ranked.append(BundleSuggestion(
            bundle_id=b, score=round(s, 2),
            rationale="; ".join(rationale_parts), role=role,
        ))

    summary_parts = [
        f"Primary bundle: {primary} (score {combined[primary]:.2f}).",
    ]
    if add_ons:
        summary_parts.append(
            f"Add-ons: {', '.join(add_ons)}."
        )
    if cross_bonus_rationale:
        summary_parts.append(
            f"Cross-bundle bonuses applied: "
            + "; ".join(cross_bonus_rationale[:2])
        )

    return BundleSuggestionResult(
        primary_bundle=primary, add_on_bundles=add_ons,
        ranked=ranked, keyword_signals=keyword_signals,
        structured_signals=structured_scores,
        summary=" ".join(summary_parts),
    )
