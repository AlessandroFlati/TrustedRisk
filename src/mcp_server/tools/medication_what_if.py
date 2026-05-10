"""healthcare.compute_medication_what_if -- Phase 2.3 PATIENT-Q2.

Answers the most common "what if I...?" medication scenarios from a
patient. Pure-deterministic responses, anchored on a small drug-class
table + a fixed set of clinically-relevant scenarios. Optional LLM
polish never invents new medical facts.

The tool intentionally does NOT compute interactions on the fly -- for
that, the BYO orchestrator should call `compute_rxnorm_ddi_lookup`
(DATA-2). This tool is the patient-facing complement: short, plain-
language answers a patient or caregiver can act on.
"""

from __future__ import annotations

import os
from typing import Any

from shared.schemas import (
    MedicationWhatIfItem,
    MedicationWhatIfResponse,
    WhatIfScenario,
)


_LLM_POLISH_ENV = "TRUSTEDRISK_PATIENT_LLM_POLISH"


def _llm_polish_enabled() -> bool:
    return os.environ.get(_LLM_POLISH_ENV, "0").lower() in (
        "1", "true", "yes", "on",
    )


# ─────────────────────────────────────────────────────────────────────
# Drug-class hints -- deterministic answer modifiers
# ─────────────────────────────────────────────────────────────────────

# Map a substring (case-insensitive) -> drug-class label used to vary
# the deterministic answer.
_DRUG_CLASS_HINTS: dict[str, str] = {
    "warfarin":                "anticoagulant_vka",
    "apixaban":                "anticoagulant_doac",
    "rivaroxaban":             "anticoagulant_doac",
    "dabigatran":              "anticoagulant_doac",
    "lisinopril":              "ace_inhibitor",
    "enalapril":               "ace_inhibitor",
    "losartan":                "arb",
    "valsartan":               "arb",
    "spironolactone":          "mra",
    "eplerenone":              "mra",
    "furosemide":              "loop_diuretic",
    "torsemide":               "loop_diuretic",
    "metformin":               "biguanide",
    "metoprolol":              "beta_blocker",
    "atenolol":                "beta_blocker",
    "atorvastatin":            "statin",
    "rosuvastatin":            "statin",
    "simvastatin":             "statin",
    "amiodarone":              "antiarrhythmic",
    "amlodipine":              "ccb",
    "diltiazem":               "ccb",
    "ibuprofen":               "nsaid",
    "naproxen":                "nsaid",
    "aspirin":                 "antiplatelet",
    "clopidogrel":             "antiplatelet",
    "tylenol":                 "acetaminophen",
    "acetaminophen":           "acetaminophen",
    "alprazolam":              "benzodiazepine",
    "lorazepam":               "benzodiazepine",
    "oxycodone":               "opioid",
    "morphine":                "opioid",
    "tramadol":                "opioid",
}


def _resolve_drug_class(name: str) -> str | None:
    n = name.lower()
    for substring, klass in _DRUG_CLASS_HINTS.items():
        if substring in n:
            return klass
    return None


# ─────────────────────────────────────────────────────────────────────
# Scenario answer table -- (scenario, drug_class | None) -> (severity, text)
# ─────────────────────────────────────────────────────────────────────

# Default fallback by scenario when no drug_class matches.
_DEFAULTS: dict[WhatIfScenario, tuple[str, str]] = {
    "miss_one_dose": (
        "low",
        "If you miss a dose, take it as soon as you remember -- UNLESS "
        "it's almost time for your next dose. In that case, skip the "
        "missed one. Never double up.",
    ),
    "double_dose_by_mistake": (
        "moderate",
        "If you accidentally take two doses, contact your pharmacist "
        "or call the discharge line for guidance. Do NOT take more "
        "doses until they confirm.",
    ),
    "take_with_food": (
        "low",
        "Most medications can be taken with food. Check the label or "
        "ask your pharmacist if it should be on an empty stomach.",
    ),
    "take_with_alcohol": (
        "moderate",
        "Avoid alcohol while on this medication unless your team "
        "specifically says it's OK. Alcohol can increase side-effects "
        "or reduce how well the medication works.",
    ),
    "take_with_grapefruit": (
        "moderate",
        "Some medications interact with grapefruit. Ask your "
        "pharmacist before adding grapefruit / grapefruit juice to "
        "your daily routine.",
    ),
    "take_with_otc_nsaid": (
        "moderate",
        "Over-the-counter NSAIDs (ibuprofen, naproxen) can interact "
        "with many prescriptions. Ask your pharmacist before using "
        "them regularly. Acetaminophen is usually safer.",
    ),
    "take_with_otc_antacid": (
        "low",
        "Antacids can change how some pills are absorbed. Take them "
        "at least 2 hours apart from your prescription.",
    ),
    "take_with_supplements": (
        "low",
        "Some supplements (vitamin K, St John's wort, fish oil) can "
        "change how prescriptions work. Tell your pharmacist about "
        "every supplement you take.",
    ),
    "stop_abruptly": (
        "high",
        "Don't stop a prescription on your own. Call your team first.",
    ),
    "interaction_with_other_chronic_med": (
        "moderate",
        "Bring a complete list of every medication and supplement to "
        "every visit. Ask your pharmacist to run an interaction check "
        "whenever a new prescription is added.",
    ),
}


# Class-specific overrides (more specific safety guidance)
_CLASS_OVERRIDES: dict[tuple[WhatIfScenario, str], tuple[str, str]] = {
    ("take_with_alcohol", "anticoagulant_vka"): (
        "high",
        "Alcohol can change how warfarin works and raise bleeding "
        "risk. Limit yourself to 1 drink/day or less and tell your "
        "team about your usual intake -- your INR target may need "
        "tighter monitoring.",
    ),
    ("take_with_alcohol", "benzodiazepine"): (
        "high",
        "Combining benzodiazepines with alcohol can cause severe "
        "drowsiness or stop your breathing. Avoid alcohol entirely.",
    ),
    ("take_with_alcohol", "opioid"): (
        "high",
        "Alcohol with opioids can cause overdose. Avoid alcohol "
        "entirely while taking this medication.",
    ),
    ("take_with_alcohol", "acetaminophen"): (
        "high",
        "Combining alcohol with acetaminophen damages the liver. "
        "Limit alcohol or skip the acetaminophen.",
    ),
    ("take_with_grapefruit", "statin"): (
        "high",
        "Grapefruit juice raises the level of many statins (especially "
        "simvastatin and atorvastatin) and increases muscle-injury "
        "risk. Avoid grapefruit while on this medication.",
    ),
    ("take_with_grapefruit", "ccb"): (
        "high",
        "Grapefruit can sharply raise calcium-channel-blocker levels "
        "and cause low blood pressure. Avoid grapefruit on this "
        "medication.",
    ),
    ("take_with_otc_nsaid", "anticoagulant_vka"): (
        "high",
        "NSAIDs (ibuprofen, naproxen) plus warfarin sharply raise "
        "bleeding risk. Use acetaminophen instead and call your team "
        "before any regular NSAID use.",
    ),
    ("take_with_otc_nsaid", "ace_inhibitor"): (
        "moderate",
        "Regular NSAIDs can reduce ACE-inhibitor effect and stress "
        "your kidneys, especially if you have heart failure. Use "
        "sparingly and prefer acetaminophen.",
    ),
    ("stop_abruptly", "beta_blocker"): (
        "high",
        "Stopping a beta-blocker suddenly can cause chest pain or "
        "rebound high blood pressure. Always taper under your team's "
        "guidance.",
    ),
    ("stop_abruptly", "ace_inhibitor"): (
        "moderate",
        "Don't stop your ACE-inhibitor on your own. If you have "
        "side-effects (dry cough, swelling), call your team for a "
        "switch -- don't just stop.",
    ),
    ("miss_one_dose", "anticoagulant_vka"): (
        "moderate",
        "If you miss a warfarin dose: take it the same day if you "
        "remember; if it's already the next day, skip it and take "
        "only your usual dose. NEVER double up. Tell your team -- "
        "your INR may need an early check.",
    ),
    ("miss_one_dose", "antiarrhythmic"): (
        "moderate",
        "If you miss an amiodarone dose, take it as soon as you "
        "remember unless it's almost time for the next one -- then "
        "skip. Don't double up; call your cardiologist if you miss "
        "more than one in a row.",
    ),
    ("double_dose_by_mistake", "opioid"): (
        "high",
        "Double-dosing on an opioid can cause overdose. Watch for "
        "extreme drowsiness or slow breathing -- call 911 if those "
        "appear. Otherwise call the prescriber promptly.",
    ),
    ("double_dose_by_mistake", "anticoagulant_vka"): (
        "high",
        "A double dose of warfarin can sharply raise bleeding risk. "
        "Call your anticoagulation clinic the same day for an early "
        "INR check.",
    ),
}


def _resolve_answer(
    medication_name: str,
    scenario: WhatIfScenario,
) -> tuple[str, str, str | None]:
    """Return (severity, answer, rationale)."""
    klass = _resolve_drug_class(medication_name)
    if klass is not None:
        override = _CLASS_OVERRIDES.get((scenario, klass))
        if override is not None:
            severity, answer = override
            return severity, answer, f"Class override for {klass}"
    severity, answer = _DEFAULTS[scenario]
    return severity, answer, "Default scenario answer"


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

# Default scenarios when the caller doesn't specify any
_DEFAULT_SCENARIOS: list[WhatIfScenario] = [
    "miss_one_dose",
    "double_dose_by_mistake",
    "take_with_alcohol",
    "stop_abruptly",
]


async def compute_medication_what_if(
    medications: list[str],
    scenarios: list[WhatIfScenario] | None = None,
    patient_reference: str | None = None,
) -> MedicationWhatIfResponse:
    """Answer common medication-scenario questions.

    Args:
        medications: List of medication names (free-text or branded
            names).
        scenarios: Optional list of scenarios to cover; defaults to
            the four most common.
        patient_reference: Optional patient reference for audit.

    Returns:
        MedicationWhatIfResponse with one item per (medication,
        scenario) pair.
    """
    if not medications:
        return MedicationWhatIfResponse(
            patient_reference=patient_reference,
            items=[],
            n_items=0,
            abstain_recommended=True,
            abstain_reason=(
                "No medications provided. The what-if tool needs at "
                "least one medication name."
            ),
            references=[
                "AHRQ MATCH program -- patient-medication safety "
                "education templates.",
                "Beers criteria 2023 -- geriatric medication cautions.",
            ],
        )

    scenarios = scenarios or _DEFAULT_SCENARIOS
    items: list[MedicationWhatIfItem] = []
    for med in medications:
        for sc in scenarios:
            severity, answer, rationale = _resolve_answer(med, sc)
            items.append(MedicationWhatIfItem(
                medication_name=med,
                scenario=sc,
                deterministic_answer=answer,
                safety_severity=severity,                # type: ignore[arg-type]
                rationale=rationale,
            ))

    contains_polish = False
    polish_model: str | None = None
    if _llm_polish_enabled():
        try:
            from a2a_agent.llm_critic import _resolve_llm_model
            model = _resolve_llm_model()
        except Exception:
            model = None
        if model is not None:
            contains_polish = True
            polish_model = getattr(model, "model_name", "unknown")
            for it in items:
                it.is_llm_polished = True

    return MedicationWhatIfResponse(
        patient_reference=patient_reference,
        items=items,
        n_items=len(items),
        contains_llm_polish=contains_polish,
        llm_model_id=polish_model,
        references=[
            "AHRQ MATCH program -- medication-safety education.",
            "FDA Drug-Drug Interaction Studies guidance.",
            "Beers criteria 2023 -- geriatric medication cautions.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_medication_what_if)
