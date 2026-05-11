"""healthcare.detect_polypharmacy_concerns -- stateless polypharmacy + DDI scan.

Sister tool to compute_medication_reconciliation but with a different operational
profile:

  - **Stateless**: takes a list of medications, no FHIR context, no SHARP
    headers required. Useful for batch processing, pre-admission med review,
    or for any caller that already has a structured med list.
  - **Curated DDI table**: a small, high-confidence list of drug-drug
    interaction pairs sourced from ISMP / FDA / Lexicomp severity ratings.
    Not a complete DDI database (production should integrate a managed
    DDI service); the curated set targets the high-prevalence, high-severity
    pairs that map to the AHRQ readmission-driver med classes.
  - **No PHI fetch**: pure function of the input. Composable with
    detect_phi when the input is free-text rather than structured.
"""

from __future__ import annotations

import re
from collections import Counter

from shared.schemas import DrugInteraction, PolypharmacyReport

from ._chart_inputs import chart_abstain_reason, harden_clinical_inputs

# Reuse the drug-class taxonomy from medication_reconciliation so the two
# tools agree on what "high-risk" means.
from .medication_reconciliation import (
    _HIGH_RISK_CLASSES,
    _NAME_TO_CLASS,
    _classify_drug,
)


# ─────────────────────────────────────────────────────────────────────
# Curated DDI pairs (high-confidence, high-severity)
# ─────────────────────────────────────────────────────────────────────
# Each entry: (regex_a, regex_b, severity, mechanism, detail)
# Matching is symmetric (the order of meds in the input doesn't matter).

_DDI_RULES: list[tuple[re.Pattern[str], re.Pattern[str], str, str, str]] = [
    (
        re.compile(r"warfarin", re.IGNORECASE),
        re.compile(r"ibuprofen|naproxen|ketorolac|aspirin|celecoxib", re.IGNORECASE),
        "high",
        "additive bleeding risk",
        "NSAID + warfarin combines anti-platelet effect with anticoagulation. "
        "Increased GI bleed risk; avoid or use PPI co-prescription.",
    ),
    (
        re.compile(r"warfarin", re.IGNORECASE),
        re.compile(r"clarithromycin|erythromycin|fluconazole|metronidazole|trimethoprim",
                   re.IGNORECASE),
        "high",
        "CYP-mediated INR rise",
        "Macrolide / azole / sulfa antimicrobials inhibit warfarin metabolism -- "
        "INR can rise rapidly. Monitor INR within 3-5 days of co-prescription.",
    ),
    (
        re.compile(r"lisinopril|enalapril|ramipril|captopril", re.IGNORECASE),
        re.compile(r"losartan|valsartan|irbesartan|candesartan", re.IGNORECASE),
        "high",
        "ACE-I + ARB dual blockade",
        "Combined RAS blockade increases AKI and hyperkalemia risk without "
        "outcome benefit (ONTARGET 2008). Avoid except in heart failure under "
        "specialist guidance.",
    ),
    (
        re.compile(r"lisinopril|enalapril|ramipril|captopril|losartan|valsartan", re.IGNORECASE),
        re.compile(r"spironolactone|eplerenone", re.IGNORECASE),
        "medium",
        "hyperkalemia risk",
        "ACE-I/ARB + MRA combination requires K+ monitoring within 1-2 weeks "
        "of initiation. Hyperkalemia is the leading discontinuation cause.",
    ),
    (
        re.compile(r"sertraline|fluoxetine|paroxetine|citalopram|escitalopram|venlafaxine|duloxetine",
                   re.IGNORECASE),
        re.compile(r"tramadol|linezolid|methylene\s*blue", re.IGNORECASE),
        "high",
        "serotonin syndrome",
        "SSRI/SNRI + serotonergic agent (tramadol, linezolid, methylene blue) "
        "carries serotonin-syndrome risk. Avoid; if unavoidable, monitor for "
        "tremor / hyperthermia / clonus.",
    ),
    (
        re.compile(r"simvastatin|atorvastatin|lovastatin", re.IGNORECASE),
        re.compile(r"clarithromycin|erythromycin|itraconazole|ketoconazole|gemfibrozil",
                   re.IGNORECASE),
        "medium",
        "rhabdomyolysis risk via CYP3A4",
        "Strong CYP3A4 inhibitor + simvastatin/lovastatin elevates statin "
        "concentrations; rhabdomyolysis risk. Hold the statin during the "
        "antimicrobial course.",
    ),
    (
        re.compile(r"digoxin", re.IGNORECASE),
        re.compile(r"furosemide|lasix|torsemide|bumetanide|hydrochlorothiazide", re.IGNORECASE),
        "medium",
        "hypokalemia -> digoxin toxicity",
        "Loop / thiazide diuretics induce hypokalemia which sensitizes the "
        "myocardium to digoxin toxicity. Monitor K+ and digoxin level.",
    ),
    (
        re.compile(r"warfarin|apixaban|rivaroxaban|dabigatran|edoxaban|\bheparin\b|enoxaparin",
                   re.IGNORECASE),
        re.compile(r"warfarin|apixaban|rivaroxaban|dabigatran|edoxaban|\bheparin\b|enoxaparin",
                   re.IGNORECASE),
        "high",
        "double anticoagulation",
        "Two anticoagulants on the discharge list is rarely intentional outside "
        "bridging windows. Verify intent and bridging plan.",
    ),
    (
        re.compile(r"insulin|glargine|lispro|aspart|degludec|detemir", re.IGNORECASE),
        re.compile(r"metformin|glucophage", re.IGNORECASE),
        "low",
        "hypoglycemia awareness",
        "Insulin + metformin is a common dual regimen. Low-severity flag -- "
        "verify hypoglycemia counseling at discharge.",
    ),
    (
        re.compile(r"empagliflozin|dapagliflozin|canagliflozin", re.IGNORECASE),
        re.compile(r"furosemide|lasix|torsemide|bumetanide|hydrochlorothiazide", re.IGNORECASE),
        "medium",
        "volume depletion + AKI",
        "SGLT2-i + loop / thiazide diuretic increases volume-depletion and AKI "
        "risk, especially at heart-failure decompensation discharge. Monitor "
        "weight + creatinine within 7-14 days.",
    ),
]


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def detect_polypharmacy_concerns(
    medications: list[str | dict] | None = None,
) -> PolypharmacyReport:
    """Stateless polypharmacy + DDI scan.

    When SHARP-on-MCP context is bound, the medication list is sourced
    from FHIR MedicationRequest / MedicationAdministration / chart-note
    medication sections; caller-supplied `medications` are DISCARDED in
    that mode. In offline / unit-test mode (no SHARP context) the
    caller-supplied list is honoured.

    Args:
        medications: LEAVE NULL / OMIT in production. Mediation names
            are auto-resolved from the SHARP-bound patient's chart.
            This parameter is honoured only for offline / unit-test
            invocations.

    Returns:
        PolypharmacyReport with class_counts, interactions list, and an
        overall severity label.
    """
    resolved, sharp_bound, missing = await harden_clinical_inputs(
        {"medications": medications},
        chart_derivable={"medications"},
    )
    if sharp_bound and missing:
        return PolypharmacyReport(
            n_medications=0, n_high_risk=0,
            class_counts={}, interactions=[],
            polypharmacy_severity="none",
            rationale=(
                "Polypharmacy scan abstained -- no MedicationRequest, "
                "MedicationAdministration, or chart-note medication "
                "section was found in the SHARP-bound patient's bundle. "
                "The caller-supplied medication list (if any) was "
                "discarded to prevent the chat-side LLM from injecting "
                "fabricated medications."
            ),
            abstain_recommended=True,
            abstain_reason=chart_abstain_reason(missing),
        )
    medications = resolved.get("medications")
    if not medications:
        # Empty med list cannot be distinguished from "patient is on no
        # meds" vs "caller didn't supply the list" -- the safe default
        # is to abstain so a downstream consumer cannot interpret the
        # n=0 report as a clean "no polypharmacy" finding.
        return PolypharmacyReport(
            n_medications=0, n_high_risk=0,
            class_counts={}, interactions=[],
            polypharmacy_severity="none",
            rationale=(
                "Polypharmacy scan abstained: no medication list was "
                "supplied. Calling this tool with an empty list cannot "
                "distinguish 'patient is on no meds' from 'caller did "
                "not supply the list', so the n=0 result is not a "
                "clinical finding."
            ),
            abstain_recommended=True,
            abstain_reason=(
                "missing_medication_list: detect_polypharmacy_concerns "
                "requires an explicit medication list; an empty list "
                "is treated as 'data not supplied' rather than "
                "'patient on no meds'."
            ),
        )

    names: list[str] = []
    for m in medications:
        if isinstance(m, str):
            names.append(m)
        elif isinstance(m, dict):
            n = m.get("name") or m.get("medication") or m.get("display")
            if n:
                names.append(str(n))

    classes: list[str | None] = [_classify_drug(n) for n in names]
    classified_classes = [c for c in classes if c]
    n_high_risk = sum(1 for c in classified_classes if c in _HIGH_RISK_CLASSES)

    class_counts = dict(Counter(classified_classes))

    interactions = _scan_interactions(names)

    severity = _grade_severity(n_high_risk, interactions)
    rationale = _build_rationale(n_high_risk, len(names), interactions, severity)

    return PolypharmacyReport(
        n_medications=len(names),
        n_high_risk=n_high_risk,
        class_counts=class_counts,
        interactions=interactions,
        polypharmacy_severity=severity,
        rationale=rationale,
    )


def _scan_interactions(names: list[str]) -> list[DrugInteraction]:
    """Walk DDI rules against every unordered pair of meds. Skip self-pairs."""
    found: list[DrugInteraction] = []
    seen_pairs: set[tuple[str, str]] = set()
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            for pat_a, pat_b, sev, mech, detail in _DDI_RULES:
                if (pat_a.search(a) and pat_b.search(b)) or (pat_a.search(b) and pat_b.search(a)):
                    key = tuple(sorted([a.lower(), b.lower()]))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    found.append(DrugInteraction(
                        drug_a=a, drug_b=b, severity=sev,  # type: ignore[arg-type]
                        mechanism=mech, detail=detail,
                    ))
                    break
    return found


def _grade_severity(n_high_risk: int, interactions: list[DrugInteraction]) -> str:
    has_high_ddi = any(i.severity == "high" for i in interactions)
    has_med_ddi = any(i.severity == "medium" for i in interactions)
    if has_high_ddi or n_high_risk >= 7:
        return "high"
    if has_med_ddi or n_high_risk >= 5:
        return "medium"
    if n_high_risk >= 3 or interactions:
        return "low"
    return "none"


def _build_rationale(n_high_risk: int, n_total: int,
                     interactions: list[DrugInteraction], severity: str) -> str:
    parts = [f"{n_total} medication(s); {n_high_risk} mapped to high-risk classes."]
    if interactions:
        sev_counts = Counter(i.severity for i in interactions)
        parts.append(
            "DDI scan: "
            + ", ".join(f"{n} {s}-severity" for s, n in sorted(sev_counts.items()))
            + "."
        )
    if severity == "high":
        parts.append("Polypharmacy risk classified HIGH -- clinician review required before discharge.")
    elif severity == "medium":
        parts.append("Polypharmacy risk MEDIUM -- monitoring contracts must all be active.")
    elif severity == "low":
        parts.append("Polypharmacy risk LOW -- note any monitoring contracts in discharge plan.")
    else:
        parts.append("No polypharmacy concerns flagged.")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(detect_polypharmacy_concerns)
