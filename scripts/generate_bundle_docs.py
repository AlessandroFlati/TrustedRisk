"""Generate one markdown walkthrough per bundle.

Each page documents:
  - When to use the bundle
  - Typical tool sequence
  - Minimum required inputs
  - Composed output / decision artifact
  - Demo scenario reference (link to e2e showcase)
  - Detailed tool list with rationale
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mcp_server.tools import BUNDLES  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Bundle metadata (hand-authored -- bundles are a curated abstraction)
# ─────────────────────────────────────────────────────────────────────

_BUNDLE_DOCS: dict[str, dict[str, object]] = {
    "core_discharge": {
        "title": "Core Discharge",
        "scope": "Adult inpatient discharge planning -- LACE-calibrated 30-day "
                  "readmission risk + medication safety + grounding + fairness audit + "
                  "counterfactual + lab trends + patient-language counseling.",
        "when_to_use": [
            "End-of-stay decision: discharge_home / home_with_care / SNF / continued_admission?",
            "Outpatient medication review (LACE = 0, follow-up window = 14-30d).",
            "Quality-assurance check on a discharge note (PHI scan + grounding).",
        ],
        "typical_sequence": [
            "compute_readmission_risk(patient_id) -> RiskEstimate (probability + LACE)",
            "compute_decision_utility(outcome_probs, n_monte_carlo=1000) -- when ranking actions",
            "compute_medication_reconciliation(patient_id) -> audit discharge med list",
            "detect_polypharmacy_concerns(medications) -> DDI scan",
            "compute_fairness_audit(risk, demographics) -> subgroup drift",
            "compute_counterfactual_explanation(risk) -> what could change the call",
            "compute_lab_trend_analysis(observations) -> optional, when labs drive concern",
            "compute_discharge_counseling(meds, lace_score, action) -> patient summary",
            "ground_claim(claim_text, patient_id) -> final disposition grounding",
            "detect_phi(text) -> before any external release of free text",
        ],
        "min_inputs": [
            "FHIR patient_id (resolves chart via SHARP context)",
            "Discharge medication list (or rely on FHIR MedicationRequest)",
            "Patient demographics for fairness audit (age, race, insurance)",
        ],
        "composed_output": "DecisionCard with recommendation + reasoning + validation + abstain triggers + audit block.",
        "demo_scenarios": ["A (75y CHF discharge)", "C (decision-utility ranking)",
                            "D (batch HTTP)", "H (geriatric outpatient med review)"],
    },
    "ed_acute": {
        "title": "Emergency Department + Acute Deterioration",
        "scope": "ED triage and inpatient deterioration. Maps chief complaint + "
                  "vitals to ESI level + disposition; computes NEWS2 with trend "
                  "delta for ward-based deterioration.",
        "when_to_use": [
            "ED arrival: triage chief complaint + vitals -> ESI 1-5 + disposition.",
            "Ward deterioration: NEWS2 score with trend vs prior shift.",
            "Sepsis early warning (NEWS2 + lactate trend).",
        ],
        "typical_sequence": [
            "detect_phi(triage_note) -- scrub PHI before downstream processing",
            "compute_admission_triage(chief_complaint, vitals, age) -> ESI + disposition",
            "compute_clinical_deterioration_score(vitals, prior_score) -- NEWS2 + trend",
            "compute_lab_trend_analysis(observations) -- lactate / WBC / Cr trends",
            "ground_claim(claim_text, patient_id) -- guideline citation",
        ],
        "min_inputs": [
            "Chief complaint (free text)",
            "Vital signs: HR, RR, SpO2, BP, T, AVPU/GCS",
            "Patient age",
        ],
        "composed_output": "AdmissionTriage + DeteriorationReport, both with deterministic safety gate (resuscitation_room / admit_inpatient / etc.).",
        "demo_scenarios": ["E (ED chest pain ESI 1)", "F (sepsis early warning ward)"],
    },
    "pediatric": {
        "title": "Pediatric",
        "scope": "Age-band-aware pediatric assessment. Pediatric vitals are "
                  "NOT just adult vitals scaled -- both normal ranges and "
                  "escalation thresholds shift with age.",
        "when_to_use": [
            "Pediatric ED triage with PEWS (Brighton/Monaghan).",
            "Weight-based pediatric medication dosing (Broselow-tape style + adult cap).",
            "Verifying age contraindications (ibuprofen <6mo, ceftriaxone <28d, etc.).",
        ],
        "typical_sequence": [
            "compute_pediatric_early_warning(age_months, vitals) -> PEWS + escalation tier",
            "compute_weight_based_dosing(drug, weight_kg, age_months, indication) -> dose + contraindications",
        ],
        "min_inputs": [
            "Age in months (drives age-band selection)",
            "Weight in kg (for dosing)",
            "Vital signs OR drug name + indication",
        ],
        "composed_output": "PEWSReport + PediatricDoseRecommendation, the latter with abstain on contraindications.",
        "demo_scenarios": ["I (4-month-old fever + lethargy)"],
    },
    "mental_health": {
        "title": "Mental Health Crisis",
        "scope": "Suicide risk + psychiatric admission with DEMOGRAPHIC BIAS "
                  "GUARD that abstains for groups (Black, Indigenous, LGBTQ+, "
                  "low-SES) where the literature documents under-prediction.",
        "when_to_use": [
            "ED psychiatric consult: suicide ideation + behavior history -> risk band.",
            "Disposition: outpatient / crisis stabilization / voluntary / involuntary hold.",
            "Free-text mental-health note PHI scrub before billing/research.",
        ],
        "typical_sequence": [
            "compute_suicide_risk_assessment(ideation, behavior, factors, demographics) -- C-SSRS-derived",
            "compute_psychiatric_admission_decision(risk_level, danger_to_*, capacity) -- 3-prong test",
            "compute_fairness_audit(risk, demographics) -- minoritized-group bias guard",
            "detect_phi(text) -- before any external release",
        ],
        "min_inputs": [
            "C-SSRS ideation + behavior (5-level + lifetime/30d)",
            "Danger-to-self / danger-to-others / grave-disability flags",
            "Voluntary capacity assessment",
            "Patient demographics (mandatory for bias guard)",
        ],
        "composed_output": "SuicideRiskAssessment + PsychiatricAdmission with state-specific legal-basis text (CO M-1 / CA 5150 / NY MHL §9.39 / FL Baker Act / IL).",
        "demo_scenarios": ["J (28y M imminent risk + bias guard)"],
    },
    "antimicrobial": {
        "title": "Antimicrobial Stewardship",
        "scope": "Empiric antibiotic selection with local antibiogram + "
                  "de-escalation with pathogen-driven narrowing + IDSA \"5 Cs\" "
                  "IV-to-PO switch criteria.",
        "when_to_use": [
            "Sepsis admission: pick empiric coverage by source + severity.",
            "Day-3 stewardship review: cultures back, narrow regimen?",
            "IV-to-PO switch decision when patient improves.",
        ],
        "typical_sequence": [
            "compute_empiric_antibiotic_selection(infection_source, severity, factors, antibiogram) -- initial regimen",
            "compute_antibiotic_de_escalation(current_regimen, pathogen, susceptibility, clinical_factors) -- narrow + switch",
            "ground_claim(claim_text) -- cite IDSA / ATS / SHEA guidelines",
        ],
        "min_inputs": [
            "Infection source (urinary / pneumonia / SSTI / etc.)",
            "Severity (uncomplicated / complicated / sepsis / septic_shock)",
            "Patient factors (allergies, eGFR, MRSA risk, ESBL history)",
            "Local antibiogram (optional but high-value)",
            "For de-escalation: pathogen + susceptibility S/I/R per drug",
        ],
        "composed_output": "AntibioticSelection (ranked options) + DeEscalationPlan (target regimen + duration).",
        "demo_scenarios": ["K (complicated UTI + ESBL)"],
    },
    "oncology": {
        "title": "Oncology Cycle Decisions",
        "scope": "Chemotherapy cycle decision (proceed/reduce/delay/hold/discontinue) "
                  "based on hematologic + renal + hepatic + ECOG gates; RECIST 1.1 "
                  "response + decision implication.",
        "when_to_use": [
            "Day-of-cycle review: proceed full / reduced / delay / hold?",
            "Restaging review: RECIST 1.1 categorical response + next-line decision.",
            "Treatment selection (e.g. T2D second-line) for cancer survivors.",
        ],
        "typical_sequence": [
            "compute_chemo_dose_adjustment(regimen, cycle, labs, ECOG) -- cycle decision",
            "compute_oncology_treatment_response(target_lesions, new_lesions, ...) -- RECIST 1.1",
            "compute_treatment_selection(condition, factors) -- when alternative regimens needed",
            "compute_lab_trend_analysis(observations) -- ANC / platelet / Cr trends",
        ],
        "min_inputs": [
            "Regimen name (slug)",
            "Cycle number",
            "Labs: ANC, platelets, eGFR, bilirubin, AST/ALT",
            "ECOG performance status",
            "For RECIST: target lesion baseline + current diameters + new-lesion / non-target flags",
        ],
        "composed_output": "ChemoDoseAdjustment (decision + reduction% + growth-factor flag) + TumorResponse (CR/PR/SD/PD/NE + decision implication).",
        "demo_scenarios": ["L (NSCLC cycle 4 with neutropenia)"],
    },
    "stroke_acs": {
        "title": "Stroke + ACS",
        "scope": "Acute stroke severity (NIHSS) + reperfusion eligibility "
                  "(IV tPA + EVT, AHA/ASA 2019 + DAWN/DEFUSE 3); chest pain "
                  "(HEART score + ACS pathway).",
        "when_to_use": [
            "Acute stroke activation: NIHSS + tPA inclusion/exclusion + EVT eligibility.",
            "Late-window stroke (6-24h): DAWN/DEFUSE 3 imaging mismatch decision.",
            "ED chest pain: HEART score + disposition (discharge / observe / admit / cath).",
        ],
        "typical_sequence": [
            "compute_stroke_severity(item_scores, last_known_well_minutes_ago) -- NIHSS + LVO indicator",
            "compute_stroke_thrombolysis_eligibility(...) -- tPA + EVT inclusion/exclusion",
            "compute_heart_score(history, ECG, age, RFs, troponin) -- chest pain risk",
            "compute_acs_disposition_decision(heart_score, STEMI, dynamic_trop, ...) -- disposition",
            "ground_claim(claim_text) -- AHA/ASA citation",
        ],
        "min_inputs": [
            "Stroke: 15 NIHSS items + last-known-well + imaging findings + labs (INR, plt, glucose, BP)",
            "ACS: history descriptor + ECG descriptor + age + risk-factor count + troponin × ULN + STEMI flag",
        ],
        "composed_output": "NIHSSReport + ThrombolysisDecision (iv_tpa_only / evt_only / iv_tpa_plus_evt / no_reperfusion / abstain); HEARTScore + ACSDisposition.",
        "demo_scenarios": ["M (acute stroke 70y M)", "N (chest pain HEART 4)"],
    },
    "obstetric_geriatric": {
        "title": "Obstetric + Geriatric",
        "scope": "Maternal early warning (MEOWS with pregnancy-adjusted vitals) + "
                  "ACOG preeclampsia classification; geriatric falls risk (Morse) + "
                  "delirium screening (CAM with Beers deliriogenic-medication flagging).",
        "when_to_use": [
            "Labor & delivery triage: MEOWS + preeclampsia classification.",
            "Inpatient geriatric: delirium screening + falls precautions.",
            "Outpatient med review with falls / delirium concerns.",
        ],
        "typical_sequence": [
            "compute_maternal_early_warning(GA_weeks, vitals, obstetric_red_flags) -- MEOWS",
            "compute_preeclampsia_assessment(GA, BP, proteinuria, factors) -- ACOG 2020",
            "compute_falls_risk_morse(...) -- Morse + Beers medication flagging",
            "compute_delirium_screening_cam(...) -- CAM + DELIRIUM mnemonic",
            "compute_fairness_audit(risk, demographics) -- maternal mortality bias",
        ],
        "min_inputs": [
            "Maternal: gestational age, BP, vitals, severe-feature flags",
            "Geriatric: falls history, ambulatory aid, gait, mental status, current medications",
        ],
        "composed_output": "MEOWSReport + PreeclampsiaReport (classification + delivery + Mg + antihtn); MorseFallsReport + CAMReport.",
        "demo_scenarios": ["O (32-week severe preeclampsia)", "P (85y delirium + falls)"],
    },
    "trauma_critical": {
        "title": "Trauma Critical Care",
        "scope": "Composite trauma severity (ISS + T-RTS) + ABC-driven massive "
                  "transfusion protocol (1:1:1 ratio + TXA window) + blunt-trauma "
                  "imaging (FAST + CT).",
        "when_to_use": [
            "Trauma team activation: rank severity, decide OR vs ICU vs ward.",
            "Hemorrhagic shock: ABC score -> MTP activation + initial product request.",
            "TXA decision (≤180 min from injury per CRASH-2).",
        ],
        "typical_sequence": [
            "compute_trauma_severity_score(injuries, GCS, SBP, RR) -- ISS + T-RTS + triage priority",
            "compute_massive_transfusion_protocol(ABC items, EBL, minutes_since_injury) -- MTP activation + TXA",
            "compute_imaging_appropriateness(blunt_abdominal_trauma, ...) -- FAST vs CT",
            "compute_clinical_deterioration_score(vitals) -- NEWS2 for ICU disposition",
        ],
        "min_inputs": [
            "Injury list with body region + AIS severity (1-6)",
            "GCS / SBP / RR for T-RTS",
            "Active hemorrhage flag",
            "ABC items: penetrating, SBP≤90, HR≥120, FAST positive",
            "Minutes since injury (TXA gate)",
        ],
        "composed_output": "TraumaSeverityReport (ISS / RTS / triage_priority) + MTPDecision (ABC + activation + 1:1:1 plan + TXA flag).",
        "demo_scenarios": ["Q (polytrauma cross-bundle)"],
    },
    "endocrine_acute": {
        "title": "Endocrine Acute",
        "scope": "DKA severity (ADA 2009 / ISPAD) + initial fluid + insulin + "
                  "K+ replacement protocol; inpatient glycemic control (basal-bolus "
                  "titration + hypoglycemia risk assessment).",
        "when_to_use": [
            "Admission for DKA: severity tier + ICU? + protocol.",
            "Inpatient hyperglycemia: titrate basal-bolus + correctional scale.",
            "Hypoglycemia review: pull basal down, switch from sliding-scale-only.",
        ],
        "typical_sequence": [
            "compute_dka_severity(pH, HCO3, glucose, mental_status, K+, weight) -- severity + protocol",
            "compute_inpatient_glycemic_control(is_icu, avg_glucose, episodes, regimen, factors) -- titration",
        ],
        "min_inputs": [
            "DKA: pH, bicarbonate, glucose, ketones, mental status, K+, weight",
            "Glycemic: is_icu, avg_glucose_24h, hypoglycemia/hyperglycemia episode counts, current regimen, hypoglycemia risk factors",
        ],
        "composed_output": "DKASeverityReport (severity tier + ICU + fluid/insulin/K+ plan + bicarb gate) + InpatientGlycemicPlan (basal change% + correctional scale + hypoglycemia risk).",
        "demo_scenarios": ["R (DKA + AKI cross-bundle)"],
    },
    "imaging": {
        "title": "Imaging Appropriateness + Contrast Safety",
        "scope": "ACR Appropriateness Criteria for common ED scenarios + "
                  "Choosing Wisely lens; ACR Manual on Contrast Media for "
                  "iodinated / gadolinium safety gating (eGFR / metformin / "
                  "allergy / pregnancy).",
        "when_to_use": [
            "Pre-imaging order: which modality is most appropriate for this scenario?",
            "Pre-contrast: is iodinated/gadolinium contrast safe given eGFR / metformin / allergy / pregnancy?",
            "Pediatric ALARA + cumulative-radiation review.",
        ],
        "typical_sequence": [
            "compute_imaging_appropriateness(clinical_scenario, age, pregnant, cumulative_radiation) -- ranked options",
            "compute_contrast_safety_check(contrast_type, eGFR, on_metformin, allergy, pregnant) -- proceed/abstain + premedication",
        ],
        "min_inputs": [
            "Clinical scenario slug (chest_pain_acs_workup, suspected_pe_high_risk, etc.)",
            "Patient age (drives pediatric ALARA)",
            "Pregnancy flag",
            "For contrast: eGFR, metformin use, prior reaction history, pregnancy",
        ],
        "composed_output": "ImagingAppropriateness (ranked modalities with appropriateness 1-9 + radiation dose) + ContrastSafetyReport (proceed flag + metformin hold + premedication).",
        "demo_scenarios": ["R (DKA + AKI + contrast gate cross-bundle)"],
    },
    "nephrology": {
        "title": "Nephrology",
        "scope": "KDIGO 2012 AKI staging (creatinine + UOP criteria + etiology "
                  "clue) + AEIOU dialysis initiation + STARRT-AKI logic for "
                  "stage-3 AKI without emergent indication.",
        "when_to_use": [
            "Hospitalized creatinine rise: stage AKI per KDIGO criteria.",
            "Emergent dialysis decision: AEIOU mnemonic + modality choice (HD vs CRRT).",
            "Pre-contrast eGFR check (composes with imaging bundle).",
        ],
        "typical_sequence": [
            "compute_aki_kdigo_stage(Cr_baseline, Cr_current, UOP, FENa, hydronephrosis) -- stage + etiology",
            "compute_dialysis_initiation_decision(aki_stage, pH, K+, refractory_*, toxin, hemodynamic) -- AEIOU + modality",
            "compute_lab_trend_analysis(observations) -- creatinine / K+ trends",
            "compute_contrast_safety_check(contrast_type, eGFR) -- pre-imaging gate",
        ],
        "min_inputs": [
            "Creatinine baseline + current",
            "Urine output (mL/kg/h × window hours)",
            "For dialysis: pH, K+, refractory flags, dialyzable toxin, hemodynamic stability",
        ],
        "composed_output": "AKIStagingReport (stage + etiology + nephrology consult flag) + DialysisInitiationReport (AEIOU indications + urgency + modality).",
        "demo_scenarios": ["R (DKA + AKI cross-bundle)"],
    },
}


# ─────────────────────────────────────────────────────────────────────
# Page renderer
# ─────────────────────────────────────────────────────────────────────

def render_page(bundle_id: str, meta: dict[str, object], tools: list[str]) -> str:
    parts: list[str] = []
    parts.append(f"# Bundle: `{bundle_id}` -- {meta['title']}\n\n")
    parts.append(f"{meta['scope']}\n\n")
    parts.append("## When to use\n\n")
    for w in meta["when_to_use"]:  # type: ignore[union-attr]
        parts.append(f"- {w}\n")
    parts.append("\n## Typical tool sequence\n\n")
    for s in meta["typical_sequence"]:  # type: ignore[union-attr]
        parts.append(f"1. {s}\n")
    parts.append("\n## Minimum inputs\n\n")
    for m in meta["min_inputs"]:  # type: ignore[union-attr]
        parts.append(f"- {m}\n")
    parts.append("\n## Composed output\n\n")
    parts.append(f"{meta['composed_output']}\n\n")
    parts.append("## Demo scenarios\n\n")
    for d in meta["demo_scenarios"]:  # type: ignore[union-attr]
        parts.append(f"- {d}\n")
    parts.append("\n## Tools in this bundle\n\n")
    for t in tools:
        parts.append(f"- `{t}`\n")
    parts.append("\n---\n\n")
    parts.append(f"_Bundle id_: `{bundle_id}` &middot; _Tools_: {len(tools)} &middot; ")
    parts.append("_Auto-generated from `mcp_server.tools.BUNDLES`_.\n")
    return "".join(parts)


def main() -> int:
    out_dir = ROOT / "docs" / "bundles"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for bundle_id, tools in BUNDLES.items():
        meta = _BUNDLE_DOCS.get(bundle_id)
        if meta is None:
            print(f"[skip] no metadata for bundle {bundle_id!r}")
            continue
        body = render_page(bundle_id, meta, tools)
        path = out_dir / f"{bundle_id}.md"
        path.write_text(body, encoding="utf-8")
        written += 1
        print(f"[ok]   wrote {path.relative_to(ROOT)} ({len(body)} chars, {len(tools)} tools)")
    print(f"\nWrote {written} bundle docs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
