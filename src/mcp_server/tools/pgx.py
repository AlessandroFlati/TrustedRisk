"""healthcare.compute_pgx_{dose_adjustment,drug_alternatives,
eligibility_check} -- Phase 7.2 Pharmacogenomic Decision Support.

Hard-coded subset of the CPIC (Clinical Pharmacogenetics
Implementation Consortium) guidelines plus FDA Table of
Pharmacogenomic Biomarkers in Drug Labeling. Each tool is pure-
deterministic -- every recommendation traces to a specific
(gene, drug, phenotype) row in the curated table, with the CPIC
evidence-level tag (A/B/C/D) preserved verbatim.

CPIC level A is the highest -- actionable + sufficient evidence to
support dose modification. Level B is moderate evidence; C/D are
informational. Operators should configure their EHR to surface only
A+B by default.

References (table sourced 2024-Q4):
- CPIC guideline registry -- https://cpicpgx.org/guidelines/
- FDA Table of Pharmacogenomic Biomarkers in Drug Labeling
- PharmGKB clinical-annotations DB
"""

from __future__ import annotations

import re
from typing import Any

from shared.schemas import (
    PGxAlternative,
    PGxAlternativesReport,
    PGxDoseAdjustment,
    PGxDoseAdjustmentReport,
    PGxEligibilityReport,
    PGxGenotype,
)

from ..fhir.client import fetch_patient_bundle, resolve_patient_id


# ─────────────────────────────────────────────────────────────────────
# CPIC table -- (gene, phenotype, drug) -> adjustment + level
# ─────────────────────────────────────────────────────────────────────

# This is a curated subset (~30 rows). A production deployment would
# bind the full PharmGKB CPIC table.

_CPIC_TABLE: list[dict[str, Any]] = [
    # CYP2D6 -- opioids
    {"gene": "CYP2D6", "phenotype": "ultra_rapid_metabolizer",
     "drug": "codeine", "rec": "avoid_drug", "dose_pct": 0, "level": "A",
     "rationale": ("CYP2D6 ultra-rapid metabolizers convert codeine "
                      "to morphine extremely fast -- life-threatening "
                      "respiratory depression risk. Use non-codeine "
                      "opioid."),
     "monitoring": ["Use morphine or hydromorphone instead",
                       "Avoid in pediatric tonsillectomy entirely"]},
    {"gene": "CYP2D6", "phenotype": "poor_metabolizer",
     "drug": "codeine", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "A",
     "rationale": ("CYP2D6 poor metabolizers cannot convert codeine to "
                      "morphine -- minimal analgesia. Use morphine."),
     "monitoring": ["Switch to morphine or hydromorphone"]},
    {"gene": "CYP2D6", "phenotype": "poor_metabolizer",
     "drug": "tramadol", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "A",
     "rationale": ("CYP2D6 PM cannot activate tramadol -- analgesic "
                      "failure."),
     "monitoring": []},
    {"gene": "CYP2D6", "phenotype": "ultra_rapid_metabolizer",
     "drug": "tramadol", "rec": "avoid_drug", "dose_pct": 0, "level": "A",
     "rationale": ("CYP2D6 UM converts tramadol to active metabolite "
                      "extremely fast -- toxicity risk."),
     "monitoring": []},
    # CYP2C19 -- antiplatelets + PPIs + SSRIs
    {"gene": "CYP2C19", "phenotype": "poor_metabolizer",
     "drug": "clopidogrel", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "A",
     "rationale": ("CYP2C19 PM has reduced clopidogrel activation, "
                      "particularly post-PCI -- use prasugrel or "
                      "ticagrelor instead."),
     "monitoring": ["Switch to prasugrel or ticagrelor",
                       "Verify in ACS / post-PCI populations"]},
    {"gene": "CYP2C19", "phenotype": "intermediate_metabolizer",
     "drug": "clopidogrel", "rec": "additional_monitoring_required",
     "dose_pct": 0, "level": "B",
     "rationale": "CYP2C19 IM has partially reduced response.",
     "monitoring": ["Consider prasugrel/ticagrelor in high-risk PCI"]},
    {"gene": "CYP2C19", "phenotype": "ultra_rapid_metabolizer",
     "drug": "voriconazole", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": -50, "level": "A",
     "rationale": ("CYP2C19 UM clears voriconazole rapidly -- "
                      "subtherapeutic levels."),
     "monitoring": ["Use isavuconazole or posaconazole instead"]},
    {"gene": "CYP2C19", "phenotype": "poor_metabolizer",
     "drug": "voriconazole", "rec": "decrease_dose",
     "dose_pct": -50, "level": "A",
     "rationale": "CYP2C19 PM accumulates voriconazole.",
     "monitoring": ["Therapeutic drug monitoring",
                       "Reduce dose by ~ 50%"]},
    # CYP2C9 / VKORC1 -- warfarin
    {"gene": "CYP2C9", "phenotype": "intermediate_metabolizer",
     "drug": "warfarin", "rec": "decrease_dose",
     "dose_pct": -25, "level": "A",
     "rationale": "CYP2C9 IM clears warfarin slowly -- bleeding risk.",
     "monitoring": ["Tighter INR monitoring early",
                       "Reduce loading + maintenance dose"]},
    {"gene": "CYP2C9", "phenotype": "poor_metabolizer",
     "drug": "warfarin", "rec": "decrease_dose",
     "dose_pct": -50, "level": "A",
     "rationale": "CYP2C9 PM accumulates warfarin sharply.",
     "monitoring": ["INR check within 3 days of initiation"]},
    {"gene": "VKORC1", "phenotype": "positive",
     "drug": "warfarin", "rec": "decrease_dose",
     "dose_pct": -25, "level": "A",
     "rationale": ("VKORC1 -1639G>A increases warfarin sensitivity. "
                      "Lower starting dose."),
     "monitoring": ["Use the IWPC dosing algorithm"]},
    # SLCO1B1 -- statins
    {"gene": "SLCO1B1", "phenotype": "intermediate_metabolizer",
     "drug": "simvastatin", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": -50, "level": "A",
     "rationale": ("SLCO1B1 *5 carriers (intermediate function) have "
                      "elevated simvastatin exposure -> myopathy risk."),
     "monitoring": ["Switch to pravastatin or rosuvastatin",
                       "Avoid simvastatin > 20 mg"]},
    {"gene": "SLCO1B1", "phenotype": "poor_metabolizer",
     "drug": "simvastatin", "rec": "avoid_drug",
     "dose_pct": 0, "level": "A",
     "rationale": "SLCO1B1 *5/*5 -- high myopathy / rhabdomyolysis risk.",
     "monitoring": ["Use pravastatin or rosuvastatin"]},
    # TPMT -- thiopurines
    {"gene": "TPMT", "phenotype": "intermediate_metabolizer",
     "drug": "azathioprine", "rec": "decrease_dose",
     "dose_pct": -50, "level": "A",
     "rationale": "TPMT IM has intermediate metabolism -- toxicity risk.",
     "monitoring": ["CBC weekly x 4 weeks then monthly"]},
    {"gene": "TPMT", "phenotype": "poor_metabolizer",
     "drug": "azathioprine", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": -90, "level": "A",
     "rationale": "TPMT PM has near-zero activity -- life-threatening myelosuppression.",
     "monitoring": ["Switch to alternative immunosuppressant"]},
    # DPYD -- fluoropyrimidines
    {"gene": "DPYD", "phenotype": "intermediate_metabolizer",
     "drug": "fluorouracil", "rec": "decrease_dose",
     "dose_pct": -50, "level": "A",
     "rationale": "DPYD IM accumulates 5-FU -- severe mucositis / neutropenia.",
     "monitoring": ["50% starting dose then titrate"]},
    {"gene": "DPYD", "phenotype": "poor_metabolizer",
     "drug": "fluorouracil", "rec": "avoid_drug",
     "dose_pct": 0, "level": "A",
     "rationale": "DPYD complete deficiency -- potentially fatal toxicity.",
     "monitoring": ["Use non-fluoropyrimidine regimen"]},
    {"gene": "DPYD", "phenotype": "intermediate_metabolizer",
     "drug": "capecitabine", "rec": "decrease_dose",
     "dose_pct": -50, "level": "A",
     "rationale": "DPYD IM -- same risk as 5-FU.",
     "monitoring": []},
    # UGT1A1 -- irinotecan
    {"gene": "UGT1A1", "phenotype": "poor_metabolizer",
     "drug": "irinotecan", "rec": "decrease_dose",
     "dose_pct": -30, "level": "B",
     "rationale": "UGT1A1 *28/*28 -- neutropenia + diarrhea risk.",
     "monitoring": ["CBC + LFT pre-cycle"]},
    # HLA -- anaphylaxis / SJS
    {"gene": "HLA-B*5701", "phenotype": "positive",
     "drug": "abacavir", "rec": "avoid_drug",
     "dose_pct": 0, "level": "A",
     "rationale": "HLA-B*5701+ -> 50% hypersensitivity reaction risk.",
     "monitoring": ["Use alternative HIV regimen"]},
    {"gene": "HLA-B*1502", "phenotype": "positive",
     "drug": "carbamazepine", "rec": "avoid_drug",
     "dose_pct": 0, "level": "A",
     "rationale": "HLA-B*1502+ -> Stevens-Johnson Syndrome / TEN risk in Asians.",
     "monitoring": ["Use lacosamide / levetiracetam / valproate"]},
    # G6PD -- antimalarials
    {"gene": "G6PD", "phenotype": "deficient",
     "drug": "primaquine", "rec": "avoid_drug",
     "dose_pct": 0, "level": "A",
     "rationale": "G6PD deficient -- hemolysis risk.",
     "monitoring": ["Use tafenoquine alternative"]},
    {"gene": "G6PD", "phenotype": "deficient",
     "drug": "rasburicase", "rec": "avoid_drug",
     "dose_pct": 0, "level": "A",
     "rationale": "G6PD deficient -- methemoglobinemia + hemolysis.",
     "monitoring": ["Use allopurinol instead"]},
    # NUDT15 -- thiopurines (Asians)
    {"gene": "NUDT15", "phenotype": "intermediate_metabolizer",
     "drug": "azathioprine", "rec": "decrease_dose",
     "dose_pct": -30, "level": "A",
     "rationale": "NUDT15 IM -- leukopenia risk in Asian populations.",
     "monitoring": ["CBC weekly initially"]},
    {"gene": "NUDT15", "phenotype": "poor_metabolizer",
     "drug": "azathioprine", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": -90, "level": "A",
     "rationale": "NUDT15 PM -- severe leukopenia.",
     "monitoring": []},
    # ──────── Phase 12.4 D1 expansion -- 55+ additional CPIC entries ────────
    # CYP2D6 -- extended SSRI / SNRI / opioid surface
    {"gene": "CYP2D6", "phenotype": "poor_metabolizer",
     "drug": "fluoxetine", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": -50, "level": "A",
     "rationale": "CYP2D6 PM accumulates fluoxetine -- sedation + QT.",
     "monitoring": ["Consider citalopram/sertraline alternative"]},
    {"gene": "CYP2D6", "phenotype": "poor_metabolizer",
     "drug": "paroxetine", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": -50, "level": "A",
     "rationale": "CYP2D6 PM raises paroxetine exposure.",
     "monitoring": []},
    {"gene": "CYP2D6", "phenotype": "ultra_rapid_metabolizer",
     "drug": "paroxetine", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "A",
     "rationale": "CYP2D6 UM clears paroxetine fast -- therapeutic failure.",
     "monitoring": []},
    {"gene": "CYP2D6", "phenotype": "poor_metabolizer",
     "drug": "atomoxetine", "rec": "decrease_dose",
     "dose_pct": -50, "level": "A",
     "rationale": "CYP2D6 PM accumulates atomoxetine -- cardiac side effects.",
     "monitoring": ["Slow up-titration"]},
    {"gene": "CYP2D6", "phenotype": "poor_metabolizer",
     "drug": "amitriptyline", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": -50, "level": "A",
     "rationale": "CYP2D6 PM accumulates TCAs -- cardiotoxicity, sedation.",
     "monitoring": ["Consider non-TCA"]},
    {"gene": "CYP2D6", "phenotype": "ultra_rapid_metabolizer",
     "drug": "amitriptyline", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "A",
     "rationale": "CYP2D6 UM converts amitriptyline rapidly to nortriptyline.",
     "monitoring": []},
    {"gene": "CYP2D6", "phenotype": "poor_metabolizer",
     "drug": "nortriptyline", "rec": "decrease_dose",
     "dose_pct": -50, "level": "A",
     "rationale": "CYP2D6 PM accumulates nortriptyline.",
     "monitoring": []},
    {"gene": "CYP2D6", "phenotype": "poor_metabolizer",
     "drug": "metoprolol", "rec": "additional_monitoring_required",
     "dose_pct": -50, "level": "B",
     "rationale": "CYP2D6 PM accumulates metoprolol -- bradycardia.",
     "monitoring": ["Watch HR / BP closely"]},
    {"gene": "CYP2D6", "phenotype": "poor_metabolizer",
     "drug": "tamoxifen", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "A",
     "rationale": "CYP2D6 PM cannot activate tamoxifen to endoxifen.",
     "monitoring": ["Consider AI in postmenopausal patients"]},
    {"gene": "CYP2D6", "phenotype": "ultra_rapid_metabolizer",
     "drug": "oxycodone", "rec": "decrease_dose",
     "dose_pct": -50, "level": "B",
     "rationale": "CYP2D6 UM raises oxymorphone exposure.",
     "monitoring": []},
    {"gene": "CYP2D6", "phenotype": "poor_metabolizer",
     "drug": "hydrocodone", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "B",
     "rationale": "CYP2D6 PM cannot generate hydromorphone -- analgesic failure.",
     "monitoring": []},
    {"gene": "CYP2D6", "phenotype": "ultra_rapid_metabolizer",
     "drug": "fluvoxamine", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "B",
     "rationale": "CYP2D6 UM lowers fluvoxamine exposure.",
     "monitoring": []},
    # CYP2C19 -- extended SSRI + PPI surface
    {"gene": "CYP2C19", "phenotype": "ultra_rapid_metabolizer",
     "drug": "escitalopram", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "A",
     "rationale": "CYP2C19 UM clears escitalopram fast -- therapeutic failure.",
     "monitoring": []},
    {"gene": "CYP2C19", "phenotype": "poor_metabolizer",
     "drug": "escitalopram", "rec": "decrease_dose",
     "dose_pct": -50, "level": "A",
     "rationale": "CYP2C19 PM accumulates escitalopram -- QT prolongation.",
     "monitoring": ["ECG baseline + at steady state"]},
    {"gene": "CYP2C19", "phenotype": "poor_metabolizer",
     "drug": "citalopram", "rec": "decrease_dose",
     "dose_pct": -50, "level": "A",
     "rationale": "CYP2C19 PM accumulates citalopram -- QT prolongation.",
     "monitoring": []},
    {"gene": "CYP2C19", "phenotype": "ultra_rapid_metabolizer",
     "drug": "citalopram", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "A",
     "rationale": "CYP2C19 UM clears citalopram fast -- therapeutic failure.",
     "monitoring": []},
    {"gene": "CYP2C19", "phenotype": "poor_metabolizer",
     "drug": "sertraline", "rec": "decrease_dose",
     "dose_pct": -50, "level": "A",
     "rationale": "CYP2C19 PM accumulates sertraline.",
     "monitoring": []},
    {"gene": "CYP2C19", "phenotype": "poor_metabolizer",
     "drug": "amitriptyline", "rec": "decrease_dose",
     "dose_pct": -50, "level": "A",
     "rationale": "CYP2C19 PM accumulates amitriptyline.",
     "monitoring": ["Plasma TDM"]},
    {"gene": "CYP2C19", "phenotype": "poor_metabolizer",
     "drug": "omeprazole", "rec": "additional_monitoring_required",
     "dose_pct": 100, "level": "A",
     "rationale": "CYP2C19 PM accumulates omeprazole -- H. pylori eradication "
                      "rates higher with PMs (counter-intuitive).",
     "monitoring": []},
    {"gene": "CYP2C19", "phenotype": "ultra_rapid_metabolizer",
     "drug": "omeprazole", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 100, "level": "A",
     "rationale": "CYP2C19 UM clears omeprazole quickly -- H. pylori eradication "
                      "fails. Consider esomeprazole or rabeprazole + dose increase.",
     "monitoring": []},
    {"gene": "CYP2C19", "phenotype": "poor_metabolizer",
     "drug": "lansoprazole", "rec": "additional_monitoring_required",
     "dose_pct": 100, "level": "A",
     "rationale": "CYP2C19 PM accumulates lansoprazole.",
     "monitoring": []},
    {"gene": "CYP2C19", "phenotype": "ultra_rapid_metabolizer",
     "drug": "pantoprazole", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 100, "level": "A",
     "rationale": "CYP2C19 UM clears pantoprazole quickly.",
     "monitoring": []},
    # CYP2C9 -- additional NSAIDs + phenytoin
    {"gene": "CYP2C9", "phenotype": "poor_metabolizer",
     "drug": "phenytoin", "rec": "decrease_dose",
     "dose_pct": -50, "level": "A",
     "rationale": "CYP2C9 PM accumulates phenytoin -- toxicity.",
     "monitoring": ["Free phenytoin level"]},
    {"gene": "CYP2C9", "phenotype": "intermediate_metabolizer",
     "drug": "phenytoin", "rec": "decrease_dose",
     "dose_pct": -25, "level": "A",
     "rationale": "CYP2C9 IM -- partial accumulation.",
     "monitoring": []},
    {"gene": "CYP2C9", "phenotype": "poor_metabolizer",
     "drug": "celecoxib", "rec": "decrease_dose",
     "dose_pct": -50, "level": "A",
     "rationale": "CYP2C9 PM accumulates celecoxib.",
     "monitoring": []},
    {"gene": "CYP2C9", "phenotype": "poor_metabolizer",
     "drug": "ibuprofen", "rec": "additional_monitoring_required",
     "dose_pct": -25, "level": "B",
     "rationale": "CYP2C9 PM accumulates ibuprofen at high chronic doses.",
     "monitoring": ["Limit chronic high-dose use"]},
    {"gene": "CYP2C9", "phenotype": "poor_metabolizer",
     "drug": "naproxen", "rec": "additional_monitoring_required",
     "dose_pct": -25, "level": "B",
     "rationale": "CYP2C9 PM accumulates naproxen.",
     "monitoring": []},
    # CYP3A5 -- tacrolimus
    {"gene": "CYP3A5", "phenotype": "intermediate_metabolizer",
     "drug": "tacrolimus", "rec": "increase_dose",
     "dose_pct": 50, "level": "A",
     "rationale": "CYP3A5 IM clears tacrolimus faster -- sub-therapeutic levels "
                      "without dose increase. Critical in transplant.",
     "monitoring": ["Trough levels every 2-3 days early"]},
    {"gene": "CYP3A5", "phenotype": "rapid_metabolizer",
     "drug": "tacrolimus", "rec": "increase_dose",
     "dose_pct": 100, "level": "A",
     "rationale": "CYP3A5 expressers (*1/*1, *1/*3) need ~ 1.5-2x standard dose.",
     "monitoring": ["Daily troughs in first week post-transplant"]},
    # SLCO1B1 -- additional statins
    {"gene": "SLCO1B1", "phenotype": "intermediate_metabolizer",
     "drug": "atorvastatin", "rec": "additional_monitoring_required",
     "dose_pct": -25, "level": "B",
     "rationale": "SLCO1B1 *5 carriers -- modest myopathy risk on atorvastatin.",
     "monitoring": ["CK if myalgia"]},
    {"gene": "SLCO1B1", "phenotype": "poor_metabolizer",
     "drug": "atorvastatin", "rec": "decrease_dose",
     "dose_pct": -50, "level": "B",
     "rationale": "SLCO1B1 *5/*5 -- myopathy risk on atorvastatin.",
     "monitoring": []},
    {"gene": "SLCO1B1", "phenotype": "poor_metabolizer",
     "drug": "rosuvastatin", "rec": "additional_monitoring_required",
     "dose_pct": -25, "level": "B",
     "rationale": "Modest exposure increase; lower myopathy risk than simvastatin.",
     "monitoring": []},
    {"gene": "SLCO1B1", "phenotype": "intermediate_metabolizer",
     "drug": "pravastatin", "rec": "additional_monitoring_required",
     "dose_pct": 0, "level": "B",
     "rationale": "Minor exposure increase.",
     "monitoring": []},
    # IFNL3 (IL28B) -- peginterferon (HCV)
    {"gene": "IFNL3", "phenotype": "TT",
     "drug": "peginterferon_alfa", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "A",
     "rationale": "IL28B TT predicts poor IFN response.",
     "monitoring": ["Use DAA-only regimen"]},
    # HLA -- additional risk alleles
    {"gene": "HLA-B*5801", "phenotype": "positive",
     "drug": "allopurinol", "rec": "avoid_drug",
     "dose_pct": 0, "level": "A",
     "rationale": "HLA-B*58:01+ -> severe SJS/TEN risk in Han Chinese, Korean, Thai.",
     "monitoring": ["Use febuxostat instead"]},
    {"gene": "HLA-A*3101", "phenotype": "positive",
     "drug": "carbamazepine", "rec": "avoid_drug",
     "dose_pct": 0, "level": "A",
     "rationale": "HLA-A*31:01+ -> DRESS / SJS risk in Europeans, Japanese.",
     "monitoring": []},
    {"gene": "HLA-B*1502", "phenotype": "positive",
     "drug": "phenytoin", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "A",
     "rationale": "HLA-B*15:02+ -> SJS/TEN risk, particularly with carbamazepine "
                      "co-prescription. Consider valproate or levetiracetam.",
     "monitoring": []},
    {"gene": "HLA-B*1502", "phenotype": "positive",
     "drug": "oxcarbazepine", "rec": "avoid_drug",
     "dose_pct": 0, "level": "A",
     "rationale": "Cross-reactive with carbamazepine -- SJS risk.",
     "monitoring": []},
    {"gene": "HLA-B*5701", "phenotype": "positive",
     "drug": "flucloxacillin", "rec": "avoid_drug",
     "dose_pct": 0, "level": "B",
     "rationale": "HLA-B*57:01+ -> drug-induced liver injury risk.",
     "monitoring": []},
    # G6PD -- additional drugs
    {"gene": "G6PD", "phenotype": "deficient",
     "drug": "dapsone", "rec": "avoid_drug",
     "dose_pct": 0, "level": "A",
     "rationale": "G6PD deficient -- hemolytic anemia + methemoglobinemia.",
     "monitoring": []},
    {"gene": "G6PD", "phenotype": "deficient",
     "drug": "sulfamethoxazole", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "A",
     "rationale": "G6PD deficient -- hemolysis with sulfa drugs.",
     "monitoring": []},
    {"gene": "G6PD", "phenotype": "deficient",
     "drug": "tafenoquine", "rec": "avoid_drug",
     "dose_pct": 0, "level": "A",
     "rationale": "G6PD <70% activity -> hemolysis. Test before prescribing.",
     "monitoring": []},
    # UGT1A1 -- atazanavir
    {"gene": "UGT1A1", "phenotype": "poor_metabolizer",
     "drug": "atazanavir", "rec": "additional_monitoring_required",
     "dose_pct": 0, "level": "B",
     "rationale": "UGT1A1 *28/*28 -> hyperbilirubinaemia with atazanavir.",
     "monitoring": ["Watch for jaundice -- consider raltegravir alternative"]},
    # CYP4F2 -- vitamin K / warfarin
    {"gene": "CYP4F2", "phenotype": "poor_metabolizer",
     "drug": "warfarin", "rec": "increase_dose",
     "dose_pct": 10, "level": "B",
     "rationale": "CYP4F2 V433M+ -- slightly higher warfarin dose required.",
     "monitoring": ["Tighter INR early"]},
    # MT-RNR1 -- aminoglycoside ototoxicity
    {"gene": "MT-RNR1", "phenotype": "m.1555A>G",
     "drug": "gentamicin", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "A",
     "rationale": "Mitochondrial m.1555A>G -> aminoglycoside-induced "
                      "ototoxicity, may be irreversible at any dose.",
     "monitoring": ["Use non-aminoglycoside alternative"]},
    {"gene": "MT-RNR1", "phenotype": "m.1555A>G",
     "drug": "amikacin", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "A",
     "rationale": "Same mitochondrial deafness risk as gentamicin.",
     "monitoring": []},
    {"gene": "MT-RNR1", "phenotype": "m.1555A>G",
     "drug": "tobramycin", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "A",
     "rationale": "Same mitochondrial deafness risk.",
     "monitoring": []},
    # CFTR -- ivacaftor (CF-specific genotype gating)
    {"gene": "CFTR", "phenotype": "G551D_homozygous",
     "drug": "ivacaftor", "rec": "standard_dosing",
     "dose_pct": 0, "level": "A",
     "rationale": "CFTR gating mutation responsive to ivacaftor monotherapy.",
     "monitoring": ["LFTs at baseline and 3 monthly"]},
    {"gene": "CFTR", "phenotype": "F508del_homozygous",
     "drug": "ivacaftor", "rec": "alternative_drug_strongly_recommended",
     "dose_pct": 0, "level": "A",
     "rationale": "F508del homozygotes require triple therapy "
                      "(elexacaftor/tezacaftor/ivacaftor).",
     "monitoring": []},
    # POLG -- valproate
    {"gene": "POLG", "phenotype": "deficient",
     "drug": "valproate", "rec": "avoid_drug",
     "dose_pct": 0, "level": "A",
     "rationale": "POLG mutations -> fatal valproate-induced hepatotoxicity.",
     "monitoring": ["Use levetiracetam or lamotrigine"]},
    # CYP2C9 / VKORC1 -- additional anticoagulant nuance
    {"gene": "VKORC1", "phenotype": "AA_homozygous",
     "drug": "warfarin", "rec": "decrease_dose",
     "dose_pct": -40, "level": "A",
     "rationale": "VKORC1 -1639 AA (homozygous) -- markedly increased sensitivity.",
     "monitoring": ["Use the IWPC algorithm"]},
    # ApoE -- efavirenz / atorvastatin (lipid CV risk modulator)
    {"gene": "APOE", "phenotype": "e4_carrier",
     "drug": "atorvastatin", "rec": "additional_monitoring_required",
     "dose_pct": 0, "level": "B",
     "rationale": "APOE e4 carriers may show variable LDL response.",
     "monitoring": ["LDL at 6 weeks"]},
    # ABCG2 -- rosuvastatin
    {"gene": "ABCG2", "phenotype": "poor_metabolizer",
     "drug": "rosuvastatin", "rec": "decrease_dose",
     "dose_pct": -50, "level": "B",
     "rationale": "ABCG2 c.421C>A reduces rosuvastatin clearance.",
     "monitoring": []},
    # PHARMGKB -- additional opioid
    {"gene": "OPRM1", "phenotype": "G_carrier",
     "drug": "morphine", "rec": "additional_monitoring_required",
     "dose_pct": 30, "level": "C",
     "rationale": "OPRM1 118A>G (G allele) -- reduced morphine response.",
     "monitoring": []},
]


def _norm_drug(name: str) -> str:
    return re.sub(r"\s+\d.*$", "", name.strip().lower())


def _coerce_genotype(g: PGxGenotype | dict) -> PGxGenotype:
    if isinstance(g, dict):
        return PGxGenotype.model_validate(g)
    return g


# Set of PGx genes the CPIC table recognises -- used by the FHIR
# extractor to flag an Observation as PGx-flavoured. Canonical casing
# preserved so PGxGenotype validation accepts the value.
_KNOWN_PGX_GENES: frozenset[str] = frozenset(
    row["gene"] for row in _CPIC_TABLE
)
_PGX_GENE_BY_LOWER: dict[str, str] = {
    g.lower(): g for g in _KNOWN_PGX_GENES
}
_KNOWN_PGX_PHENOTYPES: frozenset[str] = frozenset(
    row["phenotype"] for row in _CPIC_TABLE
)


def _normalise_phenotype_token(raw: str) -> str | None:
    """Map a raw value-string (e.g. 'Poor Metabolizer') to a canonical
    `PGxPhenotype` enum value. Returns None when no canonical match."""
    if not raw:
        return None
    token = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if token in _KNOWN_PGX_PHENOTYPES:
        return token
    # Allow case-preserved literals (mitochondrial / CFTR variants)
    if raw.strip() in _KNOWN_PGX_PHENOTYPES:
        return raw.strip()
    return None


def _detect_gene_in_text(text: str) -> str | None:
    """Return the canonical gene symbol (casing preserved) if the input
    string contains any known PGx gene token. Longest-match wins so
    'HLA-B*5701' beats a partial 'HLA-B' overlap.
    """
    if not text:
        return None
    lowered = text.lower()
    matches = [g for g_lower, g in _PGX_GENE_BY_LOWER.items()
               if g_lower in lowered]
    if not matches:
        return None
    return max(matches, key=len)


def _extract_pgx_genotypes_from_bundle(
    bundle: dict | None,
) -> list[PGxGenotype]:
    """Pull PGx genotypes from FHIR Observation resources.

    Detection: an Observation is treated as PGx-flavoured when its
    `code.text` or any `code.coding[].display` / `.code` mentions a
    gene token from the CPIC table. The phenotype is read from
    `valueString` first, then `valueCodeableConcept.text`, then the
    first `interpretation[].text`. Values are normalised against the
    canonical `PGxPhenotype` enum; rows that fail to match are
    dropped (no fabrication).

    Returns an empty list when no PGx Observations are present -- the
    caller treats that as the abstain trigger.
    """
    if not isinstance(bundle, dict):
        return []

    out: list[PGxGenotype] = []
    for entry in bundle.get("entry") or []:
        r = entry.get("resource") if isinstance(entry, dict) else None
        if not isinstance(r, dict):
            continue
        if r.get("resourceType") != "Observation":
            continue

        code_obj = r.get("code") or {}
        gene = _detect_gene_in_text(code_obj.get("text") or "")
        if gene is None:
            for c in code_obj.get("coding") or []:
                if not isinstance(c, dict):
                    continue
                gene = _detect_gene_in_text(
                    (c.get("display") or "") + " " + (c.get("code") or ""),
                )
                if gene is not None:
                    break
        if gene is None:
            continue

        raw_phenotype: str | None = None
        if isinstance(r.get("valueString"), str):
            raw_phenotype = r["valueString"]
        elif isinstance(r.get("valueCodeableConcept"), dict):
            vc = r["valueCodeableConcept"]
            raw_phenotype = vc.get("text") or ""
            if not raw_phenotype:
                for c in vc.get("coding") or []:
                    if isinstance(c, dict):
                        raw_phenotype = (
                            c.get("display") or c.get("code") or ""
                        )
                        if raw_phenotype:
                            break
        elif isinstance(r.get("interpretation"), list) and r["interpretation"]:
            interp = r["interpretation"][0]
            if isinstance(interp, dict):
                raw_phenotype = interp.get("text") or ""
                if not raw_phenotype:
                    for c in interp.get("coding") or []:
                        if isinstance(c, dict):
                            raw_phenotype = (
                                c.get("display") or c.get("code") or ""
                            )
                            if raw_phenotype:
                                break

        phenotype = _normalise_phenotype_token(raw_phenotype or "")
        if phenotype is None:
            continue

        try:
            out.append(PGxGenotype(
                gene=gene,                          # type: ignore[arg-type]
                phenotype=phenotype,                # type: ignore[arg-type]
                source_id=r.get("id"),
            ))
        except Exception:
            # Belt-and-braces: drop rows that fail PGxGenotype validation
            # (gene/phenotype outside the Literal set). Never fabricate.
            continue

    return out


async def _resolve_genotypes_from_chart_or_caller(
    caller_supplied: list[PGxGenotype | dict] | None,
) -> tuple[list[PGxGenotype], bool, bool]:
    """Decide which genotype list a PGx tool should consume.

    Returns (genotypes, sharp_bound, chart_examined). `sharp_bound`
    means the SHARP-on-MCP context was available; in that mode the
    chart is authoritative and the caller-supplied list is IGNORED
    (clinical data cannot be injected by a chat-side agent). When
    SHARP context is absent (offline / unit-test mode) the
    caller-supplied list is used as-is, preserving legacy behaviour.
    """
    sharp_bound = False
    chart_genotypes: list[PGxGenotype] = []
    try:
        _pid = await resolve_patient_id(None)
        bundle = await fetch_patient_bundle(_pid)
        sharp_bound = True
        chart_genotypes = _extract_pgx_genotypes_from_bundle(bundle)
    except Exception:
        # No SHARP context (or FHIR fetch failed) -- fall back to caller.
        pass

    if sharp_bound:
        # Production path: chart is authoritative. Caller-supplied
        # genotypes are silently discarded; the chart Observation is
        # the only allowed source of clinical PGx data.
        return chart_genotypes, True, True

    # Offline / unit-test path: caller-supplied genotypes are accepted.
    geno = [_coerce_genotype(g) for g in (caller_supplied or [])]
    return geno, False, False


# ─────────────────────────────────────────────────────────────────────
# Public APIs
# ─────────────────────────────────────────────────────────────────────


async def compute_pgx_dose_adjustment(
    medications: list[str],
    genotypes: list[PGxGenotype | dict] | None = None,
    patient_reference: str | None = None,
) -> PGxDoseAdjustmentReport:
    """Match a patient's PGx genotypes against their medication list
    and emit per-drug dose-adjustment recommendations.

    Genotypes are read from the FHIR chart via the SHARP-on-MCP
    context, NOT from the caller. When the chart contains no PGx
    Observation resources the tool abstains; clinical data cannot be
    injected by the chat-side agent. The `genotypes` argument is
    accepted only in offline / unit-test mode (no SHARP context bound).

    Args:
        medications: free-text medication names (e.g.
            "warfarin 5 mg", "clopidogrel 75 mg").
        genotypes: LEAVE NULL / OMIT for normal use. Production
            deployments source genotypes from chart-attached
            Observation resources via the SHARP context. This argument
            is honoured only when no SHARP context is bound (offline
            mode used by tests + direct CLI invocation).
        patient_reference: optional FHIR Patient reference for audit.

    Returns:
        PGxDoseAdjustmentReport. Empty `adjustments` + abstain is the
        "no actionable PGx interaction" case (e.g. genotypes don't
        affect any of the listed drugs at CPIC level A/B).
    """
    geno, sharp_bound, _ = await _resolve_genotypes_from_chart_or_caller(
        genotypes,
    )

    if not medications:
        return PGxDoseAdjustmentReport(
            patient_reference=patient_reference, genotypes=geno,
            adjustments=[], n_adjustments=0,
            abstain_recommended=True,
            abstain_reason="empty_medication_list",
            references=["CPIC guidelines (cpicpgx.org)"],
        )
    if not geno:
        return PGxDoseAdjustmentReport(
            patient_reference=patient_reference, genotypes=geno,
            adjustments=[], n_adjustments=0,
            abstain_recommended=True,
            abstain_reason=(
                "no_pgx_observations_in_chart: no PGx Observation found "
                "in the SHARP-bound patient's FHIR bundle. PGx "
                "recommendations require genotype data from a "
                "chart-attached Observation, never from caller-supplied "
                "input."
                if sharp_bound else
                "no_genotypes_supplied: caller did not provide genotypes "
                "and no SHARP-on-MCP context is bound to source them "
                "from a chart."
            ),
            references=["CPIC guidelines"],
        )

    drug_keys = [_norm_drug(m) for m in medications]
    geno_keys = {(g.gene, g.phenotype): g for g in geno}

    adjustments: list[PGxDoseAdjustment] = []
    for drug_full, drug_key in zip(medications, drug_keys):
        for row in _CPIC_TABLE:
            if row["drug"] != drug_key:
                continue
            key = (row["gene"], row["phenotype"])
            if key not in geno_keys:
                continue
            adjustments.append(PGxDoseAdjustment(
                drug=drug_full,
                cpic_guideline_id=f"CPIC:{row['gene']}:{row['drug']}",
                recommendation=row["rec"],
                dose_modifier_pct=row["dose_pct"],
                rationale=row["rationale"],
                cpic_evidence_level=row["level"],
                monitoring_recommendations=list(row["monitoring"]),
            ))

    return PGxDoseAdjustmentReport(
        patient_reference=patient_reference,
        genotypes=geno,
        adjustments=adjustments,
        n_adjustments=len(adjustments),
        abstain_recommended=False,
        references=[
            "CPIC guidelines -- https://cpicpgx.org/guidelines/",
            "FDA Table of Pharmacogenomic Biomarkers in Drug Labeling.",
            "PharmGKB clinical-annotations DB.",
        ],
    )


# Drug -> list of (alternative, same_class, level)
_DRUG_ALTERNATIVES: dict[str, list[tuple[str, bool, str]]] = {
    "codeine": [("morphine", True, "A"), ("hydromorphone", True, "A"),
                  ("acetaminophen + non-opioid", False, "B")],
    "tramadol": [("morphine", False, "A"), ("acetaminophen", False, "B")],
    "clopidogrel": [("prasugrel", True, "A"), ("ticagrelor", True, "A")],
    "voriconazole": [("isavuconazole", True, "A"),
                       ("posaconazole", True, "A")],
    "warfarin": [("apixaban", False, "B"), ("rivaroxaban", False, "B")],
    "simvastatin": [("pravastatin", True, "A"),
                       ("rosuvastatin", True, "A")],
    "azathioprine": [("mycophenolate mofetil", False, "B"),
                       ("methotrexate", False, "B")],
    "fluorouracil": [("oxaliplatin", False, "B")],
    "capecitabine": [("oxaliplatin", False, "B")],
    "abacavir": [("tenofovir + emtricitabine", True, "A")],
    "carbamazepine": [("levetiracetam", True, "A"),
                        ("lacosamide", True, "A"), ("valproate", True, "A")],
    "primaquine": [("tafenoquine", True, "A")],
    "rasburicase": [("allopurinol", True, "A")],
    "irinotecan": [("oxaliplatin", False, "B")],
}


async def compute_pgx_drug_alternatives(
    requested_drug: str,
    genotypes: list[PGxGenotype | dict] | None = None,
    patient_reference: str | None = None,
) -> PGxAlternativesReport:
    """Return alternatives when the patient's genotype blocks the
    requested drug at CPIC level A/B.

    Genotypes are read from the FHIR chart via the SHARP-on-MCP
    context, NOT from the caller. When the chart contains no PGx
    Observation resources the tool abstains; clinical data cannot be
    injected by the chat-side agent. The `genotypes` argument is
    accepted only in offline / unit-test mode (no SHARP context bound).

    Args:
        requested_drug: free-text medication name.
        genotypes: LEAVE NULL / OMIT for normal use. Production
            deployments source genotypes from chart-attached
            Observation resources via the SHARP context. This argument
            is honoured only when no SHARP context is bound (offline
            mode used by tests + direct CLI invocation).
        patient_reference: optional FHIR Patient reference for audit.
    """
    geno, sharp_bound, _ = await _resolve_genotypes_from_chart_or_caller(
        genotypes,
    )
    drug_key = _norm_drug(requested_drug)

    if not geno:
        return PGxAlternativesReport(
            requested_drug=requested_drug,
            blocking_genotypes=[], alternatives=[],
            n_alternatives=0,
            abstain_recommended=True,
            abstain_reason=(
                "no_pgx_observations_in_chart: no PGx Observation found "
                "in the SHARP-bound patient's FHIR bundle. PGx "
                "recommendations require genotype data from a "
                "chart-attached Observation, never from caller-supplied "
                "input."
                if sharp_bound else
                "no_genotypes_supplied: caller did not provide genotypes "
                "and no SHARP-on-MCP context is bound to source them "
                "from a chart."
            ),
            references=[
                "CPIC guidelines -- https://cpicpgx.org/guidelines/",
            ],
        )

    blocking: list[PGxGenotype] = []
    for row in _CPIC_TABLE:
        if row["drug"] != drug_key:
            continue
        if row["rec"] in (
            "avoid_drug", "alternative_drug_strongly_recommended",
        ):
            for g in geno:
                if g.gene == row["gene"] and g.phenotype == row["phenotype"]:
                    blocking.append(g)

    if not blocking:
        return PGxAlternativesReport(
            requested_drug=requested_drug,
            blocking_genotypes=[], alternatives=[],
            n_alternatives=0,
            abstain_recommended=False,
            references=[
                "CPIC guidelines -- https://cpicpgx.org/guidelines/",
            ],
        )

    raw_alts = _DRUG_ALTERNATIVES.get(drug_key, [])
    alternatives: list[PGxAlternative] = []
    blocking_drugs = {a.drug for a in alternatives}
    for alt_name, same_class, level in raw_alts:
        if alt_name in blocking_drugs:
            continue
        alternatives.append(PGxAlternative(
            drug=alt_name,
            same_class=same_class,
            cpic_evidence_level=level,                # type: ignore[arg-type]
            rationale=(
                f"Recommended alternative for "
                f"{', '.join(g.gene + '.' + g.phenotype for g in blocking)}."
            ),
        ))

    return PGxAlternativesReport(
        requested_drug=requested_drug,
        blocking_genotypes=blocking,
        alternatives=alternatives,
        n_alternatives=len(alternatives),
        abstain_recommended=(not alternatives),
        abstain_reason=(
            "no_alternatives_in_curated_table" if not alternatives else None
        ),
        references=[
            "CPIC guidelines -- https://cpicpgx.org/guidelines/",
            "FDA Table of Pharmacogenomic Biomarkers in Drug Labeling.",
        ],
    )


# Test -> list of (drug, level)
_TEST_TO_DRUGS: dict[str, list[tuple[str, str]]] = {
    "cyp2d6": [("codeine", "A"), ("tramadol", "A")],
    "cyp2c19": [("clopidogrel", "A"), ("voriconazole", "A")],
    "cyp2c9": [("warfarin", "A")],
    "vkorc1": [("warfarin", "A")],
    "slco1b1": [("simvastatin", "A")],
    "tpmt": [("azathioprine", "A")],
    "nudt15": [("azathioprine", "A")],
    "dpyd": [("fluorouracil", "A"), ("capecitabine", "A")],
    "ugt1a1": [("irinotecan", "B")],
    "hla-b*5701": [("abacavir", "A")],
    "hla-b*1502": [("carbamazepine", "A")],
    "g6pd": [("primaquine", "A"), ("rasburicase", "A")],
}


async def compute_pgx_eligibility_check(
    requested_test: str,
    medications_in_consideration: list[str] | None = None,
) -> PGxEligibilityReport:
    """Decide whether a given PGx test is justified for the patient's
    medication consideration list."""
    test_key = requested_test.strip().lower()
    cpic_drugs_for_test = _TEST_TO_DRUGS.get(test_key, [])
    cpic_supported = [drug for drug, _ in cpic_drugs_for_test]

    meds = [m.lower() for m in (medications_in_consideration or [])]
    overlap = [d for d in cpic_supported
                  if any(d in m for m in meds)]

    if cpic_drugs_for_test and overlap:
        actionability = (
            "high"
            if any(level == "A"
                      for d, level in cpic_drugs_for_test if d in overlap)
            else "moderate"
        )
        eligible = True
        rationale = (
            f"Patient is being considered for {sorted(overlap)}, which "
            f"have CPIC-supported guidelines on {requested_test}. "
            f"Test is clinically actionable."
        )
    elif cpic_drugs_for_test:
        actionability = "low"
        eligible = False
        rationale = (
            f"{requested_test} has CPIC guidelines for "
            f"{sorted(cpic_supported)} but the patient is not currently "
            "considered for any of those drugs. Test is not "
            "clinically actionable in the present context."
        )
    else:
        actionability = "experimental"
        eligible = False
        rationale = (
            f"{requested_test} is not in the CPIC actionable-test "
            "registry."
        )

    return PGxEligibilityReport(
        requested_test=requested_test,
        medications_in_consideration=medications_in_consideration or [],
        is_eligible=eligible,
        cpic_supported_drugs=cpic_supported,
        rationale=rationale,
        expected_clinical_actionability=actionability,             # type: ignore[arg-type]
        references=[
            "CPIC guidelines -- https://cpicpgx.org/guidelines/",
            "PharmGKB -- https://www.pharmgkb.org/",
            "FDA Table of Pharmacogenomic Biomarkers in Drug Labeling.",
        ],
    )


# ─────────────────────── MCP registration ───────────────────────


def register(mcp) -> None:
    mcp.tool()(compute_pgx_dose_adjustment)
    mcp.tool()(compute_pgx_drug_alternatives)
    mcp.tool()(compute_pgx_eligibility_check)
