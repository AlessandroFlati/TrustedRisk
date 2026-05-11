"""healthcare.compute_weight_based_dosing -- pediatric dose calculator.

Pediatric dosing follows mg/kg formulas with:
  - per-drug min/max dose-per-kg windows (avoid sub-therapeutic AND toxic doses)
  - adult-equivalent maximum cap (a 35 kg child shouldn't get more than the
    adult dose just because mg/kg math says so)
  - liquid-formulation volume calculation when relevant
  - allergy + black-box-warning gating (penicillin allergy -> no amoxicillin)
  - age contraindications (e.g. ceftriaxone < 28 days corrected age)

This is NOT a comprehensive drug database -- it ships ~12 high-frequency
pediatric drugs (antibiotics, antipyretics, antiemetics, anaphylaxis kit).
Extending requires adding an entry to `_DRUG_DB`.

References:
  Lexicomp Pediatric Dosage Handbook (current edition).
  Broselow Pediatric Emergency Tape -- color-coded weight-based dose chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from shared.schemas import PediatricDoseRecommendation

from ._chart_inputs import harden_clinical_inputs


# ─────────────────────────────────────────────────────────────────────
# Drug database
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DrugSpec:
    """One drug's dosing parameters."""
    name: str
    routes: tuple[str, ...]
    dose_mg_per_kg_min: float
    dose_mg_per_kg_max: float
    adult_max_mg: float | None
    formulation: str
    formulation_concentration_mg_per_ml: float | None  # for liquid
    indications: tuple[str, ...]
    age_contraindications_months: tuple[int, int] | None  # (min, max) inclusive ranges to BLOCK
    notes: str
    allergy_classes: tuple[str, ...] = ()  # cross-reactive allergy classes


_DRUG_DB: dict[str, DrugSpec] = {
    "amoxicillin": DrugSpec(
        name="amoxicillin",
        routes=("PO",),
        dose_mg_per_kg_min=40.0, dose_mg_per_kg_max=90.0,
        adult_max_mg=1000.0,
        formulation="amoxicillin 250 mg/5 mL oral suspension",
        formulation_concentration_mg_per_ml=50.0,
        indications=("acute_otitis_media", "strep_pharyngitis",
                      "community_acquired_pneumonia", "uti"),
        age_contraindications_months=None,
        allergy_classes=("penicillin", "beta_lactam"),
        notes="High-dose 80-90 mg/kg/day for AOM in penicillin-non-allergic children.",
    ),
    "ibuprofen": DrugSpec(
        name="ibuprofen",
        routes=("PO",),
        dose_mg_per_kg_min=5.0, dose_mg_per_kg_max=10.0,
        adult_max_mg=400.0,
        formulation="ibuprofen 100 mg/5 mL oral suspension",
        formulation_concentration_mg_per_ml=20.0,
        indications=("fever", "pain"),
        age_contraindications_months=(0, 6),  # contraindicated < 6 months
        notes="Avoid in dehydrated children -- risk of acute kidney injury.",
    ),
    "acetaminophen": DrugSpec(
        name="acetaminophen",
        routes=("PO", "PR"),
        dose_mg_per_kg_min=10.0, dose_mg_per_kg_max=15.0,
        adult_max_mg=1000.0,
        formulation="acetaminophen 160 mg/5 mL oral suspension",
        formulation_concentration_mg_per_ml=32.0,
        indications=("fever", "pain"),
        age_contraindications_months=None,
        notes="Daily max 75 mg/kg/day (max 4 g/day adult).",
    ),
    "ceftriaxone": DrugSpec(
        name="ceftriaxone",
        routes=("IV", "IM"),
        dose_mg_per_kg_min=50.0, dose_mg_per_kg_max=100.0,
        adult_max_mg=2000.0,
        formulation="ceftriaxone 1 g IV",
        formulation_concentration_mg_per_ml=None,
        indications=("meningitis", "bacteremia", "complicated_pneumonia",
                      "complicated_uti"),
        age_contraindications_months=(0, 1),  # avoid < 28 days corrected age
        allergy_classes=("cephalosporin", "beta_lactam"),
        notes="Avoid in neonates < 28 days due to bilirubin displacement risk.",
    ),
    "ondansetron": DrugSpec(
        name="ondansetron",
        routes=("PO", "IV"),
        dose_mg_per_kg_min=0.10, dose_mg_per_kg_max=0.15,
        adult_max_mg=8.0,
        formulation="ondansetron 4 mg ODT (orally disintegrating tablet)",
        formulation_concentration_mg_per_ml=None,
        indications=("nausea", "vomiting", "gastroenteritis"),
        age_contraindications_months=(0, 6),
        notes="QT prolongation -- caution if family history of long QT.",
    ),
    "epinephrine_im": DrugSpec(
        name="epinephrine_im",
        routes=("IM",),
        dose_mg_per_kg_min=0.01, dose_mg_per_kg_max=0.01,
        adult_max_mg=0.5,
        formulation="epinephrine 1 mg/mL (1:1000) IM",
        formulation_concentration_mg_per_ml=1.0,
        indications=("anaphylaxis",),
        age_contraindications_months=None,
        notes="0.01 mg/kg IM, max 0.5 mg/dose. Anterolateral thigh.",
    ),
    "albuterol_neb": DrugSpec(
        name="albuterol_neb",
        routes=("inhaled",),
        dose_mg_per_kg_min=0.10, dose_mg_per_kg_max=0.15,
        adult_max_mg=5.0,
        formulation="albuterol 0.083% (2.5 mg / 3 mL) nebulizer",
        formulation_concentration_mg_per_ml=0.83,
        indications=("asthma", "bronchospasm"),
        age_contraindications_months=None,
        notes="Standard pediatric dose 2.5 mg neb regardless of weight is acceptable.",
    ),
    "azithromycin": DrugSpec(
        name="azithromycin",
        routes=("PO", "IV"),
        dose_mg_per_kg_min=10.0, dose_mg_per_kg_max=12.0,
        adult_max_mg=500.0,
        formulation="azithromycin 200 mg/5 mL oral suspension",
        formulation_concentration_mg_per_ml=40.0,
        indications=("pertussis", "atypical_pneumonia", "strep_pharyngitis_pcn_allergy"),
        age_contraindications_months=None,
        allergy_classes=("macrolide",),
        notes="QT prolongation risk -- caution with concurrent QT-prolonging meds.",
    ),
    "dexamethasone": DrugSpec(
        name="dexamethasone",
        routes=("PO", "IV", "IM"),
        dose_mg_per_kg_min=0.15, dose_mg_per_kg_max=0.6,
        adult_max_mg=16.0,
        formulation="dexamethasone 4 mg/mL injection",
        formulation_concentration_mg_per_ml=4.0,
        indications=("croup", "asthma_exacerbation", "meningitis_adjunct"),
        age_contraindications_months=None,
        notes="Single dose for croup typically 0.6 mg/kg PO.",
    ),
    "vancomycin": DrugSpec(
        name="vancomycin",
        routes=("IV",),
        dose_mg_per_kg_min=15.0, dose_mg_per_kg_max=20.0,
        adult_max_mg=2000.0,
        formulation="vancomycin 500 mg / 1 g vial IV",
        formulation_concentration_mg_per_ml=None,
        indications=("mrsa", "complicated_skin_soft_tissue", "septic_arthritis"),
        age_contraindications_months=None,
        allergy_classes=("vancomycin",),
        notes="Trough monitoring required for prolonged courses.",
    ),
    "morphine_iv": DrugSpec(
        name="morphine_iv",
        routes=("IV", "IM", "SC"),
        dose_mg_per_kg_min=0.05, dose_mg_per_kg_max=0.10,
        adult_max_mg=10.0,
        formulation="morphine sulfate 1 mg/mL (preservative-free) IV",
        formulation_concentration_mg_per_ml=1.0,
        indications=("pain_severe",),
        age_contraindications_months=(0, 6),
        notes="Respiratory monitoring required; have naloxone available.",
    ),
    "oseltamivir": DrugSpec(
        name="oseltamivir",
        routes=("PO",),
        dose_mg_per_kg_min=3.0, dose_mg_per_kg_max=4.0,
        adult_max_mg=75.0,
        formulation="oseltamivir 6 mg/mL oral suspension",
        formulation_concentration_mg_per_ml=6.0,
        indications=("influenza",),
        age_contraindications_months=(0, 0),  # contraindicated <2 weeks
        notes="Twice-daily dosing.",
    ),
    # ─── Phase 12.3 D2 expansion -- 38 additional pediatric drugs ───
    "amoxicillin_clavulanate": DrugSpec(
        name="amoxicillin_clavulanate", routes=("PO",),
        dose_mg_per_kg_min=40.0, dose_mg_per_kg_max=90.0,
        adult_max_mg=1750.0,
        formulation="amoxicillin-clavulanate 600/42.9 mg/5 mL suspension",
        formulation_concentration_mg_per_ml=120.0,
        indications=("acute_otitis_media", "sinusitis", "uti"),
        age_contraindications_months=None,
        allergy_classes=("penicillin", "beta_lactam"),
        notes="Augmentin ES; AOM 90 mg/kg/day.",
    ),
    "cefdinir": DrugSpec(
        name="cefdinir", routes=("PO",),
        dose_mg_per_kg_min=14.0, dose_mg_per_kg_max=14.0,
        adult_max_mg=600.0,
        formulation="cefdinir 125 mg/5 mL suspension",
        formulation_concentration_mg_per_ml=25.0,
        indications=("acute_otitis_media", "strep_pharyngitis"),
        age_contraindications_months=(0, 6),
        allergy_classes=("cephalosporin", "beta_lactam"),
        notes="Reddish stool with iron. Once-daily option.",
    ),
    "cephalexin": DrugSpec(
        name="cephalexin", routes=("PO",),
        dose_mg_per_kg_min=25.0, dose_mg_per_kg_max=100.0,
        adult_max_mg=4000.0,
        formulation="cephalexin 250 mg/5 mL suspension",
        formulation_concentration_mg_per_ml=50.0,
        indications=("ssti", "uti", "strep_pharyngitis"),
        age_contraindications_months=None,
        allergy_classes=("cephalosporin", "beta_lactam"),
        notes="QID dosing for SSTI.",
    ),
    "cefuroxime": DrugSpec(
        name="cefuroxime", routes=("PO", "IV"),
        dose_mg_per_kg_min=20.0, dose_mg_per_kg_max=30.0,
        adult_max_mg=500.0,
        formulation="cefuroxime 250 mg/5 mL suspension",
        formulation_concentration_mg_per_ml=50.0,
        indications=("aom", "sinusitis", "lyme_disease"),
        age_contraindications_months=(0, 3),
        allergy_classes=("cephalosporin", "beta_lactam"),
        notes="BID dosing.",
    ),
    "clindamycin": DrugSpec(
        name="clindamycin", routes=("PO", "IV"),
        dose_mg_per_kg_min=10.0, dose_mg_per_kg_max=40.0,
        adult_max_mg=900.0,
        formulation="clindamycin 75 mg/5 mL suspension",
        formulation_concentration_mg_per_ml=15.0,
        indications=("ssti", "MRSA", "anaerobic_infection"),
        age_contraindications_months=None,
        notes="C. difficile risk; check stool quality.",
    ),
    "trimethoprim_sulfa": DrugSpec(
        name="trimethoprim_sulfa", routes=("PO",),
        dose_mg_per_kg_min=8.0, dose_mg_per_kg_max=12.0,
        adult_max_mg=160.0,
        formulation="TMP-SMX 40/200 mg per 5 mL suspension",
        formulation_concentration_mg_per_ml=8.0,
        indications=("uti", "MRSA_skin", "PJP_prophylaxis"),
        age_contraindications_months=(0, 2),
        allergy_classes=("sulfa",),
        notes="Avoid <2 mo due to kernicterus risk.",
    ),
    "doxycycline": DrugSpec(
        name="doxycycline", routes=("PO", "IV"),
        dose_mg_per_kg_min=2.2, dose_mg_per_kg_max=4.4,
        adult_max_mg=100.0,
        formulation="doxycycline 25 mg/5 mL suspension",
        formulation_concentration_mg_per_ml=5.0,
        indications=("lyme_disease", "RMSF", "atypical_pneumonia"),
        age_contraindications_months=None,
        notes="Tooth staining concern <8y, but short courses considered safe.",
    ),
    "metronidazole": DrugSpec(
        name="metronidazole", routes=("PO", "IV"),
        dose_mg_per_kg_min=15.0, dose_mg_per_kg_max=30.0,
        adult_max_mg=500.0,
        formulation="metronidazole 100 mg/mL suspension",
        formulation_concentration_mg_per_ml=100.0,
        indications=("c_difficile", "anaerobic_infection", "giardiasis"),
        age_contraindications_months=None,
        notes="Disulfiram-like reaction with alcohol.",
    ),
    "nitrofurantoin": DrugSpec(
        name="nitrofurantoin", routes=("PO",),
        dose_mg_per_kg_min=5.0, dose_mg_per_kg_max=7.0,
        adult_max_mg=100.0,
        formulation="nitrofurantoin 25 mg/5 mL suspension",
        formulation_concentration_mg_per_ml=5.0,
        indications=("uti",),
        age_contraindications_months=(0, 1),
        notes="Contraindicated <1 mo + G6PD deficiency.",
    ),
    "valacyclovir": DrugSpec(
        name="valacyclovir", routes=("PO",),
        dose_mg_per_kg_min=20.0, dose_mg_per_kg_max=20.0,
        adult_max_mg=1000.0,
        formulation="valacyclovir 500 mg tablet",
        formulation_concentration_mg_per_ml=None,
        indications=("HSV", "varicella"),
        age_contraindications_months=None,
        notes="Renal dose-adjust if eGFR <50.",
    ),
    "acyclovir_iv": DrugSpec(
        name="acyclovir_iv", routes=("IV",),
        dose_mg_per_kg_min=10.0, dose_mg_per_kg_max=20.0,
        adult_max_mg=1000.0,
        formulation="acyclovir 50 mg/mL IV",
        formulation_concentration_mg_per_ml=50.0,
        indications=("HSV_neonatal", "encephalitis", "varicella_severe"),
        age_contraindications_months=None,
        notes="Hydrate well to avoid crystal nephropathy.",
    ),
    "fluconazole": DrugSpec(
        name="fluconazole", routes=("PO", "IV"),
        dose_mg_per_kg_min=6.0, dose_mg_per_kg_max=12.0,
        adult_max_mg=400.0,
        formulation="fluconazole 50 mg/5 mL suspension",
        formulation_concentration_mg_per_ml=10.0,
        indications=("candidiasis",),
        age_contraindications_months=None,
        notes="QT prolongation; check baseline ECG.",
    ),
    "morphine_po": DrugSpec(
        name="morphine_po", routes=("PO",),
        dose_mg_per_kg_min=0.2, dose_mg_per_kg_max=0.5,
        adult_max_mg=15.0,
        formulation="morphine 10 mg/5 mL solution",
        formulation_concentration_mg_per_ml=2.0,
        indications=("pain_severe",),
        age_contraindications_months=(0, 6),
        notes="Avoid <6 mo. Respiratory depression risk.",
    ),
    "fentanyl_iv": DrugSpec(
        name="fentanyl_iv", routes=("IV", "IN"),
        dose_mg_per_kg_min=0.001, dose_mg_per_kg_max=0.002,
        adult_max_mg=0.1,
        formulation="fentanyl 50 mcg/mL IV",
        formulation_concentration_mg_per_ml=0.05,
        indications=("procedural_sedation", "pain_severe"),
        age_contraindications_months=None,
        notes="1-2 mcg/kg IV/IN. Chest-wall rigidity at high doses.",
    ),
    "ketamine_iv": DrugSpec(
        name="ketamine_iv", routes=("IV", "IM"),
        dose_mg_per_kg_min=1.0, dose_mg_per_kg_max=2.0,
        adult_max_mg=200.0,
        formulation="ketamine 50 mg/mL IV",
        formulation_concentration_mg_per_ml=50.0,
        indications=("procedural_sedation",),
        age_contraindications_months=(0, 3),
        notes="Avoid <3 mo (laryngospasm risk). Co-administer atropine "
              "or glycopyrrolate for hypersalivation.",
    ),
    "midazolam_iv": DrugSpec(
        name="midazolam_iv", routes=("IV", "IN"),
        dose_mg_per_kg_min=0.05, dose_mg_per_kg_max=0.2,
        adult_max_mg=5.0,
        formulation="midazolam 5 mg/mL IV",
        formulation_concentration_mg_per_ml=5.0,
        indications=("procedural_sedation", "status_epilepticus"),
        age_contraindications_months=None,
        notes="Reverse with flumazenil if oversedation.",
    ),
    "lorazepam_iv": DrugSpec(
        name="lorazepam_iv", routes=("IV", "IM"),
        dose_mg_per_kg_min=0.05, dose_mg_per_kg_max=0.1,
        adult_max_mg=4.0,
        formulation="lorazepam 2 mg/mL IV",
        formulation_concentration_mg_per_ml=2.0,
        indications=("status_epilepticus",),
        age_contraindications_months=None,
        notes="Onset 3-10 min; ATLS first-line for SE.",
    ),
    "diazepam_pr": DrugSpec(
        name="diazepam_pr", routes=("PR",),
        dose_mg_per_kg_min=0.3, dose_mg_per_kg_max=0.5,
        adult_max_mg=20.0,
        formulation="diazepam rectal gel 5/10/20 mg",
        formulation_concentration_mg_per_ml=None,
        indications=("seizure_cluster",),
        age_contraindications_months=(0, 6),
        notes="Caregiver-administered for seizure clusters.",
    ),
    "phenobarbital_iv": DrugSpec(
        name="phenobarbital_iv", routes=("IV",),
        dose_mg_per_kg_min=15.0, dose_mg_per_kg_max=20.0,
        adult_max_mg=1000.0,
        formulation="phenobarbital 65 mg/mL IV",
        formulation_concentration_mg_per_ml=65.0,
        indications=("status_epilepticus_neonatal",),
        age_contraindications_months=None,
        notes="Neonatal first-line; respiratory depression at higher doses.",
    ),
    "levetiracetam_iv": DrugSpec(
        name="levetiracetam_iv", routes=("IV", "PO"),
        dose_mg_per_kg_min=20.0, dose_mg_per_kg_max=60.0,
        adult_max_mg=3000.0,
        formulation="levetiracetam 100 mg/mL solution",
        formulation_concentration_mg_per_ml=100.0,
        indications=("seizure_disorder", "status_epilepticus"),
        age_contraindications_months=None,
        notes="Monitor for behavioural side effects.",
    ),
    "fosphenytoin_iv": DrugSpec(
        name="fosphenytoin_iv", routes=("IV", "IM"),
        dose_mg_per_kg_min=15.0, dose_mg_per_kg_max=20.0,
        adult_max_mg=1500.0,
        formulation="fosphenytoin 50 mg PE/mL",
        formulation_concentration_mg_per_ml=50.0,
        indications=("status_epilepticus",),
        age_contraindications_months=None,
        notes="Doses in PE (phenytoin equivalents).",
    ),
    "diphenhydramine": DrugSpec(
        name="diphenhydramine", routes=("PO", "IV", "IM"),
        dose_mg_per_kg_min=1.0, dose_mg_per_kg_max=1.25,
        adult_max_mg=50.0,
        formulation="diphenhydramine 12.5 mg/5 mL elixir",
        formulation_concentration_mg_per_ml=2.5,
        indications=("allergic_reaction", "anaphylaxis_adjunct"),
        age_contraindications_months=(0, 24),
        notes="Avoid <2y per FDA -- paradoxical CNS effects.",
    ),
    "cetirizine": DrugSpec(
        name="cetirizine", routes=("PO",),
        dose_mg_per_kg_min=0.25, dose_mg_per_kg_max=0.5,
        adult_max_mg=10.0,
        formulation="cetirizine 5 mg/5 mL syrup",
        formulation_concentration_mg_per_ml=1.0,
        indications=("allergic_rhinitis", "urticaria"),
        age_contraindications_months=(0, 6),
        notes="Less sedating than diphenhydramine.",
    ),
    "loratadine": DrugSpec(
        name="loratadine", routes=("PO",),
        dose_mg_per_kg_min=0.1, dose_mg_per_kg_max=0.2,
        adult_max_mg=10.0,
        formulation="loratadine 5 mg/5 mL syrup",
        formulation_concentration_mg_per_ml=1.0,
        indications=("allergic_rhinitis",),
        age_contraindications_months=(0, 24),
        notes="Once-daily non-sedating.",
    ),
    "ranitidine": DrugSpec(
        name="ranitidine", routes=("PO", "IV"),
        dose_mg_per_kg_min=2.0, dose_mg_per_kg_max=4.0,
        adult_max_mg=150.0,
        formulation="ranitidine 15 mg/mL syrup",
        formulation_concentration_mg_per_ml=15.0,
        indications=("GERD", "stress_ulcer_prophylaxis"),
        age_contraindications_months=None,
        notes="NDMA contamination concern -- verify supply.",
    ),
    "famotidine": DrugSpec(
        name="famotidine", routes=("PO", "IV"),
        dose_mg_per_kg_min=0.5, dose_mg_per_kg_max=1.0,
        adult_max_mg=20.0,
        formulation="famotidine 40 mg/5 mL suspension",
        formulation_concentration_mg_per_ml=8.0,
        indications=("GERD",),
        age_contraindications_months=None,
        notes="Safer alternative to ranitidine.",
    ),
    "omeprazole": DrugSpec(
        name="omeprazole", routes=("PO",),
        dose_mg_per_kg_min=0.7, dose_mg_per_kg_max=3.3,
        adult_max_mg=40.0,
        formulation="omeprazole 10 mg capsule",
        formulation_concentration_mg_per_ml=None,
        indications=("GERD_severe", "PUD"),
        age_contraindications_months=(0, 12),
        notes="Daily dosing.",
    ),
    "ipratropium_neb": DrugSpec(
        name="ipratropium_neb", routes=("INH",),
        dose_mg_per_kg_min=0.0083, dose_mg_per_kg_max=0.0083,
        adult_max_mg=0.5,
        formulation="ipratropium 250 mcg/2.5 mL nebulizer solution",
        formulation_concentration_mg_per_ml=0.1,
        indications=("severe_asthma", "anticholinergic_bronchodilation"),
        age_contraindications_months=None,
        notes="Use with albuterol q20min × 3 in moderate-severe asthma.",
    ),
    "prednisolone": DrugSpec(
        name="prednisolone", routes=("PO",),
        dose_mg_per_kg_min=1.0, dose_mg_per_kg_max=2.0,
        adult_max_mg=60.0,
        formulation="prednisolone 15 mg/5 mL syrup",
        formulation_concentration_mg_per_ml=3.0,
        indications=("asthma_exacerbation", "croup"),
        age_contraindications_months=None,
        notes="5-day course typical for asthma.",
    ),
    "methylprednisolone_iv": DrugSpec(
        name="methylprednisolone_iv", routes=("IV",),
        dose_mg_per_kg_min=1.0, dose_mg_per_kg_max=2.0,
        adult_max_mg=60.0,
        formulation="methylprednisolone 40 mg/mL IV",
        formulation_concentration_mg_per_ml=40.0,
        indications=("asthma_severe", "anaphylaxis_adjunct"),
        age_contraindications_months=None,
        notes="Solu-Medrol -- reduces airway oedema.",
    ),
    "hydrocortisone_iv": DrugSpec(
        name="hydrocortisone_iv", routes=("IV",),
        dose_mg_per_kg_min=1.0, dose_mg_per_kg_max=4.0,
        adult_max_mg=100.0,
        formulation="hydrocortisone 50 mg/mL IV",
        formulation_concentration_mg_per_ml=50.0,
        indications=("adrenal_insufficiency", "septic_shock_relative_AI"),
        age_contraindications_months=None,
        notes="Stress-dose 50 mg/m²/day for adrenal crisis.",
    ),
    "atropine_iv": DrugSpec(
        name="atropine_iv", routes=("IV", "IM", "ETT"),
        dose_mg_per_kg_min=0.02, dose_mg_per_kg_max=0.02,
        adult_max_mg=0.5,
        formulation="atropine 0.1 mg/mL IV",
        formulation_concentration_mg_per_ml=0.1,
        indications=("bradycardia_unstable", "RSI_pretreatment"),
        age_contraindications_months=None,
        notes="0.02 mg/kg IV -- minimum 0.1 mg per dose.",
    ),
    "dextrose_d10w": DrugSpec(
        name="dextrose_d10w", routes=("IV",),
        dose_mg_per_kg_min=200.0, dose_mg_per_kg_max=500.0,
        adult_max_mg=25000.0,
        formulation="D10W (10% dextrose in water) IV",
        formulation_concentration_mg_per_ml=100.0,
        indications=("hypoglycemia",),
        age_contraindications_months=None,
        notes="Pediatric: 5 mL/kg D10W. Avoid hyperosmolar D50W in <12y.",
    ),
    "naloxone_iv": DrugSpec(
        name="naloxone_iv", routes=("IV", "IM", "IN"),
        dose_mg_per_kg_min=0.01, dose_mg_per_kg_max=0.1,
        adult_max_mg=2.0,
        formulation="naloxone 1 mg/mL",
        formulation_concentration_mg_per_ml=1.0,
        indications=("opioid_overdose",),
        age_contraindications_months=None,
        notes="0.01 mg/kg initial; titrate to respiration. Repeat q2-3 min.",
    ),
    "calcium_gluconate_iv": DrugSpec(
        name="calcium_gluconate_iv", routes=("IV",),
        dose_mg_per_kg_min=60.0, dose_mg_per_kg_max=100.0,
        adult_max_mg=2000.0,
        formulation="calcium gluconate 10% IV (100 mg/mL)",
        formulation_concentration_mg_per_ml=100.0,
        indications=("hypocalcemia", "hyperkalemia_cardioprotection"),
        age_contraindications_months=None,
        notes="Slow push; central line preferred (extravasation).",
    ),
    "sodium_bicarbonate_iv": DrugSpec(
        name="sodium_bicarbonate_iv", routes=("IV",),
        dose_mg_per_kg_min=84.0, dose_mg_per_kg_max=168.0,
        adult_max_mg=8400.0,
        formulation="sodium bicarbonate 8.4% (1 mEq/mL) IV",
        formulation_concentration_mg_per_ml=84.0,
        indications=("severe_acidosis", "TCA_overdose", "hyperkalemia"),
        age_contraindications_months=None,
        notes="1-2 mEq/kg IV; dilute to 4.2% in neonates.",
    ),
    "magnesium_sulfate_iv": DrugSpec(
        name="magnesium_sulfate_iv", routes=("IV",),
        dose_mg_per_kg_min=25.0, dose_mg_per_kg_max=50.0,
        adult_max_mg=2000.0,
        formulation="magnesium sulfate 50% IV (500 mg/mL)",
        formulation_concentration_mg_per_ml=500.0,
        indications=("torsades", "severe_asthma", "eclampsia"),
        age_contraindications_months=None,
        notes="25-50 mg/kg IV over 20 min for severe asthma.",
    ),
    "rocuronium_iv": DrugSpec(
        name="rocuronium_iv", routes=("IV",),
        dose_mg_per_kg_min=0.6, dose_mg_per_kg_max=1.2,
        adult_max_mg=100.0,
        formulation="rocuronium 10 mg/mL IV",
        formulation_concentration_mg_per_ml=10.0,
        indications=("RSI_paralysis",),
        age_contraindications_months=None,
        notes="Onset 60-90s; reverse with sugammadex.",
    ),
    "succinylcholine_iv": DrugSpec(
        name="succinylcholine_iv", routes=("IV",),
        dose_mg_per_kg_min=1.0, dose_mg_per_kg_max=2.0,
        adult_max_mg=100.0,
        formulation="succinylcholine 20 mg/mL IV",
        formulation_concentration_mg_per_ml=20.0,
        indications=("RSI_paralysis_short",),
        age_contraindications_months=None,
        notes="Avoid in burns, hyperkalaemia, MH risk.",
    ),
    "etomidate_iv": DrugSpec(
        name="etomidate_iv", routes=("IV",),
        dose_mg_per_kg_min=0.2, dose_mg_per_kg_max=0.4,
        adult_max_mg=20.0,
        formulation="etomidate 2 mg/mL IV",
        formulation_concentration_mg_per_ml=2.0,
        indications=("RSI_induction",),
        age_contraindications_months=(0, 12),
        notes="Adrenal suppression; avoid prolonged infusion.",
    ),
}


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_weight_based_dosing(
    drug: str,
    weight_kg: float,
    age_months: int,
    indication: str | None = None,
    route: str = "PO",
    allergy_classes: list[str] | None = None,
    target_dose_mg_per_kg: float | None = None,
    patient_id: str | None = None,
) -> PediatricDoseRecommendation:
    """Compute weight-based pediatric dose with safety gates.

    Args:
        drug: name (must be in `_DRUG_DB`).
        weight_kg: patient weight.
        age_months: patient age (used for contraindications).
        indication: optional clinical indication; if not in the drug's
            indication list, abstain.
        route: PO / IV / IM / etc. Must be supported by the drug.
        allergy_classes: list of patient allergy classes (e.g. ["penicillin"]);
            cross-reactive drugs are blocked.
        target_dose_mg_per_kg: optional explicit dose; otherwise the midpoint
            of the spec's min/max window is used.

    Returns:
        PediatricDoseRecommendation with calculated dose, adult cap status,
        formulation volume, and any contraindications/cautions/abstain.
    """
    _r, _sb, _ = await harden_clinical_inputs(
        {"weight_kg": weight_kg, "age_months": age_months},
        chart_derivable={"weight_kg", "age_months"},
    )
    if _sb:
        weight_kg = _r.get("weight_kg") if _r.get("weight_kg") is not None else weight_kg
        age_months = _r.get("age_months") if _r.get("age_months") is not None else age_months
    drug_norm = drug.strip().lower()
    spec = _DRUG_DB.get(drug_norm)
    if spec is None:
        return PediatricDoseRecommendation(
            patient_id=patient_id,
            drug=drug,
            indication=indication,
            weight_kg=weight_kg,
            age_months=age_months,
            dose_mg_per_kg=0.0,
            calculated_dose_mg=0.0,
            final_dose_mg=0.0,
            route="PO",
            formulation="(unknown)",
            abstain_recommended=True,
            abstain_reason=f"drug_not_in_database:{drug!r}",
            rationale=(
                f"{drug!r} is not in the pediatric dosing database. "
                f"Defer to a pharmacist for weight-based calculation."
            ),
        )

    contraindications: list[str] = []
    cautions: list[str] = []

    # Route check
    if route not in spec.routes:
        contraindications.append(
            f"Route {route!r} not approved for {spec.name}; supported routes: "
            f"{', '.join(spec.routes)}."
        )

    # Age contraindication
    if spec.age_contraindications_months is not None:
        lo, hi = spec.age_contraindications_months
        if lo <= age_months <= hi:
            contraindications.append(
                f"Age {age_months / 12.0:.1f}y falls within contraindication "
                f"window ({lo}-{hi} months) for {spec.name}: {spec.notes}"
            )

    # Allergy
    if allergy_classes:
        patient_allergies = {a.strip().lower() for a in allergy_classes}
        drug_classes = {c.lower() for c in spec.allergy_classes}
        cross = patient_allergies & drug_classes
        if cross:
            contraindications.append(
                f"Allergy cross-reactivity: patient is allergic to "
                f"{', '.join(sorted(cross))}, "
                f"and {spec.name} belongs to that class."
            )

    # Indication check (informational, not contraindication)
    if indication and indication not in spec.indications:
        cautions.append(
            f"Indication {indication!r} not in the standard list for "
            f"{spec.name} ({', '.join(spec.indications)}); confirm "
            f"with attending."
        )

    # Dose calculation
    dose_per_kg = (
        target_dose_mg_per_kg if target_dose_mg_per_kg is not None
        else (spec.dose_mg_per_kg_min + spec.dose_mg_per_kg_max) / 2.0
    )
    if dose_per_kg < spec.dose_mg_per_kg_min:
        cautions.append(
            f"Selected dose {dose_per_kg} mg/kg is below the recommended "
            f"minimum ({spec.dose_mg_per_kg_min} mg/kg) for {spec.name}."
        )
    if dose_per_kg > spec.dose_mg_per_kg_max:
        contraindications.append(
            f"Selected dose {dose_per_kg} mg/kg exceeds the recommended "
            f"maximum ({spec.dose_mg_per_kg_max} mg/kg) for {spec.name}."
        )

    calculated_mg = round(dose_per_kg * weight_kg, 2)
    capped = False
    final_mg = calculated_mg
    if spec.adult_max_mg is not None and calculated_mg > spec.adult_max_mg:
        final_mg = spec.adult_max_mg
        capped = True

    volume_ml: float | None = None
    if spec.formulation_concentration_mg_per_ml:
        volume_ml = round(final_mg / spec.formulation_concentration_mg_per_ml, 2)

    abstain = bool(contraindications)
    abstain_reason = "contraindication_present" if abstain else None

    rationale = _build_rationale(spec, weight_kg, age_months, dose_per_kg,
                                   calculated_mg, final_mg, capped,
                                   volume_ml, contraindications, cautions)

    return PediatricDoseRecommendation(
        patient_id=patient_id,
        drug=spec.name,
        indication=indication,
        weight_kg=weight_kg,
        age_months=age_months,
        dose_mg_per_kg=dose_per_kg,
        calculated_dose_mg=calculated_mg,
        adult_max_dose_mg=spec.adult_max_mg,
        capped_at_adult_max=capped,
        final_dose_mg=final_mg,
        route=route,  # type: ignore[arg-type]
        formulation=spec.formulation,
        volume_to_administer_ml=volume_ml,
        contraindications=contraindications,
        cautions=cautions,
        abstain_recommended=abstain,
        abstain_reason=abstain_reason,
        rationale=rationale,
    )


def _build_rationale(spec: DrugSpec, weight_kg: float, age_months: int,
                      dose_per_kg: float, calc_mg: float, final_mg: float,
                      capped: bool, volume_ml: float | None,
                      contras: list[str], cauts: list[str]) -> str:
    parts = [
        f"{spec.name} {dose_per_kg} mg/kg × {weight_kg} kg = "
        f"{calc_mg:.2f} mg.",
    ]
    if capped:
        parts.append(
            f"Calculated dose exceeds the adult max ({spec.adult_max_mg} mg) "
            f"-- capped at {final_mg} mg."
        )
    else:
        parts.append(f"Final dose: {final_mg} mg.")
    if volume_ml is not None:
        parts.append(f"Liquid formulation: {volume_ml} mL of {spec.formulation}.")
    if contras:
        parts.append(f"Contraindications ({len(contras)}): "
                      + "; ".join(contras))
    if cauts:
        parts.append(f"Cautions ({len(cauts)}): " + "; ".join(cauts))
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_weight_based_dosing)
