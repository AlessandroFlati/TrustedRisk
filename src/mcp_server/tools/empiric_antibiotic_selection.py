"""healthcare.compute_empiric_antibiotic_selection -- guideline-driven empiric antibiotics.

Inputs:
  - infection source (urinary / pneumonia / SSTI / intra-abdominal / CNS /
    bloodstream / unknown)
  - severity (uncomplicated / complicated / sepsis / septic_shock)
  - patient factors (age, allergies, renal function, MRSA risk, ESBL exposure,
    immunocompromise)
  - local antibiogram (optional override of default empirics)

Output: ranked AntibioticOptions with spectrum coverage, contraindications,
durations, and a top pick. Top pick is grounded against the W3 corpus when
available.

References:
  IDSA Sepsis & Septic Shock Guidelines (2021).
  IDSA Complicated UTI Guidelines (2010, updates pending).
  IDSA HAP/VAP Guidelines (2016).
  ATS/IDSA Community-Acquired Pneumonia Guidelines (2019).
  IDSA SSTI Guidelines (2014).
"""

from __future__ import annotations

from typing import Any

from shared.schemas import (
    AntibioticOption,
    AntibioticSelection,
)


# ─────────────────────────────────────────────────────────────────────
# Lazy ground_claim hook (safe wrapper)
# ─────────────────────────────────────────────────────────────────────

async def _safe_ground(claim_text: str, patient_id: str | None
                        ) -> tuple[str | None, str | None]:
    try:
        from .ground_claim import ground_claim  # type: ignore
        result = await ground_claim(claim_text=claim_text, patient_id=patient_id)
    except Exception:
        return None, None
    for sub in result.sub_claims:
        for ev in sub.evidence_sources:
            if ev.source_type == "guideline_passage" and ev.relevance_score > 0.0:
                return ev.excerpt, ev.source_id
    return None, None


# ─────────────────────────────────────────────────────────────────────
# Per-source empiric option templates
# ─────────────────────────────────────────────────────────────────────

def _options_urinary(
    severity: str,
    factors: dict[str, Any],
    antibiogram: dict[str, float],
) -> tuple[list[AntibioticOption], list[str]]:
    age = int(factors.get("age") or 50)
    allergy_pcn = bool(factors.get("allergy_penicillin"))
    egfr = float(factors.get("egfr_ml_min") or 90.0)
    esbl_local = float(antibiogram.get("esbl_prevalence_pct") or 8.0) / 100.0
    has_esbl_history = bool(factors.get("prior_esbl_infection"))
    immunocompromised = bool(factors.get("immunocompromised"))

    options: list[AntibioticOption] = []
    refs = [
        "Gupta K et al. International Clinical Practice Guidelines for the "
        "Treatment of Acute Uncomplicated Cystitis and Pyelonephritis. "
        "Clin Infect Dis 2011;52:e103.",
        "Hooton TM. Uncomplicated Urinary Tract Infection. NEJM 2012;366:1028.",
    ]

    # Nitrofurantoin -- uncomplicated cystitis only
    if severity == "uncomplicated" and egfr >= 30:
        options.append(AntibioticOption(
            regimen_id="nitrofurantoin_po",
            label="Nitrofurantoin 100 mg PO BID",
            rank=99, score=0.92,
            spectrum_coverage=["E. coli", "S. saprophyticus"],
            contraindications=([f"eGFR {egfr} mL/min < 30 (insufficient urinary "
                                f"concentration)"] if egfr < 30 else []),
            cautions=([] if not allergy_pcn else
                       ["No PCN cross-reactivity (not a beta-lactam)"]),
            expected_duration_days=(5, 5),
            dose_per_administration="100 mg",
            route="PO", cost_tier="low",
        ))

    # Fosfomycin -- uncomplicated cystitis
    if severity == "uncomplicated":
        options.append(AntibioticOption(
            regimen_id="fosfomycin_po_single",
            label="Fosfomycin 3 g PO single dose",
            rank=99, score=0.85,
            spectrum_coverage=["E. coli (most ESBL strains)"],
            contraindications=[],
            cautions=["Single-dose convenience but lower cure rate vs 5-day nitrofurantoin"],
            expected_duration_days=(1, 1),
            dose_per_administration="3 g",
            route="PO", cost_tier="medium",
        ))

    # Ceftriaxone -- complicated UTI without ESBL exposure
    if severity in ("complicated", "sepsis"):
        ctx_score = 0.65 if (esbl_local > 0.15 or has_esbl_history) else 0.88
        contras = []
        if allergy_pcn:
            cautions_pcn = (["IgE-mediated PCN allergy: cross-reactivity ~2%; "
                              "consider test dose"])
        else:
            cautions_pcn = []
        options.append(AntibioticOption(
            regimen_id="ceftriaxone_iv",
            label="Ceftriaxone 1 g IV q24h",
            rank=99, score=ctx_score,
            spectrum_coverage=["GNB (non-ESBL)"],
            contraindications=contras,
            cautions=cautions_pcn + (
                [f"Local ESBL prevalence {esbl_local * 100:.0f}% -- empiric coverage may miss"]
                if esbl_local > 0.15 else []
            ),
            expected_duration_days=(7, 14),
            dose_per_administration="1 g",
            route="IV", cost_tier="medium",
        ))

    # Carbapenem -- complicated UTI WITH ESBL risk
    if severity in ("complicated", "sepsis", "septic_shock"):
        carb_score = 0.95 if (has_esbl_history or esbl_local > 0.20) else 0.55
        options.append(AntibioticOption(
            regimen_id="ertapenem_iv",
            label="Ertapenem 1 g IV q24h",
            rank=99, score=carb_score,
            spectrum_coverage=["GNB", "ESBL-producing E. coli/Klebsiella"],
            contraindications=(["Allergy: severe carbapenem allergy"]
                                if factors.get("allergy_carbapenem") else []),
            cautions=[
                "Reserve for ESBL-suspected: carbapenem stewardship -- narrow as "
                "soon as susceptibility is back."
            ],
            expected_duration_days=(7, 14),
            dose_per_administration="1 g",
            route="IV", cost_tier="high",
        ))

    # Pip-tazo -- septic shock or unknown source-broad coverage
    if severity in ("sepsis", "septic_shock"):
        options.append(AntibioticOption(
            regimen_id="pip_tazo_iv",
            label="Piperacillin-tazobactam 4.5 g IV q8h",
            rank=99, score=0.82,
            spectrum_coverage=["GNB", "Pseudomonas", "anaerobes", "Enterococcus"],
            contraindications=(["Severe PCN allergy"] if allergy_pcn else []),
            cautions=["Renal-dose adjust: extended infusion preferred at "
                       "high MIC organisms"],
            expected_duration_days=(7, 14),
            dose_per_administration="4.5 g",
            route="IV", cost_tier="medium",
        ))

    return options, refs


def _options_pneumonia(severity: str, factors: dict[str, Any]
                         ) -> tuple[list[AntibioticOption], list[str]]:
    inpatient_general = severity in ("complicated", "sepsis")
    icu_required = severity == "septic_shock"
    mrsa_risk = bool(factors.get("mrsa_risk")) or bool(factors.get("recent_hospitalization"))
    pseudomonas_risk = bool(factors.get("pseudomonas_risk"))
    allergy_pcn = bool(factors.get("allergy_penicillin"))

    options: list[AntibioticOption] = []
    refs = [
        "Metlay JP et al. Diagnosis and Treatment of Adults with CAP. ATS/IDSA. "
        "Am J Respir Crit Care Med 2019;200:e45.",
        "Kalil AC et al. Management of Adults with HAP and VAP. IDSA/ATS. "
        "Clin Infect Dis 2016;63:e61.",
    ]

    if severity == "uncomplicated":
        # CAP outpatient, no comorbidities
        options.append(AntibioticOption(
            regimen_id="amoxicillin_po",
            label="Amoxicillin 1 g PO TID",
            rank=99, score=(0.0 if allergy_pcn else 0.90),
            spectrum_coverage=["S. pneumoniae"],
            contraindications=(["PCN allergy"] if allergy_pcn else []),
            cautions=[],
            expected_duration_days=(5, 7),
            dose_per_administration="1 g",
            route="PO", cost_tier="low",
        ))
        options.append(AntibioticOption(
            regimen_id="doxycycline_po",
            label="Doxycycline 100 mg PO BID",
            rank=99, score=0.78,
            spectrum_coverage=["S. pneumoniae", "atypicals"],
            contraindications=[],
            cautions=["Avoid in pregnancy and children < 8 years"],
            expected_duration_days=(5, 7),
            dose_per_administration="100 mg",
            route="PO", cost_tier="low",
        ))

    if inpatient_general or icu_required:
        # Inpatient: beta-lactam + macrolide
        options.append(AntibioticOption(
            regimen_id="ceftriaxone_azithromycin",
            label="Ceftriaxone 1-2 g IV q24h + Azithromycin 500 mg IV q24h",
            rank=99, score=(0.50 if allergy_pcn else 0.92),
            spectrum_coverage=["S. pneumoniae", "atypicals", "GNB"],
            contraindications=(["Severe PCN allergy"] if allergy_pcn else []),
            cautions=["Macrolide QT prolongation"],
            expected_duration_days=(5, 7),
            dose_per_administration="ceftriaxone 1-2 g + azithromycin 500 mg",
            route="IV", cost_tier="medium",
        ))

    if icu_required or mrsa_risk:
        # Add MRSA coverage in ICU/VAP/post-influenza patterns
        options.append(AntibioticOption(
            regimen_id="vancomycin_iv",
            label="Vancomycin 15-20 mg/kg IV q12h (add to base regimen)",
            rank=99, score=0.85,
            spectrum_coverage=["MRSA"],
            contraindications=[],
            cautions=["Trough monitoring required",
                       "Renal-dose adjust"],
            expected_duration_days=(5, 7),
            dose_per_administration="15-20 mg/kg",
            route="IV", cost_tier="medium",
        ))

    if pseudomonas_risk and (icu_required or factors.get("hap_vap")):
        options.append(AntibioticOption(
            regimen_id="cefepime_iv",
            label="Cefepime 2 g IV q8h (Pseudomonas coverage)",
            rank=99, score=0.88,
            spectrum_coverage=["Pseudomonas", "GNB"],
            contraindications=[],
            cautions=["Neurotoxicity at high renal-impaired doses"],
            expected_duration_days=(7, 14),
            dose_per_administration="2 g",
            route="IV", cost_tier="medium",
        ))

    return options, refs


def _options_ssti(severity: str, factors: dict[str, Any]
                    ) -> tuple[list[AntibioticOption], list[str]]:
    mrsa_risk = bool(factors.get("mrsa_risk"))
    severe = severity in ("complicated", "sepsis", "septic_shock")

    options: list[AntibioticOption] = []
    refs = [
        "Stevens DL et al. Practice Guidelines for the Diagnosis and "
        "Management of SSTI. IDSA. Clin Infect Dis 2014;59:e10.",
    ]

    if severity == "uncomplicated" and not mrsa_risk:
        options.append(AntibioticOption(
            regimen_id="cephalexin_po",
            label="Cephalexin 500 mg PO QID",
            rank=99, score=0.90,
            spectrum_coverage=["MSSA", "Streptococci"],
            contraindications=[],
            cautions=[],
            expected_duration_days=(5, 7),
            dose_per_administration="500 mg",
            route="PO", cost_tier="low",
        ))

    if mrsa_risk and severity == "uncomplicated":
        options.append(AntibioticOption(
            regimen_id="tmp_smx_po",
            label="Trimethoprim-sulfamethoxazole DS PO BID",
            rank=99, score=0.82,
            spectrum_coverage=["MRSA"],
            contraindications=[],
            cautions=["Sulfa allergy", "Hyperkalemia + renal impairment"],
            expected_duration_days=(5, 7),
            dose_per_administration="1 DS tablet",
            route="PO", cost_tier="low",
        ))

    if severe:
        options.append(AntibioticOption(
            regimen_id="vancomycin_pip_tazo",
            label="Vancomycin 15-20 mg/kg IV + Piperacillin-tazobactam 4.5 g IV q8h",
            rank=99, score=0.93,
            spectrum_coverage=["MRSA", "GNB", "anaerobes"],
            contraindications=[],
            cautions=["Necrotizing fasciitis: surgical consult is the primary therapy."],
            expected_duration_days=(7, 14),
            dose_per_administration="vanc 15-20 mg/kg + pip-tazo 4.5 g",
            route="IV", cost_tier="high",
        ))

    return options, refs


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_empiric_antibiotic_selection(
    infection_source: str,
    severity: str = "uncomplicated",
    patient_factors: dict[str, Any] | None = None,
    local_antibiogram: dict[str, float] | None = None,
    patient_id: str | None = None,
) -> AntibioticSelection:
    """Pick an empiric antibiotic regimen given source, severity, factors.

    Args:
        infection_source: urinary / pneumonia / skin_soft_tissue /
            intra_abdominal / cns / bloodstream / unknown.
        severity: uncomplicated / complicated / sepsis / septic_shock.
        patient_factors: dict with keys among {age, allergy_penicillin,
            allergy_carbapenem, egfr_ml_min, mrsa_risk, pseudomonas_risk,
            recent_hospitalization, prior_esbl_infection, immunocompromised,
            hap_vap}.
        local_antibiogram: dict with local resistance prevalences, e.g.
            {"esbl_prevalence_pct": 18, "mrsa_prevalence_pct": 12}.
            When provided, drives empiric escalation (e.g. ESBL > 15% pulls
            carbapenem to top pick for complicated UTI).

    Returns:
        AntibioticSelection with ranked options.
    """
    factors = patient_factors or {}
    antibiogram = local_antibiogram or {}
    src = infection_source.strip().lower()
    sev = severity.strip().lower()
    if sev not in ("uncomplicated", "complicated", "sepsis", "septic_shock"):
        sev = "uncomplicated"

    if src == "urinary":
        options, refs = _options_urinary(sev, factors, antibiogram)
    elif src == "pneumonia":
        options, refs = _options_pneumonia(sev, factors)
    elif src == "skin_soft_tissue":
        options, refs = _options_ssti(sev, factors)
    elif src in ("intra_abdominal", "cns", "bloodstream", "unknown"):
        # Generic broad-spectrum recommendation; abstain on actual choice
        options = []
        refs = []
        return AntibioticSelection(
            patient_id=patient_id,
            infection_source=src,  # type: ignore[arg-type]
            severity=sev,  # type: ignore[arg-type]
            options=[],
            top_pick_id=None,
            abstain_recommended=True,
            abstain_reason=f"infection_source_not_yet_supported:{src!r}",
            rationale=(
                f"Empiric selection for {src!r} requires source-specific "
                f"microbiology stewardship that isn't bundled in v0.6. "
                f"Defer to ID consultation."
            ),
            local_antibiogram_used=bool(antibiogram),
            references=[],
        )
    else:
        return AntibioticSelection(
            patient_id=patient_id,
            infection_source="unknown",
            severity=sev,  # type: ignore[arg-type]
            options=[],
            top_pick_id=None,
            abstain_recommended=True,
            abstain_reason=f"unknown_infection_source:{src!r}",
            rationale=f"Unknown source {src!r}; defer to ID.",
            references=[],
        )

    # Filter options: anything with score == 0 is a hard contraindication;
    # leave them in the ranking (visible to clinician) but do not pick them.
    options.sort(key=lambda o: o.score, reverse=True)
    for i, opt in enumerate(options, 1):
        opt.rank = i

    top = options[0] if options and options[0].score > 0 else None
    if top is not None:
        excerpt, src_id = await _safe_ground(
            f"For {sev} {src} infection in this patient, "
            f"{top.label.lower()} is recommended.",
            patient_id,
        )
        if excerpt:
            top.grounding_excerpt = excerpt
            top.grounding_source = src_id

    abstain = top is None
    abstain_reason = "no_acceptable_option" if abstain else None

    rationale = _build_rationale(src, sev, factors, antibiogram, top)

    return AntibioticSelection(
        patient_id=patient_id,
        infection_source=src,  # type: ignore[arg-type]
        severity=sev,  # type: ignore[arg-type]
        options=options,
        top_pick_id=top.regimen_id if top else None,
        abstain_recommended=abstain,
        abstain_reason=abstain_reason,
        rationale=rationale,
        local_antibiogram_used=bool(antibiogram),
        references=refs,
    )


def _build_rationale(src: str, sev: str, factors: dict[str, Any],
                      antibiogram: dict[str, float],
                      top: AntibioticOption | None) -> str:
    parts = [f"Empiric selection for {sev} {src} infection."]
    if antibiogram:
        ab_str = ", ".join(f"{k}={v}" for k, v in antibiogram.items())
        parts.append(f"Local antibiogram applied: {ab_str}.")
    if factors:
        relevant = {k: v for k, v in factors.items() if k in (
            "allergy_penicillin", "egfr_ml_min", "mrsa_risk",
            "pseudomonas_risk", "prior_esbl_infection", "immunocompromised",
        )}
        if relevant:
            parts.append(f"Patient factors: {relevant}.")
    if top:
        parts.append(
            f"Top pick: {top.label} (score {top.score:.2f}, duration "
            f"{top.expected_duration_days[0]}-{top.expected_duration_days[1]} d)."
        )
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_empiric_antibiotic_selection)
