"""healthcare.compute_treatment_selection -- guideline-grounded treatment ranking.

Given a clinical condition + structured patient factors, ranks candidate
treatment options and surfaces contraindications. Each option is grounded
against the W3 grounding corpus (FAISS index) so the recommendation comes
back with a citable evidence excerpt.

Conditions bundled with v0.5:
  - atrial_fibrillation_anticoagulation     -- CHA₂DS₂-VASc-driven AC choice
  - anticoagulation_pre_op_bridging         -- peri-operative bridging plan
  - hf_reduced_ef_initial_therapy           -- GDMT 4-pillar selection
  - dm2_second_line_after_metformin         -- SGLT2/GLP-1/sulfonylurea/etc.

The dispatcher keeps the per-condition logic small and explicit -- extension
is via a new entry in `_CONDITION_HANDLERS`. Out-of-scope conditions
return `abstain_recommended=True` with `abstain_reason='condition_not_supported'`.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from shared.schemas import (
    TreatmentOption,
    TreatmentSelection,
)


# ─────────────────────────────────────────────────────────────────────
# Lazy ground_claim hook (avoid hard import order)
# ─────────────────────────────────────────────────────────────────────

async def _ground(claim_text: str, patient_id: str | None) -> tuple[str | None, str | None]:
    """Return (excerpt, source_id) from the top supporting evidence, or (None, None).

    Falls back to None silently when the grounding corpus is unavailable --
    treatment selection still works, just without the citation."""
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


async def _safe_ground(claim_text: str, patient_id: str | None
                        ) -> tuple[str | None, str | None]:
    """Wrapper around `_ground` that swallows ANY exception, including those
    raised by a substituted/monkey-patched `_ground` in tests. The resolution
    of `_ground` happens at call time so that test-time monkey-patches still
    apply (this is why we don't bind it once at module load)."""
    try:
        return await _ground(claim_text, patient_id)
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────
# Per-condition handlers
# ─────────────────────────────────────────────────────────────────────

async def _afib_anticoagulation(factors: dict[str, Any], pid: str | None
                                  ) -> tuple[list[TreatmentOption], str, list[str]]:
    """Atrial fibrillation anticoagulation ranking.

    Inputs we look at:
      cha2ds2_vasc_score: 0-9
      hasbled_score: 0-9 (bleeding risk)
      egfr_ml_min: kidney function
      mechanical_valve: bool
      age: int
      prior_intracranial_bleed: bool
    """
    cha2ds2 = int(factors.get("cha2ds2_vasc_score") or 0)
    hasbled = int(factors.get("hasbled_score") or 0)
    egfr = float(factors.get("egfr_ml_min") or 90.0)
    mechanical_valve = bool(factors.get("mechanical_valve") or False)
    prior_ich = bool(factors.get("prior_intracranial_bleed") or False)
    age = int(factors.get("age") or 70)

    options: list[TreatmentOption] = []
    refs = [
        "AHA/ACC/HRS 2023 Guideline for the Diagnosis and Management of "
        "Atrial Fibrillation. Joglar JA, et al. Circulation 2024;149.",
        "ESC 2024 Guidelines for the Management of Atrial Fibrillation. "
        "Van Gelder IC, et al. Eur Heart J 2024;45.",
    ]

    # DOAC (apixaban/rivaroxaban) -- first-line for non-valvular AF
    doac_score = 0.85
    doac_contra: list[str] = []
    doac_caut: list[str] = []
    if mechanical_valve:
        doac_score = 0.0
        doac_contra.append("Mechanical heart valve -- DOACs contraindicated (RE-ALIGN trial).")
    if egfr < 15:
        doac_score = max(0.0, doac_score - 0.5)
        doac_contra.append(f"eGFR {egfr} mL/min -- DOACs contraindicated below 15.")
    elif egfr < 30:
        doac_caut.append(f"eGFR {egfr} mL/min -- apixaban dose reduction; rivaroxaban with caution.")
    if prior_ich:
        doac_caut.append("Prior intracranial hemorrhage -- discuss reduced-dose apixaban with neurology.")
    options.append(TreatmentOption(
        option_id="doac_apixaban",
        label="Direct oral anticoagulant (apixaban first-line)",
        rank=99,  # set after sort
        score=doac_score,
        contraindications=doac_contra,
        cautions=doac_caut,
        expected_benefit=(
            f"Stroke risk reduction ~64% vs no AC; CHA₂DS₂-VASc {cha2ds2} "
            f"justifies long-term AC."
        ),
        expected_harm=(
            f"Major bleeding ~2-3%/yr; HAS-BLED {hasbled} "
            f"{'(elevated)' if hasbled >= 3 else '(acceptable)'}."
        ),
    ))

    # Warfarin -- VKA
    warf_score = 0.55
    warf_caut = ["Requires INR monitoring (target 2-3, 2.5-3.5 if mechanical valve).",
                  "Multiple drug + diet interactions."]
    warf_contra: list[str] = []
    if not mechanical_valve and age >= 75:
        warf_caut.append("Older adult -- frequent INR monitoring + fall-risk assessment.")
    if mechanical_valve:
        warf_score = 0.95  # only good option for mechanical valves
    options.append(TreatmentOption(
        option_id="warfarin",
        label="Warfarin (vitamin K antagonist)",
        rank=99,
        score=warf_score,
        contraindications=warf_contra,
        cautions=warf_caut,
        expected_benefit=(
            "Stroke risk reduction ~64%; the ONLY anticoagulant proven safe "
            "with mechanical heart valves."
            if mechanical_valve else
            "Stroke risk reduction ~64% but inferior to DOACs in most non-valvular AF."
        ),
        expected_harm="Major bleeding ~2.7%/yr; ICH ~0.5%/yr (higher than DOACs).",
    ))

    # Aspirin alone -- historically used, now NOT recommended at CHA₂DS₂-VASc ≥1
    asa_score = 0.05
    asa_caut = []
    if cha2ds2 >= 1:
        asa_caut.append(
            f"Aspirin alone is NOT recommended at CHA₂DS₂-VASc {cha2ds2} ≥ 1 "
            f"(2024 AHA/ESC guidelines)."
        )
    options.append(TreatmentOption(
        option_id="aspirin_only",
        label="Aspirin monotherapy (NOT recommended for stroke prevention in AF)",
        rank=99,
        score=asa_score,
        contraindications=[],
        cautions=asa_caut,
        expected_benefit="Modest stroke reduction (~22%) but inferior to AC and not recommended.",
        expected_harm="Bleeding without the offsetting stroke benefit AC provides.",
    ))

    # No anticoagulation
    none_score = 0.10 if cha2ds2 == 0 else 0.02
    none_caut = []
    if cha2ds2 >= 2:
        none_caut.append(
            f"CHA₂DS₂-VASc {cha2ds2} -> annual stroke risk ~4-7% off-AC; "
            f"strongly disfavored.")
    options.append(TreatmentOption(
        option_id="no_anticoagulation",
        label="No anticoagulation (close monitoring only)",
        rank=99,
        score=none_score,
        contraindications=[],
        cautions=none_caut,
        expected_benefit="No bleeding risk.",
        expected_harm=(
            f"Annual ischemic stroke risk ~{0.5 if cha2ds2 == 0 else cha2ds2 * 1.5:.1f}% "
            f"(CHA₂DS₂-VASc {cha2ds2})."
        ),
    ))

    # Ground top option
    top = max(options, key=lambda o: o.score)
    excerpt, src = await _safe_ground(
        f"For patient with non-valvular atrial fibrillation and "
        f"CHA₂DS₂-VASc {cha2ds2}, {top.label.lower()} is recommended.",
        pid,
    )
    if excerpt:
        top.grounding_excerpt = excerpt
        top.grounding_source = src

    rationale = (
        f"CHA₂DS₂-VASc {cha2ds2} ({'low' if cha2ds2 == 0 else 'moderate' if cha2ds2 == 1 else 'high'}"
        f" stroke risk), HAS-BLED {hasbled}, eGFR {egfr} mL/min, "
        f"mechanical valve = {mechanical_valve}, prior ICH = {prior_ich}. "
        f"Top pick: {top.label}."
    )
    return options, rationale, refs


async def _preop_bridging(factors: dict[str, Any], pid: str | None
                           ) -> tuple[list[TreatmentOption], str, list[str]]:
    """Peri-operative anticoagulation bridging.

    Per BRIDGE trial (NEJM 2015) + ACC 2024 peri-op guidance: most patients
    on warfarin for AF do NOT need bridging -- interrupting warfarin alone
    is non-inferior and reduces bleeding. Bridging remains indicated for
    high-thrombotic-risk groups (mechanical valve, recent VTE, very high
    CHA₂DS₂-VASc with recent stroke).
    """
    indication = str(factors.get("ac_indication") or "atrial_fibrillation")
    cha2ds2 = int(factors.get("cha2ds2_vasc_score") or 0)
    mechanical_valve = bool(factors.get("mechanical_valve") or False)
    recent_vte_weeks = factors.get("recent_vte_weeks")
    surgery_bleeding_risk = str(factors.get("surgery_bleeding_risk") or "low")  # low / high
    on_doac = bool(factors.get("on_doac") or False)

    high_thrombotic = (
        mechanical_valve
        or (isinstance(recent_vte_weeks, (int, float)) and recent_vte_weeks <= 12)
        or cha2ds2 >= 7
    )

    options: list[TreatmentOption] = []
    refs = [
        "Douketis JD et al. Perioperative Bridging Anticoagulation in "
        "Patients with Atrial Fibrillation. NEJM 2015;373:823-833 (BRIDGE).",
        "ACC 2024 Peri-operative Anticoagulation Decision Pathway.",
    ]

    # Option A: hold warfarin/DOAC, no bridging (preferred for most)
    no_bridge_score = 0.30 if high_thrombotic else 0.85
    no_bridge_caut: list[str] = []
    if high_thrombotic:
        no_bridge_caut.append(
            "High thrombotic risk -- bridging may be required. Discuss with cardiology."
        )
    options.append(TreatmentOption(
        option_id="no_bridging",
        label="Hold anticoagulation peri-operatively, no bridging (BRIDGE trial)",
        rank=99,
        score=no_bridge_score,
        contraindications=[],
        cautions=no_bridge_caut,
        expected_benefit=(
            "Lower peri-op bleeding (BRIDGE: 1.3% vs 3.2% major bleeding "
            "with bridging) without inferior thromboembolism prevention in "
            "non-high-risk AF."
        ),
        expected_harm=(
            f"Brief AC interruption: "
            f"{'unacceptable in high thrombotic risk' if high_thrombotic else 'minimal added stroke risk'}."
        ),
    ))

    # Option B: bridging with LMWH
    bridge_score = 0.85 if high_thrombotic else 0.20
    bridge_caut: list[str] = []
    if surgery_bleeding_risk == "high":
        bridge_caut.append("High-bleeding-risk surgery -- bridging multiplies post-op bleeding.")
    options.append(TreatmentOption(
        option_id="bridge_lmwh",
        label="LMWH bridging (enoxaparin therapeutic dose)",
        rank=99,
        score=bridge_score,
        contraindications=[],
        cautions=bridge_caut,
        expected_benefit=(
            "Maintains anticoagulation peri-op; recommended when thrombotic "
            "risk is high (mechanical valve, recent VTE)."
        ),
        expected_harm=(
            "Major bleeding ~3.2% vs 1.3% without bridging (BRIDGE trial); "
            "post-op haematoma at injection sites."
        ),
    ))

    # Option C: continue DOAC through low-bleeding-risk procedure
    if on_doac and surgery_bleeding_risk == "low":
        options.append(TreatmentOption(
            option_id="doac_continue",
            label="Continue DOAC through low-bleeding-risk procedure",
            rank=99,
            score=0.50,
            contraindications=[],
            cautions=[
                "Only for very low-bleeding-risk procedures (cataract, dental cleaning, "
                "skin biopsy). Most surgeries require interruption.",
            ],
            expected_benefit="No interruption of stroke prevention.",
            expected_harm="Minor bleeding at procedure site.",
        ))

    top = max(options, key=lambda o: o.score)
    excerpt, src = await _safe_ground(
        f"For peri-operative management of {indication} on anticoagulation, "
        f"{top.label.lower()} is recommended.",
        pid,
    )
    if excerpt:
        top.grounding_excerpt = excerpt
        top.grounding_source = src

    rationale = (
        f"Indication={indication}, CHA₂DS₂-VASc={cha2ds2}, mechanical_valve={mechanical_valve}, "
        f"recent_vte_weeks={recent_vte_weeks}, surgery_bleeding_risk={surgery_bleeding_risk}, "
        f"high_thrombotic_risk={high_thrombotic}. Top pick: {top.label}."
    )
    return options, rationale, refs


async def _hf_initial_therapy(factors: dict[str, Any], pid: str | None
                                ) -> tuple[list[TreatmentOption], str, list[str]]:
    """HFrEF initial 4-pillar GDMT selection.

    Per AHA/ACC/HFSA 2022 + ESC 2023 HF guidelines, all four pillars should
    be initiated in parallel within 4-6 weeks: ARNI/ACEi/ARB, beta-blocker,
    MRA, SGLT2-i. Order of initiation depends on hemodynamics + tolerability.
    """
    egfr = float(factors.get("egfr_ml_min") or 90.0)
    sbp = float(factors.get("systolic_bp") or 120.0)
    k_meq = float(factors.get("potassium_meq_l") or 4.0)
    hr = float(factors.get("heart_rate") or 75.0)

    options: list[TreatmentOption] = []
    refs = [
        "Heidenreich PA et al. 2022 AHA/ACC/HFSA Guideline for the "
        "Management of Heart Failure. Circulation 2022;145.",
        "McDonagh TA et al. 2023 Focused Update of the 2021 ESC Guidelines "
        "for HF. Eur Heart J 2023;44.",
    ]

    # ARNI (sacubitril/valsartan) -- preferred over ACEi/ARB if SBP allows
    arni_caut: list[str] = []
    arni_contra: list[str] = []
    if sbp < 100:
        arni_caut.append(f"SBP {sbp} mmHg -- start at low dose; risk of symptomatic hypotension.")
    if egfr < 30:
        arni_caut.append(f"eGFR {egfr} mL/min -- reduce dose; closely monitor K+ and Cr.")
    if k_meq > 5.5:
        arni_contra.append(f"K+ {k_meq} mEq/L -- correct hyperkalemia before initiation.")
    options.append(TreatmentOption(
        option_id="arni_sacubitril_valsartan",
        label="ARNI (sacubitril/valsartan) -- preferred RAS blocker",
        rank=99,
        score=0.0 if arni_contra else (0.70 if sbp < 100 else 0.95),
        contraindications=arni_contra,
        cautions=arni_caut,
        expected_benefit=(
            "21% reduction in CV death + HF hospitalization vs enalapril (PARADIGM-HF)."
        ),
        expected_harm="Symptomatic hypotension, hyperkalemia, angioedema.",
    ))

    # Beta-blocker
    bb_caut: list[str] = []
    bb_score = 0.95
    if hr < 55:
        bb_caut.append(f"HR {hr}/min -- defer until HR > 60.")
        bb_score = 0.40
    if sbp < 90:
        bb_caut.append(f"SBP {sbp} mmHg -- defer until SBP > 100 and patient is euvolemic.")
        bb_score = 0.30
    options.append(TreatmentOption(
        option_id="beta_blocker_carvedilol",
        label="Beta-blocker (carvedilol or metoprolol succinate)",
        rank=99,
        score=bb_score,
        contraindications=[],
        cautions=bb_caut,
        expected_benefit="35% mortality reduction in HFrEF (CIBIS-II, MERIT-HF, COPERNICUS).",
        expected_harm="Bradycardia, hypotension, fatigue.",
    ))

    # MRA (spironolactone or eplerenone)
    mra_caut: list[str] = []
    mra_contra: list[str] = []
    if k_meq > 5.0:
        mra_contra.append(f"K+ {k_meq} mEq/L -- MRA contraindicated until normalized.")
    elif k_meq > 4.5:
        mra_caut.append(f"K+ {k_meq} mEq/L -- recheck in 1 week after initiation.")
    if egfr < 30:
        mra_contra.append(f"eGFR {egfr} mL/min -- MRA contraindicated below 30.")
    options.append(TreatmentOption(
        option_id="mra_spironolactone",
        label="Mineralocorticoid receptor antagonist (spironolactone)",
        rank=99,
        score=0.0 if mra_contra else 0.90,
        contraindications=mra_contra,
        cautions=mra_caut,
        expected_benefit="30% mortality reduction (RALES); 18-21% (EMPHASIS-HF eplerenone).",
        expected_harm="Hyperkalemia, gynecomastia (spironolactone).",
    ))

    # SGLT2-i (dapagliflozin or empagliflozin)
    sglt_caut: list[str] = []
    sglt_contra: list[str] = []
    if egfr < 20:
        sglt_contra.append(f"eGFR {egfr} mL/min -- SGLT2-i not recommended below 20.")
    options.append(TreatmentOption(
        option_id="sglt2i_dapagliflozin",
        label="SGLT2 inhibitor (dapagliflozin or empagliflozin)",
        rank=99,
        score=0.0 if sglt_contra else 0.90,
        contraindications=sglt_contra,
        cautions=sglt_caut,
        expected_benefit="26% reduction in CV death + HF hospitalization (DAPA-HF, EMPEROR-Reduced).",
        expected_harm="Genital infections; rare euglycemic DKA; volume depletion.",
    ))

    top = max(options, key=lambda o: o.score)
    excerpt, src = await _safe_ground(
        f"For HFrEF initial therapy, {top.label.lower()} is recommended.",
        pid,
    )
    if excerpt:
        top.grounding_excerpt = excerpt
        top.grounding_source = src

    rationale = (
        f"HFrEF GDMT initiation. Hemodynamics: SBP {sbp}, HR {hr}. "
        f"Labs: eGFR {egfr}, K+ {k_meq}. All 4 pillars should be started "
        f"in parallel within 4-6 weeks. Top pick: {top.label}."
    )
    return options, rationale, refs


async def _dm2_second_line(factors: dict[str, Any], pid: str | None
                              ) -> tuple[list[TreatmentOption], str, list[str]]:
    """Second-line therapy after metformin in T2D.

    Per ADA Standards of Care 2024 + EASD 2022: choose by compelling
    indication. Cardiovascular disease / HF / CKD -> SGLT2-i. ASCVD (no HF)
    -> GLP-1 RA. Cost / weight neutral -> DPP-4 / sulfonylurea.
    """
    has_ascvd = bool(factors.get("ascvd") or False)
    has_hf = bool(factors.get("heart_failure") or False)
    has_ckd = bool(factors.get("ckd") or False)
    bmi = float(factors.get("bmi") or 27.0)
    egfr = float(factors.get("egfr_ml_min") or 90.0)
    a1c = float(factors.get("a1c_pct") or 8.0)

    options: list[TreatmentOption] = []
    refs = [
        "American Diabetes Association. Standards of Care in Diabetes--2024.",
        "Davies MJ et al. Management of Hyperglycemia in T2D. ADA/EASD 2022.",
    ]

    sglt_score = 0.95 if (has_hf or has_ckd) else 0.70
    sglt_contra: list[str] = []
    sglt_caut: list[str] = []
    if egfr < 20:
        sglt_contra.append(f"eGFR {egfr} mL/min -- SGLT2-i contraindicated below 20.")
    options.append(TreatmentOption(
        option_id="sglt2i",
        label="SGLT2 inhibitor (empagliflozin / dapagliflozin)",
        rank=99,
        score=0.0 if sglt_contra else sglt_score,
        contraindications=sglt_contra,
        cautions=sglt_caut,
        expected_benefit=(
            "Cardiorenal benefit: HF hospitalization ↓, CKD progression ↓; "
            "0.5-0.8% A1c reduction; weight loss 2-4 kg."
        ),
        expected_harm="Genital infections; rare euglycemic DKA.",
    ))

    glp_score = 0.93 if has_ascvd else (0.85 if bmi >= 30 else 0.70)
    glp_caut = []
    if egfr < 30:
        glp_caut.append("Adjust dose at low eGFR (semaglutide ok; exenatide avoid).")
    options.append(TreatmentOption(
        option_id="glp1_ra",
        label="GLP-1 RA (semaglutide / liraglutide)",
        rank=99,
        score=glp_score,
        contraindications=[],
        cautions=glp_caut,
        expected_benefit=(
            f"1.0-1.5% A1c reduction; weight loss 4-15 kg; "
            f"{'CV mortality benefit (LEADER, SUSTAIN-6)' if has_ascvd else 'no compelling CV indication here'}."
        ),
        expected_harm="GI side effects (nausea, vomiting); rare pancreatitis.",
    ))

    options.append(TreatmentOption(
        option_id="sulfonylurea",
        label="Sulfonylurea (glipizide) -- cost-effective second line",
        rank=99,
        score=0.40,
        contraindications=[],
        cautions=[
            "Hypoglycemia risk; weight gain 2-3 kg; no CV/renal benefit.",
        ],
        expected_benefit="0.8-1.2% A1c reduction; very cheap.",
        expected_harm="Hypoglycemia, weight gain.",
    ))

    options.append(TreatmentOption(
        option_id="basal_insulin",
        label="Basal insulin (glargine / detemir / degludec)",
        rank=99,
        score=0.55 if a1c >= 10.0 else 0.35,
        contraindications=[],
        cautions=[
            "Required teaching for self-injection + glucose monitoring.",
            "Hypoglycemia risk.",
        ],
        expected_benefit=(
            f"A1c reduction proportional to dose (no ceiling); "
            f"{'preferred starter in marked hyperglycemia (A1c ≥ 10)' if a1c >= 10.0 else 'usually held until oral options exhausted'}."
        ),
        expected_harm="Hypoglycemia, weight gain, injection burden.",
    ))

    top = max(options, key=lambda o: o.score)
    excerpt, src = await _safe_ground(
        f"For T2D second-line therapy after metformin in this patient, "
        f"{top.label.lower()} is recommended.",
        pid,
    )
    if excerpt:
        top.grounding_excerpt = excerpt
        top.grounding_source = src

    rationale = (
        f"T2D second line. ASCVD={has_ascvd}, HF={has_hf}, CKD={has_ckd}, "
        f"BMI={bmi}, eGFR={egfr}, A1c={a1c}%. Top pick: {top.label}."
    )
    return options, rationale, refs


_CONDITION_HANDLERS: dict[str, Callable[[dict[str, Any], str | None],
                                         Awaitable[tuple[list[TreatmentOption], str, list[str]]]]] = {
    "atrial_fibrillation_anticoagulation": _afib_anticoagulation,
    "anticoagulation_pre_op_bridging": _preop_bridging,
    "hf_reduced_ef_initial_therapy": _hf_initial_therapy,
    "dm2_second_line_after_metformin": _dm2_second_line,
}


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_treatment_selection(
    condition: str,
    patient_factors: dict[str, Any] | None = None,
    patient_id: str | None = None,
) -> TreatmentSelection:
    """Rank treatment options for a clinical condition.

    Args:
        condition: one of the supported condition slugs (see
            `_CONDITION_HANDLERS` keys). Unsupported conditions return an
            abstain.
        patient_factors: dict of structured patient inputs (renal function,
            CHA₂DS₂-VASc, prior bleeds, comorbidities, etc.). Per-condition
            handlers document which keys they consume.
        patient_id: opaque ID for grounding citations.

    Returns:
        TreatmentSelection with ranked options + top_pick_id + rationale.
    """
    factors = patient_factors or {}
    handler = _CONDITION_HANDLERS.get(condition)
    if handler is None:
        return TreatmentSelection(
            patient_id=patient_id,
            condition=condition,
            options=[],
            top_pick_id=None,
            abstain_recommended=True,
            abstain_reason=f"condition_not_supported:{condition!r}",
            rationale=(
                f"Condition {condition!r} is not in the supported set "
                f"({', '.join(_CONDITION_HANDLERS.keys())}). Defer to "
                f"clinician for treatment selection."
            ),
            references=[],
        )

    options, rationale, refs = await handler(factors, patient_id)

    # Sort by score descending; assign rank
    options.sort(key=lambda o: o.score, reverse=True)
    for i, opt in enumerate(options, 1):
        opt.rank = i

    top = options[0] if options else None
    top_pick = top.option_id if top and top.score > 0.0 else None
    abstain = top is None or (top and top.score < 0.40)
    abstain_reason = (
        "no_acceptable_option" if abstain and top
        else None if not abstain
        else "no_options_generated"
    )

    return TreatmentSelection(
        patient_id=patient_id,
        condition=condition,
        options=options,
        top_pick_id=top_pick,
        abstain_recommended=abstain,
        abstain_reason=abstain_reason,
        rationale=rationale,
        references=refs,
    )


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_treatment_selection)
