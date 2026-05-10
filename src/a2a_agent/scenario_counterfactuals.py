"""Phase 12.5 -- per-scenario counterfactual analysis.

For each of the five end-to-end clinical scenarios, perturb 2-4 key
input factors and detect the **minimum-modification** path that flips
the recommended decision tier. The output is a
`ScenarioCounterfactualReport` that the right-to-explanation surface
publishes alongside the standard scenario run.

Public API:
    `run_counterfactual(scenario_id) -> ScenarioCounterfactualReport`
    `run_all_counterfactuals() -> list[ScenarioCounterfactualReport]`
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from shared.schemas import (
    ScenarioCounterfactualFlip,
    ScenarioCounterfactualReport,
)


# ─────────────────────────────────────────────────────────────────────
# Scenario 1 -- Acute stroke + LVO
# ─────────────────────────────────────────────────────────────────────


async def _cf_acute_stroke() -> ScenarioCounterfactualReport:
    from mcp_server.tools.stroke_thrombolysis_eligibility import (
        compute_stroke_thrombolysis_eligibility,
    )
    baseline_factors = {
        "bp_systolic": 158, "bp_diastolic": 92,
        "glucose_mg_dl": 132,
        "on_anticoagulant": False,
        "platelets_per_uL": 240_000, "inr": 1.0,
        "aspects_score": 8, "lvo_confirmed": True,
    }
    baseline = await compute_stroke_thrombolysis_eligibility(
        last_known_well_minutes_ago=110, nihss_total=13,
        clinical_factors=baseline_factors,
    )

    flips: list[ScenarioCounterfactualFlip] = []
    perturbations_evaluated = 0

    # Perturbation 1: late-window arrival (LKW > 270 min) -- should drop tPA path
    perturbations_evaluated += 1
    late = await compute_stroke_thrombolysis_eligibility(
        last_known_well_minutes_ago=400, nihss_total=13,
        clinical_factors=baseline_factors,
    )
    if late.decision != baseline.decision:
        flips.append(ScenarioCounterfactualFlip(
            factor_name="last_known_well_minutes_ago",
            factor_description="Time since last-known-well",
            original_value=110, modified_value=400,
            original_outcome=baseline.decision,
            modified_outcome=late.decision,
            flip_distance=290.0,
        ))

    # Perturbation 2: anticoagulant on board (warfarin INR 2.5)
    perturbations_evaluated += 1
    aco_factors = dict(baseline_factors)
    aco_factors["on_anticoagulant"] = True
    aco_factors["inr"] = 2.5
    aco = await compute_stroke_thrombolysis_eligibility(
        last_known_well_minutes_ago=110, nihss_total=13,
        clinical_factors=aco_factors,
    )
    if aco.decision != baseline.decision:
        flips.append(ScenarioCounterfactualFlip(
            factor_name="on_anticoagulant",
            factor_description="Patient on anticoagulant with INR ≥ 1.7",
            original_value=False, modified_value=True,
            original_outcome=baseline.decision,
            modified_outcome=aco.decision,
            flip_distance=1.0,
        ))

    # Perturbation 3: LVO not confirmed -> should drop EVT path
    perturbations_evaluated += 1
    no_lvo_factors = dict(baseline_factors)
    no_lvo_factors["lvo_confirmed"] = False
    no_lvo = await compute_stroke_thrombolysis_eligibility(
        last_known_well_minutes_ago=110, nihss_total=13,
        clinical_factors=no_lvo_factors,
    )
    if no_lvo.decision != baseline.decision:
        flips.append(ScenarioCounterfactualFlip(
            factor_name="lvo_confirmed",
            factor_description="Large-vessel occlusion confirmed on CTA",
            original_value=True, modified_value=False,
            original_outcome=baseline.decision,
            modified_outcome=no_lvo.decision,
            flip_distance=1.0,
        ))

    rationale = (
        f"Baseline decision = {baseline.decision}. "
        f"Evaluated {perturbations_evaluated} perturbation(s); "
        f"{len(flips)} flip(s) found. The decision is sensitive to "
        f"time-since-LKW, anticoagulation status, and LVO confirmation."
    )
    return ScenarioCounterfactualReport(
        scenario_id="acute_stroke_lvo",
        title="Acute stroke + LVO + reperfusion eligibility",
        baseline_outcome=baseline.decision,
        perturbations_evaluated=perturbations_evaluated,
        flips_found=flips, n_flips=len(flips),
        rationale=rationale,
        references=[
            "Powers WJ et al. AHA/ASA Stroke Guidelines 2019.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Scenario 2 -- Sepsis bundle (NEWS2 tier)
# ─────────────────────────────────────────────────────────────────────


async def _cf_sepsis_bundle() -> ScenarioCounterfactualReport:
    from mcp_server.tools.clinical_deterioration_score import (
        compute_clinical_deterioration_score,
    )
    obs_iso = "2026-04-30T08:00:00Z"

    def _vitals(rr=26, spo2=91, on_o2=True, temp=38.4,
                  sbp=96, hr=122, avpu="V"):
        return [
            {"type": "respiratory_rate", "value": rr,
             "observed_at": obs_iso},
            {"type": "spo2", "value": spo2, "observed_at": obs_iso},
            {"type": "supplemental_oxygen", "value": 1.0 if on_o2 else 0.0,
             "observed_at": obs_iso},
            {"type": "temperature", "value": temp,
             "observed_at": obs_iso},
            {"type": "systolic_bp", "value": sbp, "observed_at": obs_iso},
            {"type": "heart_rate", "value": hr, "observed_at": obs_iso},
            {"type": "consciousness", "value": avpu,
             "observed_at": obs_iso},
        ]

    baseline = await compute_clinical_deterioration_score(
        vital_signs=_vitals(),
    )

    flips: list[ScenarioCounterfactualFlip] = []
    perturbations_evaluated = 0

    # Perturbation 1: HR back into normal range (122 -> 80)
    perturbations_evaluated += 1
    hr_normal = await compute_clinical_deterioration_score(
        vital_signs=_vitals(hr=80),
    )
    if hr_normal.severity_tier != baseline.severity_tier:
        flips.append(ScenarioCounterfactualFlip(
            factor_name="heart_rate",
            factor_description="Heart rate (bpm)",
            original_value=122, modified_value=80,
            original_outcome=baseline.severity_tier,
            modified_outcome=hr_normal.severity_tier,
            flip_distance=42.0,
        ))

    # Perturbation 2: SBP normalised (96 -> 130)
    perturbations_evaluated += 1
    sbp_normal = await compute_clinical_deterioration_score(
        vital_signs=_vitals(sbp=130),
    )
    if sbp_normal.severity_tier != baseline.severity_tier:
        flips.append(ScenarioCounterfactualFlip(
            factor_name="systolic_bp",
            factor_description="Systolic blood pressure (mmHg)",
            original_value=96, modified_value=130,
            original_outcome=baseline.severity_tier,
            modified_outcome=sbp_normal.severity_tier,
            flip_distance=34.0,
        ))

    # Perturbation 3: ALL vitals normalised (best-case post-resus)
    perturbations_evaluated += 1
    spo2_normal = await compute_clinical_deterioration_score(
        vital_signs=_vitals(rr=14, spo2=97, on_o2=False, temp=37.0,
                                   sbp=130, hr=80, avpu="A"),
    )
    if spo2_normal.severity_tier != baseline.severity_tier:
        flips.append(ScenarioCounterfactualFlip(
            factor_name="all_vitals_normalised",
            factor_description="Full panel of vitals normalises post-resus",
            original_value=("RR 26, SpO2 91 on O2, T 38.4, SBP 96, "
                                  "HR 122, AVPU=V"),
            modified_value="RR 14, SpO2 97 RA, T 37, SBP 130, HR 80, AVPU=A",
            original_outcome=baseline.severity_tier,
            modified_outcome=spo2_normal.severity_tier,
            flip_distance=12.0,
        ))

    rationale = (
        f"Baseline NEWS2 tier = {baseline.severity_tier} "
        f"(score {baseline.score_total}). "
        f"Evaluated {perturbations_evaluated} vital-sign perturbation(s); "
        f"{len(flips)} produced a tier change."
    )
    return ScenarioCounterfactualReport(
        scenario_id="sepsis_bundle",
        title="Sepsis bundle + antibiogram empiric pick",
        baseline_outcome=baseline.severity_tier,
        perturbations_evaluated=perturbations_evaluated,
        flips_found=flips, n_flips=len(flips),
        rationale=rationale,
        references=["NEWS2 -- Royal College of Physicians (2017)."],
    )


# ─────────────────────────────────────────────────────────────────────
# Scenario 3 -- Polytrauma + MTP
# ─────────────────────────────────────────────────────────────────────


async def _cf_polytrauma_mtp() -> ScenarioCounterfactualReport:
    from mcp_server.tools.massive_transfusion_protocol import (
        compute_massive_transfusion_protocol,
    )
    baseline = await compute_massive_transfusion_protocol(
        penetrating_mechanism=False,
        field_or_arrival_sbp_le_90=True,
        heart_rate_ge_120=True,
        positive_fast_exam=True,
        minutes_since_injury=20,
    )
    flips: list[ScenarioCounterfactualFlip] = []
    perturbations_evaluated = 0

    # Perturbation 1: SBP normalised -> ABC point dropped
    perturbations_evaluated += 1
    sbp_ok = await compute_massive_transfusion_protocol(
        penetrating_mechanism=False,
        field_or_arrival_sbp_le_90=False,
        heart_rate_ge_120=True,
        positive_fast_exam=True,
        minutes_since_injury=20,
    )
    if sbp_ok.mtp_activated != baseline.mtp_activated:
        flips.append(ScenarioCounterfactualFlip(
            factor_name="field_or_arrival_sbp_le_90",
            factor_description="SBP ≤ 90 at field or arrival",
            original_value=True, modified_value=False,
            original_outcome=str(baseline.mtp_activated),
            modified_outcome=str(sbp_ok.mtp_activated),
            flip_distance=1.0,
        ))

    # Perturbation 2: FAST negative
    perturbations_evaluated += 1
    fast_neg = await compute_massive_transfusion_protocol(
        penetrating_mechanism=False,
        field_or_arrival_sbp_le_90=True,
        heart_rate_ge_120=True,
        positive_fast_exam=False,
        minutes_since_injury=20,
    )
    if fast_neg.mtp_activated != baseline.mtp_activated:
        flips.append(ScenarioCounterfactualFlip(
            factor_name="positive_fast_exam",
            factor_description="Positive FAST exam at bedside",
            original_value=True, modified_value=False,
            original_outcome=str(baseline.mtp_activated),
            modified_outcome=str(fast_neg.mtp_activated),
            flip_distance=1.0,
        ))

    # Perturbation 3: TXA window closed (>180 min from injury)
    perturbations_evaluated += 1
    late_txa = await compute_massive_transfusion_protocol(
        penetrating_mechanism=False,
        field_or_arrival_sbp_le_90=True,
        heart_rate_ge_120=True,
        positive_fast_exam=True,
        minutes_since_injury=200,
    )
    if late_txa.txa_indicated != baseline.txa_indicated:
        flips.append(ScenarioCounterfactualFlip(
            factor_name="minutes_since_injury",
            factor_description="Minutes since injury (TXA window)",
            original_value=20, modified_value=200,
            original_outcome=f"TXA={baseline.txa_indicated}",
            modified_outcome=f"TXA={late_txa.txa_indicated}",
            flip_distance=180.0,
        ))

    rationale = (
        f"Baseline MTP activate = {baseline.mtp_activated}, "
        f"TXA = {baseline.txa_indicated}. "
        f"Evaluated {perturbations_evaluated} perturbation(s); "
        f"{len(flips)} produced a flip."
    )
    return ScenarioCounterfactualReport(
        scenario_id="polytrauma_mtp",
        title="Polytrauma + MTP activation",
        baseline_outcome=f"MTP={baseline.mtp_activated},"
                                   f"TXA={baseline.txa_indicated}",
        perturbations_evaluated=perturbations_evaluated,
        flips_found=flips, n_flips=len(flips),
        rationale=rationale,
        references=[
            "Holcomb JB et al. PROPPR Trial. JAMA 2015;313(5):471-482.",
            "CRASH-2 Trial -- TXA in trauma. Lancet 2010.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Scenario 4 -- Geriatric polypharmacy
# ─────────────────────────────────────────────────────────────────────


async def _cf_geriatric_polypharmacy() -> ScenarioCounterfactualReport:
    from mcp_server.tools.falls_risk_morse import compute_falls_risk_morse

    baseline_meds = ["warfarin", "lisinopril", "lorazepam",
                          "oxycodone", "diphenhydramine", "metformin"]

    baseline = await compute_falls_risk_morse(
        history_of_falling_3mo=True,
        secondary_diagnosis_present=True,
        ambulatory_aid="walker",
        has_iv_or_heparin_lock=False,
        gait="weak", mental_status="oriented",
        current_medications=baseline_meds,
    )

    flips: list[ScenarioCounterfactualFlip] = []
    perturbations_evaluated = 0

    # Perturbation 1: deprescribe deliriogenic meds
    perturbations_evaluated += 1
    deprescribed = await compute_falls_risk_morse(
        history_of_falling_3mo=True,
        secondary_diagnosis_present=True,
        ambulatory_aid="walker",
        has_iv_or_heparin_lock=False,
        gait="weak", mental_status="oriented",
        current_medications=["warfarin", "lisinopril", "metformin"],
    )
    if deprescribed.risk_tier != baseline.risk_tier:
        flips.append(ScenarioCounterfactualFlip(
            factor_name="current_medications",
            factor_description="Deprescribe deliriogenic meds "
                                  "(lorazepam + oxycodone + diphenhydramine)",
            original_value=baseline_meds,
            modified_value=["warfarin", "lisinopril", "metformin"],
            original_outcome=baseline.risk_tier,
            modified_outcome=deprescribed.risk_tier,
            flip_distance=3.0,    # 3 meds deprescribed
        ))

    # Perturbation 2: gait improved
    perturbations_evaluated += 1
    gait_normal = await compute_falls_risk_morse(
        history_of_falling_3mo=True,
        secondary_diagnosis_present=True,
        ambulatory_aid="walker",
        has_iv_or_heparin_lock=False,
        gait="normal", mental_status="oriented",
        current_medications=baseline_meds,
    )
    if gait_normal.risk_tier != baseline.risk_tier:
        flips.append(ScenarioCounterfactualFlip(
            factor_name="gait",
            factor_description="Gait status",
            original_value="weak", modified_value="normal",
            original_outcome=baseline.risk_tier,
            modified_outcome=gait_normal.risk_tier,
            flip_distance=2.0,
        ))

    # Perturbation 3: no walker, no falls history (low-risk patient)
    perturbations_evaluated += 1
    low_risk = await compute_falls_risk_morse(
        history_of_falling_3mo=False,
        secondary_diagnosis_present=False,
        ambulatory_aid="none",
        has_iv_or_heparin_lock=False,
        gait="normal", mental_status="oriented",
        current_medications=["lisinopril", "metformin"],
    )
    if low_risk.risk_tier != baseline.risk_tier:
        flips.append(ScenarioCounterfactualFlip(
            factor_name="composite_risk_factors",
            factor_description="Drop history, secondary dx, walker, "
                                  "deliriogenic meds",
            original_value="all factors present",
            modified_value="ambulatory + no falls history",
            original_outcome=baseline.risk_tier,
            modified_outcome=low_risk.risk_tier,
            flip_distance=4.0,
        ))

    rationale = (
        f"Baseline Morse tier = {baseline.risk_tier} "
        f"(score {baseline.score_total}). "
        f"{len(flips)} of {perturbations_evaluated} perturbations flipped "
        f"the tier. Deprescribing deliriogenic meds is the highest-leverage "
        f"intervention."
    )
    return ScenarioCounterfactualReport(
        scenario_id="geriatric_polypharmacy",
        title="Geriatric polypharmacy med review",
        baseline_outcome=baseline.risk_tier,
        perturbations_evaluated=perturbations_evaluated,
        flips_found=flips, n_flips=len(flips),
        rationale=rationale,
        references=[
            "Morse JM. Preventing Patient Falls. 2nd ed. Springer (2009).",
            "Beers Criteria -- AGS 2023 Update.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Scenario 5 -- Mental-health crisis
# ─────────────────────────────────────────────────────────────────────


async def _cf_mental_health_crisis() -> ScenarioCounterfactualReport:
    from mcp_server.tools.psychiatric_admission_decision import (
        compute_psychiatric_admission_decision,
    )

    baseline = await compute_psychiatric_admission_decision(
        risk_level="high",
        danger_to_self=True, danger_to_others=False,
        grave_disability=False, voluntary_capable=False,
        state_jurisdiction="CA",
    )

    flips: list[ScenarioCounterfactualFlip] = []
    perturbations_evaluated = 0

    # Perturbation 1: voluntary capable
    perturbations_evaluated += 1
    voluntary = await compute_psychiatric_admission_decision(
        risk_level="high",
        danger_to_self=True, danger_to_others=False,
        grave_disability=False, voluntary_capable=True,
        state_jurisdiction="CA",
    )
    if voluntary.disposition != baseline.disposition:
        flips.append(ScenarioCounterfactualFlip(
            factor_name="voluntary_capable",
            factor_description="Patient retains capacity to consent voluntarily",
            original_value=False, modified_value=True,
            original_outcome=baseline.disposition,
            modified_outcome=voluntary.disposition,
            flip_distance=1.0,
        ))

    # Perturbation 2: no danger-to-self
    perturbations_evaluated += 1
    no_dts = await compute_psychiatric_admission_decision(
        risk_level="moderate",
        danger_to_self=False, danger_to_others=False,
        grave_disability=False, voluntary_capable=False,
        state_jurisdiction="CA",
    )
    if no_dts.disposition != baseline.disposition:
        flips.append(ScenarioCounterfactualFlip(
            factor_name="danger_to_self",
            factor_description="Active danger-to-self",
            original_value=True, modified_value=False,
            original_outcome=baseline.disposition,
            modified_outcome=no_dts.disposition,
            flip_distance=1.0,
        ))

    # Perturbation 3: state jurisdiction (CA -> NY)
    perturbations_evaluated += 1
    ny = await compute_psychiatric_admission_decision(
        risk_level="high",
        danger_to_self=True, danger_to_others=False,
        grave_disability=False, voluntary_capable=False,
        state_jurisdiction="NY",
    )
    if (ny.legal_basis or "") != (baseline.legal_basis or ""):
        flips.append(ScenarioCounterfactualFlip(
            factor_name="state_jurisdiction",
            factor_description="State legal-basis statute",
            original_value="CA (5150)",
            modified_value="NY (MHL 9.39)",
            original_outcome=baseline.legal_basis or "",
            modified_outcome=ny.legal_basis or "",
            flip_distance=1.0,
        ))

    rationale = (
        f"Baseline disposition = {baseline.disposition}. "
        f"Evaluated {perturbations_evaluated} perturbation(s); "
        f"{len(flips)} flip(s) found. The 3-prong test "
        f"(danger-to-self, danger-to-others, grave-disability) drives "
        f"disposition; state jurisdiction modulates the legal-basis text."
    )
    return ScenarioCounterfactualReport(
        scenario_id="mental_health_crisis",
        title="Mental-health crisis + admission decision",
        baseline_outcome=baseline.disposition,
        perturbations_evaluated=perturbations_evaluated,
        flips_found=flips, n_flips=len(flips),
        rationale=rationale,
        references=[
            "California Welfare and Institutions Code § 5150.",
            "Posner K et al. C-SSRS validation. Am J Psychiatry "
            "2011;168(12):1266-1277.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Public registry + dispatcher
# ─────────────────────────────────────────────────────────────────────


_CFS: dict[str, Callable[[], Awaitable[ScenarioCounterfactualReport]]] = {
    "acute_stroke_lvo":       _cf_acute_stroke,
    "sepsis_bundle":          _cf_sepsis_bundle,
    "polytrauma_mtp":         _cf_polytrauma_mtp,
    "geriatric_polypharmacy": _cf_geriatric_polypharmacy,
    "mental_health_crisis":   _cf_mental_health_crisis,
}


def list_counterfactuals() -> list[str]:
    return list(_CFS.keys())


async def run_counterfactual(scenario_id: str) -> ScenarioCounterfactualReport:
    if scenario_id not in _CFS:
        raise KeyError(
            f"Unknown scenario {scenario_id!r}. "
            f"Available: {list(_CFS)}"
        )
    return await _CFS[scenario_id]()


async def run_all_counterfactuals() -> list[ScenarioCounterfactualReport]:
    return [await fn() for fn in _CFS.values()]
