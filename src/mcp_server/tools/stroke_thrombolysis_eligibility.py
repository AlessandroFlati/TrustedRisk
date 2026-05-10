"""healthcare.compute_stroke_thrombolysis_eligibility -- tPA + EVT eligibility.

Encodes the AHA/ASA 2019 acute ischemic stroke reperfusion eligibility tree:
  - IV alteplase (tPA): inclusion + extensive exclusion list
    - Standard window: 0-3h from last-known-well
    - Extended window: 3-4.5h with additional exclusions (>80yo, prior stroke
      + diabetes, oral anticoagulant any-INR, NIHSS >25)
  - Endovascular thrombectomy (EVT):
    - 0-6h with proximal large-vessel anterior circulation occlusion + NIHSS ≥6
    - 6-24h late-window via DAWN / DEFUSE 3 mismatch criteria

References:
  Powers WJ et al. AHA/ASA 2019 Guidelines for the Early Management of AIS.
    Stroke 2019;50:e344.
  Nogueira RG et al. NEJM 2018;378:11 (DAWN, 6-24h).
  Albers GW et al. NEJM 2018;378:708 (DEFUSE 3, 6-16h).
"""

from __future__ import annotations

from typing import Any, Literal

from shared.schemas import ThrombolysisDecision


# ─────────────────────────────────────────────────────────────────────
# IV tPA exclusion check
# ─────────────────────────────────────────────────────────────────────

def _check_tpa_exclusions(factors: dict[str, Any], extended_window: bool
                            ) -> list[str]:
    """Return list of triggered AHA/ASA exclusion criteria."""
    exclusions: list[str] = []

    # Absolute exclusions
    if factors.get("intracranial_hemorrhage_on_imaging"):
        exclusions.append("intracranial hemorrhage on imaging")
    if factors.get("history_intracranial_hemorrhage"):
        exclusions.append("history of intracranial hemorrhage")
    if factors.get("recent_ischemic_stroke_3mo"):
        exclusions.append("ischemic stroke within prior 3 months")
    if factors.get("recent_serious_head_trauma_3mo"):
        exclusions.append("serious head trauma within prior 3 months")
    if factors.get("recent_intracranial_intraspinal_surgery_3mo"):
        exclusions.append("intracranial / intraspinal surgery within prior 3 months")
    if factors.get("active_internal_bleeding"):
        exclusions.append("active internal bleeding")
    if factors.get("known_bleeding_diathesis"):
        exclusions.append("known bleeding diathesis")
    if factors.get("aortic_dissection_suspected"):
        exclusions.append("suspected aortic dissection")
    if factors.get("infective_endocarditis"):
        exclusions.append("infective endocarditis")
    if factors.get("intra_axial_intracranial_neoplasm"):
        exclusions.append("intra-axial intracranial neoplasm")

    # Lab-based exclusions
    inr = factors.get("inr")
    if isinstance(inr, (int, float)) and inr > 1.7:
        exclusions.append(f"INR {inr} > 1.7 (anticoagulant effect)")
    platelets = factors.get("platelets_per_ul")
    if isinstance(platelets, (int, float)) and platelets < 100_000:
        exclusions.append(f"platelets {platelets}/μL < 100k")
    glucose = factors.get("glucose_mg_dl")
    if isinstance(glucose, (int, float)) and glucose < 50:
        exclusions.append(f"glucose {glucose} mg/dL < 50 (must correct first)")

    # Hemodynamic
    sbp = factors.get("systolic_bp")
    dbp = factors.get("diastolic_bp")
    if isinstance(sbp, (int, float)) and sbp > 185:
        exclusions.append(f"SBP {sbp} > 185 mmHg (must lower first)")
    if isinstance(dbp, (int, float)) and dbp > 110:
        exclusions.append(f"DBP {dbp} > 110 mmHg (must lower first)")

    # DOAC use within 48h
    if factors.get("doac_use_within_48h"):
        exclusions.append("DOAC use within 48h (no validated reversal for tPA decision)")

    # Extended-window-specific (3-4.5h)
    if extended_window:
        age = factors.get("age")
        if isinstance(age, (int, float)) and age > 80:
            exclusions.append("age >80 (extended-window relative exclusion)")
        if factors.get("prior_stroke") and factors.get("diabetes"):
            exclusions.append("prior stroke + diabetes (extended-window relative exclusion)")
        if factors.get("oral_anticoagulant_any_inr"):
            exclusions.append("oral anticoagulant use (extended-window exclusion)")
        nihss = factors.get("nihss")
        if isinstance(nihss, (int, float)) and nihss > 25:
            exclusions.append(f"NIHSS {nihss} > 25 (severe stroke, extended-window exclusion)")

    return exclusions


def _tpa_inclusion_criteria(nihss: int, ischemic_imaging: bool) -> list[str]:
    inc: list[str] = []
    if nihss >= 1:
        inc.append(f"clinical stroke deficit (NIHSS {nihss})")
    if ischemic_imaging or nihss >= 1:
        inc.append("ischemic stroke pattern (CT excluding ICH; clinical syndrome)")
    return inc


# ─────────────────────────────────────────────────────────────────────
# EVT (thrombectomy) eligibility
# ─────────────────────────────────────────────────────────────────────

def _check_evt(nihss: int, lkw_min: int, factors: dict[str, Any]
                 ) -> tuple[bool, list[str], list[str], str]:
    """Return (eligible, criteria_met, criteria_unmet, window_label)."""
    met: list[str] = []
    unmet: list[str] = []
    window: str

    if lkw_min <= 360:  # 0-6h
        window = "0_6h"
        if nihss >= 6:
            met.append(f"NIHSS {nihss} ≥ 6")
        else:
            unmet.append(f"NIHSS {nihss} < 6")
        if factors.get("anterior_circulation_lvo_on_imaging"):
            met.append("anterior-circulation LVO on imaging (ICA / M1)")
        else:
            unmet.append("LVO not yet confirmed on CTA / MRA")
        if factors.get("aspects_score", 10) >= 6:
            met.append("ASPECTS ≥6 (limited core infarct on CT)")
        else:
            unmet.append("ASPECTS <6 (large core, less benefit)")
    elif lkw_min <= 1440:  # 6-24h
        window = "6_24h_dawn_defuse"
        # DAWN: clinical-imaging mismatch (NIHSS vs core volume)
        # DEFUSE 3: perfusion mismatch (penumbra > core)
        if factors.get("dawn_or_defuse_mismatch_present"):
            met.append("DAWN/DEFUSE 3 imaging mismatch confirmed")
        else:
            unmet.append("no DAWN/DEFUSE 3 imaging mismatch -- late-window EVT not supported")
        if nihss >= 6:
            met.append(f"NIHSS {nihss} ≥ 6")
        else:
            unmet.append(f"NIHSS {nihss} < 6")
        if factors.get("anterior_circulation_lvo_on_imaging"):
            met.append("anterior-circulation LVO confirmed")
        else:
            unmet.append("LVO not confirmed")
    else:
        window = "outside_window"
        unmet.append(f"last-known-well {lkw_min} min > 24h (outside any EVT window)")

    eligible = len(unmet) == 0 and len(met) >= 2
    return eligible, met, unmet, window


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_stroke_thrombolysis_eligibility(
    last_known_well_minutes_ago: int,
    nihss_total: int,
    clinical_factors: dict[str, Any] | None = None,
    patient_id: str | None = None,
) -> ThrombolysisDecision:
    """Decide IV tPA + EVT eligibility for an acute ischemic stroke patient.

    Args:
        last_known_well_minutes_ago: minutes since last-known-well.
        nihss_total: NIHSS score.
        clinical_factors: dict with imaging findings + comorbidities + labs +
            BP. See `_check_tpa_exclusions` for the recognized keys.

    Returns:
        ThrombolysisDecision with iv_tpa_eligible, evt_eligible, decision tier.
    """
    factors = clinical_factors or {}
    lkw = max(0, int(last_known_well_minutes_ago))
    nihss = max(0, min(42, int(nihss_total)))

    # tPA window
    if lkw <= 180:
        tpa_window = "0_3h"
        extended = False
    elif lkw <= 270:
        tpa_window = "3_4_5h"
        extended = True
    else:
        tpa_window = "outside_window"
        extended = False

    factors_with_nihss = dict(factors)
    factors_with_nihss.setdefault("nihss", nihss)

    if tpa_window == "outside_window":
        tpa_inclusion: list[str] = []
        tpa_exclusion: list[str] = ["last-known-well outside 4.5h tPA window"]
        tpa_eligible = False
    else:
        tpa_inclusion = _tpa_inclusion_criteria(
            nihss, factors.get("ct_excludes_ich", True),
        )
        tpa_exclusion = _check_tpa_exclusions(factors_with_nihss, extended)
        tpa_eligible = (len(tpa_inclusion) >= 1 and len(tpa_exclusion) == 0
                          and nihss >= 1)

    evt_eligible, evt_met, evt_unmet, evt_window = _check_evt(nihss, lkw, factors)

    decision: Literal[
        "iv_tpa_only",
        "evt_only",
        "iv_tpa_plus_evt",
        "no_reperfusion_supportive_care",
        "abstain_clinician_decision",
    ]
    if tpa_eligible and evt_eligible:
        decision = "iv_tpa_plus_evt"
    elif tpa_eligible:
        decision = "iv_tpa_only"
    elif evt_eligible:
        decision = "evt_only"
    elif lkw > 1440 or nihss == 0:
        decision = "no_reperfusion_supportive_care"
    else:
        decision = "abstain_clinician_decision"

    abstain = decision == "abstain_clinician_decision"
    abstain_reason = (
        "exclusions present but window open -- clinician must weigh risks"
        if abstain else None
    )

    rationale = _build_rationale(lkw, nihss, tpa_window, tpa_eligible,
                                   tpa_exclusion, evt_window, evt_eligible,
                                   evt_met, evt_unmet, decision)

    return ThrombolysisDecision(
        patient_id=patient_id,
        last_known_well_minutes_ago=lkw,
        nihss_total=nihss,
        iv_tpa_eligible=tpa_eligible,
        iv_tpa_window=tpa_window,  # type: ignore[arg-type]
        iv_tpa_inclusion_met=tpa_inclusion,
        iv_tpa_exclusion_present=tpa_exclusion,
        evt_eligible=evt_eligible,
        evt_window=evt_window,  # type: ignore[arg-type]
        evt_criteria_met=evt_met,
        evt_criteria_unmet=evt_unmet,
        decision=decision,
        rationale=rationale,
        abstain_recommended=abstain,
        abstain_reason=abstain_reason,
    )


def _build_rationale(lkw: int, nihss: int, tpa_window: str, tpa_eligible: bool,
                       tpa_excl: list[str], evt_window: str, evt_eligible: bool,
                       evt_met: list[str], evt_unmet: list[str],
                       decision: str) -> str:
    parts = [f"Last-known-well {lkw} min, NIHSS {nihss}.",
             f"IV tPA window: {tpa_window}; eligible = {tpa_eligible}."]
    if tpa_excl:
        parts.append(f"tPA exclusions ({len(tpa_excl)}): " + "; ".join(tpa_excl))
    parts.append(f"EVT window: {evt_window}; eligible = {evt_eligible}.")
    if evt_unmet:
        parts.append(f"EVT criteria unmet: " + "; ".join(evt_unmet))
    parts.append(f"Decision: {decision}.")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_stroke_thrombolysis_eligibility)
