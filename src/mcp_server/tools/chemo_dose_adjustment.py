"""healthcare.compute_chemo_dose_adjustment -- cycle decision support.

Composes:
  - hematologic recovery check (ANC, platelets) -- typical thresholds for
    proceeding: ANC ≥ 1500/μL, platelets ≥ 100k/μL
  - renal function (eGFR) -- carboplatin Calvert formula, cisplatin holds
  - hepatic function (bilirubin, AST/ALT) -- anthracyclines and taxanes
  - performance status (ECOG 0-4) -- ECOG ≥ 3 typically warrants dose reduction
    or treatment break
  - regimen-specific rules

Output: decision = proceed full / proceed reduced / delay / hold / discontinue,
plus growth-factor recommendation, dose reduction percent, and a clinical
rationale.

The tool is regimen-aware via a small dispatch table (`_REGIMEN_RULES`).
Unrecognized regimens return abstain with the structured factors so a
clinician can decide.

References:
  NCCN Clinical Practice Guidelines (Treatment of Cancer by Site).
  Smith TJ et al. Recommendations for the Use of WBC Growth Factors. ASCO.
    JCO 2015;33(28):3199.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from shared.schemas import ChemoDoseAdjustment


# ─────────────────────────────────────────────────────────────────────
# Hematologic + organ-function gates
# ─────────────────────────────────────────────────────────────────────

def _hematologic_ok(anc: float | None, platelets: float | None
                      ) -> tuple[bool, list[str]]:
    """ASCO/NCCN thresholds for proceeding with cytotoxic chemo."""
    issues: list[str] = []
    if anc is not None and anc < 1500:
        if anc < 500:
            issues.append(f"severe neutropenia ANC={anc}/μL (<500)")
        else:
            issues.append(f"neutropenia ANC={anc}/μL (<1500)")
    if platelets is not None and platelets < 100_000:
        if platelets < 50_000:
            issues.append(f"severe thrombocytopenia plt={platelets}/μL (<50k)")
        else:
            issues.append(f"thrombocytopenia plt={platelets}/μL (<100k)")
    return (len(issues) == 0), issues


def _renal_dose_adjust_pct(egfr: float, regimen: str) -> tuple[float, str | None]:
    """Renal-function-driven dose modifier.

    Returns (reduction_pct, note). reduction_pct=0 means no change.
    """
    r = regimen.lower()
    if "cisplatin" in r:
        if egfr < 30:
            return 100.0, ("eGFR <30 mL/min -- cisplatin contraindicated; "
                            "switch to carboplatin or non-platinum.")
        if egfr < 60:
            return 25.0, "Cisplatin reduce 25% for eGFR 30-59."
    if "carboplatin" in r:
        # Calvert formula adjusts AUC*GFR -- flag if GFR outside typical range
        if egfr < 30:
            return 50.0, ("eGFR <30 mL/min -- carboplatin needs Calvert "
                           "recalculation, expect ~50% reduction.")
    if "methotrexate" in r:
        if egfr < 60:
            return 50.0, "High-dose methotrexate: hold if eGFR <60; reduce otherwise."
    return 0.0, None


def _hepatic_dose_adjust_pct(bili: float | None, ast: float | None,
                                regimen: str) -> tuple[float, str | None]:
    if bili is None:
        return 0.0, None
    r = regimen.lower()
    anthracycline_regimens = ("doxorubicin", "anthracycline", "ac_t",
                                "fec_t", "ac-t", "epirubicin", "r_chop")
    if any(k in r for k in anthracycline_regimens):
        if bili > 5.0:
            return 100.0, "Bili >5: anthracycline (doxorubicin/epirubicin) held."
        if bili > 3.0:
            return 75.0, "Bili 3-5: anthracycline reduce 75%."
        if bili > 1.5:
            return 50.0, "Bili 1.5-3: anthracycline reduce 50%."
    if "paclitaxel" in r or "docetaxel" in r:
        if bili > 5.0:
            return 100.0, "Bili >5: taxane held."
        if bili > 1.5 or (ast is not None and ast > 2.5 * 40):  # ULN ~40
            return 25.0, "Bili 1.5-5 or AST >2.5×ULN: taxane reduce ~25%."
    return 0.0, None


def _ecog_modifier(ecog: int | None) -> tuple[str | None, bool]:
    """Returns (decision_note, needs_break_or_alternative)."""
    if ecog is None:
        return None, False
    if ecog >= 3:
        return ("ECOG ≥3 (largely confined to bed/chair) -- discontinue "
                 "cytotoxic chemo or switch to best supportive care."), True
    if ecog == 2:
        return ("ECOG 2 -- consider dose reduction and re-assess functional "
                 "status before next cycle."), False
    return None, False


# ─────────────────────────────────────────────────────────────────────
# Regimen rules (informational -- neutropenia/cytopenia thresholds adjust)
# ─────────────────────────────────────────────────────────────────────

_REGIMEN_KNOWN: tuple[str, ...] = (
    "carboplatin_pemetrexed",   # NSCLC platinum doublet
    "folfox",                    # CRC oxaliplatin + 5-FU
    "ac_t",                      # breast doxorubicin + cyclophosphamide -> paclitaxel
    "cisplatin_etoposide",       # SCLC
    "r_chop",                    # DLBCL rituximab + CHOP
    "fec_t",                     # breast 5-FU/epirubicin/cyclophos -> docetaxel
    "doxorubicin_alone",
    "paclitaxel_weekly",
)


def _growth_factor_indicated(regimen: str, anc: float | None,
                                cycle: int) -> bool:
    """ASCO 2015: pegfilgrastim indicated for regimens with FN risk >20%
    OR ANC <500 in prior cycle. Toy heuristic -- flag for known-high-risk."""
    high_fn_regimens = {"ac_t", "fec_t", "cisplatin_etoposide", "r_chop"}
    if regimen.lower() in high_fn_regimens:
        return True
    if anc is not None and anc < 500 and cycle > 1:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_chemo_dose_adjustment(
    regimen: str,
    cycle_number: int = 1,
    egfr_ml_min: float | None = None,
    bilirubin_mg_dl: float | None = None,
    ast_ul: float | None = None,
    alt_ul: float | None = None,
    anc_per_ul: float | None = None,
    platelets_per_ul: float | None = None,
    ecog_performance_status: int | None = None,
    patient_id: str | None = None,
) -> ChemoDoseAdjustment:
    """Decide whether to proceed / reduce / delay / hold the next cycle.

    The decision tree:
      1. If hematologic recovery is incomplete -> delay 1 week.
      2. If renal/hepatic gate fully blocks the regimen -> discontinue/switch.
      3. If renal/hepatic suggests reduction -> proceed reduced dose.
      4. If ECOG ≥ 3 -> discontinue or transition palliative.
      5. Otherwise proceed full dose.

    Args:
        regimen: known slug (see `_REGIMEN_KNOWN`) or free text.
        cycle_number: 1-24.
        egfr_ml_min / bilirubin / AST / ALT / ANC / platelets / ECOG: labs.
    """
    notes: list[str] = []
    reduction_pct = 0.0
    decision: Literal[
        "proceed_full_dose",
        "proceed_reduced_dose",
        "delay_one_week",
        "hold_until_recovery",
        "discontinue_consider_alternative",
    ]
    delay_reason: str | None = None
    abstain = False
    abstain_reason: str | None = None

    # Hard abstain when none of the safety-gate labs are present. Without
    # eGFR, ANC, or platelets the renal/hematologic gates collapse to
    # "proceed full dose" silently, which is dangerous for cytotoxic
    # regimens (e.g. cisplatin in a patient with undiagnosed AKI).
    if (egfr_ml_min is None and anc_per_ul is None
            and platelets_per_ul is None):
        return ChemoDoseAdjustment(
            patient_id=patient_id,
            regimen=regimen,
            cycle_number=cycle_number,
            egfr_ml_min=None,
            bilirubin_mg_dl=bilirubin_mg_dl,
            ast_ul=ast_ul,
            alt_ul=alt_ul,
            anc_per_ul=anc_per_ul,
            platelets_per_ul=platelets_per_ul,
            ecog_performance_status=ecog_performance_status,
            decision="hold_until_recovery",
            dose_reduction_pct=0.0,
            delay_reason="missing safety-gate labs",
            growth_factor_indicated=False,
            rationale=(
                "Chemo dose decision abstained: none of {eGFR, ANC, "
                "platelets} were supplied. The renal and hematologic "
                "safety gates cannot be applied. The 'hold' shown is "
                "a placeholder safety stance, NOT a calibrated decision."
            ),
            abstain_recommended=True,
            abstain_reason=(
                "missing_safety_gate_labs: cytotoxic chemo dosing "
                "requires at least one of eGFR (renal), ANC + platelets "
                "(hematologic) to gate the next cycle."
            ),
        )

    if regimen.strip().lower() not in _REGIMEN_KNOWN:
        # Unknown regimen -- still produce structured advisories, but flag abstain
        notes.append(
            f"Regimen {regimen!r} not in the known-rules database; "
            f"hematologic / renal / hepatic gates still applied generically."
        )
        abstain = True
        abstain_reason = f"regimen_not_in_database:{regimen!r}"

    heme_ok, heme_issues = _hematologic_ok(anc_per_ul, platelets_per_ul)
    notes.extend(heme_issues)

    # eGFR-driven adjustment only if the value is supplied. A missing
    # eGFR is treated as "no renal gate applicable" rather than "renal
    # function is normal".
    egfr_for_gate = egfr_ml_min if egfr_ml_min is not None else 90.0
    renal_pct, renal_note = _renal_dose_adjust_pct(egfr_for_gate, regimen)
    if renal_note:
        notes.append(renal_note)
    hepatic_pct, hepatic_note = _hepatic_dose_adjust_pct(
        bilirubin_mg_dl, ast_ul, regimen,
    )
    if hepatic_note:
        notes.append(hepatic_note)

    ecog_note, ecog_break = _ecog_modifier(ecog_performance_status)
    if ecog_note:
        notes.append(ecog_note)

    if renal_pct >= 100 or hepatic_pct >= 100 or ecog_break:
        decision = "discontinue_consider_alternative"
        reduction_pct = 100.0
        if ecog_break:
            delay_reason = "ECOG ≥ 3"
        elif renal_pct >= 100:
            delay_reason = "renal contraindication"
        else:
            delay_reason = "hepatic contraindication"
    elif not heme_ok:
        # Hematologic recovery first
        if anc_per_ul is not None and anc_per_ul < 500:
            decision = "hold_until_recovery"
            delay_reason = "severe neutropenia ANC <500"
        else:
            decision = "delay_one_week"
            delay_reason = "; ".join(heme_issues)
        reduction_pct = 0.0
    elif renal_pct > 0 or hepatic_pct > 0:
        decision = "proceed_reduced_dose"
        reduction_pct = max(renal_pct, hepatic_pct)
    else:
        decision = "proceed_full_dose"
        reduction_pct = 0.0

    growth_factor = _growth_factor_indicated(regimen, anc_per_ul, cycle_number)

    rationale = _build_rationale(regimen, cycle_number, decision,
                                   reduction_pct, delay_reason,
                                   growth_factor, notes)

    return ChemoDoseAdjustment(
        patient_id=patient_id,
        regimen=regimen,
        cycle_number=cycle_number,
        egfr_ml_min=egfr_ml_min,
        bilirubin_mg_dl=bilirubin_mg_dl,
        ast_ul=ast_ul,
        alt_ul=alt_ul,
        anc_per_ul=anc_per_ul,
        platelets_per_ul=platelets_per_ul,
        ecog_performance_status=ecog_performance_status,
        decision=decision,
        dose_reduction_pct=reduction_pct,
        delay_reason=delay_reason,
        growth_factor_indicated=growth_factor,
        rationale=rationale,
        abstain_recommended=abstain,
        abstain_reason=abstain_reason,
    )


def _build_rationale(regimen: str, cycle: int, decision: str,
                      reduction_pct: float, delay_reason: str | None,
                      growth_factor: bool, notes: list[str]) -> str:
    parts = [f"Regimen {regimen}, cycle {cycle}.",
             f"Decision: {decision}.",
             f"Dose reduction: {reduction_pct:.0f}%."]
    if delay_reason:
        parts.append(f"Reason: {delay_reason}.")
    if growth_factor:
        parts.append("Pegfilgrastim/G-CSF support indicated next cycle.")
    if notes:
        parts.append("Findings: " + "; ".join(notes))
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_chemo_dose_adjustment)
