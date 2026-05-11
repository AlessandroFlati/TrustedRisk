"""healthcare.compute_preeclampsia_assessment -- ACOG 2020 classification.

Per ACOG Practice Bulletin No. 222 (2020):
  Gestational hypertension: SBP ≥140 OR DBP ≥90 (≥2 occasions ≥4h apart) at
    GA ≥20w in a previously normotensive woman, WITHOUT proteinuria or
    severe features.
  Preeclampsia: same BP threshold + EITHER proteinuria (≥300 mg/24h or
    P/Cr ≥0.3 or dipstick ≥2+) OR severe features (regardless of proteinuria).
  Preeclampsia with severe features:
    - SBP ≥160 OR DBP ≥110 (≥4h apart, on bed rest)
    - Thrombocytopenia (plt <100k)
    - Impaired liver function (transaminases ≥2× ULN OR severe RUQ pain)
    - New renal insufficiency (Cr >1.1 OR doubling of baseline)
    - Pulmonary edema
    - New cerebral or visual disturbances
  Eclampsia: preeclampsia + new-onset tonic-clonic / focal / multifocal
    seizures unexplained by other cause.
  HELLP: Hemolysis + Elevated Liver enzymes + Low Platelets.

Delivery is recommended:
  - At 37+0 weeks for preeclampsia without severe features
  - At 34+0 weeks for preeclampsia with severe features (after corticosteroids)
  - Promptly for HELLP / eclampsia / inability to control BP / fetal compromise
"""

from __future__ import annotations

from typing import Literal

from shared.schemas import PreeclampsiaReport

from ._chart_inputs import harden_clinical_inputs


# ─────────────────────────────────────────────────────────────────────
# Severe-feature checks
# ─────────────────────────────────────────────────────────────────────

def _severe_features(sbp: float, dbp: float, factors: dict[str, float | bool]
                       ) -> list[str]:
    sev: list[str] = []
    if sbp >= 160:
        sev.append(f"SBP {sbp} ≥ 160 (severe range)")
    if dbp >= 110:
        sev.append(f"DBP {dbp} ≥ 110 (severe range)")
    plt = factors.get("platelets_per_ul")
    if isinstance(plt, (int, float)) and plt < 100_000:
        sev.append(f"thrombocytopenia plt {plt}/μL < 100k")
    ast = factors.get("ast_ul")
    alt = factors.get("alt_ul")
    if isinstance(ast, (int, float)) and ast >= 80:  # 2× ULN ~40
        sev.append(f"transaminitis AST {ast} U/L (≥2× ULN)")
    if isinstance(alt, (int, float)) and alt >= 80:
        sev.append(f"transaminitis ALT {alt} U/L (≥2× ULN)")
    if factors.get("severe_ruq_pain"):
        sev.append("severe RUQ / epigastric pain unresponsive to medication")
    cr = factors.get("creatinine_mg_dl")
    if isinstance(cr, (int, float)) and cr > 1.1:
        sev.append(f"creatinine {cr} mg/dL > 1.1 (renal insufficiency)")
    if factors.get("pulmonary_edema"):
        sev.append("pulmonary edema")
    if factors.get("severe_headache_persistent"):
        sev.append("severe persistent headache unresponsive to medication")
    if factors.get("visual_disturbances"):
        sev.append("new cerebral or visual disturbances (scotomata, diplopia)")
    return sev


def _hellp_present(factors: dict[str, float | bool]) -> bool:
    plt = factors.get("platelets_per_ul")
    ast = factors.get("ast_ul")
    ldh = factors.get("ldh_ul")
    haptoglobin = factors.get("haptoglobin_low")
    has_hemolysis = bool(haptoglobin) or (isinstance(ldh, (int, float)) and ldh > 600)
    has_elevated_liver = isinstance(ast, (int, float)) and ast >= 70
    has_low_platelets = isinstance(plt, (int, float)) and plt < 100_000
    return has_hemolysis and has_elevated_liver and has_low_platelets


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_preeclampsia_assessment(
    gestational_age_weeks: int,
    systolic_bp: float,
    diastolic_bp: float,
    proteinuria_present: bool = False,
    seizures_present: bool = False,
    clinical_factors: dict[str, float | bool] | None = None,
    patient_id: str | None = None,
) -> PreeclampsiaReport:
    """ACOG 2020 preeclampsia classification + disposition.

    Args:
        gestational_age_weeks: ≥20 required for preeclampsia diagnosis.
        systolic_bp / diastolic_bp: BP measurement (assumed confirmed on
            ≥2 occasions ≥4h apart per ACOG).
        proteinuria_present: 24h ≥300 mg / P/Cr ≥0.3 / dipstick ≥2+.
        seizures_present: new-onset seizures (eclampsia).
        clinical_factors: dict with keys among:
            platelets_per_ul, ast_ul, alt_ul, ldh_ul, haptoglobin_low,
            creatinine_mg_dl, severe_ruq_pain, pulmonary_edema,
            severe_headache_persistent, visual_disturbances.

    Returns:
        PreeclampsiaReport with classification + disposition.
    """
    _r, _sb, _ = await harden_clinical_inputs(
        {"gestational_age_weeks": gestational_age_weeks,
         "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp},
        chart_derivable={"gestational_age_weeks", "systolic_bp",
                          "diastolic_bp"},
    )
    if _sb:
        gestational_age_weeks = _r.get("gestational_age_weeks") if _r.get("gestational_age_weeks") is not None else gestational_age_weeks
        systolic_bp = _r.get("systolic_bp") if _r.get("systolic_bp") is not None else systolic_bp
        diastolic_bp = _r.get("diastolic_bp") if _r.get("diastolic_bp") is not None else diastolic_bp
    factors = clinical_factors or {}
    abstain_reasons: list[str] = []
    if gestational_age_weeks < 20:
        abstain_reasons.append(
            f"GA {gestational_age_weeks}w < 20w -- preeclampsia requires GA ≥20w; "
            f"consider chronic hypertension instead."
        )

    has_htn = systolic_bp >= 140 or diastolic_bp >= 90
    sev_features = _severe_features(systolic_bp, diastolic_bp, factors)
    hellp = _hellp_present(factors)

    classification: Literal[
        "no_preeclampsia",
        "gestational_hypertension",
        "preeclampsia_without_severe_features",
        "preeclampsia_with_severe_features",
        "eclampsia",
        "hellp_syndrome",
    ]

    if not has_htn and not seizures_present and not hellp:
        classification = "no_preeclampsia"
    elif seizures_present:
        classification = "eclampsia"
    elif hellp:
        classification = "hellp_syndrome"
    elif sev_features:
        classification = "preeclampsia_with_severe_features"
    elif proteinuria_present:
        classification = "preeclampsia_without_severe_features"
    elif has_htn:
        classification = "gestational_hypertension"
    else:
        classification = "no_preeclampsia"

    # Delivery recommendation
    delivery = False
    disposition: Literal[
        "outpatient_close_followup",
        "antepartum_admission_for_monitoring",
        "labor_and_delivery_for_workup",
        "labor_and_delivery_for_delivery",
        "icu_obstetric_anesthesia_consult",
    ]

    if classification in ("eclampsia", "hellp_syndrome"):
        delivery = True
        disposition = "icu_obstetric_anesthesia_consult"
    elif classification == "preeclampsia_with_severe_features":
        if gestational_age_weeks >= 34:
            delivery = True
            disposition = "labor_and_delivery_for_delivery"
        else:
            delivery = False
            disposition = "antepartum_admission_for_monitoring"
    elif classification == "preeclampsia_without_severe_features":
        if gestational_age_weeks >= 37:
            delivery = True
            disposition = "labor_and_delivery_for_delivery"
        else:
            delivery = False
            disposition = "antepartum_admission_for_monitoring"
    elif classification == "gestational_hypertension":
        if gestational_age_weeks >= 37:
            delivery = True
            disposition = "labor_and_delivery_for_delivery"
        else:
            disposition = "outpatient_close_followup"
    else:
        disposition = "outpatient_close_followup"

    magnesium = classification in (
        "preeclampsia_with_severe_features", "eclampsia", "hellp_syndrome",
    )
    antihtn = systolic_bp >= 160 or diastolic_bp >= 110

    rationale = _build_rationale(gestational_age_weeks, systolic_bp, diastolic_bp,
                                   proteinuria_present, seizures_present,
                                   sev_features, hellp, classification,
                                   delivery, disposition, magnesium, antihtn)

    return PreeclampsiaReport(
        patient_id=patient_id,
        gestational_age_weeks=gestational_age_weeks,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        proteinuria_present=proteinuria_present,
        severity_features_present=sev_features,
        classification=classification,
        delivery_recommended=delivery,
        magnesium_sulfate_indicated=magnesium,
        antihypertensive_indicated=antihtn,
        recommended_disposition=disposition,
        abstain_recommended=bool(abstain_reasons),
        abstain_reason="; ".join(abstain_reasons) if abstain_reasons else None,
        rationale=rationale,
    )


def _build_rationale(ga: int, sbp: float, dbp: float, prot: bool, seiz: bool,
                       sev: list[str], hellp: bool, cls: str, deliver: bool,
                       disposition: str, mag: bool, antihtn: bool) -> str:
    parts = [
        f"GA {ga}w; BP {sbp}/{dbp} mmHg.",
        f"Classification: {cls}.",
    ]
    if sev:
        parts.append(f"Severe features ({len(sev)}): " + "; ".join(sev))
    if hellp:
        parts.append("HELLP criteria met (hemolysis + elevated LFTs + low plt).")
    parts.append(f"Disposition: {disposition}.")
    parts.append(
        f"Magnesium sulfate: {'YES' if mag else 'no'}; "
        f"antihypertensive: {'YES' if antihtn else 'no'}; "
        f"delivery: {'YES' if deliver else 'no'}."
    )
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_preeclampsia_assessment)
