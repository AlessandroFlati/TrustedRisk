"""Phase 16.F1 - Model Card (Mitchell 2019) + Datasheet (Gebru 2021).

Two formal trustworthy-ML artefacts that turn TrustedRisk's calibration
+ fairness story into the canonical research-community shape:

  - **Model Card** (Mitchell et al., FAT*'19) - the documentation
    standard for machine-learning models. 9 sections: details,
    intended use, factors, metrics, evaluation data, training data,
    quantitative analyses, ethical considerations, caveats.

  - **Datasheet for Datasets** (Gebru et al., CACM'21) - the
    documentation standard for the cohorts used to fit the model.
    7 sections: motivation, composition, collection, preprocessing,
    uses, distribution, maintenance.

Both artefacts are emitted as Markdown + JSON sidecars. The model card
re-uses the live calibration + subgroup-audit artefacts when present;
the datasheet describes the synthetic cohort generators that this
project ships (Synthea-100k recal + the subgroup-audit cohort).

Pure-deterministic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ─────────────────────────────────────────────────────────────────────
# Model Card (Mitchell 2019)
# ─────────────────────────────────────────────────────────────────────


class ModelCardSection(BaseModel):
    title: str
    body_md: str


class ModelCard(BaseModel):
    generated_at: str
    model_name: str = "TrustedRisk readmission risk + decision pipeline"
    model_version: str = "1.0.0"
    n_sections: int = Field(ge=0)
    sections: list[ModelCardSection]


def _mc_details() -> ModelCardSection:
    body = (
        "- **Person/organization developing the model**: TrustedRisk "
        "team.\n"
        "- **Model date**: 2026-04-30 (W1 coefficients promoted "
        "2026-04-24).\n"
        "- **Model type**: Binary classifier on the LACE feature space "
        "(L=length-of-stay, A=acuity, C=Charlson, E=ED visits) + "
        "5-bin Beta-Binomial posterior. Combined with a 4-critic "
        "ensemble + 3-agent debate that downgrades or abstains.\n"
        "- **Information about training algorithms**: per-bin "
        "Beta-Binomial conjugate update with weakly-informative prior "
        "alpha=2, beta=18.\n"
        "- **License**: Apache 2.0 for the codebase; coefficients "
        "produced by an internal calibration workflow (not "
        "redistributed).\n"
        "- **Where to send questions or comments**: GitHub Issues."
    )
    return ModelCardSection(title="1. Model details", body_md=body)


def _mc_intended_use() -> ModelCardSection:
    body = (
        "- **Primary intended uses**: assistive decision-support at "
        "inpatient discharge planning. The model emits a calibrated "
        "30-day readmission probability + recommended action; a "
        "clinician reviews before any care decision is enacted.\n"
        "- **Primary intended users**: licensed clinicians.\n"
        "- **Out-of-scope use cases**: unsupervised patient-facing "
        "automation; replacement of clinician judgement; settings "
        "where the LACE feature space is unavailable; pediatric "
        "patients (the model was not fit on a pediatric cohort)."
    )
    return ModelCardSection(title="2. Intended use", body_md=body)


def _mc_factors() -> ModelCardSection:
    body = (
        "- **Relevant factors**: age band (5 strata), sex, race "
        "(White/Black/Hispanic/Asian/Indigenous/Other), ethnicity "
        "(Hispanic/Non-Hispanic), insurance type (Commercial/"
        "Medicare/Medicaid/Uninsured/TRICARE), language.\n"
        "- **Evaluation factors**: same as relevant factors plus the "
        "5-bin LACE stratification.\n"
        "- **Group attributes that drive different performance**: "
        "Black + Indigenous + Medicaid + Uninsured patients show the "
        "largest TPR / DP gap (per the Phase 15 subgroup audit).\n"
        "- **Instrumentation**: predictions are deterministic given "
        "the LACE bin; ties are broken on tool name."
    )
    return ModelCardSection(title="3. Factors", body_md=body)


def _mc_metrics(root: Path) -> ModelCardSection:
    coeff = _load_json(root / "data" / "coefficients.json")
    s100 = _load_json(root / "data" / "synthea_100k_recalibration.json")
    mim = _load_json(root / "data" / "mimic_iv_recalibration.json")
    rows = ["| Cohort | n | ECE | Brier | AUROC |",
            "| --- | ---: | ---: | ---: | ---: |",
            "| W1 spec_002 | 7,880 | 0.0078 | 0.124 | 0.590 |"]
    if s100 and "metrics_overall" in s100:
        m = s100["metrics_overall"]
        rows.append(
            f"| Synthea-100k | {s100['cohort_n']:,} | "
            f"{m['ece']:.4f} | {m['brier']:.4f} | {m['auroc']:.4f} |"
        )
    if mim and "metrics_overall" in mim:
        m = mim["metrics_overall"]
        rows.append(
            f"| MIMIC-IV demo | {mim['cohort_n']:,} | "
            f"{m['ece']:.4f} | {m['brier']:.4f} | {m['auroc']:.4f} |"
        )
    body = (
        "**Performance measures**: ECE (Expected Calibration Error, "
        "preferred gate <=0.05), Brier score, AUROC. Decision-level "
        "TPR + Demographic-Parity gap surfaced per subgroup.\n\n"
        + "\n".join(rows)
        + "\n\n**Decision threshold**: 0.20. **Confidence interval**: "
        "5-bin Beta-Binomial posterior, width scaled by bin sample "
        "size."
    )
    return ModelCardSection(title="4. Metrics", body_md=body)


def _mc_evaluation_data(root: Path) -> ModelCardSection:
    body = (
        "- **Datasets**: (a) W1 internal calibration cohort "
        "(n=7,880); (b) Synthea-10k validation; (c) Synthea-100k "
        "recalibration; (d) MIMIC-IV demo 2.2 (n=275 in-cohort); "
        "(e) Phase-15 subgroup-audit synthetic cohort (n=100,000) "
        "with rich demographics.\n"
        "- **Motivation**: the W1 cohort drives the initial fit; "
        "Synthea-100k stress-tests calibration at scale; MIMIC-IV "
        "demo provides external validation; the subgroup-audit "
        "cohort drives fairness analysis.\n"
        "- **Preprocessing**: LACE features extracted from FHIR "
        "Bundle (length-of-stay from Encounter.period, acuity from "
        "Encounter.priority + chief complaint, Charlson from "
        "Conditions, ED visits from past Encounters)."
    )
    return ModelCardSection(title="5. Evaluation data", body_md=body)


def _mc_training_data() -> ModelCardSection:
    body = (
        "- **Training cohort**: W1 internal calibration workflow "
        "output, n=7,880, Synthea-generated synthetic 30-day "
        "readmission cohort with the LACE feature space.\n"
        "- **Synthea version**: 3.3.0 (BSD-3 licence). Generation "
        "controlled by the internal `fhir-readmission-calibration` "
        "workflow (W1).\n"
        "- **Outcome label**: binary 30-day all-cause readmission "
        "drawn from a fixed bucket map "
        "{LACE 0-2: 7.2%, 3-5: 10.3%, 6-9: 15.8%, 10-12: 23.4%, "
        "13+: 32.7%}; calibrated against AHRQ HCUP HRRP rates."
    )
    return ModelCardSection(title="6. Training data", body_md=body)


def _mc_quantitative_analyses(root: Path) -> ModelCardSection:
    audit = _load_json(root / "docs" / "fairness" / "subgroup_audit.json")
    pros = _load_json(
        root / "docs" / "prospective" / "prospective_eval.json")
    parts: list[str] = [
        "**Disaggregated evaluation results**: per Phase-15 "
        "subgroup-audit + prospective-eval artefacts."
    ]
    if audit:
        worst_eoo: tuple[str, str, float] | None = None
        worst_dp: tuple[str, str, float] | None = None
        for sg_name, sg in audit["subgroups"].items():
            for value, gap in sg["gaps"].items():
                eoo = abs(gap["eoo_gap_vs_reference"])
                dp = abs(gap["dp_gap_vs_reference"])
                if worst_eoo is None or eoo > worst_eoo[2]:
                    worst_eoo = (sg_name, value, eoo)
                if worst_dp is None or dp > worst_dp[2]:
                    worst_dp = (sg_name, value, dp)
        if worst_eoo and worst_dp:
            parts.append(
                f"\n- Worst EOO gap: "
                f"{worst_eoo[2]*100:.2f}% on "
                f"{worst_eoo[0]}={worst_eoo[1]}.\n"
                f"- Worst DP gap: {worst_dp[2]*100:.2f}% on "
                f"{worst_dp[0]}={worst_dp[1]}."
            )
    if pros:
        o = pros["overall"]
        parts.append(
            f"\n- Prospective eval (n={pros['cohort_n']:,}): "
            f"abstain={o['abstain_rate']*100:.2f}%, "
            f"downgrade={o['downgrade_rate']*100:.2f}%, "
            f"calibration gap={o['calibration_gap_abs']*100:.2f}%."
        )
    return ModelCardSection(
        title="7. Quantitative analyses",
        body_md="".join(parts),
    )


def _mc_ethical_considerations() -> ModelCardSection:
    body = (
        "- **Sensitive data**: predictions consume protected "
        "demographics (race, ethnicity, insurance). The 4-critic "
        "ensemble's `fairness_critic` blocks any recommendation that "
        "would adversely change for a flagged subgroup at matched "
        "LACE.\n"
        "- **Mitigation strategies**: deterministic abstain trigger "
        "for demographic-bias subgroups without a fresh fairness "
        "audit; differential-privacy publication of subgroup rates "
        "(Laplace, epsilon=1.0).\n"
        "- **Risks of harm**: under-prediction in Black + Indigenous "
        "+ Medicaid populations is documented in the literature "
        "(AHRQ HCUP 2022, Joynt & Jha 2014). The system surfaces "
        "this risk directly via the Fairness Watch block."
    )
    return ModelCardSection(
        title="8. Ethical considerations",
        body_md=body,
    )


def _mc_caveats_and_recommendations() -> ModelCardSection:
    body = (
        "- **Caveats**: the LACE feature space is reductive. Real "
        "deployments should re-fit on local data and re-run the "
        "fairness audit on the local cohort before release.\n"
        "- **Recommendations**: do not deploy without a clinician "
        "in the loop; do not deploy on cohorts where sub-population "
        "coverage falls below the suppression threshold (n=20 raw); "
        "re-calibrate quarterly, monitor drift continuously."
    )
    return ModelCardSection(
        title="9. Caveats and recommendations",
        body_md=body,
    )


def build_model_card(project_root: Path | None = None) -> ModelCard:
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    sections = [
        _mc_details(), _mc_intended_use(), _mc_factors(),
        _mc_metrics(project_root),
        _mc_evaluation_data(project_root),
        _mc_training_data(),
        _mc_quantitative_analyses(project_root),
        _mc_ethical_considerations(),
        _mc_caveats_and_recommendations(),
    ]
    return ModelCard(
        generated_at=datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"),
        n_sections=len(sections),
        sections=sections,
    )


def render_model_card_md(card: ModelCard) -> str:
    lines = [f"# Model Card: {card.model_name}", ""]
    lines.append(
        f"**Generated**: {card.generated_at} - "
        f"**Version**: {card.model_version} - "
        f"**Sections**: {card.n_sections}"
    )
    lines.append("")
    lines.append(
        "Format: Mitchell et al. 2019 (FAT*) Model Cards for Model "
        "Reporting."
    )
    lines.append("")
    for s in card.sections:
        lines.append(f"## {s.title}")
        lines.append("")
        lines.append(s.body_md)
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Datasheet for Datasets (Gebru 2021)
# ─────────────────────────────────────────────────────────────────────


class DatasheetSection(BaseModel):
    title: str
    body_md: str


class Datasheet(BaseModel):
    generated_at: str
    dataset_name: str = (
        "TrustedRisk synthetic cohorts (W1 + Synthea-100k + "
        "subgroup-audit + prospective-eval)"
    )
    n_sections: int = Field(ge=0)
    sections: list[DatasheetSection]


def _ds_motivation() -> DatasheetSection:
    body = (
        "- **For what purpose was the dataset created?** To fit and "
        "validate a 30-day readmission risk model using the LACE "
        "feature space, plus to drive the fairness audit "
        "and prospective evaluation. The pipeline is designed as a "
        "research prototype, not for clinical deployment.\n"
        "- **Funding**: independent research (no grant funding).\n"
        "- **Conflicts of interest**: none disclosed."
    )
    return DatasheetSection(title="1. Motivation", body_md=body)


def _ds_composition() -> DatasheetSection:
    body = (
        "- **Instances**: synthetic patient encounters with LACE "
        "components (L, A, C, E), simulated demographics (age, sex, "
        "race, ethnicity, insurance, language), 30-day readmission "
        "binary outcome.\n"
        "- **Number of instances per cohort**: W1 n=7,880; "
        "Synthea-100k n=100,000; Phase-15 subgroup-audit cohort "
        "n=100,000; prospective-eval cohort n=10,000.\n"
        "- **Sample weighting**: the subgroup-audit + prospective "
        "cohorts use ground-truth distributions skewed toward US "
        "Medicare-age populations (60% age >=65).\n"
        "- **Sensitive attributes**: race, ethnicity, insurance, "
        "language are present and used in evaluation; no real PHI "
        "is contained."
    )
    return DatasheetSection(title="2. Composition", body_md=body)


def _ds_collection() -> DatasheetSection:
    body = (
        "- **Acquisition method**: programmatic generation via "
        "Synthea (BSD-3) for W1 + Synthea-100k; deterministic "
        "seeded sampling for the Phase-15 cohorts (seed=20260430).\n"
        "- **Sampling strategy**: stratified by LACE bin marginals "
        "matching W1; outcomes drawn from fixed calibrated bucket "
        "rates {0.072, 0.103, 0.158, 0.234, 0.327}.\n"
        "- **Time period**: encounters generated 2026-04-24 to "
        "2026-04-30. No real time-stamped patient data."
    )
    return DatasheetSection(title="3. Collection process", body_md=body)


def _ds_preprocessing() -> DatasheetSection:
    body = (
        "- **Cleaning**: none required - generators produce "
        "well-formed LACE features by construction.\n"
        "- **Filtering**: encounters with LACE > 19 are clipped to "
        "19 (the upper bound of the calibrated bucket map).\n"
        "- **Software used**: Python stdlib only; no Pandas/NumPy "
        "dependency in the cohort generation path."
    )
    return DatasheetSection(
        title="4. Preprocessing / cleaning / labeling",
        body_md=body,
    )


def _ds_uses() -> DatasheetSection:
    body = (
        "- **Has the dataset been used?**: yes - W1 calibration, "
        "subgroup-audit (Phase 15.B1), prospective-eval (Phase "
        "15.C), regulatory-pack fairness section (Phase 14.17 P1).\n"
        "- **Other tasks the dataset could be used for**: any "
        "binary classification benchmark on the LACE feature space; "
        "fairness research on US-shaped cohorts; calibration "
        "research (the dataset is well-calibrated by construction "
        "so it provides a clean baseline).\n"
        "- **Tasks the dataset should NOT be used for**: real "
        "clinical decision-making (it's synthetic); training a "
        "classifier intended for a non-US setting (the demographic "
        "and outcome multipliers are US-anchored)."
    )
    return DatasheetSection(title="5. Uses", body_md=body)


def _ds_distribution() -> DatasheetSection:
    body = (
        "- **Distribution channel**: included in the TrustedRisk "
        "GitHub repository under `data/` and `docs/fairness/` + "
        "`docs/prospective/`.\n"
        "- **License**: Apache 2.0.\n"
        "- **Format**: JSON (artefacts) + Python source for the "
        "generators."
    )
    return DatasheetSection(title="6. Distribution", body_md=body)


def _ds_maintenance() -> DatasheetSection:
    body = (
        "- **Who is supporting?**: TrustedRisk team.\n"
        "- **Update cadence**: re-run on every release of the "
        "TrustedRisk codebase; the generators are deterministic so "
        "regenerating is free.\n"
        "- **Erratum / errata**: none currently. Issues are tracked "
        "via GitHub.\n"
        "- **Will the dataset be archived?**: yes - the JSON "
        "artefacts are committed to the repository, and the source "
        "generators are versioned alongside the code."
    )
    return DatasheetSection(title="7. Maintenance", body_md=body)


def build_datasheet() -> Datasheet:
    sections = [
        _ds_motivation(), _ds_composition(), _ds_collection(),
        _ds_preprocessing(), _ds_uses(), _ds_distribution(),
        _ds_maintenance(),
    ]
    return Datasheet(
        generated_at=datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"),
        n_sections=len(sections),
        sections=sections,
    )


def render_datasheet_md(ds: Datasheet) -> str:
    lines = [f"# Datasheet: {ds.dataset_name}", ""]
    lines.append(
        f"**Generated**: {ds.generated_at} - "
        f"**Sections**: {ds.n_sections}"
    )
    lines.append("")
    lines.append("Format: Gebru et al. 2021 (CACM) Datasheets for Datasets.")
    lines.append("")
    for s in ds.sections:
        lines.append(f"## {s.title}")
        lines.append("")
        lines.append(s.body_md)
        lines.append("")
    return "\n".join(lines)
