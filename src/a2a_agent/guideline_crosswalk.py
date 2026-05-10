"""Phase 17.P - Clinical guideline crosswalk for the 145-tool surface.

For every registered MCP tool produces a row of:

    (tool_name, bundle, guideline_source, guideline_url, year,
     level_of_evidence)

Pure-data; the table is hand-curated for ~50 canonical scores +
falls back to a per-bundle default for the long tail. Re-derives
the live tool list from `mcp_server.tools.BUNDLES` so a new tool
automatically lands in the report (under its bundle's default).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GuidelineRow(BaseModel):
    tool_name: str
    bundle: str
    guideline_source: str
    guideline_url: str
    year: int = Field(ge=1900, le=2030)
    level_of_evidence: str


class GuidelineCrosswalkReport(BaseModel):
    n_rows: int = Field(ge=0)
    n_tools_total: int = Field(ge=0)
    n_with_specific_guideline: int
    rows: list[GuidelineRow]


# Per-tool authoritative overrides
_TOOL_OVERRIDES: dict[str, tuple[str, str, int, str]] = {
    "compute_readmission_risk": (
        "AHRQ HCUP Statistical Brief #248 (HRRP cohort)",
        "https://www.hcup-us.ahrq.gov/reports/statbriefs/sb248.jsp",
        2019, "Level III - retrospective cohort"),
    "compute_heart_score": (
        "Six et al. 2008 - HEART score for chest pain",
        "https://doi.org/10.1097/HCO.0b013e328329caa6",
        2008, "Level I - prospective validation"),
    "compute_stroke_severity": (
        "NIH Stroke Scale (NINDS 1989, AHA/ASA 2019)",
        "https://www.stroke.org/-/media/stroke-files/nihss.pdf",
        2019, "Level I - clinical guideline"),
    "compute_stroke_thrombolysis_eligibility": (
        "AHA/ASA 2019 - Guidelines for the Early Management of "
        "Acute Ischemic Stroke",
        "https://doi.org/10.1161/STR.0000000000000211",
        2019, "Level I - clinical guideline"),
    "compute_aki_kdigo_stage": (
        "KDIGO Clinical Practice Guideline for AKI",
        "https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf",
        2012, "Level I - clinical guideline"),
    "compute_dialysis_initiation_decision": (
        "STARRT-AKI 2020 + KDIGO 2012",
        "https://doi.org/10.1056/NEJMoa2000741",
        2020, "Level I - RCT"),
    "compute_cha2ds2_vasc": (
        "Lip et al. 2010 - refined stroke risk in non-valvular AF",
        "https://doi.org/10.1378/chest.09-1584",
        2010, "Level I - large cohort + ESC 2020"),
    "compute_has_bled": (
        "Pisters et al. 2010 - HAS-BLED score",
        "https://doi.org/10.1378/chest.10-0134",
        2010, "Level I - prospective derivation"),
    "compute_grace_acs_score": (
        "GRACE Investigators 2003 / 2014",
        "https://doi.org/10.1136/heart.89.7.755",
        2014, "Level I - large registry"),
    "compute_timi_acs_score": (
        "Antman et al. 2000 - TIMI risk score for UA/NSTEMI",
        "https://doi.org/10.1001/jama.284.7.835",
        2000, "Level I - derivation + validation"),
    "compute_apache_ii_score": (
        "Knaus et al. 1985 - APACHE II severity of disease",
        "https://doi.org/10.1097/00003246-198510000-00009",
        1985, "Level I - large cohort"),
    "compute_sofa_score": (
        "Singer et al. 2016 - Sepsis-3 + Vincent 1996 SOFA",
        "https://doi.org/10.1001/jama.2016.0287",
        2016, "Level I - international consensus"),
    "compute_meld_score": (
        "Kamath et al. 2007 - MELD-Na update",
        "https://doi.org/10.1056/NEJMoa0801209",
        2007, "Level I - validation cohort"),
    "compute_rifle_aki_classification": (
        "ADQI Group 2004 - RIFLE consensus",
        "https://doi.org/10.1186/cc2872",
        2004, "Level I - international consensus"),
    "compute_qsofa_score": (
        "Sepsis-3 (JAMA 2016)",
        "https://doi.org/10.1001/jama.2016.0287",
        2016, "Level I - international consensus"),
    "compute_lactate_clearance": (
        "Surviving Sepsis Campaign 2021",
        "https://doi.org/10.1097/CCM.0000000000005337",
        2021, "Level I - clinical guideline"),
    "compute_das28_rheumatoid_arthritis": (
        "Prevoo et al. 1995 - DAS28 derivation",
        "https://doi.org/10.1002/art.1780380107",
        1995, "Level I - validation cohort"),
    "compute_asdas_axspa": (
        "ASAS - ASDAS validation 2009",
        "https://doi.org/10.1136/ard.2008.094870",
        2009, "Level I - validation"),
    "compute_acr_eular_ra_classification": (
        "ACR/EULAR 2010 RA classification criteria",
        "https://doi.org/10.1002/art.27584",
        2010, "Level I - international consensus"),
    "compute_rcri_cardiac_risk": (
        "Lee et al. 1999 - Revised Cardiac Risk Index",
        "https://doi.org/10.1161/01.CIR.100.10.1043",
        1999, "Level I - prospective derivation"),
    "compute_ariscat_pulmonary_risk": (
        "Canet et al. 2010 - ARISCAT",
        "https://doi.org/10.1097/ALN.0b013e3181fc6e0a",
        2010, "Level I - prospective"),
    "compute_caprini_vte_risk": (
        "Caprini 2005 - VTE risk assessment model",
        "https://doi.org/10.1016/j.jvs.2009.03.027",
        2005, "Level I - validation cohort"),
    "compute_maddrey_alcoholic_hepatitis": (
        "Maddrey et al. 1978 + AASLD 2019 update",
        "https://doi.org/10.1002/hep.30866",
        2019, "Level I - clinical guideline"),
    "compute_fib4_liver_fibrosis": (
        "Sterling et al. 2006 - FIB-4 derivation",
        "https://doi.org/10.1002/hep.21178",
        2006, "Level I - cohort validation"),
    "compute_glasgow_blatchford_ugib": (
        "Blatchford et al. 2000 - GBS validation",
        "https://doi.org/10.1016/S0140-6736(00)02816-6",
        2000, "Level I - prospective"),
    "compute_rome_iv_ibs": (
        "Rome IV Foundation 2016",
        "https://theromefoundation.org/rome-iv/",
        2016, "Level I - international consensus"),
    "compute_hunt_hess_sah": (
        "Hunt-Hess scale (1968) + Connolly et al. 2012 AHA/ASA",
        "https://doi.org/10.1161/STR.0b013e3182587839",
        2012, "Level I - clinical guideline"),
    "compute_ich_score": (
        "Hemphill et al. 2001 - ICH score",
        "https://doi.org/10.1161/01.STR.32.4.891",
        2001, "Level I - prospective derivation"),
    "compute_modified_rankin": (
        "Quinn et al. 2009 - mRS reliability validation",
        "https://doi.org/10.1161/STROKEAHA.108.541128",
        2009, "Level I - international consensus"),
    "compute_hauser_ambulation_index": (
        "Hauser et al. 1983 - Ambulation Index",
        "https://doi.org/10.1212/WNL.33.11.1444",
        1983, "Level II - clinical scoring"),
    "compute_bishop_induction_score": (
        "Bishop 1964 + ACOG 2009",
        "https://www.acog.org/clinical/clinical-guidance",
        2009, "Level I - clinical guideline"),
    "compute_apgar_score": (
        "Apgar 1953 + AAP/ACOG 2015 reaffirmation",
        "https://doi.org/10.1542/peds.2015-2651",
        2015, "Level I - clinical guideline"),
    "compute_bell_nec_stage": (
        "Bell et al. 1978 (modified Bell + Walsh 1986)",
        "https://doi.org/10.1542/peds.78.3.460",
        1986, "Level I - large cohort"),
    "compute_bilirubin_nomogram": (
        "AAP 2022 - Hyperbilirubinemia clinical practice guideline",
        "https://doi.org/10.1542/peds.2022-058859",
        2022, "Level I - clinical guideline"),
    "compute_iss_myeloma_staging": (
        "Greipp et al. 2005 + IMWG 2015 (R-ISS)",
        "https://doi.org/10.1200/JCO.2005.04.242",
        2015, "Level I - international consensus"),
    "compute_ipss_r_mds_score": (
        "Greenberg et al. 2012 - IPSS-R",
        "https://doi.org/10.1182/blood-2012-03-420489",
        2012, "Level I - international consensus"),
    "compute_ecog_performance_status": (
        "Oken et al. 1982 - ECOG criteria",
        "https://doi.org/10.1097/00000421-198212000-00014",
        1982, "Level I - clinical scoring"),
    "compute_karnofsky_performance": (
        "Karnofsky & Burchenal 1949",
        "",
        1949, "Level II - clinical scoring"),
    "compute_thyroid_management": (
        "ATA 2014 + ETA 2013 hypothyroidism guidelines",
        "https://doi.org/10.1089/thy.2014.0028",
        2014, "Level I - clinical guideline"),
    "compute_kdpi_kidney_donor": (
        "OPTN/UNOS KDPI calculator (2014)",
        "https://optn.transplant.hrsa.gov/data/allocation-calculators/kdpi-calculator/",
        2014, "Level I - operational"),
    "compute_epts_recipient_score": (
        "OPTN/UNOS EPTS calculator (2014)",
        "https://optn.transplant.hrsa.gov/data/allocation-calculators/epts-calculator/",
        2014, "Level I - operational"),
    "compute_epworth_sleepiness_scale": (
        "Johns 1991 - Epworth scale",
        "https://doi.org/10.1093/sleep/14.6.540",
        1991, "Level I - validation cohort"),
    "compute_stop_bang_osa_screen": (
        "Chung et al. 2008 - STOP-BANG questionnaire",
        "https://doi.org/10.1097/ALN.0b013e31816d83e4",
        2008, "Level I - prospective"),
    "compute_dn4_neuropathic_pain": (
        "Bouhassira et al. 2005 - DN4",
        "https://doi.org/10.1016/j.pain.2004.12.010",
        2005, "Level I - validation"),
    "compute_pediatric_early_warning": (
        "Brighton/Monaghan PEWS (2005, validated 2017)",
        "https://doi.org/10.1542/peds.2017-2966",
        2017, "Level I - large cohort"),
    "compute_clinical_deterioration_score": (
        "RCP NEWS2 (2017)",
        "https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2",
        2017, "Level I - clinical guideline"),
    "compute_dka_severity": (
        "ADA Position Statement 2009 + ISPAD 2018",
        "https://doi.org/10.1111/pedi.12701",
        2018, "Level I - clinical guideline"),
    "compute_charlson_elixhauser_index": (
        "Quan et al. 2005 + van Walraven 2009",
        "https://doi.org/10.1097/01.mlr.0000182534.19832.83",
        2009, "Level I - cohort validation"),
    "compute_falls_risk_morse": (
        "Morse et al. 1989 - Morse Falls Scale",
        "https://doi.org/10.1037/t02948-000",
        1989, "Level I - validation"),
    "compute_delirium_screening_cam": (
        "Inouye et al. 1990 - Confusion Assessment Method",
        "https://doi.org/10.7326/0003-4819-113-12-941",
        1990, "Level I - validation"),
    "compute_maternal_early_warning": (
        "Singh et al. 2012 - MEOWS",
        "https://doi.org/10.1111/j.1365-2044.2011.06916.x",
        2012, "Level I - validation"),
    "compute_preeclampsia_assessment": (
        "ACOG 2020 Practice Bulletin 222",
        "https://doi.org/10.1097/AOG.0000000000003891",
        2020, "Level I - clinical guideline"),
    "compute_trauma_severity_score": (
        "Baker et al. 1974 (ISS) + Champion 1989 (RTS)",
        "",
        1989, "Level I - international convention"),
    "compute_massive_transfusion_protocol": (
        "PROPPR 2015 + CRASH-2 2010",
        "https://doi.org/10.1001/jama.2015.12",
        2015, "Level I - RCT"),
    "compute_imaging_appropriateness": (
        "ACR Appropriateness Criteria + Choosing Wisely",
        "https://www.acr.org/Clinical-Resources/ACR-Appropriateness-Criteria",
        2024, "Level I - clinical guideline"),
    "compute_contrast_safety_check": (
        "ACR Manual on Contrast Media v2024 + Davenport 2020",
        "https://www.acr.org/Clinical-Resources/Contrast-Manual",
        2024, "Level I - clinical guideline"),
    "compute_hl7v2_message_parse": (
        "HL7 v2.x Standard, Chapter 3 (Patient Administration)",
        "https://www.hl7.org/implement/standards/product_brief.cfm?product_id=185",
        2007, "Level I - standards body"),
    "compute_ccda_document_parse": (
        "HL7 C-CDA R2.1 Implementation Guide",
        "https://www.hl7.org/implement/standards/product_brief.cfm?product_id=492",
        2018, "Level I - standards body"),
    "compute_cox_proportional_hazards": (
        "Cox 1972 - Regression models and life-tables",
        "https://doi.org/10.1111/j.2517-6161.1972.tb00899.x",
        1972, "Level I - methodological"),
    "compute_shap_attribution": (
        "Lundberg & Lee 2017 - A unified approach to interpreting "
        "model predictions",
        "https://arxiv.org/abs/1705.07874",
        2017, "Level I - methodological"),
    "compute_ensemble_stacking": (
        "Wolpert 1992 - Stacked generalization",
        "https://doi.org/10.1016/S0893-6080(05)80023-1",
        1992, "Level I - methodological"),
}


# Per-bundle defaults
_BUNDLE_DEFAULTS: dict[str, tuple[str, str, int, str]] = {
    "core_discharge": (
        "AHRQ HCUP HRRP cohort",
        "https://www.hcup-us.ahrq.gov/reports/statbriefs/sb248.jsp",
        2019, "Level III - retrospective cohort"),
    "ed_acute": (
        "AHRQ ED triage standards + RCP NEWS2",
        "https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2",
        2017, "Level I - clinical guideline"),
    "antimicrobial": (
        "IDSA / ATS clinical guidelines",
        "https://www.idsociety.org/practice-guideline/practice-guidelines/",
        2024, "Level I - clinical guideline"),
    "oncology": (
        "NCCN Clinical Practice Guidelines in Oncology",
        "https://www.nccn.org/guidelines/category_1",
        2024, "Level I - clinical guideline"),
    "stroke_acs": (
        "AHA/ACC + AHA/ASA composite",
        "https://professional.heart.org/en/guidelines-and-statements",
        2024, "Level I - clinical guideline"),
    "obstetric_geriatric": (
        "ACOG + AGS Beers Criteria composite",
        "https://www.acog.org/clinical",
        2024, "Level I - clinical guideline"),
    "trauma_critical": (
        "ATLS 10th Edition + ACS-COT",
        "https://www.facs.org/quality-programs/trauma/atls/",
        2024, "Level I - clinical guideline"),
    "endocrine_acute": (
        "ADA Standards of Care",
        "https://diabetesjournals.org/care/issue/47/Supplement_1",
        2024, "Level I - clinical guideline"),
    "imaging": (
        "ACR Appropriateness Criteria",
        "https://www.acr.org/Clinical-Resources/ACR-Appropriateness-Criteria",
        2024, "Level I - clinical guideline"),
    "nephrology": (
        "KDIGO Clinical Practice Guidelines",
        "https://kdigo.org/guidelines/",
        2024, "Level I - clinical guideline"),
    "rheumatology": (
        "ACR/EULAR Recommendations",
        "https://www.rheumatology.org/Practice-Quality/Clinical-Support/Clinical-Practice-Guidelines",
        2024, "Level I - clinical guideline"),
    "peri_op_risk": (
        "ACC/AHA peri-operative + AAGBI",
        "https://professional.heart.org/en/guidelines-and-statements",
        2024, "Level I - clinical guideline"),
    "infectious_disease": (
        "IDSA / Surviving Sepsis Campaign",
        "https://www.survivingsepsis.org/",
        2024, "Level I - clinical guideline"),
    "gi_hepatology_depth": (
        "AGA / AASLD clinical practice guidelines",
        "https://www.aasld.org/publications/practice-guidelines",
        2024, "Level I - clinical guideline"),
    "neurology_depth": (
        "AAN / AHA-ASA composite",
        "https://www.aan.com/Guidelines",
        2024, "Level I - clinical guideline"),
    "ob_peds_advanced": (
        "ACOG + AAP composite",
        "https://www.aap.org/en/practice-management/policies",
        2024, "Level I - clinical guideline"),
    "cardiology_depth": (
        "ACC/AHA + ESC composite",
        "https://professional.heart.org/en/guidelines-and-statements",
        2024, "Level I - clinical guideline"),
    "heme_onc_depth": (
        "NCCN + IMWG + IPSS composite",
        "https://www.nccn.org/guidelines/category_1",
        2024, "Level I - clinical guideline"),
    "endocrinology_advanced": (
        "Endocrine Society + ATA + AACE composite",
        "https://www.endocrine.org/clinical-practice-guidelines",
        2024, "Level I - clinical guideline"),
    "sleep_pain": (
        "AASM + IASP composite",
        "https://aasm.org/clinical-resources/practice-standards/",
        2024, "Level I - clinical guideline"),
    "transplant": (
        "OPTN/UNOS allocation policy + KDIGO transplantation",
        "https://optn.transplant.hrsa.gov/policies-bylaws/policies/",
        2024, "Level I - operational policy"),
    "specialty_clinics": (
        "AAD / AAO / ATS composite",
        "https://www.aad.org/member/clinical-quality/guidelines",
        2024, "Level I - clinical guideline"),
    "critical_care": (
        "SCCM ICU practice + Surviving Sepsis",
        "https://www.sccm.org/Clinical-Resources/Guidelines",
        2024, "Level I - clinical guideline"),
    "model_research": (
        "Methodological - peer-reviewed source",
        "",
        2017, "Level I - methodological"),
    "legacy_ehr_parsers": (
        "HL7 standards body specifications",
        "https://www.hl7.org/implement/standards/",
        2024, "Level I - standards body"),
    "patient_facing": (
        "AHRQ Re-engineered Discharge (RED) toolkit",
        "https://www.ahrq.gov/patient-safety/settings/hospital/red/toolkit/index.html",
        2017, "Level I - clinical guideline"),
    "data_normalization": (
        "Regenstrief LOINC + RxNorm + UMLS Metathesaurus",
        "https://www.nlm.nih.gov/research/umls/index.html",
        2024, "Level I - standards body"),
    "clinical_workflow": (
        "AHRQ National Quality Forum measures",
        "https://www.qualityforum.org/Measures_Reports_Tools.aspx",
        2024, "Level I - clinical guideline"),
    "external_knowledge": (
        "PubMed + ClinicalTrials.gov + NIH RePORTER",
        "https://pubmed.ncbi.nlm.nih.gov/",
        2024, "Level I - primary literature"),
    "chart_intelligence": (
        "AHRQ NLP guidelines for clinical text",
        "https://digital.ahrq.gov/",
        2024, "Level II - methodological"),
    "context_resolution": (
        "SMART on FHIR + SHARP launch context",
        "https://hl7.org/fhir/smart-app-launch/",
        2022, "Level I - standards body"),
    "diagnosis": (
        "USMLE / NEJM diagnostic reasoning curricula",
        "https://www.nejm.org/medical-articles/case-records-of-the-massachusetts-general-hospital",
        2024, "Level II - editorial"),
    "auto_coding": (
        "ICD-10-CM Official Guidelines + AMA CPT",
        "https://www.cms.gov/medicare/coding-billing/icd-10-codes",
        2024, "Level I - operational policy"),
    "pharmacogenomics": (
        "CPIC Clinical Pharmacogenetics Implementation Consortium",
        "https://cpicpgx.org/guidelines/",
        2024, "Level I - clinical guideline"),
    "preadmit_triage": (
        "ACEP red-flag + AHRQ self-care triage",
        "https://www.acep.org/patient-care/clinical-policies",
        2024, "Level I - clinical guideline"),
    "research_design": (
        "BMJ N-of-1 trials methodology + ICH-GCP",
        "https://www.bmj.com/content/336/7637/152",
        2008, "Level I - methodological"),
    "quality_stars": (
        "CMS Star Rating Methodology + NCQA HEDIS",
        "https://www.cms.gov/medicare/health-plans/medicareadvtgspecratestats/medicareadvantagepartcandd",
        2024, "Level I - operational policy"),
    "population_health": (
        "CDC NSSP + ACIP recommendations",
        "https://www.cdc.gov/nssp/",
        2024, "Level I - operational policy"),
    "insurance_appeals": (
        "ERISA + ACA external review + state insurance commissioner",
        "https://www.dol.gov/agencies/ebsa/laws-and-regulations/laws/erisa",
        2024, "Level I - statutory"),
    "multimodal": (
        "AHA/ACC/HRS QT analysis + DICOM SR Part-3",
        "https://www.dicomstandard.org/current",
        2024, "Level I - standards body"),
    "fhir_writeback": (
        "HL7 FHIR R4 write-back transactions",
        "https://www.hl7.org/fhir/http.html",
        2022, "Level I - standards body"),
    "prior_authorization": (
        "AMA PA reform + AHIP Fast PASS",
        "https://www.ama-assn.org/practice-management/sustainability/prior-authorization-reform-resources",
        2024, "Level I - operational policy"),
    "clinical_documentation": (
        "Joint Commission documentation standards",
        "https://www.jointcommission.org/standards/",
        2024, "Level I - operational policy"),
    "patient_qa": (
        "AHRQ Re-engineered Discharge (RED) + health-literacy",
        "https://www.ahrq.gov/patient-safety/settings/hospital/red/toolkit/index.html",
        2017, "Level I - clinical guideline"),
    "mental_health": (
        "C-SSRS + APA practice guidelines",
        "https://www.psychiatry.org/psychiatrists/practice/clinical-practice-guidelines",
        2024, "Level I - clinical guideline"),
    "pediatric": (
        "AAP Bright Futures + Brighton/Monaghan PEWS",
        "https://www.aap.org/en/practice-management/policies",
        2024, "Level I - clinical guideline"),
    "economics": (
        "AHRQ HCUP cost data + ICER value-based pricing",
        "https://www.hcup-us.ahrq.gov/",
        2024, "Level I - operational"),
}


def build_guideline_crosswalk() -> GuidelineCrosswalkReport:
    from mcp_server.tools import BUNDLES   # type: ignore

    seen: set[str] = set()
    rows: list[GuidelineRow] = []
    n_specific = 0
    for bundle, tools in BUNDLES.items():
        if bundle not in _BUNDLE_DEFAULTS:
            raise KeyError(
                f"Bundle {bundle!r} has no entry in _BUNDLE_DEFAULTS. "
                f"Add a guideline reference before proceeding."
            )
        default = _BUNDLE_DEFAULTS[bundle]
        for tool in tools:
            if tool in seen:
                continue
            seen.add(tool)
            override = _TOOL_OVERRIDES.get(tool)
            if override is not None:
                src, url, year, level = override
                n_specific += 1
            else:
                src, url, year, level = default
            rows.append(GuidelineRow(
                tool_name=tool, bundle=bundle,
                guideline_source=src,
                guideline_url=url, year=year,
                level_of_evidence=level,
            ))
    rows.sort(key=lambda r: (r.bundle, r.tool_name))
    return GuidelineCrosswalkReport(
        n_rows=len(rows),
        n_tools_total=len(rows),
        n_with_specific_guideline=n_specific,
        rows=rows,
    )


def render_guideline_crosswalk_md(
    report: GuidelineCrosswalkReport,
) -> str:
    lines: list[str] = [
        "# TrustedRisk - Clinical Guideline Crosswalk", ""]
    lines.append(
        f"**Tools mapped**: {report.n_rows} | "
        f"**With tool-specific guideline**: "
        f"{report.n_with_specific_guideline} | "
        f"**With bundle-default guideline**: "
        f"{report.n_rows - report.n_with_specific_guideline}"
    )
    lines.append("")
    current_bundle: str | None = None
    for r in report.rows:
        if r.bundle != current_bundle:
            current_bundle = r.bundle
            lines.append(f"## {r.bundle}")
            lines.append("")
            lines.append(
                "| Tool | Guideline source | Year | Level of "
                "evidence |"
            )
            lines.append("| --- | --- | ---: | --- |")
        url_label = (
            f"[{r.guideline_source}]({r.guideline_url})"
            if r.guideline_url else r.guideline_source
        )
        lines.append(
            f"| `{r.tool_name}` | {url_label} | {r.year} | "
            f"{r.level_of_evidence} |"
        )
    lines.append("")
    return "\n".join(lines)
