# TrustedRisk - Clinical Guideline Crosswalk

**Tools mapped**: 145 | **With tool-specific guideline**: 61 | **With bundle-default guideline**: 84

## antimicrobial

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_antibiotic_de_escalation` | [IDSA / ATS clinical guidelines](https://www.idsociety.org/practice-guideline/practice-guidelines/) | 2024 | Level I - clinical guideline |
| `compute_empiric_antibiotic_selection` | [IDSA / ATS clinical guidelines](https://www.idsociety.org/practice-guideline/practice-guidelines/) | 2024 | Level I - clinical guideline |
## auto_coding

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_coding_audit` | [ICD-10-CM Official Guidelines + AMA CPT](https://www.cms.gov/medicare/coding-billing/icd-10-codes) | 2024 | Level I - operational policy |
| `compute_cpt_suggest` | [ICD-10-CM Official Guidelines + AMA CPT](https://www.cms.gov/medicare/coding-billing/icd-10-codes) | 2024 | Level I - operational policy |
| `compute_hcpcs_suggest` | [ICD-10-CM Official Guidelines + AMA CPT](https://www.cms.gov/medicare/coding-billing/icd-10-codes) | 2024 | Level I - operational policy |
| `compute_icd10_suggest` | [ICD-10-CM Official Guidelines + AMA CPT](https://www.cms.gov/medicare/coding-billing/icd-10-codes) | 2024 | Level I - operational policy |
## cardiology_depth

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_cha2ds2_vasc` | [Lip et al. 2010 - refined stroke risk in non-valvular AF](https://doi.org/10.1378/chest.09-1584) | 2010 | Level I - large cohort + ESC 2020 |
| `compute_grace_acs_score` | [GRACE Investigators 2003 / 2014](https://doi.org/10.1136/heart.89.7.755) | 2014 | Level I - large registry |
| `compute_has_bled` | [Pisters et al. 2010 - HAS-BLED score](https://doi.org/10.1378/chest.10-0134) | 2010 | Level I - prospective derivation |
| `compute_timi_acs_score` | [Antman et al. 2000 - TIMI risk score for UA/NSTEMI](https://doi.org/10.1001/jama.284.7.835) | 2000 | Level I - derivation + validation |
## chart_intelligence

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_clinical_ner` | [AHRQ NLP guidelines for clinical text](https://digital.ahrq.gov/) | 2024 | Level II - methodological |
| `compute_negation_temporal` | [AHRQ NLP guidelines for clinical text](https://digital.ahrq.gov/) | 2024 | Level II - methodological |
| `compute_structure_discharge_summary` | [AHRQ NLP guidelines for clinical text](https://digital.ahrq.gov/) | 2024 | Level II - methodological |
## clinical_documentation

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_admission_hnp_draft` | [Joint Commission documentation standards](https://www.jointcommission.org/standards/) | 2024 | Level I - operational policy |
| `compute_consult_letter_draft` | [Joint Commission documentation standards](https://www.jointcommission.org/standards/) | 2024 | Level I - operational policy |
| `compute_discharge_summary_draft` | [Joint Commission documentation standards](https://www.jointcommission.org/standards/) | 2024 | Level I - operational policy |
| `compute_progress_note_draft` | [Joint Commission documentation standards](https://www.jointcommission.org/standards/) | 2024 | Level I - operational policy |
## clinical_workflow

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_care_gap_detector` | [AHRQ National Quality Forum measures](https://www.qualityforum.org/Measures_Reports_Tools.aspx) | 2024 | Level I - clinical guideline |
| `compute_medication_adherence_predictor` | [AHRQ National Quality Forum measures](https://www.qualityforum.org/Measures_Reports_Tools.aspx) | 2024 | Level I - clinical guideline |
| `compute_order_set` | [AHRQ National Quality Forum measures](https://www.qualityforum.org/Measures_Reports_Tools.aspx) | 2024 | Level I - clinical guideline |
| `compute_prom_influence` | [AHRQ National Quality Forum measures](https://www.qualityforum.org/Measures_Reports_Tools.aspx) | 2024 | Level I - clinical guideline |
## context_resolution

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_resolve_patient_from_query` | [SMART on FHIR + SHARP launch context](https://hl7.org/fhir/smart-app-launch/) | 2022 | Level I - standards body |
## core_discharge

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_counterfactual_explanation` | [AHRQ HCUP HRRP cohort](https://www.hcup-us.ahrq.gov/reports/statbriefs/sb248.jsp) | 2019 | Level III - retrospective cohort |
| `compute_decision_utility` | [AHRQ HCUP HRRP cohort](https://www.hcup-us.ahrq.gov/reports/statbriefs/sb248.jsp) | 2019 | Level III - retrospective cohort |
| `compute_discharge_counseling` | [AHRQ HCUP HRRP cohort](https://www.hcup-us.ahrq.gov/reports/statbriefs/sb248.jsp) | 2019 | Level III - retrospective cohort |
| `compute_fairness_audit` | [AHRQ HCUP HRRP cohort](https://www.hcup-us.ahrq.gov/reports/statbriefs/sb248.jsp) | 2019 | Level III - retrospective cohort |
| `compute_lab_trend_analysis` | [AHRQ HCUP HRRP cohort](https://www.hcup-us.ahrq.gov/reports/statbriefs/sb248.jsp) | 2019 | Level III - retrospective cohort |
| `compute_medication_reconciliation` | [AHRQ HCUP HRRP cohort](https://www.hcup-us.ahrq.gov/reports/statbriefs/sb248.jsp) | 2019 | Level III - retrospective cohort |
| `compute_readmission_risk` | [AHRQ HCUP Statistical Brief #248 (HRRP cohort)](https://www.hcup-us.ahrq.gov/reports/statbriefs/sb248.jsp) | 2019 | Level III - retrospective cohort |
| `detect_phi` | [AHRQ HCUP HRRP cohort](https://www.hcup-us.ahrq.gov/reports/statbriefs/sb248.jsp) | 2019 | Level III - retrospective cohort |
| `detect_polypharmacy_concerns` | [AHRQ HCUP HRRP cohort](https://www.hcup-us.ahrq.gov/reports/statbriefs/sb248.jsp) | 2019 | Level III - retrospective cohort |
| `ground_claim` | [AHRQ HCUP HRRP cohort](https://www.hcup-us.ahrq.gov/reports/statbriefs/sb248.jsp) | 2019 | Level III - retrospective cohort |
## critical_care

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_apache_ii_score` | [Knaus et al. 1985 - APACHE II severity of disease](https://doi.org/10.1097/00003246-198510000-00009) | 1985 | Level I - large cohort |
| `compute_meld_score` | [Kamath et al. 2007 - MELD-Na update](https://doi.org/10.1056/NEJMoa0801209) | 2007 | Level I - validation cohort |
| `compute_rifle_aki_classification` | [ADQI Group 2004 - RIFLE consensus](https://doi.org/10.1186/cc2872) | 2004 | Level I - international consensus |
| `compute_sofa_score` | [Singer et al. 2016 - Sepsis-3 + Vincent 1996 SOFA](https://doi.org/10.1001/jama.2016.0287) | 2016 | Level I - international consensus |
## data_normalization

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_charlson_elixhauser_index` | [Quan et al. 2005 + van Walraven 2009](https://doi.org/10.1097/01.mlr.0000182534.19832.83) | 2009 | Level I - cohort validation |
| `compute_normalize_observations` | [Regenstrief LOINC + RxNorm + UMLS Metathesaurus](https://www.nlm.nih.gov/research/umls/index.html) | 2024 | Level I - standards body |
| `compute_rxnorm_ddi_lookup` | [Regenstrief LOINC + RxNorm + UMLS Metathesaurus](https://www.nlm.nih.gov/research/umls/index.html) | 2024 | Level I - standards body |
| `compute_umls_concept_map` | [Regenstrief LOINC + RxNorm + UMLS Metathesaurus](https://www.nlm.nih.gov/research/umls/index.html) | 2024 | Level I - standards body |
## diagnosis

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_differential_diagnosis_ranker` | [USMLE / NEJM diagnostic reasoning curricula](https://www.nejm.org/medical-articles/case-records-of-the-massachusetts-general-hospital) | 2024 | Level II - editorial |
## economics

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_expected_value_of_intervention` | [AHRQ HCUP cost data + ICER value-based pricing](https://www.hcup-us.ahrq.gov/) | 2024 | Level I - operational |
| `compute_fairness_advanced` | [AHRQ HCUP cost data + ICER value-based pricing](https://www.hcup-us.ahrq.gov/) | 2024 | Level I - operational |
## ed_acute

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_admission_triage` | [AHRQ ED triage standards + RCP NEWS2](https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2) | 2017 | Level I - clinical guideline |
| `compute_clinical_deterioration_score` | [RCP NEWS2 (2017)](https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2) | 2017 | Level I - clinical guideline |
## endocrine_acute

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_dka_severity` | [ADA Position Statement 2009 + ISPAD 2018](https://doi.org/10.1111/pedi.12701) | 2018 | Level I - clinical guideline |
| `compute_inpatient_glycemic_control` | [ADA Standards of Care](https://diabetesjournals.org/care/issue/47/Supplement_1) | 2024 | Level I - clinical guideline |
## endocrinology_advanced

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_adrenal_insufficiency_workup` | [Endocrine Society + ATA + AACE composite](https://www.endocrine.org/clinical-practice-guidelines) | 2024 | Level I - clinical guideline |
| `compute_hypocalcemia_severity` | [Endocrine Society + ATA + AACE composite](https://www.endocrine.org/clinical-practice-guidelines) | 2024 | Level I - clinical guideline |
| `compute_thyroid_management` | [ATA 2014 + ETA 2013 hypothyroidism guidelines](https://doi.org/10.1089/thy.2014.0028) | 2014 | Level I - clinical guideline |
## external_knowledge

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_clinical_trials_matcher` | [PubMed + ClinicalTrials.gov + NIH RePORTER](https://pubmed.ncbi.nlm.nih.gov/) | 2024 | Level I - primary literature |
| `compute_drug_pricing` | [PubMed + ClinicalTrials.gov + NIH RePORTER](https://pubmed.ncbi.nlm.nih.gov/) | 2024 | Level I - primary literature |
| `compute_nih_reporter_search` | [PubMed + ClinicalTrials.gov + NIH RePORTER](https://pubmed.ncbi.nlm.nih.gov/) | 2024 | Level I - primary literature |
| `compute_pubmed_search` | [PubMed + ClinicalTrials.gov + NIH RePORTER](https://pubmed.ncbi.nlm.nih.gov/) | 2024 | Level I - primary literature |
## fhir_writeback

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_write_decision_to_fhir` | [HL7 FHIR R4 write-back transactions](https://www.hl7.org/fhir/http.html) | 2022 | Level I - standards body |
## gi_hepatology_depth

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_fib4_liver_fibrosis` | [Sterling et al. 2006 - FIB-4 derivation](https://doi.org/10.1002/hep.21178) | 2006 | Level I - cohort validation |
| `compute_glasgow_blatchford_ugib` | [Blatchford et al. 2000 - GBS validation](https://doi.org/10.1016/S0140-6736(00)02816-6) | 2000 | Level I - prospective |
| `compute_maddrey_alcoholic_hepatitis` | [Maddrey et al. 1978 + AASLD 2019 update](https://doi.org/10.1002/hep.30866) | 2019 | Level I - clinical guideline |
| `compute_rome_iv_ibs` | [Rome IV Foundation 2016](https://theromefoundation.org/rome-iv/) | 2016 | Level I - international consensus |
## heme_onc_depth

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_ecog_performance_status` | [Oken et al. 1982 - ECOG criteria](https://doi.org/10.1097/00000421-198212000-00014) | 1982 | Level I - clinical scoring |
| `compute_ipss_r_mds_score` | [Greenberg et al. 2012 - IPSS-R](https://doi.org/10.1182/blood-2012-03-420489) | 2012 | Level I - international consensus |
| `compute_iss_myeloma_staging` | [Greipp et al. 2005 + IMWG 2015 (R-ISS)](https://doi.org/10.1200/JCO.2005.04.242) | 2015 | Level I - international consensus |
| `compute_karnofsky_performance` | Karnofsky & Burchenal 1949 | 1949 | Level II - clinical scoring |
## imaging

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_aki_kdigo_stage` | [KDIGO Clinical Practice Guideline for AKI](https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf) | 2012 | Level I - clinical guideline |
| `compute_contrast_safety_check` | [ACR Manual on Contrast Media v2024 + Davenport 2020](https://www.acr.org/Clinical-Resources/Contrast-Manual) | 2024 | Level I - clinical guideline |
## infectious_disease

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_hiv_management_tier` | [IDSA / Surviving Sepsis Campaign](https://www.survivingsepsis.org/) | 2024 | Level I - clinical guideline |
| `compute_lactate_clearance` | [Surviving Sepsis Campaign 2021](https://doi.org/10.1097/CCM.0000000000005337) | 2021 | Level I - clinical guideline |
| `compute_qsofa_score` | [Sepsis-3 (JAMA 2016)](https://doi.org/10.1001/jama.2016.0287) | 2016 | Level I - international consensus |
| `compute_tb_risk_screen` | [IDSA / Surviving Sepsis Campaign](https://www.survivingsepsis.org/) | 2024 | Level I - clinical guideline |
## insurance_appeals

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_appeal_escalation_path` | [ERISA + ACA external review + state insurance commissioner](https://www.dol.gov/agencies/ebsa/laws-and-regulations/laws/erisa) | 2024 | Level I - statutory |
| `compute_appeal_letter_draft` | [ERISA + ACA external review + state insurance commissioner](https://www.dol.gov/agencies/ebsa/laws-and-regulations/laws/erisa) | 2024 | Level I - statutory |
| `compute_denial_letter_parse` | [ERISA + ACA external review + state insurance commissioner](https://www.dol.gov/agencies/ebsa/laws-and-regulations/laws/erisa) | 2024 | Level I - statutory |
## legacy_ehr_parsers

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_ccda_document_parse` | [HL7 C-CDA R2.1 Implementation Guide](https://www.hl7.org/implement/standards/product_brief.cfm?product_id=492) | 2018 | Level I - standards body |
| `compute_hl7v2_message_parse` | [HL7 v2.x Standard, Chapter 3 (Patient Administration)](https://www.hl7.org/implement/standards/product_brief.cfm?product_id=185) | 2007 | Level I - standards body |
## mental_health

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_psychiatric_admission_decision` | [C-SSRS + APA practice guidelines](https://www.psychiatry.org/psychiatrists/practice/clinical-practice-guidelines) | 2024 | Level I - clinical guideline |
| `compute_suicide_risk_assessment` | [C-SSRS + APA practice guidelines](https://www.psychiatry.org/psychiatrists/practice/clinical-practice-guidelines) | 2024 | Level I - clinical guideline |
## model_research

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_cox_proportional_hazards` | [Cox 1972 - Regression models and life-tables](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x) | 1972 | Level I - methodological |
| `compute_ensemble_stacking` | [Wolpert 1992 - Stacked generalization](https://doi.org/10.1016/S0893-6080(05)80023-1) | 1992 | Level I - methodological |
| `compute_shap_attribution` | [Lundberg & Lee 2017 - A unified approach to interpreting model predictions](https://arxiv.org/abs/1705.07874) | 2017 | Level I - methodological |
## multimodal

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_dicom_sr_ingest` | [AHA/ACC/HRS QT analysis + DICOM SR Part-3](https://www.dicomstandard.org/current) | 2024 | Level I - standards body |
| `compute_ecg_qt_analyzer` | [AHA/ACC/HRS QT analysis + DICOM SR Part-3](https://www.dicomstandard.org/current) | 2024 | Level I - standards body |
## nephrology

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_dialysis_initiation_decision` | [STARRT-AKI 2020 + KDIGO 2012](https://doi.org/10.1056/NEJMoa2000741) | 2020 | Level I - RCT |
## neurology_depth

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_hauser_ambulation_index` | [Hauser et al. 1983 - Ambulation Index](https://doi.org/10.1212/WNL.33.11.1444) | 1983 | Level II - clinical scoring |
| `compute_hunt_hess_sah` | [Hunt-Hess scale (1968) + Connolly et al. 2012 AHA/ASA](https://doi.org/10.1161/STR.0b013e3182587839) | 2012 | Level I - clinical guideline |
| `compute_ich_score` | [Hemphill et al. 2001 - ICH score](https://doi.org/10.1161/01.STR.32.4.891) | 2001 | Level I - prospective derivation |
| `compute_modified_rankin` | [Quinn et al. 2009 - mRS reliability validation](https://doi.org/10.1161/STROKEAHA.108.541128) | 2009 | Level I - international consensus |
## ob_peds_advanced

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_apgar_score` | [Apgar 1953 + AAP/ACOG 2015 reaffirmation](https://doi.org/10.1542/peds.2015-2651) | 2015 | Level I - clinical guideline |
| `compute_bell_nec_stage` | [Bell et al. 1978 (modified Bell + Walsh 1986)](https://doi.org/10.1542/peds.78.3.460) | 1986 | Level I - large cohort |
| `compute_bilirubin_nomogram` | [AAP 2022 - Hyperbilirubinemia clinical practice guideline](https://doi.org/10.1542/peds.2022-058859) | 2022 | Level I - clinical guideline |
| `compute_bishop_induction_score` | [Bishop 1964 + ACOG 2009](https://www.acog.org/clinical/clinical-guidance) | 2009 | Level I - clinical guideline |
## obstetric_geriatric

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_delirium_screening_cam` | [Inouye et al. 1990 - Confusion Assessment Method](https://doi.org/10.7326/0003-4819-113-12-941) | 1990 | Level I - validation |
| `compute_falls_risk_morse` | [Morse et al. 1989 - Morse Falls Scale](https://doi.org/10.1037/t02948-000) | 1989 | Level I - validation |
| `compute_maternal_early_warning` | [Singh et al. 2012 - MEOWS](https://doi.org/10.1111/j.1365-2044.2011.06916.x) | 2012 | Level I - validation |
| `compute_preeclampsia_assessment` | [ACOG 2020 Practice Bulletin 222](https://doi.org/10.1097/AOG.0000000000003891) | 2020 | Level I - clinical guideline |
## oncology

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_chemo_dose_adjustment` | [NCCN Clinical Practice Guidelines in Oncology](https://www.nccn.org/guidelines/category_1) | 2024 | Level I - clinical guideline |
| `compute_oncology_treatment_response` | [NCCN Clinical Practice Guidelines in Oncology](https://www.nccn.org/guidelines/category_1) | 2024 | Level I - clinical guideline |
| `compute_treatment_selection` | [NCCN Clinical Practice Guidelines in Oncology](https://www.nccn.org/guidelines/category_1) | 2024 | Level I - clinical guideline |
## patient_facing

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_caregiver_summary` | [AHRQ Re-engineered Discharge (RED) toolkit](https://www.ahrq.gov/patient-safety/settings/hospital/red/toolkit/index.html) | 2017 | Level I - clinical guideline |
| `compute_patient_faq` | [AHRQ Re-engineered Discharge (RED) toolkit](https://www.ahrq.gov/patient-safety/settings/hospital/red/toolkit/index.html) | 2017 | Level I - clinical guideline |
| `compute_translate_discharge_counseling` | [AHRQ Re-engineered Discharge (RED) toolkit](https://www.ahrq.gov/patient-safety/settings/hospital/red/toolkit/index.html) | 2017 | Level I - clinical guideline |
## patient_qa

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_caregiver_handoff` | [AHRQ Re-engineered Discharge (RED) + health-literacy](https://www.ahrq.gov/patient-safety/settings/hospital/red/toolkit/index.html) | 2017 | Level I - clinical guideline |
| `compute_discharge_qa` | [AHRQ Re-engineered Discharge (RED) + health-literacy](https://www.ahrq.gov/patient-safety/settings/hospital/red/toolkit/index.html) | 2017 | Level I - clinical guideline |
| `compute_medication_what_if` | [AHRQ Re-engineered Discharge (RED) + health-literacy](https://www.ahrq.gov/patient-safety/settings/hospital/red/toolkit/index.html) | 2017 | Level I - clinical guideline |
## pediatric

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_pediatric_early_warning` | [Brighton/Monaghan PEWS (2005, validated 2017)](https://doi.org/10.1542/peds.2017-2966) | 2017 | Level I - large cohort |
| `compute_weight_based_dosing` | [AAP Bright Futures + Brighton/Monaghan PEWS](https://www.aap.org/en/practice-management/policies) | 2024 | Level I - clinical guideline |
## peri_op_risk

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_ariscat_pulmonary_risk` | [Canet et al. 2010 - ARISCAT](https://doi.org/10.1097/ALN.0b013e3181fc6e0a) | 2010 | Level I - prospective |
| `compute_caprini_vte_risk` | [Caprini 2005 - VTE risk assessment model](https://doi.org/10.1016/j.jvs.2009.03.027) | 2005 | Level I - validation cohort |
| `compute_rcri_cardiac_risk` | [Lee et al. 1999 - Revised Cardiac Risk Index](https://doi.org/10.1161/01.CIR.100.10.1043) | 1999 | Level I - prospective derivation |
## pharmacogenomics

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_pgx_dose_adjustment` | [CPIC Clinical Pharmacogenetics Implementation Consortium](https://cpicpgx.org/guidelines/) | 2024 | Level I - clinical guideline |
| `compute_pgx_drug_alternatives` | [CPIC Clinical Pharmacogenetics Implementation Consortium](https://cpicpgx.org/guidelines/) | 2024 | Level I - clinical guideline |
| `compute_pgx_eligibility_check` | [CPIC Clinical Pharmacogenetics Implementation Consortium](https://cpicpgx.org/guidelines/) | 2024 | Level I - clinical guideline |
## population_health

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_outbreak_heatmap` | [CDC NSSP + ACIP recommendations](https://www.cdc.gov/nssp/) | 2024 | Level I - operational policy |
| `compute_syndromic_surveillance` | [CDC NSSP + ACIP recommendations](https://www.cdc.gov/nssp/) | 2024 | Level I - operational policy |
| `compute_vaccine_reminder_cohort` | [CDC NSSP + ACIP recommendations](https://www.cdc.gov/nssp/) | 2024 | Level I - operational policy |
## preadmit_triage

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_symptom_followup_questions` | [ACEP red-flag + AHRQ self-care triage](https://www.acep.org/patient-care/clinical-policies) | 2024 | Level I - clinical guideline |
| `compute_symptom_red_flag_check` | [ACEP red-flag + AHRQ self-care triage](https://www.acep.org/patient-care/clinical-policies) | 2024 | Level I - clinical guideline |
| `compute_when_to_seek_care` | [ACEP red-flag + AHRQ self-care triage](https://www.acep.org/patient-care/clinical-policies) | 2024 | Level I - clinical guideline |
## prior_authorization

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_pa_appeal_likelihood` | [AMA PA reform + AHIP Fast PASS](https://www.ama-assn.org/practice-management/sustainability/prior-authorization-reform-resources) | 2024 | Level I - operational policy |
| `compute_pa_evidence_pack` | [AMA PA reform + AHIP Fast PASS](https://www.ama-assn.org/practice-management/sustainability/prior-authorization-reform-resources) | 2024 | Level I - operational policy |
| `compute_pa_letter_draft` | [AMA PA reform + AHIP Fast PASS](https://www.ama-assn.org/practice-management/sustainability/prior-authorization-reform-resources) | 2024 | Level I - operational policy |
| `compute_pa_payer_rules_match` | [AMA PA reform + AHIP Fast PASS](https://www.ama-assn.org/practice-management/sustainability/prior-authorization-reform-resources) | 2024 | Level I - operational policy |
## quality_stars

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_care_gap_priority_ranking` | [CMS Star Rating Methodology + NCQA HEDIS](https://www.cms.gov/medicare/health-plans/medicareadvtgspecratestats/medicareadvantagepartcandd) | 2024 | Level I - operational policy |
| `compute_quality_measures_aggregate` | [CMS Star Rating Methodology + NCQA HEDIS](https://www.cms.gov/medicare/health-plans/medicareadvtgspecratestats/medicareadvantagepartcandd) | 2024 | Level I - operational policy |
| `compute_stars_rating_forecast` | [CMS Star Rating Methodology + NCQA HEDIS](https://www.cms.gov/medicare/health-plans/medicareadvtgspecratestats/medicareadvantagepartcandd) | 2024 | Level I - operational policy |
## research_design

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_causal_treatment_effect` | [BMJ N-of-1 trials methodology + ICH-GCP](https://www.bmj.com/content/336/7637/152) | 2008 | Level I - methodological |
| `compute_n_of_1_trial_design` | [BMJ N-of-1 trials methodology + ICH-GCP](https://www.bmj.com/content/336/7637/152) | 2008 | Level I - methodological |
## rheumatology

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_acr_eular_ra_classification` | [ACR/EULAR 2010 RA classification criteria](https://doi.org/10.1002/art.27584) | 2010 | Level I - international consensus |
| `compute_asdas_axspa` | [ASAS - ASDAS validation 2009](https://doi.org/10.1136/ard.2008.094870) | 2009 | Level I - validation |
| `compute_das28_rheumatoid_arthritis` | [Prevoo et al. 1995 - DAS28 derivation](https://doi.org/10.1002/art.1780380107) | 1995 | Level I - validation cohort |
## sleep_pain

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_dn4_neuropathic_pain` | [Bouhassira et al. 2005 - DN4](https://doi.org/10.1016/j.pain.2004.12.010) | 2005 | Level I - validation |
| `compute_epworth_sleepiness_scale` | [Johns 1991 - Epworth scale](https://doi.org/10.1093/sleep/14.6.540) | 1991 | Level I - validation cohort |
| `compute_stop_bang_osa_screen` | [Chung et al. 2008 - STOP-BANG questionnaire](https://doi.org/10.1097/ALN.0b013e31816d83e4) | 2008 | Level I - prospective |
## specialty_clinics

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_diabetic_retinopathy_severity` | [AAD / AAO / ATS composite](https://www.aad.org/member/clinical-quality/guidelines) | 2024 | Level I - clinical guideline |
| `compute_lesion_triage` | [AAD / AAO / ATS composite](https://www.aad.org/member/clinical-quality/guidelines) | 2024 | Level I - clinical guideline |
| `compute_pft_interpretation` | [AAD / AAO / ATS composite](https://www.aad.org/member/clinical-quality/guidelines) | 2024 | Level I - clinical guideline |
## stroke_acs

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_acs_disposition_decision` | [AHA/ACC + AHA/ASA composite](https://professional.heart.org/en/guidelines-and-statements) | 2024 | Level I - clinical guideline |
| `compute_heart_score` | [Six et al. 2008 - HEART score for chest pain](https://doi.org/10.1097/HCO.0b013e328329caa6) | 2008 | Level I - prospective validation |
| `compute_stroke_severity` | [NIH Stroke Scale (NINDS 1989, AHA/ASA 2019)](https://www.stroke.org/-/media/stroke-files/nihss.pdf) | 2019 | Level I - clinical guideline |
| `compute_stroke_thrombolysis_eligibility` | [AHA/ASA 2019 - Guidelines for the Early Management of Acute Ischemic Stroke](https://doi.org/10.1161/STR.0000000000000211) | 2019 | Level I - clinical guideline |
## transplant

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_epts_recipient_score` | [OPTN/UNOS EPTS calculator (2014)](https://optn.transplant.hrsa.gov/data/allocation-calculators/epts-calculator/) | 2014 | Level I - operational |
| `compute_immunosuppression_dose_check` | [OPTN/UNOS allocation policy + KDIGO transplantation](https://optn.transplant.hrsa.gov/policies-bylaws/policies/) | 2024 | Level I - operational policy |
| `compute_kdpi_kidney_donor` | [OPTN/UNOS KDPI calculator (2014)](https://optn.transplant.hrsa.gov/data/allocation-calculators/kdpi-calculator/) | 2014 | Level I - operational |
## trauma_critical

| Tool | Guideline source | Year | Level of evidence |
| --- | --- | ---: | --- |
| `compute_imaging_appropriateness` | [ACR Appropriateness Criteria + Choosing Wisely](https://www.acr.org/Clinical-Resources/ACR-Appropriateness-Criteria) | 2024 | Level I - clinical guideline |
| `compute_massive_transfusion_protocol` | [PROPPR 2015 + CRASH-2 2010](https://doi.org/10.1001/jama.2015.12) | 2015 | Level I - RCT |
| `compute_trauma_severity_score` | Baker et al. 1974 (ISS) + Champion 1989 (RTS) | 1989 | Level I - international convention |
