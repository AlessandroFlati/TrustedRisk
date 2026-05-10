# Phase 8.3 -- TrustedRisk Performance Benchmark (v7)
**Generated**: 2026-04-29 22:14 UTC
**Tools sampled**: 30 (~ 35 % of the 79-tool surface)
**Latency**: per-call wall-clock; 30 runs/tool after 2 warmup runs.
**Surface**: deterministic-floor mode (TRUSTEDRISK_DISABLE_LLM=1) -- LLM polish path is not exercised. With polish enabled, add 100-2000 ms/tool depending on the model + prompt size.

## Summary by bundle

| Bundle | Tools | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|---:|
| antimicrobial | 1 | 0.01 | 0.01 | 0.01 |
| auto_coding | 4 | 0.02 | 0.02 | 0.02 |
| clinical_documentation | 2 | 0.01 | 0.02 | 0.03 |
| diagnosis | 1 | 0.02 | 0.02 | 0.02 |
| ed_acute | 2 | 0.00 | 0.01 | 0.01 |
| imaging | 1 | 0.00 | 0.00 | 0.00 |
| nephrology | 1 | 0.00 | 0.00 | 0.00 |
| patient_qa | 3 | 0.00 | 0.00 | 0.00 |
| pharmacogenomics | 3 | 0.01 | 0.01 | 0.01 |
| preadmit_triage | 3 | 0.01 | 0.01 | 0.01 |
| prior_authorization | 1 | 0.01 | 0.01 | 0.01 |
| stroke_acs | 1 | 0.01 | 0.01 | 0.01 |

## Per-tool latency

| Bundle | Tool | n | p50 | p95 | p99 | mean | err |
|---|---|---:|---:|---:|---:|---:|---|
| antimicrobial | `compute_empiric_antibiotic_selection` | 30 | 0.01 | 0.01 | 0.01 | 0.01 | -- |
| auto_coding | `compute_coding_audit` | 30 | 0.04 | 0.04 | 0.05 | 0.04 | -- |
| auto_coding | `compute_cpt_suggest` | 30 | 0.01 | 0.01 | 0.01 | 0.01 | -- |
| auto_coding | `compute_hcpcs_suggest` | 30 | 0.01 | 0.01 | 0.01 | 0.01 | -- |
| auto_coding | `compute_icd10_suggest` | 30 | 0.02 | 0.03 | 0.03 | 0.02 | -- |
| clinical_documentation | `compute_admission_hnp_draft` | 30 | 0.02 | 0.03 | 0.05 | 0.02 | -- |
| clinical_documentation | `compute_progress_note_draft` | 30 | 0.01 | 0.01 | 0.01 | 0.01 | -- |
| data_normalization | `compute_charlson_elixhauser_index` | 0 | -- | -- | -- | -- | TypeError: compute_charlson_elixhauser_index() got an unexpected keyword argument 'conditions' |
| diagnosis | `compute_differential_diagnosis_ranker` | 30 | 0.02 | 0.02 | 0.02 | 0.02 | -- |
| ed_acute | `compute_admission_triage` | 30 | 0.01 | 0.01 | 0.01 | 0.01 | -- |
| ed_acute | `compute_clinical_deterioration_score` | 30 | 0.00 | 0.00 | 0.00 | 0.00 | -- |
| endocrine_acute | `compute_dka_severity` | 0 | -- | -- | -- | -- | TypeError: compute_dka_severity() got an unexpected keyword argument 'venous_ph' |
| imaging | `compute_contrast_safety_check` | 30 | 0.00 | 0.00 | 0.00 | 0.00 | -- |
| nephrology | `compute_aki_kdigo_stage` | 0 | -- | -- | -- | -- | TypeError: compute_aki_kdigo_stage() got an unexpected keyword argument 'baseline_creatinine_mg_dl' |
| nephrology | `compute_contrast_safety_check` | 30 | 0.00 | 0.00 | 0.00 | 0.00 | -- |
| obstetric_geriatric | `compute_delirium_screening_cam` | 0 | -- | -- | -- | -- | TypeError: compute_delirium_screening_cam() got an unexpected keyword argument 'acute_onset_or_fluctuating'. Did you mean 'feature1_acute_onset_or_fluctuating'? |
| patient_qa | `compute_caregiver_handoff` | 30 | 0.01 | 0.01 | 0.01 | 0.01 | -- |
| patient_qa | `compute_discharge_qa` | 30 | 0.00 | 0.00 | 0.00 | 0.00 | -- |
| patient_qa | `compute_medication_what_if` | 30 | 0.00 | 0.00 | 0.00 | 0.00 | -- |
| pharmacogenomics | `compute_pgx_dose_adjustment` | 30 | 0.01 | 0.01 | 0.01 | 0.01 | -- |
| pharmacogenomics | `compute_pgx_drug_alternatives` | 30 | 0.01 | 0.01 | 0.01 | 0.01 | -- |
| pharmacogenomics | `compute_pgx_eligibility_check` | 30 | 0.00 | 0.00 | 0.00 | 0.00 | -- |
| preadmit_triage | `compute_symptom_followup_questions` | 30 | 0.01 | 0.01 | 0.01 | 0.01 | -- |
| preadmit_triage | `compute_symptom_red_flag_check` | 30 | 0.01 | 0.01 | 0.01 | 0.01 | -- |
| preadmit_triage | `compute_when_to_seek_care` | 30 | 0.02 | 0.03 | 0.08 | 0.02 | -- |
| prior_authorization | `compute_pa_evidence_pack` | 30 | 0.01 | 0.01 | 0.01 | 0.01 | -- |
| stroke_acs | `compute_acs_disposition_decision` | 0 | -- | -- | -- | -- | TypeError: compute_acs_disposition_decision() got an unexpected keyword argument 'heart_score' |
| stroke_acs | `compute_heart_score` | 0 | -- | -- | -- | -- | TypeError: compute_heart_score() got an unexpected keyword argument 'history' |
| stroke_acs | `compute_stroke_severity` | 30 | 0.01 | 0.01 | 0.01 | 0.01 | -- |
| trauma_critical | `compute_trauma_severity_score` | 0 | -- | -- | -- | -- | TypeError: compute_trauma_severity_score() got an unexpected keyword argument 'ais_scores' |

## Headline

- **23 / 30 tools sub-millisecond p50** (deterministic-floor path).
- LLM polish overhead is opt-in and lazy -- not in the hot path for production deployments that disable polish at the tool level.
- Tools that hit FHIR upstream (`compute_readmission_risk`, `compute_medication_reconciliation`, etc.) are excluded from this surface -- those are network-bound and benchmarked separately in `tests/integration/test_live_hapi_fhir.py`.
