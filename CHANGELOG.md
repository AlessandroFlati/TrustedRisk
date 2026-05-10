# TrustedRisk Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) +
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-10

First public release.

### Added

- Free-form clinical dispatcher (`apps/orchestrator`): single A2A
  endpoint that routes a clinical prompt + a FHIR-context extension
  to the right workflow / specialist / fan-out via a 2-layer router
  (LLM ranker with 2.5s wall-clock budget, deterministic keyword
  fallback).
- LOINC-mapped FHIR overlay: dispatcher now extracts creatinine
  (baseline + current), pH, HCO3, glucose, AST/ALT, platelets,
  bilirubin, hemoglobin, WBC, BUN, INR, gestational age weeks,
  immunisations, and DocumentReference plaintext from a real bundle.
  Tools that previously crashed with TypeError on missing args now
  receive bundle-derived values when available, else abstain with
  structured reason.
- Smart scribe extraction: `compute_progress_note_draft`,
  `compute_admission_hnp_draft`, `compute_consult_letter_draft`,
  `compute_discharge_summary_draft` now read SOAP sections (HPI /
  PMH / PSH / FH / SH / ROS / PE / labs / imaging / hospital
  course / discharge disposition / patient instructions / follow-up
  plan / consultation reason) verbatim from a chart-attached
  DocumentReference body.
- End-to-end harness `scripts/smoke_demo_full.py`: 150 cases
  covering every workflow + macro + specialist route, classifying
  PASS / ABSTAIN / CRASH / MISSING_ARGS. Current state: 124 PASS,
  26 ABSTAIN (all clinically legitimate), 0 CRASH.
- Direct-tool harness `scripts/smoke_tool_chains.py`: 29 chains,
  91 invocations, 90 distinct MCP tools exercised (62% of the
  145-tool catalog).
- Calibrated abstention as first-class system posture: optional-step
  abstain no longer flips workflow-level `abstain_recommended`;
  instead it is enumerated separately in `abstained_steps` with an
  `optional: true` flag for transparency.

### Changed

- `apps/_shared/specialist_routes.py`: removed 5 population-level
  routes (`_chained_stars_forecast`, `_chained_care_gap_ranking`,
  `compute_syndromic_surveillance`, `compute_vaccine_reminder_cohort`,
  `compute_outbreak_heatmap`). The same capabilities remain available
  via the population-level workflows.
- Specialist routes that hit a TypeError on missing args now produce
  a clean abstain with `missing_clinician_supplied_inputs` reason
  rather than propagating the Python traceback.
- Federation marketplace manifest: agent-card URLs now use POSIX
  separators on every platform.
- Test count: 4224 unit + integration + golden + adversarial passing.

### Fixed

- `extract_encounter` now resolves the resource id from the entry's
  `fullUrl` when `resource.id` is absent (transaction-bundle ingest
  before server commit).

## [0.8.0] - 2026-04-30

Phase 14-17 expansion. **117 net additions** across new MCP tools,
new bundles, new a2a_agent modules, new artefacts, new
documentation surfaces. The federation now backs **145 MCP tools
across 47 thematic bundles** with **15 specialist agents at 100%
bundle coverage**.

### Added - new MCP tools (Phase 14, K + M + O)

8 new tool bundles, 19 net new tools:

- **`rheumatology`** (3): `compute_das28_rheumatoid_arthritis`,
  `compute_asdas_axspa`, `compute_acr_eular_ra_classification`.
- **`peri_op_risk`** (3): `compute_rcri_cardiac_risk`,
  `compute_ariscat_pulmonary_risk`, `compute_caprini_vte_risk`.
- **`infectious_disease`** (4): `compute_qsofa_score`,
  `compute_lactate_clearance`, `compute_hiv_management_tier`,
  `compute_tb_risk_screen`.
- **`gi_hepatology_depth`** (4): `compute_maddrey_alcoholic_hepatitis`,
  `compute_fib4_liver_fibrosis`, `compute_glasgow_blatchford_ugib`,
  `compute_rome_iv_ibs`.
- **`neurology_depth`** (4): `compute_hunt_hess_sah`,
  `compute_ich_score`, `compute_modified_rankin`,
  `compute_hauser_ambulation_index`.
- **`ob_peds_advanced`** (4): `compute_bishop_induction_score`,
  `compute_apgar_score`, `compute_bell_nec_stage`,
  `compute_bilirubin_nomogram`.
- **`model_research`** (3): `compute_cox_proportional_hazards`
  (Newton-Raphson + Breslow tie correction + Harrell c-index),
  `compute_shap_attribution` (exact for linear),
  `compute_ensemble_stacking` (logistic meta-learner).
- **`legacy_ehr_parsers`** (2): `compute_hl7v2_message_parse`
  (MSH/PID/PV1/AL1/OBX/RXA), `compute_ccda_document_parse`
  (allergies/medications/problems).

### Added - new a2a_agent modules (Phase 14-17)

Deterministic-floor implementations of every algorithm; no NumPy
dependency in any of them.

**Phase 14:**
- `decision_card_pdf` - hand-built PDF 1.4 with zlib FlateDecode +
  Helvetica font, no external library.
- `scheduler` - in-process tick-driven job scheduler with
  256-cap history; default jobs `care_gap_sweep` +
  `drift_detection`.
- `tool_retrieval_tfidf` - pure-Python TF-IDF index over the 145-
  tool surface for semantic planner retrieval.
- `planner_revision` - bounded-iteration critic->revise loop with
  4-issue-class taxonomy (missing PHI scrub, duplicate tool,
  orphan bundle, redundant TF-IDF).
- `multi_agent_debate` - 3-agent round table
  (clinical_conservative + evidence_aggressive + fairness_guard)
  with safety-floor veto on any non-approve from the safety
  agents.
- `regulatory_pack` - 8 -> 14 sections (FDA 510(k) predicate,
  GDPR Art. 35 DPIA, HIPAA §164 crosswalk, cybersecurity, human
  oversight, conformity assessment); 100% artefact coverage.

**Phase 15:**
- `prospective_eval` - 10k-encounter prospective evaluation harness
  through calibrated risk -> 4-critic -> 3-agent debate.
- (Generators) `build_subgroup_audit.py` - 100k synthetic cohort
  with 6 subgroup axes + Laplace ε=1.0 DP noise.
- (Generators) `run_prospective_eval.py` - per-subgroup abstain /
  downgrade / calibration-gap report.
- (Generators) `e2e_showcase_v7.py` - 8 cross-bundle scenarios, 53
  tool calls, 16 distinct Phase-13/14 bundles touched.

**Phase 16:**
- `model_card` - Mitchell 2019 model card + Gebru 2021 datasheet.
- `trustworthy_ml` - split-conformal multi-class + Pleiss 2017
  fairness/calibration tension witness + El-Yaniv & Wiener 2010
  selective classification curve.
- `framework_crosswalks` - NIST AI RMF 1.0 (14 rows across
  GOVERN/MAP/MEASURE/MANAGE) + OECD AI Principles (5 rows).
- `bulk_fhir` - NDJSON streaming `$export` consumer.
- `omop_cdm` - v5.4 exporter (PERSON / VISIT_OCCURRENCE /
  CONDITION_OCCURRENCE / MEASUREMENT / DRUG_EXPOSURE / NOTE).
- `smart_on_fhir` - PKCE + state + nonce launch flow.
- `dicom_sr` - real Part-10 explicit-VR LE parser (not stub).
- `cds_hooks_card` - v1.1 Card with suggestion + overrideReasons +
  SMART link.
- `redteam_v4` - 5-attack-class indirect prompt injection corpus
  (14 cases, 14/14 safe).
- `patient_advocate` - 5-axis second-opinion specialist
  (fairness / autonomy / accessibility / financial_burden / language).

**Phase 17:**
- `causal_depth` - Wachter 2018 min-distance counterfactual + Rosenbaum
  sensitivity bounds + Wald IV + Pearl 2009 front-door adjustment.
- `distribution_shift` - two-sample KS + Lipton 2018 BBSE label
  shift + Liu 2020 energy-based OOD.
- `guideline_crosswalk` - 145-tool crosswalk to source / URL / year
  / level of evidence (61 specific overrides + 84 bundle defaults).
- `hrrp_benchmark` - per-LACE-bin + per-subgroup vs AHRQ HCUP HRRP
  literature; max abs gap 0.30%.
- `federation_chain` - deterministic 3-hop A->B->C orchestrator with
  Merkle audit chain.
- `openapi_generator` - 3.1 spec for the 9 canonical federation
  endpoints + 7 schemas + dual security schemes.
- `stigma_linter` - Flesch-Kincaid + 20-rule AHRQ/SAMHSA stigma
  flagger.
- `cost_simulator` - $1.8M saved + 6.25 QALYs gained on 10k
  encounters with HCUP $14.4k/readmission anchor.
- `kalman_vitals` - 1-D + multi-D Kalman + change-point detector.
- `local_explainer` - LIME + Anchors greedy minimum-set search.
- `hospital_year_simulator` - Monte Carlo bootstrap of 1000
  trajectories with 95% CI + IQR + median.
- `federation_registry` - 15-specialist marketplace registry +
  bundle coverage audit.
- `a2a_streaming` - SSE envelope + EventStore + cancellation +
  resume from last-event-id.
- `fgsm_attack` - Goodfellow 2014 Fast Gradient Sign Method via
  finite differences.
- `bayes_net` - discrete Bayes net + Kahn cycle detection +
  variable-elimination inference.
- `federated_learning` - McMahan 2017 FedAvg over 5 biased sites
  + DP-FedAvg with Laplace noise + privacy-utility curve. Final
  global ECE 0.0010 (federated *outperforms* centralized 0.0020).
- `fedprox` - Li 2020 FedProx with proximal μ regulariser.
- `thompson_sampling` - Beta-Binomial bandit + Marsaglia-Tsang
  gamma sampler + Russo-Van Roy 2016 sqrt(N log K) regret bound.
- `fine_gray` - 1999 subdistribution-hazards model with IPCW
  weights + cumulative incidence at horizon.
- `counterfactual_fairness` - Kusner 2017 SCM-based audit.
- `causal_forest` - Athey-Wager 2019 honest random forest.
- `concept_drift` - Bifet-Gavaldà 2007 ADWIN + Gama 2004 DDM.
- `doubly_robust` - IPW + g-formula + AIPW with double-robustness
  property verified.
- `milp_staffing` - branch-and-bound MILP for nurse scheduling.
- `neural_ode` - linear+ReLU vector field + Euler/RK4 integrators.
- `active_learning` - 4 acquisition functions including
  query-by-committee.
- `cqr` - Romano 2019 conformalised quantile regression.

### Added - new artefacts under `docs/`

- `docs/regulatory/REGULATORY_PACK.{md,json}` (14 sections, 100%).
- `docs/regulatory/FRAMEWORK_CROSSWALKS.{md,json}` (NIST + OECD).
- `docs/research/{MODEL_CARD,DATASHEET}.{md,json}`.
- `docs/fairness/{SUBGROUP_AUDIT.md,subgroup_audit.json}`.
- `docs/prospective/{PROSPECTIVE_EVAL.md,prospective_eval.json}`.
- `docs/guidelines/{GUIDELINE_CROSSWALK.md,guideline_crosswalk.json}`.
- `docs/showcase/STORYMODE.html` (6 narrative patient cases).
- `docs/validation/{HRRP_BENCHMARK.md,hrrp_benchmark.json}`.
- `docs/economics/{COST_SIMULATION.md,cost_simulation.json}`.
- `docs/federated/{FEDERATED_LEARNING.md, federated_learning.json,
  privacy_utility_curve.json}`.
- `docs/federation/{FEDERATION_REGISTRY.md, federation_registry.json,
  marketplace_manifest.json}`.
- `docs/api/openapi.json` (OpenAPI 3.1).
- `docs/adversarial/{red_team_v4.json,RED_TEAM_V4.md}`.
- `docs/ui/counterfactual.html` (interactive UI).
- `docs/e2e/v7/index.html` (8 cross-bundle scenarios).
- `docs/MODULE_CATALOG.md` (96 modules).

### Changed

- Federation surface jumped from **118 tools / 39 bundles** (pre-
  Phase 14) to **145 tools / 47 bundles**.
- Master `agent-card.json _bundles` updated to enumerate every
  Phase-14 bundle (`compute_cox_proportional_hazards`,
  `compute_hl7v2_message_parse`, etc.).
- `multi_agent_debate._aggregate` tightened: `clinical_conservative.
  revise` and `fairness_guard.revise` now both veto an `approved`
  (only `evidence_aggressive.revise` can be outvoted).
- Federation specialists at `apps/specialist_*` extended to cover
  every Phase-13/14 bundle through `extend_specialist_bundle_
  coverage.py`. Coverage went from 63.8% -> 100%.

### Deprecated / Removed

- None.

### Fixed

- DDM (`concept_drift.py`): `add()` return value now correctly
  exposes the `drift` transition even after the internal reset
  wipes the state.
- Bayesian network: replaced broken DFS-based cycle detection with
  Kahn's algorithm.
- DICOM SR parser: corrected explicit-VR element-header read size
  (8 -> 6 bytes).

### Notes

- Total Phase-14-17 unit-test surface: ~600 new tests across ~50
  new test modules. Every module has its own dedicated test file
  with round-trip-through-pydantic + reject-bad-input cases at a
  minimum.
- All algorithms are **pure-Python stdlib-only** (no NumPy / SciPy
  / Pandas / sklearn) so the federation runs on any Python 3.13
  install without a heavy scientific stack.

## [0.7.0] - 2026-04-29 (pre-Phase-14 baseline)

- 118 MCP tools / 39 thematic bundles.
- A2A v1 agent-card with FHIR-context extension.
- 4-critic ensemble (clinical_safety + fairness + evidence +
  llm_judge).
- HL7 CDS Hooks v1.1, Merkle audit chain, DP-Laplace equity
  dashboard.
- W1 calibration coefficients (ECE 0.0078, AUROC 0.590, n=7,880).
