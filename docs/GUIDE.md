# TrustedRisk -- Owner's Comprehensive Guide

> A complete walkthrough of TrustedRisk as it stands at v1.0.0.
> Goes from high-level inspiration to low-level module-by-module
> implementation, with concrete commands you can run on your own
> machine to verify each piece.

> **How to read this guide.** Every section ends with a "**Test it
> yourself**" block listing the exact `pytest` invocation + script
> command that exercises the surface described above it. Run the
> commands in order -- each section is self-contained.

---

## Table of contents

1. [The 30-second elevator pitch](#1-the-30-second-elevator-pitch)
2. [Mental model -- what kind of system is this?](#2-mental-model)
3. [Architecture in five layers](#3-architecture-in-five-layers)
4. [Layer 1: the federation marketplace + 15 specialists](#4-layer-1-federation--15-specialists)
5. [Layer 2: 145 MCP tools across 47 bundles](#5-layer-2-145-mcp-tools--47-bundles)
6. [Layer 3: the 98-module a2a_agent governance layer](#6-layer-3-the-98-module-a2a_agent-layer)
7. [Layer 4: calibrated artefacts + the regulatory pack](#7-layer-4-calibrated-artefacts--regulatory-pack)
8. [Layer 5: testing, reproducibility + the build pipeline](#8-layer-5-testing--reproducibility)
9. [The decision flow end-to-end](#9-the-decision-flow-end-to-end)
10. [Patient-advocate + multi-agent debate in detail](#10-patient-advocate--multi-agent-debate)
11. [Trustworthy ML: every formalism + how to test it](#11-trustworthy-ml-formalisms)
12. [Causal inference + counterfactual reasoning](#12-causal-inference--counterfactuals)
13. [Federated learning (FedAvg + FedProx + DP-FedAvg)](#13-federated-learning)
14. [Drift detection + adversarial robustness](#14-drift-detection--adversarial-robustness)
15. [Interoperability surfaces](#15-interoperability)
16. [Compliance + governance crosswalks](#16-compliance--governance)
17. [Demo assets you can show a judge](#17-demo-assets)
18. [Submission readiness checklist](#18-submission-readiness)
19. [Troubleshooting](#19-troubleshooting)

---

## 1. The 30-second elevator pitch

TrustedRisk is a **federation of 15 specialist A2A agents** sharing a
**145-tool MCP backend** that turns a FHIR Bundle (or HL7 v2 / C-CDA
chart) into an **auditable DecisionCard** for clinical disposition. It
is the answer to "what does a healthcare AI system look like when you
take calibration, fairness, abstention, and patient autonomy
seriously?"

**Test it yourself**:
```bash
PYTHONPATH=src python scripts/build_all_artefacts.py
```
This runs the entire artefact pipeline (17 steps, ~7 s) and shows you
that every doc artefact regenerates cleanly.

---

## 2. Mental model

Think of TrustedRisk as five concentric rings, from clinician-facing
to algorithmic floor:

```
    Clinician  ->  DecisionCard  ->  CDS Hooks  ->  EHR
                       ↑
            Patient-advocate veto
                       ↑
            Multi-agent debate (3 voices)
                       ↑
            4-critic ensemble (clinical_safety + fairness +
                                evidence + llm_judge)
                       ↑
                   Tool chain
                  (MCP federation)
                       ↑
              Calibrated risk + conformal CI
                       ↑
                Calibrated artefacts
                  (W1 + Synthea-100k + MIMIC)
```

Three principles bind the rings together:

- **No autonomy without abstention.** The system has explicit
  abstain triggers when uncertainty or fairness signals are
  insufficient.
- **No prediction without provenance.** Every DecisionCard chains
  cryptographically into a Merkle audit trail (`a2a_agent.merkle_audit`)
  so any downstream tampering is detectable.
- **No fairness without veto.** The fairness-guard agent in the
  multi-agent debate has unilateral veto power on flagged subgroups.

---

## 3. Architecture in five layers

| Layer | Surface | Count | Where to look |
|---|---|---:|---|
| 1 | Federation marketplace + specialists | 15 | `apps/specialist_*` + `docs/federation/` |
| 2 | MCP tools + bundles | 145 / 47 | `src/mcp_server/tools/` |
| 3 | Algorithm + governance modules | 96 | `src/a2a_agent/` |
| 4 | Calibrated artefacts + regulatory docs | 17 | `data/` + `docs/` |
| 5 | Tests + reproducibility | ~1,900 | `tests/` |

**Test it yourself**:
```bash
# See the full layer 3 catalog
PYTHONPATH=src python scripts/generate_module_catalog.py
cat docs/MODULE_CATALOG.md | head -50
```

---

## 4. Layer 1: federation + 15 specialists

The federation has 15 specialist A2A agents on ports 8770-8784:

| Specialist | Port | Owns bundles |
|---|---:|---|
| `trustedrisk-discharge` | 8770 | core_discharge, antimicrobial, oncology, clinical_workflow, data_normalization, patient_facing, fhir_writeback, heme_onc_depth |
| `trustedrisk-acute` | 8771 | ed_acute, stroke_acs, obstetric_geriatric, trauma_critical, endocrine_acute, imaging, nephrology, critical_care, cardiology_depth, endocrinology_advanced, gi_hepatology_depth, infectious_disease, neurology_depth, ob_peds_advanced, peri_op_risk, rheumatology, transplant |
| `trustedrisk-pedi-mh` | 8772 | pediatric, mental_health |
| `trustedrisk-evidence` | 8773 | context_resolution, diagnosis, external_knowledge, chart_intelligence, research_design, model_research, legacy_ehr_parsers, sleep_pain, specialty_clinics |
| `trustedrisk-population` | 8774 | economics |
| `trustedrisk-pa` | 8775 | prior_authorization |
| `trustedrisk-scribe` | 8776 | clinical_documentation |
| `trustedrisk-patient` | 8777 | patient_facing, patient_qa |
| `trustedrisk-coder` | 8778 | auto_coding |
| `trustedrisk-pgx` | 8779 | pharmacogenomics |
| `trustedrisk-preadmit` | 8780 | preadmit_triage |
| `trustedrisk-quality` | 8781 | quality_stars |
| `trustedrisk-pophealth` | 8782 | population_health |
| `trustedrisk-appeals` | 8783 | insurance_appeals |
| `trustedrisk-multimodal` | 8784 | multimodal |

The federation registry (`a2a_agent.federation_registry`) aggregates
every `apps/specialist_*/agent_card.json` into a single marketplace
manifest + verifies 100% bundle coverage.

**Test it yourself**:
```bash
# Regenerate + verify the registry
PYTHONPATH=src python scripts/generate_federation_registry.py
# Should print: "15 specialists, 47 bundles covered, 0 missing (100.0% coverage)."

# Inspect the marketplace manifest
cat docs/federation/marketplace_manifest.json | python -m json.tool | head -30

# Run the federation registry tests
PYTHONPATH=src pytest tests/unit/test_federation_registry.py -v
```

The federation chain orchestrator (`a2a_agent.federation_chain`)
demonstrates a real specialist-to-specialist call sequence:
`discharge -> pa-agent -> patient-advocate`, each contributing an audit
record into a Merkle chain.

```bash
# Run the federation chain demo
PYTHONPATH=src python -c "
from a2a_agent.federation_chain import run_federation_chain
trace = run_federation_chain(initial_payload={
    'patient_id': 'demo-1', 'encounter_id': 'enc-1', 'lace': 9,
    'demographics': {'age': 70, 'race': 'white', 'insurance': 'medicare',
                     'language': 'english', 'n_chronic_meds': 5,
                     'has_home_caregiver_available': True,
                     'has_transportation': True,
                     'fairness_audit_present': True},
})
print('verdict:', trace.final_verdict, 'audit_root:', trace.audit_root[:16])
"
```

---

## 5. Layer 2: 145 MCP tools + 47 bundles

The `mcp_server.tools` package exposes 145 async `compute_*` functions
grouped into 47 thematic bundles. Each tool has an inline Pydantic v2
input schema, a deterministic floor implementation, and (where
applicable) an LLM-assisted re-rank that runs only when the deterministic
floor passes.

The 47 bundles span:
- 12 clinical-vertical (core_discharge, ed_acute, pediatric,
  mental_health, antimicrobial, oncology, stroke_acs,
  obstetric_geriatric, trauma_critical, endocrine_acute, imaging,
  nephrology)
- 8 cross-cutting (economics, context_resolution, diagnosis,
  patient_facing, data_normalization, clinical_workflow,
  external_knowledge, chart_intelligence)
- 3 Phase-2 pain-point (prior_authorization, clinical_documentation,
  patient_qa)
- 3 Phase-7 expansion (auto_coding, pharmacogenomics, preadmit_triage)
- 3 Phase-10 outpatient (quality_stars, population_health,
  insurance_appeals)
- 1 Phase-11 multi-modal (multimodal)
- 6 Phase-13 depth (critical_care, specialty_clinics, cardiology_depth,
  heme_onc_depth, endocrinology_advanced, sleep_pain, transplant)
- 6 Phase-14 K-bundles (rheumatology, peri_op_risk, infectious_disease,
  gi_hepatology_depth, neurology_depth, ob_peds_advanced)
- 2 Phase-14 platform (model_research, legacy_ehr_parsers,
  fhir_writeback, research_design)

**Test it yourself**:
```bash
# Confirm the runtime surface
PYTHONPATH=src python -c "
from mcp_server.tools import BUNDLES
unique = set()
for b in BUNDLES.values(): unique.update(b)
print(f'{len(unique)} unique tools across {len(BUNDLES)} bundles')
"
# Expected: 145 unique tools across 47 bundles

# Run the federation healthz endpoint as a smoke
PYTHONPATH=src pytest tests/integration/test_full_federation_smoke.py -q
```

---

## 6. Layer 3: the 98-module a2a_agent layer

The `src/a2a_agent/` package is the heart of TrustedRisk. It contains
98 modules organised by concern:

### 6.1 Calibration + risk
- `coefficients` -- read W1 spec_002 5-bin Beta-Binomial.
- `incremental_risk` -- partial-update risk over streaming inputs.
- `deterioration_nowcast` -- short-horizon deterioration probability.
- `trajectory_predictor` -- Bayesian state-space patient trajectory.

### 6.2 Trustworthy ML (Phase 16.F)
- `trustworthy_ml` -- split-conformal multi-class (Romano 2020 LAC),
  Pleiss 2017 fairness/calibration tension witness, El-Yaniv & Wiener
  2010 selective classification curve.
- `model_card` -- Mitchell 2019 model card + Gebru 2021 datasheet
  generators.
- `cqr` -- Romano 2019 conformalised quantile regression with
  pinball-loss SGD.

### 6.3 Causal inference (Phase 17.M, 17.AR, 17.AS, 17.AU)
- `causal_inference` -- DoWhy ATE with backdoor adjustment (legacy).
- `causal_depth` -- Wachter 2018 min-distance counterfactual,
  Rosenbaum sensitivity bounds, Wald IV, Pearl 2009 front-door.
- `causal_forest` -- Athey-Wager 2019 honest random forest for CATE.
- `counterfactual_fairness` -- Kusner 2017 SCM-based audit.
- `doubly_robust` -- IPW + g-formula + AIPW with Robins 1994 property.
- `fine_gray` -- 1999 subdistribution-hazards model with competing
  risks.

### 6.4 Drift + OOD (Phase 16.N, 17.AT, 14.X)
- `distribution_shift` -- KS test + Lipton 2018 BBSE label shift +
  Liu 2020 energy-based OOD.
- `concept_drift` -- Bifet-Gavaldà 2007 ADWIN + Gama 2004 DDM.
- `drift_monitor` -- sliding-window mean-prob + LACE histogram KS.

### 6.5 Explainability (Phase 17.AG)
- `local_explainer` -- LIME locality-weighted least squares + Anchors
  greedy minimum-set search.

### 6.6 Federated learning (Phase 17.AN, 17.AO)
- `federated_learning` -- McMahan 2017 FedAvg + DP-FedAvg + privacy-
  utility curve.
- `fedprox` -- Li 2020 FedProx with proximal regulariser.

### 6.7 Online + active learning (Phase 17.AP, 17.AX)
- `thompson_sampling` -- Beta-Binomial bandit with Marsaglia-Tsang
  gamma sampler.
- `active_learning` -- least_confidence / margin / entropy /
  query-by-committee acquisitions.

### 6.8 Multi-agent (Phase 14.16, 16.J)
- `multi_agent_debate` -- clinical_conservative + evidence_aggressive
  + fairness_guard with safety-floor veto.
- `patient_advocate` -- 5-axis second-opinion specialist.
- `planner_revision` -- bounded critic->revise loop.
- `federation_chain` -- A->B->C orchestrator with Merkle audit.

### 6.9 Operational (Phase 14.12, 17.AV, 17.AW, 17.AL)
- `scheduler` -- in-process tick-driven job scheduler.
- `milp_staffing` -- branch-and-bound MILP for nurse-staffing.
- `neural_ode` -- linear+ReLU vector field with Euler/RK4 integrators.
- `bayes_net` -- discrete Bayes net + Kahn cycle detection +
  variable-elimination inference.

### 6.10 Audit + compliance (Phase 14.17, 16.G)
- `regulatory_pack` -- 14-section EU AI Act + FDA 510(k) + GDPR DPIA
  + HIPAA + ISO 13485 + cybersecurity + human oversight + conformity
  assessment.
- `framework_crosswalks` -- NIST AI RMF + OECD AI Principles.
- `merkle_audit` -- RFC 6962 Merkle audit chain.
- `audit` -- HIPAA-style PHI-redacted JSONL log.
- `data_lineage` -- coefficient -> training-data hash chain.
- `right_to_explanation` -- GDPR Art. 22 surface.
- `patient_audit_summary` -- patient-friendly audit story.

### 6.11 Interoperability (Phase 16.H, 14.13)
- `bulk_fhir` -- NDJSON $export streaming consumer.
- `omop_cdm` -- v5.4 exporter.
- `smart_on_fhir` -- PKCE + state + nonce launch flow.
- `dicom_sr` -- Part-10 explicit-VR LE binary parser.
- `cds_hooks_card` -- v1.1 cards with overrideReasons + SMART links.
- `openapi_generator` -- OpenAPI 3.1 spec.
- `legacy_ehr_parsers` -- HL7 v2 + C-CDA (in `mcp_server.tools`).

### 6.12 Patient-facing + lint (Phase 17.Z)
- `stigma_linter` -- Flesch-Kincaid + 20-rule AHRQ/SAMHSA stigma flagger.
- `notification_formatter` -- multichannel alert formatting.

### 6.13 Adversarial (Phase 16.I, 17.AK)
- `redteam_v3` -- multi-target red-team (legacy).
- `redteam_v4` -- 5-class indirect prompt injection corpus.
- `safety_redteam` -- adversarial fragility + OOD + pen-test +
  chaos engineering.
- `fgsm_attack` -- Fast Gradient Sign Method via finite differences.

### 6.14 Bench + simulation (Phase 17.S, 17.AA, 17.AH, 15.C)
- `hrrp_benchmark` -- per-LACE-bin + per-subgroup vs literature.
- `cost_simulator` -- TrustedRisk vs LACE-only baseline cost analysis.
- `hospital_year_simulator` -- Monte Carlo 1000-trajectory bootstrap.
- `prospective_eval` -- 10k-encounter "if we had deployed" simulation.
- `outcomes_simulator` -- hospital-year Monte Carlo.

### 6.15 Streaming (Phase 17.AJ)
- `a2a_streaming` -- SSE envelope + EventStore + cancellation +
  resume.

**Test it yourself**:
```bash
# Auto-discover the catalog
PYTHONPATH=src python scripts/generate_module_catalog.py
less docs/MODULE_CATALOG.md
```

---

## 7. Layer 4: calibrated artefacts + regulatory pack

Every artefact under `docs/` is auto-regenerable. The 17 generators
and their outputs:

| Generator | Output(s) |
|---|---|
| `build_subgroup_audit.py` | `docs/fairness/subgroup_audit.json` + `SUBGROUP_AUDIT.md` |
| `run_prospective_eval.py` | `docs/prospective/prospective_eval.json` + `PROSPECTIVE_EVAL.md` |
| `generate_hrrp_benchmark.py` | `docs/validation/HRRP_BENCHMARK.md` |
| `generate_model_card.py` | `docs/research/MODEL_CARD.md` + `DATASHEET.md` |
| `generate_framework_crosswalks.py` | `docs/regulatory/FRAMEWORK_CROSSWALKS.md` |
| `generate_regulatory_pack.py` | `docs/regulatory/REGULATORY_PACK.md` |
| `generate_guideline_crosswalk.py` | `docs/guidelines/GUIDELINE_CROSSWALK.md` |
| `extend_specialist_bundle_coverage.py` | `apps/specialist_*/agent_card.json` |
| `generate_federation_registry.py` | `docs/federation/FEDERATION_REGISTRY.md` |
| `e2e_showcase_v7.py` | `docs/e2e/v7/index.html` |
| `generate_storymode.py` | `docs/showcase/STORYMODE.html` |
| `generate_cost_simulation.py` | `docs/economics/COST_SIMULATION.md` |
| `generate_federated_learning_report.py` | `docs/federated/FEDERATED_LEARNING.md` |
| `run_redteam_v4.py` | `docs/adversarial/red_team_v4.json` |
| `generate_counterfactual_ui.py` | `docs/ui/counterfactual.html` |
| `generate_openapi.py` | `docs/api/openapi.json` |
| `generate_module_catalog.py` | `docs/MODULE_CATALOG.md` |

**Test it yourself**:
```bash
# Build everything
PYTHONPATH=src python scripts/build_all_artefacts.py
# Expected: "17 OK / 0 FAIL" in ~7 seconds.

# Run the smoke test
PYTHONPATH=src pytest tests/integration/test_build_all_artefacts.py -v
# Expected: 26/26 passing.
```

---

## 8. Layer 5: testing + reproducibility

Test directory layout:

```
tests/
├── unit/                        # ~110 modules, ~1,500 tests
├── integration/                 # ~25 modules, ~300 tests
├── property/                    # Hypothesis property-based, ~16 tests
├── golden/                      # snapshot tests on cohorts
├── adversarial/                 # red-team v2/v3/v4 corpora
├── functional/                  # ~181 pipeline-style tests
├── eval/                        # evaluation framework tests
├── llm_integration/             # tests with real LLM calls (skipped by default)
└── regression/                  # historical-bug regression
```

The two integration tests that prove cross-module composability:

- `test_phase17_cross_module_pipeline.py` -- 11 tests touching ~10
  Phase-14-17 modules in their natural composition order (HL7v2 ->
  SHAP/Cox -> conformal -> debate -> advocate -> CDS Hooks -> OMOP ->
  federation chain -> regulatory pack -> stigma lint -> ADWIN).
- `test_build_all_artefacts.py` -- 26 tests verifying the build
  pipeline produces every expected artefact above its min-size
  threshold + 100% regulatory + federation coverage.

**Test it yourself**:
```bash
# Run everything (~30 s)
PYTHONPATH=src pytest tests/ -q

# Run just the cross-module pipeline test
PYTHONPATH=src pytest tests/integration/test_phase17_cross_module_pipeline.py -v

# Run just the build pipeline smoke test
PYTHONPATH=src pytest tests/integration/test_build_all_artefacts.py -v
```

---

## 9. The decision flow end-to-end

For a single encounter, the canonical decision flow is:

1. **Ingestion**. The FHIR Bundle arrives via A2A `Message.metadata`
   (the `fhir-context` extension, camelCase fields), or
   via legacy HL7 v2 / C-CDA through `compute_hl7v2_message_parse` /
   `compute_ccda_document_parse`. The before-model hook strips the
   bearer token before any LLM prompt is built.

2. **Planning**. `a2a_agent.planner` produces a deterministic
   `ToolUsePlan` (intent rules) optionally augmented by TF-IDF
   retrieval (`tool_retrieval_tfidf`) when no intent rule matches.

3. **Plan revision**. `planner_revision` runs a critic over the
   plan + emits issues (missing PHI scrub, duplicate tool, orphan
   bundle, redundant TF-IDF). A bounded N-iteration loop revises
   the plan before any tool runs.

4. **Tool execution**. The MCP backend runs the tool chain. Each
   tool has a deterministic floor; LLM-using tools optionally
   re-rank when configured.

5. **Risk + explanation**. `compute_readmission_risk` produces a
   calibrated point estimate + 95% Beta-Binomial CI. SHAP
   attribution + Wachter min-distance counterfactual produce the
   explanation surface.

6. **Conformal multi-class**. `trustworthy_ml.split_conformal_
   multiclass` produces a prediction set with marginal coverage
   guarantee.

7. **4-critic ensemble**. `clinical_safety` + `fairness` +
   `evidence` + `llm_judge` critics vote; `apply_critique` applies
   the verdict (approve / downgrade_confidence / force_abstain /
   request_replay).

8. **3-agent debate**. `multi_agent_debate.run_debate` runs the
   three voices; safety-agent revise has veto power.

9. **Patient-advocate**. `patient_advocate.evaluate_patient_advocate`
   runs the 5-axis review (fairness, autonomy, accessibility,
   financial_burden, language) and produces a concur / challenge
   / escalate verdict.

10. **DecisionCard composition**. The final card includes the
    recommendation, the conformal prediction set, the SHAP
    attribution, the counterfactual, the 4-critic ensemble verdict,
    the debate verdict, the patient-advocate verdict, the audit
    hash, the timestamp, and the right-to-explanation summary.

11. **Output surfaces**. Multiple consumers can be served from the
    same DecisionCard: a CDS Hooks v1.1 card, an OMOP CDM export,
    a Joint-Commission discharge summary, a multi-language patient-
    facing counseling sheet, a SMART app launch URL.

12. **Audit + storage**. Every step logs to the append-only Merkle
    chain (`a2a_agent.merkle_audit`) for tamper-evidence and to the
    reproducibility archive (SQLite) for byte-identical replay.

**Test it yourself**:
```bash
# Run the cross-module integration test that touches every step
PYTHONPATH=src pytest tests/integration/test_phase17_cross_module_pipeline.py -v
```

---

## 10. Patient-advocate + multi-agent debate

### Multi-agent debate (`a2a_agent.multi_agent_debate`)

Three voices, all deterministic:

- **clinical_conservative** -- risk + CI width drives verdict.
  - Discharge home with predicted risk ≥ 20% -> abstain.
  - Discharge home with CI width > 10% -> revise.
  - High risk regardless of action -> revise.

- **evidence_aggressive** -- CI width drives verdict.
  - CI ≤ 5% (half the moderate bar) -> approve.
  - CI ≥ 20% (double the moderate bar) -> revise.

- **fairness_guard** -- flagged subgroup + audit presence drives verdict.
  - Flagged subgroup (Black, Indigenous, LGBTQ+, low-SES, peripartum)
    + no fairness audit -> abstain.
  - Flagged subgroup + audit present -> revise.
  - Unflagged -> approve.

Aggregation:
- `clinical_conservative.abstain` OR `fairness_guard.abstain` ->
  `force_abstain` (hard veto).
- `clinical_conservative.revise` OR `fairness_guard.revise` ->
  `revise` (soft veto -- only `evidence_aggressive.revise` can be
  outvoted).
- All-approve -> `approved`.

### Patient advocate (`a2a_agent.patient_advocate`)

5 axes, all deterministic:

- **fairness** -- flagged race + insurance + (audit present -> concern,
  audit absent -> blocker).
- **autonomy** -- low-risk + continued_admission proposed -> concern
  (shared decision-making is being skipped).
- **accessibility** -- discharge_home + (no caregiver OR no
  transportation) -> blocker.
- **financial_burden** -- uninsured + ≥ 4 chronic medications ->
  concern (340B / patient-assistance referral needed).
- **language** -- preferred language outside the 5 supported
  (en/es/zh/vi/ar) -> concern (interpreter referral needed).

Aggregation: any blocker -> `escalate`; any concern -> `challenge`;
otherwise `concur`.

**Test it yourself**:
```bash
# Run both component tests
PYTHONPATH=src pytest tests/unit/test_multi_agent_debate.py \
  tests/unit/test_patient_advocate.py -v

# Try the integration on the storymode patients
python scripts/generate_storymode.py
open docs/showcase/STORYMODE.html
# Each of the 6 narrative patients runs through both debate +
# advocate; the verdicts surface as colour-coded badges.
```

---

## 11. Trustworthy ML formalisms

The `trustworthy_ml` module provides three flagship formalisms:

### Split-conformal multi-class (Romano 2020 LAC)

Given calibration probabilities + true labels + test probabilities,
produce a per-instance prediction set with marginal coverage
:math:`P(y_test \in C(x)) \geq 1 - \alpha`:

```python
from a2a_agent.trustworthy_ml import split_conformal_multiclass
res = split_conformal_multiclass(
    cal_probabilities=[{"a": 0.7, "b": 0.2, "c": 0.1}, ...],
    cal_true_labels=["a", "b", ...],
    test_probabilities=[{"a": 0.5, "b": 0.3, "c": 0.2}],
    target_coverage=0.90,
)
print(res.prediction_sets, res.empirical_coverage_test)
```

### Pleiss 2017 fairness/calibration tension witness

Demonstrates the impossibility theorem on the actual cohort: with
non-trivial base-rate variation across subgroups, you cannot have
calibration AND equal generalised FPR/FNR simultaneously.

```python
from a2a_agent.trustworthy_ml import per_subgroup_calibration_tension
res = per_subgroup_calibration_tension(
    predictions=[0.1, 0.8, ...],
    labels=[0, 1, ...],
    subgroup_labels=["a", "b", ...],
)
print(res.impossibility_witnessed)   # True if base-rate spread > 5%
                                     # AND FPR or FNR spread > 2%
```

### El-Yaniv & Wiener 2010 selective classification

Sweeps the abstain threshold τ over confidence quantiles and reports
the coverage(τ) -> risk(τ) curve. Optimal point minimises
risk × (1 - coverage).

```python
from a2a_agent.trustworthy_ml import selective_classification_curve
res = selective_classification_curve(
    predictions=[0.1, 0.4, 0.6, 0.9],
    confidences=[0.5, 0.6, 0.8, 0.95],
    labels=[0, 0, 1, 1],
    n_thresholds=11,
)
print(res.optimal_threshold, res.optimal_coverage, res.optimal_risk)
```

### Conformalised quantile regression (Romano 2019)

Train quantile regressors at α/2 and 1-α/2 levels, then conformalise
via the (1-α)(1+1/n_cal) quantile of the calibration conformity
scores:

```python
from a2a_agent.cqr import fit_cqr
res = fit_cqr(
    X_train=X_tr, y_train=y_tr,
    X_calibration=X_cal, y_calibration=y_cal,
    X_test=X_te, y_test=y_te,
    target_coverage=0.90,
)
print(res.empirical_coverage_test, res.average_interval_width)
```

**Test it yourself**:
```bash
PYTHONPATH=src pytest tests/unit/test_trustworthy_ml.py \
  tests/unit/test_cqr.py -v
```

---

## 12. Causal inference + counterfactuals

The `causal_depth` + `causal_forest` + `counterfactual_fairness` +
`doubly_robust` modules cover the canonical causal-inference toolkit:

- **Wachter 2018 min-distance counterfactual** --
  `causal_depth.find_min_distance_counterfactual` projects gradient
  descent on a sigmoid model toward the minimum-L1 perturbation that
  flips the prediction class. Used by `docs/ui/counterfactual.html`.

- **Rosenbaum sensitivity bounds** --
  `causal_depth.rosenbaum_gamma_bound` returns the smallest unmeasured-
  confounder strength Γ that would explain away a sign-test result.

- **Wald instrumental variables** -- `causal_depth.wald_iv_estimate`
  computes the LATE estimator cov(Y,Z) / cov(D,Z).

- **Pearl 2009 front-door adjustment** --
  `causal_depth.front_door_adjustment` computes
  P(Y=y | do(X=x)) = Σ_m P(M=m|X=x) × Σ_x' P(X=x') × P(Y=y|X=x',M=m).

- **Fine-Gray 1999 subdistribution hazards** --
  `fine_gray.fit_fine_gray` fits a Cox-like model with IPCW weights
  + cumulative-incidence at horizon for outcomes with competing risks
  (death-before-readmission).

- **Doubly-robust ATE** -- `doubly_robust.estimate_ate_doubly_robust`
  produces IPW + g-formula + AIPW estimates; AIPW is consistent if
  EITHER propensity OR outcome model is correct.

- **Causal forest (Athey-Wager 2019)** -- `causal_forest.fit_causal_
  forest` builds an honest random forest with splitting/estimation
  half + heterogeneity-maximising splits + leaf CATE.

- **Counterfactual fairness (Kusner 2017)** --
  `counterfactual_fairness.audit_counterfactual_fairness` runs the
  Pearl abduction-action-prediction algorithm: for each instance,
  flip the protected attribute via a structural causal model + score
  the counterfactual + compare.

**Test it yourself**:
```bash
PYTHONPATH=src pytest tests/unit/test_causal_depth.py \
  tests/unit/test_causal_forest.py \
  tests/unit/test_counterfactual_fairness.py \
  tests/unit/test_doubly_robust.py \
  tests/unit/test_fine_gray.py -v

# Open the interactive counterfactual UI
python scripts/generate_counterfactual_ui.py
# Open docs/ui/counterfactual.html in any browser; move the LACE
# sliders + watch the recommendation flip in real time + see the
# closest counterfactual that would change the disposition.
```

---

## 13. Federated learning

Federated learning across 5 biased hospital sites is implemented in
`federated_learning` + `fedprox`:

- 5 sites: `urban_academic`, `rural_community`, `safety_net`,
  `geriatric_specialty`, `pediatric_adjacent`. Each carries a
  per-site LACE skew + outcome multiplier.
- Per-round flow: each site fits its own Beta-Binomial conjugate
  posterior on its local cohort -> optionally adds Laplace noise (DP) ->
  ships only the parameters -> server aggregates with sample-size
  weighting.
- Counterfactual baselines: centralised (fit on the union) +
  local-only (each site alone, no federation).
- Privacy-utility curve: ε ∈ {0.1, 0.5, 1.0, 5.0, 10.0}.
- FedProx adds a proximal regulariser μ/2‖w-w_global‖² to handle
  non-IID clients; `mu=0` recovers FedAvg.

**Result on the synthetic cohort**: final FedAvg global ECE 0.0010
vs centralised baseline 0.0020 (federated *outperforms* due to
implicit Bayesian shrinkage).

**Test it yourself**:
```bash
PYTHONPATH=src pytest tests/unit/test_federated_learning.py \
  tests/unit/test_fedprox.py -v

# Run the full report
python scripts/generate_federated_learning_report.py
cat docs/federated/FEDERATED_LEARNING.md
```

---

## 14. Drift detection + adversarial robustness

### Drift detection (`concept_drift` + `distribution_shift`)

- **ADWIN** (Bifet-Gavaldà 2007) -- adaptive sliding window with
  Hoeffding-bound cut.
- **DDM** (Gama 2004) -- error-rate state machine with in-control ->
  warning -> drift transitions.
- **KS test** (`distribution_shift.two_sample_ks_test`) -- covariate
  shift detection with asymptotic p-value.
- **BBSE** (Lipton 2018) -- label-shift correction via inverting the
  source confusion matrix.
- **Energy-based OOD** (Liu 2020) -- E(x) = -logsumexp(z(x)) per
  instance.

### Adversarial robustness (`fgsm_attack` + `redteam_v4`)

- **FGSM** (Goodfellow 2014) -- Fast Gradient Sign Method via
  finite-difference numerical gradients on a sigmoid model. Walks ε
  to find the minimum L∞ perturbation that flips the prediction.
- **Red-team v4** -- 5-attack-class indirect prompt injection corpus
  (14 cases): hostile content embedded in `Observation.note`,
  `MedicationRequest.dosageInstruction.text`, `Patient.alias`,
  patient narrative jailbreak, dose-tampering unit confusion, HIPAA
  leak via PHI rephrasing. **All 14 cases land in `safe` posture**.

**Test it yourself**:
```bash
PYTHONPATH=src pytest tests/unit/test_concept_drift.py \
  tests/unit/test_distribution_shift.py \
  tests/unit/test_fgsm_attack.py \
  tests/unit/test_redteam_v4.py -v

# Run the full red-team v4 report
python scripts/run_redteam_v4.py
cat docs/adversarial/red_team_v4.json | python -m json.tool | head -30
```

---

## 15. Interoperability

TrustedRisk speaks every standard a real EHR integration requires:

- **HL7 FHIR R4** -- native consumer; SMART-on-FHIR launch flow with
  PKCE + state + nonce verified end-to-end (`smart_on_fhir.py`).
- **HL7 Bulk Data Access v2.0** -- NDJSON `$export` streaming consumer
  (`bulk_fhir.py`).
- **HL7 v2.x** -- pipe-delimited message parser
  (`legacy_ehr_parsers.compute_hl7v2_message_parse`).
- **HL7 C-CDA R2.1** -- XML chart parser
  (`legacy_ehr_parsers.compute_ccda_document_parse`).
- **HL7 CDS Hooks v1.1** -- Card builder with suggestion +
  overrideReasons + SMART links (`cds_hooks_card.py`).
- **DICOM Part-10** -- explicit-VR LE binary parser for SR objects
  (`dicom_sr.py`).
- **OMOP CDM v5.4** -- exporter producing PERSON / VISIT_OCCURRENCE /
  CONDITION_OCCURRENCE / MEASUREMENT / DRUG_EXPOSURE / NOTE rows
  (`omop_cdm.py`).
- **OpenAPI 3.1** -- auto-generated spec for the federation
  endpoints (`openapi_generator.py`).

**Test it yourself**:
```bash
PYTHONPATH=src pytest tests/unit/test_h3_interop.py \
  tests/unit/test_bulk_fhir.py \
  tests/unit/test_omop_cdm.py \
  tests/unit/test_openapi_generator.py -v

# Generate the OpenAPI spec + open in Swagger UI / Postman
python scripts/generate_openapi.py
cat docs/api/openapi.json | python -m json.tool | head -30
```

---

## 16. Compliance + governance

The regulatory pack (`docs/regulatory/REGULATORY_PACK.md`) has 14
sections at 100% artefact coverage:

1. Intended purpose + risk classification (EU AI Act Art. 6 + Annex IV)
2. Calibration + external validation (FDA 510(k) Bench)
3. Fairness + bias governance (EU AI Act Art. 10)
4. Audit trail + record-keeping (EU AI Act Art. 12-13)
5. Adversarial robustness + design verification (ISO 13485 §7.3)
6. Performance + scalability (FDA SVV)
7. Quality Management System (ISO 13485)
8. Post-market monitoring + incident reporting (EU AI Act Art. 17 + 20)
9. **FDA 510(k) predicate device comparison** (added Phase 16.G1)
10. **GDPR Art. 35 Data Protection Impact Assessment**
11. **HIPAA §164 Privacy + Security Rule crosswalk**
12. **Cybersecurity + SBOM** (FDA 524B + NIST SP 800-63B)
13. **Human oversight + emergency override** (EU AI Act Art. 14)
14. **EU AI Act conformity assessment readiness**
    (Articles 9-15 + 43 + Annex VI)

The framework crosswalks doc (`docs/regulatory/FRAMEWORK_CROSSWALKS.md`)
maps every TrustedRisk capability to NIST AI RMF 1.0 (14 rows across
GOVERN/MAP/MEASURE/MANAGE) + OECD AI Principles 2019 (5 rows).

The model card (`docs/research/MODEL_CARD.md`) follows Mitchell 2019
(9 sections) + the datasheet (`docs/research/DATASHEET.md`) follows
Gebru 2021 (7 sections).

**Test it yourself**:
```bash
PYTHONPATH=src pytest tests/unit/test_regulatory_pack.py \
  tests/unit/test_framework_crosswalks.py \
  tests/unit/test_model_card.py -v

# Regenerate + read
python scripts/generate_regulatory_pack.py
python scripts/generate_framework_crosswalks.py
python scripts/generate_model_card.py
less docs/regulatory/REGULATORY_PACK.md
```

---

## 17. Demo assets

Five visual / interactive demos for a judge:

1. **`docs/showcase/STORYMODE.html`** -- 6 narrative patient cases
   (Eleanor Thompson, Marcus Williams, Ana Lucia Hernandez, Vinh
   Nguyen, Chayton Bear, Sarah Cohen) with chart text + structured
   demographics + full pipeline run + colour-coded debate + advocate
   verdicts.

2. **`docs/e2e/v7/index.html`** -- 8 cross-bundle scenarios
   (S1 ICU sepsis, S2 cardiology AFib + ACS, S3 heme/onc staging,
   S4 peri-op rheumatology, S5 neuro emergency, S6 OB/peds advanced,
   S7 GI/hepatology + ID, S8 outpatient panel + research models +
   EHR ingest) -- 53 tool calls, 16 distinct Phase-13/14 bundles
   touched.

3. **`docs/ui/counterfactual.html`** -- interactive counterfactual
   explorer; LACE sliders update calibrated risk + recommended action
   in real time + show the minimum-L1 LACE perturbation that would
   flip the recommendation. Self-contained vanilla JS, 7 KB.

4. **`docs/regulatory/REGULATORY_PACK.md`** -- 14-section EU AI Act
   + FDA 510(k) + ISO 13485 + GDPR + HIPAA pack at 100% artefact
   coverage.

5. **`docs/api/openapi.json`** -- OpenAPI 3.1 spec for the 9
   canonical federation endpoints; can be opened directly in Postman
   or Swagger UI.

**Test it yourself**:
```bash
# Build everything in one shot
python scripts/build_all_artefacts.py

# Open the demos in your browser
start docs/showcase/STORYMODE.html       # Windows
open docs/showcase/STORYMODE.html        # macOS
xdg-open docs/showcase/STORYMODE.html    # Linux
```

---

## 18. Submission readiness checklist

Run through this before the 2026-05-11 submission:

- [ ] `python scripts/build_all_artefacts.py` exits with `17 OK / 0 FAIL`.
- [ ] `pytest tests/ -q` exits with all green.
- [ ] `pytest tests/integration/test_phase17_cross_module_pipeline.py
        tests/integration/test_build_all_artefacts.py -v` passes.
- [ ] `docs/regulatory/REGULATORY_PACK.md` shows
        "Artefact coverage**: 100.0%".
- [ ] `docs/federation/FEDERATION_REGISTRY.md` shows
        "100.0% coverage".
- [ ] `docs/MODULE_CATALOG.md` lists ≥ 70 `a2a_agent` modules.
- [ ] `docs/api/openapi.json` validates as OpenAPI 3.1 in your
        editor of choice.
- [ ] CHANGELOG.md is up to date for v0.8.0.
- [ ] `pyproject.toml` version field is `"0.8.0"`.
- [ ] The five demo HTML files render correctly in a fresh browser
        window.

---

## 19. Troubleshooting

### `ImportError: No module named 'mcp_server'`

Cause: missing `PYTHONPATH=src` prefix on the command. All commands
in this guide assume `PYTHONPATH=src` is set.

### `ECE > 0.05` after recalibration

Cause: the calibration cohort doesn't match the W1 generator. Either
re-run `scripts/synthea_100k_recalibrate.py` to refresh the cohort
or check that `data/coefficients.json` hasn't been overwritten.

### Federation registry shows < 100% bundle coverage

Cause: a new bundle was added to `mcp_server.tools.BUNDLES` but no
specialist owns it. Run
`python scripts/extend_specialist_bundle_coverage.py` to extend the
default specialist mapping; if the bundle is genuinely new, add it
to `_BUNDLE_TO_OWNER` in `a2a_agent.federation_registry`.

### Encoding errors on Windows when running scripts

Cause: cp1252 console can't render UTF-8 special characters. Use
plain-ASCII characters in `print()` statements; the artefacts on
disk are always written with `encoding="utf-8"` so they render fine
in any text editor.

### Build pipeline times out

Cause: a generator is hanging on a network call (the calibration
workflows can do this if `TRUSTEDRISK_FAIRNESS_BASELINE_PATH` points
to a remote URL). Ensure no remote dependencies are configured
before `build_all_artefacts.py` runs; it should complete in under
10 seconds locally.

---

## Final note

If you can run `python scripts/build_all_artefacts.py` and see
"17 OK / 0 FAIL" + open `docs/showcase/STORYMODE.html` in a
browser, you have a working, end-to-end TrustedRisk. The release
is the codebase + the artefacts + this guide; nothing has been
hand-curated outside the deterministic generators.
