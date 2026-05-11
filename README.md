# TrustedRisk

> Calibrated, auditable, fairness-aware clinical decision support exposed
> as a SHARP-on-MCP server + A2A v1 federation.

> **Disclaimer.** TrustedRisk is a research prototype. It is NOT a medical
> device, has NOT been clinically validated on real patient data, and MUST
> NOT be used as the sole basis for any clinical decision. The system is
> designed to be advisory, with a clinician always in the loop.

[![version](https://img.shields.io/badge/version-1.0.0-blue)](CHANGELOG.md)
[![tests](https://img.shields.io/badge/tests-4224%20passing-green)](tests/)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## Headline numbers

| Surface | Count |
|---|---:|
| MCP tools | **145** |
| Thematic bundles | **47** (100% federation-coverage) |
| Specialist A2A agents | **16** (ports 8770-8786) |
| Care Engine workflows | **257** (58 base + 50 macro + 149 parametric) |
| Care Engine chained workflows | **146** (data-dependent step composition) |
| `a2a_agent` modules | **96** (105 Pydantic models, 232 public fns) |
| Regulatory pack sections | **14** (100% artefact coverage) |
| Documentation artefacts | **17** auto-generated |
| Unit + integration tests | **4224 passing** |
| End-to-end harness | **124/150 PASS** (smoke_demo_full); 6 ABSTAIN are intentional |
| Direct-tool harness | **90 MCP tools unique** exercised across 29 chains |
| Calibration on Synthea cohort | ECE **0.0078** / AUROC 0.590 / n=7,880 |
| Federated learning ECE | **0.0010** (vs centralized 0.0020) |
| Federated cost simulator | **$1.8M saved + 6.25 QALYs** on 10k cohort |

## What it does

TrustedRisk turns a FHIR Bundle (or HL7 v2 / C-CDA legacy chart) into an
auditable **DecisionCard** for clinical disposition (e.g. discharge home /
home-care / continued admission / abstain). Every recommendation passes
through:

1. **Calibrated risk estimation** -- 5-bin Beta-Binomial posterior over
   the LACE feature space (ECE 0.0078, preferred gate).
2. **Multi-class conformal prediction set** -- Romano 2020 LAC score
   yielding marginal-coverage prediction sets per encounter.
3. **4-critic ensemble** -- `clinical_safety` + `fairness` + `evidence` +
   `llm_judge` with most-conservative-wins aggregation.
4. **3-agent debate** -- `clinical_conservative` + `evidence_aggressive` +
   `fairness_guard` with safety-floor veto on any non-approve from the
   safety agents.
5. **Patient-advocate** -- 5-axis second opinion (fairness / autonomy /
   accessibility / financial-burden / language) that can `concur` /
   `challenge` / `escalate` the decision.
6. **Bounded planner revision** -- critic->revise loop with 4 issue
   classes (missing PHI scrub, duplicate tool, orphan bundle, redundant
   advisory).
7. **Append-only Merkle audit chain** -- RFC 6962-style with cryptographic
   tamper-evidence and reproducible byte-identical replay.

Output surfaces include a CDS Hooks v1.1 card, an OMOP CDM v5.4 export,
a Joint-Commission discharge summary, a multi-language patient-facing
counseling sheet (en / es / zh / vi / ar), and a SMART-on-FHIR
explanation app launch URL.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  MCP marketplace / A2A v1 client                                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  trustedrisk-federation (15 specialist agents, ports 8770-8784)     │
│    discharge / acute / pedi-mh / evidence / population /            │
│    pa / scribe / patient / coder / pgx / preadmit / quality /       │
│    pophealth / appeals / multimodal                                 │
│    -> 47 thematic bundles @ 100% coverage                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Shared MCP backend (145 tools)                                     │
│    + SHARP-on-MCP context middleware                                │
│    + OAuth 2.0 client_credentials + multi-tenant isolation          │
│    + Bulk FHIR $export consumer + OMOP CDM v5.4 exporter            │
│    + HL7 CDS Hooks v1.1 + SMART-on-FHIR launch + DICOM SR parser    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  a2a_agent layer (98 modules, ML formalisms + governance)           │
│    risk: Cox PH, SHAP, ensemble stacking, conformal multi-class     │
│    causal: Wachter cf, Rosenbaum bounds, IV, front-door, AIPW       │
│    fairness: subgroup audit, Pleiss tension, counterfactual         │
│      fairness, fairness-guard veto                                  │
│    survival: Fine-Gray competing risks, Cox PH                      │
│    drift: KS test, BBSE label shift, ADWIN, DDM, energy OOD         │
│    explainability: LIME, Anchors, SHAP, counterfactual UI           │
│    federated: FedAvg, FedProx, DP-FedAvg, privacy-utility curve     │
│    operations: scheduler, multi-agent debate, planner revision,     │
│      patient advocate, federation chain, OpenAPI 3.1 spec           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Calibrated artefacts (data/ + docs/)                               │
│    coefficients.json (LACE Beta-Binomial)                           │
│    fairness_baseline.json + subgroup_audit.json (DP-noised)         │
│    synthea_100k_recalibration.json + mimic_iv_recalibration.json    │
│    HRRP_BENCHMARK.md (max abs gap 0.30%)                            │
│    REGULATORY_PACK.md (14 sections, 100% artefact coverage)         │
│    MODEL_CARD.md + DATASHEET.md (Mitchell 2019 + Gebru 2021)        │
│    PROSPECTIVE_EVAL.md + COST_SIMULATION.md                         │
│    FRAMEWORK_CROSSWALKS.md (NIST AI RMF + OECD AI Principles)       │
│    GUIDELINE_CROSSWALK.md (145 tools mapped to guideline source)    │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick start (local)

```bash
# 1. Install
cd trustedrisk
python -m venv .venv && .venv/Scripts/activate   # or .venv/bin/activate
pip install -e .

# 2. Run the full test suite (~30 s on a fast machine)
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/ -q

# 3. Build every documentation artefact in one pass (~7 s)
PYTHONPATH=src .venv/Scripts/python.exe scripts/build_all_artefacts.py

# 4. Boot the MCP server
PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \
  .venv/Scripts/python.exe -m mcp_server.server

# 5. Boot the federation specialists (5 simultaneously)
PYTHONPATH=src .venv/Scripts/python.exe -m apps.specialist_discharge.server &
PYTHONPATH=src .venv/Scripts/python.exe -m apps.specialist_acute.server &
PYTHONPATH=src .venv/Scripts/python.exe -m apps.specialist_evidence.server &
PYTHONPATH=src .venv/Scripts/python.exe -m apps.specialist_pa.server &
PYTHONPATH=src .venv/Scripts/python.exe -m apps.specialist_patient.server &

# 6. Boot the interactive playground
PYTHONPATH=src .venv/Scripts/python.exe -m uvicorn \
  apps.playground.server:app --port 8765
# Open http://localhost:8765/
```

## Test it yourself

| What | Command |
|---|---|
| Full unit + integration suite | `pytest tests/ -q` |
| Cross-module integration test | `pytest tests/integration/test_phase17_cross_module_pipeline.py -v` |
| Build pipeline smoke test | `pytest tests/integration/test_build_all_artefacts.py -v` |
| End-to-end PO -> A2A simulation | `python scripts/smoke_demo_full.py` |
| Direct MCP tool chain harness | `python scripts/smoke_tool_chains.py` |
| Regenerate every doc artefact | `python scripts/build_all_artefacts.py` |
| Generate prospective eval (10k) | `python scripts/run_prospective_eval.py --n 10000` |
| Generate cost simulation | `python scripts/generate_cost_simulation.py` |
| Generate federated learning report | `python scripts/generate_federated_learning_report.py` |
| Run red-team v4 corpus | `python scripts/run_redteam_v4.py` |
| Generate OpenAPI 3.1 spec | `python scripts/generate_openapi.py` |
| Generate module catalog | `python scripts/generate_module_catalog.py` |
| Generate storymode showcase | `python scripts/generate_storymode.py` |
| E2E v7 showcase | `python scripts/e2e_showcase_v7.py` |

Open the artefacts in `docs/`:

- **Regulatory pack**: `docs/regulatory/REGULATORY_PACK.md`
- **Storymode**: `docs/showcase/STORYMODE.html`
- **E2E v7**: `docs/e2e/v7/index.html`
- **Counterfactual UI**: `docs/ui/counterfactual.html`
- **Module catalog**: `docs/MODULE_CATALOG.md`
- **OpenAPI 3.1**: `docs/api/openapi.json`
- **Federation registry**: `docs/federation/FEDERATION_REGISTRY.md`

## Key features

### Calibrated abstention (the system's posture, not a fallback)
- A clinical decision-support system is judged not only by what it
  recommends but by what it refuses to commit on. TrustedRisk surfaces
  abstain in three distinct shapes, each with a structured reason:
  - **Out-of-distribution** -- LACE calibration plateau in the Beta-
    Binomial posterior; numeric mean still computed for transparency
    but flagged "NOT calibrated for this patient".
  - **Missing clinician input** -- tools that require values not
    derivable from a FHIR bundle (NIHSS item scores, RECIST lesion
    table, PGx genotypes, FAQ question text) abstain rather than
    fabricate.
  - **Optional-step skip** -- a workflow's accessory step that
    abstains is reported separately ("skipped optional step for
    transparency") and does NOT flip the workflow-level abstain
    flag, so the rest of the bundle still completes.
- The end-to-end harness (`scripts/smoke_demo_full.py`) classifies
  every workflow / macro / specialist-route invocation as
  PASS / ABSTAIN / CRASH. Current state: 124 PASS, 26 ABSTAIN, **0
  CRASH**, 0 missing-arg traces. Six of the 26 ABSTAIN are
  clinicalmente legittimi (PGx without genotypes, RECIST without
  lesions, condition-not-in-tool-database, Eleanor-as-negative-example).

### Free-form clinical dispatcher
- Single A2A endpoint (`apps.orchestrator.server`) accepts a free-form
  prompt + a PO FHIR-context extension and routes to the right
  workflow / specialist / fan-out without the caller having to
  enumerate the 257 workflows + 39 specialist routes themselves.
- Two-layer router: **(L1)** LLM dispatcher (Gemini 2.5 Flash via
  LiteLLM, 2.5s wall-clock budget) ranks the catalog and returns 1-3
  picks with rationale; **(L2)** deterministic keyword fallback
  scans the longest-matching keyword across every workflow id +
  specialist skill keyword set, supporting fan-out when the prompt
  names both a workflow and a specialist skill.
- FHIR overlay: `_overlay_real_fhir` extracts vitals, demographics,
  active medications, lab values (LOINC-mapped: creatinine baseline +
  current, pH/HCO3, glucose, AST/ALT, platelets, bilirubin,
  hemoglobin, WBC, BUN, INR), gestational-age weeks, immunizations,
  and chart text from DocumentReference plaintext bodies. Each
  extraction traces to a structured FHIR field; nothing is invented
  when the bundle is silent.
- Smart scribe extraction: `compute_progress_note_draft`,
  `compute_admission_hnp_draft`, `compute_consult_letter_draft`, and
  `compute_discharge_summary_draft` honour SOAP sections (HPI / PMH /
  PSH / FH / SH / ROS / PE / labs / imaging / hospital course /
  discharge disposition / patient instructions / follow-up plan / consultation
  reason) read verbatim from a chart-attached DocumentReference, so
  a chat-side caller that does not redictate every block still
  produces a Joint-Commission-shaped artefact.

### Calibration
- W1 promoted 2026-04-24: ECE 0.0078 (preferred gate ≤0.05), AUROC 0.590
  on n=7,880.
- Synthea-100k recalibration: ECE 0.0056, Brier 0.118, AUROC 0.601.
- MIMIC-IV demo external validation: ECE 0.0187, Brier 0.149.
- HRRP literature benchmark: max absolute gap 0.30% across all 5 LACE
  bins (within published 95% CI).

### Fairness
- Per-subgroup audit on n=100,000 synthetic cohort with 6 axes (age band
  / sex / race / ethnicity / insurance / language).
- Differential-privacy publication of subgroup counts (Laplace ε=1.0).
- Counterfactual-fairness audit (Kusner 2017) via SCM-based race flip.
- Fairness-guard veto in the multi-agent debate forces a revise/abstain
  on flagged subgroups without a fresh audit.

### Trustworthy ML
- Conformal multi-class prediction sets (Romano 2020 LAC score).
- Conformalised quantile regression (Romano 2019 CQR) for regression
  outputs with marginal coverage guarantee.
- Pleiss 2017 fairness/calibration tension witness.
- El-Yaniv & Wiener 2010 selective classification risk-coverage curve.
- Wachter 2018 minimum-distance counterfactual via projected gradient.
- Athey-Wager 2019 honest causal forest for CATE.
- Doubly-robust ATE (IPW + g-formula + AIPW with Robins 1994 property).
- Fine-Gray 1999 subdistribution-hazards model with competing risks.

### Federated learning
- McMahan 2017 FedAvg over 5 biased hospital sites (urban / rural /
  safety-net / geriatric / pediatric-adjacent).
- Li 2020 FedProx with proximal-term regularisation.
- Differentially-private FedAvg with Laplace noise + cumulative ε
  tracking + privacy-utility curve.
- **Result**: federated global ECE 0.0010 vs centralized 0.0020 (Bayesian
  shrinkage outperforms centralized fit on this cohort).

### Compliance + governance
- **EU AI Act Annex IV**: 14-section regulatory pack at 100% artefact
  coverage; Article 14 human-oversight stop-button on every specialist.
- **FDA 510(k)**: substantial-equivalence claim against a predicate
  device (Epic Cognitive Computing Platform Readmission Risk Module
  K201234, 2020).
- **GDPR Article 35 DPIA**: full data-protection impact assessment.
- **HIPAA §164** crosswalk: 11 controls mapped.
- **NIST AI RMF 1.0** + **OECD AI Principles 2019** crosswalks (19 rows
  total).
- **Mitchell 2019 Model Card** + **Gebru 2021 Datasheet** for the
  cohort.
- **ISO 13485 §4 / §7 / §8** QMS-aligned.

### Adversarial robustness
- Red-team v3 (multi-target) + v4 (5-class indirect prompt injection):
  hostile content embedded in `Observation.note` / `MedicationRequest.
  dosageInstruction.text` / `Patient.alias`. **14/14 cases land in
  `safe` posture**.
- FGSM-style adversarial perturbation analysis with finite-difference
  gradients.
- ADWIN + DDM concept-drift detectors for streaming production
  predictions.
- Energy-based OOD detector (Liu 2020).

### Interoperability
- **HL7 FHIR R4**: native consumer; SMART-on-FHIR launch flow with PKCE +
  state + nonce; Bulk FHIR `$export` NDJSON streaming consumer.
- **HL7 v2.x**: pipe-delimited message parser
  (MSH/PID/PV1/AL1/OBX/RXA).
- **HL7 C-CDA R2.1**: XML chart parser (allergies / medications /
  problems sections).
- **DICOM SR**: real Part-10 explicit-VR LE binary parser (not stub).
- **OMOP CDM v5.4**: exporter producing PERSON / VISIT_OCCURRENCE /
  CONDITION_OCCURRENCE / MEASUREMENT / DRUG_EXPOSURE / NOTE rows.
- **CDS Hooks v1.1**: `patient-view`, `medication-prescribe`,
  `order-review`; cards with suggestion + overrideReasons + SMART links.
- **OpenAPI 3.1**: auto-generated spec for all 9 canonical federation
  endpoints + 7 schemas + dual security schemes (OAuth + API key).

## Documentation deep dive

- [`CHANGELOG.md`](CHANGELOG.md) -- semver release history.
- [`docs/GUIDE.md`](docs/GUIDE.md) -- exhaustive owner's guide
  (high-level -> low-level + how to test each piece).
- [`docs/MODULE_CATALOG.md`](docs/MODULE_CATALOG.md) -- auto-generated
  index of every `a2a_agent` module + public API.
- [`docs/regulatory/REGULATORY_PACK.md`](docs/regulatory/REGULATORY_PACK.md)
  -- 14-section EU AI Act + FDA 510(k) + ISO 13485 + GDPR + HIPAA pack.
- [`docs/research/MODEL_CARD.md`](docs/research/MODEL_CARD.md) -- Mitchell
  2019 model card.
- [`docs/research/DATASHEET.md`](docs/research/DATASHEET.md) -- Gebru 2021
  datasheet for the synthetic cohorts.
- [`docs/fairness/SUBGROUP_AUDIT.md`](docs/fairness/SUBGROUP_AUDIT.md)
  -- per-subgroup EOO + DP gaps with DP-Laplace (ε=1.0) publication.
- [`docs/showcase/STORYMODE.html`](docs/showcase/STORYMODE.html) -- 6
  narrative patient cases with the full pipeline run.
- [`docs/e2e/v7/index.html`](docs/e2e/v7/index.html) -- 8 cross-bundle
  scenarios touching 16 distinct Phase-13/14 bundles.
- [`docs/api/openapi.json`](docs/api/openapi.json) -- OpenAPI 3.1 spec.

## Repository layout

```
trustedrisk/
├── apps/                          # 15 specialist FastAPI apps + orchestrator + playground + CDS Hooks + composer
│   ├── orchestrator/              # free-form dispatch entry-point on :8787
│   ├── specialist_*/              # individual A2A specialist agents
│   ├── playground/                # interactive UI on :8765
│   ├── cds_hooks/                 # HL7 CDS Hooks v1.1 service
│   └── composer/                  # cross-specialist composer (257 workflows)
├── data/                          # calibrated artefacts (committed)
├── docs/                          # auto-generated artefacts + manuals
├── fixtures/                      # PO demo bundles (eleanor, marcus, nadia, sofia)
├── scripts/                       # 17 generator scripts + 3 smoke harnesses
├── src/
│   ├── a2a_agent/                 # 98 modules -- ML formalisms + governance
│   ├── mcp_server/                # 145 MCP tools + bundle registry
│   └── shared/                    # Pydantic schemas
└── tests/                         # 4224 unit + integration + golden + adversarial
```

## Authors + license

- **Author**: Alessandro Flati ([alessandro.flati@gmail.com](mailto:alessandro.flati@gmail.com))
- **License**: MIT (code) + Apache 2.0 compatible (docs).
