# TrustedRisk Bundles

TrustedRisk's 58 MCP tools are grouped into **20 thematic bundles** to aid
agent discovery and client-side filtering. Each bundle is a curated subset
of tool function names that an upstream A2A client can request via MCP
capability negotiation.

## Clinical-vertical bundles (1-12)

| Bundle | Tools | Use case |
|---|---:|---|
| [`core_discharge`](core_discharge.md) | 10 | Original safe-discharge workflow (LACE risk + grounding + medication safety + counseling) |
| [`ed_acute`](ed_acute.md) | 5 | Emergency department triage + inpatient deterioration |
| [`pediatric`](pediatric.md) | 2 | Age-band-aware pediatric assessment |
| [`mental_health`](mental_health.md) | 4 | Suicide risk + psychiatric admission with bias guard |
| [`antimicrobial`](antimicrobial.md) | 3 | Empiric antibiotic selection + de-escalation |
| [`oncology`](oncology.md) | 4 | Chemo dose adjustment + RECIST 1.1 response |
| [`stroke_acs`](stroke_acs.md) | 5 | Acute stroke (NIHSS + tPA/EVT) + chest pain (HEART + ACS pathway) |
| [`obstetric_geriatric`](obstetric_geriatric.md) | 5 | MEOWS + preeclampsia + falls + delirium |
| [`trauma_critical`](trauma_critical.md) | 3 | ISS/RTS + massive transfusion + blunt-trauma imaging |
| [`endocrine_acute`](endocrine_acute.md) | 2 | DKA severity + inpatient glycemic control |
| [`imaging`](imaging.md) | 3 | ACR appropriateness + contrast safety |
| [`nephrology`](nephrology.md) | 4 | KDIGO AKI staging + dialysis initiation |

## Cross-cutting bundles (13-20)

| Bundle | Tools | Use case |
|---|---:|---|
| `economics` | 3 | Cost-effectiveness + EVOI per intervention; ICER ladder feed (IMPACT-1) |
| `context_resolution` | 3 | Conversational SHARP patient resolver (LLM with deterministic regex floor) (LLM-3) |
| `diagnosis` | 3 | Grounded differential diagnosis ranker; can't-miss diagnoses surface unconditionally (LLM-4) |
| `patient_facing` | 4 | Discharge counseling + 5-language translator (en/es/zh/vi/ar) + FAQ + caregiver summary (PATIENT-1/2/3) |
| `data_normalization` | 4 | Charlson + Elixhauser + LOINC + RxNorm DDI + UMLS crosswalk (DATA-1/2/3/4) |
| `clinical_workflow` | 4 | Order-set + adherence + HEDIS care-gap + PROM influence (LIB-1/2/3/4) |
| `external_knowledge` | 4 | PubMed (Entrez) + ClinicalTrials.gov v2 + drug pricing + NIH RePORTER v2 (KNOWLEDGE-1/2/3/4) |
| `chart_intelligence` | 3 | Clinical NER + negation/temporal context (NegEx) + discharge-summary structurer (CHART-1/2/3) |

Walkthrough pages for the 8 cross-cutting bundles are TODO; the runtime
contract is the same as for the clinical-vertical bundles.

Tools that are generally useful (`ground_claim`, `compute_fairness_audit`,
`compute_lab_trend_analysis`, `detect_phi`, `compute_aki_kdigo_stage`,
`compute_imaging_appropriateness`, `compute_contrast_safety_check`,
`compute_admission_triage`, `compute_readmission_risk`) appear in multiple
bundles. The deduplicated tool count is 58.

## Cross-bundle composition

Some clinical workflows naturally cross bundle boundaries. The TrustedRisk
runtime supports this -- tool calls from different bundles can be composed
freely. The end-to-end showcase at `docs/e2e/index.html` includes three
explicit cross-bundle scenarios:

- **Q (polytrauma)**: `trauma_critical` + `ed_acute` + `core_discharge`
- **R (DKA + AKI + contrast)**: `endocrine_acute` + `nephrology` + `imaging`
- **S (severe preeclampsia -> PPH -> discharge)**: `obstetric_geriatric` +
  `trauma_critical` + `core_discharge`

## How agents request a bundle

The bundle list is exposed both:
1. Via the agent-card at `src/a2a_agent/agent-card.json` under `_bundles`.
2. Via the runtime helper `mcp_server.tools.list_bundles()` (returns
   `dict[str, list[str]]`).

A2A clients can call `list_bundles()` over the MCP capabilities response
and filter the tool surface they expose to their LLM by bundle ID.

## When NOT to use bundles

- If the patient's situation crosses bundles (e.g. trauma + AKI), expose
  the union -- bundles are a discovery aid, not a hard partition.
- If the client is internal/trusted, expose the full 58-tool surface and
  let the LLM router pick.
- If the use case is brand-new (not in any bundle), default to
  `core_discharge` + `ground_claim` + the agent's self-critique step.

## Beyond bundles -- agent-level capabilities

Bundles describe *which tools* are available. The TrustedRisk agent adds
several capabilities that operate ACROSS bundles:

- **Multi-critic ensemble** (4 independent critics: clinical_safety,
  fairness, evidence, llm_judge) votes on every candidate DecisionCard --
  see [`src/a2a_agent/critique.py`](../../src/a2a_agent/critique.py).
- **Plan revision loop** with bounded retries -- see
  [`src/a2a_agent/plan_revision.py`](../../src/a2a_agent/plan_revision.py).
- **Longitudinal memory** with patient timeline + drift detection --
  see [`src/a2a_agent/memory.py`](../../src/a2a_agent/memory.py).
- **Batch endpoint** for throughput-bound use cases (overnight
  triage, pre-admission screening) -- see
  [`src/a2a_agent/batch.py`](../../src/a2a_agent/batch.py).
- **Time-bounded validity** (LACE-scaled re-call windows) and
  **multi-tenant OAuth** with per-client FHIR allowlist.

## Validation evidence

Three layers of validation:

1. **External calibration** -- `scripts/external_validation.py` runs the
   calibrated readmission model on a cross-shifted cohort with
   bootstrap subgroup CIs.
2. **Golden cases** -- `tests/golden/cases/*.json` regression suite
   (36 cases, ~18 tools).
3. **Unit + integration + property + adversarial + functional tests** --
   1847 tests across `tests/{unit,integration,property,golden,
   llm_integration,adversarial,eval,regression,functional}/`. The
   `functional/` layer (181 tests / 15 files) exercises every pipeline
   end-to-end.

Full report: [`docs/validation/VALIDATION.md`](../validation/VALIDATION.md).

## Interactive playground

To explore the bundles interactively, run the FastAPI playground:

```bash
PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \
    .venv/Scripts/python.exe -m uvicorn apps.playground.server:app \
    --port 8765
```

Then open `http://localhost:8765/`. Three tabs: Scenarios (19 demo
cases), Patient timeline (cross-session memory + drift flags), A2A
federation (mock external-agent handshake).
