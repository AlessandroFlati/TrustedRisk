# Bundle: `clinical_workflow` -- Discharge Order Set + Adherence + Care Gaps + PROM

Workflow-shaping tools that translate a `DecisionCard` into operational
artifacts: a structured discharge order set, an adherence-risk score, a
HEDIS-grounded care-gap report, and a PROM-influence surface that
re-weights action dominance based on patient-reported outcomes.

## When to use

- Generating a discharge order set that an EHR can ingest directly
  (medications, follow-up appointments, monitoring labs, specialty
  consults, patient education).
- Predicting medication adherence risk before a prescription is sent
  (e.g. cost barriers, polypharmacy fatigue, regimen complexity).
- Detecting open HEDIS care gaps for population-health programs.
- Re-weighting the action dominance based on PROMIS-29 / EQ-5D-5L /
  PROMIS-10 patient-reported scores.

## Typical tool sequence

1. compute_order_set(discharge_action, risk_estimate, medications,
   chief_complaint) -> categorised orders with rationale + timing
1. compute_medication_adherence_predictor(patient_id, medications) ->
   adherence risk + driver list
1. compute_care_gap_detector(patient_id, conditions) -> HEDIS-mapped
   open gaps (HbA1c overdue, BP control, mammography, etc.)
1. compute_prom_influence(prom_scores) -> QALY delta + action-dominance
   shift hint

## Tools in this bundle

- `compute_order_set` (LIB-1)
- `compute_medication_adherence_predictor` (LIB-2)
- `compute_care_gap_detector` (LIB-3, HEDIS)
- `compute_prom_influence` (LIB-4, PROMIS / EQ-5D)

## Determinism

- Order set is a pure-template generator -- no LLM. Each order carries
  a rationale + a reference chain so an EHR ingestion layer can import
  directly without human re-keying.
- Adherence prediction is a deterministic logistic over structured
  drivers (cost, complexity, prior adherence).
- Care-gap detection compares the FHIR Conditions + Observations against
  the HEDIS measure spec.
- PROM influence translates validated PROMIS / EQ-5D scoring tables;
  the action-dominance shift is heuristic.

## Minimum inputs

- For order set: `discharge_action`, optional `risk_estimate`,
  optional `medications[]`, optional `chief_complaint`
- For adherence: `patient_id`, `medications[]`
- For care-gap: `patient_id`, `conditions[]`
- For PROM influence: `prom_scores` (PROMIS-29 / EQ-5D-5L)

## Composed output

- `OrderSet` (categorised orders + per-order rationale + timing + priority)
- `AdherenceRiskReport` (risk score + driver list)
- `CareGapReport` (open HEDIS gaps + recommended actions)
- `PROMInfluenceReport` (QALY delta + action-dominance shift hint)

## Demo scenarios

- v5 showcase: scenario A (75y CHF) consumes order_set + adherence.
- v6 showcase: SCRIBE-discharge-summary uses these alongside the
  document drafter.

## Caveats

- Action-dominance shift is a heuristic -- institutions should re-fit
  on local outcomes before clinical use.
- HEDIS measure specs are versioned; the bundled snapshot is for the
  current measurement year and must be refreshed annually.
