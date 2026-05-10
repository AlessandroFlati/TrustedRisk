# Bundle: `economics` -- Health-Economics Evaluation

Cost-effectiveness analysis (CEA) and Expected Value of Information (EVOI)
on candidate interventions, anchored on the calibrated readmission risk and
subgroup fairness audit so the economic claims trace back to the same
deterministic substrate as the clinical recommendation.

## When to use

- Comparing alternative discharge interventions (e.g. home_with_care vs
  SNF) on cost-per-QALY against a literature benchmark.
- Pre-computing the population-level EVOI of running an additional test
  (e.g. lab panel) before committing to a higher-acuity action.
- Feeding the cost-effectiveness ladder
  (`a2a_agent.cost_effectiveness_ladder`) for `/api/economics/ladder`.

## Typical tool sequence

1. compute_readmission_risk(patient_id) -> calibrated baseline
1. compute_fairness_audit(risk, demographics) -> subgroup decomposition
1. compute_expected_value_of_intervention(baseline_risk, intervention_specs)
   -> ICER + per-intervention EVOI

## Tools in this bundle

- `compute_expected_value_of_intervention` (IMPACT-1)
- `compute_readmission_risk` (baseline-risk feed)
- `compute_fairness_audit` (subgroup decomposition feeds CEA)

## Companion modules

- `a2a_agent.outcomes_simulator` -- Monte Carlo hospital-year simulator
  (n=1000 default).
- `a2a_agent.cost_effectiveness_ladder` -- ICER ladder per intervention.
- `a2a_agent.impact_aggregator` -- population-level rollups (cumulative
  QALYs, $$ saved against a counterfactual).
- `a2a_agent.causal_inference` -- DoWhy ATE with backdoor adjustment for
  confounder-aware effect estimates.

## Outputs

`InterventionEVReport` with per-intervention ICER, EVOI, point estimate
+ CI, and the reference benchmark used for the ICER comparison.

## Minimum inputs

- A baseline risk (typically from `compute_readmission_risk`) -- point
  estimate + CI95.
- One or more candidate interventions, each with: name, RRR (relative
  risk reduction), per-patient cost, per-avoided-event cost.
- Optional: subgroup fairness decomposition for stratified ICER.

## Composed output

`InterventionEVReport` (one per intervention) with:
- ICER ($/QALY)
- EVOI ($)
- Point estimate + 95% CI
- Reference benchmark used for the ICER comparison

## Demo scenarios

- v5 showcase: scenario S (severe preeclampsia -> emergency C-section
  -> PPH -> discharge) -- cross-bundle composition with `core_discharge`
- v6 showcase: scenarios PA-imaging-mri-lumbar (cost-per-QALY framing
  for diagnostic imaging)

## Caveats

- US-centric cost benchmarks (HCUP Statistical Brief #248). Institutions
  outside the US should re-fit on local cost data.
- The cost-per-QALY threshold is configurable; the default 100k USD / QALY
  is a US literature benchmark, not a regulatory threshold.
