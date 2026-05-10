# Bundle: `core_discharge` -- Core Discharge

Adult inpatient discharge planning -- LACE-calibrated 30-day readmission risk + medication safety + grounding + fairness audit + counterfactual + lab trends + patient-language counseling.

## When to use

- End-of-stay decision: discharge_home / home_with_care / SNF / continued_admission?
- Outpatient medication review (LACE = 0, follow-up window = 14-30d).
- Quality-assurance check on a discharge note (PHI scan + grounding).

## Typical tool sequence

1. compute_readmission_risk(patient_id) -> RiskEstimate (probability + LACE)
1. compute_decision_utility(outcome_probs, n_monte_carlo=1000) -- when ranking actions
1. compute_medication_reconciliation(patient_id) -> audit discharge med list
1. detect_polypharmacy_concerns(medications) -> DDI scan
1. compute_fairness_audit(risk, demographics) -> subgroup drift
1. compute_counterfactual_explanation(risk) -> what could change the call
1. compute_lab_trend_analysis(observations) -> optional, when labs drive concern
1. compute_discharge_counseling(meds, lace_score, action) -> patient summary
1. ground_claim(claim_text, patient_id) -> final disposition grounding
1. detect_phi(text) -> before any external release of free text

## Minimum inputs

- FHIR patient_id (resolves chart via SHARP context)
- Discharge medication list (or rely on FHIR MedicationRequest)
- Patient demographics for fairness audit (age, race, insurance)

## Composed output

DecisionCard with recommendation + reasoning + validation + abstain triggers + audit block.

## Demo scenarios

- A (75y CHF discharge)
- C (decision-utility ranking)
- D (batch HTTP)
- H (geriatric outpatient med review)

## Tools in this bundle

- `compute_readmission_risk`
- `compute_decision_utility`
- `ground_claim`
- `detect_phi`
- `compute_medication_reconciliation`
- `detect_polypharmacy_concerns`
- `compute_fairness_audit`
- `compute_counterfactual_explanation`
- `compute_lab_trend_analysis`
- `compute_discharge_counseling`

---

_Bundle id_: `core_discharge` &middot; _Tools_: 10 &middot; _Auto-generated from `mcp_server.tools.BUNDLES`_.
