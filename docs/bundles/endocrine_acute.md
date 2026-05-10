# Bundle: `endocrine_acute` -- Endocrine Acute

DKA severity (ADA 2009 / ISPAD) + initial fluid + insulin + K+ replacement protocol; inpatient glycemic control (basal-bolus titration + hypoglycemia risk assessment).

## When to use

- Admission for DKA: severity tier + ICU? + protocol.
- Inpatient hyperglycemia: titrate basal-bolus + correctional scale.
- Hypoglycemia review: pull basal down, switch from sliding-scale-only.

## Typical tool sequence

1. compute_dka_severity(pH, HCO3, glucose, mental_status, K+, weight) -- severity + protocol
1. compute_inpatient_glycemic_control(is_icu, avg_glucose, episodes, regimen, factors) -- titration

## Minimum inputs

- DKA: pH, bicarbonate, glucose, ketones, mental status, K+, weight
- Glycemic: is_icu, avg_glucose_24h, hypoglycemia/hyperglycemia episode counts, current regimen, hypoglycemia risk factors

## Composed output

DKASeverityReport (severity tier + ICU + fluid/insulin/K+ plan + bicarb gate) + InpatientGlycemicPlan (basal change% + correctional scale + hypoglycemia risk).

## Demo scenarios

- R (DKA + AKI cross-bundle)

## Tools in this bundle

- `compute_dka_severity`
- `compute_inpatient_glycemic_control`

---

_Bundle id_: `endocrine_acute` &middot; _Tools_: 2 &middot; _Auto-generated from `mcp_server.tools.BUNDLES`_.
