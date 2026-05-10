# Bundle: `pediatric` -- Pediatric

Age-band-aware pediatric assessment. Pediatric vitals are NOT just adult vitals scaled -- both normal ranges and escalation thresholds shift with age.

## When to use

- Pediatric ED triage with PEWS (Brighton/Monaghan).
- Weight-based pediatric medication dosing (Broselow-tape style + adult cap).
- Verifying age contraindications (ibuprofen <6mo, ceftriaxone <28d, etc.).

## Typical tool sequence

1. compute_pediatric_early_warning(age_months, vitals) -> PEWS + escalation tier
1. compute_weight_based_dosing(drug, weight_kg, age_months, indication) -> dose + contraindications

## Minimum inputs

- Age in months (drives age-band selection)
- Weight in kg (for dosing)
- Vital signs OR drug name + indication

## Composed output

PEWSReport + PediatricDoseRecommendation, the latter with abstain on contraindications.

## Demo scenarios

- I (4-month-old fever + lethargy)

## Tools in this bundle

- `compute_pediatric_early_warning`
- `compute_weight_based_dosing`

---

_Bundle id_: `pediatric` &middot; _Tools_: 2 &middot; _Auto-generated from `mcp_server.tools.BUNDLES`_.
