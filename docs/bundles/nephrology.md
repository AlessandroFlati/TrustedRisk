# Bundle: `nephrology` -- Nephrology

KDIGO 2012 AKI staging (creatinine + UOP criteria + etiology clue) + AEIOU dialysis initiation + STARRT-AKI logic for stage-3 AKI without emergent indication.

## When to use

- Hospitalized creatinine rise: stage AKI per KDIGO criteria.
- Emergent dialysis decision: AEIOU mnemonic + modality choice (HD vs CRRT).
- Pre-contrast eGFR check (composes with imaging bundle).

## Typical tool sequence

1. compute_aki_kdigo_stage(Cr_baseline, Cr_current, UOP, FENa, hydronephrosis) -- stage + etiology
1. compute_dialysis_initiation_decision(aki_stage, pH, K+, refractory_*, toxin, hemodynamic) -- AEIOU + modality
1. compute_lab_trend_analysis(observations) -- creatinine / K+ trends
1. compute_contrast_safety_check(contrast_type, eGFR) -- pre-imaging gate

## Minimum inputs

- Creatinine baseline + current
- Urine output (mL/kg/h × window hours)
- For dialysis: pH, K+, refractory flags, dialyzable toxin, hemodynamic stability

## Composed output

AKIStagingReport (stage + etiology + nephrology consult flag) + DialysisInitiationReport (AEIOU indications + urgency + modality).

## Demo scenarios

- R (DKA + AKI cross-bundle)

## Tools in this bundle

- `compute_aki_kdigo_stage`
- `compute_dialysis_initiation_decision`
- `compute_lab_trend_analysis`
- `compute_contrast_safety_check`

---

_Bundle id_: `nephrology` &middot; _Tools_: 4 &middot; _Auto-generated from `mcp_server.tools.BUNDLES`_.
