# Bundle: `oncology` -- Oncology Cycle Decisions

Chemotherapy cycle decision (proceed/reduce/delay/hold/discontinue) based on hematologic + renal + hepatic + ECOG gates; RECIST 1.1 response + decision implication.

## When to use

- Day-of-cycle review: proceed full / reduced / delay / hold?
- Restaging review: RECIST 1.1 categorical response + next-line decision.
- Treatment selection (e.g. T2D second-line) for cancer survivors.

## Typical tool sequence

1. compute_chemo_dose_adjustment(regimen, cycle, labs, ECOG) -- cycle decision
1. compute_oncology_treatment_response(target_lesions, new_lesions, ...) -- RECIST 1.1
1. compute_treatment_selection(condition, factors) -- when alternative regimens needed
1. compute_lab_trend_analysis(observations) -- ANC / platelet / Cr trends

## Minimum inputs

- Regimen name (slug)
- Cycle number
- Labs: ANC, platelets, eGFR, bilirubin, AST/ALT
- ECOG performance status
- For RECIST: target lesion baseline + current diameters + new-lesion / non-target flags

## Composed output

ChemoDoseAdjustment (decision + reduction% + growth-factor flag) + TumorResponse (CR/PR/SD/PD/NE + decision implication).

## Demo scenarios

- L (NSCLC cycle 4 with neutropenia)

## Tools in this bundle

- `compute_chemo_dose_adjustment`
- `compute_oncology_treatment_response`
- `compute_treatment_selection`
- `compute_lab_trend_analysis`

---

_Bundle id_: `oncology` &middot; _Tools_: 4 &middot; _Auto-generated from `mcp_server.tools.BUNDLES`_.
