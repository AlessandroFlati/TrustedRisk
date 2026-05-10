# Bundle: `imaging` -- Imaging Appropriateness + Contrast Safety

ACR Appropriateness Criteria for common ED scenarios + Choosing Wisely lens; ACR Manual on Contrast Media for iodinated / gadolinium safety gating (eGFR / metformin / allergy / pregnancy).

## When to use

- Pre-imaging order: which modality is most appropriate for this scenario?
- Pre-contrast: is iodinated/gadolinium contrast safe given eGFR / metformin / allergy / pregnancy?
- Pediatric ALARA + cumulative-radiation review.

## Typical tool sequence

1. compute_imaging_appropriateness(clinical_scenario, age, pregnant, cumulative_radiation) -- ranked options
1. compute_contrast_safety_check(contrast_type, eGFR, on_metformin, allergy, pregnant) -- proceed/abstain + premedication

## Minimum inputs

- Clinical scenario slug (chest_pain_acs_workup, suspected_pe_high_risk, etc.)
- Patient age (drives pediatric ALARA)
- Pregnancy flag
- For contrast: eGFR, metformin use, prior reaction history, pregnancy

## Composed output

ImagingAppropriateness (ranked modalities with appropriateness 1-9 + radiation dose) + ContrastSafetyReport (proceed flag + metformin hold + premedication).

## Demo scenarios

- R (DKA + AKI + contrast gate cross-bundle)

## Tools in this bundle

- `compute_imaging_appropriateness`
- `compute_contrast_safety_check`
- `compute_aki_kdigo_stage`

---

_Bundle id_: `imaging` &middot; _Tools_: 3 &middot; _Auto-generated from `mcp_server.tools.BUNDLES`_.
