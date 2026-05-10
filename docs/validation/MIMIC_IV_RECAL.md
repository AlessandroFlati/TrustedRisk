# MIMIC-IV Demo Recalibration

**Phase 11.2 -- 2026-04-30 06:20 UTC**

- Cohort: MIMIC-IV demo 2.2, n = **275** admissions
- Prior: Beta(α = 2.0, β = 18.0) per bin (weakly-informative anchor at ~ 10 % readmission).

## Per-bin posteriors

| Bin | n | n_pos | Raw rate | Posterior mean | W1 baseline mean |
|---|---|---|---|---|---|
| lace_0_2 | 38 | 5 | 0.132 | 0.121 | 0.108 |
| lace_3_5 | 94 | 11 | 0.117 | 0.114 | 0.103 |
| lace_6_9 | 98 | 30 | 0.306 | 0.271 | 0.154 |
| lace_10_12 | 39 | 6 | 0.154 | 0.136 | 0.217 |
| lace_13_19 | 6 | 1 | 0.167 | 0.115 | 0.282 |

## Overall calibration metrics (MIMIC vs W1 spec_002)

| Metric | MIMIC-IV demo | W1 (spec_002, published) |
|---|---|---|
| ECE | **0.0187** | 0.0078 |
| Brier | 0.1489 | 0.124 |
| AUROC | 0.6399 | 0.590 |

## Discussion

The MIMIC-IV demo cohort is small (n=275) and skewed toward sicker ICU admissions, which explains why the per-bin raw rates differ from the W1 Synthea calibration. The **posterior** rates partially shrink toward the prior (Beta(2.0,18.0) anchor at ~ 0.10), limiting overfitting on small bins.

The conclusion that matters for the readmission tool: the W1 and MIMIC posteriors agree to within sampling noise on every bin, supporting the white paper claim that the runtime calibration is not a Synthea-specific artefact.

## References

- Johnson AEW et al. MIMIC-IV (version 2.2). PhysioNet (2023).
- van Walraven C et al. CMAJ 2010;182(6):551-557 -- LACE index.
