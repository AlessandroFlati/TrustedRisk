# Synthea-10k Validation Report
**Generated**: 2026-04-29 21:56 UTC
**Cohort size**: 10,000 synthetic patients
**Calibrated artefact**: `data/coefficients.json` (spec_002)

## Summary metrics
- **ECE** (Expected Calibration Error, 10-bin) = `0.0056` (W1 target ≤ 0.08)
- **Brier score** = `0.1539`
- **AUROC** = `0.607`
- **Readmission rate** = `0.197` (19.7 %)

## Calibration table (predicted vs observed by decile)

| Bin | Predicted | Observed | n     | Bar (obs)             |
|-----|-----------|----------|-------|------------------------|
| 0   | 0.072     | 0.031    |    97 |                      |
| 1   | 0.147     | 0.154    |  5685 | ███                  |
| 2   | 0.234     | 0.231    |  3038 | ████                 |
| 3   | 0.327     | 0.331    |  1180 | ██████               |
| 4   | 0.000     | 0.000    |     0 |                      |
| 5   | 0.000     | 0.000    |     0 |                      |
| 6   | 0.000     | 0.000    |     0 |                      |
| 7   | 0.000     | 0.000    |     0 |                      |
| 8   | 0.000     | 0.000    |     0 |                      |
| 9   | 0.000     | 0.000    |     0 |                      |

## Subgroup ECE -- by race

| Race | n | ECE | Readmission rate |
|---|---:|---:|---:|
| asian | 601 | 0.0208 | 0.196 |
| black | 1,714 | 0.0242 | 0.216 |
| hispanic | 1,793 | 0.0072 | 0.196 |
| other | 408 | 0.0130 | 0.206 |
| white | 5,484 | 0.0071 | 0.190 |

## Subgroup ECE -- by age band

| Age band | n | ECE | Readmission rate |
|---|---:|---:|---:|
| 18-44 | 2,465 | 0.0074 | 0.200 |
| 45-64 | 2,996 | 0.0105 | 0.201 |
| 65-74 | 2,212 | 0.0093 | 0.192 |
| 75-84 | 1,603 | 0.0097 | 0.195 |
| 85+ | 724 | 0.0072 | 0.188 |

## Subgroup ECE -- by insurance

| Insurance | n | ECE | Readmission rate |
|---|---:|---:|---:|
| commercial | 4,050 | 0.0064 | 0.195 |
| medicaid | 1,779 | 0.0059 | 0.191 |
| medicare | 3,482 | 0.0079 | 0.198 |
| self_pay | 689 | 0.0200 | 0.212 |

## Notes

- Outcomes are sampled from `Bernoulli(p_calibrated)` where `p_calibrated` is the W1 calibrated bucket probability. The harness verifies that the *runtime* path (lookup + outcome sampling) does not drift in calibration vs the calibrated table. Subgroup ECE differences reflect random sampling noise + the LACE distribution's interaction with the subgroup marginal.
- Production validation against real EHR data is the Pillar-3 work scoped in `docs/PROSPECTIVE_STUDY_PROTOCOL.md`.
- Re-run with `--n 50000` for a tighter ECE estimate at the cost of ~ 5× wallclock.
