# TrustedRisk - Synthetic Prospective Evaluation

**Generated**: 2026-05-11 17:12 UTC - **Cohort n**: 2,000 - **Seed**: 20260430 - **Fairness audit present**: True

Counterfactual 'if we had deployed' analysis: every synthetic encounter is run through the calibrated readmission risk model -> 4-critic ensemble (deterministic-floor approximation) -> 3-agent debate. The cohort + per-encounter demographics are sampled from the same distribution the subgroup-audit builder uses (Phase 15.B1).

## 1. Overall pipeline behaviour

| Metric | Value |
| --- | ---: |
| Cohort n | 2,000 |
| Approve rate | 35.90% |
| Downgrade rate (4-critic / debate revise) | 64.10% |
| Abstain rate (fairness-guard / safety floor) | 0.00% |
| Mean predicted readmission risk | 19.46% |
| Observed outcome rate | 22.50% |
| Absolute calibration gap | 3.04% |
| Fairness-flagged share of cohort | 35.75% |

**Action distribution (TrustedRisk pipeline)**: {'discharge_with_homecare': 1135, 'continued_admission': 855, 'discharge_home': 10}

## 2. Counterfactual: LACE-threshold baseline

Without the multi-agent debate / fairness-guard layer the system would deploy the LACE-threshold proposed action on every encounter.

| Metric | Pipeline | LACE-only baseline | Delta |
| --- | ---: | ---: | ---: |
| Approve rate | 35.90% | 100.00% | -64.10% |
| Abstain rate | 0.00% | 0.00% | 0.00% |
| Downgrade rate | 64.10% | 0.00% | 64.10% |

**Baseline action distribution**: {'discharge_with_homecare': 1128, 'continued_admission': 855, 'discharge_home': 17}

## 3. Per-subgroup metrics

### age_band

| Value | n | abstain | downgrade | predicted | observed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 18-39 | 182 | 0.00% | 68.13% | 20.30% | 25.82% |
| 40-64 | 707 | 0.00% | 66.90% | 19.78% | 20.65% |
| 65-74 | 504 | 0.00% | 61.31% | 18.84% | 19.64% |
| 75-84 | 413 | 0.00% | 61.74% | 19.41% | 25.67% |
| 85+ | 194 | 0.00% | 62.37% | 19.29% | 26.80% |

### sex

| Value | n | abstain | downgrade | predicted | observed |
| --- | ---: | ---: | ---: | ---: | ---: |
| female | 1,037 | 0.00% | 64.51% | 19.51% | 21.89% |
| male | 963 | 0.00% | 63.66% | 19.42% | 23.16% |

### race

| Value | n | abstain | downgrade | predicted | observed |
| --- | ---: | ---: | ---: | ---: | ---: |
| asian | 108 | 0.00% | 56.48% | 19.30% | 17.59% |
| black | 258 | 0.00% | 100.00% | 19.11% | 25.97% |
| hispanic | 347 | 0.00% | 56.20% | 19.18% | 24.21% |
| indigenous | 16 | 0.00% | 100.00% | 20.18% | 25.00% |
| other | 43 | 0.00% | 55.81% | 18.88% | 18.60% |
| white | 1,228 | 0.00% | 59.28% | 19.64% | 21.82% |

### ethnicity

| Value | n | abstain | downgrade | predicted | observed |
| --- | ---: | ---: | ---: | ---: | ---: |
| hispanic | 348 | 0.00% | 65.80% | 19.25% | 24.43% |
| non_hispanic | 1,652 | 0.00% | 63.74% | 19.51% | 22.09% |

### insurance_type

| Value | n | abstain | downgrade | predicted | observed |
| --- | ---: | ---: | ---: | ---: | ---: |
| commercial | 786 | 0.00% | 52.54% | 19.61% | 18.19% |
| medicaid | 342 | 0.00% | 100.00% | 19.09% | 26.61% |
| medicare | 621 | 0.00% | 50.89% | 19.52% | 22.06% |
| tricare | 78 | 0.00% | 48.72% | 19.11% | 32.05% |
| uninsured | 173 | 0.00% | 100.00% | 19.46% | 31.21% |

### language

| Value | n | abstain | downgrade | predicted | observed |
| --- | ---: | ---: | ---: | ---: | ---: |
| arabic | 50 | 0.00% | 76.00% | 19.47% | 34.00% |
| chinese | 110 | 0.00% | 68.18% | 20.46% | 23.64% |
| english | 1,548 | 0.00% | 63.89% | 19.41% | 21.51% |
| spanish | 253 | 0.00% | 62.45% | 19.58% | 23.32% |
| vietnamese | 39 | 0.00% | 56.41% | 18.05% | 38.46% |

## 4. References

- TrustedRisk Phase 15.C - synthetic prospective evaluation.
- Multi-agent debate: a2a_agent.multi_agent_debate (Phase 14.16 Q1).
- Calibration: W1 spec_002 5-bin Beta-Binomial, AUROC 0.590, ECE 0.0078.
