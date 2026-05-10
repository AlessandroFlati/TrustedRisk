# TrustedRisk -- Subgroup Fairness Audit

**Generated**: 2026-05-10 10:35 UTC - **Cohort n**: 100,000 - **DP**: epsilon=1.0

Per-subgroup calibration + Equality-of-Opportunity + Demographic-Parity gaps. Reference categories per subgroup are marked with `*`. DP-noised counts are published separately for external release.

## age_band (reference = 40-64)

| Value | n | prevalence | mean p_hat | TPR@0.20 | EOO gap | DP gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 40-64* | 35,108 | 20.80% | 0.2280 | 56.83% | +0.00% | +0.00% |
| 65-74 | 24,864 | 22.95% | 0.2276 | 57.00% | +0.17% | -0.25% |
| 18-39 | 9,941 | 19.35% | 0.2287 | 56.39% | -0.43% | +0.51% |
| 75-84 | 20,053 | 25.18% | 0.2283 | 57.12% | +0.29% | +0.48% |
| 85+ | 10,034 | 28.21% | 0.2281 | 57.97% | +1.14% | +0.15% |

## sex (reference = male)

| Value | n | prevalence | mean p_hat | TPR@0.20 | EOO gap | DP gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| male* | 49,198 | 23.19% | 0.2276 | 57.34% | +0.00% | +0.00% |
| female | 50,802 | 22.45% | 0.2284 | 56.74% | -0.59% | +0.26% |

## race (reference = white)

| Value | n | prevalence | mean p_hat | TPR@0.20 | EOO gap | DP gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| white* | 60,103 | 21.64% | 0.2278 | 56.93% | +0.00% | +0.00% |
| other | 1,949 | 22.11% | 0.2282 | 56.61% | -0.32% | -0.68% |
| hispanic | 18,015 | 24.59% | 0.2281 | 56.00% | -0.93% | +0.16% |
| black | 12,952 | 25.99% | 0.2286 | 58.38% | +1.45% | +0.66% |
| asian | 5,991 | 21.48% | 0.2284 | 56.72% | -0.21% | +0.77% |
| indigenous | 990 | 29.80% | 0.2321 | 64.07% | +7.14% | +1.85% |

## ethnicity (reference = non_hispanic)

| Value | n | prevalence | mean p_hat | TPR@0.20 | EOO gap | DP gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| non_hispanic* | 81,915 | 22.45% | 0.2278 | 57.02% | +0.00% | +0.00% |
| hispanic | 18,085 | 24.46% | 0.2288 | 57.13% | +0.12% | +0.55% |

## insurance_type (reference = commercial)

| Value | n | prevalence | mean p_hat | TPR@0.20 | EOO gap | DP gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| medicare | 29,843 | 22.09% | 0.2273 | 57.17% | -0.06% | -0.73% |
| commercial* | 40,010 | 21.07% | 0.2286 | 57.22% | +0.00% | +0.00% |
| medicaid | 18,152 | 25.70% | 0.2271 | 55.80% | -1.43% | -1.08% |
| uninsured | 8,044 | 27.95% | 0.2297 | 57.78% | +0.56% | +0.52% |
| tricare | 3,951 | 22.27% | 0.2278 | 58.98% | +1.75% | -0.32% |

## language (reference = english)

| Value | n | prevalence | mean p_hat | TPR@0.20 | EOO gap | DP gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| english* | 77,958 | 22.58% | 0.2279 | 57.00% | +0.00% | +0.00% |
| spanish | 13,052 | 24.01% | 0.2282 | 56.22% | -0.78% | +0.08% |
| chinese | 4,946 | 23.31% | 0.2284 | 58.63% | +1.63% | +0.18% |
| vietnamese | 2,070 | 22.42% | 0.2301 | 59.91% | +2.92% | +1.63% |
| arabic | 1,974 | 23.15% | 0.2292 | 57.33% | +0.33% | -0.57% |
