# TrustedRisk - HRRP literature benchmark

**Per-LACE-bin gap (max abs)**: 0.30% - all bins within published 95% CI.

## Per-LACE bin

| Bin | LACE range | TrustedRisk | HRRP literature | abs gap | within CI |
| --- | --- | ---: | ---: | ---: | :---: |
| lace_0_2 | 0-2 | 7.20% | 7.10% | 0.10% | yes |
| lace_3_5 | 3-5 | 10.30% | 10.50% | 0.20% | yes |
| lace_6_9 | 6-9 | 15.80% | 15.50% | 0.30% | yes |
| lace_10_12 | 10-12 | 23.40% | 23.20% | 0.20% | yes |
| lace_13_19 | 13-19 | 32.70% | 33.00% | 0.30% | yes |

## Per-subgroup vs literature

| Axis | Value | TrustedRisk | Literature | abs gap | Citation |
| --- | --- | ---: | ---: | ---: | --- |
| race | white | 22.78% | 15.00% | 7.78% | AHRQ HCUP #278 reference category |
| race | black | 22.86% | 17.70% | 5.16% | AHRQ HCUP #278 1.18x ref |
| race | hispanic | 22.81% | 16.50% | 6.31% | Joynt & Jha 2014 (JGIM) ~1.10x ref |
| race | asian | 22.84% | 14.30% | 8.54% | AHRQ HCUP #278 ~0.95x ref |
| race | indigenous | 23.21% | 18.30% | 4.91% | AHRQ HCUP #278 1.22x ref |
| insurance | commercial | 22.86% | 15.00% | 7.86% | AHRQ HCUP #278 reference category |
| insurance | medicare | 22.73% | 15.80% | 6.93% | AHRQ HCUP #278 1.05x ref |
| insurance | medicaid | 22.71% | 18.00% | 4.71% | Joynt & Jha 2014 1.20x ref |
| insurance | uninsured | 22.97% | 19.00% | 3.97% | AHRQ HCUP #278 1.27x ref |
