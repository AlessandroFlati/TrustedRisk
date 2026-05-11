# TrustedRisk - Federated Learning report

**Generated**: 2026-05-11 17:12 UTC - **Sites**: 5 - **Rounds**: 10 - **Centralized baseline ECE**: 0.0020

Simulates McMahan 2017 FedAvg over five biased hospital sites with no raw-data sharing. Each round broadcasts only the Beta-Binomial parameters (alpha_post, beta_post per LACE bin).

## 1. Per-round convergence (no DP)

| round | global ECE | site_urban | site_rural | site_safety_net | site_geri | site_pedi |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0091 | 0.0220 | 0.0283 | 0.0308 | 0.0444 | 0.0163 |
| 1 | 0.0048 | 0.0213 | 0.0305 | 0.0248 | 0.0385 | 0.0174 |
| 2 | 0.0033 | 0.0210 | 0.0312 | 0.0227 | 0.0364 | 0.0176 |
| 3 | 0.0025 | 0.0209 | 0.0316 | 0.0216 | 0.0354 | 0.0178 |
| 4 | 0.0020 | 0.0208 | 0.0318 | 0.0209 | 0.0347 | 0.0181 |
| 5 | 0.0017 | 0.0207 | 0.0319 | 0.0205 | 0.0343 | 0.0184 |
| 6 | 0.0015 | 0.0207 | 0.0320 | 0.0201 | 0.0340 | 0.0186 |
| 7 | 0.0013 | 0.0207 | 0.0321 | 0.0199 | 0.0337 | 0.0187 |
| 8 | 0.0011 | 0.0206 | 0.0321 | 0.0197 | 0.0336 | 0.0188 |
| 9 | 0.0010 | 0.0206 | 0.0322 | 0.0196 | 0.0334 | 0.0189 |

## 2. Local-only baseline ECE

| site | local ECE |
| --- | ---: |
| `site_geriatric_specialty` | 0.0106 |
| `site_pediatric_adjacent` | 0.0056 |
| `site_rural_community` | 0.0054 |
| `site_safety_net` | 0.0098 |
| `site_urban_academic` | 0.0090 |

## 3. Privacy-utility curve (DP-FedAvg)

| epsilon | final ECE | centralized | overhead |
| ---: | ---: | ---: | ---: |
| 0.1 | 0.0079 | 0.0020 | +0.0059 |
| 0.5 | 0.0018 | 0.0020 | -0.0002 |
| 1.0 | 0.0014 | 0.0020 | -0.0007 |
| 5.0 | 0.0011 | 0.0020 | -0.0009 |
| 10.0 | 0.0011 | 0.0020 | -0.0010 |

## References

- McMahan et al. 2017 - Communication-Efficient Learning of Deep Networks from Decentralized Data (AISTATS).
- Geyer et al. 2017 - Differentially Private Federated Learning.
- Dwork et al. 2006 - Calibrating Noise to Sensitivity.