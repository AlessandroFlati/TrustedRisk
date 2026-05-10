# Conformal Prediction vs CI95 -- Readmission Risk

**Phase 11.1 -- 2026-04-30 06:13 UTC**

We extend the `compute_readmission_risk` tool with a **split-conformal** prediction surface that runs alongside the existing CI95 band. Two complementary tracks ship in `data/conformal_readmission.json`:

- **`abs_residual`** -> a regression-style interval `[p − q, p + q]` clipped to `[0, 1]`. Marginal coverage at the calibrated target on the validation fold.
- **`binary_one_minus_p`** -> a binary prediction set `{c ∈ {0, 1} : 1 − p_c ≤ q}`. An empty set is the abstention signal.

## Calibration

| Track | Score function | n_calibration | Quantile q | Validation coverage |
|---|---|---|---|---|
| Regression | `abs_residual` | 2,500 | 0.0812 | **0.888** (target 0.90) |
| Binary | `binary_one_minus_p` | 2,500 | 0.7660 | **0.916** (target 0.90) |

(Synthetic IID Beta-Binomial cohort matching the W1 5-bin marginals; 5,000 samples; 50/50 calibration/validation split. Reproduce with `scripts/calibrate_conformal_readmission.py`.)

## Comparison vs CI95

| Property | CI95 | Conformal interval |
|---|---|---|
| Coverage guarantee | Bayesian 95% credible band on the W1 posterior | Frequentist marginal 90% coverage under exchangeability |
| Distribution-free? | No (parametric Beta posterior) | **Yes** -- works against any predictive function |
| Model assumptions | Beta-Binomial conjugacy | Only IID/exchangeability of (x, y) |
| Failure mode | Mis-calibrated when posterior is mis-specified | Conservative when calibration set is small |
| Empirical width on W1 cohort | ~ 0.18 (depends on bin) | ~ 0.16 (q = 0.0812 -> ± 0.0812 around p) |

For the readmission decision, the two bands are **not redundant**: the CI95 surfaces uncertainty about the *parameter* (where the true probability lies given the W1 calibration cohort), while the conformal interval surfaces uncertainty about the *next observation* (a fresh patient's outcome). A clinician who needs decision-grade abstention thresholds should anchor on the conformal prediction set; a population-health analyst tracking calibration drift should anchor on CI95.

## Wiring

`compute_readmission_risk` lazy-loads `data/conformal_readmission.json` (override path: `TRUSTEDRISK_CONFORMAL_PATH`) and attaches four optional fields to the `RiskEstimate`:

- `conformal_interval_lower`
- `conformal_interval_upper`
- `conformal_prediction_set` (list, may be `[]` for abstain)
- `conformal_target_coverage`

When the artefact is absent or corrupt the four fields stay `None` and the runtime falls back gracefully to CI95-only output (no breaking change for existing consumers).

## References

- Vovk V, Gammerman A, Shafer G. *Algorithmic Learning in a Random World*. 2nd ed. Springer (2022).
- Angelopoulos AN, Bates S. *A Gentle Introduction to Conformal Prediction*. arXiv:2107.07511 (2022).
