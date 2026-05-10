# TrustedRisk Performance Benchmarks v10

**Phase 11.7 -- captured 2026-04-30 06:47 UTC**

All measurements run in-process against the same ASGI factories used by `make demo-up-full`. No subprocess, no live network. `TRUSTEDRISK_DISABLE_LLM=1` enforces the deterministic floor.

## Layer 1 -- Phase 10/11 per-tool latency (n=200 each)

| Label | n | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) | errors |
|---|---|---|---|---|---|---|
| compute_quality_measures_aggregate | 200 | 0.01 | 0.01 | 0.01 | 0.01 | 0 |
| compute_stars_rating_forecast | 200 | 0.01 | 0.01 | 0.02 | 0.05 | 0 |
| compute_care_gap_priority_ranking | 200 | 0.01 | 0.01 | 0.02 | 0.17 | 0 |
| compute_syndromic_surveillance | 200 | 0.00 | 0.00 | 0.01 | 0.03 | 0 |
| compute_vaccine_reminder_cohort | 200 | 0.00 | 0.00 | 0.01 | 0.03 | 0 |
| compute_outbreak_heatmap | 200 | 0.01 | 0.01 | 0.01 | 0.03 | 0 |
| compute_denial_letter_parse | 200 | 0.01 | 0.02 | 0.03 | 0.25 | 0 |
| compute_appeal_escalation_path | 200 | 0.00 | 0.00 | 0.00 | 0.02 | 0 |
| compute_ecg_qt_analyzer | 200 | 0.01 | 0.01 | 0.01 | 0.04 | 0 |
| compute_dicom_sr_ingest | 200 | 0.00 | 0.00 | 0.01 | 0.03 | 0 |
| plan_tool_use | 200 | 0.04 | 0.04 | 0.07 | 0.25 | 0 |

## Layer 2 -- Clinical-scenario latency (n=50 each)

| Label | n | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) | errors |
|---|---|---|---|---|---|---|
| scenario:acute_stroke_lvo | 50 | 0.04 | 0.05 | 0.53 | 0.53 | 0 |
| scenario:sepsis_bundle | 50 | 0.04 | 0.07 | 0.22 | 0.22 | 0 |
| scenario:polytrauma_mtp | 50 | 0.03 | 0.04 | 0.10 | 0.10 | 0 |
| scenario:geriatric_polypharmacy | 50 | 0.17 | 0.22 | 0.37 | 0.37 | 0 |
| scenario:mental_health_crisis | 50 | 0.07 | 0.09 | 0.26 | 0.26 | 0 |

## Layer 3 -- Federation concurrency (100 concurrent /healthz)

| Label | n | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) | errors |
|---|---|---|---|---|---|---|
| federation_healthz_concurrent_100 | 100 | 34.72 | 47.96 | 49.89 | 50.41 | 0 |

## Notes

- p99 ≪ 100 ms confirms the deterministic floor stays in the interactive-latency band.
- Scenario-level p99 includes the chain of 5-7 tool calls, so it tracks tool-fan-out cost.
- Federation concurrency uses the Starlette TestClient -- there is no real socket layer here, so wall-clock numbers reflect ASGI dispatch + tool dispatch only. The next-generation benchmark (out of scope for this submission) would re-run against an actual uvicorn process to also measure socket / loopback overhead.
