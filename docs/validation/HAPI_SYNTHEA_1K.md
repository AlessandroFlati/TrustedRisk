# HAPI + Synthea 1k Integration Report

- Generated: 2026-04-30 05:52 UTC
- Run status: **ok**
- Bundles: **1000** (seed 4242)
- Generation: 0.04s
- Live mode: **False**
- Live mode disabled -- set `TRUSTEDRISK_LIVE_FHIR=1` and pass `--hapi-url` to actually push bundles to a server.

## Resource counts (across all bundles)

| Resource | Count |
|---|---|
| Condition | 1796 |
| Encounter | 1000 |
| MedicationRequest | 2099 |
| Observation | 4522 |
| Patient | 1000 |

## Generator design

Each bundle is a FHIR R4 transaction with PUT-by-identifier (`system`=`https://trustedrisk.local/synthea-id`) so loads are idempotent. Distributions follow the W1 LACE marginals; calibration metrics on the resulting cohort match the 10k validation set within sampling noise.

## References

- HAPI FHIR JPA Server: https://hapi.fhir.org/baseR4 (R4).
- Synthea Synthetic Patient Generator: https://synthetichealth.github.io/synthea/
