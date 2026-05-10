# Synthea 1000-patient validation cohort

**Status**: methodology + reproduction script ship in this repo. The actual
1000-patient run is a one-shot heavy job (Java + Synthea, ~10-15 minutes,
~70 MB jar download); the script that produces this report on demand is
`scripts/synthea_cohort_runner.py`.

**Generated**: see `cohort_label` field in the report block below the first
time you run the script.
**Companion**: `docs/MODEL_CARD.md` (model-level claims) +
`docs/validation/VALIDATION.md` (MIMIC-IV demo cohort).

---

## 1. Why Synthea?

Synthea (Walonoski JAMIA 2018) is a GPL-licensed, free, fully-synthetic FHIR
R4 generator. We use it as a second external-validation cohort because:

- It produces FHIR R4 bundles with realistic encounter sequences ->
  drop-in input for `compute_readmission_risk` without any preprocessing.
- It carries no PHI (it's synthetic by construction) -> no IRB / DUA needed.
- It's reproducible by seed (`-s 42`) -> byte-identical FHIR every time.
- It generates 1000 patients in ~10-15 minutes on a 4 GB JVM.

These properties make it the ideal second validation surface alongside the
MIMIC-IV demo (which is real but small, n=100).

## 2. Methodology

The runner:
1. Downloads `synthea-with-dependencies.jar` v3.3.1 from GitHub releases
   (checksum-trusted via HTTPS; cached locally at `data/synthea_runs/`).
2. Generates 1000 synthetic FHIR Bundles via
   `java -Xmx4g -jar synthea.jar -p 1000 -s 42 --exporter.fhir.export true`.
3. For each Bundle:
   - Extract LACE features via the production `_compute_lace_components`
     helper from `src/mcp_server/tools/readmission_risk.py`.
   - Look up the calibrated posterior from `data/coefficients.json`
     (`spec_002`, ECE 0.0078 internal).
   - Derive a 30-day readmission label using the CMS-style definition:
     1 if any inpatient/emergency encounter starts within 30 days of an
     earlier inpatient encounter's discharge; 0 otherwise; None when no
     inpatient encounter is found.
4. Aggregate predictions + labels -> ECE / AUROC / Brier + 5-bin
   calibration table.

## 3. Reproduction

```bash
PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \
    .venv/Scripts/python.exe scripts/synthea_cohort_runner.py \
    --n 1000 --seed 42 \
    --output-dir data/synthea_runs/run_1k \
    --report docs/validation/SYNTHEA_1000.md
```

Re-runs are idempotent in spirit: pass `--skip-generate` to skip the JAR
call and re-evaluate against an already-generated FHIR directory. The
script writes back into this exact file.

## 4. Expected Behavior

The Synthea LACE distribution skews lower than MIMIC-IV (Synthea's
synthetic patients have shorter average LOS and fewer comorbidities), so
expectations are:

| Metric | MIMIC-IV demo (real) | Synthea 1000 (synthetic, expected) |
|---|---|---|
| ECE (10-bin) | 0.082 | 0.06–0.12 |
| AUROC | 0.555 | 0.52–0.58 (LACE has known modest discrimination) |
| Base rate readmission | 0.18 | 0.05–0.15 (Synthea undersamples high-acuity) |

A Synthea ECE > 0.20 or AUROC < 0.50 would indicate either (a) a Synthea
encounter-class drift since v3.3.1, or (b) a calibration regression in our
shipped coefficients. Both are addressable: re-run the calibration on
Synthea-style data to refit `coefficients.json`.

## 5. Limitations

- Synthea models are clinically reasonable but not epidemiologically
  identical to a real institutional cohort. Calibration on Synthea is
  necessary but not sufficient evidence for clinical deployment.
- The Synthea encounter-class taxonomy uses HL7 v3 ActCode (`IMP`, `EMER`)
  which may not match a particular EHR's local Encounter coding. The
  `_is_inpatient` heuristic in `synthea_evaluator.py` covers both ActCode
  and SNOMED inpatient codes -- if your EHR uses something else, extend
  that helper.
- The LACE feature extractor inherits the same limitations the production
  model has on Synthea: synthetic ED-visit history is sparser than real
  Medicare data, so the E component skews low.

## 6. Result block (auto-rewritten by the runner)

The first run of the script overwrites the section below with the actual
ECE / AUROC / Brier / calibration-table for the n=1000, seed=42 cohort.
Until then, this section is a placeholder.

```text
[awaiting first scripts/synthea_cohort_runner.py run]
```

## References

- Walonoski J, et al. *Synthea: An approach, method, and software mechanism
  for generating synthetic patients and the synthetic electronic health
  care record.* JAMIA 2018;25:230.
- Walraven C, et al. *LACE index.* CMAJ 2010;182:551.
- Centers for Medicare & Medicaid Services. *Hospital Readmissions
  Reduction Program (HRRP).* (CMS-style 30-day readmission definition.)
