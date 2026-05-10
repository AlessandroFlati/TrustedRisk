# TrustedRisk synthetic + open datasets

**Phase 9.2** -- datasheet-style cards (Gebru et al. *Datasheets for
Datasets*, FAccT 2018) for every cohort the runtime ships against.
Each dataset card answers the standard 7 questions: Motivation /
Composition / Collection / Preprocessing / Uses / Distribution /
Maintenance.

| Dataset | Size | Card |
|---|---:|---|
| 600-case bundle cohort | 600 (50 × 12 bundles) | [`bundle_cohort_600.md`](bundle_cohort_600.md) |
| Synthea-10k validation | 10 000 | [`synthea_10k.md`](synthea_10k.md) |
| MIMIC-IV demo 2.2 cross-fold | 100 | [`mimic_iv_demo.md`](mimic_iv_demo.md) |

These are the canonical datasets the test suite exercises (excluding
the operator's own production data, which never leaves the deployment).

## When to cite which dataset
- **Bundle cohort 600**: golden-case regression in `tests/golden/`.
- **Synthea-10k**: calibration + subgroup ECE (`docs/validation/SYNTHEA_10K.md`).
- **MIMIC-IV demo**: external Pillar-2 validation
  (`docs/validation/VALIDATION.md`).

## License notes
- Synthea cohorts: GPL-2.0 -- free for research + commercial.
- MIMIC-IV demo 2.2: PhysioNet Open License v1.0 (no DUA required).
- Bundle 600: MIT (the same license as TrustedRisk source).

## Refresh cadence

Each card is timestamped + versioned. Re-run the generator scripts
quarterly or whenever a dependent calibration shifts:

```bash
PYTHONPATH=src .venv/Scripts/python.exe \
    scripts/generate_synthetic_cohorts.py     # 600-case set
PYTHONPATH=src .venv/Scripts/python.exe \
    scripts/synthea_10k_validation.py         # 10k cohort + report
```
