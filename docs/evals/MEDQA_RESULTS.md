# MedQA-USMLE-style Eval Results

**Phase 11.8 -- captured 2026-04-30 06:50 UTC**

- Bench size: **30** items (synthetic clone -- see harness module for license note).
- Floor accuracy: **96.7%** (29 / 30).
- LLM accuracy: skipped (no LLM client configured).

## Per-category accuracy (floor)

| Category | Floor accuracy |
|---|---|
| ECG | 1.000 |
| anticoagulation | 0.500 |
| antimicrobial | 1.000 |
| appeals | 1.000 |
| cardiology | 1.000 |
| diabetes | 1.000 |
| emergency | 1.000 |
| endocrine | 1.000 |
| geriatric | 1.000 |
| hematology | 1.000 |
| imaging | 1.000 |
| informatics | 1.000 |
| nephrology | 1.000 |
| obstetric | 1.000 |
| pediatric | 1.000 |
| pharmacogenomics | 1.000 |
| psychiatric | 1.000 |
| quality | 1.000 |
| research_methods | 1.000 |
| statistics | 1.000 |
| stroke | 1.000 |
| trauma | 1.000 |

## Failures

| ID | Category | Correct | Floor pick | Floor ✓? | LLM pick | LLM ✓? |
|---|---|---|---|---|---|---|
| Q01 | anticoagulation | B | A | False | - | None |

## Notes

- The bench is a 30-item **synthetic clone** of the MedQA-USMLE style; it is not the official Jin et al. corpus (license-restricted). The clone exercises the same reasoning surface and is what we publish in this submission. Re-run with a licensed MedQA shard by passing it to `run_medqa(bench=<items>)`.
- The deterministic floor uses keyword-routed cues per category. It is intentionally simple -- meant to provide a reproducible lower bound, not to chase state of the art.
- Set `ANTHROPIC_API_KEY` (or any of the other supported polish providers) and re-run with `use_llm=True` to get the LLM row. The reply parser accepts the first A-D letter found in the LLM output.

## References

- Jin D et al. What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams (MedQA-USMLE). arXiv:2009.13081 (2020).
- USMLE Content Outlines, NBME 2024.
