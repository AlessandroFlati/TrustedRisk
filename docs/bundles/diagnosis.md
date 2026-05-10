# Bundle: `diagnosis` -- Grounded Differential Diagnosis Ranker

Re-rank a deterministic differential-diagnosis table using an LLM,
preserving a "can't-miss" floor: high-acuity diagnoses (e.g. STEMI, PE,
aortic dissection) surface unconditionally regardless of LLM output. Each
ranked diagnosis carries a `grounding_verdict` so hallucinated diagnoses
surface as `ungrounded`.

## When to use

- ED triage where a clinician wants a structured DDx with grounded
  evidence per item.
- Inpatient cross-cover when a presenting complaint is ambiguous and
  the rule-based table alone is too narrow.

## Typical tool sequence

1. compute_admission_triage(patient_id, presenting_complaint) -> triage
   level + initial differential surface
1. compute_differential_diagnosis_ranker(presenting_complaint, candidates)
   -> re-ranked DDx with per-item evidence
1. ground_claim(diagnosis_label, patient_id) -> final-call grounding

## Determinism floor

- Rule-based table is the floor; the LLM only re-ranks (no new diagnoses).
- Can't-miss diagnoses are hard-coded per presenting complaint and
  surface regardless of LLM output.
- Each item has a `grounding_verdict` field (`grounded` / `ungrounded` /
  `partial`) -- clients can filter on it.

## Tools in this bundle

- `compute_differential_diagnosis_ranker` (LLM-4)
- `ground_claim`
- `compute_admission_triage`

## Minimum inputs

- A chief complaint (must map to a recognised key in the DDx table)
- Optional: structured features (vitals, labs, demographics)
- Optional: free-text summary

## Composed output

`DifferentialDiagnosisReport` with:
- Ranked `items[]` (DifferentialItem with diagnosis + probability + supporting features + grounding_verdict)
- `cant_miss_diagnoses`: list of diagnoses that surface unconditionally
- A2A INPUT_REQUIRED transition: `task_state = "input_required"` +
  `clarification_request` when the chief complaint is too vague,
  with the supported chief-complaint keys as candidates

## Demo scenarios

- v5 showcase: triage scenarios E (sepsis), J (mental-health crisis)
  use the DDx ranker as the first step
- v6 showcase: PA-imaging-mri-lumbar exercises the DDx grounding path

## Caveats

- The rule-based table is curated, not exhaustive -- rare diagnoses may
  not appear at all, and the LLM cannot add them.
- The grounding-verdict assumes the FAISS guideline corpus is loaded
  (`data/grounding_index_mpnet.pkl`).
