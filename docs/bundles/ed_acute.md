# Bundle: `ed_acute` -- Emergency Department + Acute Deterioration

ED triage and inpatient deterioration. Maps chief complaint + vitals to ESI level + disposition; computes NEWS2 with trend delta for ward-based deterioration.

## When to use

- ED arrival: triage chief complaint + vitals -> ESI 1-5 + disposition.
- Ward deterioration: NEWS2 score with trend vs prior shift.
- Sepsis early warning (NEWS2 + lactate trend).

## Typical tool sequence

1. detect_phi(triage_note) -- scrub PHI before downstream processing
1. compute_admission_triage(chief_complaint, vitals, age) -> ESI + disposition
1. compute_clinical_deterioration_score(vitals, prior_score) -- NEWS2 + trend
1. compute_lab_trend_analysis(observations) -- lactate / WBC / Cr trends
1. ground_claim(claim_text, patient_id) -- guideline citation

## Minimum inputs

- Chief complaint (free text)
- Vital signs: HR, RR, SpO2, BP, T, AVPU/GCS
- Patient age

## Composed output

AdmissionTriage + DeteriorationReport, both with deterministic safety gate (resuscitation_room / admit_inpatient / etc.).

## Demo scenarios

- E (ED chest pain ESI 1)
- F (sepsis early warning ward)

## Tools in this bundle

- `compute_admission_triage`
- `compute_clinical_deterioration_score`
- `detect_phi`
- `ground_claim`
- `compute_lab_trend_analysis`

---

_Bundle id_: `ed_acute` &middot; _Tools_: 5 &middot; _Auto-generated from `mcp_server.tools.BUNDLES`_.
