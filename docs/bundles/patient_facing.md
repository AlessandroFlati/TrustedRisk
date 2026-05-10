# Bundle: `patient_facing` -- Patient + Caregiver Rendering

Patient-facing rendering of the discharge plan: counseling text,
multi-language translation, FAQs, and caregiver-summary surfaces. All
LLM uses are paraphrase-only -- the structured medical content (drug
names, dosages, monitoring, red flags) is never invented.

## When to use

- After a `core_discharge` flow has produced a `DecisionCard` and the
  client needs a patient-language artifact for printing or in-app
  display.
- Multilingual hand-off: same structured input rendered in 5 languages
  (en / es / zh / vi / ar).
- Caregiver summary for family / home health nurse hand-off.

## Typical tool sequence

1. compute_discharge_counseling(meds, lace_score, action) -> 5-section
   counseling (deterministic)
1. compute_translate_discharge_counseling(counseling, target_language)
   -> translated counseling (PATIENT-1)
1. compute_patient_faq(decision_card) -> top-N anticipated patient
   questions + answers (PATIENT-2)
1. compute_caregiver_summary(decision_card) -> caregiver-language summary
   (PATIENT-3)

## Determinism floor

- Discharge counseling is fully deterministic (5 sections, 13 drug
  classes mapped to patient-language label + purpose + monitoring +
  red flags).
- Translation uses LLM paraphrase but bullets are immutable -- content
  cannot drift between languages.
- FAQ + caregiver summary derive their facts from the structured
  `DecisionCard` payload, not from the LLM's own knowledge.

## Tools in this bundle

- `compute_discharge_counseling`
- `compute_translate_discharge_counseling` (PATIENT-1)
- `compute_patient_faq` (PATIENT-2)
- `compute_caregiver_summary` (PATIENT-3)

## Minimum inputs

- A `DecisionCard` (output of `core_discharge` flow)
- Target language (for translator)
- Optional: caregiver type / patient demographic adjustments

## Composed output

- `DischargeCounseling` (5 sections, deterministic)
- Translated counseling (camelCase per-language `body_translated_<lang>`)
- `PatientFAQ` (top-N anticipated questions + answers)
- `CaregiverSummary` (caregiver-language hand-off prose)

## Demo scenarios

- v5 showcase: scenario A (75y CHF discharge) exercises counseling
- v6 showcase: PATIENT-multi-turn-qa demonstrates the patient-facing
  layer composed with the patient-agent specialist

## Caveats

- Reading level targeted at 6th-grade Flesch-Kincaid for English.
  Other languages preserve the structure but reading-level audits per
  language are out of scope for this release.
- Patient ID is for audit only -- never appears in the output text.
  Tests enforce this.
