# PO Demo Fixtures

FHIR R4 transaction Bundles ready for upload to a Prompt Opinion workspace.
Each Bundle drives a specific TrustedRisk Care Engine workflow that is
visually compelling for a 3-minute demo video.

All Patient resources use the `demo-` ID prefix so they are easy to filter
out of the workspace later. Resource IDs are human-readable
(`Condition/demo-eleanor-chf` rather than UUIDs) so the trace artefacts
are easier to read in the demo.

## Demo posture (one abstain, three positive)

The four bundles split deliberately along the abstain / positive axis so
the recorded demo can show both halves of the engine's safety contract:

- **Eleanor Greene (abstain demo)**: triggers calibrated abstain on the
  LACE OOD plateau (LACE >=12), missing admission/discharge med pair, and
  missing decision-card action. The engine refuses to substitute a number
  it cannot calibrate or fields it has no evidence for.
- **Marcus Reyes / Nadia Okafor / Sofia Ramirez (positive demos)**:
  every workflow input the dispatcher overlays from the bundle (vital
  signs, chief complaint, infection source, immunization history,
  allergies, insurance type) is present, so the corresponding tool chain
  runs end-to-end without abstaining and produces a calibrated artefact.

## Patients in this directory

| File | Patient | Posture | Workflow target | Prompt to use in PO chat |
|---|---|---|---|---|
| `eleanor_greene.json` | Eleanor Greene, 78F | abstain demo | `discharge_planning` | `Run TrustedRisk's full discharge bundle for this patient.` |
| `marcus_reyes.json` | Marcus Reyes, 64M | positive | `sepsis_workup` + `complete_sepsis_pipeline` | `Run TrustedRisk's complete sepsis pipeline on this patient.` |
| `nadia_okafor.json` | Nadia Okafor, 41F | positive | `denial_appeal_pipeline` + `prior_auth_pipeline` | `Use TrustedRisk to draft an appeal for the denied adalimumab on this patient.` |
| `sofia_ramirez.json` | Sofia Ramirez, 7F | positive | `well_child_visit` + `vaccine_schedule_check` | `Run TrustedRisk's well-child workflow and identify any overdue ACIP vaccines.` |

## What each Patient drives in the dispatch

### 1. Eleanor Greene (78F) -- abstain demo

**Profile**: 78yo female with CHF, AKI, T2DM, HTN, hyperlipidemia, paroxysmal AFib.
Recent IMP admission (5 days, 2026-04-22 to 2026-04-27), 4 ED visits in last 6 months.
On 5 chronic meds incl. warfarin + furosemide. Sulfa allergy.

**Why she abstains** (intentional, every gate is real):
- LACE ~14-16 lands in the OOD plateau band (>=12) where the shipped
  `coefficients.json` lookup degenerates to a constant prob_mean. The
  tool emits `abstain_recommended=True` with reason
  `lace_calibration_plateau_OOD` and downgrades `confidence` to
  `degraded` -- the numeric 28.2% is exposed for transparency but
  flagged unsafe to use.
- The bundle has `MedicationRequest` entries marked `active` only, with
  no separate `admission` vs `discharge` phase, so
  `compute_medication_reconciliation` abstains on
  `missing_admission_meds_and_discharge_meds`.
- No upstream `decision_card.recommendation.action` is supplied, so
  `compute_caregiver_handoff` abstains on `missing_decision_card_action`
  rather than fabricate a default disposition.

**Expected LACE** (for transparency, not for clinical use):
- L (LOS 5 days) -> 4 points
- A (IMP encounter) -> 3 points
- C (6 conditions) -> 5 points
- E (4 non-IMP ED visits) -> 4 points

**Demo highlights**:
- The `[ABSTAIN WARNING]` banner appears on the status_message ahead of
  any narrative; the chat-LLM cannot present 28.2% as a clean estimate.
- `_bundle_provenance` still shows the 24 FHIR resources consumed --
  abstain is a calibration verdict, not a fetch failure.

### 2. Marcus Reyes (64M) -- positive demo

**Profile**: 64yo Black male with recent UTI treated with cipro 2 weeks ago,
now ED with fever 39.4 C, HR 118, BP 86/50, RR 26, SpO2 91% on room air,
GCS 13, lactate 4.1, WBC 17.8, Cr 1.7, platelets 142k, bilirubin 1.4.
Suspected urosepsis with shock. Penicillin allergy (urticaria + angioedema 2018),
covered by Medicare Part B.

**Bundle additions for full positive run**:
- `Coverage/Medicare Part B` -> `insurance_type=medicare` for fairness
  audit subgroup.
- `AllergyIntolerance` Penicillin (high criticality) -> drives empiric
  antibiotic selection away from beta-lactams without prior PCN cross-
  reactivity testing.
- `Observation` GCS 13 -> qSOFA AMS criterion is met explicitly rather
  than inferred from hypotension.
- `Observation` Platelets 142k + Bilirubin 1.4 -> SOFA coag and liver
  domains have data instead of abstaining.
- `Condition` "Suspected urosepsis with septic shock" (SNOMED bacterial
  sepsis + ICD-10 A41.51) -> infection_source resolves to `urinary` for
  the empiric selection step.

**Expected workflow output** (sepsis_workup macro, 5 hops):
- `triage` -> ESI 2 (septic shock pattern + abnormal vitals).
- `deterioration` -> NEWS2 high tier, urgent escalation.
- `ddx` -> urosepsis at top, bacteremia + acute pyelonephritis below.
- `antibiotic` -> ceftriaxone or piperacillin-tazobactam (cipro recent
  failure pulls it off the list; PCN allergy pushes ceftriaxone to a
  desensitization plan or alt agent).
- `consult` -> ID consult letter chained on the antibiotic rationale.

**Demo highlights**:
- Macro `complete_sepsis_pipeline` (3-hop: sepsis_workup ->
  antibiotic_stewardship -> transitional_care_management) inflates to
  every inner step in the trace artefact.
- Real lactate 4.1 + low MAP triggers the Surviving-Sepsis hour-1
  bundle citation in the antibiotic step.

### 3. Nadia Okafor (41F) -- positive demo

**Profile**: 41yo Black female with seropositive RA on UnitedHealthcare
PPO. Adalimumab ordered but PA denied for step-therapy: payer requires
documented inadequate response to two conventional DMARDs first.
Documented failures of methotrexate (transaminitis ALT 3x ULN) and
sulfasalazine (maculopapular rash). Active disease (CDAI 18.4, ESR 42,
CRP 28.5).

**Bundle additions for full positive run**:
- `DocumentReference` for the UnitedHealthcare denial letter (full
  denial body in `attachment.data` base64, including claim id, deadline,
  contested charge, denial category) -> `compute_denial_letter_parse`
  extracts the payer, deadline, and step-therapy denial category.
- `DocumentReference` for the rheumatology progress note (full SOAP body
  base64, including CDAI score, MTX/SSZ failure dates and reasons, ACR
  guideline reference) -> `compute_pa_evidence_pack` chart-excerpts the
  med-failure timeline.

**Expected workflow output**:
- `denial_letter_parse` -> payer=UnitedHealthcare, denial_category=
  step_therapy_not_met, deadline 2026-11-02, contested $6,842/mo.
- `pa_evidence_pack` -> assembles MedicationRequest.statusReason for MTX
  (transaminitis) + SSZ (rash), plus CDAI/ESR/CRP active-disease
  observations, flags step-therapy criterion already met.
- `appeal_letter_draft` -> 8-section internal first-level appeal citing
  ACR 2021 conditional recommendation for TNF-i after csDMARD failure.

**Demo highlights**:
- Real payer interaction (UnitedHealthcare, step-therapy denial).
- The tool reads `MedicationRequest.statusReason.text` on the stopped
  meds, not just the `status` enum, to assemble a defensible argument.
- Bundle provenance shows 15 FHIR resources, including 2
  DocumentReferences with verifiable plaintext bodies.

### 4. Sofia Ramirez (7F) -- positive demo

**Profile**: 7yo Hispanic female (Spanish-preferred) on MassHealth
Medicaid, with mild asthma on albuterol PRN. Vital signs at well-child
visit: HR 92, RR 22, SpO2 99%, T 36.8 C (PEWS = 0, low tier). BMI 62nd
percentile (normal). Immunization history covers infant series (HepB x3,
IPV x3, Hib x3, PCV13 x3, MMR, DTaP, Varicella, 2024-25 flu, 2025-26
flu) -- all caught up except the routine school-age boosters that are
not yet age-due.

**Bundle additions for full positive run**:
- `Coverage/MassHealth Standard` -> `insurance_type=medicaid` for
  fairness audit subgroup.
- 4 vital-sign Observations (HR, RR, SpO2, T) at the well-child visit
  encounter -> `compute_pediatric_early_warning` runs to a low-tier
  PEWS score instead of abstaining on missing vitals.
- 14 additional Immunization records across HepB, IPV, Hib, PCV13,
  Varicella, and 2025-26 flu -> `compute_vaccine_schedule_check`
  evaluates against a complete history rather than the prior 3-record
  stub.

**Expected workflow output**:
- `well_child_visit` -> BMI 62nd percentile (healthy), height/weight
  tracked, asthma reviewed, low PEWS.
- `vaccine_schedule_check` -> flags MMR-2 as overdue (should have
  happened by age 6), queues HPV as upcoming at age 9, TDAP booster at
  age 11-12; all infant doses caught up.
- Patient-language counseling in **Spanish** (locale=es) when the
  discharge counseling tool detects the
  `Patient.communication[0].preferred=true` block.

**Demo highlights**:
- Pediatric workflow with ACIP schedule logic.
- Multi-language counseling triggered by `Patient.communication`.
- Contrast with adult workflow (no LACE, different parametric variant
  set; PEWS replaces NEWS2 with age-band-specific cutoffs).

## Upload to Prompt Opinion

PO workspace FHIR endpoint pattern:
```
https://app.promptopinion.ai/api/workspaces/<WS_ID>/fhir
```

Each Bundle is a transaction whose entries POST to the resource type endpoint
(e.g. `Patient`, `Encounter`) with `fullUrl: urn:uuid:<slug>`. The PO server
**does not expose `updateCreate`** so a PUT-with-id transaction returns
`not-supported`; we use POST + urn:uuid so the server assigns the resource ids
and we still bind intra-transaction references via the urn:uuid placeholders.
If you need to regenerate the bundles from a PUT-shaped source, run
`fixtures/po_demo/_convert_to_post.py` -- it rewrites in place. Example:

```bash
TOKEN=<paste from PO Generate Token dialog>
WS_BASE="https://app.promptopinion.ai/api/workspaces/<WS_ID>/fhir"
for f in eleanor_greene.json marcus_reyes.json nadia_okafor.json sofia_ramirez.json; do
  echo "=== uploading $f ==="
  curl -sS -X POST "$WS_BASE" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/fhir+json" \
    -H "Accept: application/fhir+json" \
    --data @"$f" | python -m json.tool | head -40
done
```

After upload, the four Patient resources appear in the PO workspace. Pin them
to the chat sidebar and run the prompts above.

## Notes on FHIR-context propagation

When PO chat dispatches via SendA2AMessage, it injects:

```
metadata["https://app.promptopinion.ai/schemas/a2a/v1/fhir-context"] = {
  "fhirUrl": "<workspace fhir base>",
  "fhirToken": "<short-lived JWT, 1h TTL>",
  "patientId": "<bare id, no Patient/ prefix>"
}
```

The TrustedRisk orchestrator unwraps this, binds `FHIRContext` to the SHARP
ContextVar, and every tool downstream that needs FHIR data reads from the
PO workspace using the same token. The dispatcher additionally calls
`_overlay_real_fhir` once per workflow which fetches the bundle and
overlays demo-derived inputs (admission_meds, discharge_meds, fhir_bundle,
patient_demographics) so tools without a SHARP fallback also see real data.

## What is *not* in these bundles

To keep the bundles auditable for the demo:
- **No PHI**: all names, addresses, phones, dates of birth are synthetic.
- **No `medicationReference`** to external `Medication/` resources: every
  med uses an inline `medicationCodeableConcept` with RxNorm coding so the
  display name is always available even on FHIR servers that scope out
  `patient/Medication.rs`.
- **No DocumentReference**: PO's pattern of stuffing meds into clinical
  notes' markdown body is not exercised here -- the structured
  MedicationRequest path is the only source of truth.
- **No DiagnosticReport / Coverage** for Eleanor / Marcus: kept minimal.
  `Coverage/demo-nadia-uhc` is included for Nadia because the appeals
  workflow reads payer name from it.
