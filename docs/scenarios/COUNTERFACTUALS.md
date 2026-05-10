# Counterfactual Explanations -- Per-scenario right-to-explanation surface

**Phase 12.5 -- captured 2026-04-30 07:31 UTC**

For each of the 5 end-to-end clinical scenarios, the harness perturbs 2-4 key input factors and detects the **minimum-modification** path that changes the recommended decision tier. The table below is regenerated on every CI build by `scripts/generate_scenario_counterfactuals_doc.py`.

**Aggregate:** 8 flip(s) across 15 perturbation(s) over 5 scenarios.

## Acute stroke + LVO + reperfusion eligibility (`acute_stroke_lvo`)

- Baseline outcome: **iv_tpa_only**
- Perturbations evaluated: 3 -- flips found: **2**

| Factor | Original | Modified | Original outcome | Modified outcome | Distance |
|---|---|---|---|---|---|
| last_known_well_minutes_ago | 110 | 400 | iv_tpa_only | **abstain_clinician_decision** | 290 |
| on_anticoagulant | False | True | iv_tpa_only | **abstain_clinician_decision** | 1 |

_Rationale_: Baseline decision = iv_tpa_only. Evaluated 3 perturbation(s); 2 flip(s) found. The decision is sensitive to time-since-LKW, anticoagulation status, and LVO confirmation.

**References:**
- Powers WJ et al. AHA/ASA Stroke Guidelines 2019.

## Sepsis bundle + antibiogram empiric pick (`sepsis_bundle`)

- Baseline outcome: **high**
- Perturbations evaluated: 3 -- flips found: **1**

| Factor | Original | Modified | Original outcome | Modified outcome | Distance |
|---|---|---|---|---|---|
| all_vitals_normalised | RR 26, SpO2 91 on O2, T 38.4, SBP 96, HR 122, AVPU=V | RR 14, SpO2 97 RA, T 37, SBP 130, HR 80, AVPU=A | high | **low** | 12 |

_Rationale_: Baseline NEWS2 tier = high (score 16). Evaluated 3 vital-sign perturbation(s); 1 produced a tier change.

**References:**
- NEWS2 -- Royal College of Physicians (2017).

## Polytrauma + MTP activation (`polytrauma_mtp`)

- Baseline outcome: **MTP=True,TXA=True**
- Perturbations evaluated: 3 -- flips found: **1**

| Factor | Original | Modified | Original outcome | Modified outcome | Distance |
|---|---|---|---|---|---|
| minutes_since_injury | 20 | 200 | TXA=True | **TXA=False** | 180 |

_Rationale_: Baseline MTP activate = True, TXA = True. Evaluated 3 perturbation(s); 1 produced a flip.

**References:**
- Holcomb JB et al. PROPPR Trial. JAMA 2015;313(5):471-482.
- CRASH-2 Trial -- TXA in trauma. Lancet 2010.

## Geriatric polypharmacy med review (`geriatric_polypharmacy`)

- Baseline outcome: **high**
- Perturbations evaluated: 3 -- flips found: **1**

| Factor | Original | Modified | Original outcome | Modified outcome | Distance |
|---|---|---|---|---|---|
| composite_risk_factors | all factors present | ambulatory + no falls history | high | **low** | 4 |

_Rationale_: Baseline Morse tier = high (score 65). 1 of 3 perturbations flipped the tier. Deprescribing deliriogenic meds is the highest-leverage intervention.

**References:**
- Morse JM. Preventing Patient Falls. 2nd ed. Springer (2009).
- Beers Criteria -- AGS 2023 Update.

## Mental-health crisis + admission decision (`mental_health_crisis`)

- Baseline outcome: **involuntary_hold_evaluation**
- Perturbations evaluated: 3 -- flips found: **3**

| Factor | Original | Modified | Original outcome | Modified outcome | Distance |
|---|---|---|---|---|---|
| voluntary_capable | False | True | involuntary_hold_evaluation | **voluntary_inpatient** | 1 |
| danger_to_self | True | False | involuntary_hold_evaluation | **crisis_stabilization_unit** | 1 |
| state_jurisdiction | CA (5150) | NY (MHL 9.39) | California 5150 hold (72h, Welf. & Inst. Code §5150) -- prong(s) identified: danger to self. | **NY MHL §9.39 emergency admission (15-day initial) -- prong(s) identified: danger to self.** | 1 |

_Rationale_: Baseline disposition = involuntary_hold_evaluation. Evaluated 3 perturbation(s); 3 flip(s) found. The 3-prong test (danger-to-self, danger-to-others, grave-disability) drives disposition; state jurisdiction modulates the legal-basis text.

**References:**
- California Welfare and Institutions Code § 5150.
- Posner K et al. C-SSRS validation. Am J Psychiatry 2011;168(12):1266-1277.

## Notes

- Single-factor perturbations are scored against the original deterministic-floor output. A flip means the perturbed run produced a different `decision` / `disposition` / `severity_tier`.
- These reports are part of the GDPR Art. 22 / EU AI Act Art. 13 right-to-explanation surface -- every emitted DecisionCard can be paired with the corresponding scenario counterfactuals on demand.
