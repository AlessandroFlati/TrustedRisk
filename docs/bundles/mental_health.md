# Bundle: `mental_health` -- Mental Health Crisis

Suicide risk + psychiatric admission with DEMOGRAPHIC BIAS GUARD that abstains for groups (Black, Indigenous, LGBTQ+, low-SES) where the literature documents under-prediction.

## When to use

- ED psychiatric consult: suicide ideation + behavior history -> risk band.
- Disposition: outpatient / crisis stabilization / voluntary / involuntary hold.
- Free-text mental-health note PHI scrub before billing/research.

## Typical tool sequence

1. compute_suicide_risk_assessment(ideation, behavior, factors, demographics) -- C-SSRS-derived
1. compute_psychiatric_admission_decision(risk_level, danger_to_*, capacity) -- 3-prong test
1. compute_fairness_audit(risk, demographics) -- minoritized-group bias guard
1. detect_phi(text) -- before any external release

## Minimum inputs

- C-SSRS ideation + behavior (5-level + lifetime/30d)
- Danger-to-self / danger-to-others / grave-disability flags
- Voluntary capacity assessment
- Patient demographics (mandatory for bias guard)

## Composed output

SuicideRiskAssessment + PsychiatricAdmission with state-specific legal-basis text (CO M-1 / CA 5150 / NY MHL §9.39 / FL Baker Act / IL).

## Demo scenarios

- J (28y M imminent risk + bias guard)

## Tools in this bundle

- `compute_suicide_risk_assessment`
- `compute_psychiatric_admission_decision`
- `compute_fairness_audit`
- `detect_phi`

---

_Bundle id_: `mental_health` &middot; _Tools_: 4 &middot; _Auto-generated from `mcp_server.tools.BUNDLES`_.
