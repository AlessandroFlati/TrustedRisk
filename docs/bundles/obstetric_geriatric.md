# Bundle: `obstetric_geriatric` -- Obstetric + Geriatric

Maternal early warning (MEOWS with pregnancy-adjusted vitals) + ACOG preeclampsia classification; geriatric falls risk (Morse) + delirium screening (CAM with Beers deliriogenic-medication flagging).

## When to use

- Labor & delivery triage: MEOWS + preeclampsia classification.
- Inpatient geriatric: delirium screening + falls precautions.
- Outpatient med review with falls / delirium concerns.

## Typical tool sequence

1. compute_maternal_early_warning(GA_weeks, vitals, obstetric_red_flags) -- MEOWS
1. compute_preeclampsia_assessment(GA, BP, proteinuria, factors) -- ACOG 2020
1. compute_falls_risk_morse(...) -- Morse + Beers medication flagging
1. compute_delirium_screening_cam(...) -- CAM + DELIRIUM mnemonic
1. compute_fairness_audit(risk, demographics) -- maternal mortality bias

## Minimum inputs

- Maternal: gestational age, BP, vitals, severe-feature flags
- Geriatric: falls history, ambulatory aid, gait, mental status, current medications

## Composed output

MEOWSReport + PreeclampsiaReport (classification + delivery + Mg + antihtn); MorseFallsReport + CAMReport.

## Demo scenarios

- O (32-week severe preeclampsia)
- P (85y delirium + falls)

## Tools in this bundle

- `compute_maternal_early_warning`
- `compute_preeclampsia_assessment`
- `compute_falls_risk_morse`
- `compute_delirium_screening_cam`
- `compute_fairness_audit`

---

_Bundle id_: `obstetric_geriatric` &middot; _Tools_: 5 &middot; _Auto-generated from `mcp_server.tools.BUNDLES`_.
