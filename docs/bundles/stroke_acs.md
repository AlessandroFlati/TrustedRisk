# Bundle: `stroke_acs` -- Stroke + ACS

Acute stroke severity (NIHSS) + reperfusion eligibility (IV tPA + EVT, AHA/ASA 2019 + DAWN/DEFUSE 3); chest pain (HEART score + ACS pathway).

## When to use

- Acute stroke activation: NIHSS + tPA inclusion/exclusion + EVT eligibility.
- Late-window stroke (6-24h): DAWN/DEFUSE 3 imaging mismatch decision.
- ED chest pain: HEART score + disposition (discharge / observe / admit / cath).

## Typical tool sequence

1. compute_stroke_severity(item_scores, last_known_well_minutes_ago) -- NIHSS + LVO indicator
1. compute_stroke_thrombolysis_eligibility(...) -- tPA + EVT inclusion/exclusion
1. compute_heart_score(history, ECG, age, RFs, troponin) -- chest pain risk
1. compute_acs_disposition_decision(heart_score, STEMI, dynamic_trop, ...) -- disposition
1. ground_claim(claim_text) -- AHA/ASA citation

## Minimum inputs

- Stroke: 15 NIHSS items + last-known-well + imaging findings + labs (INR, plt, glucose, BP)
- ACS: history descriptor + ECG descriptor + age + risk-factor count + troponin × ULN + STEMI flag

## Composed output

NIHSSReport + ThrombolysisDecision (iv_tpa_only / evt_only / iv_tpa_plus_evt / no_reperfusion / abstain); HEARTScore + ACSDisposition.

## Demo scenarios

- M (acute stroke 70y M)
- N (chest pain HEART 4)

## Tools in this bundle

- `compute_stroke_severity`
- `compute_stroke_thrombolysis_eligibility`
- `compute_heart_score`
- `compute_acs_disposition_decision`
- `ground_claim`

---

_Bundle id_: `stroke_acs` &middot; _Tools_: 5 &middot; _Auto-generated from `mcp_server.tools.BUNDLES`_.
