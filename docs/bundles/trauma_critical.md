# Bundle: `trauma_critical` -- Trauma Critical Care

Composite trauma severity (ISS + T-RTS) + ABC-driven massive transfusion protocol (1:1:1 ratio + TXA window) + blunt-trauma imaging (FAST + CT).

## When to use

- Trauma team activation: rank severity, decide OR vs ICU vs ward.
- Hemorrhagic shock: ABC score -> MTP activation + initial product request.
- TXA decision (≤180 min from injury per CRASH-2).

## Typical tool sequence

1. compute_trauma_severity_score(injuries, GCS, SBP, RR) -- ISS + T-RTS + triage priority
1. compute_massive_transfusion_protocol(ABC items, EBL, minutes_since_injury) -- MTP activation + TXA
1. compute_imaging_appropriateness(blunt_abdominal_trauma, ...) -- FAST vs CT
1. compute_clinical_deterioration_score(vitals) -- NEWS2 for ICU disposition

## Minimum inputs

- Injury list with body region + AIS severity (1-6)
- GCS / SBP / RR for T-RTS
- Active hemorrhage flag
- ABC items: penetrating, SBP≤90, HR≥120, FAST positive
- Minutes since injury (TXA gate)

## Composed output

TraumaSeverityReport (ISS / RTS / triage_priority) + MTPDecision (ABC + activation + 1:1:1 plan + TXA flag).

## Demo scenarios

- Q (polytrauma cross-bundle)

## Tools in this bundle

- `compute_trauma_severity_score`
- `compute_massive_transfusion_protocol`
- `compute_imaging_appropriateness`

---

_Bundle id_: `trauma_critical` &middot; _Tools_: 3 &middot; _Auto-generated from `mcp_server.tools.BUNDLES`_.
