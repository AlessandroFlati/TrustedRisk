# Bundle: `context_resolution` -- Conversational SHARP Patient Resolution

Resolve a free-text user query (e.g. "the 78-year-old gentleman from yesterday with the chest pain")
into a SHARP-on-MCP patient context (`X-Patient-ID`) that downstream tools
can consume. LLM-assisted with a deterministic regex floor so the tool
never silently invents a match.

## When to use

- Conversational front-ends that pre-populate the SHARP context before
  invoking the agentic loop.
- Backfilling SHARP headers when an upstream agent omits `X-Patient-ID`
  but provides a free-text reference to a patient.

## Typical tool sequence

1. detect_phi(query_text) -> strip PHI before logging the query
1. compute_resolve_patient_from_query(query_text, candidate_bundle) ->
   ranked patient candidates with score + margin
1. ground_claim(query_text, patient_id) -> final-call grounding once a
   match is selected

## Determinism floor

- LLM is optional; the tool ships a deterministic regex extractor that
  matches MRN, name fragments, age, and date-of-birth tokens.
- Top match is set ONLY when score ≥ 0.6 AND margin ≥ 0.2 -- otherwise
  the tool abstains and returns ranked candidates without a top pick.
- Operators can disable the LLM entirely with `TRUSTEDRISK_DISABLE_LLM=1`.

## Tools in this bundle

- `compute_resolve_patient_from_query` (LLM-3)
- `detect_phi`
- `ground_claim`

## Minimum inputs

- A free-text query (e.g. "the older gentleman with chest pain from
  this morning")
- A pre-fetched candidate cohort (from SHARP context -- Patient + recent
  Encounters)
- Optional: time anchor (hours back from now)

## Composed output

`PatientResolutionResult` with:
- `candidates`: ranked list with `match_score` + matched_features
- `resolved_patient_id`: set ONLY when score ≥ 0.6 AND margin ≥ 0.2
- A2A INPUT_REQUIRED transition: `task_state = "input_required"` +
  `clarification_request` when the top match is ambiguous, so the BYO
  orchestrator can ask the user to pick from the candidate list

## Demo scenarios

- 5-turn COIN demo (v6 showcase, scenario PATIENT-multi-turn-qa)
  exercises this surface as the BYO orchestrator's first step.

## Caveats

- The candidate cohort must be pre-fetched via SHARP context -- the tool
  does not query the FHIR server itself.
- The LLM never invents new candidates; it only re-ranks the
  deterministic candidate list.
