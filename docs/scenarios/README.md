# Clinical Scenarios -- End-to-End Walk-throughs

**Phase 11.6 -- 2026-04-30**

Each scenario is a deterministic, reproducible chain of MCP tool calls that solves a real clinical problem end-to-end. The scenarios are exercised by `tests/integration/test_clinical_scenarios.py` on every CI run, and you can step through them interactively in `docs/notebooks/04_clinical_scenarios.ipynb`.

## Scenario index

| ID | Title | Tools chained |
|---|---|---|
| `acute_stroke_lvo` | Acute stroke + LVO + reperfusion eligibility | NIHSS -> tPA/EVT eligibility -> ACR appropriateness -> contrast safety -> AKI staging -> PHI scrub |
| `sepsis_bundle` | Sepsis bundle + antibiogram empiric pick | NEWS2 -> empiric antibiotic (with local antibiogram) -> creatinine trend -> discharge counseling -> PHI scrub |
| `polytrauma_mtp` | Polytrauma + MTP activation | ISS / RTS -> ABC / MTP -> ACR appropriateness -> AKI staging -> PHI scrub |
| `geriatric_polypharmacy` | Geriatric polypharmacy med review | Polypharmacy DDI -> medication reconciliation -> Morse falls -> CAM delirium -> discharge counseling -> medication what-if -> PHI scrub |
| `mental_health_crisis` | Mental-health crisis + admission decision | C-SSRS -> psychiatric admission decision -> discharge counseling -> ES translation -> caregiver instruction set -> PHI scrub |

## Running

```python
import asyncio
from a2a_agent.clinical_scenarios import run_all_scenarios, run_scenario

# Single scenario
stroke = asyncio.run(run_scenario("acute_stroke_lvo"))
print(stroke.final_summary)

# All five
for r in asyncio.run(run_all_scenarios()):
    print(f"{r.scenario_id}: {r.n_tools_invoked} tools -- {r.final_summary}")
```

## Reference numbers (current build)

```
acute_stroke_lvo: 6 tools -- NIHSS 13 -> iv_tpa_only; ACR top modality; AKI stage no_aki.
sepsis_bundle: 5 tools -- NEWS2 (low); empiric pick honours local ESBL prevalence.
polytrauma_mtp: 5 tools -- ISS 16; MTP = True; AKI stage_1.
geriatric_polypharmacy: 7 tools -- 6->4 meds; Morse 65 (high); CAM negative.
mental_health_crisis: 6 tools -- C-SSRS high; disposition involuntary_hold_evaluation; ES counseling.
```

Cumulative: **29 tool calls across 5 scenarios** in the current build.

## Design rationale

- **Inline inputs.** Every scenario provides its own structured inputs so the runs are deterministic and reproducible without a live FHIR server. Production deployments would swap the inline data for FHIR Bundle extracts (`fetch_patient_bundle` under the SHARP context).
- **No LLM in the floor.** Each tool's deterministic floor is what runs by default. `enable_llm_polish=true` flags can be flipped on individual sections (the polished prose then routes through the audited Anthropic / Ollama / Gemini polisher with cite-back enforcement -- see `docs/WHITE_PAPER.md` §4).
- **Reproducibility.** Same inputs -> byte-identical outputs. The smoke test asserts on tool count + non-empty summaries and is run on every CI build.

## References

- Vohra S et al. CONSORT extension for N-of-1 trials. *BMJ* 2015;350:h1738.
- Powers WJ et al. *AHA/ASA Stroke Guidelines* 2019.
- Rhodes A et al. *Surviving Sepsis Campaign 2016*.
- Holcomb JB et al. *PROPPR Trial* (1:1:1 transfusion ratio). *JAMA* 2015;313(5):471-482.
- Inouye SK et al. *Confusion Assessment Method*. *Ann Intern Med* 1990;113(12):941-948.
- Posner K et al. *C-SSRS validation*. *Am J Psychiatry* 2011;168(12):1266-1277.
