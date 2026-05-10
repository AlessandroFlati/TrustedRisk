# TrustedRisk Roadmap -- Phase 0 -> 6

**Version**: 1.0
**Authored**: 2026-04-29

This is the operational plan that takes TrustedRisk from a single
calibrated MCP/A2A agent to a federation of marketplace-publishable
specialists composable in any A2A v1 workspace. The strategic
shift is captured in §0 -- without it, the rest reads as a pile of
features. With it, every phase below is a deliberate compounding step.

---

## §0. Strategic positioning

A2A v1 marketplaces do **not** offer an orchestrator-monolith. They
publish **specialist agents** that a workspace-side BYO agent
composes via a `Consult with another agent` UX --
the explicit "Agent Composition" pattern of A2A v1.

**Implication**: maximising challenge coverage = (a) **becoming a
federation** of focused specialists rather than one monolithic agent,
plus (b) **adding genuinely new pain-point specialists** beyond the
clinical-decision core. Both reuse the existing 58 MCP tools + 35 a2a
modules; both are marketplace-listable; both are visible as separate
tiles in a workspace composition surface. Three to five specialists is
the sweet spot -- more would dilute the demo, fewer would understate
the composition story.

The judging criteria align with this:

| Criterion | How the roadmap addresses it |
|---|---|
| **AI Factor** | Phase 2 specialists are LLM-native (PA letter drafting, scribe note synthesis, patient Q&A) |
| **Potential Impact** | Phase 2 covers $40 B/yr (PA), #1 burnout driver (documentation), patient self-service |
| **Feasibility** | Phase 0 SHARP/PO compliance + Phase 5 regulatory pack + existing Merkle audit + DP equity + multi-tenant OAuth |

---

## §1. Phase 1 -- Multi-specialist federation (4-5 days)

Split the single `trustedrisk-agent` into 4-5 marketplace-publishable
specialists, each with its own agent-card / port / focused skill
catalog, all sharing the underlying 58-tool MCP backend.

| Specialist | Port | Bundles reused | Skill catalog |
|---|---|---|---|
| `trustedrisk-discharge` | 8770 | core_discharge + clinical_workflow + patient_facing + data_normalization | safe_discharge_review, medication_safety_review, discharge_counseling, clinical_workflow_orders, data_normalization, patient_facing_translation |
| `trustedrisk-acute` | 8771 | ed_acute + stroke_acs + trauma_critical + endocrine_acute + obstetric_geriatric + nephrology + imaging | ed_admission_triage, deterioration_score, acute_stroke, chest_pain_acs, trauma_critical_care, endocrine_acute, imaging_appropriateness_safety, nephrology, maternal_obstetric, geriatric_assessment |
| `trustedrisk-evidence` | 8772 | diagnosis + external_knowledge + chart_intelligence + context_resolution | differential_diagnosis, external_knowledge_retrieval, chart_intelligence, context_resolution, treatment_selection |
| `trustedrisk-population` | 8773 | economics + DP equity + impact aggregator | cost_effectiveness, fairness_audit (population view), equity_dashboard |
| `trustedrisk-pedi-mh` | 8774 | pediatric + mental_health | pediatric_assessment, mental_health_crisis |

### Deliverables

- `apps/_shared/specialist_factory.py` -- common factory: builds an ASGI
  app that wraps the shared MCP backend + a focused agent-card +
  standard well-known routes
- Per specialist: `apps/specialist_<id>/__init__.py`,
  `apps/specialist_<id>/server.py`, `apps/specialist_<id>/agent_card.json`
- `docker-compose.yml` entries for ports 8770-8774
- `tests/integration/test_specialist_federation.py` -- every
  specialist's `/.well-known/agent-card.json` validates A2A v1 schema
  and declares the PO FHIR-context extension
- Optional: short bundle-orchestrator router that routes a free-text
  request to the right specialist (for the BYO orchestrator on the
  platform)

### Why this matters first

Phases 2, 3, 4 all consume the federation surface. We need it solid
before we build new specialists on top.

---

## §2. Phase 2 -- Pain-point specialists (6-8 days)

### S1. `pa-agent` -- Prior Authorization (3-4 days)

**Pain-point**: $40 B/yr US administrative cost; AMA top-3 clinician
burden; insurance denial rate ~15-20% in 2025.

**LLM-native**: drafts a payer-specific PA letter or appeal grounded
on the patient chart + current guidelines. Rule-based templates
exhaust the deterministic floor; the natural-language argumentation
must be generative.

**New MCP tools** (4):

- `compute_pa_evidence_pack` -- extracts the evidence bundle (FHIR
  Conditions + Observations + MedicationRequests + chart-intelligence
  NER) needed to support a specific PA request
- `compute_pa_letter_draft` -- LLM-mediated letter drafting with
  deterministic floor template; cite-back to chart line for every
  claim
- `compute_pa_appeal_likelihood` -- calibrated probability of approval
  using payer denial-rate priors + evidence-strength score
- `compute_pa_payer_rules_match` -- hard-coded payer-specific
  requirements + LLM gap analysis

**New a2a module**: `apps/specialist_pa/orchestrator.py` -- bounded
retry + 4-critic ensemble vote on the drafted letter (ground_claim
+ medication safety + fairness + LLM judge)

**Reuses**: chart_intelligence, ground_claim, drug_pricing, KNOWLEDGE
bundle, audit Merkle.

### S2. `scribe-agent` -- Documentation drafting (3 days)

**Pain-point**: clinicians spend 2-3 h/day on documentation; the #1
cited driver of burnout (Mayo Clinic 2024 survey).

**LLM-native**: drafts progress notes, discharge summaries, consult
letters, and admission H&P documents from a structured DecisionCard +
the clinician's voice or text input + the FHIR chart. Every claim
cite-backs to a source line.

**New MCP tools** (4):

- `compute_progress_note_draft` -- daily progress note draft
- `compute_discharge_summary_draft` -- multi-section discharge summary
- `compute_consult_letter_draft` -- specialty consult letter
- `compute_admission_hnp_draft` -- admission History and Physical

Each tool: deterministic template floor (sections, structure) + LLM
polish on the prose, with mandatory cite-back to FHIR observation IDs
or chart line spans.

**Reuses**: chart_intelligence, patient_facing, ground_claim, audit.

### S3. `patient-agent` -- Q&A post-discharge (2 days)

**Pain-point**: patients call the discharge line for clarification;
hospital callback queues are saturated.

**LLM-native** + demonstrates **A2A composition explicitly**: this
agent runs on its own port and **calls `trustedrisk-discharge`** via
A2A `Consult with another agent` to fetch the structured DecisionCard,
then renders Q&A via patient_facing tools.

**New MCP tools** (3):

- `compute_discharge_qa` -- Q&A loop with patient (multi-turn via
  A2A `contextId`)
- `compute_medication_what_if` -- "what if I take this on an empty
  stomach?" answered with deterministic guidance + LLM phrasing
- `compute_caregiver_handoff` -- caregiver-language summary

**Reuses**: patient_facing bundle entirely + multi-turn A2A.

---

## §3. Phase 3 -- A2A v1 protocol depth (3-4 days)

Implement the protocol features we currently declare but don't fully
support.

| Feature | Implementation | Tools touched |
|---|---|---|
| `INPUT_REQUIRED` lifecycle | DDx ranker and patient resolver pause mid-task and ask the client for clarification rather than abstaining | diagnosis + context_resolution bundles |
| `AUTH_REQUIRED` lifecycle | When the access token expires mid-task, transition to AUTH_REQUIRED + auto-refresh via `refresh_fhir_token()` (Phase 0) | sharp/refresh.py + middleware |
| Streaming | `outcomes_simulator` (Monte Carlo n=1000), `pubmed_search` paginated, batch endpoint | core MCP server + a2a |
| Push notifications | `alert_agent` wired into A2A push notification webhooks (`pushNotifications: true` in agent-card) | drift_monitor + alert_agent |
| `referenceTaskIds` | Cross-encounter timeline references (the patient timeline pulls prior task IDs) | memory.py + patient timeline |
| Multi-turn `contextId` continuity | Functional test that proves COIN works (5-turn patient↔discharge consultation) | patient_agent + discharge specialist |

---

## §4. Phase 4 -- Showcase + live integration (3 days)

- **v6 E2E showcase** -- rebuild `docs/e2e/index.html` to exercise all
  58 tools across the 20 bundles via the federation surface; one
  scenario per specialist + 5 cross-specialist composition scenarios.
  Output ~800-1000 KB.
- **Live HAPI FHIR test** -- `tests/integration/test_live_hapi_fhir.py`
  hits the public `https://hapi.fhir.org/baseR4` with a
  Synthea-generated Patient resource and exercises the entire
  SHARP->MCP->A2A->DecisionCard chain end-to-end.
- **PO public-instance compliance smoke** --
  `tests/integration/test_po_compliance.py` parses the agent-card
  served by `https://ts.fhir-mcp.promptopinion.ai/mcp` and
  `https://dotnet.fhir-mcp.promptopinion.ai/mcp` to confirm our
  capability advertisement matches their consumer expectations.
- **Documentation pass** -- flesh out the 8 stub bundle pages from
  Phase 0 (`economics.md`, `context_resolution.md`, `diagnosis.md`,
  `patient_facing.md`, `data_normalization.md`,
  `clinical_workflow.md`, `external_knowledge.md`,
  `chart_intelligence.md`) into full pages following
  `core_discharge.md`'s template.

---

## §5. Phase 5 -- Regulatory + scientific deepening (2-4 days)

- **`docs/PROSPECTIVE_STUDY_PROTOCOL.md`** -- IRB-ready study protocol
  for the FDA SaMD Pillar-3 clinical-validation gap (1000+ patients,
  1+ institution, paired clinician-vs-algorithm comparison)
- **SBOM** -- generate `docs/sbom.json` via `cyclonedx-py` over
  `requirements.lock`, ready for FDA submission
- **`SECURITY.md`** -- vulnerability disclosure policy + bug bounty
  terms + responsible-disclosure email
- **ISO 13485 design-controls checklist** --
  `docs/ISO_13485_CHECKLIST.md` with per-control evidence pointer
  (test name, doc section, code path)
- **CVE scan summary** -- `docs/CVE_SCAN.md` with the latest pip-audit
  + safety check output

---

## §6. Phase 6 -- Composer + Real LLM + Cloud Run (S1 -> S3 -> S2)

User-elected sequence (2026-04-29): execute S1 (composer agent),
then S3 (real LLM polish), then S2 (Cloud Run deploy).

### §6.1 Composer agent + workflow templates (3-4 days)
- `apps/composer/` on port 8780 with workflow engine
- 5 workflow templates: chf_admission, sepsis_workup,
  discharge_planning, outpatient_med_review, stroke_alert
- BYO-orchestrator pattern that consults the 9 federation specialists
- Tests demonstrating end-to-end multi-specialist composition

### §6.2 Real LLM polish wired (2-3 days)
- Cable Ollama (RTX 5090, gpt-oss-20b/120b) + Gemini API for
  PA letter polish + scribe note polish
- Hallucination detection harness (cite-back preservation)
- Streaming LLM tokens via the SSE infra of Phase 3.3
- Real-LLM regression tests

### §6.3 Cloud Run deployment (2-3 days)
- All 9 specialists live on URL pubblici (Cloud Run free tier)
- Custom domain + TLS
- Rate limiting + OpenTelemetry traces
- Make targets `make deploy` idempotenti
- Note: actual `gcloud` invocation is manual per scope rule

## §7. Phase 7 -- Tier 1 expansions (5 specialists/features)

- **§7.1** auto-coding agent (`apps/specialist_coder/`, port 8778)
- **§7.2** pharmacogenomic DSS (`apps/specialist_pgx/`, port 8779)
- **§7.3** Synthea-10k validation (`docs/validation/SYNTHEA_10K.md`)
- **§7.4** explainability depth (SHAP + ALE + concept-bottleneck +
  constitutional critic)
- **§7.5** pre-arrival triage agent (`apps/specialist_preadmit/`)

## §8. Phase 8 -- Tier 2 quality

- **§8.1** v7 interactive showcase (extends playground)
- **§8.2** architecture diagrams + tutorials
- **§8.3** performance benchmark + Prometheus + OTel
- **§8.4** production-grade FHIR client (circuit breaker, retry,
  pooling, pagination)
- **§8.5** more integration tests (cross-specialist, audit replay,
  DP epsilon budget)

## §9. Phase 9 -- Tier 3 optionality

- **§9.1** multi-language support (10 languages)
- **§9.2** synthetic data marketplace (`docs/datasets/`)
- **§9.3** workflow YAML templates (composer DSL)
- **§9.4** compliance attestations (HIPAA + GDPR + EU AI Act)
- **§9.5** N-of-1 trial design tool

---

## §7. Risk + sequencing notes

- **Federation cost**: 5 deployment endpoints. Cloud Run free tier (2 M
  req/month + scale-to-zero) absorbs this. If we ever need
  consolidation, all 5 specialists share the same binary -- we can
  collapse to a single deployment + path-based routing.
- **PA-agent payer rules**: payer-specific requirements vary by US
  insurer. We will hard-code the top 5 (Aetna, BCBS, Cigna, Humana,
  UnitedHealth) plus Medicare/Medicaid; the remainder fall back to
  generic AHRQ-style PA criteria. Documented as known limitation.
- **Live HAPI dependency**: `hapi.fhir.org/baseR4` has occasional
  downtime. The integration test marks it `@pytest.mark.live` and
  skips when the endpoint returns 5xx, so CI doesn't break on remote
  outages.
- **Schedule buffer**: with 12 days of net build time before the
  2026-05-11 deadline and the 5 phases summing to ~17-22 days, we
  will compress Phases 4-5 to the most-impactful items if Phase 2
  goes long. The pivot points are noted in the per-phase deliverables
  (e.g., Phase 5 items are independent and can ship in any order).

---

## §8. Tracking

Each phase will be split into numbered tasks via the in-session task
tracker. Phase boundaries are reflected in commit messages
(`feat(phase-1):`, `feat(phase-2-pa):`, etc.) so the git log doubles
as a phase-progress marker.
