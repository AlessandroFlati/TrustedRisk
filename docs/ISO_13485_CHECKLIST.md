# ISO 13485 / IEC 62304 Design-Controls Checklist

**Standard**: ISO 13485:2016 (medical device QMS) + IEC 62304:2006/A1:2015 (medical-device software lifecycle)
**Software safety classification**: Class B (life-affecting but non-life-threatening) -- matches the IMDRF SaMD Class II categorisation in `docs/FDA_SAMD_ANALYSIS.md` §3.
**Document version**: 0.1 (pre-submission feasibility)
**Generated**: 2026-04-29 UTC
**Companion**: `docs/FDA_SAMD_ANALYSIS.md`, `docs/MODEL_CARD.md`, `docs/PROSPECTIVE_STUDY_PROTOCOL.md`, `SECURITY.md`, `docs/SBOM.md`

---

## How to read this checklist

Each design-control row maps to:
- **Standard reference** -- clause in ISO 13485 / IEC 62304
- **Status** -- Met / Partially met / Open
- **Evidence** -- concrete artefacts (test names, doc paths, code paths) that demonstrate compliance
- **Gap (when status ≠ Met)** -- what's needed to close

The checklist is intentionally stricter than this prototype release
needs. It maps the runtime to the production-grade evidence a real
SaMD audit would expect, so the gap to a Class II QMS audit is
visible.

---

## §1. Design + Development Planning (ISO 13485 §7.3.2)

| Control | Status | Evidence | Gap |
|---|---|---|---|
| Documented development plan | Met | `docs/ROADMAP.md` (Phase 0 -> 6), `TRUSTEDRISK_DESIGN.md` v0.3 | -- |
| Multi-disciplinary review | Partially met | Solo developer + LLM-mediated review; per-PR ultraview when scoped | A formal review board (clinician + biostatistician + security officer) is required before clinical deployment |
| Risk-management plan | Met | `docs/FDA_SAMD_ANALYSIS.md` §3 + `docs/PROSPECTIVE_STUDY_PROTOCOL.md` §9 | -- |

---

## §2. Design Inputs (ISO 13485 §7.3.3)

| Control | Status | Evidence | Gap |
|---|---|---|---|
| Functional + performance requirements | Met | `docs/MODEL_CARD.md`, `src/shared/schemas.py` (every Pydantic model is the runtime contract) | -- |
| Regulatory requirements identified | Met | `docs/FDA_SAMD_ANALYSIS.md` §1 (CDS exclusion) + §3 (SaMD class) + §5 (510(k)/De Novo/PCCP path) | -- |
| User requirements + intended use | Met | `docs/MODEL_CARD.md` §2 (Intended Use), agent-card descriptions | Validate via clinician interviews at IRB submission |

---

## §3. Design Outputs (ISO 13485 §7.3.4 + IEC 62304 §5.1.1-2)

| Control | Status | Evidence | Gap |
|---|---|---|---|
| Software architecture documented | Met | `docs/ARCHITECTURE.md` (system overview), `docs/GUIDE.md` §6 (A2A v1 + MCP integration), `docs/ROADMAP.md` (phase sequencing) | -- |
| Each output traceable to inputs | Met | Pydantic schemas pin every tool I/O contract; `compute_data_lineage` tool emits per-decision integrity hash | -- |
| Output accepted before release | Partially met | CI gate on full test suite (~ 2 800 tests) + 92 % coverage | Formal release-gate signoff (vs CI auto-pass) needed pre-clinical-deployment |
| Complete + verifiable | Met | All schemas exported; mypy clean; 92 % coverage | -- |

---

## §4. Verification + Validation (ISO 13485 §7.3.6-7 + IEC 62304 §5.5-5.7)

| Control | Status | Evidence | Gap |
|---|---|---|---|
| Unit verification | Met | `tests/unit/` -- ~ 2 200 tests; `pytest --cov` ≥ 92 % global | -- |
| Property-based testing | Met | `tests/property/` -- Hypothesis invariants on 10 critical scoring tools | -- |
| Adversarial verification | Met | `tests/adversarial/` -- 30 tests, 9 prompt-injection payloads; safety_redteam suite (fragility, OOD, pen-test, chaos) | -- |
| Integration testing | Met | `tests/integration/` -- federation specialists, multi-turn COIN, live HAPI (opt-in), PO compliance smoke (opt-in) | -- |
| Functional testing | Met | `tests/functional/` -- 181 pipeline-style tests across 15 files | -- |
| Golden-case regression | Met | `tests/golden/cohorts/` -- 600 synthetic cases × 12 bundles; `tests/golden/cases/` -- 36 hand-authored | -- |
| External validation | Partially met | MIMIC-IV demo 2.2 (n=275, ECE 0.082); `scripts/external_validation.py` | Real-cohort clinical validation pending -- see `docs/PROSPECTIVE_STUDY_PROTOCOL.md` |
| Live FHIR integration | Met (opt-in) | `tests/integration/test_live_hapi_fhir.py` -- passes against `https://hapi.fhir.org/baseR4` | -- |

---

## §5. Risk Management (ISO 14971 + IEC 62304 §7)

| Control | Status | Evidence | Gap |
|---|---|---|---|
| Hazard analysis | Met | `docs/FDA_SAMD_ANALYSIS.md` §4 (boundary-tool analysis) + §10 (open items) | -- |
| Risk-control measures | Met | Deterministic safety gate, abstain triggers, 4-critic ensemble, fairness audit, OOD detector, demographic bias guard | -- |
| Residual risk acceptance | Partially met | Documented in `docs/MODEL_CARD.md` §8 | Formal risk-management report (per ISO 14971 §10) needed pre-clinical-deployment |

---

## §6. Software Item Identification (IEC 62304 §5.3)

| Control | Status | Evidence | Gap |
|---|---|---|---|
| SBOM | Met | `docs/sbom.json` (CycloneDX 1.6, 298 components); generation procedure in `docs/SBOM.md` | -- |
| Versioning | Met | Semantic versioning `0.7.0`; coefficient bundle versioned via `model_version` field | -- |
| Configuration management | Met | Git history + `requirements.lock` + `Dockerfile` pin the runtime classpath | -- |

---

## §7. Cybersecurity (FDA 2023 + IEC 62304 §9.7)

| Control | Status | Evidence | Gap |
|---|---|---|---|
| Authentication | Met | OAuth 2.0 client_credentials, API-key, mTLS -- see `SECURITY.md` §4.2 | -- |
| Authorisation | Met | Multi-tenant FHIR allowlist per OAuth client; SHARP §3.2 enforcement | -- |
| Data integrity | Met | RFC 6962 Merkle audit chain + reproducibility archive | -- |
| Vulnerability disclosure | Met | `SECURITY.md` policy + 90-day responsible-disclosure window | -- |
| CVE scanning | Met | `docs/CVE_SCAN.md` summary; pip-audit run on every release | -- |
| Penetration testing | Partially met | In-house pen-test harness in `safety_redteam` module | Independent third-party pen test needed pre-clinical-deployment |

---

## §8. Maintenance (ISO 13485 §7.5.1, IEC 62304 §6)

| Control | Status | Evidence | Gap |
|---|---|---|---|
| Change-control plan | Met | Pre-determined Change Control Plan (PCCP) sketched in `docs/FDA_SAMD_ANALYSIS.md` §6 | Full PCCP submission pending |
| Re-calibration procedure | Met | `scripts/external_validation.py` + drift_monitor + W1 internal calibration workflow | -- |
| Patch / hotfix process | Partially met | Standard git -> CI -> release-tag flow | Clinical-impact patch SLO (≤ 14 days for high-severity) needs formalisation |

---

## §9. Post-Market Surveillance (ISO 13485 §8.2.1, FDA 2023)

| Control | Status | Evidence | Gap |
|---|---|---|---|
| Drift monitoring | Met | `a2a_agent.drift_monitor` (sliding-window mean-prob, KS distance, abstain rate, CI width) | -- |
| Alert routing | Met | `apps/alert_agent/` (port 8768, COMPOSE-1) wired into A2A push notifications (Phase 3.4) | -- |
| Outcome reporting / re-validation | Open | Requires post-market surveillance database tied to the live deployment | Operational, not engineering -- pending clinical deployment |

---

## §10. Document Control (ISO 13485 §4.2)

| Control | Status | Evidence | Gap |
|---|---|---|---|
| Approved document templates | Met | `docs/MODEL_CARD.md`, `docs/FDA_SAMD_ANALYSIS.md`, `docs/PROSPECTIVE_STUDY_PROTOCOL.md`, `docs/ISO_13485_CHECKLIST.md`, `docs/SBOM.md`, `docs/CVE_SCAN.md`, `SECURITY.md` | -- |
| Version-controlled | Met | Git for all docs; commit signoff per release | -- |
| Periodic review | Open | Currently event-driven (per release) | Quarterly review cadence to be formalised |

---

## §11. Summary

| Control category | Met | Partial | Open |
|---|---:|---:|---:|
| Planning | 1 | 1 | 0 |
| Inputs | 3 | 0 | 0 |
| Outputs | 3 | 1 | 0 |
| V&V | 7 | 1 | 0 |
| Risk | 2 | 1 | 0 |
| ID + config | 3 | 0 | 0 |
| Cybersecurity | 5 | 1 | 0 |
| Maintenance | 2 | 1 | 0 |
| Post-market | 2 | 0 | 1 |
| Document control | 2 | 0 | 1 |
| **Total** | **30** | **6** | **2** |

The gaps are operational (review board convening, third-party pen test,
post-market surveillance database) -- not engineering. The runtime is
ready for institutional QMS integration; the open items become
actionable once a clinical-deployment partner is engaged.
