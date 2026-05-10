# Security Policy

**Project**: TrustedRisk -- Calibrated clinical decision-support agent
**Version**: 0.7.0
**Last reviewed**: 2026-04-29 UTC

---

## 1. Scope

This security policy covers:

- The TrustedRisk MCP server (`src/mcp_server/server.py`) and the 8
  specialist apps (`apps/specialist_*/`).
- The SHARP-on-MCP middleware (`src/mcp_server/sharp/`).
- The 5 federated A2A apps under `apps/` (federation_partner,
  alert_agent, scheduler_agent, playground, cds_hooks).
- The MCP tools in `src/mcp_server/tools/` (69 tools / 23 bundles as
  of 2026-04-29).

The scope **excludes**:

- Internal calibration workflow scaffolding under `workflows/` (internal R&D, not
  part of the deliverable surface).
- Synthetic test fixtures under `fixtures/` (not patient-identifying).

---

## 2. Reporting a vulnerability

**Email**: alessandro.flati@gmail.com
**Subject prefix**: `[TrustedRisk-Security]`

We aim to acknowledge reports within **3 business days** and provide a
preliminary triage assessment within **10 business days**. Critical
vulnerabilities (RCE, auth bypass, PHI exfiltration) are prioritised
ahead of all other work.

### 2.1 What to include

- Affected version (or git SHA)
- Component (specialist name, tool name, middleware, etc.)
- Reproduction steps
- Expected vs observed behaviour
- Proof-of-concept (if available)
- Suggested remediation (if known)

### 2.2 What we ask in return

- Don't publicly disclose the vulnerability before we've had a chance
  to coordinate a fix (90-day responsible-disclosure window -- extended
  for clinical-impact vulnerabilities).
- Don't access, modify, or exfiltrate patient data while researching.
- Don't perform denial-of-service attacks against production
  deployments. Local reproduction (`docker compose up`) is preferred.

---

## 3. Coordinated-disclosure timeline

| Day | Activity |
|---|---|
| 0 | Report received, acknowledged. |
| 0–3 | Initial triage; severity classification (CVSS 4.0). |
| 3–14 | Reproduction + root-cause analysis. |
| 14–60 | Patch development + internal validation against the test suite (~ 2 850+ tests). |
| 60–90 | Coordinated public disclosure with the reporter; CVE assignment if applicable; users notified via release notes. |
| 90+ | Public disclosure regardless. |

For vulnerabilities affecting clinical-impact components (PHI, audit
log integrity, the deterministic safety gate), the timeline tightens
to ≤ 30 days.

---

## 4. Known security surface

The platform's security model is layered:

### 4.1 Network / transport
- TLS 1.2+ recommended for all FHIR + A2A calls (TLS 1.3 preferred).
- The MCP server enforces SHARP `X-FHIR-...` headers per request.
- 403 returned on missing required FHIR context (Phase 0 / SHARP §3.2).
- 401 + `AuthRequiredHint` returned on FHIR-token expiry that the
  refresh path could not recover (Phase 3.2).

### 4.2 Authentication / authorisation
- OAuth 2.0 client_credentials grant (RFC 6749 §4.4) -- optional, off
  by default; enable via `TRUSTEDRISK_OAUTH_ENABLED=1`.
- Multi-tenant FHIR allowlist per OAuth client.
- API-key authentication for marketplace-mediated A2A (header
  `X-API-Key`).
- mTLS supported via the standard Starlette/uvicorn TLS stack.

### 4.3 PHI handling
- All tool inputs are PHI-scrubbed before hashing in the audit log
  (`a2a_agent.audit._hash`).
- The `compute_detect_phi` tool surfaces residual PHI in any free-text
  output before it leaves the runtime.
- The reproducibility archive stores structured inputs without
  identifying fields.

### 4.4 Audit + tamper-evidence
- Append-only HIPAA-style audit log (JSONL).
- RFC 6962 Merkle audit chain (`a2a_agent.merkle_audit`) -- every audit
  event hashed into a Merkle tree; root signed and published.
- GDPR Art. 22 / EU AI Act Art. 13 right-to-explanation surface
  (`a2a_agent.right_to_explanation`).

### 4.5 Adversarial robustness
- L1-perturbation fragility analysis (`a2a_agent.safety_redteam`).
- Mahalanobis-distance OOD detector with abstain trigger.
- Pen-test harness covering auth bypass / FHIR injection / SMART
  launch tampering / XSS.
- Chaos engineering (4 fault types, safe-rate ≥ 75 % required).

### 4.6 Differential privacy
- Equity dashboard publishes subgroup rates with Laplace noise
  (`a2a_agent.dp_equity`).

---

## 5. Known limitations

- **No formal threat model document yet**. Stride/PASTA-style threat
  model is queued for the v1.0 release.
- **No SOC 2 / HITRUST attestation**. Prototype-stage code; production
  deployment requires institutional security review.
- **Refresh-token storage**. The `X-FHIR-Refresh-Token` header is
  ephemeral per request (no server-side storage), but operators must
  ensure the upstream EHR / IDP rotates refresh tokens per OAuth 2.1
  best practices.

---

## 6. Hall of Fame

(Future security researchers' acknowledgements will be listed here as
disclosures land.)

---

## 7. References

- FDA. *Cybersecurity in Medical Devices: Quality System Considerations
  and Content of Premarket Submissions* (Sept 2023).
- *Executive Order 14028* -- Improving the Nation's Cybersecurity (May 2021).
- HIPAA Privacy Rule, 45 CFR §164.501 et seq.
- GDPR Art. 22 (right to not be subject to a decision based solely on
  automated processing).
- EU AI Act (Regulation (EU) 2024/1689), Art. 13 (transparency).
