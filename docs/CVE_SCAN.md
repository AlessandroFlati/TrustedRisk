# CVE Scan Summary

**Tool**: `pip-audit` (PyPI advisory database)
**Source**: `docs/sbom.json` + active venv classpath (298 packages)
**Scan date**: 2026-04-29 UTC
**Total findings**: 6 vulnerabilities across 3 packages
**Critical (CVSS ≥ 9.0)**: 1
**High (CVSS 7.0-8.9)**: 2
**Medium (CVSS 4.0-6.9)**: 3

> Re-run on every release tag and on every dependency bump:
>
> ```bash
> .venv/Scripts/python.exe -m pip_audit --strict --format json \
>     -o docs/pip_audit_<date>.json
> ```

---

## 1. Findings table

| Package | Installed | CVE / GHSA | Fix version | Severity | Surface |
|---|---|---|---|---|---|
| **langgraph** | 0.4.7 | CVE-2026-28277 | 1.0.10 | **High** | Optional dev -- not in production runtime |
| **litellm** | 1.82.6 | CVE-2026-35029 | 1.83.0 | **Critical** (admin auth bypass) | Optional dev (LLM critic dev surface only) |
| litellm | 1.82.6 | CVE-2026-35030 | 1.83.0 | **High** (JWT cache reuse) | Optional dev |
| litellm | 1.82.6 | GHSA-69x8-hrgq-fjj8 | 1.83.0 | **Critical** (full auth bypass chain) | Optional dev |
| litellm | 1.82.6 | GHSA-xqmj-j6mv-4862 | 1.83.7 | **Medium** (SSTI in /prompts/test) | Optional dev |
| pip | 26.0.1 | CVE-2026-3219 | (unfixed) | **Medium** (concatenated ZIP/TAR) | Build-time tooling |

All 6 findings are in **dev / build-time surface** -- none touch the
production runtime classpath that ships in `Dockerfile` (the production
image does NOT include langgraph, litellm, or pip-as-package-install).

---

## 2. Per-package remediation

### 2.1 `langgraph` (1 finding)

- **CVE-2026-28277**: msgpack checkpointer can deserialise arbitrary
  Python objects -> RCE on a malicious checkpoint.
- **Mitigation**: not used in TrustedRisk runtime (langgraph is a
  transitive dev-only dependency for an LLM benchmark we don't ship).
- **Action**: pin to `langgraph >= 1.0.10` in `pyproject.toml [dev]`
  next release; verify no regressions.

### 2.2 `litellm` (4 findings)

- All 4 affect the litellm proxy server (`/config/update`,
  `/prompts/test`, JWT cache). TrustedRisk uses litellm only as a
  client library inside `a2a_agent.llm_critic` -- it does NOT run the
  litellm proxy.
- **Action**: bump to `>= 1.83.7` (covers all 4 advisories) on next
  release.

### 2.3 `pip` (1 finding)

- **CVE-2026-3219**: pip mis-handles concatenated tar+ZIP archives.
- **Mitigation**: TrustedRisk does not run `pip` from inside the
  runtime; pip is build-time tooling only. The production image is
  immutable (no `pip install` at runtime).
- **Action**: track upstream; bump when a fixed pip release ships.

---

## 3. Risk assessment

The headline counts (1 critical / 2 high) **understate** the true
exposure because every finding is on a dev-only / build-time surface.
The production runtime -- the classpath inside the published Docker
image -- has zero open critical / high CVEs.

| Surface | Critical | High | Medium | Low |
|---|---:|---:|---:|---:|
| Production runtime (Dockerfile classpath) | 0 | 0 | 0 | 0 |
| Dev/build-time tooling | 1 | 2 | 3 | 0 |

The dev-surface findings are still tracked because:
- developers running tests locally are exposed
- a CI image that re-uses dev tooling could expand the blast radius

---

## 4. Refresh cadence

| Trigger | Action |
|---|---|
| Quarterly | Re-run `pip-audit`; archive output as `docs/pip_audit_YYYY-MM-DD.json` |
| Pre-release (`v0.X.Y`) | Mandatory re-run + remediation gate |
| Critical advisory | 24-hour SLO for production-runtime fixes; 7-day for dev-only |
| SBOM regeneration | Triggers a fresh CVE scan |

---

## 5. Cross-references

- `docs/sbom.json` -- SBOM inventory the scan ran against
- `docs/SBOM.md` -- SBOM walkthrough + refresh policy
- `SECURITY.md` -- disclosure policy + cybersecurity overview
- `docs/FDA_SAMD_ANALYSIS.md` §8 -- FDA cybersecurity considerations
- `docs/ISO_13485_CHECKLIST.md` §7 -- ISO 13485 cybersecurity controls
