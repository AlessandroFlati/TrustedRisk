# Software Bill of Materials (SBOM)

**Format**: [CycloneDX](https://cyclonedx.org/) 1.6
**Source artifact**: `docs/sbom.json`
**Generated**: 2026-04-29 UTC via `cyclonedx-bom`
**Components inventoried**: 298 Python packages (full transitive closure)

---

## 1. Purpose

The SBOM is required for FDA SaMD submissions per the *Cybersecurity
in Medical Devices: Quality System Considerations* (Sept 2023)
guidance. It is also a critical input to:

- CVE/NVD vulnerability triage (`docs/CVE_SCAN.md`)
- Supply-chain provenance audits (Executive Order 14028)
- Marketplace publishers wanting to verify the runtime's surface
  before listing TrustedRisk in a compliant workspace

---

## 2. Generation

Re-run after any `pyproject.toml` / `requirements.lock` change:

```bash
.venv/Scripts/python.exe -m pip install cyclonedx-bom
.venv/Scripts/python.exe -m cyclonedx_py environment \
    --output-format json --output-file docs/sbom.json
```

The generator inspects the active Python environment, so the SBOM
exactly matches the runtime classpath. Any component missing from
`docs/sbom.json` is also missing from the runtime -- no hidden
dependencies.

---

## 3. Top-level direct dependencies

The full transitive closure is in `docs/sbom.json`. The project's
direct dependencies (per `pyproject.toml`) are:

- `fastmcp` ≥ 3.0 -- MCP server runtime
- `pydantic` ≥ 2.13 -- schema layer
- `httpx` -- async HTTP client (FHIR, refresh-token, push notifications)
- `starlette`, `uvicorn` -- ASGI surface
- `fhirpy` -- async FHIR R4 client
- `numpy`, `scipy`, `statsmodels` -- calibration + Monte Carlo
- `sentence-transformers`, `faiss-cpu` -- grounding corpus retrieval
- `google-adk` -- A2A agent runtime (orchestrator, root LlmAgent)
- `python-dotenv` -- env loading
- `presidio-analyzer` (optional) -- PHI detection

Test/dev dependencies (excluded from production listings via
`extras_require = ["dev"]`):

- `pytest`, `pytest-asyncio`, `pytest-cov`, `hypothesis`
- `cyclonedx-bom` (SBOM tooling itself)
- `pip-audit` (vulnerability scanner)
- `ruff`, `mypy`, `mutmut` (linting / typing / mutation testing)

---

## 4. Cross-references

- `docs/CVE_SCAN.md` -- vulnerability scan + remediation status against this SBOM
- `docs/FDA_SAMD_ANALYSIS.md` §8 -- cybersecurity considerations
- `docs/SECURITY.md` -- vulnerability disclosure policy
- `pyproject.toml` -- direct-dependency manifest
- `Dockerfile` -- final runtime classpath

---

## 5. Refresh policy

The SBOM should be regenerated on every:
- Production release tag (`v0.X.Y`)
- Major dependency update
- Security advisory affecting any component
- Quarterly cadence (minimum)

Each regeneration must be archived (filename pattern `docs/sbom-v0.X.Y-YYYY-MM-DD.json`)
so historical decisions can be re-validated against the SBOM in effect
at the time the decision was made -- completing the
`a2a_agent.audit.fetch_archived_decision` reproducibility loop.
