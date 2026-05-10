# Cloud Run Deployment -- Pre-Deploy Dry-Run Validation

**Phase 12.8 -- captured 2026-04-30 07:43 UTC**

- Services in `scripts/deploy_cloudrun.sh`: **18**
- Imports + `/healthz` 200: **18**
- Failures: **0**
- Duplicate service names: **0** (none)

## Per-service validation

| Service | Module | Import | /healthz | Agent card | Notes |
|---|---|---|---|---|---|
| trustedrisk-mcp | `mcp_server.server:build_http_app` | ✓ | 200 | 200 | - |
| trustedrisk-discharge | `apps.specialist_discharge.server:app` | ✓ | 200 | 200 | - |
| trustedrisk-acute | `apps.specialist_acute.server:app` | ✓ | 200 | 200 | - |
| trustedrisk-evidence | `apps.specialist_evidence.server:app` | ✓ | 200 | 200 | - |
| trustedrisk-population | `apps.specialist_population.server:app` | ✓ | 200 | 200 | - |
| trustedrisk-pedi-mh | `apps.specialist_pedi_mh.server:app` | ✓ | 200 | 200 | - |
| trustedrisk-pa | `apps.specialist_pa.server:app` | ✓ | 200 | 200 | - |
| trustedrisk-scribe | `apps.specialist_scribe.server:app` | ✓ | 200 | 200 | - |
| trustedrisk-patient | `apps.specialist_patient.server:app` | ✓ | 200 | 200 | - |
| trustedrisk-coder | `apps.specialist_coder.server:app` | ✓ | 200 | 200 | - |
| trustedrisk-pgx | `apps.specialist_pgx.server:app` | ✓ | 200 | 200 | - |
| trustedrisk-preadmit | `apps.specialist_preadmit.server:app` | ✓ | 200 | 200 | - |
| trustedrisk-quality | `apps.specialist_quality.server:app` | ✓ | 200 | 200 | - |
| trustedrisk-pophealth | `apps.specialist_pophealth.server:app` | ✓ | 200 | 200 | - |
| trustedrisk-appeals | `apps.specialist_appeals.server:app` | ✓ | 200 | 200 | - |
| trustedrisk-multimodal | `apps.specialist_multimodal.server:app` | ✓ | 200 | 200 | - |
| trustedrisk-composer | `apps.composer.server:app` | ✓ | 200 | 200 | - |
| trustedrisk-cds-hooks | `apps.cds_hooks.server:app` | ✓ | 200 | 404 | - |

## How to deploy for real

The user must run `gcloud auth login` + `gcloud config set project <id>` once on their workstation, then:

```bash
make deploy                # all 18 services
make deploy-mcp            # only the MCP server
make deploy-composer       # only the composer
DEPLOY_TARGETS=quality,pophealth,appeals \
    bash scripts/deploy_cloudrun.sh   # subset deploy
```

The dry-run validation in this file confirms every service boots in-process before the user pays for a Cloud Run rollout.
