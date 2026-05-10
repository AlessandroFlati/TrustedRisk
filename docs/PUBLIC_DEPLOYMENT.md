# Public deployment -- flati.work

TrustedRisk MCP + 16-agent A2A federation exposed on
`https://flati.work` via Caddy + DDNS + home-router NAT.

## Topology

```
  Internet
     |
     | TCP 443
     v
  [Router NAT 443 -> LAN_IP:8181]
     |
     v
  Caddy (:8181)                  <-- TLS termination (Let's Encrypt, TLS-ALPN-01)
     |
     +--- /a2a/* ----------> Federation (127.0.0.1:9001)
     |                          |
     |                          +-- /a2a/trustedrisk-discharge   -> sub-app
     |                          +-- /a2a/trustedrisk-acute       -> sub-app
     |                          +-- ... 13 more specialists ...
     |                          +-- /a2a/trustedrisk-composer    -> sub-app
     |
     +--- everything else --> MCP server (127.0.0.1:9000)
                                (OAuth + SHARP + rate-limit)
```

The federation server (`apps.federation.server:app`) hosts all 15
specialists + the composer in a single uvicorn process. Each agent
keeps its own SHARP / OAuth middleware stack; `apps.federation.server`
only adds a thin `_StripRootPath` ASGI wrapper to make Starlette 1.0+
Mount semantics line up with the sub-app's whitelist checks.

Why no port 80: only 443 is forwarded by the router. ACME falls back to
TLS-ALPN-01, which is satisfied on the same 443 path NAT'd to 8181, so
no separate :80 listener is required.

## Files

| Path | Purpose |
|---|---|
| `Caddyfile` | reverse proxy + TLS config (routes `/a2a/*` to :9001, rest to :9000) |
| `apps/federation/server.py` | unified A2A federation (16 sub-apps under `/a2a/<name>`) |
| `.env` | MCP host/port + OAuth secret + clients path |
| `data/oauth_clients.json` | OAuth client registry (gitignored -- contains hashes) |
| `scripts/stack_ctl.ps1` | start/stop/restart/status for MCP + Federation + Caddy |
| `scripts/start_public.ps1` | foreground boot of MCP + Caddy (legacy, no federation) |
| `scripts/firewall_open.ps1` | one-time firewall rule (run as admin) |
| `caddy_data/` | Caddy ACME state (gitignored) |
| `logs/caddy_access.log`, `logs/{mcp,federation}.*.log` | runtime logs |

## One-time setup (already done)

1. `winget install CaddyServer.Caddy` -- Caddy 2.11.x
2. JWT signing secret + `flati-work-primary` client provisioned in
   `.env` and `data/oauth_clients.json`
3. Router NAT: `TCP 443 -> <LAN_IP>:8181`
4. DNS A record: `flati.work -> <public IP>`

## One-time setup (still to do -- admin required)

Open the inbound firewall rule for Caddy on port 8181. **Run PowerShell
as Administrator** and execute:

```powershell
cd C:\Users\aless\PycharmProjects\TrustedRisk
.\scripts\firewall_open.ps1
```

## Boot

Background-managed (recommended):

```powershell
cd C:\Users\aless\PycharmProjects\TrustedRisk
.\scripts\stack_ctl.ps1 start    # boots MCP + Federation + Caddy detached
.\scripts\stack_ctl.ps1 status   # shows PID + alive flag for each
.\scripts\stack_ctl.ps1 stop     # tears all three down
```

Foreground (legacy, no federation):

```powershell
.\scripts\start_public.ps1
```

`stack_ctl.ps1 start` brings up:
- MCP server on `127.0.0.1:9000` (`logs/mcp.*.log`)
- Federation server on `127.0.0.1:9001` (`logs/federation.*.log`)
- Caddy on `:8181` with auto-TLS for `flati.work` (`logs/caddy.*.log`)

First boot will fetch a Let's Encrypt cert via TLS-ALPN-01. Watch
Caddy's stdout for `certificate obtained successfully`.

## Smoke tests

```powershell
# Local loopback (bypasses Caddy)
curl http://127.0.0.1:9000/healthz
# {"status":"ok","tools_registered":145,"bundles_registered":47,...}

curl http://127.0.0.1:9001/healthz
# {"status":"ok","n_agents":16,"agents":["trustedrisk-acute",...]}

curl http://127.0.0.1:9001/a2a/trustedrisk-discharge/.well-known/agent-card.json

# Through Caddy + public DNS (after first cert issuance, ~10-60s)
curl https://flati.work/healthz                                        # MCP root
curl https://flati.work/a2a/trustedrisk-discharge/.well-known/agent-card.json
curl https://flati.work/a2a/trustedrisk-composer/.well-known/agent-card.json

# OAuth token
$body = "grant_type=client_credentials&client_id=flati-work-primary&client_secret=<PLAINTEXT>&scope=discharge.read"
curl -X POST https://flati.work/oauth/token -H "Content-Type: application/x-www-form-urlencoded" -d $body

# Authenticated MCP probe
curl https://flati.work/.well-known/agent-card.json
```

## Public surface map

| Path | Auth | Purpose |
|---|---|---|
| `/healthz` | none | liveness |
| `/readyz` | none | (Caddy blocks -- internal only) |
| `/metrics` | none | (Caddy blocks -- internal only) |
| `/oauth/token` | none (issues tokens) | OAuth2 client_credentials |
| `/.well-known/agent-card.json` | none | A2A root agent (MCP) |
| `/.well-known/marketplace.json` | none | Federation marketplace manifest |
| `/.well-known/sharp-capabilities.json` | none | SHARP capability declaration |
| `/a2a/trustedrisk-<id>/.well-known/agent-card.json` | none | per-specialist A2A card (15 specialists + composer) |
| `/a2a/trustedrisk-<id>/mcp/*` | OAuth + SHARP | per-specialist MCP dispatch |
| `/a2a/trustedrisk-<id>/api/*` | OAuth + SHARP | per-specialist batch / stream / push |
| `/openapi.json`, `/docs` | none | (only when agent_card_path is passed) |
| `/mcp/*` | OAuth Bearer + SHARP headers | MCP tool dispatch (root) |
| `/api/batch/decision-cards` | OAuth + SHARP | batch pipeline |
| `/api/stream/*` | OAuth + SHARP | SSE streaming |
| `/api/push/configs` | OAuth + SHARP | webhook registry |
| `/ws/chat` | OAuth + SHARP | A2A WebSocket chat |
| `/smart/*` | none | SMART-on-FHIR launch |

## Credential reference

OAuth client credentials are in `data/oauth_clients.json` (hash) and
the original plaintext lives only in `.env`-adjacent secrets storage:

- **client_id**: `flati-work-primary`
- **client_secret**: see local password manager -- never commit
- **tenant**: `flati-work`
- **allowed_scopes**: `discharge.read`, `discharge.execute`, `phi.scan`, `grounding.read`
- **allowed_fhir_servers**: `[]` (wildcard, single-tenant trust)

To rotate the client secret:

```powershell
$venv = ".venv\Scripts\python.exe"
$env:PYTHONPATH = "src"
$new = -join ((1..32) | ForEach-Object { '{0:x2}' -f (Get-Random -Maximum 256) })
$hash = & $venv -c "from mcp_server.oauth import OAuthClient; print(OAuthClient.hash_secret('$new'))"
# Replace client_secret_hash in data/oauth_clients.json with $hash
# Distribute $new to clients
```

## Operational notes

- **Cert renewal**: Caddy auto-renews ~30 days before expiry. No action required.
- **DDNS update**: if your public IP changes, update no-ip's record before
  the cert expires; otherwise TLS-ALPN-01 challenge fails.
- **Rate limit**: per-IP limiter is active (see `a2a_agent.observability`).
  Caddy is the visible client to MCP, so the limit is effectively global --
  tune `RATE_LIMIT_*` env vars if needed.
- **Logs**: `logs/caddy_access.log` (JSON, rotated 10MB×7) and
  `logs/mcp.{stdout,stderr}.log` (truncated on each restart).
