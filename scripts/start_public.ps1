# TrustedRisk public launcher — boots MCP (loopback :9000) + Caddy (:8181 → 443 NAT).
#
# Usage:  .\scripts\start_public.ps1
# Stop:   Ctrl+C in this window stops Caddy; the MCP child process is killed by the trap.
#
# Prereqs (one-time):
#   1. Caddy installed (winget install CaddyServer.Caddy)
#   2. .env populated (TRUSTEDRISK_OAUTH_* + TRUSTEDRISK_PORT=9000)
#   3. data/oauth_clients.json present
#   4. Firewall inbound TCP 8181 allowed (run scripts/firewall_open.ps1 as admin once)
#   5. Router NAT 443 -> <LAN IP>:8181 configured
#   6. flati.work A record points to public IP
#
# The script:
#   - activates .venv
#   - exports PYTHONPATH=src
#   - starts MCP via uvicorn in a background job
#   - starts Caddy in foreground (so the script exits with Ctrl+C cleanly)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# ---- venv -----------------------------------------------------------------
$venvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    throw "Virtual environment not found at $venvActivate. Run: python -m venv .venv && .venv\Scripts\pip install -e .[deploy]"
}
. $venvActivate

# ---- env ------------------------------------------------------------------
$env:PYTHONPATH = "src"
$pythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"

# ---- locate caddy ---------------------------------------------------------
$caddyExe = (Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter "caddy.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if (-not $caddyExe) {
    $caddyCmd = Get-Command caddy -ErrorAction SilentlyContinue
    if ($caddyCmd) { $caddyExe = $caddyCmd.Path }
}
if (-not $caddyExe) { throw "Caddy not found. Install: winget install CaddyServer.Caddy" }

# ---- start MCP (background) ----------------------------------------------
Write-Host "[+] Starting MCP on 127.0.0.1:9000 ..." -ForegroundColor Cyan
$mcpProc = Start-Process -FilePath $pythonExe `
    -ArgumentList "-m", "mcp_server.server" `
    -WorkingDirectory $RepoRoot `
    -PassThru `
    -NoNewWindow `
    -RedirectStandardOutput (Join-Path $RepoRoot "logs\mcp.stdout.log") `
    -RedirectStandardError  (Join-Path $RepoRoot "logs\mcp.stderr.log")

Write-Host "    MCP PID: $($mcpProc.Id)  (logs: logs\mcp.*.log)" -ForegroundColor DarkGray

# Trap: kill MCP if Caddy exits / Ctrl+C.
$null = Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action {
    if (-not $mcpProc.HasExited) {
        Stop-Process -Id $mcpProc.Id -Force -ErrorAction SilentlyContinue
    }
}

# Brief wait so MCP binds before Caddy starts proxying.
Start-Sleep -Seconds 3
if ($mcpProc.HasExited) {
    Get-Content (Join-Path $RepoRoot "logs\mcp.stderr.log") -Tail 30
    throw "MCP exited during startup. Check logs\mcp.stderr.log"
}

# ---- start Caddy (foreground) --------------------------------------------
Write-Host "[+] Starting Caddy on :8181 (TLS-ALPN-01 for flati.work) ..." -ForegroundColor Cyan
Write-Host "    Caddy: $caddyExe" -ForegroundColor DarkGray
Write-Host "    Press Ctrl+C to stop both processes." -ForegroundColor Yellow
Write-Host ""

try {
    & $caddyExe run --config (Join-Path $RepoRoot "Caddyfile")
}
finally {
    Write-Host "[~] Stopping MCP (PID $($mcpProc.Id)) ..." -ForegroundColor Cyan
    if (-not $mcpProc.HasExited) {
        Stop-Process -Id $mcpProc.Id -Force -ErrorAction SilentlyContinue
    }
}
