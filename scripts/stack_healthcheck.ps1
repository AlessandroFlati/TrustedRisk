# Health-check probe for the TrustedRisk public stack. Registered as a
# Scheduled Task that fires every 5 minutes.
#
# What it does:
#   1. GET https://flati.work/.well-known/oauth-authorization-server
#      (the public OAuth metadata endpoint -- always responds 200 if
#      Caddy + MCP + OAuth handler are alive).
#   2. If the request fails twice in a row (state file in .stack_pids),
#      attempt `stack_ctl restart` and notify ntfy.
#   3. If the restart fails too, escalate ntfy to urgent.
#
# Idempotent: safe to run on a 5-minute cron without piling up.

$ErrorActionPreference = "Continue"

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$LogDir     = Join-Path $RepoRoot "logs"
$StateDir   = Join-Path $RepoRoot ".stack_pids"
New-Item -ItemType Directory -Force -Path $LogDir, $StateDir | Out-Null
$LogFile    = Join-Path $LogDir "healthcheck.log"
$StateFile  = Join-Path $StateDir "healthcheck.state"

function Write-Log($msg) {
    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss")
    Add-Content -Path $LogFile -Value "[$ts UTC] $msg"
}

function Read-EnvVar($name) {
    $envFile = Join-Path $RepoRoot ".env"
    if (-not (Test-Path $envFile)) { return $null }
    foreach ($line in Get-Content $envFile) {
        if ($line -match "^\s*$name\s*=\s*(.+)\s*$") { return $Matches[1].Trim('"',"'") }
    }
    return $null
}

function Send-Ntfy($title, $body, $priority = "default") {
    $topic = Read-EnvVar "TRUSTEDRISK_NTFY_TOPIC"
    if (-not $topic) { return }
    try {
        Invoke-RestMethod -Uri "https://ntfy.sh/$topic" -Method Post `
            -Body $body `
            -Headers @{ Title = $title; Priority = $priority; Tags = "trustedrisk" }
    } catch {
        Write-Log "ntfy POST failed: $($_.Exception.Message)"
    }
}

$DeployHost = Read-EnvVar "TRUSTEDRISK_DEPLOY_HOST"
if (-not $DeployHost) { $DeployHost = "flati.work" }
$Probe = "https://$DeployHost/healthz"

# Read the consecutive-failure counter (0 if file missing).
$failCount = 0
if (Test-Path $StateFile) {
    $raw = (Get-Content $StateFile -Raw).Trim()
    if ($raw -match "^\d+$") { $failCount = [int]$raw }
}

$alive = $false
try {
    $resp = Invoke-WebRequest -Uri $Probe -Method Get -TimeoutSec 10 -UseBasicParsing
    if ($resp.StatusCode -eq 200) { $alive = $true }
} catch {
    Write-Log "Probe failed: $($_.Exception.Message)"
}

if ($alive) {
    if ($failCount -gt 0) {
        Write-Log "Probe recovered after $failCount consecutive failures."
        Send-Ntfy "TrustedRisk recovered" "Probe OK after $failCount failures." "low"
    }
    Set-Content -Path $StateFile -Value "0"
    return
}

$failCount++
Set-Content -Path $StateFile -Value $failCount
Write-Log "Consecutive failures: $failCount."

# Single transient failure: log only, don't restart yet.
if ($failCount -lt 2) { return }

# Two or more consecutive failures: try restart.
Write-Log "Attempting stack_ctl restart (failCount=$failCount)."
Send-Ntfy "TrustedRisk DOWN" "Probe failed $failCount times; restarting stack." "high"

try {
    & (Join-Path $PSScriptRoot "stack_ctl.ps1") restart *>> $LogFile
} catch {
    Write-Log "stack_ctl restart threw: $($_.Exception.Message)"
    Send-Ntfy "TrustedRisk RESTART FAILED" $_.Exception.Message "urgent"
    return
}

# Wait a beat, re-probe to confirm recovery.
Start-Sleep -Seconds 15
try {
    $resp = Invoke-WebRequest -Uri $Probe -Method Get -TimeoutSec 10 -UseBasicParsing
    if ($resp.StatusCode -eq 200) {
        Write-Log "Restart verified, probe OK."
        Send-Ntfy "TrustedRisk recovered" "Auto-restart succeeded after $failCount failures." "default"
        Set-Content -Path $StateFile -Value "0"
        return
    }
} catch {
    Write-Log "Post-restart probe failed: $($_.Exception.Message)"
}

Send-Ntfy "TrustedRisk STILL DOWN" "Auto-restart did not recover; manual intervention required." "urgent"
