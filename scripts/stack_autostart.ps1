# Boot-time wrapper that waits for network connectivity, then starts
# the TrustedRisk public stack. Registered as a Scheduled Task by
# scripts\install_hardening.ps1.
#
# Behavior:
#   1. Wait up to 5 minutes for the public hostname to be DNS-resolvable
#      and reachable (Caddy needs DNS to acquire / refresh Let's Encrypt
#      certs, MCP backend just needs the local network).
#   2. Call scripts\stack_ctl.ps1 start.
#   3. On failure, write to logs\autostart.log and POST to ntfy if
#      TRUSTEDRISK_NTFY_TOPIC is set in .env.

$ErrorActionPreference = "Continue"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogDir   = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile  = Join-Path $LogDir "autostart.log"

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

Write-Log "autostart.ps1 invoked, RepoRoot=$RepoRoot"

# Wait up to 5 minutes for outbound network. Use a TCP connect to
# 1.1.1.1:443 since it works on every PowerShell version (no
# -TimeoutSeconds dependency).
function Test-Internet {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $task = $client.ConnectAsync("1.1.1.1", 443)
        if ($task.Wait(2000) -and $client.Connected) {
            $client.Close()
            return $true
        }
        $client.Close()
    } catch { }
    return $false
}

$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    if (Test-Internet) { $ready = $true; break }
    Start-Sleep -Seconds 10
}
if (-not $ready) {
    Write-Log "Network never came up after 5 minutes; aborting autostart."
    Send-Ntfy "TrustedRisk autostart FAILED" "Network not reachable after 5 min." "high"
    exit 1
}
Write-Log "Network reachable after $($i*10)s."

try {
    & (Join-Path $PSScriptRoot "stack_ctl.ps1") start *>> $LogFile
    if ($LASTEXITCODE -eq 0) {
        Write-Log "Stack started."
        Send-Ntfy "TrustedRisk stack UP" "Boot autostart succeeded." "low"
    } else {
        Write-Log "stack_ctl start returned exit $LASTEXITCODE"
        Send-Ntfy "TrustedRisk autostart FAILED" "stack_ctl exit $LASTEXITCODE; check logs\autostart.log" "high"
    }
} catch {
    Write-Log "stack_ctl start threw: $($_.Exception.Message)"
    Send-Ntfy "TrustedRisk autostart FAILED" $_.Exception.Message "high"
}
