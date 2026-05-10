# Background controller for the TrustedRisk public stack (MCP + Caddy).
#
# Subcommands:
#   start    — boot MCP + Caddy detached, write PIDs to .stack_pids/
#   stop     — kill both processes, delete pid files
#   restart  — stop + start
#   status   — show running state + last log lines
#
# Usage:
#   .\scripts\stack_ctl.ps1 start
#   .\scripts\stack_ctl.ps1 stop
#   .\scripts\stack_ctl.ps1 restart
#   .\scripts\stack_ctl.ps1 status

param([Parameter(Mandatory=$true)][ValidateSet('start','stop','restart','status')][string]$Action)

$ErrorActionPreference = "Stop"

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$PidDir     = Join-Path $RepoRoot ".stack_pids"
$McpPidFile        = Join-Path $PidDir "mcp.pid"
$FederationPidFile = Join-Path $PidDir "federation.pid"
$CaddyPidFile      = Join-Path $PidDir "caddy.pid"
$LogDir     = Join-Path $RepoRoot "logs"

New-Item -ItemType Directory -Force -Path $PidDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Get-CaddyExe {
    $exe = (Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter "caddy.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
    if (-not $exe) {
        $cmd = Get-Command caddy -ErrorAction SilentlyContinue
        if ($cmd) { $exe = $cmd.Path }
    }
    if (-not $exe) { throw "Caddy not found." }
    return $exe
}

function Test-ProcessAlive($pidFile) {
    if (-not (Test-Path $pidFile)) { return $false }
    $procPid = Get-Content $pidFile -ErrorAction SilentlyContinue
    if (-not $procPid) { return $false }
    $proc = Get-Process -Id $procPid -ErrorAction SilentlyContinue
    return [bool]$proc
}

function Stop-IfRunning($pidFile, $label) {
    if (Test-Path $pidFile) {
        $procPid = Get-Content $pidFile
        try {
            Stop-Process -Id $procPid -Force -ErrorAction Stop
            Write-Host "[~] $label stopped (PID $procPid)" -ForegroundColor Cyan
        } catch {
            Write-Host "[~] $label was not running (PID $procPid)" -ForegroundColor DarkGray
        }
        Remove-Item $pidFile -ErrorAction SilentlyContinue
    }
}

function Start-Stack {
    if (Test-ProcessAlive $McpPidFile) {
        throw "MCP already running (PID $(Get-Content $McpPidFile)). Run 'stop' first."
    }
    if (Test-ProcessAlive $FederationPidFile) {
        throw "Federation already running (PID $(Get-Content $FederationPidFile)). Run 'stop' first."
    }
    if (Test-ProcessAlive $CaddyPidFile) {
        throw "Caddy already running (PID $(Get-Content $CaddyPidFile)). Run 'stop' first."
    }

    $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) { throw "venv missing at $venvPython" }
    $caddyExe = Get-CaddyExe

    # Truncate logs so each start is a fresh slate.
    foreach ($n in @("mcp", "federation", "caddy")) {
        Set-Content -Path (Join-Path $LogDir "$n.stdout.log") -Value "" -Encoding utf8
        Set-Content -Path (Join-Path $LogDir "$n.stderr.log") -Value "" -Encoding utf8
    }

    $env:PYTHONPATH = "src"

    $mcp = Start-Process `
        -FilePath $venvPython `
        -ArgumentList "-m", "mcp_server.server" `
        -WorkingDirectory $RepoRoot `
        -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $LogDir "mcp.stdout.log") `
        -RedirectStandardError  (Join-Path $LogDir "mcp.stderr.log")
    $mcp.Id | Out-File $McpPidFile -Encoding ascii
    Write-Host "[+] MCP started (PID $($mcp.Id))" -ForegroundColor Green

    $federation = Start-Process `
        -FilePath $venvPython `
        -ArgumentList "-m", "apps.federation.server" `
        -WorkingDirectory $RepoRoot `
        -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $LogDir "federation.stdout.log") `
        -RedirectStandardError  (Join-Path $LogDir "federation.stderr.log")
    $federation.Id | Out-File $FederationPidFile -Encoding ascii
    Write-Host "[+] Federation started (PID $($federation.Id))" -ForegroundColor Green

    Start-Sleep -Seconds 5

    if ($mcp.HasExited) {
        Get-Content (Join-Path $LogDir "mcp.stderr.log") -Tail 30
        throw "MCP exited during startup. See logs/mcp.stderr.log"
    }
    if ($federation.HasExited) {
        Get-Content (Join-Path $LogDir "federation.stderr.log") -Tail 30
        throw "Federation exited during startup. See logs/federation.stderr.log"
    }

    $caddy = Start-Process `
        -FilePath $caddyExe `
        -ArgumentList "run", "--config", (Join-Path $RepoRoot "Caddyfile") `
        -WorkingDirectory $RepoRoot `
        -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $LogDir "caddy.stdout.log") `
        -RedirectStandardError  (Join-Path $LogDir "caddy.stderr.log")
    $caddy.Id | Out-File $CaddyPidFile -Encoding ascii
    Write-Host "[+] Caddy started (PID $($caddy.Id))" -ForegroundColor Green
}

function Stop-Stack {
    Stop-IfRunning $CaddyPidFile      "Caddy"
    Stop-IfRunning $FederationPidFile "Federation"
    Stop-IfRunning $McpPidFile        "MCP"
}

function Show-Status {
    $mcpAlive        = Test-ProcessAlive $McpPidFile
    $federationAlive = Test-ProcessAlive $FederationPidFile
    $caddyAlive      = Test-ProcessAlive $CaddyPidFile
    $mcpPid          = if (Test-Path $McpPidFile)        { Get-Content $McpPidFile }        else { '<no pid>' }
    $federationPid   = if (Test-Path $FederationPidFile) { Get-Content $FederationPidFile } else { '<no pid>' }
    $caddyPid        = if (Test-Path $CaddyPidFile)      { Get-Content $CaddyPidFile }      else { '<no pid>' }

    Write-Host "MCP        : PID=$mcpPid  alive=$mcpAlive"
    Write-Host "Federation : PID=$federationPid  alive=$federationAlive"
    Write-Host "Caddy      : PID=$caddyPid  alive=$caddyAlive"
}

switch ($Action) {
    'start'   { Start-Stack }
    'stop'    { Stop-Stack }
    'restart' { Stop-Stack; Start-Sleep -Seconds 1; Start-Stack }
    'status'  { Show-Status }
}
