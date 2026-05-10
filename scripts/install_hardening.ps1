# Register Windows Scheduled Tasks that keep the TrustedRisk stack
# alive across reboots and transient failures during a demo /
# evaluation window.
#
# Tasks created (idempotent -- safe to re-run):
#   - TrustedRisk-Autostart : runs once at user logon, calls
#     scripts\stack_autostart.ps1.
#   - TrustedRisk-HealthCheck : runs every 5 minutes after logon,
#     calls scripts\stack_healthcheck.ps1.
#
# Both run as the current user (not SYSTEM) because the .venv and
# Caddy installs live under the user profile.
#
# Run this script ONCE from an elevated PowerShell:
#   .\scripts\install_hardening.ps1
#
# To uninstall:
#   .\scripts\install_hardening.ps1 -Uninstall

param(
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

# Self-elevate. Register-ScheduledTask with -RunLevel Highest needs admin.
$current = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $current.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    $args = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    if ($Uninstall) { $args += " -Uninstall" }
    Write-Host "[~] Re-launching elevated..." -ForegroundColor Yellow
    Start-Process -FilePath "powershell.exe" -ArgumentList $args -Verb RunAs -Wait
    exit
}

$RepoRoot     = Split-Path -Parent $PSScriptRoot
$AutostartPs1 = Join-Path $PSScriptRoot "stack_autostart.ps1"
$HealthPs1    = Join-Path $PSScriptRoot "stack_healthcheck.ps1"
$HiddenVbs    = Join-Path $PSScriptRoot "run_hidden.vbs"

$AutostartTaskName = "TrustedRisk-Autostart"
$HealthTaskName    = "TrustedRisk-HealthCheck"

function Remove-TaskIfExists($name) {
    $existing = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
        Write-Host "[~] Removed task '$name'" -ForegroundColor DarkGray
    }
}

if ($Uninstall) {
    Remove-TaskIfExists $AutostartTaskName
    Remove-TaskIfExists $HealthTaskName
    Write-Host "[done] Hardening uninstalled." -ForegroundColor Green
    return
}

if (-not (Test-Path $AutostartPs1)) { throw "Missing $AutostartPs1" }
if (-not (Test-Path $HealthPs1))    { throw "Missing $HealthPs1" }
if (-not (Test-Path $HiddenVbs))    { throw "Missing $HiddenVbs" }

# Both tasks run as the currently logged-on user.
$User = "$env:USERDOMAIN\$env:USERNAME"
Write-Host "[+] Registering tasks as '$User'" -ForegroundColor Cyan

# Common action template: invoke wscript.exe -> run_hidden.vbs -> powershell.
# Going through wscript avoids the brief console flash that happens when
# Task Scheduler launches `powershell.exe` directly in the user session.
function New-PsAction($scriptPath) {
    return New-ScheduledTaskAction `
        -Execute "wscript.exe" `
        -Argument "`"$HiddenVbs`" `"$scriptPath`"" `
        -WorkingDirectory $RepoRoot
}

# ----- Autostart task: at user logon, one-shot -----
Remove-TaskIfExists $AutostartTaskName

$autostartTrigger = New-ScheduledTaskTrigger -AtLogOn -User $User
$autostartSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

Register-ScheduledTask `
    -TaskName $AutostartTaskName `
    -Action (New-PsAction $AutostartPs1) `
    -Trigger $autostartTrigger `
    -Settings $autostartSettings `
    -User $User `
    -RunLevel Highest `
    -Description "Boot the TrustedRisk public stack (MCP + Federation + Caddy) on user logon." | Out-Null
Write-Host "[+] Registered '$AutostartTaskName' (trigger: AtLogOn)" -ForegroundColor Green

# ----- Healthcheck task: every 5 minutes, indefinitely -----
Remove-TaskIfExists $HealthTaskName

$healthTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Minutes 5)
$healthSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 4) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $HealthTaskName `
    -Action (New-PsAction $HealthPs1) `
    -Trigger $healthTrigger `
    -Settings $healthSettings `
    -User $User `
    -RunLevel Highest `
    -Description "Probe https://flati.work every 5 minutes; restart stack on consecutive failure; notify via ntfy.sh." | Out-Null
Write-Host "[+] Registered '$HealthTaskName' (trigger: every 5 min)" -ForegroundColor Green

Write-Host ""
Write-Host "Verify with:" -ForegroundColor Cyan
Write-Host "  Get-ScheduledTask -TaskName TrustedRisk-*" -ForegroundColor White
Write-Host ""
Write-Host "Logs land in:" -ForegroundColor Cyan
Write-Host "  $RepoRoot\logs\autostart.log" -ForegroundColor White
Write-Host "  $RepoRoot\logs\healthcheck.log" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Pick an ntfy topic (any string), e.g. 'tr-flati-9k3v2x'" -ForegroundColor White
Write-Host "  2. Add to .env: TRUSTEDRISK_NTFY_TOPIC=tr-flati-9k3v2x" -ForegroundColor White
Write-Host "  3. Install ntfy.sh app on phone, subscribe to that topic" -ForegroundColor White
Write-Host "  4. Disable auto-reboot for Windows Update via gpedit.msc:" -ForegroundColor White
Write-Host "     Computer Config -> Admin Templates -> Windows Components ->" -ForegroundColor DarkGray
Write-Host "     Windows Update -> Configure Automatic Updates ->" -ForegroundColor DarkGray
Write-Host "     'Notify for download and auto install'" -ForegroundColor DarkGray
