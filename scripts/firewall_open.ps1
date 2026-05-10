# Open Windows Firewall for Caddy on TCP 8181.
# RUN ONCE, AS ADMINISTRATOR (right-click PowerShell -> Run as Administrator).

$ErrorActionPreference = "Stop"

$caddyExe = (Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter "caddy.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if (-not $caddyExe) {
    $caddyCmd = Get-Command caddy -ErrorAction SilentlyContinue
    if ($caddyCmd) { $caddyExe = $caddyCmd.Path }
}
if (-not $caddyExe) { throw "Caddy not found. Install: winget install CaddyServer.Caddy" }

$ruleName = "TrustedRisk-Caddy-8181"

# Idempotent — remove any pre-existing rule with the same name first.
Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule

$rule = New-NetFirewallRule `
    -DisplayName $ruleName `
    -Description "Allow inbound TCP 8181 for Caddy reverse proxy (flati.work)" `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort 8181 `
    -Program $caddyExe `
    -Profile Any

Write-Host "Firewall rule '$($rule.DisplayName)' created. Caddy ($caddyExe) can accept inbound on 8181." -ForegroundColor Green
