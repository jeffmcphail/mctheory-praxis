# Praxis meta-registrar: runs every register_*_task.ps1 in this directory.
#
# Usage (elevated PowerShell):
#   .\services\register_all_tasks.ps1                 # discover + register all
#   .\services\register_all_tasks.ps1 -DryRun         # list what would run, do nothing
#   .\services\register_all_tasks.ps1 -Only "crypto_1m","smart_money"
#                                                     # run only the named scripts
#                                                     # (match is on the part between
#                                                     # 'register_' and '_task.ps1')
#
# Pattern: new collectors drop a register_<name>_task.ps1 into services/ and
# this meta-script picks them up automatically. No edits here required when
# new tasks are added.

param(
    [switch]$DryRun,
    [string[]]$Only
)

$ErrorActionPreference = "Stop"
$scriptsDir = $PSScriptRoot

$allScripts = Get-ChildItem -Path $scriptsDir -Filter "register_*_task.ps1" |
    Sort-Object Name

if ($Only) {
    $filtered = @()
    foreach ($s in $allScripts) {
        # Extract the <name> portion between "register_" and "_task.ps1"
        if ($s.Name -match "^register_(.+)_task\.ps1$") {
            $key = $Matches[1]
            if ($Only -contains $key) {
                $filtered += $s
            }
        }
    }
    $scripts = $filtered
} else {
    $scripts = $allScripts
}

Write-Host ""
Write-Host "============================================================="
Write-Host "  Praxis meta-registrar"
Write-Host "  Scripts directory: $scriptsDir"
if ($Only) {
    Write-Host "  Filter: -Only $($Only -join ', ')"
}
Write-Host "============================================================="

if (-not $scripts -or $scripts.Count -eq 0) {
    Write-Host "  No matching register_*_task.ps1 scripts found."
    exit 0
}

Write-Host "  Discovered $($scripts.Count) registration script(s):"
foreach ($s in $scripts) {
    Write-Host "    - $($s.Name)"
}

if ($DryRun) {
    Write-Host ""
    Write-Host "  DRY RUN -- no tasks registered. Remove -DryRun to execute."
    exit 0
}

# Advisory elevation check. Not hard-fail; some S4U tasks may register under
# specific UAC policies. If this warns and the registrations fail, right-click
# the shell and choose 'Run as administrator', then re-run.
$currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host ""
    Write-Warning "Not running elevated. Register-ScheduledTask may fail with 'Access is denied'."
    Write-Warning "If failures appear below, rerun from an elevated PowerShell."
}

$results = @()
foreach ($s in $scripts) {
    Write-Host ""
    Write-Host "-------- Running $($s.Name) --------"
    $errMsg = ""
    $status = "OK"
    try {
        & $s.FullName
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            $status = "FAIL"
            $errMsg = "script exited with code $LASTEXITCODE"
        }
    } catch {
        $status = "FAIL"
        $errMsg = $_.Exception.Message
        Write-Host "ERROR: $errMsg" -ForegroundColor Red
    }
    $results += [PSCustomObject]@{
        Script = $s.Name
        Status = $status
        Error  = $errMsg
    }
}

Write-Host ""
Write-Host "============================================================="
Write-Host "  SUMMARY"
Write-Host "============================================================="
$results | Format-Table -AutoSize

$failed = @($results | Where-Object { $_.Status -eq "FAIL" }).Count
if ($failed -gt 0) {
    Write-Host "$failed script(s) failed. Review errors above." -ForegroundColor Yellow
    exit 1
}

Write-Host "All $($results.Count) task registration script(s) completed successfully."
exit 0
