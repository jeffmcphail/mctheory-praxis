# Register the Praxis Funding Regime-Turn Alert as a Windows Scheduled Task
# (Cycle 58 Track A).
#
# Read-only tripwire: watches production CEX-majors funding on the executor
# venue (binance) and fires a single ntfy.sh push (via PRAXIS_ALERT_URL) when
# the cross-asset N-day mean annualized funding clears a threshold and stays
# there for M consecutive complete UTC days. Carry was parked in Cycle 57
# (regime dead); this is the re-measure trigger, not a deploy signal.
#
# Run as Administrator (required for Register-ScheduledTask):
#   .\services\register_funding_regime_alert_task.ps1
#
# Sibling of register_funding_monitor_task.ps1; kept a SEPARATE task per
# memory #12 (exit-code honesty) and #13 (process pattern verification) so
# its failure is never masked by, and never masks, the other funding tasks.

$TaskName = "PraxisFundingRegimeAlert"
$PraxisDir = "C:\Data\Development\Python\McTheoryApps\praxis"
$BatFile = "$PraxisDir\services\funding_regime_alert_service.bat"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`"" `
    -WorkingDirectory $PraxisDir

# Once daily. The signal evaluates on COMPLETE UTC days, so a single daily
# run is sufficient; 00:30 LOCAL keeps it clear of the funding collector /
# monitor cluster (00:05 / 00:15). Time is LOCAL (Task Scheduler convention).
$Trigger = New-ScheduledTaskTrigger -Daily -At "00:30"

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew

$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType S4U `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Praxis funding regime-turn alert (Cycle 58 Track A). Read-only; reads funding_rates scoped to the executor venue, fires a single ntfy.sh push (PRAXIS_ALERT_URL) on a sustained cross-asset funding turn. Re-measure trigger for the parked carry strategy, not a deploy signal. Idempotent per episode via the funding_regime_alerts ledger."

Write-Host ""
Write-Host "============================================================="
Write-Host "  Task registered: $TaskName"
Write-Host "  Trigger: Daily at 00:30 LOCAL time"
Write-Host "  Bat: $BatFile"
Write-Host "  Logs: $PraxisDir\logs\funding_regime_alert.log"
Write-Host "  Ledger: funding_regime_alerts table in data\crypto_data.db"
Write-Host "============================================================="
Write-Host ""
Write-Host "  Requires PRAXIS_ALERT_URL in .env for the POST to fire."
Write-Host ""
Write-Host "  Start immediately (smoke test):"
Write-Host "  Start-ScheduledTask -TaskName $TaskName"
Write-Host ""
Write-Host "  Check status:"
Write-Host "  Get-ScheduledTaskInfo -TaskName $TaskName"
Write-Host ""
