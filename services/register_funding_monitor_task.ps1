# Register Praxis Funding Monitor as a Windows Scheduled Task (Cycle 41 pilot).
#
# Runs the funding-carry monitor 3x daily at 00:15 / 08:15 / 16:15 LOCAL
# time (~10 min after PraxisFundingCollector at 00:05/08:05/16:05). The
# monitor loads the Cycle 40 verified phase3 models, computes 11 features
# for BTC + ETH, runs RF inference, and writes one row per asset into
# funding_signals via INSERT OR IGNORE.
#
# Run as Administrator (required for Register-ScheduledTask):
#   .\services\register_funding_monitor_task.ps1
#
# This is the Cycle 41 sibling of register_funding_task.ps1 (collector);
# they are intentionally two separate scheduled tasks per memory #12
# (exit-code honesty) and memory #13 (process pattern verification) --
# collector failure must not mask monitor failure and vice versa.

$TaskName = "PraxisFundingMonitor"
$PraxisDir = "C:\Data\Development\Python\McTheoryApps\praxis"
$BatFile = "$PraxisDir\services\funding_monitor_service.bat"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`"" `
    -WorkingDirectory $PraxisDir

# Three daily triggers at 8-hour spacing, offset 15 min past the hour so
# the corresponding PraxisFundingCollector run (00:05/08:05/16:05) has
# settled its INSERT OR REPLACE into funding_rates before the monitor
# queries it. Times below are LOCAL (Task Scheduler convention), not UTC.
# Local-vs-UTC framing: Toronto is UTC-4/UTC-5, so a 00:15 local trigger
# fires roughly 4-5h after the corresponding 00:00 UTC funding event;
# the strategy trades at funding-window cadence (8h), not microseconds,
# so this is the right operational frame for funding-carry signals.
$T1 = New-ScheduledTaskTrigger -Daily -At "00:15"
$T2 = New-ScheduledTaskTrigger -Daily -At "08:15"
$T3 = New-ScheduledTaskTrigger -Daily -At "16:15"
$Trigger = @($T1, $T2, $T3)

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
    -Description "Praxis funding-carry monitor (Cycle 41 pilot, BTC+ETH) -- runs ~10 min after PraxisFundingCollector each cycle, writes inference rows to funding_signals table in data/crypto_data.db. Uses the Cycle 40 verified phase3 models at outputs/funding_carry_repro/cpo/phase3_models_funding.joblib."

Write-Host ""
Write-Host "============================================================="
Write-Host "  Task registered: $TaskName"
Write-Host "  Triggers: Daily at 00:15, 08:15, 16:15 LOCAL time"
Write-Host "  Bat: $BatFile"
Write-Host "  Logs: $PraxisDir\logs\funding_monitor.log"
Write-Host "  Output: funding_signals table in data\crypto_data.db"
Write-Host "============================================================="
Write-Host ""
Write-Host "  Start immediately (smoke test):"
Write-Host "  Start-ScheduledTask -TaskName $TaskName"
Write-Host ""
Write-Host "  Check status:"
Write-Host "  Get-ScheduledTaskInfo -TaskName $TaskName"
Write-Host ""
