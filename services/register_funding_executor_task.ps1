# Register Praxis Funding Executor as a Windows Scheduled Task (Cycle 51).
#
# Runs the paper-trading executor 3x daily at 00:20 / 08:20 / 16:20 LOCAL,
# 5 min after PraxisFundingMonitor (at :15). Reads funding_alerts; applies
# 9-control risk layer; logs decisions to paper_trades.
#
# Run as Administrator (required for Register-ScheduledTask):
#   .\services\register_funding_executor_task.ps1
#
# Same registration pattern as register_funding_monitor_task.ps1 (Cycle 41)
# and register_funding_task.ps1 (collector). S4U / Limited principal.

$TaskName = "PraxisFundingExecutor"
$PraxisDir = "C:\Data\Development\Python\McTheoryApps\praxis"
$BatFile = "$PraxisDir\services\funding_executor_service.bat"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`"" `
    -WorkingDirectory $PraxisDir

# Three daily triggers at 8-hour spacing, offset 20 min past the hour so the
# corresponding PraxisFundingMonitor run (at :15 LOCAL) has settled its
# INSERT OR IGNORE into funding_alerts before the executor queries it.
# Times below are LOCAL (Task Scheduler convention), not UTC.
$T1 = New-ScheduledTaskTrigger -Daily -At "00:20"
$T2 = New-ScheduledTaskTrigger -Daily -At "08:20"
$T3 = New-ScheduledTaskTrigger -Daily -At "16:20"
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
    -Description "Praxis funding-carry paper-trading executor (Cycle 51). Runs ~5 min after PraxisFundingMonitor each cycle. NO exchange API calls; writes decisions to paper_trades table."

Write-Host ""
Write-Host "============================================================="
Write-Host "  Task registered: $TaskName"
Write-Host "  Triggers: Daily at 00:20, 08:20, 16:20 LOCAL time"
Write-Host "  Bat: $BatFile"
Write-Host "  Logs: $PraxisDir\logs\funding_executor.log"
Write-Host "  Output: paper_trades table in data\crypto_data.db"
Write-Host "  KILL SWITCH: set EXECUTOR_KILL_SWITCH=1 in .env to disable entries"
Write-Host "============================================================="
Write-Host ""
Write-Host "  Smoke trigger:  Start-ScheduledTask -TaskName $TaskName"
Write-Host "  Check status:   Get-ScheduledTaskInfo -TaskName $TaskName"
Write-Host ""
