# Register Praxis Funding Rate Collector as a Windows Scheduled Task
# Run as Administrator:
#   .\services\register_funding_task.ps1

$TaskName = "PraxisFundingCollector"
$PraxisDir = "C:\Data\Development\Python\McTheoryApps\praxis"
$BatFile = "$PraxisDir\services\funding_collector_service.bat"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`"" `
    -WorkingDirectory $PraxisDir

# Three daily triggers at 8-hour spacing, offset 5 min past the hour to ensure
# the Binance funding event has settled before we pull. Times below are LOCAL
# (Task Scheduler convention), not UTC -- see banner note.
$T1 = New-ScheduledTaskTrigger -Daily -At "00:05"
$T2 = New-ScheduledTaskTrigger -Daily -At "08:05"
$T3 = New-ScheduledTaskTrigger -Daily -At "16:05"
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
    -Description "Praxis funding rate collector -- BTC+ETH, every 8 hours"

Write-Host ""
Write-Host "============================================================="
Write-Host "  Task registered: $TaskName"
Write-Host "  Trigger: Every 8 hours (00:05, 08:05, 16:05 local time)"
Write-Host "  Collects: BTC + ETH funding rates (last 7 days per run, idempotent)"
Write-Host "  Logs: $PraxisDir\logs\funding_collector.log"
Write-Host "  NOTE: triggers run at LOCAL time (Task Scheduler convention), not UTC."
Write-Host "  Aligns approximately with Binance funding events given Toronto is UTC-4/UTC-5."
Write-Host "============================================================="
Write-Host ""
Write-Host "  Start immediately:"
Write-Host "  Start-ScheduledTask -TaskName $TaskName"
Write-Host ""
