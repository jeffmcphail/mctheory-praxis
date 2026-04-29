# Register Praxis Fear and Greed Index Collector as a Windows Scheduled Task
# Run as Administrator:
#   .\services\register_fear_greed_task.ps1

$TaskName = "PraxisFearGreedCollector"
$PraxisDir = "C:\Data\Development\Python\McTheoryApps\praxis"
$BatFile = "$PraxisDir\services\fear_greed_collector_service.bat"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`"" `
    -WorkingDirectory $PraxisDir

# Run once daily at 00:30 local time, after alternative.me's daily 00:00 UTC
# publication. Pulls 7 days of overlap each run for idempotent safety margin.
$Trigger = New-ScheduledTaskTrigger -Daily -At "00:30"

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
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
    -Description "Praxis Fear and Greed Index collector -- daily at 00:30"

Write-Host ""
Write-Host "============================================================="
Write-Host "  Task registered: $TaskName"
Write-Host "  Trigger: Daily at 00:30"
Write-Host "  Collects: alternative.me Fear and Greed Index (last 7 days, idempotent)"
Write-Host "  Logs: $PraxisDir\logs\fear_greed_collector.log"
Write-Host "============================================================="
Write-Host ""
Write-Host "  Start immediately:"
Write-Host "  Start-ScheduledTask -TaskName $TaskName"
Write-Host ""
