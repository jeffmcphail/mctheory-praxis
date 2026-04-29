# Register Praxis 4-Hour OHLCV Collector as a Windows Scheduled Task
# Run as Administrator:
#   .\services\register_ohlcv_4h_task.ps1

$TaskName = "PraxisOhlcv4hCollector"
$PraxisDir = "C:\Data\Development\Python\McTheoryApps\praxis"
$BatFile = "$PraxisDir\services\ohlcv_4h_collector_service.bat"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`"" `
    -WorkingDirectory $PraxisDir

# Run once daily at 00:20 local time (5 min offset from the daily OHLCV
# collector at 00:15 to avoid log-file contention or transient API stress).
$Trigger = New-ScheduledTaskTrigger -Daily -At "00:20"

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
    -Description "Praxis 4h OHLCV collector -- BTC+ETH, last 7 days, daily at 00:20"

Write-Host ""
Write-Host "============================================================="
Write-Host "  Task registered: $TaskName"
Write-Host "  Trigger: Daily at 00:20"
Write-Host "  Collects: BTC + ETH 4h candles (last 7 days per run, idempotent)"
Write-Host "  Logs: $PraxisDir\logs\ohlcv_4h_collector.log"
Write-Host "============================================================="
Write-Host ""
Write-Host "  Start immediately:"
Write-Host "  Start-ScheduledTask -TaskName $TaskName"
Write-Host ""
