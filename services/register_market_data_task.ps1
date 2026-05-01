# Register Praxis CoinGecko Market Data Collector as a Windows Scheduled Task
# Run as Administrator:
#   .\services\register_market_data_task.ps1

$TaskName = "PraxisMarketDataCollector"
$PraxisDir = "C:\Data\Development\Python\McTheoryApps\praxis"
$BatFile = "$PraxisDir\services\market_data_collector_service.bat"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`"" `
    -WorkingDirectory $PraxisDir

# Run once daily at 00:35 local time, after fear_greed at 00:30 and before
# any end-of-day reporting cron. /coins/{id} is current-state-only on the
# free tier; one call per day captures macro-state for BTC, ETH, SOL.
$Trigger = New-ScheduledTaskTrigger -Daily -At "00:35"

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
    -Description "Praxis market data collector -- daily at 00:35 (CoinGecko per-asset stats + BTC dominance from /global)"

Write-Host ""
Write-Host "============================================================="
Write-Host "  Task registered: $TaskName"
Write-Host "  Trigger: Daily at 00:35"
Write-Host "  Collects: CoinGecko /coins/{id} for BTC, ETH, SOL + /global BTC dominance"
Write-Host "  Logs: $PraxisDir\logs\market_data_collector.log"
Write-Host "============================================================="
Write-Host ""
Write-Host "  Start immediately:"
Write-Host "  Start-ScheduledTask -TaskName $TaskName"
Write-Host ""
