# Register Crypto 1-Minute Data Collector as a Windows Scheduled Task
# Run as Administrator:
#   .\services\register_crypto_1m_task.ps1

$TaskName = "PraxisCrypto1mCollector"
$PraxisDir = "C:\Data\Development\Python\McTheoryApps\praxis"
$BatFile = "$PraxisDir\services\crypto_1m_collector_service.bat"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`"" `
    -WorkingDirectory $PraxisDir

# Run every 6 hours to keep 1m data fresh
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 6)

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 60) `
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
    -Description "Praxis 1-minute crypto OHLCV collector -- BTC+ETH every 6 hours"

Write-Host ""
Write-Host "============================================================="
Write-Host "  Task registered: $TaskName"
Write-Host "  Trigger: Every 6 hours"
Write-Host "  Collects: BTC + ETH 1-minute candles (last 2 days per run)"
Write-Host "  Logs: $PraxisDir\logs\crypto_1m_collector.log"
Write-Host "============================================================="
Write-Host ""
Write-Host "  FIRST RUN: Collect full 30 days manually first:"
Write-Host "  python -m engines.crypto_data_collector collect-ohlcv-1m --asset BTC --days 30"
Write-Host "  python -m engines.crypto_data_collector collect-ohlcv-1m --asset ETH --days 30"
Write-Host ""
Write-Host "  Then start the task:"
Write-Host "  Start-ScheduledTask -TaskName $TaskName"
Write-Host ""
