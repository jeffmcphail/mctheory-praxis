# Register Praxis BTC On-Chain Metrics Collector as a Windows Scheduled Task
# Run as Administrator:
#   .\services\register_onchain_task.ps1

$TaskName = "PraxisOnchainCollector"
$PraxisDir = "C:\Data\Development\Python\McTheoryApps\praxis"
$BatFile = "$PraxisDir\services\onchain_collector_service.bat"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`"" `
    -WorkingDirectory $PraxisDir

# Run once daily at 00:45 local time, after blockchain.info's daily UTC
# midnight publication. Pulls 7 days of overlap each run for idempotent
# safety margin (INSERT OR IGNORE on the `date` PK).
$Trigger = New-ScheduledTaskTrigger -Daily -At "00:45"

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
    -Description "Praxis BTC on-chain metrics collector -- daily at 00:45"

Write-Host ""
Write-Host "============================================================="
Write-Host "  Task registered: $TaskName"
Write-Host "  Trigger: Daily at 00:45"
Write-Host "  Collects: blockchain.info BTC on-chain metrics (last 7 days, idempotent)"
Write-Host "  Logs: $PraxisDir\logs\onchain_collector.log"
Write-Host "============================================================="
Write-Host ""
Write-Host "  Start immediately:"
Write-Host "  Start-ScheduledTask -TaskName $TaskName"
Write-Host ""
