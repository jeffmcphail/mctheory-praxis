# Register Praxis Info Bars Live Collector as a Windows Scheduled Task.
# Run as Administrator:
#   .\services\register_info_bars_task.ps1
#
# Task name: PraxisInfoBarsCollector
# Cadence:   every 5 minutes, indefinite repetition
# Action:    cmd.exe /c services\info_bars_collector_service.bat
#
# After registration, verify with:
#   Get-ScheduledTask -TaskName PraxisInfoBarsCollector
#   Get-ScheduledTaskInfo -TaskName PraxisInfoBarsCollector
# Initially LastTaskResult = 267011 (task ready, not yet run); after the
# first successful run it should transition to 0.

$TaskName = "PraxisInfoBarsCollector"
$PraxisDir = "C:\Data\Development\Python\McTheoryApps\praxis"
$BatFile = "$PraxisDir\services\info_bars_collector_service.bat"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`"" `
    -WorkingDirectory $PraxisDir

# Run every 5 minutes starting at the next 5-minute boundary from now.
# Trigger is one-shot with repetition; matches the pattern used by
# PraxisTradesCollector for sub-hour cadence.
$Start = (Get-Date).AddMinutes(1)
$Trigger = New-ScheduledTaskTrigger `
    -Once `
    -At $Start `
    -RepetitionInterval (New-TimeSpan -Minutes 5)

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 4) `
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
    -Description "Praxis Info Bars live collector -- 5-minute cadence appending newly-closed info bars per slice"

Write-Host ""
Write-Host "============================================================="
Write-Host "  Task registered: $TaskName"
Write-Host "  Trigger: Every 5 minutes (starting $Start)"
Write-Host "  Runs:    engines.info_bars.writer --live"
Write-Host "  Logs:    $PraxisDir\logs\info_bars_collector.log"
Write-Host "============================================================="
Write-Host ""
Write-Host "  Start immediately:"
Write-Host "    Start-ScheduledTask -TaskName $TaskName"
Write-Host ""
Write-Host "  Inspect:"
Write-Host "    Get-ScheduledTaskInfo -TaskName $TaskName"
Write-Host ""
