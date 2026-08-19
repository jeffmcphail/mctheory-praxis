# Register the Praxis Bybit Liquidation Collector as a Windows Scheduled Task.
# Run as Administrator:
#   .\services\register_bybit_liquidation_task.ps1
#
# This is the ACTIVE T1 liquidation collector. The Binance equivalent has no
# registration script on purpose -- its stream is unreachable from this host
# and it stays unscheduled until that changes.
#
# Hourly trigger against a 3550s run duration: each invocation exits ~50s
# before the next fires, so instances never overlap and the MultipleInstances
# IgnoreNew silent-skip race (Cycles 7-8, 10) cannot occur. Single writer.

$TaskName = "PraxisBybitLiquidationCollector"
$PraxisDir = "C:\Data\Development\Python\McTheoryApps\praxis"
$BatFile = "$PraxisDir\services\bybit_liquidation_collector_service.bat"

if (-not (Test-Path $BatFile)) {
    Write-Error "Service script not found: $BatFile"
    exit 1
}

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`"" `
    -WorkingDirectory $PraxisDir

# Hourly, starting at the next whole hour. RepetitionInterval runs it
# indefinitely; the collector's own --duration is what bounds each run.
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date.AddHours((Get-Date).Hour + 1) `
    -RepetitionInterval (New-TimeSpan -Hours 1)

# ExecutionTimeLimit must sit ABOVE the run duration: 3550s is 59.2 minutes, so
# 65 gives headroom. If the scheduler kills a healthy capture mid-window the
# run leaves an unclosed gap row and is indistinguishable from a crash.
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 65) `
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
    -Description "Bybit allLiquidation stream -> liquidations table (Cycle 62A T1). Forward-only: no backfill exists on any path, so every unrecorded hour is lost permanently. Counts are NOT comparable to Binance-based priors."

Write-Host ""
Write-Host "Registered $TaskName (hourly, 3550s per run)."
Write-Host "Start it now with:  Start-ScheduledTask -TaskName $TaskName"
Write-Host "Log: $PraxisDir\logs\bybit_liquidation_collector.log"
