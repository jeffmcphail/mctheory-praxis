# ===================================================================
#  Register Praxis Live Collector as a Windows Scheduled Task
#  Run this script ONCE as Administrator to set up the service.
#
#  To run: Right-click PowerShell -> Run as Administrator, then:
#    cd C:\Data\Development\Python\McTheoryApps\praxis
#    .\services\register_collector_task.ps1
#
#  To remove:
#    Unregister-ScheduledTask -TaskName "PraxisLiveCollector" -Confirm:$false
#
#  To check status:
#    Get-ScheduledTask -TaskName "PraxisLiveCollector"
#    Get-ScheduledTaskInfo -TaskName "PraxisLiveCollector"
# ===================================================================

$TaskName = "PraxisLiveCollector"
$PraxisDir = "C:\Data\Development\Python\McTheoryApps\praxis"
$BatFile = "$PraxisDir\services\live_collector_service.bat"

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Action: run the batch file
$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`"" `
    -WorkingDirectory $PraxisDir

# Trigger: at system startup
$Trigger = New-ScheduledTaskTrigger -AtStartup

# Settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -RestartCount 999 `
    -ExecutionTimeLimit (New-TimeSpan -Days 365) `
    -MultipleInstances IgnoreNew

# Principal: run whether user is logged on or not (requires password)
# Use -RunLevel Highest for admin privileges if needed
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType S4U `
    -RunLevel Limited

# Register
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Praxis Polymarket live price collector -- samples top 50 markets every 60s"

Write-Host ""
Write-Host "============================================================="
Write-Host "  Task registered: $TaskName"
Write-Host "  Trigger: At system startup"
Write-Host "  Restart: Every 1 min on failure (up to 999 times)"
Write-Host "  Time limit: 365 days"
Write-Host "  Logs: $PraxisDir\logs\live_collector.log"
Write-Host "============================================================="
Write-Host ""
Write-Host "  To start NOW:     Start-ScheduledTask -TaskName $TaskName"
Write-Host "  To stop:          Stop-ScheduledTask -TaskName $TaskName"
Write-Host "  To check status:  Get-ScheduledTaskInfo -TaskName $TaskName"
Write-Host "  To view logs:     Get-Content $PraxisDir\logs\live_collector.log -Tail 50"
Write-Host "  To remove:        Unregister-ScheduledTask -TaskName $TaskName"
Write-Host ""
