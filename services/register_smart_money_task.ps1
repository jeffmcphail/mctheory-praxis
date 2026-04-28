# Register Smart Money Snapshot as a Windows Scheduled Task
# Run as Administrator:
#   cd C:\Data\Development\Python\McTheoryApps\praxis
#   .\services\register_smart_money_task.ps1

$TaskName = "PraxisSmartMoney"
$PraxisDir = "C:\Data\Development\Python\McTheoryApps\praxis"
$BatFile = "$PraxisDir\services\smart_money_service.bat"

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

# Trigger: every 6 hours
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 6)

# Settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -MultipleInstances IgnoreNew

# Principal
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
    -Description "Praxis smart money tracker -- discovers top wallets and snapshots positions every 6 hours"

Write-Host ""
Write-Host "============================================================="
Write-Host "  Task registered: $TaskName"
Write-Host "  Trigger: Every 6 hours"
Write-Host "  Logs: $PraxisDir\logs\smart_money.log"
Write-Host "============================================================="
Write-Host ""
Write-Host "  To start NOW:     Start-ScheduledTask -TaskName $TaskName"
Write-Host "  To stop:          Stop-ScheduledTask -TaskName $TaskName"
Write-Host "  To check status:  Get-ScheduledTaskInfo -TaskName $TaskName"
Write-Host "  To view logs:     Get-Content $PraxisDir\logs\smart_money.log -Tail 20"
Write-Host "  To remove:        Unregister-ScheduledTask -TaskName $TaskName"
Write-Host ""
