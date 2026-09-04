@echo off
REM Smart Money Snapshot Service
REM Runs every 6 hours via Windows Task Scheduler.
REM Step 1: discover top wallets (refresh tracked_wallets list)
REM Step 2: snapshot all tracked wallets' current positions
REM
REM Cycle 43h: bat hardened to track per-step python exit codes via
REM FAIL_COUNT. Previously, a `discover` failure followed by a successful
REM `snapshot` would surface as LastResult=0 in Task Scheduler, masking
REM the discover failure. Now each step's failure increments FAIL_COUNT
REM and the bat exits non-zero if any step failed (memory #12 --
REM exit-code honesty).

setlocal enabledelayedexpansion

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\smart_money.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1
set FAIL_COUNT=0

echo [%date% %time%] Starting smart money snapshot... >> "%LOG_FILE%"
python -u -m engines.smart_money discover --category ALL >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    set /a FAIL_COUNT+=1
    echo [%date% %time%]   STEP FAILED: discover ^(errorlevel !ERRORLEVEL!^) >> "%LOG_FILE%"
)
REM Cycle 64: --alert fires the PRAXIS_ALERT_URL push on an incomplete or stale
REM run. The snapshot exit code is now meaningful and is preserved verbatim
REM below (1=incomplete, 2=stale, 3=fatal) rather than collapsed to 1, so Task
REM Scheduler's LastTaskResult distinguishes them without reading the log.
python -u -m engines.smart_money snapshot --alert >> "%LOG_FILE%" 2>&1
set SNAP_RC=!ERRORLEVEL!
if !SNAP_RC! neq 0 (
    set /a FAIL_COUNT+=1
    echo [%date% %time%]   STEP FAILED: snapshot ^(errorlevel !SNAP_RC!^) >> "%LOG_FILE%"
)

REM `exit /b` ends the batch, which implicitly ends setlocal -- so no explicit
REM endlocal here. An explicit `endlocal & exit /b !SNAP_RC!` would discard the
REM variable before the exit could read it, collapsing every failure back to 1.
if !FAIL_COUNT! gtr 0 (
    echo [%date% %time%] Snapshot completed with !FAIL_COUNT! step failure^(s^); rc=!SNAP_RC!. >> "%LOG_FILE%"
    if !SNAP_RC! neq 0 exit /b !SNAP_RC!
    exit /b 1
)
echo [%date% %time%] Snapshot complete. >> "%LOG_FILE%"
endlocal
