@echo off
REM Funding Regime-Turn Alert (Cycle 58 Track A).
REM
REM Read-only tripwire that watches production CEX-majors funding on the
REM executor venue and fires a single ntfy.sh push when the cross-asset
REM funding level climbs back above threshold and stays there (sustained
REM M days). Carry was PARKED in Cycle 57 (regime went dead); this is the
REM "wake me when it's worth re-measuring" signal -- NOT a deploy trigger.
REM
REM Short-lived run-and-exit task; once daily. Evaluates on COMPLETE UTC
REM days (1-day resolution), so exact run time within the day is not
REM load-bearing. Idempotent per turn-episode via the funding_regime_alerts
REM ledger (re-runs in the same episode are a no-op).
REM
REM Honest exit code (memory #12): captures the python errorlevel and
REM exits with it, so Task Scheduler's LastTaskResult reflects a real
REM POST/read failure rather than the trailing echo's exit code.

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\funding_regime_alert.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting funding regime-turn alert... >> "%LOG_FILE%"
python -u -m scripts.funding_regime_alert --alert >> "%LOG_FILE%" 2>&1
set EXITCODE=%ERRORLEVEL%
echo [%date% %time%] Regime-alert complete (exit code: %EXITCODE%). >> "%LOG_FILE%"
exit /b %EXITCODE%
