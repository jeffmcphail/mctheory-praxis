@echo off
REM Funding Rate Collector
REM Runs every 8 hours via Windows Task Scheduler, time-aligned to Binance
REM funding events at 00:00, 08:00, 16:00 UTC. Pulls last 7 days for safety
REM overlap (idempotent via INSERT OR REPLACE).
REM
REM Cycle 42a: extended from BTC+ETH to the full atlas Exp 13 deployment
REM universe (BNB excluded per atlas's degenerate base_rate finding).
REM Historical funding-rate data for the 4 new assets was backfilled by
REM Cycle 40 D1; this collector now keeps all 6 assets fresh going forward.
REM
REM Per memory #12 (exit-code honesty): each per-asset python call's
REM ERRORLEVEL is checked inside the FOR loop and aggregated into
REM FAIL_COUNT. If any asset fails, the bat exits non-zero so Task
REM Scheduler's LastResult reflects the failure honestly (without this,
REM the FOR-loop ERRORLEVEL leaks the last call's status and masks
REM middle-of-loop failures -- exactly the trap that surfaced during
REM Cycle 42a verification when SUPPORTED_ASSETS was missing XRP/ADA/AVAX).

setlocal enabledelayedexpansion

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\funding_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1
set FAIL_COUNT=0

echo [%date% %time%] Starting funding rate collection... >> "%LOG_FILE%"
for %%A in (BTC ETH SOL XRP ADA AVAX) do (
    python -u -m engines.crypto_data_collector collect-funding --asset %%A --days 7 >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        set /a FAIL_COUNT+=1
        echo [%date% %time%]   ASSET FAILED: %%A ^(errorlevel !ERRORLEVEL!^) >> "%LOG_FILE%"
    )
)

if !FAIL_COUNT! gtr 0 (
    echo [%date% %time%] Collection completed with !FAIL_COUNT! asset failure^(s^). >> "%LOG_FILE%"
    endlocal & exit /b 1
)
echo [%date% %time%] Collection complete. >> "%LOG_FILE%"
endlocal
