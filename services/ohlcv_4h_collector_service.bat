@echo off
REM 4-Hour OHLCV Collector
REM Runs once daily via Windows Task Scheduler.
REM Pulls last 7 days of 4h candles for BTC and ETH (overlap = idempotent
REM safety margin against missed runs; INSERT OR REPLACE handles dupes).
REM
REM Cycle 43h: bat hardened to track per-asset python exit codes via
REM FAIL_COUNT so middle-of-sequence failures surface to Task Scheduler's
REM LastResult instead of being masked by the trailing call's status
REM (memory #12 -- exit-code honesty).

setlocal enabledelayedexpansion

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\ohlcv_4h_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1
set FAIL_COUNT=0

echo [%date% %time%] Starting 4h OHLCV collection... >> "%LOG_FILE%"
python -u -m engines.crypto_data_collector collect-ohlcv-4h --asset BTC --days 7 >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    set /a FAIL_COUNT+=1
    echo [%date% %time%]   ASSET FAILED: BTC ^(errorlevel !ERRORLEVEL!^) >> "%LOG_FILE%"
)
python -u -m engines.crypto_data_collector collect-ohlcv-4h --asset ETH --days 7 >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    set /a FAIL_COUNT+=1
    echo [%date% %time%]   ASSET FAILED: ETH ^(errorlevel !ERRORLEVEL!^) >> "%LOG_FILE%"
)

if !FAIL_COUNT! gtr 0 (
    echo [%date% %time%] Collection completed with !FAIL_COUNT! asset failure^(s^). >> "%LOG_FILE%"
    endlocal & exit /b 1
)
echo [%date% %time%] Collection complete. >> "%LOG_FILE%"
endlocal
