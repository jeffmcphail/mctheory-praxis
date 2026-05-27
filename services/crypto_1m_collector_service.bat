@echo off
REM Crypto 1-Minute Data Collector
REM Runs daily via Windows Task Scheduler to accumulate 1m candles.
REM
REM Cycle 43h: bat hardened to track per-asset python exit codes via
REM FAIL_COUNT (the cmd.exe FOR-loop / sequential-call ERRORLEVEL
REM propagation otherwise reflects only the LAST command, silently
REM masking mid-list failures -- memory #12 trap).

setlocal enabledelayedexpansion

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\crypto_1m_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1
set FAIL_COUNT=0

echo [%date% %time%] Starting 1m data collection... >> "%LOG_FILE%"
python -u -m engines.crypto_data_collector collect-ohlcv-1m --asset BTC --days 2 >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    set /a FAIL_COUNT+=1
    echo [%date% %time%]   ASSET FAILED: BTC ^(errorlevel !ERRORLEVEL!^) >> "%LOG_FILE%"
)
python -u -m engines.crypto_data_collector collect-ohlcv-1m --asset ETH --days 2 >> "%LOG_FILE%" 2>&1
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
