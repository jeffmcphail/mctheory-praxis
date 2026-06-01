@echo off
REM Funding Executor (Cycle 51 paper-trading scaffold)
REM Runs 3x daily at 00:20/08:20/16:20 LOCAL (~5 min after PraxisFundingMonitor
REM at :15). Reads funding_alerts; applies 9-control risk layer; logs decision
REM to paper_trades. NO EXCHANGE API CALLS.
REM
REM EXECUTOR_KILL_SWITCH env var (in .env): set "1"/"true" to force all
REM pending alerts to be marked skip with reason "EXECUTOR_KILL_SWITCH=on".
REM Emergency stop without redeploying code.
REM
REM Single python invocation -- exit code propagates naturally; no FAIL_COUNT
REM aggregation needed (vs the collector's nested 6x2 loop).

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\funding_executor.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting executor cycle... >> "%LOG_FILE%"
python -u -m engines.funding_executor >> "%LOG_FILE%" 2>&1
echo [%date% %time%] Executor cycle complete (exit code: %ERRORLEVEL%). >> "%LOG_FILE%"
