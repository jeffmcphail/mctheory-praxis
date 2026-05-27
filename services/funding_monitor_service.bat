@echo off
REM Funding Carry Monitor (Cycle 41 pilot)
REM Runs ~10 min after PraxisFundingCollector each cycle (collector at
REM 00:05/08:05/16:05 LOCAL; this task at 00:15/08:15/16:15 LOCAL).
REM
REM Loads the Cycle 40 verified phase3 models, computes the 11 hand-crafted
REM features for BTC + ETH, runs RF inference, and INSERT OR IGNOREs one
REM row per asset into the funding_signals table (PK (asset, timestamp)
REM where timestamp is the most recent funding-window UTC time).
REM
REM Idempotent on re-run within the same 8h window.

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\funding_monitor.log
set MODELS=%PRAXIS_DIR%\outputs\funding_carry_repro\cpo\phase3_models_funding.joblib

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting funding monitor cycle... >> "%LOG_FILE%"
python -u -m scripts.funding_monitor --assets BTC,ETH --models "%MODELS%" --persist >> "%LOG_FILE%" 2>&1
echo [%date% %time%] Monitor cycle complete (exit code: %ERRORLEVEL%). >> "%LOG_FILE%"
