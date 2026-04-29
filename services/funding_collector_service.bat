@echo off
REM Funding Rate Collector
REM Runs every 8 hours via Windows Task Scheduler, time-aligned to Binance
REM funding events at 00:00, 08:00, 16:00 UTC. Pulls last 7 days for safety
REM overlap (idempotent via INSERT OR REPLACE).

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\funding_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting funding rate collection... >> "%LOG_FILE%"
python -u -m engines.crypto_data_collector collect-funding --asset BTC --days 7 >> "%LOG_FILE%" 2>&1
python -u -m engines.crypto_data_collector collect-funding --asset ETH --days 7 >> "%LOG_FILE%" 2>&1
echo [%date% %time%] Collection complete. >> "%LOG_FILE%"
