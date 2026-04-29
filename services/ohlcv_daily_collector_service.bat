@echo off
REM Daily OHLCV Collector
REM Runs once daily via Windows Task Scheduler.
REM Pulls last 7 days of daily candles for BTC and ETH (overlap = idempotent
REM safety margin against missed runs; INSERT OR REPLACE handles dupes).

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\ohlcv_daily_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting daily OHLCV collection... >> "%LOG_FILE%"
python -u -m engines.crypto_data_collector collect-ohlcv --asset BTC --days 7 >> "%LOG_FILE%" 2>&1
python -u -m engines.crypto_data_collector collect-ohlcv --asset ETH --days 7 >> "%LOG_FILE%" 2>&1
echo [%date% %time%] Collection complete. >> "%LOG_FILE%"
