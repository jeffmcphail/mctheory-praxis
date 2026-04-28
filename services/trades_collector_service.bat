@echo off
REM Trade Flow Collector
REM Runs every hour via Windows Task Scheduler; each invocation polls
REM Binance for BTC+ETH trade history at 30s cadence for 3600 seconds,
REM so back-to-back scheduling provides continuous coverage.

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\trades_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting trades collection loop... >> "%LOG_FILE%"
python -u -m engines.crypto_data_collector collect-trades-loop --assets BTC ETH --interval 30 --duration 3600 >> "%LOG_FILE%" 2>&1
echo [%date% %time%] Loop invocation exited. >> "%LOG_FILE%"
