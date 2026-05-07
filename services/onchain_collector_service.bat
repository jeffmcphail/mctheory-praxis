@echo off
REM BTC On-Chain Metrics Collector
REM Runs once daily after blockchain.info's daily UTC-midnight publication.
REM Pulls last 7 days for safety overlap (idempotent via INSERT OR IGNORE
REM on the `date` PK).

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\onchain_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting on-chain collection... >> "%LOG_FILE%"
python -u -m engines.crypto_data_collector collect-onchain --days 7 >> "%LOG_FILE%" 2>&1
echo [%date% %time%] Collection complete. >> "%LOG_FILE%"
