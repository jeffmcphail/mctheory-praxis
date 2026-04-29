@echo off
REM Fear and Greed Index Collector
REM Runs once daily after alternative.me's daily 00:00 UTC publication.
REM Pulls last 7 days for safety overlap (idempotent via INSERT OR REPLACE).

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\fear_greed_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting Fear and Greed collection... >> "%LOG_FILE%"
python -u -m engines.crypto_data_collector collect-fear-greed --days 7 >> "%LOG_FILE%" 2>&1
echo [%date% %time%] Collection complete. >> "%LOG_FILE%"
