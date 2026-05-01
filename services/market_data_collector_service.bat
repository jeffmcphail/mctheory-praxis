@echo off
REM CoinGecko market data + BTC dominance collector.
REM Runs daily at 00:35 local Toronto time.
REM Uses /global for BTC dominance + /coins/{id} per asset (BTC, ETH, SOL).
REM Free tier: ~4 calls/day, well under the 30/min and 10k/month caps.

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\market_data_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting market_data collection... >> "%LOG_FILE%"
python -u -m engines.crypto_data_collector collect-market-data --asset all >> "%LOG_FILE%" 2>&1
echo [%date% %time%] Collection complete. >> "%LOG_FILE%"
