@echo off
REM Crypto 1-Minute Data Collector
REM Runs daily via Windows Task Scheduler to accumulate 1m candles

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\crypto_1m_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting 1m data collection... >> "%LOG_FILE%"
python -u -m engines.crypto_data_collector collect-ohlcv-1m --asset BTC --days 2 >> "%LOG_FILE%" 2>&1
python -u -m engines.crypto_data_collector collect-ohlcv-1m --asset ETH --days 2 >> "%LOG_FILE%" 2>&1
echo [%date% %time%] Collection complete. >> "%LOG_FILE%"
