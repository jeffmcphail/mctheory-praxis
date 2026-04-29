@echo off
REM Order Book Snapshot Collector
REM Runs every hour via Windows Task Scheduler; each invocation polls
REM Binance for BTC+ETH order books at 10s cadence for 3550 seconds,
REM exiting 50s before the next hourly trigger to ensure clean handoff
REM between back-to-back invocations and avoid the MultipleInstances
REM IgnoreNew silent-skip race condition. See Cycle 7 retro and
REM Cycle 8 brief for the diagnosis.

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\order_book_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting order book collection loop... >> "%LOG_FILE%"
python -u -m engines.crypto_data_collector collect-order-book-loop --assets BTC ETH --interval 10 --duration 3550 >> "%LOG_FILE%" 2>&1
echo [%date% %time%] Loop invocation exited. >> "%LOG_FILE%"
