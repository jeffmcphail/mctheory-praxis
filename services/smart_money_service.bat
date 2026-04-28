@echo off
REM Smart Money Snapshot Service
REM Runs every 6 hours via Windows Task Scheduler

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\smart_money.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting smart money snapshot... >> "%LOG_FILE%"
python -u -m engines.smart_money discover --category ALL >> "%LOG_FILE%" 2>&1
python -u -m engines.smart_money snapshot >> "%LOG_FILE%" 2>&1
echo [%date% %time%] Snapshot complete. >> "%LOG_FILE%"
