@echo off
REM ===================================================================
REM  Praxis Live Collector Service
REM  Runs the Polymarket live price collector with auto-restart on crash.
REM  Designed for Windows Task Scheduler (run at startup, restart on failure).
REM ===================================================================

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\live_collector.log

REM Create logs directory if needed
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Activate venv
cd /d "%PRAXIS_DIR%"
call "%VENV%"

REM Force Python UTF-8 mode (prevents cp1252 encoding errors in log files)
set PYTHONUTF8=1

REM Log start
echo. >> "%LOG_FILE%"
echo =================================================== >> "%LOG_FILE%"
echo [%date% %time%] Live Collector starting >> "%LOG_FILE%"
echo =================================================== >> "%LOG_FILE%"

:loop
echo [%date% %time%] Starting collector... >> "%LOG_FILE%"
python -u -m engines.live_collector start --top 50 --interval 60 >> "%LOG_FILE%" 2>&1

REM If we get here, the collector exited (crash or error)
echo [%date% %time%] Collector exited with code %ERRORLEVEL%. Restarting in 30s... >> "%LOG_FILE%"
timeout /t 30 /nobreak >nul
goto loop
