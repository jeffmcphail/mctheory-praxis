@echo off
REM Praxis Info Bars Live Collector (Cycle 34).
REM Short-lived run-and-exit task; every 5 minutes via PraxisInfoBarsCollector.
REM Iterates DISTINCT (asset, bar_type, threshold_value) from info_bars and
REM appends newly-closed bars since each slice's MAX(end_timestamp). New
REM thresholds added via the backfill script auto-pick-up here next run.

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\info_bars_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting info_bars live collection... >> "%LOG_FILE%"
python -u -m engines.info_bars.writer --live >> "%LOG_FILE%" 2>&1
set EXITCODE=%errorlevel%
echo [%date% %time%] Collection complete (exit=%EXITCODE%). >> "%LOG_FILE%"
exit /b %EXITCODE%
