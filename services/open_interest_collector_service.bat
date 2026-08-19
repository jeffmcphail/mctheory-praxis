@echo off
REM Open Interest Collector (Cycle 62A T2)
REM
REM Runs HOURLY via Windows Task Scheduler. Hourly is chosen because the
REM feature this feeds is oi_change_7d -- OI now against OI ~7 days ago --
REM which 168 points across the window serves roughly 24x over. Hourly also
REM matches Binance's finest OI-history granularity, so seeded rows and live
REM rows are the same kind of measurement rather than two regimes stitched
REM together. Cost is 6 assets x 2 venues x 24 = 288 rows/day.
REM
REM URGENCY: both venues wall off OI history and neither wall can be crossed
REM later. Binance rejects startTime beyond ~30 days at every granularity;
REM Bybit caps at 200 rows per request. A day not collected is a permanent
REM hole. Seeded once on 2026-08-19 (binance floor 2026-07-20T15:00Z, bybit
REM floor 2026-02-02T00:00Z); everything after that boundary is live capture.
REM
REM EXIT CODES: the collector returns 2 when a series has gone stale, not
REM merely when it wrote 0 rows -- 0 new rows is the correct outcome when the
REM table is already current, so freshness is what decides the exit code.
REM This bat propagates that so Task Scheduler's LastResult stays honest.

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\open_interest_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting open interest collection... >> "%LOG_FILE%"
python -u -m engines.open_interest_collector --verbose 2 collect >> "%LOG_FILE%" 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] Collection exited with %RC%. >> "%LOG_FILE%"
exit /b %RC%
