@echo off
REM Multi-Asset OHLCV Collector (Cycle 62A T4)
REM
REM Runs DAILY via Windows Task Scheduler. Keeps ohlcv_daily and ohlcv_4h
REM current across the layered universe:
REM   base   -- BTC ETH SOL XRP ADA AVAX (the funding_rates universe)
REM   unlock -- config/unlock_universe.json, where the asset lists on Binance
REM
REM WHY IT EXISTS: regime class K (cross-sectional dispersion) needs >= 3
REM universe assets. Every OHLCV table held exactly BTC and ETH, so K was
REM uncomputable and sat in RegimeState.missing on 100% of evaluations
REM (Cycle 61 T5). After Cycle 62A, ohlcv_daily holds 20 assets and K is
REM verified computable at the acting layer.
REM
REM This does NOT replace ohlcv_daily_collector_service.bat or
REM ohlcv_4h_collector_service.bat. Those keep their BTC/ETH scope and their
REM own cadence; this one is purely additive and writes into the same tables
REM with INSERT OR IGNORE, so the two cannot conflict.
REM
REM --days 30 is a top-up window, not a backfill: the 400-day history was
REM seeded once on 2026-08-19. A short window keeps the daily run cheap while
REM still self-repairing a few missed days.

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\ohlcv_universe_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting multi-asset OHLCV collection... >> "%LOG_FILE%"
python -u -m engines.ohlcv_universe_collector --verbose 2 collect --days 30 --days-4h 30 >> "%LOG_FILE%" 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] Collection exited with %RC%. >> "%LOG_FILE%"
exit /b %RC%
