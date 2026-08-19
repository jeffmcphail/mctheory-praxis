@echo off
REM Bybit Liquidation Collector -- allLiquidation stream (Cycle 62A T1 venue)
REM
REM THIS IS THE ACTIVE T1 LIQUIDATION COLLECTOR.
REM Binance's equivalent (services\liquidation_collector_service.bat) stays in
REM place but UNSCHEDULED -- its stream is unreachable from this host.
REM
REM WHY BYBIT AND NOT BINANCE
REM   No liquidation backfill exists on ANY path. Binance withdrew the public
REM   allForceOrders REST endpoint (stream-only), and data.binance.vision has
REM   no USD-margined liquidation dataset at all while its coin-margined one
REM   ends 2024-10-14 against trades coverage starting 2026-04-29 -- zero
REM   overlap. The series can therefore only ever be built FORWARD, and every
REM   hour not recorded is lost permanently. A reversible venue choice against
REM   an irreversible data loss is why this runs now.
REM
REM READ BEFORE USING THE DATA -- COUNTS DO NOT TRANSFER FROM BINANCE
REM   Bybit is a VENUE SUBSTITUTION, not a drop-in:
REM     1. Perp market share differs -- Binance carries substantially more
REM        perpetual OI and volume, so the same cascade yields a different
REM        number of events at different sizes.
REM     2. The liquidation ENGINE differs -- margin tiers, partial-liquidation
REM        rules, ADL and insurance-fund behaviour are not the same, so the
REM        same position is closed at a different price in a different number
REM        of pieces.
REM     3. The stream THROTTLE differs, and this one was measured: Binance's
REM        !forceOrder@arr caps at ONE event per symbol per second; Bybit's
REM        allLiquidation has no such cap (up to 18 distinct events in a single
REM        BTCUSDT second on this host). Bybit is the more complete record AND
REM        the less comparable one.
REM   Any Binance-derived prior on event RATES -- per-minute thresholds,
REM   burst-size percentiles -- must be re-estimated on Bybit data. What
REM   transfers is the event CLASS, which is what scenario A2 needs.
REM
REM COVERAGE IS A CHOSEN SUBSET
REM   Bybit has no all-market liquidation topic; allLiquidation is per-symbol.
REM   Default coverage is the six-asset funding/open-interest universe
REM   (BTC ETH SOL XRP ADA AVAX). Anything outside it is UNOBSERVED, which is
REM   a coverage boundary, not a quiet market.
REM
REM The 3550s duration against an hourly trigger is the trades_collector
REM handoff pattern: each invocation exits 50s before the next fires, so
REM invocations never overlap and the MultipleInstances IgnoreNew silent-skip
REM race (Cycles 7-8, 10) cannot occur. Single writer, by construction.
REM
REM EXIT CODES
REM   0  healthy -- connected, control topic live, rows written
REM   1  never connected, or a DB/config error
REM   2  connected but untrustworthy: a subscribe was rejected, or the control
REM      topic delivered nothing (a live socket with a silently dead
REM      subscription -- the Cycle 62A trap), or a genuinely empty window

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\bybit_liquidation_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting Bybit liquidation stream capture... >> "%LOG_FILE%"
python -u -m engines.bybit_liquidation_collector --verbose 2 collect --duration 3550 >> "%LOG_FILE%" 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] Capture exited with %RC%. >> "%LOG_FILE%"
exit /b %RC%
