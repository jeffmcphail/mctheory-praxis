@echo off
REM Liquidation Collector -- Binance forced-order stream (Cycle 62A T1)
REM
REM ############################################################
REM #  DO NOT REGISTER THIS TASK YET -- THE FEED IS BLOCKED.    #
REM ############################################################
REM
REM Binance's futures WebSocket (fstream.binance.com) accepts the connection
REM from this host (HTTP 101, socket stays open, no close code) and then
REM delivers ZERO frames -- including on !markPrice@arr@1s, which pushes
REM unconditionally every second. Measured 2026-08-19:
REM
REM   Binance SPOT  btcusdt@aggTrade  -> 347 frames / 20s   DATA FLOWS
REM   Binance PERP  btcusdt@aggTrade  ->   0 frames / 20s   SILENT
REM   Binance PERP  !markPrice@arr@1s ->   0 frames / 20s   SILENT
REM   Binance PERP  !forceOrder@arr   ->   0 frames / 119s  SILENT
REM   Bybit   PERP  publicTrade       -> 271 frames / 20s   DATA FLOWS
REM   Hyperliquid   trades            ->  96 frames / 20s   DATA FLOWS
REM
REM Binance futures REST (fapi.binance.com) works normally from the same host,
REM which is why the open-interest collector is unaffected. Reproduced inside
REM and outside the tool sandbox, so it is not a harness artefact. Binance
REM removed the public allForceOrders REST endpoint, so forced orders are
REM stream-only and there is no same-venue fallback.
REM
REM Registering this task now would produce an hourly stream of exit-code-2
REM runs -- which is at least honest, but it is noise, not data. Register it
REM only once the feed is reachable (VPN/egress change), or after a decision
REM to collect Bybit's allLiquidation topic instead, which is a VENUE
REM SUBSTITUTION and changes what forced-trade scenario A2 is measured
REM against.
REM
REM The 3550s duration against an hourly trigger is the trades_collector
REM handoff pattern: each invocation exits 50s before the next fires, so
REM invocations never overlap and the MultipleInstances IgnoreNew silent-skip
REM race (Cycles 7-8, 10) cannot occur. Single writer, by construction.

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\liquidation_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting liquidation stream capture... >> "%LOG_FILE%"
python -u -m engines.liquidation_collector --verbose 2 collect --duration 3550 >> "%LOG_FILE%" 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] Capture exited with %RC%. >> "%LOG_FILE%"
exit /b %RC%
