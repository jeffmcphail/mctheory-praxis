@echo off
REM Unlock-Universe Market Data Collector (Cycle 62A T3)
REM
REM Runs DAILY via Windows Task Scheduler. Collects circulating_supply and
REM total_supply for the ~25 unlock-bearing assets in
REM config/unlock_universe.json into the existing market_data table.
REM
REM WHY IT EXISTS: market_data held 427 rows across five mega-caps whose
REM supply moves by block subsidy, emission, burn and escrow -- no VC or team
REM vesting cliffs anywhere in it. Cycle 61 measured zero supply jumps
REM clearing 1% across that universe; the largest was 0.41%. Forced-trade
REM scenario F1 was unfalsifiable against it: not disconfirmed, unmeasurable.
REM
REM DOES NOT REPLACE market_data_collector_service.bat. That one keeps its
REM SUPPORTED_ASSETS scope and owns the btc_dominance global; this one is
REM purely additive and writes only unlock-universe rows. Widening
REM SUPPORTED_ASSETS instead would have widened the OHLCV and funding
REM services too, which the brief forbids.
REM
REM PACING: CoinGecko's free tier rate-limits aggressively and the budget is
REM shared across the whole run. --sleep 6.5 plus a 7-step backoff ladder is
REM tuned to outlast a depleted window rather than a transient blip; expect
REM this task to take several minutes and occasionally log 429 retries. That
REM is normal, not failure.
REM
REM The collector verifies both supply fields per asset against the same
REM response it stores from, and REFUSES to store an asset missing either --
REM a supply series with a synthesised zero in it is worse than a short one.

set PRAXIS_DIR=C:\Data\Development\Python\McTheoryApps\praxis
set VENV=%PRAXIS_DIR%\.venv\Scripts\activate.bat
set LOG_DIR=%PRAXIS_DIR%\logs
set LOG_FILE=%LOG_DIR%\unlock_market_data_collector.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PRAXIS_DIR%"
call "%VENV%"
set PYTHONUTF8=1

echo [%date% %time%] Starting unlock-universe market data collection... >> "%LOG_FILE%"
python -u -m engines.unlock_market_data_collector --verbose 2 collect >> "%LOG_FILE%" 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] Collection exited with %RC%. >> "%LOG_FILE%"
exit /b %RC%
