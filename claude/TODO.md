# Praxis TODO

> **About this file**: Claude's Praxis-specific work queue. These are tasks
> Claude is responsible for executing or remembering across sessions.
>
> Pure-Praxis-domain content (trading rules, market findings, multi-phase
> roadmaps) lives in `docs/`, not here. This file is "what Claude has to
> do," not "what's true about Praxis."
>
> Maintained by `praxis_main_current`. Updated whenever a new TODO comes
> up, when an item completes, or when priority shifts.

---

## Active TODOs

Priority-grouped, then domain-grouped within each priority.

### High priority -- short and high-leverage

- **Cycle 25.5: position_snapshots Phase 5 cleanup.** Run after
  24-48h burn-in confirms the post-cutover dual-write writer is
  stable. Two-task cleanup:
  1. Modify `engines/smart_money.py` `cmd_snapshot` and `cmd_monitor`
     to single-write to `position_snapshots` only (drop the
     `_legacy` INSERT branch and the `_position_snapshots_pre_cutover`
     introspection helper). Also drop the `position_snapshots_v2`
     CREATE statement from `init_db()`.
  2. DROP TABLE `position_snapshots_legacy` AND DROP TABLE
     `position_snapshots_v2` (the empty v2 artifact left over from
     init_db's idempotent CREATE post-cutover).
  Plus standard cleanup: doc updates marking row #9 as DONE
  (no longer DONE-PARTIAL), retro at
  `claude/retros/RETRO_position_snapshots_phase5_cleanup.md`. Note
  the smart_money task is a 6h scheduled task (not long-lived), so
  the next scheduled invocation picks up the writer change
  automatically -- no kill-and-relaunch step needed.
  *(Source: Cycle 25 retro; deferred per Rule 35.6 Phase 5 pattern.)*

- **Cycle 26: Migrate `trades` per the dual-write recipe.** Largest
  remaining table (~6.5M rows). Already near-conforming -- `timestamp`
  is already INTEGER ms; primarily needs PK shape change (likely
  compound on `(asset, trade_id)`) and the `datetime` cosmetic fix
  (`Z` suffix -> `+00:00`). **Investigate before Brief**: is the
  trades collector long-lived (PraxisTradesCollector runs continuously
  per the migration plan) or scheduled? This affects the
  kill-and-relaunch decision per Cycle 24.1's process notes. If
  long-lived, plan the writer-restart step into Phase 0.
  *(Source: docs/SCHEMA_MIGRATION_PLAN.md row #10)*

- **Spike-DB (data/spike_scanner.db) reader audit.** Cycle 24's
  `cmd_export` in `engines/live_collector.py` divides ms back to
  seconds at export to preserve the spike DB's seconds contract.
  Audit other readers/writers of `spike_scanner.db.price_history`
  and decide whether to migrate that DB to ms in a future cycle.
  *(Source: Cycle 24 retro)*

- **`engines/live_collector.py` `yes_bid`/`yes_ask`/`spread` writer
  completion.** The `price_snapshots` schema reserves these columns
  but the writer only populates `yes_mid`. Pre-existing incomplete
  writer; preserved across Cycle 24's migration unchanged. Either
  populate them from `get_clob_prices` (already implemented but
  unused in `sample_all_markets`) or formally drop them from the
  schema in a future cycle.
  *(Source: Cycle 24 retro)*

- **Run `services/register_market_data_task.ps1` from elevated
  PowerShell** (one-shot admin step). Files in place; manual first-run
  has already seeded 3 rows for 2026-05-01. The scheduled task only
  needs registering so it picks up daily at 00:35 going forward.
  *(Source: Cycle 19 -- session lacked admin privileges)*

- **TRADING_ATLAS.md count reconciliation**: 15 numbered headings vs 17
  prose-claimed. Either find the two missing experiments or fix the prose.
  Atlas search would benefit from accuracy. ~30-45 min Code time including
  atlas_sync re-run. *(Source: Cycle 12 retro section 3 / Cycle 12 commit)*

- **Experiment 10 addendum parser fix**: Cycle 12 parser stops Exp 10 at
  the next `## ` divider, losing the "leverage runaway, not strategy
  failure" final conclusion. Two paths: (a) extend parser to fold orphan
  addendums by experiment-number reference, or (b) move addendum text into
  Exp 10's body in the markdown. (b) is much simpler and keeps
  markdown-as-source rule clean. ~30 min Code time. *(Source: Cycle 12
  retro section 7.1)*

### Mid priority -- structural improvements

- **`phase3_models.joblib` retrain**: funding_rates ground truth is back
  post-recovery. Auto-run phase2+3 with updated training window, hot-swap
  joblib. Brief writing is cheap; execution wall-time is whatever the
  trainer takes (10-60 min). Trigger condition for ongoing automation:
  when `phase3_models.joblib` is 6+ months old.

- **`engines/burgess.py` legacy cleanup**: `engines/burgess.py` (legacy)
  coexists with `src/praxis/models/burgess.py` (canonical, post-frequency-
  redo migration). Decide which version stays and remove the other.
  Decision-heavy -- needs human judgment on migration completeness.
  *(Source: Recovery Plan post_disk_failure section 1.2 item 6)*

- **Mojibake in `smart_money.log`**: emoji bytes getting mangled in log
  display. Doesn't affect the data in the DB, just human-readable log
  output. Cause is probably PYTHONUTF8 not propagating through cmd.exe.
  Small fix when convenient. *(Source: Cycle 13 reactivation; verified in
  morning health snapshot 2026-04-30)*

- **Atlas tools not surfacing in Claude.ai chat surface**: works in Claude
  Desktop's new chats only. Filed as known asymmetry, not blocking.
  Investigate when Anthropic ships Claude.ai chat updates that might fix
  the underlying tool-discovery mechanism.

### Lower priority -- meaningful but not urgent

- **Register scheduled collector for `onchain_btc` table**: Cycle 17
  added it to MCP `get_collector_health` (48h threshold) but no
  scheduled task is currently running. Latest data is from 2026-04-28
  (~2.7 days stale at the 48h threshold), so health alarms will fire
  until a collector lands. `engines/crypto_data_collector.py`
  `collect_onchain` exists but is not invoked by any registered
  scheduled task. Pattern would mirror the existing daily collector
  service registrations. *(Source: BRIEF_temporal_standard_pilot.md
  Task 3 deferred follow-up)*

- **VR profile experimental framework**: Design framework for using VR
  profile info (multi-timescale mean-reversion) to inform entry/exit
  signals, trade time horizons, and regime detection. Conceptual work
  before any code.

- **Funding rate flip-positive alert**: Alert when crypto funding rates
  flip positive and sustain (bull phase returning) -- funding monitor
  will show active signals. Check periodically.

- **5-minute BTC momentum strategy ("4-minute rule")**: Watch 4 mins of
  5-min Polymarket BTC markets, bet continuation in final minute. Evolve
  trigger threshold with Kalman-like filter over time (3.8 min may pay
  better than 4.0 min). Easily backtestable with collected data. High-
  frequency, small edge, many trades.

- **Convergence speed detector**: Build using live_collector data
  (now 30k+ snapshots and growing). Compute rolling volatility -- when
  vol drops sharply while price trends toward 0 or 1, that's convergence.
  Use as entry/exit trigger.

- **Crypto prediction ensemble Corvino enhancements**: (1) Add LightGBM
  as 3rd model alongside XGBoost+LSTM, (2) Collect liquidation data from
  Binance futures API, (3) Build ATR-based position management module --
  TP1=1.5xATR, TP2=3xATR, SL=2xATR, move SL to breakeven after TP1 hit.
  Key insight: "management matters more than entry."

- **AI ensemble probability engine**: Use AI Agent Factory's QPT +
  ProviderBridge. Multi-LLM consensus (Anthropic, Groq, DeepSeek) on
  market questions, compare to Polymarket prices, flag divergences >15%.
  Use QPT evolution to optimize prompt loadings. Add: FinTral
  (Mistral-7B financial multimodal, runs locally, free inference) as
  3rd+ provider for real consensus. Also explore FinGPT-Forecaster for
  granular per-asset sentiment scores beyond Fear & Greed Index.

- **Polymarket market making bot**: Provide liquidity on both sides of
  markets, earn the spread (1-3%/month = 13-45% APR). Needs inventory
  management, news event detection, spread widening on volatility.
  "Boring money printer" -- keeps capital working instead of sitting
  idle.

- **Actuarial engine for Praxis prediction market trading**: Compares
  holding off on graduated-scale trades (e.g. sports seasons with games
  remaining) against alternative short-term opportunities from the
  scanner. Must verify a better capital deployment exists before
  deferring entry. Start with sports leagues, expand to other trade
  types. Build after examining remaining trade types.

- **Cross-asset features for crypto LSTM**: Add XAU/USD daily as macro
  feature -- structurally tied to BTC via real-rates and de-dollarization
  tailwind. Also add DXY (USD index) and 10Y TIPS real yields. Use
  yfinance (free). Optional: CFTC COT report for gold futures positioning
  (free, weekly, public, real institutional positioning data). Cheap data
  additions when LSTM training day arrives. Apply Hurst multi-timescale
  framework to gold same as BTC.

- **MCP wrappers for Alpaca and QuantConnect**: Jeff has accounts on both.
  Alpaca = live trading/data, QuantConnect = backtesting/multi-asset
  historical. Note: smart_money.db MCP coverage landed in Cycle 14;
  Atlas MCP tools landed in Cycle 12 -- only Alpaca + QuantConnect remain
  pending from the original "MCP extensions" plan.

- **Becker Polymarket dataset (33GB) ingestion**: Tier 2 data import.
  Big lift, scope a Brief when ready.

- **UTC-aware funding scheduler refactor**: Only relevant if/when the
  Toronto-vs-UTC offset starts mattering operationally. Currently the
  17h threshold (Cycle 14) absorbs the offset cleanly. Defer.

### Goals / long-term action items

- **Sign up for Kraken Breakout prop trading platform**: Pay for trial
  test, pass it, trade with their capital at 90% profit share. Build
  toward this as a goal.

---

## State / context (not active TODOs but useful for sessions)

### LSTM + Quantamental crypto prediction system: BUILT

- `engines/lstm_predictor.py` (1069 lines)
- LSTM features: 7 features x 60 timesteps -- close, high, low, volume,
  FearGreed, funding, Hurst
- XGBoost quantamental: ~40 features incl. multi-timescale Hurst from
  Volterra framework
- Data collected (2.5 years): 1800 daily OHLCV, 2160 4h candles, 900
  Fear & Greed, 2190 funding rates, 365 on-chain. Post-recovery counts
  may differ; verify before training run.
- **Pending work**: training run, walk-forward backtest validation, apply
  to Polymarket crypto markets. (Now unblocked since funding_rates ground
  truth is restored post-recovery.)

---

## Reading queue

Books / papers to absorb in priority order. References to specific
techniques in this list show up across multiple TODOs.

1. **Lopez de Prado, Machine Learning for Asset Managers (MLAM, ~150pp)**:
   deflated Sharpe ratio, combinatorial purged CV, meta-labeling,
   triple-barrier method (intrabar confluence v8.1 used this). Highest
   yield-per-page.
2. **Carver, Systematic Trading + Leveraged Trading**: multi-strategy
   framework, diversification multiplier math is missing piece for
   Manager Agent Kelly allocation. Carver's Substack "Investment
   Idiocy" = Tier 2 allowlist.

---

## Recently closed (last 30 days)

For context on what just shipped, see `docs/praxis_main_series_transcript.md`
and the recent `claude/retros/RETRO_*.md` files.

Highlights of the recovery + post-recovery sequence (2026-04-29 / 30):

- Cycle 8: OrderBook --duration race fix
- Cycles 9-10: Historical data backfill (6 tables) + 4 new scheduled
  collectors registered + trades audit
- Cycle 11: MCP get_collector_health expansion (post-Cycle-10 awareness)
- Cycle 12: Atlas DB v0.1 + 2 new MCP tools (atlas_search, atlas_get)
- Cycle 13: CLAUDE_CODE_RULES.md v1.2 + Live/SmartMoney reactivation
- Cycle 14: MCP get_collector_health sidecar DB monitoring +
  funding_rates threshold widening (9h -> 17h)
- Cycle 15: CLAUDE_CODE_RULES.md v1.3 -- new Rule 34 (SQLite read
  transaction management)
- Cycle 16: Meta-docs convention setup, memory cleanup, claude/TODO.md
  plus META_DOCS.md and docs/TRADING_*.md created.
- Cycle 17: CLAUDE_CODE_RULES.md v1.4 -- new Rule 35
  (temporal data storage standard); pilot migration of `fear_greed`
  table to ms-since-epoch UTC PK schema; `onchain_btc` added to MCP
  `get_collector_health` monitoring (48h threshold, intentionally
  alarming until a collector is registered); `docs/SCHEMA_NOTES.md`
  created documenting all 17 tables across the 3 Praxis SQLite DBs
  with Rule 35 conformance status.
- Cycle 18: `docs/SCHEMA_MIGRATION_PLAN.md` written
  (ordered roadmap for the remaining ~8 nonconforming tables, with
  pattern annotations and per-cycle execution log); `ohlcv_daily`
  migrated as the second table to Rule 35 (compound PK on
  `(asset, timestamp)`, timestamp seconds -> ms; 1,802 rows preserved,
  latest UTC delta 0s); writer in `engines/crypto_data_collector.py`
  updated; reader in `engines/lstm_predictor.py:68` verified
  unchanged; MCP `get_collector_health` autodetect heuristic confirmed
  working across the now-mixed (ms + seconds) tables.
- Cycle 19: `market_data` migrated to Rule 35
  (compound PK on `(asset, timestamp)`, ms timestamp, no `id`);
  `collect_market_data` rewritten to fetch `/global` once per cycle
  and populate the previously-unfilled `btc_dominance` column;
  `collect-market-data` CLI subcommand added (parser + dispatch +
  `cmd_collect_market_data` handler); `services/market_data_collector_service.bat`
  and `services/register_market_data_task.ps1` created (CRLF); MCP
  `get_collector_health` extended to monitor `market_data` (25h
  threshold); `docs/SCHEMA_NOTES.md` updated to mark `market_data`
  CONFORMING; manual first-run seeded 3 rows (BTC, ETH, SOL) for
  2026-05-01 with dominance 58.47%. **Outstanding**: scheduled-task
  registration needs an elevated PowerShell -- see Active TODOs.
- Cycle 20: `ohlcv_4h` migrated to Rule 35
  (compound PK on `(asset, timestamp)`, timestamp seconds -> ms,
  datetime rewritten naive -> ISO `+00:00`; 10,830 rows preserved
  with latest UTC delta 0s); writer in
  `engines/crypto_data_collector.py` `collect_ohlcv_4h` updated
  (init_db schema + INSERT path); MCP `get_collector_health`
  autodetect verified for the now-mixed ms/seconds tables;
  `docs/SCHEMA_NOTES.md` + `docs/SCHEMA_MIGRATION_PLAN.md` updated.
  Note: prior plan-doc note that ohlcv_4h.datetime was already
  `+00:00` was empirically wrong (it was naive); migration re-derived
  datetime from `timestamp` for defense in depth.
- **Cycle 25**: `smart_money.position_snapshots` migrated to
  Rule 35 via the **third use of the dual-write recipe** (Phases
  0-4; Phase 5 cleanup deferred to Cycle 25.5 after 24-48h
  burn-in). 68,812 rows preserved at cutover; compound PK on
  `(snapshot_id, wallet, market_slug, outcome)` (the natural key,
  promoted from a UNIQUE constraint), dropped `id`. **First
  schema-shape migration in the recipe**: the legacy `timestamp
  TEXT` (microsecond ISO) was renamed to `datetime`, and a NEW
  `timestamp INTEGER` ms column was added derived via
  julianday/ROUND. Phase 2 backfill: 65,376 rows via pure-SQL
  INSERT-SELECT in 0.273s wall-clock (Brief budgeted "well under
  5s"). Phase 4 atomic cutover: 9ms wall-clock. **ZERO reader
  fixes required** -- cross-engine grep confirmed every reader
  keys on `snapshot_id`, never on `timestamp` (huge simplification
  vs Cycle 24's 4 reader fixes). Both writer sites
  (`cmd_snapshot` L335-379 and `cmd_monitor` L681-712) refactored
  through a shared `_insert_position_pair` helper using runtime PK
  introspection (`_position_snapshots_pre_cutover`). MCP
  `SIDECAR_DBS["smart_money"]["position_snapshots"]
  ["timestamp_format"]` flipped `"iso_text"` -> `"ms"` in same
  Phase 4 commit; schema comment block at server.py:74-79 updated.
  PraxisSmartMoney is a 6h scheduled task (not long-lived), so the
  next scheduled invocation picked up the new code automatically;
  no kill-and-relaunch step. **New recipe nuance documented**:
  ROUND of microsecond-precision floats in SQLite produces ~50%
  rate of +1ms drift vs Python's `int(... .timestamp() * 1000)`
  TRUNC convention (whenever microsecond fraction is >= 500us);
  drift is harmless for this table where readers key on snapshot_id;
  verify script tolerates +/-1ms on Check 5. Per Cycle 24.1 process
  notes, AC #17 included a HARD-restart-protocol live-MCP exercise
  step (Chat pastes get_collector_health post-restart). Phase 0
  commit: `36fb44a`. Phases 2-4 commit: `874bf81`.
- **Cycle 24.1 (2026-05-05)**: `_to_latest_ms` ms-format sidecar
  hotfix -- closed as **retro-only; no code change required**.
  Brief hypothesized a missing `/1000` divide (or missing `"ms"`
  branch) inside the sidecar staleness helper, producing a
  year-58000 OSError(22) on Windows reachable through
  `get_collector_health.databases.live_collector`. Diagnosis: the
  on-disk `_to_latest_ms` at `servers/praxis_mcp/tools/meta.py:
  481-544` already had a correct `"ms"` branch returning
  `int(latest)` directly (no `datetime.fromtimestamp` call on
  numeric input, so the OSError path is unreachable from on-disk
  code). Real root cause: a stale FastMCP stdio subprocess holding
  the pre-Cycle-24 `SIDECAR_DBS` config and module state in memory;
  the subprocess survived a normal close-and-reopen of Claude
  Desktop and was only fully cleared by ending the python.exe MCP
  children via Task Manager (a "hard" restart). Post-hard-restart
  verification at 12:20 UTC: `price_snapshots` reports
  `row_count=374,755`, `latest=2026-05-05T12:19:48.522Z`,
  `is_stale=false`, `staleness_seconds=37`. Two durable process
  notes captured in the retro for Cycles 25-26: (a) future
  dual-write Briefs must include a live-MCP exercise AC where Chat
  pastes the post-restart `get_collector_health` response, since
  Cycle 24's AC #20 was claimed-but-not-actually-verified by the
  same tautology pattern; (b) "restart Claude Desktop" is
  ambiguous -- replace with "hard restart via Task Manager / End
  Process on python.exe MCP subprocesses" or, at minimum, verify
  via `Get-Process python` that the subprocesses are gone before
  re-exercising. Retro at
  `claude/retros/RETRO_to_latest_ms_hotfix.md`.
- **Cycle 24**: `live_collector.price_snapshots`
  migrated to Rule 35 via the **second use of the dual-write recipe**
  established in Cycle 23 (Phases 0-4; Phase 5 cleanup deferred to
  Cycle 24.5 after 24-48h burn-in). 358,715 rows preserved at cutover;
  compound PK on `(slug, timestamp)`, dropped `id`, timestamp seconds
  -> ms via clean `legacy_ts * 1000` multiply (no julianday/ROUND --
  legacy data had no sub-second precision to recover). NEW `datetime
  TEXT` column derived from `timestamp` via SQLite
  `strftime('%Y-%m-%dT%H:%M:%S+00:00', timestamp, 'unixepoch')` for
  backfill. Phase 2 backfill: 358,661 rows via pure-SQL INSERT-SELECT
  in 2.243s wall-clock (Brief budgeted ~30s; integer multiply beats
  Cycle 23's julianday/ROUND by ~3x even with 4x rows). Phase 4
  atomic cutover: 4ms wall-clock. **Long-lived-process gotcha**:
  PraxisLiveCollector launches python once and runs indefinitely;
  file changes do NOT auto-pick-up (different from
  PraxisOrderBookCollector's hourly restart pattern). Phase 0 commit
  was paired with an explicit kill-and-relaunch step. Three Brief-
  named reader fixes shipped atomically with the writer change in
  Phase 0 (`check_for_spikes` in-process, `mev_executor.py` window
  query, stats display); a fourth (`dashboards/data_collector.py`)
  surfaced during the cross-engine audit and was added to the same
  commit. Plus a fifth touch point caught at Phase 4: MCP
  `SIDECAR_DBS.price_snapshots.timestamp_format` was hardcoded `"s"`
  (Brief implied autodetect would handle it); changed to `"ms"` in
  the cutover commit. Pre-existing incomplete writer
  (`yes_bid`/`yes_ask`/`spread` reserved but unwritten) preserved as
  out-of-scope; deferred TODO. Spike-DB export divides ms back to
  seconds at write time to preserve `spike_scanner.db`'s seconds
  contract (audit deferred). Phase 0 commit: `b8fa847`. Phases 2-4
  commit: `6ca1796`.
- **Cycle 24.5**: `live_collector.price_snapshots` Phase 5 cleanup
  (commit `<CYCLE_24_5_HASH>`). Dropped `_legacy` (448,941 rows) +
  empty `_v2` stub via
  `scripts/migrations/cycle24_5_price_snapshots_cleanup.py`
  (legacy/live ratio at drop = 99.25%; `_v2` had been empty since
  Cycle 24's cutover). Collapsed `sample_all_markets` to
  single-write (removed runtime PK introspection, dual-INSERT
  branch, and the `_v2` CREATE block in `init_db()`). Row #8 of
  `SCHEMA_MIGRATION_PLAN.md` flipped DONE-PARTIAL -> DONE.
  Post-cleanup live-MCP state: row_count=452,387, staleness=5.5s,
  `is_stale=false`; `live_collector.unmonitored` now contains only
  `["collection_log", "spike_alerts", "tracked_markets"]` (legacy
  and v2 gone). **First cycle to apply the corrected ordering**
  surfaced by the Cycle 23.5 retro: writer-collapse-FIRST, then
  kill the long-lived PraxisLiveCollector process, then run the
  cleanup script. The cleanup script's pre-flight #4 (refuses to
  drop `_legacy` if it was written within 60s) is the load-bearing
  prevention against the Cycle 23.5 cascade pattern; legacy's last
  write at run time was 260s old, well past the threshold. Second
  hybrid-workflow cycle.
- Cycle 23.5: `order_book_snapshots` Phase 5 cleanup (commit
  `c21a679`). Dropped `_legacy` (104,776 rows) +
  empty `_v2` stub via `scripts/migrations/cycle23_5_order_book_cleanup.py`
  (legacy/live ratio at drop = 99.99%). Collapsed
  `collect_order_book_snapshot` to single-write (removed runtime
  PK introspection, dual-INSERT branch, and the `_v2` CREATE
  block in `init_db()`). Row #7 of `SCHEMA_MIGRATION_PLAN.md`
  flipped from DONE-PARTIAL to DONE; first hybrid-workflow cycle
  (Claude drafted the cleanup script + writer-collapse brief,
  Code applied the on-disk edit).
- Cycle 23: `order_book_snapshots` migrated to
  Rule 35 via the **first dual-write pilot** in the migration program
  (Phases 0-4; Phase 5 cleanup deferred to Cycle 23.5 after 24-48h
  burn-in). 88,894 rows preserved (BTC + ETH); compound PK on
  `(asset, timestamp)`, dropped `id`, timestamp seconds -> ms with
  full sub-second precision recovered from the existing `datetime`
  column (which had always carried microseconds even though the old
  ts column truncated to seconds). Phase 2 backfill: 87,668 rows
  via pure-SQL INSERT-SELECT in 7.219s; required a follow-up
  ROUND-correction UPDATE on 43,596 off-by-1ms rows
  (SQLite's `julianday * 86400000` lands ~1 ULP below the integer
  for half of .NNN-precision datetimes; CAST AS INTEGER truncates,
  ROUND fixes). Phase 4 atomic cutover: 5ms wall-clock.
  Two pre-existing MCP tool bugs silently fixed by the migration:
  `get_order_book_range` (returned 0 rows for any sane ms input
  pre-Cycle-23) and `get_order_book_snapshot` (ABS math was
  unit-mismatched). Mid-cycle gotcha: cutover RENAME pair invalidated
  the writer's hardcoded `_v2` table reference, breaking dual-write
  for ~5 min until the writer was retrofitted with runtime PK-shape
  introspection. Documented as a load-bearing lesson in the new
  "Dual-write recipe" section of `docs/SCHEMA_MIGRATION_PLAN.md`,
  which becomes the durable template for Cycles 24-26.
- Cycle 22: `ohlcv_1m` migrated to Rule 35
  (compound PK on `(asset, timestamp)`, timestamp seconds -> ms,
  datetime rewritten naive -> ISO `+00:00`; 530,836 rows preserved
  with latest UTC delta 0s); writer in
  `engines/crypto_data_collector.py` `collect_ohlcv_1m` updated
  (init_db schema + INSERT path); MCP `ohlcv.py` docstring updated
  to specify ms units (no body change). **First non-cosmetic reader
  fix in the migration program**: `engines/intrabar_predictor.py`
  line 110 `bar_seconds = bar_minutes * 60 -> bar_minutes * 60 *
  1000` -- pre-fix the bar-bucketing arithmetic silently returned
  zero aggregated bars for any `bar_minutes >= 2` post-migration.
  Verified empirically post-fix: `bar_minutes=5` returns aggregated
  bars at exact 5-min boundaries. Cycle 21.5's writer-alignment-
  audit prescription caught this pre-merge. Performance datapoint:
  530,836-row INSERT-SELECT completed in 0.567s wall-clock (Brief
  budgeted 5-30s). Writer-alignment audit confirmed for kline
  endpoints: Binance kline `openTime` is bar-aligned by contract
  (no jitter); only event-driven endpoints like
  `fetch_funding_rate_history` carry reporting jitter requiring
  writer-side truncation. Durable result for future cycles
  touching kline data.
- Cycle 21.5: funding_rates writer alignment hotfix.
  Caught during post-Cycle-21 independent verification: writer was
  preserving Binance's sub-second jitter (e.g., 1777795200003 vs
  migration's 1777795200000), accumulating duplicate rows for each
  hourly event. Fixed via 2-task hotfix: writer now truncates to
  seconds-aligned ms before storage; deduplication script collapsed
  26 existing dupes (lossless -- duplicate pairs had identical
  funding_rate values). Cross-table sanity check confirmed the bug
  pattern is isolated to funding_rates (fear_greed, ohlcv_daily,
  ohlcv_4h, market_data all show 0 dupes; their writers feed off
  bar-aligned openTime). Future migrations should sanity-check
  post-cycle row growth against expected cadence to catch this class
  of bug earlier.
- Cycle 21: `funding_rates` migrated to Rule 35
  (compound PK on `(asset, timestamp)`, timestamp seconds -> ms,
  datetime rewritten naive -> ISO `+00:00`; 2,212 rows preserved with
  latest UTC delta 0s); writer in `engines/crypto_data_collector.py`
  `collect_funding_rates` updated (init_db schema + INSERT path); MCP
  `funding.py` comment header refreshed (the autodetect-aware reader
  needed zero logic changes); cross-engine SQL audit verified no engine
  has hardcoded seconds-since-epoch SQL filters against `funding_rates`
  -- phase3 model retrain unblocked from this migration's perspective
  (the standalone retrain TODO remains tracked separately under Mid
  priority); Cycle 14 17h staleness threshold confirmed valid
  post-migration via `get_collector_health` (~11h staleness, is_stale
  false); `docs/SCHEMA_NOTES.md` + `docs/SCHEMA_MIGRATION_PLAN.md`
  updated. Process note: per the Brief, Mode B requires a Brief from
  Chat every cycle -- Cycle 20's "Mode B-lite" self-dispatch was the
  exception, not a new pattern. This cycle ran from a proper Brief.

---

## Maintenance protocol

- **Adding a TODO**: Drop it under the appropriate priority bucket. Keep
  prose terse. Reference source (which retro / which session / which
  Brief surfaced it) so future sessions can find context.
- **Closing a TODO**: Move to "Recently closed" with a one-line summary
  and the cycle/commit hash that did it. Prune from active TODOs.
- **Reordering**: Acceptable any time; the priority buckets are
  approximate, not contractual.
- **Renaming**: Don't rename established TODOs unless the work
  fundamentally changed. Use sub-bullets if a TODO grew multiple parts.
