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

- **Cycle 18: Write `docs/SCHEMA_MIGRATION_PLAN.md` and start migrating
  second table per Rule 35.** Cycle 17 landed Rule 35 (Temporal data
  storage standard) and the `fear_greed` pilot migration. Cycle 18
  produces the ordered plan covering the remaining ~8 nonconforming
  tables, then begins the next migration. Decision points: which table
  goes second, dual-write vs. stop-migrate-start per table, ordering
  by risk/value. *(Source: BRIEF_temporal_standard_pilot.md
  out-of-scope list)*

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
- **Cycle 17 (this cycle)**: CLAUDE_CODE_RULES.md v1.4 -- new Rule 35
  (temporal data storage standard); pilot migration of `fear_greed`
  table to ms-since-epoch UTC PK schema; `onchain_btc` added to MCP
  `get_collector_health` monitoring (48h threshold, intentionally
  alarming until a collector is registered); `docs/SCHEMA_NOTES.md`
  created documenting all 17 tables across the 3 Praxis SQLite DBs
  with Rule 35 conformance status. Closes prior TODOs:
  "docs/SCHEMA_NOTES.md documenting timestamp heterogeneity" and
  "Add onchain_btc to MCP get_collector_health monitoring".

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
