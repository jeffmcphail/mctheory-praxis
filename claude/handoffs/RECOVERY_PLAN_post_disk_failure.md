# Praxis Recovery Plan — Post-Disk-Failure Build & Data Queue

> **Document type:** Strategic roadmap, NOT a Brief. Individual Briefs for each cycle are drafted separately and live alongside this document in `claude/handoffs/`. This file is the source of "what's queued and in what order"; each cycle gets its own dedicated Brief written immediately before that cycle's Code session, following the standard Brief contract (single Cycle number, Mode A/B/C, acceptance criteria, kill switch, estimated scope). Treat §5 as itinerary, not as cycle Briefs themselves.

**Source:** Synthesized from praxis_main_current chat session of 2026-04-23 to 2026-04-28, after the disk-failure event of ~2026-04-24.
**Status:** Recovery commits `40ec923` + `ad77064` pushed; venv operational; `data/*.db`, `phase3_models.joblib`, and `.env` permanently lost.
**Cycle counter at crash:** 7 (Cycle 7 MCP health-signal fix delivered; Cycle 8 Atlas DB Brief written but not yet executed).
**Authoring conventions:** Atlas DB v0.1 (Cycle 8) will turn this kind of doc into a queryable artifact; until then this lives as markdown.

---

## 0. The Pivot Point

The "last pre-crash message" was the wrap-up of Cycle 7 — delivery of `praxis_mcp_health_signal_fix.zip` containing the patched `servers/praxis_mcp/tools/meta.py` (per-table cadence thresholds, orphan-table acknowledgment) and matching README. That message ended `Cycle counter: 7`.

Everything captured below was generated **after** that message, before the crash, and the disk failure prevented it from becoming files-on-disk artifacts. This document recovers the substance.

---

## 1. The Brief / Cycle State at Crash Time

### 1.1 Briefs that exist in repo and were ready or in flight

| Cycle | Brief | Status at crash | Action |
|---|---|---|---|
| 7 | (none — the meta.py patch was Mode A inline) | ✅ Delivered to user, applied to repo, awaiting Desktop relaunch verification | Verify when collectors are back online |
| 8 | `BRIEF_atlas_db_v0_1.md` | ✅ Written, delivered as delta zip, **never executed by Code** | **PENDING — Code execution next session** |
| (parked) | `BRIEF_polymarket_fee_structure_PARKED.md` | Parked — fee investigation deferred | Hold until Polymarket strategy work resumes |

The Atlas DB Brief (Cycle 8) is the **single highest-priority pending Code action.** It is fully written, internally consistent, and ready to hand off. The crash interrupted only the execution, not the planning.

### 1.2 Briefs that needed to be written but were not

These were post-Cycle-7 follow-up Briefs the chat had identified but not yet drafted. Each needs to be written before Code can execute. Listed in the priority order established in the chat:

1. **OrderBook `--duration` tweak (Cycle 9 candidate).** Mode A, one-line edit to `services/order_book_collector_service.bat` changing `--duration 3600` to `--duration 3550` so the current invocation exits before the next hourly fire. Restores continuous (10s cadence) microstructure coverage. Was the explicitly queued #2 priority after the MCP health signal fix.

2. **Funding/sentiment scheduled collectors.** Decision was pending: do `funding_rates` and `fear_greed` need scheduled tasks, or remove them from health monitoring? Chat recommendation was **"schedule both"** because (a) the LSTM quantamental model needs `fear_greed` as a feature (memory item on cross-asset features), (b) live funding carry execution needs current `funding_rates`, (c) both are latent prerequisites for items 1, 2 of the strategy queue. Brief would register two new scheduled tasks: `collect-funding` (8h cadence) and `collect-fear-greed` (daily cadence).

3. **`/register-collector-task` slash command.** Idea raised when discussing the howtoprofitai article on skill-file workflow. Since funding + sentiment task registration is structurally identical work and the .ps1 + .bat + ASCII-check + admin-shell dance is the same every time, a Claude Code slash command would canonicalize it. Becomes one-liner Code execution rather than full Brief + Retro.

4. **MCP server v0.2 — extend to `smart_money.db`.** Memory item #28 captures it: 60-min Brief, wraps the existing `engines/smart_money.py` and `smart_money_alerts.py` (already committed) behind MCP tools. Architecture identical to Cycle 7's `crypto_data.db` MCP wrapping.

5. **OrderBook + Crypto1m fee structure investigation (Polymarket).** Parked Brief covers this for Polymarket fees specifically; the broader question is whether the dynamic taker-fee structure (~3.15% on 50¢ contracts) kills latency-arb on 5-min markets too, or only 15-min. 30-minute Code Brief.

6. **`engines/burgess.py` legacy cleanup.** Legacy engine file coexists with `src/praxis/models/burgess.py` canonical (post-frequency-redo migration). `scripts/run_burgess.py` and `scripts/run_chan_cpo.py` still import from legacy location; tests and battle scripts use new. Documented as known issue in recovery commit message. Deferred but should be cleaned up as a future cycle to avoid confusion.

---

## 2. Strategic Build Queue (Post-Recovery)

Captured from the chat's strategic discussions during the article-triage rounds and the long-arc Praxis vision conversation. Organized by category, with prerequisites flagged.

### 2.1 Atlas DB Architecture (Cycle 8 — already-written brief, plus Cycle 9 follow-on)

The pending Cycle 8 Brief delivers:
- `data/praxis_meta.db` sidecar SQLite with structured rows for ~25 Atlas experiments
- Embedding vectors per experiment for semantic similarity search (Voyage 3-lite preferred, OpenAI ada-002 fallback)
- `engines/atlas_sync.py` migration tool — markdown is source of truth, DB is derived + augmented
- New MCP tools `atlas_search` and `atlas_get` for queryable Atlas access from Chat
- `docs/ATLAS_DB.md` documenting the workflow

**This is the bidirectional sync architecture you specified:** edit md → run `python -m engines.atlas_sync` → review printed diff → commit both files together. No git pre-commit hook (premature lock-in).

**Prerequisite for:** the Discovery Agent (semantic similarity over experiments) and for the workflow rule "Atlas DB substitutes for inline historical context in chat handovers" (memory #25 update).

#### 2.1.1 Cycle 9 Schema Augmentation: methodology-aware Atlas

The original Cycle 8 Brief's `result_class` field with values `POSITIVE | NEGATIVE | INCONCLUSIVE | PARTIAL` is the wrong abstraction in isolation. **A negative result is conditional on the methodology under which it was tested**, and as the methodology evolves (event-bar sampling, meta-labeling, triple-barrier labeling, deflated Sharpe correction, regime-stratified validation), previously-failed strategies become conditionally-revisitable rather than absolutely dead.

**Cycle 9 (or as a Cycle 8.5 augmentation if scope allows) adds three structured fields to `atlas_experiments`:**

```sql
-- Methodology fingerprint: structured tags describing how the result was produced.
-- Used to identify which negative results become candidates for re-testing under new methodology.
methodology_fingerprint TEXT,    -- JSON: {"bar_type": "time|volume|dollar|imbalance",
                                 --        "validation_scheme": "k_fold|walk_forward|cpcv",
                                 --        "labeling_method": "raw|meta_label|triple_barrier",
                                 --        "sharpe_correction": "none|deflated|probabilistic",
                                 --        "regime_stratification": "none|single|multiple",
                                 --        "feature_scope": "ohlcv_only|with_microstructure|with_cross_asset"}

-- Revisit triggers: list of methodology changes that would warrant re-testing this result.
-- Mostly applies to NEGATIVE entries; POSITIVE entries can use it for "verify under stricter test."
revisit_when TEXT,               -- JSON array of trigger names, e.g.
                                 -- ["event_bar_partition_built", "meta_labeling_implemented",
                                 --  "deflated_sharpe_correction_applied"]

-- Last methodology audit: when this entry was last reviewed against current methodology.
-- Drives the "stale negative results that should be re-evaluated" Discovery Agent query.
last_methodology_audit_at TEXT,  -- ISO timestamp
```

**The query pattern this enables** is the operational point: when the methodology evolves (e.g., event-bar partition lands), the Atlas DB is queryable as "show me all NEGATIVE results whose `revisit_when` field includes 'event_bar_partition_built'." That's how negative results become **productive infrastructure** rather than dead weight. A confirmed-failed result with proper methodology tagging is reusable; a confirmed-failed result with no methodology context is silent skipped work that no future advance can claim against.

**Companion principle:** `engines/atlas_sync.py` parser must extract these fields from the markdown source. The markdown convention for `revisit_when` is a structured tag block at the bottom of each experiment entry; `engines/atlas_sync.py` recognizes the syntax and populates the DB column. This is a small parser extension on top of the original Cycle 8 spec.

**Migration of existing entries.** When Cycle 8 ships, all existing experiments inherit a default `methodology_fingerprint` reflecting how the project actually evaluated them at the time (mostly time-bar / k-fold / raw labels / no-Sharpe-correction / no-regime-stratification / OHLCV-only). Cycle 9 then walks the existing entries and tags each NEGATIVE entry with appropriate `revisit_when` triggers based on the explicit list in §3.4. This is a one-time migration, ~30 minutes of human review.

### 2.2 Crypto Prediction System (LSTM + Quantamental)

Memory items capture this as a queue item with substantial complexity:
- **LSTM timeseries model** (`engines/lstm_predictor.py` exists, 1,669 lines, currently pure-numpy not PyTorch)
- **Quantamental model** combining fundamental/sentiment/news/on-chain/whale data
- **Combine for crypto direction** with probability/confidence/horizon outputs
- **Apply to Polymarket crypto markets** (buckets, directional bets)

**Architectural decisions captured in the chat:**

1. **Cross-asset features for the LSTM** (gold-trading discussion, memory item):
   - **XAU/USD daily** (gold) — structurally tied to BTC via real-rates and de-dollarization tailwind
   - **DXY** (USD index)
   - **10Y TIPS real yields**
   - **CFTC COT report** for gold futures positioning (free, weekly)
   - All free via yfinance. Apply Hurst multi-timescale framework to gold the same way as BTC.

2. **Add LightGBM as third ensemble model** alongside XGBoost + LSTM (Corvino-inspired).

3. **Liquidation data from Binance futures API** as additional feature.

4. **ATR-based position management module** — TP1=1.5×ATR, TP2=3×ATR, SL=2×ATR, move SL to breakeven after TP1 hit. Key insight: "management matters more than entry."

**Prerequisite:** crypto_data.db rebuilt with sufficient history (see §3 data priorities).

### 2.3 AI Ensemble Probability Engine

Multi-LLM consensus on Polymarket market questions vs implied prices, flag divergences >15%, use QPT evolution to optimize prompt loadings for probability assessment accuracy.

**Discoveries from chat that update this:**
- **FinTral** (Mistral-7B financial multimodal, runs locally, free inference) as 3rd+ provider for real consensus.
- **FinGPT-Forecaster** for granular per-asset sentiment scores beyond Fear & Greed Index.
- **Karlsson Polymarket retro pattern** — persistent learning across sessions via lessons extracted to memory. Worth borrowing the persistence-of-strategy-scorecard pattern rather than the agent code itself.

**Prerequisite for first useful run:** validation against historical Polymarket data. **Becker dataset (33GB) unblocks this** — see §3.

### 2.4 Convergence Speed Detector for Polymarket

Memory item #26 captures it. 213k+ Polymarket 60-sec snapshots from PraxisLiveCollector were lost with the disk, but Becker's dataset replaces them and then some. Detect convergence via volatility collapse as price trends toward 0 or 1 near resolution. Use as entry/exit trigger.

**Connection from chat:** this is structurally the **Polymarket-native version of buffer-ETF gamma hedging** discussed during the leveraged-ETF article. Same mathematical structure (nonlinear delta as price approaches threshold), different market. Worth tagging that explicitly when implementing — the cross-asset analogy strengthens the framing.

**Connection from chat:** also touches the **credit-risk modeling literature** angle from the Iriarte Cabrera article triage. Polymarket binary markets resolving on event outcomes share mathematical structure with credit-default models (KMV/Merton, default cascades, recovery rates) more cleanly than they share with traditional asset-price models. Non-obvious neighbor; worth mining when this gets built.

### 2.5 Polymarket Market-Making Bot (memory #15)

`engines/market_maker.py` exists in repo, 1,257 lines, was already substantial pre-crash. Provides liquidity both sides, earns spread (1-3%/month target), needs inventory management + news-event detection + spread widening on volatility.

**Critical update from chat — Polymarket fee structure findings:**
- Fee formula: `fee = shares × feeRate × price × (1 − price)`, where feeRate = 0.072 for crypto markets
- Fee maximized at price = 0.50 (where it equals ~1.8% of contract value)
- **GTC (maker) orders pay no fee** — this is the architecturally-favored strategy
- **Sum-of-prices arb on near-50/50 markets needs spread > ~3.5-4% to clear taker fees both sides** — which is exactly why Polymarket killed that strategy with this fee model

**Implication:** maker-only strategy is the only architecturally-favored Polymarket play for retail. The market-making bot is exactly that.

**Sub-finding from chat:** Polymarket BTC 5-min markets have a deterministic slug pattern `btc-updown-5m-{window_ts}` where `window_ts = now - (now % 300)`. Saves API calls, more reliable.

### 2.6 5-min BTC Momentum Strategy ("4-minute rule") — Methodology-Conditionally Open

**Status update from chat — original time-bar formulation didn't survive validation.** The Liu paper post-mortem provided independent confirmation that 5-min BTC binary options approximate a random walk **when sampled on time bars with conventional methodology**. Liu's v3 with 10-min trend filter and rebalanced momentum weights still produced 25-27% win rate vs ~53% breakeven.

**That is a time-bar / conventional-methodology finding, not a structural-no-edge finding.** The "directional momentum at 5-min horizon doesn't exist" claim presumes the test methodology was capable of detecting one. Several methodology changes from the López de Prado discipline (§3.6) plausibly invalidate the negative result and warrant re-testing:

1. **Information-bar partition.** Resample on volume bars / dollar bars / imbalance bars rather than time bars. Time bars systematically violate IID by sampling during low-information periods and undersampling bursts. The 5-min horizon expressed in volume-bar equivalents may surface signal that time-bar slicing washed out.
2. **Triple-barrier labeling with volatility-adjusted geometry** rather than simple win/loss-at-bar-close. Stop and profit barriers tuned to local realized volatility produce different labels than fixed-horizon close prices, and the resulting target distribution is genuinely different.
3. **Meta-labeling: "should I bet" separately from "which way."** Often a much easier prediction. A directional model with 51% accuracy plus a bet-quality filter that abstains on low-confidence cases can produce a much higher conditional accuracy on selected trades.
4. **Deflated Sharpe correction** to ensure observed underperformance isn't itself a multi-test artifact. The original v8.1 work tested ~25 parameter variants; the inverse correction applies to negative findings too — what looked like clear failure may have been a deflation-floor effect.

**Convergence/vol-collapse measurement (§2.4) remains the alive specific instance** of this idea, but it should not be treated as the only surviving reframe. Atlas entry should be tagged `revisit_when: [event_bar_partition_built, meta_labeling_implemented, deflated_sharpe_correction_applied]`.

### 2.7 Smart Money Tracker

Memory captures it. Monitor top Polymarket leaderboard wallets as a **spoiler signal (not copy trading)** — before entering a trade, check if smart money is positioned against you.

`engines/smart_money.py` (903 lines) and `smart_money_alerts.py` (684 lines) exist in repo. MCP wrapping is the deferred Brief.

### 2.8 NegRisk Market Rebalancing — Conditionally Non-Executable (Reframe Candidates Below)

Multi-outcome markets where bucket probabilities must sum to 100%. Apparent mispricings are **phantom in the tested context**, caused by illiquid placeholder outcomes with empty ask-side order books. Flash loan approach didn't work for taker execution (CLOB is off-chain).

**The honest finding: non-executable as a taker on mainstream Polymarket given current liquidity profile.** That is narrower than "structurally non-executable." Three reframes are open and untested:

1. **Decentralized prediction markets beyond Polymarket/Kalshi** — lower volume but "wild west" pricing where the same arb structure may work. Watch for resolution-manipulation risk.
2. **Brief executable windows on larger NegRisk markets at moments of high real activity** (election nights, major sports events, Fed announcements). The placeholder outcomes briefly get real liquidity and the sum-of-prices arb may have measurable windows we never sampled in the original test.
3. **As a maker rather than taker** — providing liquidity in the illiquid legs and earning the spread someone else wants to cross. The structural feature that killed taker execution becomes income under the maker frame. Conceptually adjacent to the §2.5 market-making bot.

The `engines/negrisk_arb.py` and `engines/flash_executor.py` files stay in repo as the taker-frame reference implementation. **A maker-frame negrisk variant has not been written and would be a distinct engine.** Atlas entry should be tagged `revisit_when: [maker_frame_implementation, decentralized_market_data_available, high_liquidity_event_window_data_available]`.

### 2.9 Funding Rate Carry — Highest EV Item

Sharpe +4.45 to +10.78 confirmed via backtest. Strategy file `engines/funding_rate_strategy.py` exists. Currently monitored only.

**Critical caveat captured in chat (Taleb-pilled framing):** Sharpe number assumes regime continuity. Basis-collapse events aren't in our sample. The Sharpe assumes the second moment is bounded and that the mean is meaningful — both assumptions are testable but currently untested. **Atlas entry should explicitly disclose: "Sharpe 4.45-10.78 assumes regime continuity; not validated against discrete basis-collapse events; treat as conditional Sharpe pending stress-regime sampling."** This is the López de Prado / Taleb-pilled discipline applied honestly.

**Prerequisite for live execution:** `funding_rates` table needs to be live (currently orphan, see §1.2 item 2). Also needs `phase3_models.joblib` rebuilt (lost with disk).

### 2.10 MEV Build Plan (Memory ordering)

Phase 1 Scanner ✓ (`engines/mev_scanner.py` committed) → Phase 1c-AI Event-Driven Spike Predictor (transformer, classify event types, predict remaining move using price + news + social velocity) → Phase 2 Executor → Phase 3 AI Combinatorial Engine → Phase 4 Cross-Chain Monitor → Phase 5 Cross-Chain Executor → DEX arb + flash loans.

No new updates from this session. Long-arc plan stays as captured in memory.

### 2.11 Kraken Breakout Prop Trading

Memory item; sign up, pay trial fee, pass test, trade with their capital at 90% profit share.

**Critical update captured in this chat (Kelly math discipline):** the prop firm article triage produced the honest math: even with genuine positive expectancy, ~20-30% probability of randomly hitting drawdown floor before profit ceiling. **Budget for 3-5 attempts in EV calculation.** Also: verify Kraken Breakout is A-book vs B-book — read program terms carefully for "simulated environment" or hedging language about firm discretion. Counterparty risk lower than random Discord prop firms but B-book structure independent of firm reputation.

### 2.12 Long-Arc Multi-Agent Vision (Discovery → Research → Backtest → Manager)

The single most ambitious item discussed. Architecture I sketched in chat, with your refinements:

**Five agents (refined boundaries from chat):**
- **Discovery Agent** — web scanning, idea extraction, structured `TradingIdea` output
- **Research Agent** — Atlas lookup (already invalidated / partial overlap / genuinely new), `IdeaScorecard` output
- **Data Agent** — confirms required data exists or schedules collection, `DataAvailability` output
- **Backtest Agent** — smoke test → parameter battery (uses NautilusTrader if evan-kolberg adapters check out)
- **Manager Agent** — coordinates, runs strategy scorecard, makes Kelly vector allocations, halts live book on drawdown limits

**Critical design rule captured:** Manager Agent is the only agent with live execution authority. A Discovery Agent reading a Medium article should never be in the call graph that places an order.

**Build order (lowest-waste path):** Atlas DB migration first (Cycle 8) → NautilusTrader integration → Manager Agent doing risk management on existing manual strategies → Backtest Agent automating parameter sweeps → Research Agent testing against known-bad ideas → Discovery Agent last.

**Discovery Agent source-tier design (captured during the SetupAlpha 10-websites article):**
- **Tier 1 (continuous monitoring):** arXiv q-fin RSS, SSRN q-fin, QuantPedia, QuantRocket, NBER working papers (finance section), Hugging Face datasets
- **Tier 2 (author allowlist seeds):** Bawa, Khirman, Karlsson, López de Prado, Carver, Asness, Chan, Cirillo
- **Tier 3 / blocklist:** SetupAlpha, FXM Brand, Moonsat, pumpparade, Iriarte Cabrera (pattern-matched content farms)
- **Filter axis (the Ilinski / Iriarte Cabrera realization):** operationalizability + disclosed validation methodology, NOT vocabulary sophistication. Vocabulary score fails to discriminate Ilinski-tier work from López-de-Prado-tier work.
- **Promotion/relegation protocol:** when an idea sourced from author X produces a high-confidence Atlas entry, X gets monitoring weight. Three failed backtests in a row → downweight.
- **Citation-graph traversal beats site-coverage breadth.**

**Manager Agent prerequisites — reading queue from memory #30:**
1. **López de Prado *Machine Learning for Asset Managers* (MLAM, ~150pp)** — deflated Sharpe ratio (multi-test inflation correction), combinatorial purged CV, meta-labeling, triple-barrier method. Highest yield-per-page for our use case. Retroactively explains v8.1 work.
2. **Carver *Systematic Trading + Leveraged Trading*** — multi-strategy framework, **diversification multiplier math is the missing piece for Manager Agent Kelly-vector allocation across strategies.** Without this, Manager Agent allocations will systematically over-allocate to strategies that look diversifying but aren't. Carver's Substack "Investment Idiocy" = Tier 2 allowlist.

---

## 3. Data Priorities — What to Rebuild and What Not To

This is the more important half of the analysis, per your prompt. Data is permanently gone; what we choose to recreate determines what's possible in the next 4 weeks.

### 3.1 Inventory of What Was Lost

| Source | Volume | Recoverability | Strategic value |
|---|---|---|---|
| `crypto_data.db` — trades table | 1.36M trades, ~6 days continuous | Zero recovery. **Restart from now.** | High — microstructure features, v8.3 dependency |
| `crypto_data.db` — order_book_snapshots | 6,498 snapshots, ~18 hours | Zero recovery. **Restart from now.** | High — same |
| `crypto_data.db` — ohlcv_1m | 520k bars (1m), 2.5 years | **Re-pullable from Binance API** | High — LSTM training data |
| `crypto_data.db` — ohlcv_4h, ohlcv_daily | 2,160 4h candles, 1,800 daily | **Re-pullable from Binance API** | High — same |
| `crypto_data.db` — funding_rates | 2,190 obs (8h cadence × ~2 years) | **Re-pullable from Binance API** | High — funding carry strategy ground truth |
| `crypto_data.db` — fear_greed | 900 daily observations | **Re-pullable from Alternative.me API** (free) | Medium — LSTM feature |
| `crypto_data.db` — onchain_btc | 365 days metrics | **Re-pullable from glassnode/coinmetrics** (some free, some paid) | Medium — quantamental feature |
| `live_collector.db` — Polymarket price snapshots | 213k 60-sec snapshots since Apr 22 | **Becker dataset replaces this and goes back further** | Critical — convergence detector |
| `smart_money.db` — wallet positions | Months of accumulated wallet snapshots | Partial — current state queryable from Polymarket Data API; history gone | Medium — smart money tracker |
| `phase3_models.joblib` | Trained funding rate model | **Retrainable** once funding_rates is back | Medium — funding strategy live |

### 3.2 The Atlas-Driven Reframe: What Has Changed in Our Priorities?

This is the key question you flagged. Several Atlas-level realizations from this chat materially change which data matters most:

**1. Standard TA on OHLCV is dead — five lines of evidence.** v8.1 triple-barrier probe: 41.4% test accuracy vs 42% baseline. Liu paper independent confirmation. Five total lines of evidence captured in memory. **Implication:** rebuilding ohlcv_1m to high accuracy is no longer urgent for direction prediction. It's still needed as **input to the LSTM cross-asset feature model** (which uses 7 features × 60 timesteps), but the standalone "OHLCV will tell us the direction" thesis is closed. Lower the priority of dense OHLCV history; it's a feature, not the answer.

**2. Microstructure is the open frontier.** Trade flow + order book snapshots are the v8.3 thesis. The disk crash erased our accumulation runway. **Implication:** restarting trades and order_book collectors is **the single highest-priority data action** because the clock for v8.3 readiness restarts now. Memory said "earliest probe May 7, robust training May 21" — that was based on April 22 collector start. New equivalent: earliest probe ~May 12 (assuming restart May 28 + 2 weeks), robust training ~May 26.

**3. Funding rate carry is highest EV but data-prerequisite gated.** Funding history is re-pullable from Binance, so this is a one-shot script away from being whole again. **Implication:** funding_rates re-pull is high-priority **and cheap** — should be done immediately, not deferred.

**4. Becker Polymarket dataset (33GB parquet) is the unlock.** This was the genuinely useful find from the article queue. Includes years of trade + orderbook + resolution data. **Implication:** several queued items (convergence detector, AI ensemble probability engine, market-making bot backtest, even a retroactive 5-min binary direction test à la Liu) become testable via historical data **without needing local accumulation**. Ingesting Becker's dataset is the single highest-leverage data action for prediction-market strategies.

**5. The Atlas DB migration changes how we use rebuilt data.** Once Cycle 8 ships, every strategy result lands in queryable structured form. **Implication:** rebuild data with the upcoming Atlas integration in mind — as soon as a backtest produces a result, it gets logged into `atlas_experiments` rather than scattered across markdown notes. This argues for **deferring serious backtesting until after Cycle 8 ships**, so the new results land in the right structured form rather than needing migration later.

**6. Daily-bar features are confirmed irrelevant to intraday decisions** (Chan/Burgess pairs trading realization captured in memory item, the migration that produced `src/praxis/models/burgess.py`). **Implication:** historical daily-aggregate computations that were planned should be re-scoped down. We don't need 5 years of daily covariance matrices for pairs trading; we need minute-bar lookback windows of the right horizons.

### 3.3 Priority-Ordered Data Recovery Plan

Three tiers, ordered by EV-per-effort.

**Tier 1: Do this week.**

| # | Action | Cost | Yield |
|---|---|---|---|
| 1 | Restart trades + order_book collectors with the Cycle 7 OrderBook `--duration 3550` fix and the new MCP per-table thresholds. | 1-2h Brief + Code | Restarts v8.3 microstructure clock; without this we lose another week. |
| 2 | One-shot Binance API pull of full ohlcv_1m / ohlcv_4h / ohlcv_daily history (BTC + ETH, 2.5 years). | 1-2h script | Restores LSTM training inputs. Cheap and fully recoverable. |
| 3 | One-shot Binance API pull of full funding_rates history (BTC + ETH perp, 2+ years). | 30 min script | Restores funding carry model. Same Binance endpoints. Same script as #2 with different params. |
| 4 | Pull full Alternative.me Fear & Greed Index history (free, daily, 900+ days). | 15 min script | LSTM cross-asset feature. |
| 5 | Schedule new tasks for `collect-funding` (8h) and `collect-fear-greed` (daily) so these stop being orphan tables. | 30 min Brief + Code | Permanent fix for orphan-table problem identified in Cycle 7 triage. |
| 6 | Re-issue `.env` API keys: Anthropic, OpenAI, Voyage, Gemini, Binance, ElevenLabs, Twilio. **Polymarket private key is the high-stakes one** — needs seed phrase recovery or Polymarket support escalation. | 1-2h | Most strategies inert until this is done. |

**Tier 2: Do over the next 2 weeks.**

| # | Action | Cost | Yield |
|---|---|---|---|
| 7 | **Ingest Becker Polymarket dataset (33GB).** Mode B Brief: download from Cloudflare R2, validate schema, ingest to a new `data/polymarket_history.parquet/` directory next to crypto_data.db, sanity-check evan-kolberg NautilusTrader adapters. | 60-90 min Brief, 4-6h Code (mostly waiting on download/ingest) | Unblocks: convergence detector backtest, AI ensemble probability validation, market-making bot historical sim, retroactive 5-min direction test. Multiple items at once. |
| 8 | Pull Binance liquidation data history (Corvino-inspired ensemble feature). | 1h script | Adds liquidation as quantamental feature. |
| 9 | Pull cross-asset macro features for LSTM: XAU/USD daily, DXY, 10Y TIPS. yfinance, free. | 30 min script | Memory-flagged feature additions. |
| 10 | Retrain `phase3_models.joblib` on rebuilt funding_rates data. | 1-2h Code | Restores funding strategy live capability. |
| 11 | Build `data/RECOVERY_NOTES.md` documenting what was reconstructed-from-API vs what was permanently lost vs what's regenerating live, with timestamps of restart events. | 30 min, manual | Audit trail. Critical for any future crisis attribution and for honest Atlas entries. |

**Tier 3: Defer until specific strategies need them.**

| # | Action | Triggered by |
|---|---|---|
| 12 | Pull on-chain BTC metrics history. | LSTM training begins |
| 13 | CFTC COT report ingestion (gold positioning). | Cross-asset LSTM feature day |
| 14 | Smart money historical wallet snapshots (rebuild via Polymarket Data API, partial). | Smart money tracker build |
| 15 | Hugging Face SII-WANGZJ 1.1B Polymarket dataset (bigger Becker alternative). | If Becker proves insufficient |
| 16 | pmxt.dev hourly Polymarket orderbook archive subscription. | If we want continuous historical orderbook depth |

### 3.4 What Specifically Should NOT Be Rebuilt — Conditionally

**Critical reframe.** "Don't rebuild this" is shorthand for "don't rebuild this *under the current methodology profile*." Each item below is conditionally skipped pending specific methodology advances that would warrant re-testing. Every entry includes its `revisit_when` triggers so that when the methodology evolves we don't have to re-derive the right re-test scope.

This framing matters because **previously-failed strategies don't stay failed under a paradigm shift in how we evaluate them.** Event-bar partitioning, meta-labeling, triple-barrier with volatility-adjusted geometry, deflated Sharpe correction, and combinatorial purged CV are each potentially capable of reviving a result that failed under conventional methodology. The Atlas should preserve the option to revisit, not foreclose it.

1. **Pre-migration Chan/Burgess daily-bar feature computations.** Daily-aggregate features have no bearing on intraday decisions — that was the realization driving the migration to `src/praxis/models/burgess.py`. **But this is a feature-frequency finding, not a strategy-class finding.** Mean-reversion pairs trading is one of the oldest live strategy categories in finance; the migration is the *fix*, not evidence the underlying strategy class doesn't work. The post-migration minute-bar framework has not been validated on real data yet.
   - `revisit_when`: minute-bar Chan/Burgess validation completes; if it produces an inconclusive or marginal result, then test event-bar variants before declaring the strategy class itself non-viable.

2. **NegRisk arb scanner output history.** Phantom mispricings caused by illiquid placeholder outcomes — confirmed under taker-execution-on-mainstream-Polymarket framing. **Three reframes are open** (see §2.8): decentralized prediction markets, brief high-activity windows on Polymarket, maker-frame variant. The `negrisk_arb.py` and `flash_executor.py` files stay in repo as taker-frame reference; the maker-frame variant doesn't exist yet and a maker-frame negative result would be a separate finding.
   - `revisit_when`: maker-frame negrisk implementation; decentralized prediction market data ingestion; Polymarket high-liquidity event-window data captured.

3. **v8.1 standard-TA backtests.** Five lines of evidence that 25-feature OHLCV on time bars with conventional cross-validation can't predict 5-min BTC direction. **None of those features were computed on event-bar data, none used meta-labeling, none used triple-barrier with volatility-adjusted geometry, none applied deflated Sharpe correction.** The honest claim is "naive TA features on time-bar data with conventional CV doesn't predict 5-min direction." Narrower than "TA is dead on this asset."
   - `revisit_when`: event-bar partition built; meta-labeling implemented; triple-barrier labeling with volatility-adjusted barriers implemented; deflated Sharpe correction integrated into result reporting.

4. **Pre-fix battle results from before the dimension-path-bug fix in `runner.py`.** AI Factory artifact, not Praxis. Re-run rather than restored.
   - `revisit_when`: not applicable (these are obsolete by definition, no methodology evolution would revive them).

5. **5-min BTC binary direction backtests à la 4-minute rule.** Per Liu independent confirmation, this approximates random walk **under time-bar sampling with conventional methodology** (see §2.6 for the methodology-conditional framing).
   - `revisit_when`: event-bar partition built; meta-labeling implemented; deflated Sharpe correction applied.

**The general principle this section now establishes:** every NEGATIVE Atlas entry should carry an explicit `revisit_when` tag listing the methodology changes that would warrant re-testing. Without this tag, negative results become silent dead weight — confirmed-failed work that no future methodology advance can claim against. With this tag, negative results become **productive infrastructure** — when the Atlas DB is queried for "what should I re-test now that I've added meta-labeling," the right candidates surface automatically.

This principle is encoded into the Atlas DB schema augmentation in §2.1 (`methodology_fingerprint` and `revisit_when` fields). It is the most important methodological commitment captured in this document.

### 3.5 Atlas Entries That Should Get Updated From This Cycle

These are findings from this chat that should land in the Atlas as part of or after Cycle 8. Note that NEGATIVE entries below are **methodology-conditional NEGATIVE** per §2.1.1's schema augmentation — each carries explicit `revisit_when` triggers, not blanket dead-weight tagging.

| New / Updated entry | Result class | Revisit triggers | Source |
|---|---|---|---|
| Funding rate carry — Sharpe 4.45-10.78 with **regime-continuity caveat** | POSITIVE_CONDITIONAL | basis_collapse_event_in_sample, longer_history_window | Memory + Taleb-pilled framing |
| Forced flow / mandate exploitation as a unifying category — leveraged ETF rebalancing (decayed), index reconstitution (alive but competed), buffered-ETF gamma (under-explored), funding rate carry (alive), Polymarket convergence (queued) | CATEGORICAL | n/a (taxonomy, not result) | SetupAlpha leveraged-ETF article + Chan funding strategy |
| 5-min BTC binary direction prediction — five lines of evidence of non-predictability **on time bars with conventional methodology** | NEGATIVE_CONDITIONAL | event_bar_partition_built, meta_labeling_implemented, deflated_sharpe_correction_applied | v8.1 + Liu paper |
| Polymarket fee structure (formula `fee = shares × feeRate × price × (1-price)`, feeRate=0.072 crypto, GTC orders fee-free) | INFRASTRUCTURE_FACT | n/a (fact, not result) | Polymarket BTC trading engine article triage |
| NegRisk arb on Polymarket multi-outcome markets — non-executable as taker on mainstream Polymarket given current liquidity | NEGATIVE_CONDITIONAL | maker_frame_implementation, decentralized_market_data_available, high_liquidity_event_window_data_captured | Cycle pre-this work + chat reframe |
| Polymarket 5-min BTC market deterministic slug pattern `btc-updown-5m-{window_ts}` | INFRASTRUCTURE_FACT | n/a | Patange article triage |
| Black-Scholes false-assumptions framing — every arb monetizes a P-vs-Q gap | THEORETICAL_FRAME | n/a | Bawa article + Girsanov discussion |
| Discovery Agent source-tier seed list (Tier 1 + Tier 2 + blocklist) | INFRASTRUCTURE_RECORD | n/a | SetupAlpha 10-websites article triage |
| Methodology-aware negative-result schema (this section's principle codified) | METHODOLOGY_PRINCIPLE | n/a (foundational) | This document's revision |
| **Forced Flow Pressure as 13th regime class in REGIME_MATRIX.md** — generalizes the "forced flow / mandate exploitation" Atlas category into a regime-detection primitive. Five-state scheme (−2 to +2) measuring crowding + trigger-imminence + cascade-active across mandate-driven flows. Subsumes short squeezes (equity), crowding cascades in correlated factor positioning (canonical: 2007 quant quake — Khandani-Lo unwind hypothesis), funding rate cascades + liquidation maps (crypto perps), MEV opportunities, Polymarket convergence, leveraged ETF rebalancing flow, options gamma hedging. Two amplifying sub-primitives distinguished: **cross-asset trigger** (originating event in unrelated market — credit stress, rate shock, counterparty distress in another book) and **marketmaker risk-capital withdrawal** (liquidity providers pull back at moment of maximum stress, widening spreads and worsening the cascade — separate stage detectable via spread/depth telemetry). Detection signal stack varies per market: borrow rate + short interest + OI gamma concentration for equity squeezes; factor return correlations + stock-loan utilization + prime broker concentration for crowding cascades; funding rate + open interest + liquidation price levels for crypto perps; price proximity to 0/1 + market-maker net delta for prediction market convergence. Becomes a feature input to existing strategies: funding rate carry decomposes Sharpe across forced-flow regimes (the Taleb-pilled "basis collapse blow-up" failure mode is a +2 against the trade, with the 2007 quake as the canonical cross-asset-triggered analog), convergence detector tagged with Polymarket-specific forced-flow indicators, MEV phase 1c spike predictor uses forced-flow precursors. Implementation: extend `docs/REGIME_MATRIX.md` with the new class plus markets-specific signal stacks, add to Atlas DB schema migration as part of `regime_classes` table population. | METHODOLOGY_PRINCIPLE / REGIME_PRIMITIVE | n/a (foundational) | RB Trading squeeze article triage + earlier "forced flow / mandate exploitation" category synthesis (§3.5 leveraged ETF discussion) + datadriveninvestor "quant world is doing things" article triage surfacing Khandani-Lo 2007 case study |

### 3.6 The López de Prado Discipline Gates

**This is the most important strategic shift.** Once MLAM is digested (which I'd schedule as parallel reading during the data rebuild), the validation gates for any strategy entering the live book change:

1. **Walk-forward validation on at least 12 months of out-of-sample data.**
2. **Regime-stratified performance check across at least 3 distinct regimes** from REGIME_MATRIX.md.
3. **Ablation test where one core feature is removed** — strategy should degrade smoothly, not catastrophically (smooth degradation = real edge; catastrophic = overfit).
4. **Explicit edge-mechanism documentation** that survives the "would Renaissance/Two-Sigma have already arbed this if it were real" sanity check.
5. **Paper trading for some minimum live time before real capital** — not a backtest, real-time decisions on real-time data, even if no money moves.
6. **Automatic demotion** when a live strategy's running Sharpe drops below floor for N days, Manager Agent shrinks Kelly weight or pulls.
7. **Deflated Sharpe correction** for any strategy where multiple parameter variations were tested. Reported Sharpe S after testing N strategies should be deflated by ~S × √(2 ln N / N). For v8.1 with N≈25 variants, the correction is non-trivial.
8. **Methodology fingerprint required on every Atlas entry, NEGATIVE results carry `revisit_when` triggers.** Per §2.1.1 schema augmentation. Without this, the Atlas accumulates dead-weight negative findings; with it, negative results are productive infrastructure that surface as re-test candidates when methodology evolves.

Items 1-6 are existing reasoning; item 7 (deflated Sharpe) is the MLAM-specific addition; **item 8 is the methodology-aware extension established by the §3.4 reframe and codified in the §2.1.1 schema augmentation.** Worth embedding all three into the Atlas DB schema's structured fields.

---

## 4. Workflow & Architecture Updates

### 4.1 Workflow Changes Captured in Chat

1. **Atlas DB substitutes for inline historical context.** Memory #25 was updated to reflect this. Once Cycle 8 ships, chat handovers carry only state + conventions; experiment lookup happens via `atlas_search` / `atlas_get` on demand. Pattern generalizes: queryable DB tables replace inline context for any structured project knowledge.

2. **Brief structure can include "Recommended model tier per phase".** Code defaults to Sonnet for routine work, escalates to Opus 4.6 1M for complex tasks. Future Briefs can hint per-phase complexity to save Code-side cost.

3. **Brief-as-skill / template directory.** Recurring Brief patterns (collector registration, MCP tool addition, parameter sweep, outage triage, feature-column addition to data table) are candidates for `claude/handoffs/templates/` skeleton briefs. Defer until 5-6 cycles establish what actually recurs.

4. **Domain blocklist for Discovery Agent is workflow rule, not technical.** Grows as we encounter new content farms. Currently: SetupAlpha, FXM Brand, Moonsat, pumpparade, Iriarte Cabrera. Will extend.

5. **Republish-pattern detection as a quality signal.** FXM Brand reposts identical content under different post IDs. Discovery Agent should hash article content + author and downweight repeat-template authors.

### 4.2 Architecture Updates Captured

1. **Skill + MCP-tool duality** (Khirman's pattern). When adding new MCP tools, build the parallel Claude Code skill that wraps the same logic. Adopted as principle going forward.

2. **MCP server scope — three .db files.** Cycle 7 covered crypto_data.db (10 tools). Cycle 8 brings Atlas DB MCP tools (atlas_search, atlas_get). Future cycle wraps smart_money.db. Pattern: each new operationally-relevant SQLite gets MCP-wrapped.

3. **Cross-asset features for LSTM.** Memory item flagged. Daily macro features (XAU/USD, DXY, TIPS) get added when LSTM training day arrives.

4. **"Forced flow / mandate exploitation" as Atlas category.** Unifies funding rate carry, leveraged ETF rebalancing, index reconstitution, buffered-ETF gamma, Polymarket convergence. Cross-references the Girsanov / P-vs-Q framing.

5. **Operationalizability filter for Discovery Agent**, not vocabulary sophistication. The Ilinski / Iriarte Cabrera convergence proves vocabulary fails to discriminate. Filter axes: discloses assumptions, specifies out-of-sample validation, engages with multi-test inflation, reports failures alongside successes.

---

## 5. Suggested Order of Operations Next 2-4 Weeks

Concretely, what to do in what order. Items in **bold** are bottleneck-unblockers. Cycle numbers assume the dual-Claude rhythm.

**Week 1 (recovery completion + Cycle 8):**
1. **Re-issue API keys, rebuild `.env`.** Polymarket private key triage first.
2. **Cycle 8: Atlas DB migration** — execute the already-written Brief. Code session.
3. Tier 1 data items 1, 2, 3, 4, 5 (collectors restart, ohlcv re-pull, funding re-pull, fear/greed re-pull, schedule new tasks). Probably 2-3 small Briefs combined.
4. Begin reading MLAM in parallel. Add Khandani-Lo 2011 *"What Happened to the Quants in August 2007?"* (NBER WP #14465, freely available) as case-study companion — short read, canonical empirical specimen of the cross-asset-triggered crowding cascade that the Forced Flow Pressure regime class is designed to detect.

**Week 2 (data foundation + first strategy):**
5. Tier 2 data item 7: Becker dataset ingestion. Mode B Brief.
6. Tier 1 data item 11: RECOVERY_NOTES.md.
7. **Cycle 9: Atlas DB methodology-aware schema augmentation** (per §2.1.1) — adds `methodology_fingerprint`, `revisit_when`, `last_methodology_audit_at` fields. Migrates existing entries with default fingerprints. ~30 min of Code work plus ~30 min of human review.
8. Cycle 10: write `engines/burgess.py` legacy cleanup Brief and execute.
9. **Cycle 11: Funding carry live execution wiring** with the Taleb-pilled regime-continuity caveat as part of the Atlas entry. The entry uses the new `methodology_fingerprint` to record the validation methodology under which Sharpe was measured.

**Week 3-4 (strategy build resumes):**
10. Begin reading Carver's *Systematic Trading* (Manager Agent prerequisite).
11. Cycle 12: smart_money.db MCP wrapping (60-min Brief).
12. Tier 2 data items 8, 9, 10 (liquidation history, cross-asset macro, retrain phase3_models.joblib).
13. Cycle 13: Convergence speed detector first cut, validated against Becker historical data. Atlas entry uses methodology fingerprint to record the validation profile.
14. Cycle 14+: AI ensemble probability engine first cut, validated against Becker historical resolved markets.

**Continuous in parallel:** as the methodology profile evolves (event-bar partition, meta-labeling, deflated Sharpe), the Discovery Agent (when built) or a manual sweep (sooner) queries the Atlas for `revisit_when` triggers and surfaces NEGATIVE_CONDITIONAL entries that warrant re-testing. **This is the operational payoff of the §2.1.1 schema augmentation.**

**At completion of weeks 3-4:** the build queue is back to where it was pre-crash, with Atlas DB infrastructure as a new advantage, methodology-aware result tagging integrated, and MLAM/Carver discipline applied to all new entries.

---

## 6. Open Questions Requiring User Decision

These are the pending decisions the chat had identified but hadn't resolved:

1. **Schedule funding/sentiment collectors, or remove orphan tables from health check?** Chat recommendation: schedule both. Awaiting confirmation.
2. **Brief-templates directory now or after 5-6 more cycles?** Chat recommendation: defer.
3. **`engines/burgess.py` cleanup priority** — soon or after Cycle 8 ships?
4. **Becker dataset urgency** — wait for Atlas DB to land first (so results land structured), or ingest immediately to begin convergence-detector work in parallel?
5. **Polymarket private key recovery** — escalate to Polymarket support, or accept loss?
6. **`gui/mcb_studio/backend/requirements.txt` deletion** — separate cleanup commit now or defer?
7. **Atlas DB methodology-aware schema augmentation (§2.1.1) — Cycle 8.5 augmentation or separate Cycle 9?** Chat recommendation: separate Cycle 9 to keep the Cycle 8 Brief unchanged and the schema change deliberate. But if Cycle 8 finds the original schema underspecified during execution, fold it in as 8.5. Awaiting your call.

---

## 7. What This Doc Replaces

This document is the synthesis of post-Cycle-7 chat substance that didn't make it to artifact files before the disk crash. Once the Atlas DB ships, much of this content becomes proper Atlas entries (entries flagged in §3.5) plus structured queue items in `claude/handoffs/`. Until then, this is the canonical recovery roadmap.

**Authored:** 2026-04-28 (v1 initial recovery roadmap), revised same day (v2 — methodology-aware reframe of §2.6, §2.8, §3.4, §3.5, §3.6, with new §2.1.1 schema augmentation), revised 2026-04-29 (v3 — added Forced Flow Pressure as 13th regime class to §3.5 Atlas entries table, generalizing squeeze / funding cascade / convergence / MEV under one regime primitive), revised same day (v4 — Forced Flow Pressure entry extended with crowding cascade as enumerated primitive plus cross-asset trigger and marketmaker withdrawal as named sub-stages; Khandani-Lo 2007 paper added to week 1 reading), revised same day (v5 — added document-type header clarifying this is a roadmap not a Brief, to prevent confusion since it lives in `claude/handoffs/`).
**Author:** praxis_main_current chat (Claude).
**Recommended location:** `claude/handoffs/RECOVERY_PLAN_post_disk_failure.md` (so it's discoverable next session and becomes part of the Brief/Retro permanent record).
