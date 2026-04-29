# Retro: Historical Data Backfill (Tier 1 Recovery) — COMPLETE

**Date:** 2026-04-29
**Status:** COMPLETE — all 7 acceptance criteria met
**Brief:** `claude/handoffs/BRIEF_historical_backfill.md`
**Series / Cycle:** praxis / 9
**Mode:** B (CLI invocations against existing collector; no source edits)
**Scope:** S (six one-shot subcommands + one validation pass)

---

## 1. TL;DR

All six `engines.crypto_data_collector` subcommands ran cleanly on the first attempt with no rate-limit hits, no retries, and no failures. `crypto_data.db` is now repopulated with 1,800 daily bars, 10,800 4h bars, 518,405 1m bars, 2,190 funding rate observations, 900 fear-greed readings, and 364 days of on-chain BTC metrics. The strategically critical `funding_rates` table is at 100% of expected coverage (2,190 rows, BTC+ETH, 365 days). The OrderBook collector kept running in parallel throughout (verified Running before and after) and accumulated 2,946 snapshots in the same window. No source files were modified.

---

## 2. Phase 2 validation output (verbatim)

```
ohlcv_daily                    1,800 rows  (2023-11-12 -> 2026-04-29)
ohlcv_4h                      10,800 rows  (2023-11-11 -> 2026-04-29)
ohlcv_1m                     518,405 rows  (2025-10-31 -> 2026-04-29)
funding_rates                  2,190 rows  (2025-04-30 -> 2026-04-29)
fear_greed                       900 rows  (2023-11-11 -> 2026-04-29)
onchain_btc                      364 rows  (2025-04-30 -> 2026-04-28)
order_book_snapshots           2,946 rows  (2026-04-29 -> 2026-04-29)
```

`order_book_snapshots` row count reflects the parallel-running Cycle 8 collector and is not produced by this Brief. The other six rows are the Brief's deliverable.

---

## 3. Acceptance criteria check

| # | Criterion | Status |
|---|---|---|
| 1 | All six invocation commands run | PASS (all six completed first-try) |
| 2 | Phase 2 validation output captured | PASS (§2 above) |
| 3 | `funding_rates` > 1,800 rows for BTC+ETH | PASS (2,190 rows) |
| 4 | At least three tables (`ohlcv_daily`, `ohlcv_4h`, `funding_rates` minimum) at >=80% expected | PASS (all three at 100%) |
| 5 | No edits to `engines/crypto_data_collector.py` or any other source file | PASS (`git status` shows only untracked files in `claude/`) |
| 6 | Retro at `claude/retros/RETRO_historical_backfill.md` | PASS (this file) |
| 7 | `git status` shows `data/crypto_data.db*` as the only modified files | PASS (db files are gitignored as expected; modified-file set is empty; untracked additions are confined to `claude/` working dir) |

---

## 4. Per-invocation results

| # | Subcommand | Asset | --days | Rows stored | Notes |
|---|---|---|---|---|---|
| 1.1a | `collect-ohlcv` | BTC | 900 | 900 daily | Full coverage |
| 1.1b | `collect-ohlcv` | ETH | 900 | 900 daily | Full coverage |
| 1.2a | `collect-ohlcv-4h` | BTC | 900 | 5,400 | Brief flagged a possible 180d cap; **did not trigger** — full 900d came back (5,400 = 900 * 6). See §6. |
| 1.2b | `collect-ohlcv-4h` | ETH | 900 | 5,400 | Same — full 900d. |
| 1.3a | `collect-ohlcv-1m` | BTC | 180 | 259,202 | Single pass, ~260 batched requests, no 429s. Progress prints every 50 batches as documented. |
| 1.3b | `collect-ohlcv-1m` | ETH | 180 | 259,203 | Same shape as BTC. |
| 1.4a | `collect-funding` | BTC | 365 | 1,095 | 365 * 3 = 1,095 funding events, exact. |
| 1.4b | `collect-funding` | ETH | 365 | 1,095 | Exact. |
| 1.5  | `collect-fear-greed` | — | 900 | 900 | Single endpoint, instant. Latest reading: 26 (Fear). |
| 1.6  | `collect-onchain` | — | 365 | 364 | One day short of nominal — see §6. |

Wall-time: not recorded with `Measure-Command`; informal observation was the entire Phase 1 sequence completed in well under the 15-25 min estimate from the Brief, dominated by the two 1m pulls (each ~90s of paginated fetching).

---

## 5. Files modified

**Source files:** none.

**Data files** (all gitignored; do not appear in `git diff`):
- `data/crypto_data.db` — populated rows in `ohlcv_daily`, `ohlcv_4h`, `ohlcv_1m`, `funding_rates`, `fear_greed`, `onchain_btc`
- `data/crypto_data.db-wal`, `data/crypto_data.db-shm` — WAL sidecars

**Working files** (untracked, not committed; helpers I authored to sidestep PowerShell quote-mangling on inline `python -c` calls):
- `claude/scratch/check_schema.py` — Phase 0 schema lister
- `claude/scratch/check_columns.py` — column inspector used after the first validation script hit a `no such column: timestamp` error on `onchain_btc`
- `claude/scratch/validate_backfill.py` — Phase 2 validator (final corrected version with per-table time-column dispatch)

The `claude/scratch/` directory existed but was empty before this session. Leaving the three scripts in place; they are dirt-cheap to delete and are useful if this Brief needs to be re-run.

---

## 6. Anomalies and things worth flagging

### 6.1 The 4h "180-day cap" did not exist (or did not trigger)

The Brief warned that the collector might internally cap the 4h pull at 180 days regardless of `--days 900`. **It did not.** Both BTC and ETH returned the full 900-day window — 5,400 candles each, exactly `900 * 6 bars/day`. Either the cap referenced in the Brief was at a different code location than I would expect from the docstring, or it was removed in an earlier cycle and the Brief's caveat is now stale. This is purely upside; flagging in case the Chat-side mental model of the collector is out-of-date.

### 6.2 Date range starts 2023-11-12, not 2023-11-11

The 900-day pull from a launch date of 2026-04-29 should reach back to 2023-11-11 (using `today - 900*86400`). `ohlcv_daily` starts at 2023-11-12 (one day shy), while `ohlcv_4h` and `fear_greed` start at 2023-11-11. Likely a `>` vs `>=` boundary or a per-bar-window-alignment thing in the daily collector specifically. Cosmetic; 899-day-vs-900-day distinction is not meaningful for the LSTM use case.

### 6.3 On-chain came back 364 rows instead of 365

`collect-onchain` printed individual series row counts that varied: `active_addresses: 360`, `transaction_count: 361`, `hash_rate: 361`, `difficulty: 361`, `block_size: 361`, `market_cap: 1476`, with a final "Stored 364 days of on-chain data". The blockchain.info charts API appears to have slightly different per-metric coverage windows, and the collector aligns to the intersection. Not blocking for any current use; just noting.

The `market_cap: 1476` line stands out — that's nearly 4x the other counts. Likely the endpoint returns sub-daily ticks for that metric and the collector down-samples on insert. Did not investigate further; out of scope.

### 6.4 `onchain_btc` schema uses `date`, not `timestamp`

My initial Phase 2 validator assumed every table had a `timestamp` column. `onchain_btc` uses a `date` (string `YYYY-MM-DD`) column instead. First validator run threw `OperationalError: no such column: timestamp`. Trivial fix — added a per-table column map. Just a heads-up that the schema isn't perfectly uniform, in case a future MCP tool or feature builder assumes it is.

### 6.5 No 429s, no retries, no kill-switch invocations

The "known pitfalls" around Binance rate limits and long fetch times all turned out to be no-ops on this run — every command completed first try without any rate-limit response from Binance, blockchain.info, or alternative.me. The collector's existing `enableRateLimit: True` + 0.3s sleep is sufficient for 180 days of 1m candles per asset with no observable backoff.

### 6.6 OrderBook collector ran cleanly throughout

`Get-ScheduledTask` reported `State: Running` both at Phase 0 (before) and after all Phase 1 commands completed. The 2,946 `order_book_snapshots` rows accumulated during the backfill window confirm SQLite WAL mode handled the concurrent reads/writes without contention, exactly as the Brief predicted.

### 6.7 PowerShell here-string + `python -` does not work in this harness

I tried the Brief's exact Phase 2 validator (a PowerShell here-string piped via `python -c @"...`@`"`) and the harness's `PowerShell` tool fed it to a backgrounded shell that hung on stdin. Pivoted to writing the validator as a `.py` file and invoking it via `bash` + `.venv/Scripts/python.exe`. **Recommendation for future Briefs:** prefer "write a small `.py` file then invoke it" over inline `python -c` with embedded quoting, especially when the snippet contains both single and double quotes, format strings, and decorative characters. The Brief's `python -c @"..."@` block is a great spec but a poor invocation pattern for harnessed Windows execution.

---

## 7. Open Items for Chat

1. **Mental model of the 4h cap** (§6.1): Brief said the collector capped at 180 days for 4h candles, but the actual run pulled the full 900. Worth confirming whether the cap was lifted in an earlier cycle (and updating Chat's project memory) or whether there's a parameter path I missed.

2. **Phase 3 prerequisite** is met: the `phase3_models.joblib` retrain that the Brief explicitly excluded is now unblocked — `funding_rates` has its full 2,190-row ground-truth window. Chat can decide when to write the retrain Brief (Recovery Plan §3.3 Tier 2 territory).

3. **Scheduled-task gap** (Recovery Plan §1.2 item 2): going forward, `funding_rates`, `fear_greed`, and `ohlcv_daily` will go stale unless the `engines.crypto_data_collector` subcommands run on a schedule. This Brief explicitly excluded that. Suggest Chat draft a "scheduled-collectors" Brief next to install Windows Task Scheduler entries for `collect-ohlcv` (daily), `collect-funding` (every 8h aligned with funding events), and `collect-fear-greed` (daily).

4. **OrderBook collector duration race** (Cycle 8 retro §6) is unaffected by this work. The trades + crypto_1m collectors may share the same race; auditing them is queued for a separate Brief and not in scope here.

---

## 8. Kill switch / safety check

- Kill switch from Brief: stop after one retry on persistent failure, document, continue with remaining tables. **Not invoked** — no failures occurred.
- Source files modified: **0** (verified via `git status` showing only untracked files in `claude/`).
- `engines/crypto_data_collector.py` edited: **NO**.
- `services/*.bat` or `services/*.ps1` edited: **NO**.
- OrderBook collector disturbed: **NO** (verified Running both before and after).
- Real-money path touched: **NO** (read-only public APIs only).
