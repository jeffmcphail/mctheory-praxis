# RETRO — Cycle 63: Handover Blockers (Measurement Only)

**Date:** 2026-09-04
**Mode:** Code (measurement + verification). No behaviour changes, no writes to
`crypto_data.db`, no orders placed or simulated.
**Brief:** `claude/handoffs/BRIEF_cycle63_handover_blockers.md`
**Consumer:** `ai_factory_main_current`, blocked on T1 + T2 before scheduling the
MCRMINI1 → MCRMINI2 collector handover.

---

## T1 — Can `funding_executor` place real orders?

### **NO. Unconditionally.**

`services/funding_executor_service.bat` scheduled on MCRMINI2 tomorrow **could not
place a real order.** This is not conditional on an env var, a flag, or a config key —
there is no live order path in the module to gate.

### Why the question was genuinely open

Chat's prior was "funding_executor is a paper path," resting on the assumption that
**no live order path exists in this repo at all.** That assumption is false, and it was
worth re-deriving rather than confirming:

- `engines/carry_executor.py:270` — `order = exchange.create_market_order(symbol, side, qty)`
- `engines/carry_executor.py:169-170, 183-184` — reads `BINANCE_API_KEY` / `BINANCE_API_SECRET`
- `scripts/run_carry.py:263` — `--live` flag, *"Use real money (default: paper trading)"*

So a working real-money order path **does** exist. The correct framing is not "no live
path exists" but **"there are two disjoint paths, and the scheduled one is not the live one."**

### The evidence that they are disjoint

| Check | Method | Result |
|---|---|---|
| Import closure | AST walk (catches function-level imports, unlike grep) | `funding_executor.py` + `engines/trading_session.py`. Nothing else in-repo. |
| Network modules | Runtime import, inspect `sys.modules` | `socket` absent, `ssl` absent, `urllib.request` absent. Only `urllib.parse` — pure string parsing, pulled in by `dotenv`. |
| Dynamic escape hatches | grep `__import__` / `importlib` / `eval` / `exec` / `subprocess` / `os.system` | none, in either module |
| Reference to carry path | grep `carry` in `funding_executor.py` | no import, no call — only the words "funding-carry" in prose |
| Who imports `carry_executor` | repo-wide grep | **only** `scripts/run_carry.py` (4 sites) |
| Is `run_carry` scheduled | grep all 20 `services/*.bat` | **no bat references it** |
| Order-shaped methods | `dir(FundingExecutor)` | empty — none |
| Write side-effects | SQL extraction | exactly two: `INSERT OR IGNORE INTO paper_trades`, `INSERT OR IGNORE INTO paper_position_exits` |

The runtime check is the decisive one: the process never loads `socket` or `ssl`, so it
**physically cannot open a connection to an exchange**, regardless of intent.

> Note: the module docstring asserts this invariant and states it is "phrased to describe
> the invariant without containing the exact forbidden tokens, so the grep is clean." A
> safety-belt grep designed around its own docstring's phrasing is not evidence. The
> findings above were derived independently of it, and they happen to confirm it.

### 2. What gates it?

**Nothing — there is no live path to gate.** `EXECUTION_VENUE = "binance"` (line 130) is a
string used to scope the funding *read* (Cycle 57 P4), not an exchange connection.

### 3. What `--trigger-source scheduled` actually reaches

An attribution label, and nothing else. Full trace:

```
bat: python -u -m engines.funding_executor --trigger-source scheduled
  |- main() L872: add_argument("--trigger-source", default="cli",
  |                            choices=["cli","scheduled"])
  |- main() L912: create_session(..., trigger_source=args.trigger_source, ...)
  |                 -> writes trading_sessions.trigger_source
  |- main() L919: print(f"(trigger_source={...}, mode=paper_live)")
```

Those are its only two uses. **No behavioural branch anywhere.** `mode` is hard-coded
`"paper_live"` regardless of the flag.

> **Naming trap for the handover:** `mode="paper_live"` means *"paper trading, live
> (ambient) session"* — **not** live money. Anyone auditing `trading_sessions` and seeing
> `paper_live` on scheduled rows could easily read it the other way.

### 4. Credentials

`load_dotenv()` **is** called (line 108), but the module reads **exactly one env var** in
the entire file: `EXECUTOR_KILL_SWITCH` (line 226). No API key, no secret, no token.
`BINANCE_API_KEY` / `BINANCE_API_SECRET` are present in the repo but read **only** by
`carry_executor.py`, which nothing schedules.

### 5. Kill switch — and an honest caveat

`EXECUTOR_KILL_SWITCH`, read at line 226, `"1"/"true"/"yes"` forces every pending alert to
`skip`. **It is not present in `.env` today, so it is currently OFF.**

**It does not default to safe.** It defaults to *off*, i.e. permissive — entries proceed.
That is the right default here only because safety is **structural** (no order path exists),
not switch-based. If this switch were ever the primary protection on a live path, its
default would be wrong.

### Handover implication

The live path (`carry_executor` + `run_carry.py --live`) **travels with the repo to
MCRMINI2.** It is dormant because nothing schedules it, not because it is disabled — and
`BINANCE_API_KEY` / `SECRET` in `.env` travel with it. Cycle 55 recommended quarantining it;
that recommendation still stands and is now a handover-time concern.

---

## T2 — Do any collectors use IP-allowlisted API keys?

### **NONE. No allowlist updates are required for the handover.**

All 20 `services/*.bat` resolve to 12 distinct modules. Every one was checked for
credential env reads and for HTTP auth.

| Module (scheduled) | Authenticates? | Source |
|---|---|---|
| `engines.crypto_data_collector` (7 bats) | No | Binance public, CoinGecko public, blockchain.info public |
| `engines.bybit_liquidation_collector` | No | Bybit public WS |
| `engines.liquidation_collector` | No | Binance public WS |
| `engines.open_interest_collector` | No | public |
| `engines.ohlcv_universe_collector` | No | public |
| `engines.unlock_market_data_collector` | No | CoinGecko public |
| `engines.info_bars.writer` | No | public |
| `engines.live_collector` | No | Polymarket gamma / clob public |
| `engines.smart_money` | No | Polymarket data-api public |
| `engines.funding_executor` | No | local SQLite only |
| `scripts.funding_monitor` | No* | `PRAXIS_ALERT_URL`, `TEAMS_WEBHOOK_URL` |
| `scripts.funding_regime_alert` | No* | `PRAXIS_ALERT_URL`, `TEAMS_WEBHOOK_URL` |

Credential env reads across the entire scheduled set: **zero.** The only env vars read at
all are the two outbound webhook URLs (\*), which are notification sinks — not exchange
credentials and not IP-allowlisted.

Every HTTP call in the Polymarket collectors is a bare `requests.get(url, params=...)` —
**no `headers=`, no `auth=`.** Initial keyword matches on "token" were false positives
(market/side *tokens*, and Polymarket CTF tokens — identifiers, not auth).

### On Chat's claim about self-service portal edits

**Could not confirm, and it is moot.** Verifying whether IP restriction is enabled on the
Binance/Bybit keys requires signing into those API-management pages under Jeff's account —
outside what this cycle can reach. But no scheduled task uses those keys, so no allowlist
edit is on the handover critical path either way.

That question becomes live **only** if the carry path is ever activated on MCRMINI2. At
that point `BINANCE_API_KEY`'s allowlist must be checked before the first `--live` run.

### One quiet-failure item that is *not* an allowlist item

`.env` must carry over to MCRMINI2 for `PRAXIS_ALERT_URL` and `TEAMS_WEBHOOK_URL`. If it
does not, `funding_monitor` and `funding_regime_alert` **lose alerting silently** — the same
failure mode the allowlist question was guarding against, by a different cause.

---

## T3 — CoinGecko historical reach

### Endpoints and tier

- `GET /api/v3/coins/{id}` — `collect-market-data` (line 942) and `unlock_market_data_collector`
- `GET /api/v3/global` — BTC dominance (line 912)
- Host `https://api.coingecko.com` — **unauthenticated public free tier.**

**There is no CoinGecko key.** None in `.env`, no `x-cg-demo-api-key` / `x-cg-pro-api-key`
header anywhere in the code, no `pro-api.coingecko.com` reference. The instruction to use
the key in `.env` could not be followed because no such key exists — so the probes were run
exactly as production calls it, unauthenticated, which measures the real reach.

### Measured reach — 365 days

| Probe | Result |
|---|---|
| `/coins/bitcoin` | HTTP 200 — `circulating_supply = 20,079,618` (current) |
| `/coins/bitcoin/market_chart?days=365&interval=daily` | HTTP 200 — **366 points, 2025-09-05 → 2026-09-04** |
| `/coins/bitcoin/market_chart?days=max` | **HTTP 401**, `error_code 10012`: *"Public API users are limited to querying historical data within the past 365 days."* |
| `/coins/bitcoin/history?date=01-01-2024` | **HTTP 401**, same 10012 |

### `circulating_supply` — called out separately

### **NOT retrievable historically. At any date, including 3 days ago.**

`/coins/{id}/history` was probed at three dates *inside* the 365-day window:

| Date probed | HTTP | `market_data` keys returned | `circulating_supply` |
|---|---|---|---|
| 3 days ago (`01-09-2026`) | 200 | `current_price`, `market_cap`, `total_volume` | **absent** |
| 90 days ago (`06-06-2026`) | 200 | `current_price`, `market_cap`, `total_volume` | **absent** |
| 360 days ago (`09-09-2025`) | 200 | `current_price`, `market_cap`, `total_volume` | **absent** |

The endpoint succeeds and simply does not carry a supply field. `market_chart` does not
either (keys: `prices`, `market_caps`, `total_volumes`). This confirms the code comment at
`crypto_data_collector.py:925` and means **`market_data.circulating_supply` is snapshot-only,
forward-only** — a gap in that column cannot be refilled directly. Current state: 30 assets,
`2026-05-01` → `2026-09-04`.

### But it is derivable — F1 is not dead

`market_cap` and price **are** both historical, and `market_cap / price = circulating_supply`.
Validated against BTC:

```
derived (from /history, 3d ago):  20,077,862
current (from /coins/{id}):       20,079,618
delta:                            -1,756  (-0.0087%)
```

The delta is the correct magnitude *and direction* for 3 days of BTC issuance — this is
measuring real supply change, not error. **Caveat to document if used:** CoinGecko computes
`market_cap` from its own supply figure, so this recovers *their* number, not an independent
one. For F1 (token unlocks) that is the right number anyway, since it is the same series the
forward-collected column contains.

### Rate limit

No `x-ratelimit-*` headers are returned. Production evidence is better than a synthetic
burst here (a deliberate 429 would risk throttling the live collectors' egress IP mid-handover):

- `logs/market_data_collector.log`: **201 x HTTP 429**, recurring ~2/day, most recently **2026-09-04**.
- Cause: `collect-market-data --asset all` runs 6 assets at `time.sleep(2)` plus one `/global`
  = **7 calls in ~12s, about 35/min**, well over the public ceiling.
- `unlock_market_data_collector` is gentler: `--sleep` default **6.5s**, about 9/min.

So the free tier is **already being exceeded at current volume**, before the handover.

### Refilling a 3-day gap across 30 assets

- For `circulating_supply` **directly: impossible** — the field does not exist historically.
- Via the derivation: 30 assets x 3 days = **90 `/history` calls**
  - at 6.5s spacing: **~9.8 min** (may still 429 intermittently)
  - at a safe 12s spacing (~5/min): **~18 min**

**The move to MCRMINI2 does not raise this ceiling** — the limit is per-IP and the tier is
unchanged. A fresh IP resets accumulated reputation, nothing more.

---

## T4 — `smart_money` and `onchain` sources

### `engines/smart_money.py` — Polymarket, public, unauthenticated

- `https://data-api.polymarket.com` — `/v1/leaderboard`, `/positions`, `/trades`, `/activity`
- `https://gamma-api.polymarket.com`
- **No authentication.** Bare `requests.get(..., params=..., timeout=15)`; no headers, no keys.
- `discover --category ALL` covers 5 categories (`OVERALL, POLITICS, SPORTS, CRYPTO, CULTURE`),
  period MONTH, top 25, `time.sleep(0.5)` between categories.

### Connection lifetime — this is the mirror-breaker

`snapshot` opens **one** SQLite connection (line 344, `PRAGMA journal_mode=WAL`) and holds it
across a loop over **every active wallet — 1,586 of them** — each iteration making a network
call and sleeping 0.3s, closing only at line 434.

Measured from `logs/smart_money.log`:

| Run | Start | End | Held |
|---|---|---|---|
| 2026-09-04 | 09:05:23 | 09:26:48 | **21m 25s** |
| 2026-09-04 | 10:24:03 | 10:44:39 | **20m 36s** |

So: **~21 minutes of continuously-held WAL connection, 4x daily** (every 6h).

### The actual root cause is worse than a window overlap

The **2026-09-03 04:24 run died mid-loop at wallet 416/1571** (last output 04:30:23) and
**never reached `conn.close()`.** Nothing then ran until a manual, off-cadence restart at
**2026-09-04 09:05** — a **~29-hour outage** with the WAL sidecars orphaned throughout.

A mirror running at 09:05 on 09-03 would have hit `smart_money.db-shm` **4h35m into that
dead run.** That matches the reported failure far better than a 21-minute window collision.

Corroborated in the data — committed `snapshot_id`s jump straight across the outage:

```
20260903_022409   <- last good
20260904_130554   <- manual restart
```

**5 scheduled runs missed.** The died run never reached `conn.commit()` (line 407), so it
produced no snapshot row at all.

*(Secondary anomaly: the 09-02 10:24 run completed its Python output — full closing banner —
but never printed the bat's "Snapshot complete." echo. Milder, but the same shape: a run whose
termination the bat did not observe.)*

> The `-shm` / `-wal` files currently dated 14:54–14:55 are **mine** — created by the read-only
> queries run during this cycle. Not a stuck writer. Worth stating because in WAL mode *any*
> reader materialises `-shm`, which is itself part of the problem below.

### Implication for the handover

**Moving machines does not fix this.** The failure is crash-safety, not host. Two changes are
needed, neither in scope this cycle:

1. `snapshot` / `discover` should hold the connection under `try/finally` or a context manager
   so a mid-loop death still closes it — and ideally commit incrementally rather than once at
   the end, so a crash at wallet 416 does not discard 416 wallets of work.
2. The mirror needs to tolerate or checkpoint WAL sidecars, since a healthy run legitimately
   holds `-shm` for ~21 minutes 4x a day even when nothing is wrong.

### `collect-onchain` — blockchain.info, public

- `https://api.blockchain.info/charts/{metric}`, `timespan={days}days`, `rollingAverage=24hours`,
  `timeout=30`, `time.sleep(1)` between metrics. No key, no auth.
- 6 metrics: `n-unique-addresses`, `n-transactions`, `hash-rate`, `difficulty`,
  `avg-block-size`, `market-cap`.

### Backfillability

| Source | Backfillable? | How far |
|---|---|---|
| `collect-onchain` (blockchain.info) | **Yes** | Arbitrary `timespan`; collector currently clamps to `min(days, 365)` at line 1377. Raise the clamp for more. |
| `smart_money` (Polymarket) | **No** | The data-api returns *current* positions/leaderboard only. `position_snapshots` is a time series built by repeated sampling — a missed snapshot is gone permanently. **The 5 missed runs are unrecoverable.** |

---

## T5 — Binance liquidation stream from MCRMINI2

### **SKIPPED — pending MCRMINI2 availability.**

Not run. MCRMINI2 was not available this cycle. Running it from MCRMINI1 would only
re-derive the known result: Cycle 62A established the `fstream` block is **server-side for
this host** — handshake completes, `LIST_SUBSCRIPTIONS` and `SUBSCRIBE` are both answered,
then zero market-data frames arrive. Repeating that here would prove nothing.

**Pick this up at handover time**, on MCRMINI2:

```powershell
python -m engines.liquidation_collector collect --duration 60 --verbose 2
```

Report frames received, or the same zero-frame signature. A positive result is a real gain:
Binance throttles to one event per symbol-second while Bybit does not, so Binance is the
*less complete* record but the one every published prior is based on — having both venues is
strictly better than either alone.

---

## Summary for `ai_factory_main_current`

| # | Question | Answer |
|---|---|---|
| **T1** | Could `funding_executor` place a real order on MCRMINI2? | **NO — unconditional.** No network modules load at runtime (`socket` / `ssl` / `urllib.request` all absent). Writes only to `paper_trades` / `paper_position_exits`. **Safe to schedule.** |
| **T2** | Credentials needing an IP-allowlist update? | **NONE.** Zero credential reads across all 12 scheduled modules. **No allowlist edit blocks the move.** |
| T3 | CoinGecko reach | Public tier, **365 days**. `circulating_supply` **not historical at any date** — but **derivable** as `market_cap / price` (validated to -0.0087%). Already 429-ing ~2/day today. |
| T4 | smart_money / onchain | Both **public, unauthenticated** (Polymarket data-api; blockchain.info). smart_money holds WAL open **~21 min x 4/day**, and a **crashed 09-03 run left it orphaned for ~29h** — that, not a window overlap, broke the mirror. On-chain backfillable; smart_money **not** — 5 snapshots permanently lost. |
| T5 | Binance stream from MCRMINI2 | **Skipped — pending MCRMINI2.** Command to run at handover is above. |

### Carried forward (not in scope this cycle)

1. Quarantine `carry_executor` / `run_carry.py --live` before or during the move — it is
   dormant only because nothing schedules it (Cycle 55 recommendation, now handover-relevant).
2. Fix `smart_money` connection crash-safety and incremental commit; make the mirror
   WAL-tolerant. **The handover does not fix this.**
3. Carry `.env` to MCRMINI2 or alerting stops silently (`PRAXIS_ALERT_URL`, `TEAMS_WEBHOOK_URL`).
4. Consider a derived-supply backfill path for F1, with the CoinGecko-circularity caveat documented.
5. CoinGecko free tier is **already exceeded** at current volume — adding assets will worsen it;
   the new IP does not raise the ceiling.

---

*Cycle 63 — measurement only. No behaviour changed, no orders placed or simulated, no writes
to `crypto_data.db`. All database access was `mode=ro`.*
