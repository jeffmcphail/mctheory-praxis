# BRIEF — Cycle 63: Handover Blockers — Measurement Only

**Estimated Scope:** S–M (30–90 min)
**Estimated Cost:** none (read-only; no orders, no writes)
**Mode:** Code (measurement + verification)
**Retro:** `claude/retros/RETRO_cycle63_handover_blockers.md`
**Context:** `ai_factory_main_current` is scheduling the MCRMINI1 → MCRMINI2 collector
handover and is blocked on four answers. Two of them block scheduling outright.

---

## Rules for this cycle

**This is a MEASUREMENT cycle. No behaviour changes.**

- **Do not place, simulate, or trigger any order.** T1 involves reading an executor's code
  path; it does not involve running it.
- Read-only against `crypto_data.db`. No `VACUUM`, no migration, no bulk delete — the
  database is 24 GiB and mid-handover, and destructive operations need a fresh verified
  snapshot first.
- Where a question can be answered by *reading code* or by *making one read-only API call*,
  do that. Do not infer from documentation or from module names.
- If something cannot be established, **say so plainly** rather than reporting a guess.
  Two of these gate a machine handover.

---

## T1 — Can `funding_executor` place real orders? (BLOCKING)

`services/funding_executor_service.bat` runs
`python -u -m engines.funding_executor --trigger-source scheduled` on a schedule.
It will be re-registered on MCRMINI2, and `ai_factory_main_current` will not schedule it
until this is settled.

Chat's recollection is that this is a **paper** path — Cycle 54c produced a paper tie-out
and Cycle 57 found that executor basis-blind (`compute_exit`: funding − TC only). **That
recollection is explicitly not good enough here.** Read the code.

Answer:

1. Does `engines/funding_executor.py` contain a **live order-placement path** at all —
   any call that would reach an exchange's trading endpoint?
2. If yes, what gates it? An env var, a CLI flag, a config key, a hard-coded constant?
   Name the exact gate and its current value.
3. What does `--trigger-source scheduled` actually reach? Trace the path from that flag to
   whatever it ultimately does.
4. Does it load API credentials? From where, and are they present in `.env` today?
5. Is there a kill switch, and does it default to safe?

**Deliverable: a plain yes/no on "could this place a real order if scheduled on MCRMINI2
tomorrow," with the code path that justifies the answer.** If the answer is conditional,
state the exact condition.

## T2 — Do any collectors use IP-allowlisted API keys? (BLOCKING)

If any do, the allowlist must be updated for MCRMINI2's egress address **before** the move,
or those tasks fail on first run — and likely fail quietly.

1. Which collectors authenticate at all, versus using public market-data endpoints?
   Public endpoints need no keys and are IP-agnostic; those are not at risk.
2. For any that authenticate: which venue, which credential, and where is it loaded from?
3. **Is IP allowlisting actually enabled on those keys?** For Binance and Bybit this is
   visible in the API-management page; if the key is restricted, note it.
4. Prime candidates are `engines/funding_executor.py` and `engines/smart_money.py`, but
   check the full set rather than assuming.

**Deliverable: the list of credentials needing an allowlist update, or a clean "none."**
Chat has told `ai_factory_main_current` that Binance/Bybit allowlist edits are self-service
portal changes taking minutes; confirm or correct that if you can see the key settings.

## T3 — CoinGecko historical reach, especially `circulating_supply`

`market_data` is written by `engines.crypto_data_collector collect-market-data` and
`engines.unlock_market_data_collector` (30-asset universe added Cycle 62A). If its history
cannot be backfilled, `market_data` joins the ephemeral list and the handover has to treat
it like liquidations.

1. Which CoinGecko endpoint(s) do those collectors call, and on which plan/tier?
2. How far back can that tier retrieve **daily** history?
3. **Specifically: is `circulating_supply` retrievable historically, or only as a current
   snapshot?** This is the field scenario F1 (token unlocks) depends on, and Cycle 61 found
   the existing 5-asset universe useless for it.
4. What is the rate limit, and how long would refilling a 3-day gap across 30 assets take?

## T4 — Confirm the `smart_money` and `onchain` sources

Both are unconfirmed in the inventory, and **smart-money is the collector whose
`smart_money.db-shm` lock failed the nightly mirror at 09:05 on 2026-09-03.**

1. `engines/smart_money.py` — what venue or API does `discover --category ALL` and
   `snapshot` read? Does it authenticate? Does it hold a long-lived SQLite connection, and
   for how long per run?
2. `collect-onchain` — which provider, what history depth, what rate limit?
3. For both: is the data backfillable, and how far?

## T5 — Opportunistic: is the Binance liquidation stream reachable from MCRMINI2?

**Only when MCRMINI2 is available; skip otherwise and note it.**

Cycle 62A established the `fstream` block is **server-side** for MCRMINI1 — the handshake
completes, `LIST_SUBSCRIPTIONS` and `SUBSCRIBE` are both answered, then zero market-data
frames arrive. Suppression inside the TLS tunnel, which only Binance can do. Not a local
middlebox.

If MCRMINI2 egresses from a different address, the stream may work there. That would be a
real gain: Binance throttles to one event per symbol-second while Bybit does not, so Binance
is the *less complete* record but the one every published prior is based on. Having both
venues is strictly better than having either.

```powershell
python -m engines.liquidation_collector collect --duration 60 --verbose 2
```

Report frames received, or the same zero-frame signature.

---

## Explicitly NOT in scope

- No collector or backfill code (that is the next cycle, after these answers)
- No gap-detector implementation
- No changes to any scheduled task
- No orders, live or simulated
- No writes to `crypto_data.db`

---

## Hand back

Retro containing, in this order:

1. **T1 — the yes/no on live orders**, with the code path
2. **T2 — the allowlist list, or "none"**
3. T3 — CoinGecko reach, with `circulating_supply` called out separately
4. T4 — smart-money and on-chain sources, plus smart-money's connection lifetime
5. T5 — Binance stream result from MCRMINI2, or noted as skipped

Items 1 and 2 are what `ai_factory_main_current` is waiting on; put them first.

Final step, standing:

```
.venv\Scripts\python split_zip.py zip --repo-delta
```

---

*Last updated: 2026-09-04 (Chat: praxis_main_current)*
*Changes: Initial brief. Cycle 63 measurement-only cycle answering the four handover
blockers raised in the two-machine estate exchange, plus an opportunistic test of whether
the Binance liquidation stream is reachable from the target machine.*
