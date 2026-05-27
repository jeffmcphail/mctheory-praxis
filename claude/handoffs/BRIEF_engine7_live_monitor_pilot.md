# Cycle 41 -- Engine 7 live monitor pilot (BTC + ETH)

**Predecessor:** Cycle 40 -- Engine 7 full-universe paper reproduction
(commits `082459b` + `cd948d3`). Atlas Exp 13 verified CONFIRMED +
VERIFIED. Cycle 41 is item 41a from Cycle 40's "Open items" list.

**Mode:** RECON-then-implementation in a single cycle. The brief
asked for RECON first if details weren't obvious; Cycle 41 RECON
report (inline in chat) answered all 4 brief questions cleanly,
user approved Option A + Step 4 in-scope + dual-gate columns, and
implementation followed without a Cycle 41b.

**Goal:** Deploy a live funding-carry monitor using the Cycle 40
verified phase3 models on the 2 assets the live PraxisFundingCollector
already covers (BTC + ETH). Generate real-time signals into a new
monitoring table; observe sit-out behavior in the current bear-side
BTC funding regime as a forward confirmation of atlas's
unverified-2022-bear hypothesis.

## Scope (in)

- New `funding_signals` table in `data/crypto_data.db` (Rule-35-
  conforming compound PK on `(asset, timestamp)`).
- Extend `scripts/funding_monitor.py` with `--persist`, `--funding-source`,
  `--db` flags; switch funding-history source from CCXT to DB (default);
  fix the previously-wrong `DEFAULT_MODELS` path.
- New `services/funding_monitor_service.bat` (mirrors funding collector bat).
- New `services/register_funding_monitor_task.ps1` registers
  `PraxisFundingMonitor` task at 00:15 / 08:15 / 16:15 LOCAL time
  (~10 min after PraxisFundingCollector).
- Persist both gate outcomes per row: `above_gate` (P>0.70 atlas-
  recommended live) AND `above_gate_050` (P>0.50 atlas headline).
- Acceptance: deployed task + ≥1 funding cycle of observation.

## Scope (out)

- Universe extension to 6 assets (deferred to Cycle 42)
- Cross-venue funding spreads (Bybit, OKX, Hyperliquid)
- `--feature-mode` default fix in `run_cpo.py` (memory #23)
- Executor integration / live trading
- Real-time alerting (Slack/email)

## Operational frame

"Near-real-time, ≤8h post-event." The strategy trades at funding-window
cadence (8h), not microseconds. The LOCAL-time triggers inherit
PraxisFundingCollector's local-vs-UTC offset (Toronto UTC−4/−5), so
each funding event is captured within one local-trigger cycle.

## Expected first-cycle observation

Per Cycle 39 finding: BTC last-30d funding annualized −2.28%, pos_share
0.333; ETH similar. Both below the 5% min-funding-ann config threshold
in any of the 36 configs. Expected `above_gate = 0` and `above_gate_050 = 0`
for every funding window in current regime. Sit-out is the successful
observation outcome, not a failure of the monitor. If any
`p_profitable > 0.70` row appears, capture as additional data and
surface before commit.

## Memory anchors

- #4 (dotenv): unchanged (no new secrets surface in monitor)
- #5 (max validation + verbose): existing funding_monitor.py prints
  verbose progress; `--persist` is additive, not gated
- #12 (exit-code honesty): monitor and collector run as separate
  scheduled tasks so failures don't mask each other
- #13 (process pattern verification): monitor is a one-shot scheduled
  task, matching all 13 existing Praxis* collector tasks
- #14 (Rule 35): new `funding_signals` table follows the compound
  natural-key PK + ms timestamp + ISO datetime pattern
