# RETRO — Cycle 54c: Funding Studio Analyze View + Health Strip + Live Rollup

Predecessors: Cycle 54 backend (`c8c70c5`…`33facdd`), 54b frontend
(`e7722b2`, `4ad6a2b`). 54c adds the **analytical layer**: a charted equity
curve, cross-session comparison, deep skip breakdown, an always-on
collector-health header strip, and the on-demand live rollup. With it, the
**paper-trading Funding Studio is feature-complete** (Control + Monitor +
Analyze). Paper-first; frontend holds zero trade logic.

## Commits

| SHA | What |
|---|---|
| `c8c70c5`…`33facdd` | Cycle 54 backend foundation |
| `e7722b2`, `4ad6a2b` | Cycle 54b frontend (control + live monitor) |
| `82a1461` | Cycle 54c: equity endpoint + Analyze view + health strip |

(Commits stay local — Jeff pushes to origin at the next session start.)

## RECON (R0c–R5c)

- **R0c (premise correction):** mcb's `EquityChart` is a hand-rolled `<canvas>`
  (index-based, no time axis); lightweight-charts (4.1) is used by
  `PriceChart`/`MCBPane`. For 54c's cross-session **overlay** (shared time
  axis + tooltips), lightweight-charts is required (the canvas can't overlay on
  time) — so we added it.
- **R1c = (b):** the equity series is computed **server-side**
  (`compute_equity_series`), not JS cumsum — keeping "frontend renders, backend
  computes" crisp and co-located with `compute_rollup`.
- **R2c:** already implemented in Cycle 54 — `GET /{id}` computes the rollup
  on demand when `pnl_rollup_json` is NULL (live). Added the **"computed live"**
  label this cycle.
- **R3c:** new `[Analyze]` tab; metric-table-always, overlay gated to
  same-window; cross-window normalization deferred.
- **R4c:** collector-health **App-header strip**, tri-state + near-limit.
- **R5c:** `lightweight-charts ^4.1.3` (matches mcb).

## Equity curve + the tie-out (verification of record)

`compute_equity_series` (controller) groups exits by `exit_timestamp` (so curve
points are unique on the time axis — multiple assets can exit the same instant)
and running-sums `net_pnl_usd`; `GET /api/sessions/{id}/equity` serves it
mode-aware (replay → harness DB, live → MAIN). `EquityCurve` renders it with
lightweight-charts as a **stepped line** (honest — the executor books P&L only
at exit; flat between).

**The tie-out is the check that matters:** the curve's terminal `cum_pnl_usd`
must equal `compute_rollup`'s `net_pnl_usd` (two computations of the same total
over the same exits). Verified exact, both gates, **corroborated from the
persisted rollups**: 0.50 → 28.60022 == 28.60022; 0.70 → 29.959065 ==
29.959065. The UI renders the tie-out inline (green "✓ terminal == rollup net").

## Comparison finding (the analytical payoff)

Same window (Q1 2025), two gates overlaid:

| | gate 0.50 | gate 0.70 |
|---|---|---|
| Trades (entries=exits) | **143** | 84 |
| Funding (gross) | +$85.80 | +$63.56 |
| TC | −$57.20 | −$33.60 |
| **Net P&L** | **+$28.60** | **+$29.96** |

The looser 0.50 gate trades **70% more**, collecting more gross funding — but
the extra marginal trades pay enough TC to net **slightly LESS**. This
corroborates the Cycle-53 finding: the 0.50 stream picks up marginal trades the
config-gate would zero; the disciplined 0.70 gate nets more on fewer trades.
The Analyze overlay makes that visible at a glance (two distinct curves).

## On-demand live rollup + "computed live" label (R2c)

A live session leaves `pnl_rollup_json` NULL (MAIN rows persist by
`session_id`); `GET /{id}` computes the rollup on demand. The RollupBar now
labels it: **"● computed live · not settled"** (amber) while running,
**"computed live"** once stopped, **"final · persisted"** for replay. This
isn't cosmetic — it signals a live snapshot may still change. Verified: a
running live session shows the amber label with correct sit-out zeros
(`entries=0/exits=0/net=$0.00`, `pnl_rollup_json` NULL).

## Collector-health header strip (R4c)

Always-on strip rendering `GET /api/health`, tri-state per table from the
actual fields: **fresh** (green) = `is_stale:false`; **stale** (red) =
`is_stale:true`; **empty** (amber) = `error:"empty table"` (benign sit-out,
explicitly NOT stale); **near-limit** (amber) = `staleness/threshold > 0.8`.
Foregrounds the funding tables; click expands full per-DB detail. Verified
live: `funding_alerts`/`paper_trades` amber-empty, **`onchain_btc` near-limit
at 89%** — the empty≠stale distinction holds on real data.

## Split preserved

The `[Analyze]` tab holds the analytical layer; **Monitor and Control are
structurally unchanged** (the only Monitor touch is the R2c label, a prop +
badge on RollupBar). Operational vs analytical stays clean.

## Verification

- `npm run build` clean (56 modules). lightweight-charts in the lockfile;
  `node_modules`/`dist` gitignored; no playwright in the committed lockfile.
- End-to-end via the UI (Playwright + headless Edge): kicked a Q1/0.50 replay
  for a distinct overlay; deep-dive (curve + tie-out + skip breakdown),
  comparison (table + overlay), health strip (tri-state incl. near-limit),
  live rollup (computed-live + zeros). Tie-out exact both gates.
- R3 isolation re-verified: MAIN `paper_trades` = 0 after the new Q1/0.50
  replay (replay rows live only in the harness DB).
- Safety-belt clean (equity endpoint read-only; frontend zero trade logic).
  Servers torn down; temp scripts removed.

## Cycle 54c status

- **Funding Studio frontend COMPLETE** — Control + Monitor + Analyze. The
  paper-trading control GUI is now feature-complete.
- **Real money STILL GATED (unchanged):** D8-live, `max_daily_loss_pct` wiring,
  SOL conservative sizing, human-in-the-loop, retire/coexist the
  PraxisFundingExecutor scheduled task.
- Deferred (noted, not blocking): intra-hold mark-to-market equity (realized-
  at-exit only), cross-window equity normalization.
