# Retro: Cycle 53 (D7+D8) -- backtest replay + monitor/executor config-gate fix

**Brief:** `claude/handoffs/BRIEF_cycle53_backtest_replay_config_gate.md`
**Date:** 2026-06-02
**Mode:** Build-then-verify, one cycle. Option 1 (build as-is, decompose the gap).
**Status:** DONE -- D7 decomposition reproduces atlas Exp 13 exactly;
criterion-6 structural gap CONFIRMED (corrects an initial mis-read);
D8 closes it at both monitor + executor layers; all verifications green.
**Predecessor:** Cycle 52 (`67157a5` + `f188a5b`).
**Commit:** `533ee75`

---

## Summary

D7 reframed "verified" away from the literal Sharpe-match target (the
wrong criterion for a deliberately funding-only executor) toward a P&L
*decomposition*: prove the executor's funding-only P&L equals atlas Exp
13's reported P&L minus a small, measured basis term, with config-gate
zeroing accounted for explicitly.

- **D7a** regenerates per-(asset, day) OOS predictions via `predict_model`
  over the cached OOS window (`data/funding_cache`, 2025-01-01..2026-03-26)
  + `phase3_models_funding.joblib`. Reproduces atlas `phase4_model_stats`
  to the digit (ADA 75 days / +9.45%, BTC 36 / +3.51%, etc.).
- **D7b** synthesizes a self-contained harness DB (funding_rates from the
  identical cache series; funding_signals + funding_alerts per gate stream)
  and runs the PRODUCTION executor against it with an injected post-OOS
  clock + relaxed caps.
- **D7c** decomposes per asset into shared-set (funding+basis-TC, residual
  ~1e-15) vs atlas-zeroed leak (zero% / execOnly%).
- **D8a + D8b** add the config-threshold gate (`ann_rate >= min_funding_ann
  AND pct_positive >= min_pct_positive`) at BOTH the monitor (alert
  emission) and the executor (11th risk check), enforcing the SAME strategy
  atlas verified.

Acceptance results:

| Criterion | Result |
|---|---|
| 1 funding-term exact match | PASS -- max \|Δ\| = 6.66e-16 USD/trade, both gates |
| 2 basis drag/asset | small (shared-set mean +0.3..+2.2 bps/hold), mostly favorable to atlas; resid confirms it |
| 3 config-gate zeroing | 3.3% @ 0.70 (SOL 33%), 17.0% @ 0.50 (XRP 35%, ETH 18%) |
| 4 concurrency relaxed | documented (∞ concurrency/notional/staleness/daily-loss; $500 kept) |
| 5 residual ~0 | ~1e-15 over shared set -- genuine, non-tautological |
| 6 live-monitor gate | structural gap CONFIRMED -> D8 fired |

D8 verification (post-fix): atlas-zeroed leak goes to **0 booked / 0.0000%
execOnly** at both gates; monitor suppresses exactly the leaked alerts
(243->235 @ 0.70, 702->583 @ 0.50); executor 11th-check smoke PASS
(NEG skip `config_thresholds_not_met`, POS enter).

Net change:
- `engines/funding_executor.py`: `now_func` injectable clock (optional,
  defaults to real now -- zero production change); `enforce_config_gate`
  flag (default True); JOIN extended to pull ann_rate/pct_positive/
  min_funding_ann/min_pct_positive; RiskChecks gains 5 fields (11th check
  `config_thresholds_met` + 4 forensic values); `all_ok()` + skip-reason
  updated; docstring 10->11 checks.
- `scripts/funding_monitor.py`: `run_inference` captures min_pct_positive
  + computes config_ok; `persist_signals` writes min_pct_positive;
  `process_alerts` suppresses alerts when config_ok is False.
- `scripts/migrations/cycle53_funding_signals_add_min_pct_positive.py` (new):
  idempotent ALTER TABLE ADD COLUMN.
- `engines/crypto_data_collector.py`: init_db funding_signals gains
  min_pct_positive (fresh-DB symmetry).
- `scripts/cycle53_backtest_replay.py` (new): the D7+D8 harness.
- `outputs/cycle53_backtest_replay/`: SUMMARY.md + decomposition.json + harness DBs.
- `data/crypto_data.db`: funding_signals.min_pct_positive added (126 rows
  -> NULL, treated as config-not-verifiable by the executor; the next
  monitor run repopulates).

---

## The self-correction story (criterion 6)

This is the cycle's central lesson and worth recording verbatim for future
recon discipline.

**Initial recon claim (WRONG).** A first probe over ADA + BTC found
`nonzero_realized == P>gate_days` (75==75, 36==36) and I generalized to
"config-gate zeroing = 0% at both gates -> the RF soft-gate fully subsumes
the hard gate -> not a Cycle 54 blocker." I surfaced this as a *favorable*
finding.

**Two compounding errors:**
1. **Over-generalization from a 2-asset sample.** ADA and BTC happen to have
   zero zeroing. SOL (33% @ 0.70) and XRP/ETH (up to 35% @ 0.50) do not.
2. **A buggy detector.** My first decompose keyed config-gate zeroing on
   `funding_term == 0`. But config-gating never zeroes the funding term --
   `run_funding_single_day` returns `daily_return = 0` (and `gross = 0`)
   while the funding series is still non-zero. The correct key is
   `atlas_gross == 0 and atlas_net == 0`. The buggy detector reported 0.0%
   everywhere, falsely corroborating the over-generalization.

**What caught it: criterion 5's independent residual.** When I fixed the
residual to be non-tautological -- basis summed INDEPENDENTLY from
regenerated spot/perp price returns, not defined as `atlas - exec` -- the
residual stopped being identically zero and surfaced an exact, quantized
`+0.0008` per offending trade (one round-trip TC). That signature
(`n_zeroed x TC`, matching 8x0.0008=0.64% @ 0.70 and 119x0.0008=9.52% @
0.50 to the digit) is what forced the investigation that found the real
config-gate zeroing. **Had I left basis_drag defined as `atlas - exec`,
residual would have been 0 by construction and the gap would have shipped
silently.** The rigor of an independent cross-check -- not a tautological
one -- was the whole ballgame.

Discipline takeaways: (a) never generalize a per-asset rate from 2 of 6
assets; (b) a "verification" residual must difference INDEPENDENT sources
or it verifies nothing; (c) when the credibility-cycle framing says
"surface before continuing," a *favorable* early finding deserves the same
adversarial scrutiny as an unfavorable one -- I surfaced the wrong answer
confidently, and only the later rigor saved it.

---

## "Unverified strategy since Cycle 41" -- structural finding

atlas Exp 13's headline Sharpe (+4.65) and per-asset cum returns are
INCLUSIVE of `run_funding_single_day`'s config gate (Condition 1 +
Condition 2). The live monitor, since the Cycle 41 alerting build, has
gated alerts on **argmax-P > gate ALONE** -- it never re-applied those
hard thresholds. The executor (Cycle 51/52) then books any alert.

Therefore the deployed system has been a DIFFERENT strategy than the one
atlas verified: it would book trades atlas zeroed. The decomposition
quantifies the divergence -- 3.3% of high-conviction (P>0.70) signals,
rising to 17% at P>0.50, where the leaked trades net **-0.90%** (atlas's
gate was correctly skipping losers). atlas's Sharpe does not carry forward
to that system.

It has not bitten only by luck: `funding_alerts` has stayed at 0 rows
through the current sit-out regime (no natural P>0.70 firings), so the
mismatch has never produced a real (paper) trade. The fix lands before
that regime changes.

---

## D8 design decisions

### Same strategy definition enforced at two layers (approved Option 1)

- **D8a (monitor, source fix):** funding_alerts becomes atlas-faithful --
  no spurious push notifications, nothing for the executor to book.
- **D8b (executor, defense-in-depth):** an independent 11th check so a
  real-money entry never rides on an unverifiable or config-failing signal,
  regardless of what wrote the alert.

The two are deliberately redundant: D8a stops the noise at the source; D8b
guarantees the invariant at the gate where money would actually move.

### funding_signals schema: 3 of 4 fields already present

Cycle 41's monitor write-path already persisted `min_funding_ann`,
`ann_rate`, `pct_positive`. Only `min_pct_positive` was missing -- a
5-line idempotent ALTER TABLE migration. The 126 pre-existing rows get
NULL; the executor treats any NULL config field as
**config-not-verifiable -> skip** (conservative: no real money on a
strategy definition you can't confirm). The next monitor run repopulates
with the full four.

### now_func injection over monkeypatch

The executor keys staleness / exit-elapsed / daily-loss on `datetime.now()`.
Replaying 2025 data needs a fixed post-OOS clock. `now_func` was cleanly
factorable (single `self._now` wrapper, defaults to real UTC now), so it
went in as an optional constructor arg with zero production behavior
change -- preferred over a test-only monkeypatch per the brief.

### enforce_config_gate flag for reproducible baseline

The executor's 11th check is production-on by default. The flag exists so
the backtest can reproduce the pre-fix leak deterministically
(`enforce_config_gate=False`) in the same script that demonstrates the fix
(`=True`). Production never sets it False.

---

## Decomposition mechanics (for future readers)

- Per trade: `exec_pct = funding_term - TC`; `atlas_net = clip(spot_ret +
  perp_ret + funding_term - TC)`; so `atlas_net - exec_pct = basis_term`
  exactly (clip never bites for carry-sized returns).
- The harness sets `signal_timestamp = day-midnight ms` and reads
  funding_rates from the identical cache series, so the executor's window
  `(signal_ts, signal_ts + hold_days*86400000]` is byte-identical to
  atlas's `(hold_start, hold_end]`. That is WHY criterion-1 matches to
  6.66e-16 (float epsilon), not approximately.
- Shared set = days atlas booked (atlas_net != 0). Zeroed set = days atlas
  zeroed (config gate). Basis stats + residual computed over the shared set
  only, so they describe genuine basis -- the zeroed days are reported
  separately as the leak.

---

## Acceptance criteria

| # | Criterion | Status |
|:-:|---|:-:|
| 1 | D7 PASSED (corrected detector), commit as-is | ✅ funding match 6.66e-16; resid ~1e-15 |
| 2 | D8a monitor config-gate; re-decomp zeroed booked=0 | ✅ alerts 243->235 / 702->583 |
| 3 | D8b executor 11th check; smoke confirms skip path | ✅ NEG skip / POS enter |
| 4 | funding_signals schema extended (migration) | ✅ min_pct_positive added |
| 5 | Safety-belt grep clean | ✅ only the docstring prohibition line matches |
| 6 | Standard commit + push + SHA insertion follow-up | ✅ |
| 7 | Retro captures the four required threads | ✅ this file |

---

## Cycle 54 real-money unblock status

**Unblocked, gated on D8 being live.** The strategy the live system runs
now matches the one atlas verified. Remaining pre-real-money items:

- **Confirm the fix is on the scheduled path.** PraxisFundingExecutor
  picks up the new executor automatically next trigger; the monitor task
  picks up D8a next run. No natural alert has fired (sit-out regime), so
  the first real exercise will be the first P>0.70 + config_ok signal.
- **Basis drag is small but SOL is the watch item** -- shared-set basis is
  mostly favorable to atlas, but SOL's per-hold basis is the noisiest
  (the one asset where funding-only can overstate). Size SOL conservatively.
- **`max_daily_loss_pct` still unwired** (carried from Cycle 52); wire to
  portfolio notional before real money.
- **Exchange API / order state machine / real KILL_SWITCH path** -- the
  Cycle 54 build proper.
- Standing queue unchanged (44d, 44b, 44h-refactor, 44q, per-venue funding
  health, TEAMS_WEBHOOK_URL fallback removal).
