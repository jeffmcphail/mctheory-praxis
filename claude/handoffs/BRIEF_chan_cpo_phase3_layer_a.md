# CYCLE 59 PHASE 3 -- Chan CPO Layer A (unconditional) replication

**Estimated Scope:** M (30 min - 2 hr)
**Estimated Cost:** none (no API calls)

**Retro:** `claude/retros/RETRO_chan_cpo_phase3_layer_a.md`

## Governing document

`claude/handoffs/CYCLE59_CHAN_CPO_PREREGISTRATION.md`, committed at `e21e0e9`.
It is authoritative. Do not deviate from it. If something in it is ambiguous or
appears wrong, STOP and report rather than choosing -- an unspecified decision
resolved silently is a Chat defect, not a Code judgement call.

## Why the file name matters mechanically

`split_zip.py --repo-delta` resolves its anchor by walking `git log` over
`claude/handoffs/` and accepting ONLY files matching `BRIEF_*.md`. A handoff
named anything else can never anchor a delta, and an UNCOMMITTED brief is
invisible to it regardless of name. Cycles 55-60 broke both rules; the result
was a 102-file delta spanning eight cycles. This brief is committed as a
standalone commit before any Phase 3 code is written.

## Objective

Reproduce the UNCONDITIONAL (no ML) leg of Chan's GLD/GDX result, GROSS, and
compare against the pre-registered target: **Sharpe 1.947, annual return
17.29%**.

This is the single most diagnostic number in the whole arc. Layer A exercises
only the backtest engine, the data path, and the strategy logic -- zero machine
learning. If it cannot be reproduced after exhausting the four pre-registered
ambiguities, the framework-is-broken hypothesis is CONFIRMED and every prior
negative verdict in the atlas becomes suspect.

## Scope -- what to build (`engines/chan_cpo/`)

### `signal.py`
- `spread(t) = GLD_close(t) - GDX_close(t) * GDX_weight`
- causal EWMA recursions for `Spread_EMA` and `Spread_VAR`, `alpha = 2/lookback`,
  exactly as specified in prereg section 3
- z-score with denominator a PARAMETER (ambiguity A1: var vs std)
- Bollinger entry/exit, `exit_threshold = -0.6 * entry_threshold`
- intraday session filter 09:30-15:59 ET, force-liquidate 16:00
- per-combination intraday round-trip simulation -> daily return series
- `return_mode` a PARAMETER (A2: simple vs log)

### `unconditional.py`
- exhaustive 400-combination grid (5 `GDX_weight` x 10 `entry_threshold` x
  8 `lookback`) over the TRAIN window
- select the single combination maximising cumulative in-sample return
- FREEZE it; apply unchanged to the TEST window
- metrics: annual return, Sharpe, Calmar, 3-year cumulative

### `run_chan_cpo.py`
- CLI, `--validate` and `--verbose` with levels, defaulting to MAXIMUM
- all four ambiguities exposed as flags: `--zscore-denominator {var,std}`,
  `--return-mode {simple,log}`, `--adjustment {adj,unadj}`, plus the session
  bounds and grid as parameters. Nothing hard-coded.

## Windows (from the prereg, do not alter)

    TRAIN 2009-09-28 .. 2017-12-31
    TEST  2018-01-01 .. 2020-12-31

The walk-forward window is NOT touched in this phase.

## No-lookahead requirements

- EWMA recursions strictly causal
- grid selection uses TRAIN data only; the TEST window must not influence which
  combination is chosen
- add automated leakage self-checks that MUST PASS before any metric is
  reported (follow the pattern in `engines/infobar_lstm/leakage_checks.py`):
  perturb future bars and assert current signals are unchanged; assert the
  frozen combination is identical whether or not TEST data is present

## Reporting order -- mandatory, and the order is not arbitrary

1. **ROUND TRIPS PER DAY**, mean and median, on the TEST window. Report this
   FIRST, before any Sharpe. At the pre-registered 1.26 bps per round trip,
   5 round trips/day is ~15.9% annualised against a claimed 17.29% annual
   return. The trade count, not the spread, is what decides this experiment.
2. **GROSS metrics on TEST**: Sharpe, annual return, Calmar, 3-year cumulative,
   each against its pre-registered target.
3. **The winning parameter combination** (`GDX_weight`, `entry_threshold`,
   `lookback`) and which ambiguity settings produced it.
4. **Benchmarks per prereg section 7:**
   - a. buy-and-hold GLD over TEST
   - b. INTRADAY-ONLY long GLD (enter 09:30, exit 16:00, daily) -- the
     like-for-like control
   - c. the strategy's realised long/short exposure split and its correlation
     to (b)

   2018-2020 is a major gold rally. If the strategy does not beat (b), the
   result is intraday gold drift, not conditional mean reversion, whatever the
   Sharpe says.

## Ambiguity sweep

If the primary configuration misses the target, sweep A1-A4 and report the full
grid of outcomes. Iterating on those four is LEGITIMATE -- they are genuine
interpretation gaps in the paper. Iterating on anything else is not, and is
listed in prereg section 11 as invalidating.

## Do not, in this phase

- no transaction costs (Phase 4 -- Layer A is GROSS)
- no conditional/ML layer (Phase 5, gated on Layer A passing)
- no walk-forward (Phase 6)
- do not tune anything on TEST
- do not commit Kibot data or parquet panels

## Final step (standing, not optional)

    .venv\Scripts\python split_zip.py zip --repo-delta

## Hand back

The retro, the commit hashes, the delta zip, and the reporting items above in
the stated order -- round trips per day before any Sharpe.
