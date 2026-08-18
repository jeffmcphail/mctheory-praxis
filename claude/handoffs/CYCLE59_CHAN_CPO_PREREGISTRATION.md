# CYCLE 59 PRE-REGISTRATION -- Chan Conditional Parameter Optimization (GLD/GDX)

**Committed before any strategy code ran.** Data loaded and validated in Phase 2
(commit 2e3205c); no signal, grid, Sharpe, or comparison to Chan's figures had
been computed at the time of writing. The cost model in section 6 is calibrated
from MEASURED spreads -- data characterisation, not result tuning.

## 1. Purpose

NOT an edge hunt. A test of the INSTRUMENT.

Every negative result Praxis has produced is consistent with two rival
explanations that are perfectly confounded: the market is efficient, or our
backtest framework is broken. Replicating a KNOWN, PUBLISHED, profitable
historical result holds the market constant and tests the framework. Every
outcome is informative.

Target: Chan, Belov & Ciobanu, "Conditional Parameter Optimization" (March 2021),
GLD/GDX example. Paper in project as CPOPaper__Chan.pdf. This is the PARAMETER
paper, not the later Conditional PORTFOLIO Optimization paper.

## 2. Pre-registered targets (out-of-sample, 3 years ending 2020-12-31)

|                   | Unconditional | Conditional |
|-------------------|---------------|-------------|
| Annual return     | 17.29%        | 19.77%      |
| Sharpe            | 1.947         | 2.325       |
| Calmar            | 0.984         | 1.454       |
| 3-year cumulative | 73%           | 83%         |

**The paper reports NO transaction costs and NO slippage**, on a 1-minute
intraday strategy doing multiple round trips per day. These are almost certainly
GROSS figures from a vendor demonstration (PredictNow is Chan's own company).
That fact is the crux of the experiment.

## 3. Strategy specification (from the paper)

- GLD and GDX, 1-minute bars.
- **TRADE GLD ONLY.** GDX is signal-only, never traded. Load-bearing for the
  cost model (section 6).
- Spread(t) = GLD_close(t) - GDX_close(t) * GDX_weight
- EWMA recursions, alpha = 2/lookback:
    Spread_EMA(0) = Spread(0)
    Spread_EMA(t+1) = alpha*Spread(t+1) + (1-alpha)*Spread_EMA(t)
    Spread_VAR(t+1) = alpha*(Spread(t+1)-Spread_EMA(t+1))^2 + (1-alpha)*Spread_VAR(t)
- Z(t) = (Spread(t) - Spread_EMA(t)) / Spread_VAR(t)   [ambiguity A1]
- Bollinger, exit_threshold = -0.6 * entry_threshold:
    buy GLD    if Z < -entry_threshold
    short GLD  if Z > +entry_threshold
    exit long  if Z > exit_threshold
    exit short if Z < -exit_threshold
- Intraday only: signals 09:30-15:59 ET, force-liquidate 16:00. Multiple round
  trips/day permitted. Daily return = sum of round-trip GLD returns.
- Grid, 5 x 10 x 8 = 400 combinations:
    GDX_weight       {2, 2.5, 3, 3.5, 4}
    entry_threshold  {0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.25, 1.5, 2, 2.5}
    lookback (min)   {30, 60, 90, 120, 180, 240, 360, 720}
- UNCONDITIONAL (Layer A): exhaustive grid maximising cumulative IN-SAMPLE
  return; single best combination FROZEN and applied to TEST.
- CONDITIONAL (Layer B): GBDT predicts next-day strategy return from the 3
  parameters plus 112 technical features (8 indicators x {GLD, GDX} x 7
  lookbacks {50,100,200,400,800,1600,3200} min). Each test day: predict all 400,
  take argmax, use next day.

## 4. Data as delivered and validated (Phase 2, commit 2e3205c)

Source: Kibot, 1-minute, US Eastern wall clock, kept tz-naive (the only
ambiguous wall-clock hour is the autumn DST repeat at 01:xx, far outside any
session; naive ET keeps both legs on one clock without DST failures).

CONFIRMED SCHEMA -- bid/ask files are 10 columns, TWO FULL OHLC QUARTETS:
    0 date, 1 time, 2 bid_open, 3 bid_high, 4 bid_low, 5 bid_close,
    6 ask_open, 7 ask_high, 8 ask_low, 9 ask_close
Timestamp = cols 0+1 parsed as %m/%d/%Y %H:%M.
Bid/ask files contain NO volume and NO trade price. Anything requiring both a
traded price and a spread must join the two layouts on timestamp; build_panel.py
does this once.

Usable ranges (ET):
    with bid/ask: 2009-09-28 .. 2026-08-11, 4,243 session days,
                  1,946,045 common bars, median 390/390 RTH bars/day
    price only:   2009-01-02 .. 2026-08-11, 4,428 session days

CROSSED-BAR HANDLING (pre-registered):
Crossed bars are RETAINED, not deleted, behind --allow-crossed. Two columns with
a load-bearing distinction:
    spread / spread_bps            signed, negative on crossed bars -- DIAGNOSTIC ONLY
    cost_spread / cost_spread_bps  floored at 0 -- the ONLY spread a cost
                                   calculation may consume
Rationale: clipping at source would hide the defect the validator detects.
Magnitude justifying the floor: worst signed spread is -153.48 bps (GLD) /
-162.68 bps (GDX). Unfloored, the deepest crossed bars would credit the backtest
roughly 1.5% per crossing. After flooring: 0 negative bars in both panels.

Other quality notes: seven zero-ask bars at 04:00 dropped; GDX overnight jumps
all traced to genuine events (COVID, Brexit); no unhandled splits; adjustment
variants proven distinct from data (GDX factor 0.865425, volume inverse 1.1555).

## 5. Windows (AMENDED -- deviation stated in advance)

    TRAIN         2009-09-28 .. 2017-12-31   (bid/ask coverage start)
    TEST          2018-01-01 .. 2020-12-31   (Chan's EXACT OOS window)
    WALK-FORWARD  2021-01-01 .. 2026-08-11   (decay check, only after TEST)

- The paper implies a 2006 start and 80/20. Bid/ask begins 2009-09-28, giving
  73/27. TEST is Chan's exactly and untouched; only training breadth differs.
- Not a concession but a pre-registered robustness probe: after the primary
  result, re-run with TRAIN start 2012 and 2015, TEST fixed. A robust edge
  should be insensitive to training-window length; large sensitivity is itself
  a finding.
- COVID (March 2020) sits inside TEST, matching Chan.

## 6. Cost model (DECIDED; corrects the Phase 2 two-leg figure)

**The strategy trades GLD only.** The "round-trip crossing both legs = 10.6 bps"
figure applies to a two-leg pairs trade -- that is atlas Exp 1's geometry, not
Chan's. GDX's 4.51 bps is IRRELEVANT to cost because GDX is never traded. This
materially changes the economics and is precisely why GLD/GDX may survive where
the SP500 two-leg pairs did not.

MEASURED, RTH, locked bars excluded (quoted spread):
    GLD  FULL median 0.79 / p95 1.45      TEST median 0.79 / p95 1.14
    GDX  FULL median 3.52 / p95 5.58      TEST median 4.51 / p95 5.30

PRIMARY cost assumption:
- QUOTED spread on GLD, per bar, from cost_spread_bps (floored).
- Marketable fills: long entry at ask, long exit at bid; short entry at bid,
  short exit at ask.
- Round-trip spread cost ~= 0.79 bps (one full spread per round trip).
- Commission parametrised, default 0.23 bps/side (~$0.0035/share on GLD near
  $150), ~0.47 bps per round trip.
- **Total pre-registered round-trip cost ~= 1.26 bps.**

QUOTED chosen over EFFECTIVE deliberately. Effective (GLD 0.71 test) reflects
what the AVERAGE trade paid including passive and price-improved fills, which a
signal-driven marketable order cannot assume. Effective is retained as a
declared sensitivity, never as the primary.

SENSITIVITY GRID (declared now, not after results):
    cost multiplier {0x pure gross, 1x primary, 2x, 4x}
    plus one run substituting effective spread for quoted.

**MANDATORY: every result must state ROUND TRIPS PER DAY, reported BEFORE any
Sharpe.** Cost scales linearly with it and the paper leaves "multiple round trips
per day" unquantified. At 1.26 bps per round trip, 5 round trips/day = 6.3
bps/day ~= 15.9% annualised against a claimed 17.29% annual return. **The trade
count, not the spread, is the variable that decides this experiment.**

## 7. Benchmarks (MANDATORY -- the strategy is UNHEDGED)

Trading GLD alone on a GLD-GDX spread signal means the book carries outright GLD
exposure; it is not market-neutral despite the pairs-style signal. The TEST
window 2018-2020 contains a major gold rally (GLD roughly $120 -> $180), so a
long-biased strategy could harvest drift rather than mean reversion.

Force-liquidation at 16:00 excludes overnight drift, which mitigates but does not
eliminate this. Every reported result must therefore include:
  a. buy-and-hold GLD over the same window;
  b. an INTRADAY-ONLY long-GLD control (enter 09:30, exit 16:00, every day) --
     the correct like-for-like benchmark;
  c. the strategy's realised net long/short exposure split and its correlation
     to (b).
If the strategy does not beat (b), the result is intraday gold drift, not
conditional mean reversion, regardless of Sharpe.

## 8. Pre-registered ambiguities (PARAMETERS; test which reproduces Chan)

A1. zscore_denominator in {var, std}. The paper divides by VAR; a conventional
    Bollinger Z divides by STD = sqrt(VAR). Try both.
A2. return_mode in {simple, log} for round-trip aggregation.
A3. eighth_indicator: paper names only 7 (BB Z-score, Money Flow, Force Index,
    Donchian Channel, ATR, Awesome Oscillator, ADX). Parametrise a candidate
    (RSI or %B) and STATE which was used.
A4. adjustment in {adj, unadj}. Unadjusted = actual traded prices, likely
    primary; test both.

Iterating on A1-A4 to hit the target is LEGITIMATE -- they are interpretation
gaps in the source. Iterating on anything else is not.

## 9. Success criteria

LAYER A (unconditional, GROSS, no ML) -- the diagnostic that matters most:
- Reproduce Sharpe 1.947 +/- 0.2 AND annual return 17.29% +/- 2pp
  => BACKTEST ENGINE VALIDATED.
- If not, iterate ONLY on A1-A4. If still missing after all four, the engine has
  a genuine defect: the framework-is-broken hypothesis is CONFIRMED and every
  prior negative verdict in the atlas becomes suspect.
- Layer B does not run until Layer A passes.

LAYER B (conditional, GROSS): reproduce ~2.325. The uplift is modest
(1.947 -> 2.325), so most of the edge is the mean reversion, not the ML.

## 10. Decision tree (declared in advance; every branch decisive)

  A. Cannot reproduce gross 1.947 after A1-A4
     -> FRAMEWORK BROKEN. Highest diagnostic value. All prior atlas negatives
        require re-examination.
  B. Reproduce gross, dies under 1x cost
     -> FRAMEWORK SOUND; every prior dead end vindicated. Published results are
        gross; transaction cost is the gap between a backtest and a tradeable
        strategy. Most likely outcome, and it answers the question that started
        this arc.
  C. Reproduce gross AND survives 1x cost AND beats benchmark 7(b)
     -> Genuine historical edge. Proceed to walk-forward 2021-2026 for decay.
  D. Reproduce gross, survives 1x, dies at 2x
     -> Marginal. Report as cost-sensitive, do not deploy, record the breakeven
        cost explicitly.

## 11. What would invalidate this experiment

Stated now so it cannot be rationalised later:
- Any parameter tuned on the TEST window.
- Selecting the unconditional combination using test-window data.
- Reporting a net figure without stating round trips per day.
- Changing the cost assumption after seeing a result.
- Iterating beyond A1-A4 to reach the target.
- Consuming the signed spread column in any cost calculation.
- Reporting a result without the section 7 benchmarks.
- Running the walk-forward before the TEST verdict is recorded.

## 12. Sequencing

  Phase 0  this document, committed.                        <- gate
  Phase 3  Layer A engine; TRAIN grid; freeze; TEST gross.
  Phase 4  cost overlay + sensitivity; round trips/day; benchmarks.
  Phase 5  Layer B conditional (gated on Layer A).
  Phase 6  walk-forward (gated on branch C or D).
