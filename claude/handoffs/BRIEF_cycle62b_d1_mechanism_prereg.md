# BRIEF — Cycle 62B: D1 Leveraged-Token Rebalance Mechanism Study

**Version 3 (2026-10-16).** Supersedes v2 of 2026-10-15. Nothing is withdrawn. v3 folds in
what Cycle 66 established, which changes this cycle in two ways: the T1 gate is now a DESIGN
gate rather than only a feasibility gate, and T3's measurement instruments are specified
explicitly, including one that is ruled out.

**Why v3 exists.** Cycle 66 ran the A2 liquidation-cascade study against eight weeks of Bybit
ground truth and refuted it (`claude/retros/RETRO_cycle66_a2_labelled_validation.md`, `43c529b`).
Two of its findings bear directly on this brief and would have cost this cycle its verdict if
left out.

**Estimated Scope:** L (2–4 hr)
**Estimated Cost:** none (Binance archive is free)
**Mode:** Code (implementation + execution)
**Retro:** `claude/retros/RETRO_cycle62b_d1_mechanism.md`
**Screen:** `docs/FORCED_TRADE_SCREEN.md` · **Taxonomy:** `docs/FORCED_TRADE_TAXONOMY.md`

---

## What this is, and what it is not

**Not a strategy hunt.** Cycle 61 T3 established that all 50 genuine Binance USDT leveraged
tokens are **delisted since 2024-02**, last data 2024-02. Nothing studied here can be
traded, now or ever.

This is a **mechanism test of the entire forced-flow premise**, using the only scenario in
the taxonomy with a large sample: **~28,500 rebalance events**, on an instrument whose flow
direction is knowable in advance.

The logic: a leveraged token *must* increase exposure after the underlying rises and reduce
it after it falls. Mechanical, price-insensitive, direction predictable before the event. If
forced rebalancing does **not** produce measurable price pressure *here* — huge N, a
mechanical trigger, perfect foreknowledge of direction — then the forced-flow thesis is weak
in general and we learn that after one cycle instead of five. If it **does**, that justifies
hunting a live equivalent product.

**Both branches are decisive. That is the entire justification for running this on a dead
instrument.**

---

## T1 — Establish the actual rebalance mechanics FIRST. This is a hard gate.

`docs/FORCED_TRADE_TAXONOMY.md` classifies D1 as `SCHEDULED` (daily reset).

**Chat is not confident that is correct and is flagging it rather than asserting it.**
Binance's leveraged tokens (BLVT) are believed to have used a **variable-leverage** model
with rebalancing triggered by leverage drifting outside a target band — not a fixed daily
reset at a fixed time, as US leveraged ETFs use. Confirm or correct the taxonomy row.

**Sub-question 1 now decides whether this cycle runs at all, not merely how it runs.**
Cycle 66 found that liquidation cascades cannot separate cause from effect: price runs about
27 bps in the forced direction over the nine minutes *before* the first forced fill, so the move
triggers the forcing at least as much as the forcing moves the price. The entire justification
for running 62B on a delisted instrument is that a **scheduled** rebalance breaks that
simultaneity - the triggering move has already happened, and the forced flow arrives at a known
later time, which is what makes the pressure identifiable. **If T1 establishes that the
rebalance is drift-triggered rather than scheduled, this cycle inherits A2's endogeneity and
that advantage is gone. STOP and report.** Do not run T3 against a triggered mechanism and
describe the result as pressure; it would be the same unidentifiable mixture Cycle 66 spent
itself proving unidentifiable.

Six sub-questions. **All six are part of the gate; do not proceed past any unresolved one.**

1. Is the rebalance **scheduled, triggered, or both**?
2. What is the trigger condition, and what is the target leverage band?
3. Is there a published rebalance **time**, or is it opportunistic?
4. Is a rebalance **detectable in the token's own price/NAV series** — e.g. as a
   discontinuity between the token's return and the leverage-multiple of the underlying's?
5. **GRANULARITY.** Do 1m/1h klines exist in the archive for these tokens across 2020–2024?
   Cycle 61 worked in 1d bars. **If the mechanism is drift-triggered and only daily bars
   exist, the study can neither locate the event nor measure pressure around it.** Settle
   this at the gate; do not discover it in T3.
6. **NAV AVAILABILITY.** Is NAV in the archive at all, or only market price? BLVT NAV was
   published on a separate endpoint. Sub-question 4's detection method leans on NAV; if only
   market price exists, state that detection falls back to a return-discontinuity proxy and
   what that costs in confidence.

**On documentation:** the product has been delisted since 2024-02, so Binance's BLVT docs
may no longer exist. "Establish from the data itself" is an acceptable substitute — but say
which route was used and rate the confidence accordingly. An answer established from data
with stated confidence is fine; an answer asserted without saying how is not.

**If the mechanism cannot be established with confidence, STOP and report.** A study built
on a guessed trigger measures nothing.

## T2 — Pre-registration, committed standalone before any measurement code exists

Write `claude/handoffs/CYCLE62B_D1_PREREGISTRATION.md` and **commit it on its own** before
writing any measurement code.

### Hypothesis

> A leveraged token must increase exposure after the underlying rises and decrease it after
> it falls — mechanically, and price-insensitively. That flow is predictable in direction,
> and approximately in size, from the underlying's move **before** the rebalance occurs, and
> it exerts measurable pressure on the underlying.

### Screen answers — all three, in writing

Per `docs/FORCED_TRADE_SCREEN.md`:

- **Q1 — alpha per trade vs cost per trade, on an envelope, BEFORE backtesting.** Estimate
  the rebalance flow size relative to the underlying's ADV. **If the flow is under ~1% of
  ADV, say so** — a mechanism that moves nothing is not tradeable however forced it is, and
  that is worth knowing before the backtest rather than after.
- **Q2 — who is compelled, and can we reach them?** The compelled party is the issuer's
  rebalancing agent. The accessibility half has an honest answer: **this product is dead, so
  it is not accessible.** State that plainly, and state that the purpose is mechanism
  validation rather than deployment.
- **Q3 — what carries the volatility, and does the backtest compute it?** Answer at design,
  then confirm at implementation. The Cycle 57 defect lived in the gap between those two.

### Pre-declared target

State **what magnitude of price pressure, in basis points, would count as confirming the
mechanism** — before measuring it.

### Decision tree — every branch decisive

| Branch | Condition | Meaning |
|---|---|---|
| **A** | No measurable pressure | Forced-flow thesis is weak. The whole program downgrades — worth knowing after one cycle instead of five. |
| **B** | Pressure exists, below realistic cost | Mechanism real, uneconomic at retail. Same TC-bound shape as Exp 1 / Cycle 60 / Chan. |
| **C** | Pressure exists and exceeds cost | Mechanism validated. Justifies searching for a live equivalent product. |
| **D** | Pressure only in some regimes | First real evidence for the scenario × regime hypothesis. |

**A guard on branch A.** "No measurable pressure" is a valid reading only if the measurement
used an instrument capable of detecting it. See T3: a null from aggressor imbalance means the
instrument was wrong, not that pressure was absent. **Branch A requires a null in depth and
displacement against matched controls**, not a null in one-sidedness.

## T3 — The study

### What to measure, and what NOT to measure

Cycle 66 is the reason this section is explicit rather than left to implementation.

**Do not use aggressor imbalance as a pressure measure.** The Cycle 61 detector required
|imbalance| > 0.8 in a 60 s window and met it in **0% of labelled liquidation clusters >= $1M**,
including every one of the fifteen largest. Inside those clusters imbalance ran *lower* than at
random times (0.44 against a 0.49 null on BTC; 0.34 against 0.44 on ETH). Large forced flow
attracts counterparty aggression in proportion to its size, so the tape during a genuine forced
event is **more** two-sided than a quiet minute, not less. A null from imbalance is not a null
from pressure. Reading one as the other would send this cycle to branch A for the wrong reason
and retire the whole programme on an instrument error.

**Use these, which Cycle 66 showed do work:**

- **Hit-side book depth against a matched control.** The cleanest instrument in Cycle 66 by a
  wide margin: 0.66 (BTC) and 0.62 (ETH) of control at onset, against controls sitting at
  1.40 and 1.30. If the archive has no book data for the underlying at the needed granularity,
  say so at T1 rather than discovering it here.
- **Price displacement in bps**, signed into the forced direction, from a pre-event reference.
- **Spread**, reported but treated as a size-threshold indicator rather than a primary measure:
  Cycle 66 found it unchanged (1.00) at all but the four largest events, where it went to 5.7x.

### Method, ported from Cycle 66

- **Matched controls:** same hour-of-day, same trailing-volatility quintile, drawn from observed
  time at a stated minimum distance from any event. Report the control level beside every event
  figure. The matched control is the number to read - a raw ratio against a trailing median of a
  right-skewed series reads above 1.0 for free, which is how Cycle 61 produced a spurious result.
- **Start the profile at least 10 minutes BEFORE the event.** Cycle 66's single most important
  finding - that the move begins before the forced flow does - would have been invisible to a
  profile starting at t=0. For a scheduled rebalance this window is also where the predictable
  part lives.
- **Carry the profile to +1 hour.** Cycle 66's move never reverted: +25 bps (BTC) and +57 (ETH)
  an hour on, against a control at zero. Pressure that does not revert is a repricing, not a
  liquidity effect, and the difference decides whether branch C means anything tradeable.
- **Day-clustered standard errors, day-block bootstrap CIs.** Rebalance events cluster in time.
- **A cost grid, not a single number.** Cycle 66 used 2 / 9 / 20 / 30 bps round trip (maker,
  Hyperliquid taker, Binance spot taker, Cycle 55 real-money). Gross bps first, then the grid.

### Knowable entry vs look-ahead

Cycle 66's sharpest methodological result: measured from a cascade's last forced fill, reversion
was +6.9 bps at 5 minutes; measured from the first instant that fill was *knowable* to be the
last, it was -0.6. The entire apparent edge lived in 30 seconds of hindsight.

Here the same discipline cuts the favourable way, and that is the point of the cycle. State
explicitly, for whatever mechanism T1 establishes, **the earliest instant a trader could have
known the event was coming and in which direction**, and measure from there. Label any figure
that depends on information available only after the fact as look-ahead, as Cycle 66 did.

### The study itself


- Reuse the Cycle 60 archive path (`engines/xsec_reversal/archive.py`) for both the
  leveraged tokens and their underlyings. Note the **ms→µs timestamp switch guard** already
  in that loader (Binance changed spot archive timestamps on 2025-01-01; naive parsing sends
  2025+ data to the year ~55000).
- Measure underlying price behaviour around rebalance events — before, during, after — at
  whatever granularity the T1 mechanism implies.
- **Control: matched non-event periods**, same time-of-day, same volatility bucket. Cycle 61
  T1 showed exactly why this matters: raw event ratios sat slightly above 1.0 purely because
  a short-window mean was compared against a 24h median of a right-skewed series. **The
  matched control is the number to read.**
- Report **gross pressure in bps first**, then net of a stated cost model.
- Report the **event count actually used**, not the theoretical 28,500.

## T4 — Fix the leveraged-pattern substring bug (independent of T1–T3)

`DEFAULT_LEVERAGED_PATTERNS` in `engines/xsec_reversal/universe.py:138` is matched with `in`,
not `endswith`, so `JUPUSDT` (Jupiter) and `SYRUPUSDT` (Maple) contain `UPUSDT` and were
**silently deleted from the Cycle 60 study universe**. Chat wrote the warning about exactly
this trap into the same file — for the tokenized-equity `endswith("BUSDT")` rule — and left
the identical bug one constant above it.

- Port the structural fix from `engines/forced_trade/leveraged.py:split_leveraged`:
  **require the implied underlying to actually trade** before treating a name as leveraged.
  Structural, not another pattern list. Cycle 61 already built it and recorded the false
  positives in `outputs/forced_trade/t3_substring_false_positives.json`.
- Add a test asserting `JUPUSDT` and `SYRUPUSDT` survive the filter.
- **Re-measure the impact against the real archive list.** The "2 symbols of ~655" figure is
  Cycle 61's and should not be restated — the bug is a **class** (any base name ending in
  UP/DOWN/BULL/BEAR), not those two names. Report how many symbols the class actually
  catches. **The Cycle 60 verdict does not change.**

**Suggested order: T4 first.** It is independent, has a proven fix to port rather than
invent, and lands a real defect fix regardless of how the T1 gate resolves. Then T1, and
stop there if the mechanism turns out to be triggered or the granularity/NAV gaps bite.

---

## Prohibited

- **Do not proceed past T1** if the rebalance mechanism is unclear — report instead.
- Do not tune any threshold to reach the pre-declared target.
- No live-product search in this cycle (gated on branch C).
- No ML.

---

## Hand back

1. **The T1 mechanism answer** — scheduled vs triggered, how it was established, and the
   confidence rating. Plus explicit answers on granularity and NAV.
2. The pre-registration commit hash, timestamped **before** any measurement code
3. Gross pressure in bps, **with the matched control alongside**, measured in depth and
   displacement - never in aggressor imbalance (T3)
4. The profile from at least -10 min to +1 hour, so a repricing is distinguishable from a
   reverting liquidity effect
5. Net across the cost grid, and the earliest knowable entry instant the figures assume
6. Actual event count used
7. **Which decision-tree branch, and why** - and if branch A, state which instrument
   produced the null
8. T4 confirmed by test, with the re-measured symbol-class impact

Final step, standing:

```
.venv\Scripts\python split_zip.py zip --repo-delta
```

---

*Last updated: 2026-10-16 (Chat: praxis_main_current)*
*Changes: v3 - T1 sub-question 1 raised to a design gate (a triggered mechanism inherits A2's
endogeneity and kills the cycle's identification advantage); T3 given an explicit measurement
spec that rules out aggressor imbalance and ports Cycle 66's matched-control, pre-event window,
permanence, clustered-SE and cost-grid method; branch A given a guard against instrument error.*
