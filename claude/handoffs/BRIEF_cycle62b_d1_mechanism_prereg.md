# BRIEF — Cycle 62B: D1 Leveraged-Token Rebalance Mechanism Study

**Version 2 (2026-10-15).** Supersedes the version committed 2026-09-04. Nothing was
executed against v1 and no pre-registration was written, so amending in place is clean —
this is a work specification, not a tamper-evident record.

**What changed in v2:** the T1 gate grew from four sub-questions to six, folding in the
three feasibility risks Code raised when it first read the brief (granularity, NAV
availability, documentation survival). Those were answered in chat and never committed.

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

## T3 — The study

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
3. Gross pressure in bps, **with the matched control alongside**
4. Net of the stated cost model
5. Actual event count used
6. **Which decision-tree branch, and why**
7. T4 confirmed by test, with the re-measured symbol-class impact

Final step, standing:

```
.venv\Scripts\python split_zip.py zip --repo-delta
```

---

*Last updated: 2026-10-15 (Chat: praxis_main_current)*
*Changes: v2 — T1 gate expanded from four sub-questions to six, adding granularity and NAV
availability (raised by Code against v1, answered in chat, never committed). Documentation-
survival guidance and suggested task ordering added.*
