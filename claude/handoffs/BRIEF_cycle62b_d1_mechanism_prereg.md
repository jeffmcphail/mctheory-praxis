# BRIEF — Cycle 62B: D1 Leveraged-Token Rebalance Mechanism Study

**Estimated Scope:** L (2–4 hr)
**Estimated Cost:** none (Binance archive is free)
**Mode:** Code (implementation + execution)
**Retro:** `claude/retros/RETRO_cycle62b_d1_mechanism.md`
**Predecessor:** Cycle 61 (`400cd64`) — forced-trade data audit
**Companion:** `BRIEF_cycle62a_forced_trade_collectors.md` (independent; runs in parallel)
**Screen:** `docs/FORCED_TRADE_SCREEN.md` · **Taxonomy:** `docs/FORCED_TRADE_TAXONOMY.md`

---

## What this is, and what it is not

**Not a strategy hunt.** Cycle 61 T3 established that all 50 genuine Binance USDT leveraged
tokens are **delisted since 2024-02**. Nothing studied here can be traded.

This is a **mechanism test of the entire forced-flow premise**, using the one scenario
Praxis has with a large sample: ~28,500 rebalance events, on an instrument whose flow
direction is knowable in advance.

The logic: if mechanically forced rebalancing does **not** produce exploitable price
pressure *here* — with a huge sample, a mechanical trigger, and perfect foreknowledge of
direction — then the forced-flow thesis is weak in general, and we learn that after one
cycle instead of five. If it **does**, that justifies searching for a live equivalent
product. Both branches are decisive, which is the property that makes the cycle worth
running on a dead instrument.

---

## T1 — Establish the actual rebalance mechanics FIRST. This is a gate.

`docs/FORCED_TRADE_TAXONOMY.md` classifies D1 as `SCHEDULED` (daily reset).

**That classification may be wrong for Binance specifically.** Binance's leveraged tokens
(BLVT) used a **variable-leverage** model with rebalancing triggered by leverage drifting
outside a target band — not a fixed daily reset at a fixed time, as US leveraged ETFs use.
Chat is **not certain** of this and is flagging it rather than asserting it. The taxonomy
entry should be corrected or confirmed by this task.

Establish, from Binance documentation and from the data itself:

1. Is the rebalance **scheduled, triggered, or both**?
2. What is the trigger condition, and what is the target leverage band?
3. Is there a published rebalance **time**, or is it opportunistic?
4. Can a rebalance be **detected in the token's own price/NAV series** — e.g. as a
   discontinuity between the token's return and the leverage-multiple of the underlying's
   return?

**The trade design depends entirely on this answer, so it is a gate.** If the mechanism is
triggered rather than scheduled, say so and re-derive the signal before proceeding. Report
the answer in the retro **before** any backtest exists.

If the mechanism cannot be established with confidence, **stop and report that** — a study
built on a guessed trigger measures nothing.

## T2 — Pre-registration, committed before any result exists

Write `claude/handoffs/CYCLE62B_D1_PREREGISTRATION.md` and **commit it standalone** before
writing any measurement code.

It must contain:

### Hypothesis

> A leveraged token must increase exposure after the underlying rises and decrease it after
> it falls — mechanically, and price-insensitively. That flow is predictable in direction,
> and approximately in size, from the underlying's move **before** the rebalance occurs, and
> it exerts measurable pressure on the underlying.

### Screen answers — all three, in writing

Per `docs/FORCED_TRADE_SCREEN.md`:

- **Q1 — alpha per trade vs cost per trade, on an envelope, BEFORE backtesting.**
  One rebalance per token per event. Estimate the rebalance flow size relative to the
  underlying's ADV. **If the flow is under ~1% of ADV, say so** — a mechanism that moves
  nothing is not tradeable however forced it is, and that is worth knowing before the
  backtest rather than after.
- **Q2 — who is compelled, and can we reach them?**
  The compelled party is the issuer's rebalancing agent. The accessibility half has an
  honest answer: **this product is dead, so it is not accessible.** State that plainly, and
  state that the study's purpose is mechanism validation rather than deployment.
- **Q3 — what carries the volatility, and does the backtest compute it?**
  Answer at design, then confirm at implementation. The Cycle 57 defect lived in the gap
  between those two answers.

### Pre-declared targets

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
  whatever granularity the mechanism from T1 implies.
- **Control:** matched non-event periods, same time-of-day, same volatility bucket. Cycle 61
  T1 showed why this matters: raw event ratios sat slightly above 1.0 purely because a
  short-window mean was compared to a 24h median of a right-skewed series. The matched
  control is the number to read.
- Report **gross pressure in bps first**, then net of a stated cost model.
- Report the **event count actually used**, not the theoretical 28,500.

## T4 — Fix the leveraged-pattern substring bug

`DEFAULT_LEVERAGED_PATTERNS` in `engines/xsec_reversal/universe.py` is matched with `in`,
not `endswith`, so:

- `JUPUSDT` (Jupiter) contains `UPUSDT`
- `SYRUPUSDT` (Maple SYRUP) contains `UPUSDT`

Both are ordinary spot assets, and both were **silently deleted from the Cycle 60 study
universe**. Chat wrote the warning about exactly this trap into the same file — for the
tokenized-equity `endswith("BUSDT")` rule, which would have deleted BNBUSDT, SHIBUSDT,
ARBUSDT and others — and then left the identical class of bug one constant above it.

- Apply the structural fix from `engines/forced_trade/leveraged.py:split_leveraged`:
  **require the implied underlying to actually trade** before treating a name as a leveraged
  token. Structural, not another pattern list.
- Add a test asserting `JUPUSDT` and `SYRUPUSDT` survive the filter.
- Note in the retro that Cycle 60's universe was 2 symbols short of ~655. Impact is small,
  the bug is real, **and the Cycle 60 verdict does not change.**

---

## Prohibited

- **Do not proceed past T1** if the rebalance mechanism is unclear — report instead.
- Do not tune any threshold to reach the pre-declared target.
- No live-product search in this cycle (gated on branch C).
- No ML (tier 3, gated on a validated scenario existing).

---

## Hand back

Retro containing:

1. **The T1 mechanism answer** — scheduled vs triggered, and how it was established
2. The pre-registration commit hash, timestamped **before** any measurement code
3. Gross pressure in bps, with the matched control alongside
4. Net of the stated cost model
5. Actual event count used
6. **Which decision-tree branch, and why**
7. Confirmation of the T4 fix with its test

Final step, standing:

```
.venv\Scripts\python split_zip.py zip --repo-delta
```

---

*Last updated: 2026-08-19 (Chat: praxis_main_current)*
*Changes: Initial brief. Cycle 62B D1 mechanism study on the delisted Binance leveraged-token
universe — mechanism gate, pre-registration requirement, matched-control measurement, and the
leveraged-pattern substring fix.*
