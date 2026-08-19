# The Forced-Trade Screen

*A design-stage filter applied BEFORE any backtest is written.*

**Status:** Cycle 61 deliverable. Methodology document, permanent.
**Companion:** `docs/FORCED_TRADE_TAXONOMY.md` (the scenario catalogue),
`docs/REGIME_MATRIX.md` (the regime axis).

---

## Why this exists

Praxis has run ~60 research cycles. Nearly all produced a negative result, and the
negatives cluster into **two distinct failure modes** that must not be conflated:

| Mode | Shape | Examples | Fix |
|---|---|---|---|
| **1. Turnover death** | Real gross alpha, too many trades | Exp 1 (SP500 pairs), Cycle 60 (crypto reversal), Cycle 59 (Chan GLD/GDX) | Longer holding period |
| **2. Regime death** | Sound structure, the payment stopped | Engine 7 (funding carry) | Wait, or find another compelled counterparty |

Three of our four major negatives are mode 1, and **every one of them was predictable on
an envelope before a line of code was written.** The screen exists to make that
prediction mandatory rather than optional.

---

## The three questions

A strategy proposal must answer all three **in writing, before implementation**.

### Q1 — What is the holding period, and what is the alpha-to-cost ratio per trade?

    net edge = (alpha per trade) − (cost per trade)

Estimate both sides on an envelope. If the ratio does not clear **~3×** on paper, it will
not clear in a backtest, and building the backtest is a waste of a cycle.

**The empirical regularity to lean on** (measured, Cycle 59, on real GLD/GDX minute data):

| Lookback | Alpha per round trip | Round trips/day |
|---|---|---|
| 30 min | 0.385 bps | ~50 |
| 720 min | **1.498 bps** | ~3.3 |

**Alpha per trade rises with holding period; cost per trade is roughly fixed.** Roughly
4× more alpha per trade for 15× less trading. In the same study only **47 of 400**
parameter cells earned more per trip than a trip cost — the viable region exists and it
is the low-turnover corner.

**Rule of thumb:** anything implying more than a few trades per week starts in a hole.

### Q2 — Who is on the other side, and why are they trading at a bad price?

Three possible answers, with very different life expectancies:

| Answer | Durability |
|---|---|
| "They made a mistake" | **Decays.** Someone will notice and compete it away. |
| "They are compelled" | **Persists.** A rule forces the trade regardless of price. |
| "They are paid to bear risk" | **Persists.** It is a premium, not an error. |

**The hard definition of compulsion** — the category leaks badly without one:

> **Is there a rule, contract, or mechanism that forces the trade regardless of the
> trader's opinion? If a human could simply choose not to trade, it is not compulsion.**

| Passes | Fails |
|---|---|
| Funding payment — the contract debits you every 8h | "Panic selling" — voluntary, opinion-driven |
| Margin liquidation — the engine closes you out | "FOMO buying" — voluntary |
| Index tracker buying an addition — mandate requires it | "Retail is dumb" — not a mechanism |
| Token unlock — vesting contract releases supply | "Weak hands" — a description, not a rule |

**Q2 has a second half that is easy to skip and must not be:** *can we still reach it?*
Forced flows are the most-studied phenomena in professional finance. Index-rebalance
arbitrage is a textbook strategy with dedicated desks. A flow can be genuinely compelled
and genuinely inaccessible. Answer both parts.

### Q3 — What carries the volatility, and does the backtest actually compute it?

**The Cycle 57 lesson, and the most expensive mistake in the atlas.**

Funding carry showed Sharpe +4.65. Investigation found funding is a *smooth hourly drip*
contributing almost **zero** volatility, while the **basis** term carried essentially all
of it — measured, `pnl_vol ≈ basis_vol` in every row. The paper-trading executor computed
`funding − TC` and omitted basis entirely. A near-zero denominator produces an enormous
Sharpe from a modest return.

Real basis-inclusive numbers were 1.5–1.9, not 4.65.

**So: name the volatile term before believing any risk-adjusted number, then confirm the
backtest computes it.** An omitted denominator is not a small overstatement — it is a
division by almost nothing.

---

## Retroactive validation

**A screen that cannot kill our known failures is not a filter.** Applied below using only
what was knowable at design time — no hindsight.

### Exp 1 — S&P 500 minute-bar pairs (Engine 1)

| Q | Answer at design time | Verdict |
|---|---|---|
| Q1 | Minute-bar pairs, ~2.7 trades/model-day, two legs crossed per round trip at ~4 bps each = ~11 bps cost. Gross alpha unknown but a minute-scale mean-reversion signal is worth single-digit bps. | **FAIL — ratio below 1× on the envelope** |
| Q2 | No compelled counterparty identified. The premise was a statistical relationship, not a mechanism. | **FAIL** |
| Q3 | Two-leg spread; both legs volatile; the backtest did compute it. | Pass |

**Screen result: KILLED on Q1 and Q2.** Actual outcome: gross +5.3 bps, TC 11.0 bps, net
−5.6 bps. The envelope was right.

### Cycle 60 — Crypto cross-sectional reversal

| Q | Answer at design time | Verdict |
|---|---|---|
| Q1 | 4h bars, daily rebalance, turnover ≈ 1.6 of book per rebalance. At 10 bps/side taker, **fees alone cost 16 bps per rebalance** — computable before any data was collected. Gross alpha per rebalance unknown. | **FAIL — fees alone exceeded plausible gross in T1/T2** |
| Q2 | Reversal after non-informational moves. A *partial* pass: liquidation cascades are genuine compulsion, but the strategy traded the whole cross-section, not the compelled subset. | **PARTIAL** |
| Q3 | Both legs volatile; correctly computed. | Pass |

**Screen result: KILLED on Q1.** Actual outcome: gross 42–43 bps vs cost 24–33 bps; net
edge failed the multiple-testing correction in all twelve cells.

**Note the Q2 partial — it is the seed of this whole program.** The compelled mechanism
was real; we diluted it by trading everything instead of only the forced flow.

### Cycle 59 — Chan GLD/GDX replication

| Q | Answer at design time | Verdict |
|---|---|---|
| Q1 | "Multiple round trips per day" at ~1.26 bps round-trip cost. The pre-registration's own worked example: 5 round trips/day ⇒ 15.9% annualised cost vs a claimed 17.29% return. **Even the optimistic assumption barely cleared.** | **FAIL** |
| Q2 | No compelled counterparty. A statistical spread relationship. | **FAIL** |
| Q3 | Single traded leg, price-volatile, correctly computed. | Pass |

**Screen result: KILLED on Q1 and Q2.** Actual outcome: 50 round trips/day, 0.425 bps
gross per trip against 1.26 bps cost — needs 3× more.

*(This cycle was still worth running: its purpose was framework validation, not edge
discovery, and it succeeded at that. The screen would have correctly predicted the
economic result while leaving the diagnostic value intact.)*

### Engine 7 — Funding carry (the control case)

The screen must not kill everything, or it is merely pessimism.

| Q | Answer at design time | Verdict |
|---|---|---|
| Q1 | Hold 3–12 days. `minHold = cost_RT × 365 / funding_ann`; at 9 bps and 9.6% funding, breakeven at 3.4 days. | **PASS** |
| Q2 | Perp longs **must** pay funding — contractual, every 8h, regardless of opinion. Retail-accessible venue. | **PASS** |
| Q3 | **This is where it would have caught the real problem.** Funding is a smooth drip; basis carries the volatility; the executor omitted basis. Q3 asks precisely this. | **FAIL — and correctly so** |

**Screen result: PASSES Q1/Q2, FLAGS Q3.** This is the right answer. Carry's structure is
sound and its economics work; the Sharpe was inflated by an omitted denominator. The
screen separates "this strategy is bad" from "this measurement is wrong" — which is the
distinction the atlas most needed and did not have.

### Validation summary

| Strategy | Q1 | Q2 | Q3 | Screen | Actual |
|---|---|---|---|---|---|
| Exp 1 | ❌ | ❌ | ✅ | KILL | Net −5.6 bps ✓ |
| Cycle 60 | ❌ | ⚠️ | ✅ | KILL | DSR fails all cells ✓ |
| Cycle 59 (Chan) | ❌ | ❌ | ✅ | KILL | Needs 3× more alpha ✓ |
| Engine 7 (carry) | ✅ | ✅ | ❌ | PASS + flag | Structure sound, Sharpe inflated ✓ |

**The screen kills all three turnover deaths, passes the one structurally sound strategy,
and flags exactly the defect that inflated it.** It discriminates rather than merely
rejecting. Adopted.

---

## Application rules

1. **Written before implementation.** Answers go in the cycle's brief, not in the retro.
2. **Q1 is arithmetic, not opinion.** Show the numbers. "Should be cheap enough" is not an answer.
3. **A Q2 failure is disqualifying on its own.** No mechanism means no reason to expect persistence, whatever a backtest says.
4. **Q3 is answered twice** — once at design (what should carry the volatility) and once at implementation (confirm the code computes it). The Cycle 57 defect lived in the gap between those two.
5. **Failing the screen does not always mean "do not run."** Cycle 59 failed Q1 and Q2 and was still correct to run, because its objective was framework validation. It means *do not run it expecting an edge* — state the real objective instead.

---

*Last updated: 2026-08-18 (Chat: praxis_main_current)*
*Changes: Initial version. Three-question design-stage screen (alpha-to-cost ratio,
compelled counterparty with accessibility, volatility denominator), the hard definition of
compulsion, and retroactive validation against Exp 1, Cycle 60, Cycle 59 and Engine 7.*
