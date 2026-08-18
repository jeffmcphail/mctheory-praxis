# CYCLE 60 CLOSEOUT — Tiered Cross-Sectional Reversal × Crypto

**Status:** VERDICT REACHED on the in-sample test. OOS holdout **NOT spent** —
nothing passed IS, so an OOS run would be uninformative.

**Result in one line:** the gross tier structure predicted by the hypothesis is
**confirmed and strong**; the net-of-cost edge is **not established** — the
multiple-testing correction fails decisively in all twelve cells.

---

## 1. Verdict against the pre-registered criteria

| # | Criterion | Result |
|---|---|---|
| 1 | Net Sharpe > 0.85 across plausible spreads | **MARGINAL FAIL.** T2 = 2.03/1.61/**0.82**/−0.13 at 0.5×/1×/2×/4×; T3 = 1.86/**0.85**/−0.31/−1.73. Neither survives a 2× spread assumption. |
| 2 | Deflated Sharpe > 0.95 vs the declared grid | **FAIL, decisively.** All 12 (tier × holding) cells fail. Best DSR = **0.337** (T2, h=24); the T2 h=6 winner scores **0.015**. |
| 3 | Mean IC negative | **PASS**, and monotone by tier: −0.0137 / −0.0806 / −0.0998. |
| 4 | `demean` beats `none` | **UNTESTABLE AS DESIGNED** — see §3. The control was inert; the beta-dispersion alternative was never ruled out. |
| 5 | Survives IS → OOS | **NOT RUN.** Nothing passed IS; and the holdout is already weakened (Amendment 2, D1). |
| 6 | Capacity ≥ $100k | **PASS.** T1 $8.3M, T2 $1.86M, T3 $642k at 1% ADV participation. |

**Decision-tree outcome:** between **A** (no tier passes) and **D** (gross
positive, net negative). The pre-registered verdict is therefore **NEGATIVE on a
tradeable edge**, with a substantive positive finding on structure (§2).

---

## 2. What IS established (and is worth keeping)

Best IS config per tier, spread multiplier 1.0:

| Tier | best config | gross SR | net SR | gross bps | cost bps | mean IC |
|---|---|---|---|---|---|---|
| T1_liquid | f=6, h=24, q=0.1 | 0.558 | 0.097 | 24.00 | 19.8 | −0.0137 |
| T2_mid | f=12, h=6, q=0.1 | 3.548 | 1.613 | 43.31 | 23.6 | −0.0806 |
| T3_illiquid | f=24, h=6, q=0.1 | 3.830 | 0.846 | 42.23 | 32.9 | −0.0998 |

1. **Cross-sectional reversal in crypto is real and liquidity-tiered.** Mean IC
   is negative in every tier and rises monotonically in magnitude as liquidity
   falls (−0.014 → −0.081 → −0.100). This is an IC result, computed from raw rank
   correlations, independent of any cost model.
2. **It reconciles Fayez (2026).** That study's ten liquid USDT perps map onto
   **T1**, which is precisely the tier where we also find essentially nothing
   (net 0.097, IC −0.014). Their null was not wrong; it was tier-specific.
3. **Costs, not signal, are the binding constraint.** Gross alpha of 42–43 bps
   per rebalance is real and large; cost of 24–33 bps eats most of it, and the
   remainder does not clear a best-of-N correction.
4. **The concentration pattern is monotone.** q=0.1 wins in every tier, and
   gross Sharpe declines monotonically through q=0.2 and q=0.3 — the signal is
   strongest in the extreme ranks, which is what a genuine cross-sectional
   effect should look like (and is *not* what noise looks like).

---

## 3. DEFECT D4 — `demean` is a no-op, so the beta control never ran

Criterion 4 reported `demean` and `none` as **identical to three decimals, on
both the best and the median**, in all three tiers. That is not a violated
criterion; it is a bug in the experiment design.

**Mechanism.** `residualize(mode="demean")` subtracts the cross-sectional mean —
the *same scalar for every symbol at that timestamp*. `build_positions` ranks the
signal. **Ranking is invariant to subtracting a constant**, so demeaned and raw
signals produce identical rankings, identical positions, and identical P&L,
always. (Winsorisation is also shift-equivariant, so it does not break the
identity either.)

**Consequences.**
- The pre-registered control against "this is just beta dispersion, not relative
  value" **was never actually exercised**. Criterion 4 is unresolved, not passed.
  Only `residualize="beta"` — which subtracts a *per-symbol* βᵢ·r_mkt — can
  distinguish the two.
- The 96-config grid contained only **48 distinct configs**, each duplicated, so
  each DSR cell had 12 real trials rather than 24.

**Does the duplicate correction rescue anything?** No. For the T2/h=6 winner the
observed per-period Sharpe is 0.0844 against an expected-max under the null of
0.158 at 24 trials and 0.140 at 12. It remains below the null expectation either
way. The DSR verdict stands.

---

## 4. Pre-registered prediction about the untouched OOS window

Recorded **before** any OOS run: IS-only gross Sharpe exceeds full-sample gross
Sharpe for both live tiers (T2: 3.548 vs 2.490; T3: 4.200 vs 3.147). Since
full = IS + OOS, the OOS window (2024-07 → 2026-06) must be **weaker** than the
IS window. Any future OOS test should be expected to underperform IS, and that
expectation is on the record in advance.

---

## 5. Proposed atlas entry

- **Cell:** `MEAN_REVERSION × CRYPTO` — previously empty; now filled.
- **Engine:** cross-sectional reversal, dollar-neutral, liquidity-tiered.
- **Verdict:** ❌ **NO TRADEABLE EDGE** (net), ✅ **STRUCTURE CONFIRMED** (gross).
- **Conditions:** 600 Binance USDT symbols (survivorship-free, 169/600 delisted),
  4h bars, IS 2021-01 → 2024-06, tiered assumed spreads 3/15/40 bps,
  10 bps/side taker fee, non-overlapping rebalancing.
- **Key numbers:** gross SR 0.56 / 3.55 / 3.83 by tier; mean IC −0.014 / −0.081 /
  −0.100; best net SR 1.61 (T2) but DSR 0.015; best DSR anywhere 0.337.
- **Why it failed:** transaction costs. At 10 bps/side and turnover ≈ 1.0–1.7,
  costs consume 24–33 bps of a 42–43 bps gross signal, and the residual does not
  survive a best-of-N correction. **Same TC-bound pattern as atlas Exp 1.**
- **Open threads (do not re-run blind):**
  1. the beta-dispersion control (D4) was never executed;
  2. fee tier is the untested lever — at maker or VIP fees the arithmetic changes
     materially, but that must be pre-registered as a *factual* execution
     assumption, not tuned after seeing results;
  3. the OOS window remains unspent and is predicted weaker (§4).

---

## 6. Cross-cutting lessons for the framework

- **A control that cannot fail is not a control.** Demeaning looked like a
  rigorous beta adjustment and was arithmetically inert under rank-based
  position construction. Every pre-registered control should be paired with a
  test that it *can* change the output.
- **Report identical numbers as a red flag, not a result.** The bug was visible
  only because the criterion-4 line printed best *and* median for both arms; a
  single summary statistic would have hidden it.
- **The gross/net split keeps paying.** Three cycles running (Exp 1, Cycle 59
  design, Cycle 60), the finding is the same shape: gross alpha is real,
  transaction cost is the binding constraint. That is now the platform's most
  replicated result and belongs in the cross-cutting principles.
