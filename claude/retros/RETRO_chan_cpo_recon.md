# Retro: Cycle 38 RECON -- Chan/Burgess pairs trading code archaeology + design memo

**Brief:** `claude/handoffs/BRIEF_chan_cpo_recon.md`
**Date:** 2026-05-15
**Mode:** Read-only investigation (4 parallel streams, single pause point)
**Status:** DONE
**Predecessor:** Cycle 37 (`f732c3e` + `ebfb438`); atlas Exp 1 NEGATIVE per 2026-04-02 documented rebuild
**Commit:** `b0ace4b`
**Resolution:** Option D-prime -- close cycle with documentation cleanup only, no further investigation, no experimental run.

---

## Summary

Investigation cycle that scoped four streams (code archaeology, Chan paper review, info-bars feasibility, TC arithmetic stress test) to answer "should Cycle 39 attempt to revive Chan/Burgess pairs trading on SP500?" The cycle landed on **Option D-prime**: the atlas's existing analysis is sufficient, the conclusion is mechanical, and no experimental cycle would materially change the verdict at acceptable cost. Three documentation changes captured the investigation's substantive findings; no code, no atlas DB updates, no experimental output.

Net change:
- `TRADING_ATLAS.md` Exp 1 (id=1): two factorization fixes (the "7 indicators × 16 lookback combinations" line was structurally wrong) + a Cycle 38 RECON addendum documenting (a) the implementation is faithful to Chan's spec, (b) the 5.3/11/-5.6 bps chain is internally consistent but only reproducible by re-running deleted artifacts, (c) revival hypotheses re-prioritized post-RECON.
- atlas_sync: 1 TRADING entry updated (Exp 1), 1 embedding regenerated. All 14 other entries byte-identical md_hash pre/post.

---

## Why this matters

The cycle's central value isn't a corrected number -- it's clarity about what Cycle 39 should NOT spend compute on. The atlas's TC arithmetic (5.3 bps gross vs 11 bps TC at 2.7 trades/day) is mechanical and the break-even trade count (~1.325/day, half of current) is hard to reach without changing either the venue economics or the bar type. Reproducing the 5.3 figure (Option A in Code's report) would confirm what's already known; Option B (dollar-bar rebuild) is the only path that could plausibly flip the verdict, but Mechanism B's magnitude is genuinely unknowable from atlas-state alone, so the expected outcome is "modest improvement, possibly still TC-bound."

Given confirmed-positive Engine 7 (funding carry, Exp 13) and open Engine 8 (alternative data) threads, expected research value is higher elsewhere.

The cycle also surfaced a project-memory clarification: a `lookback ∈ {30, 60, 90, 120, 180, 240, 360, 720}` min set referenced in memory was being treated as a feature-lookback alternative to Chan's `{50, 100, 200, 400, 800, 1600, 3200}` set. It's actually the z-score TRADING-PARAM grid (a different knob). Future cycles proposing an "8-lookback rebuild" need to clarify which knob is being changed.

---

## Investigation findings (compressed from Code's report)

### Code archaeology (I1-I3)

- `engines/chan_cpo.py` absent at HEAD; deleted in commit `a2202a7` (2026-04-03). Git log shows only two commits ever touched it (`c48ccf9` Mar 23 + `a2202a7` Apr 3); no intermediate state.
- Logic split into three successor modules at HEAD: `engines/pairs_trading.py` (PairSpec, ParamConfig, run_pair_year), `engines/cpo_training.py` (RF train/predict), `engines/minute_features.py` (the Chan-paper-spec feature computation).
- The recovered `git show a2202a7^:engines/chan_cpo.py` contains only 17 daily-bar features -- not the 112 minute-feature spec the atlas's 2026-04-02 rebuild claims. The 112-feature implementation lives entirely in `engines/minute_features.py`, which post-dates the chan_cpo.py deletion. Cycle 37's "suspect-LOW" for the 2026-04-02 result upgrades to **"feature-spec verifiable; OOS run unreproducible without re-running"** -- the code is at HEAD; the artifact is not.
- Trade count (2.7/day) is a derived backtest statistic, not a configured cap. Mechanically, changing bar type changes trade count -- this is the load-bearing fact for Option B in the design memo.

### Chan paper spec (I4-I5)

- **Brief premise contradicted:** the two Chan PDFs (`Conditional_Portfolio_Optimization_..._Chan.pdf`, `CPOPaper__Chan.pdf`) are not in the repo. Not in root, not anywhere under user home. Workaround: `engines/minute_features.py:1-33` embeds the paper spec as docstring with explicit page citations (Chan 2021 Ch.7 p.142; paper p.5).
- Canonical spec: 8 indicators × 2 assets per pair × 7 lookbacks `{50, 100, 200, 400, 800, 1600, 3200}` min = 112 features. The 8 indicators (`bb_zscore, bb_bandwidth, mfi, force_index, donchian_width, atr, awesome_osc, adx`) match between paper-cited spec and current implementation. The implementer noted (line 16-17) that the paper says "8" but names only 7; `bb_bandwidth` was inferred as the natural 8th to make the math work.
- Spec comparison: 2026-04-02 implementation matches Chan canonical exactly. Project-memory 8-lookback set is the z-score trading-param grid (a different knob); confusion documented in atlas addendum.

### Info-bars feasibility (I6-I8)

- SP500 minute-bar data NOT on disk. `data/` has only crypto hourly cache. Polygon re-fetch would take ~15-60 min wall-clock for 38 pairs × ~76 unique tickers × ~98k bars each, depending on tier.
- Dollar-bar construction: standard Lopez de Prado recipe, ~30-50 lines pandas, one-pass per ticker. Implementation cost ~1 hour.
- Threshold tuning arithmetic: at target 20 bars/day on median SP500 pair (~$2B daily volume), threshold ≈ $100M per pair. Range across the universe (mega-cap to less-liquid SP500): 5-30 bars/day, acceptable.

### TC arithmetic (I9-I11)

- Atlas's 5.3 / 11 / -5.6 chain internally consistent: 5.3 - (2.7 × 4) = -5.5 ≈ -5.6 within rounding. 5.3 gross figure is load-bearing and only reproducible by re-running.
- Break-even table at 5.3 bps gross, 4 bps RT: trades/day must drop from 2.7 to 1.325 (a 50% reduction) to clear zero net.
- Two knobs flip the verdict mechanically: (1) halve trade count, (2) halve TC. Either alone reaches break-even at the existing gross alpha.
- Info-bar trade-count plausibility: Mechanism A (bar count drops 10-30× at $100M threshold) alone gets trade count well below break-even. Mechanism B (per-bar gross alpha shift from event-time sampling) is the uncertain piece -- could lift per-trade alpha by 20-50% in best case, or reduce it if mean-reversion opportunities cluster in quiet periods that dollar bars compress. Cycle 38 RECON cannot predict B's magnitude from atlas-state alone.

---

## Resolution: Option D-prime

The brief surfaced four options (A: reproduce 2026-04-02 result, B: dollar-bar rebuild, C: TC sensitivity sweep, D: accept NEGATIVE). The user chose **Option D-prime**: D's content (accept NEGATIVE) but with a small documentation-cleanup commit so the investigation's substantive findings (spec verified, factorization fixed, revival hypotheses re-prioritized) don't evaporate.

Rationale (captured during the pause-point exchange):
- Option A's compute (~3-5 h) confirms what's already known; doesn't open new optionality.
- Option B's outcome distribution centers on "modest improvement, possibly still TC-bound"; expected value low relative to alternative Engine 7/8 work.
- Option C is information-only; doesn't enable deployment without lower-TC broker access the user doesn't currently have.
- Option D-prime captures the cycle's clarifying value (factorization fix + revival-hypothesis re-ranking + project-memory clarification) at zero experimental compute.

---

## What was learned about project memory

The user's memory included a `lookback ∈ {30, 60, 90, 120, 180, 240, 360, 720}` min set associated with a "Chan/Burgess rebuild" framing. The cycle revealed this is the z-score TRADING-PARAM grid (per `engines/minute_features.py:25-28` annotation), not a feature-lookback alternative. The Chan paper's feature lookbacks are the geometric `{50, 100, 200, 400, 800, 1600, 3200}` min set, and `engines/minute_features.py` implements that faithfully.

The conflation in memory phrasing risked future cycles proposing "rebuild with 8 lookbacks" as a feature redesign, when the actually-implemented feature set is already the canonical Chan spec. Memory entry for this clarification is the user's to update directly (per cycle 38 close instructions); not in Code's scope.

---

## Cycle 39 candidates

Direction call is the user's; investigation surfaced three plausible Engine-pivot directions:

1. **Engine 7 scaling (Funding Carry).** Exp 13 is the atlas's lone CONFIRMED POSITIVE (Sharpe +4.65 primary, +10.78 validation). Scaling cycle could (a) extend to additional venues beyond Binance, (b) tighten the gate calibration at P>0.80 (live-deployment-ready), (c) instrument live-vs-paper PnL reconciliation to convert downstream live-deployment signals into an atlas-feedback loop. Suspect-LOW per Cycle 37 for the original Sharpe numbers; scaling cycle would re-verify operationally.
2. **Engine 8 thread (Alternative data on crypto).** Atlas's "Pending experiments" section lists `ALTERNATIVE × CRYPTO (on-chain)` with hypothesis: "Alpha from information edge rather than pattern edge." `onchain_btc` collector pipeline now in place via Cycles 30+31. A CPO experiment built on on-chain features (active addresses, exchange flows, whale movements, hash-rate) would be the first crypto experiment outside the TA-pattern paradigm.
3. **LSTM v2 with funding-carry features.** Per atlas/retro discussion, LSTM v2 (info bars + triple-barrier + DL refresh) was queued post-Cycle 36c. The cycle's most defensible motivation is now "test whether DL on the *one* confirmed-positive signal stream (funding carry) materially improves Sharpe/calibration" rather than re-attacking TA-on-crypto. Pre-requisite: Engine 7 scaling cycle establishes a clean test bed.

Ordering implication: #1 → #3 if DL becomes useful; #2 is independent and can run in parallel.

---

## Notes

### Methodological observation: read-only RECON cycles as a pattern

Cycle 36 RECON (which fed Cycles 36a/b/c) and Cycle 38 RECON share a useful shape: a multi-stream investigation with a single pause point between "findings report" and "implementation decision." The pattern works when:
- The investigation has clear streams that can run roughly parallel
- The decision space at the pause point is concrete (a handful of options, not open-ended exploration)
- "Option D: accept the existing analysis and pivot" is a credible outcome the cycle is allowed to converge on

Cycle 38 RECON's Option D-prime resolution is the cleanest possible RECON outcome: the cycle answered the question, the answer didn't motivate further work on this thread, and the answer itself is preserved as atlas documentation.

### Cycle that doesn't produce a result is not a failed cycle

The brief explicitly framed Option D as valid ("the atlas's existing analysis is sufficient, accept NEGATIVE and pivot to different research direction"). The cycle landed there. The atlas correction + retro documentation captures what was learned; future cycles can read the addendum without re-doing the investigation. The cost was ~2 hours of investigation budget; the benefit is permanent clarity about why Chan/Burgess on retail SP500 isn't a productive revival target.

### Brief premise on Chan PDFs

The brief assumed the two Chan PDFs were in repo root; they weren't anywhere on disk. The cycle workaround (recover spec from `engines/minute_features.py` docstring + citations) was sufficient for verification purposes, but if a future cycle wants Chan-paper-level detail on feature construction methodology (raw vs z-score vs aggregated), the PDFs would need to be re-acquired. Worth noting as a small data-engineering item if Engine 7+ work later motivates revisiting Chan's methodology beyond the indicator-list level.

---

## Open items / next cycle inputs

- **Cycle 39 direction call** (user): Engine 7 scaling, Engine 8 thread, or LSTM v2 with funding features.
- **Memory update** (user, directly): correct the 8-lookback-set conflation per the addendum's clarification. Out of Code's scope this cycle.
- **`atlas_search` engine-filter parameter** (deferred TODO from Cycle 35) -- still open.
- **PMA backfill** (separate cycle) -- still open.
- **Chan PDF re-acquisition** (low priority): only relevant if a future cycle wants paper-level methodology nuance beyond the indicator-list spec.
- **Cycle 36c's Sharpe-invariance methodological finding** remains the cycle's reusable methodological asset; applicable to any future cycle proposing a "leverage knob fixes the strategy" revival hypothesis.
