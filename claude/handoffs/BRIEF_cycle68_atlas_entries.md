# BRIEF - Cycle 68: Two Outstanding Atlas Entries

**Estimated Scope:** S (under 1 hr)
**Estimated Cost:** none (no computation; both verdicts are already adjudicated)
**Mode:** Code (documentation only)
**Retro:** `claude/retros/RETRO_cycle68_atlas_entries.md`

---

## What this is

Two finished verdicts never reached `TRADING_ATLAS.md`. Both are adjudicated and neither
needs re-running - this is transcription, not analysis.

**Chat does not have the atlas file and does not know its schema.** The content below is
supplied as findings; **read the existing entries and match their structure, field names,
ordering and tone.** If an entry below carries something the atlas has no field for, drop it
rather than inventing a field. If the atlas has a required field not covered here, say so in
the retro rather than guessing a value.

**Do not re-run anything.** If a number below disagrees with the source retro, the retro
wins and the discrepancy goes in the retro as a Chat error.

---

## T1 - Entry: MEAN_REVERSION x CRYPTO (Cycle 60)

Drafted in `claude/retros/CYCLE60_CLOSEOUT.md` (`8f71828`) and never transcribed.

**Verdict: gross tier structure confirmed, net edge not established.**

- Tiered cross-sectional reversal across a crypto universe. Information coefficient by tier
  was **-0.0137 / -0.0806 / -0.0998**, monotone in the predicted direction, so the gross
  effect is real and ordered.
- **Net edge fails.** All twelve per-cell deflated Sharpe cells fail. Criterion 1 fails at
  2x quoted spread. Criterion 4 was untestable for the reason below.
- **The out-of-sample holdout was never spent** and remains available.

**Record the known defect alongside the verdict.** The pre-registered beta-dispersion
control never ran: `residualize(mode='demean')` subtracts the same scalar from every symbol,
and `build_positions` ranks the signal, so ranking is invariant to a constant shift. The
control was inert and the nominal 96-config grid was 48 configurations duplicated. The
verdict stands on the remaining evidence, but the entry should not imply the control was
exercised. A regression test now pins this
(`test_demean_is_a_noop_for_rank_based_positions`).

## T2 - Entry: liquidation cascades / forced-trade scenario A2 (Cycle 66)

Source: `claude/retros/RETRO_cycle66_a2_labelled_validation.md` (`43c529b`).

**Verdict: refuted as a tradeable scenario. This entry closes the open "cannot be
determined" that Cycle 61 left in the atlas** - note the closure explicitly, since the point
of the entry is that an unresolved cell is now resolved.

**Lead with the oracle result, because it is what makes the negative decisive.** Reversion
was measured from the liquidation labels themselves, so detection error is zero by
construction. From the first instant a cascade is knowably over, reversion across the 210
BTC+ETH clusters at or above $1M is **-0.6 bps at 5 minutes and -2.7 bps at 15**
(day-clustered SE 3.0 and 4.6) against a **9 bps** cheapest-taker cost floor. Indistinguishable
from zero. **Detector quality is not the binding constraint** - a perfect detector buys
access to a move that does not pay.

Supporting, not load-bearing:

- The Cycle 61 detector finds **0 of 8** clusters at or above $10M, 0 of the two
  nine-figure manual cascades, and 0 of 174 clusters with 100 or more liquidations.
- **Why it fails, which is the transferable part:** it requires one-sided aggressor
  imbalance, and real cascades are *two-sided* on spot - imbalance inside the largest
  clusters runs **lower** than at random times (0.44 against a 0.49 null on BTC). Large
  forced flow attracts counterparty aggression in proportion to its size.
- Book depth at labelled events is **0.92x** its matched control, spread unchanged.
- Price runs about **27 bps in the forced direction before the first forced fill**, and the
  move is permanent (+25 bps BTC, +57 ETH an hour later against a control at zero). The
  move triggers the liquidations at least as much as the reverse.

**Classification: branch B shape** - a real effect whose tradeable part sits below cost.
This is the **fourth independent instance** of that pattern (Experiment 1, Cycle 60, the Chan
replication, A2). If the atlas has somewhere to record a cross-entry pattern, that is now
the most robust finding the programme has; if not, note it in the retro instead of forcing
it into an entry.

## T3 - Consistency pass

While both entries are open:

- Check no other atlas cell still reads "cannot be determined" for a question since
  resolved.
- Check the two new entries use the same verdict vocabulary as existing ones. If Cycle 60's
  "gross confirmed, net not established" has no existing counterpart, say so in the retro
  rather than inventing a category.

---

## Explicitly NOT in scope

- No re-running, re-measuring or re-adjudicating either cycle
- No new analysis, no new entries beyond these two
- No atlas schema changes

## Hand back

1. The two entries as written, quoted in the retro
2. Any field the atlas required that this brief did not supply, and what was used
3. Any discrepancy found between this brief and the source retros, with the retro winning
4. Result of the T3 consistency pass

Final step, standing:

```
.venv\Scripts\python split_zip.py zip --repo-delta
```

---

*Last updated: 2026-10-16 (Chat: praxis_main_current)*
*Changes: initial brief. Transcribes the Cycle 60 mean-reversion verdict and the Cycle 66 A2
refutation into TRADING_ATLAS.md, closing the open "cannot be determined" from Cycle 61.*
