# Retro: Cycle 37 -- a2202a7 sibling-fabrication sweep

**Brief:** `claude/handoffs/BRIEF_a2202a7_fabrication_sweep.md`
**Date:** 2026-05-15
**Mode:** Hybrid, single pause point (between investigation report and corrections)
**Status:** DONE
**Predecessor:** Cycle 36c (`fc9dff8` + `58add68`); memory #19 / `project_a2202a7_fabrication_sweep_queued.md`
**Commit:** `<CYCLE_37_HASH>`

---

## Summary

Verdict: **ISOLATED.** One additional confirmed fabrication beyond the two Exp 10 items already caught by Cycles 36a + 36c: the free-standing `## CRITICAL FINDING: PORTFOLIO LEVERAGE CAP NEEDED` block (atlas lines 892-918 pre-correction). Same fabrication family as the Exp 10 Addendum -- asserts a `Required fix in cpo_core.py` and projects "~-12% with 50% leverage cap" without surviving artifact. Cycle 36b superseded the fix mechanism; Cycle 36c measured the actual cap=0.5 result at -26.58% (~2x off the "-12%" projection).

**Pattern observation:** the fabrication was narrow -- one false revival hypothesis applied three times to one experiment via three syntactic surfaces (Exp 10 main block + Addendum + this free-standing CRITICAL FINDING block). **Not distributed** across a2202a7's broader atlas authorship. The other 10 a2202a7-authored experiment entries (Exps 1-correction, 8, 9, 11, 12, 13, 13-Addendum, 14, 15, 16, 17) each scanned clean on P1/P2/P3 patterns -- with four entries (Exps 11/12/13/14) having "suspect-deferred" P3 verifiability gaps caused by artifact deletion in the same commit, but no internal-inconsistency smoking guns.

Net change:
- `TRADING_ATLAS.md` -- one block replacement (~22 line delta) at lines 892-918 (pre-correction numbering). Block retitled to "CRITICAL FINDING [SUPERSEDED]"; -12% projection retraction added; pending-actions list updated to reflect Cycle 36b/36c completion status.
- atlas_sync: **0 added / 0 updated / 0 removed; 0 embeddings regenerated.** Confirmed pre/post md_hash byte-identical on all 15 parsed experiment entries -- the edited block sits outside any parsed-experiment boundary, identical scoping to Cycle 36a's Addendum retraction.

---

## Why this matters

The cycle confirms that a2202a7's fabrication pattern is bounded to the Exp 10 leverage-cap narrative. Without this audit, future readers might have continued to encounter the unmodified `CRITICAL FINDING` block claiming a non-existent fix is pending and projecting a non-existent measurement, even after Cycles 36a + 36b + 36c had collectively superseded all three elements. The block sat between Exp 12 and the v4 landscape matrix, outside any structured DB entry, so it was easy to miss in cycles scoped to specific experiments.

The methodological win is the **defensive md_hash sweep**: by snapshotting all 15 parsed entries' md_hashes pre-edit and re-running atlas_sync post-edit, the cycle proves the markdown edit stayed outside parsed-block boundaries to byte-equality on every entry. This is the canonical safety check for atlas-markdown edits that target inter-experiment scaffolding rather than experiment bodies.

---

## Execution log

### Step S1: Diff inventory

`git show a2202a7 --stat`: 72 files changed, +1919 / -79498 lines. The -79k is almost entirely deletion of `output/{burgess,crypto_ta,crypto_ta_val,futures_ta,futures_ta_val,fx_ta,mcb_ta,momentum,momentum_ethsol}/cpo/*` (60 files) plus `engines/chan_cpo.py` and `praxis.zip`.

Modified: `.env`, `.gitignore`, `TRADING_ATLAS.md`, `docs/praxis_main_series.md`, `engines/{cpo_core,momentum_strategy,ta_models}.py`, `scripts/{run_chan_cpo,run_cpo}.py`.

**Surprise from inventory: a2202a7 added ZERO files.** The commit message lists "Files added: engines/dex_scanner.py, engines/dex_quoter.py, engines/momentum_signals.py, scripts/run_dex_scanner.py, scripts/run_momentum.py" -- but `git log --diff-filter=A` confirms those 5 files were actually added in the **immediately preceding commit `0cddac0`** ("carry trade executor + paper trading pipeline"). The a2202a7 commit message is incorrect about its own contents. This is a commit-message authorship error rather than a fabrication of experimental content -- the engine files do exist and behave as documented; the commit-message attribution is just wrong. Worth recording as a secondary observation; not material for atlas correctness.

### Step S2: Atlas entries touched

a2202a7 inserted 679 lines into TRADING_ATLAS.md (vs 8 removed). It authored **11 new experiment blocks** (Exps 8-17 plus the Exp 10 Addendum) and updated existing Exp 1 with "Corrected results (2026-04-02 ... 112 minute features)". Atlas metadata total bumped from "2 complete, 6 pending" to "17 complete" in a single commit.

PMA was not touched (PMA was created in a later cycle).

### Step S3: Per-entry scan

Already-corrected entries (out of Cycle 37 scope):
- Exp 10 main block (Cycle 36c)
- Exp 10 Addendum (Cycle 36a retracted, Cycle 36c rewrote)

NEW CONFIRMED:
- **`## CRITICAL FINDING: PORTFOLIO LEVERAGE CAP NEEDED`** free-standing block (atlas lines 892-918). P1 hit: "Required fix in cpo_core.py: Add max_portfolio_weight parameter" describes a fix that was both never implemented as described AND has since been superseded by `--max-leverage` (Cycle 36b). P3 hit: "~-12% with 50% leverage cap" doesn't match any single leverage ratio applied to -83.78% (linear scaling gives -20% to -24% depending on which gross level is used as the baseline). Cycle 36c's actual cap=0.5 result is -26.58%, materially different from "-12%" by ~2x. The figure is an aspirational guess rather than derived arithmetic -- the additional liberty compared to the sibling Exp 10 Addendum's retracted -27.95%, which at least had the correct linear-scaling structure (just no actual run behind it). The Cycle 36a + 36c progression validates linear scaling as the correct relationship for binding caps; the -27.95% projection turns out to have been roughly correct in magnitude (off by 1.4 pp from Cycle 36c's actual -26.58% at cap=0.5); the -12% projection is uniquely off both in structure (no derivation) and in magnitude (~2x off).

CLEAN entries (no P1/P2/P3 hits):
- **Exp 8 (MOMENTUM TSMOM ETH+SOL)**: per-model AUC table, headline +1.91% Sharpe +0.545. No config-count assertion to verify; results internally consistent.
- **Exp 9 (MOMENTUM Triple Barrier)**: "1728 configs" P2-VERIFIED against `engines/momentum_strategy.py:177-188`: TSMOM_4H = 4 × 3 × 72 = 864 + TSMOM_DAILY = 4 × 3 × 72 = 864 = **1,728**. Matches exactly.
- **Exp 11 (TA × FUTURES TB)**: "7920 configs" consistent with universal_ta path (Exp 4 sibling uses identical phrasing pre-a2202a7). 47-day OOS caveat documented in-line.
- **Exp 12 (TA × FX G10 TB)**: "7920 configs" same provenance. 52-day OOS caveat documented.
- **Exp 13 (FUNDING Carry)**: Sharpe +4.65 primary / +10.78 validation are extraordinary but downstream live deployment exists (Cycle 30+ scheduled collectors, `funding_monitor.py`, dashboard). Live trading would surface fabrication; suspect-LOW. Per-asset table (BTC +52%, ETH +111%, AVAX +63%) is internally consistent.
- **Exp 13 Addendum (Regime Feature Comparison 2026-04-02)**: AUC delta table A:0.9862 / B:0.9791 / C:0.9813 across BTC/ETH/SOL/BNB, pattern A > C > B holds in all 4 rows. Difficult to fabricate this internal consistency without a real run.
- **Exp 14 (GRID BOT)**: "Configs: 36 (4 spacings × 3 range widths × 3 hold durations)" P2-VERIFIED against `engines/grid_bot_strategy.py:138-165`. Headline Sharpe -10.79 / cum -112.84% catastrophic failure; harder to motivate fabricating failure than success.
- **Exp 15 (VRP × BTC/ETH)**: explicitly INCONCLUSIVE / BLOCKED due to Deribit geo-block + tenor collapse bug. No specific result claim to verify.
- **Exp 16 (Cross-DEX ARB)**: mechanical observations from `engines/dex_scanner.py` + `engines/dex_quoter.py` -- pool counts, price impact -- easily reproducible. NO EDGE verdict matches depth result.
- **Exp 17 (1-min Momentum)**: "88 trades, 31% WR, -827.5 bps" backtest result; "3x leverage = -2,482 bps" is exact arithmetic (827.5 × 3 = 2482.5). Engine files exist (`engines/momentum_signals.py` from `0cddac0`). Suspect-LOW; cheap to re-verify if ever operationally relevant.

SUSPECT-DEFERRED (no smoking guns, artifact-deletion blocks verification):
- Exp 11, Exp 12, Exp 13, Exp 14 P3 headline-number verifiability gaps. All `output/{futures_ta,fx_ta,funding_rate,grid_bot}/` deleted in a2202a7 itself. No internal inconsistencies found; verification requires re-running each (~3-5h compute per entry; out of cycle scope).

### Step S4: Non-atlas docs

`docs/praxis_main_series.md` was substantively rewritten by a2202a7 (file structure changed completely). Per brief, report-only -- no corrections proposed. No specific P1/P2/P3 hits surface from a quick scan; the doc is narrative chat-series history rather than experiment-result claims.

### Step S5: New files

a2202a7 added zero new files. The commit message's "Files added" list is incorrect (see S1 surprise note). Engine/script files credited to a2202a7 were actually added in `0cddac0`. No fabrication of experimental content; just sloppy commit-message authorship.

### Step apply: Correction 1

Single Edit-tool replacement at `TRADING_ATLAS.md` lines 892-918. Heading retitled from `## CRITICAL FINDING: PORTFOLIO LEVERAGE CAP NEEDED` to `## CRITICAL FINDING [SUPERSEDED]: Portfolio leverage cap reasoning`. Block body replaced with cycle-trail context (36a + 36b + 36c history), the user-refined wording on the -27.95% vs -12% distinction (the Addendum had cleanly-derived projection; the CRITICAL FINDING block had a free aspirational guess that doesn't match any leverage ratio applied to -83.78%), pending-actions list updated to reflect completion status. The trailing v4 landscape matrix block remains intact; its "Pending actions" sublist was folded into the SUPERSEDED block to keep the documentation chronology in one place.

### Step verify: defensive md_hash sweep

Pre-correction snapshot of all 15 atlas_experiments.md_hash captured via `outputs/snapshot_md_hashes.py pre`. atlas_sync ran: 0 added / 0 updated / 15 unchanged / 0 embeddings regenerated -- matches predicted outcome exactly. Post-snapshot via `outputs/snapshot_md_hashes.py post` captured; comparison shows all 15 hashes byte-identical pre/post. The edited block lives outside any parsed-experiment boundary; the atlas_sync parser confirmed it correctly via the unchanged structured-entry count.

Spot-check via `outputs/verify_cycle37.py` on id=8 (Exp 10, the experiment whose narrative the block originally distorted), id=10 (Exp 12, the entry immediately preceding the edited block), and id=11 (Exp 13, the entry immediately following the edited block). All three: result_class stable, md_hash unchanged, no leaked phrases from the SUPERSEDED block in their parsed full_markdown.

---

## Notes

### Pattern characterization

The fabrication pattern from a2202a7 in Exp 10 was **narrow**: a single false revival hypothesis ("leverage cap revives the strategy") applied to one experiment, expressed across three syntactic surfaces (the main Exp 10 block's "construction failure" verdict, the Exp 10 Addendum's projected -27.95% result, and the free-standing `CRITICAL FINDING` block's "Required fix" + "~-12%" projection). All three surfaces have now been corrected across Cycles 36a (Addendum), 36c (main block), and 37 (CRITICAL FINDING).

The pattern is **not distributed** across a2202a7's broader atlas authorship. Future cycles should treat a2202a7-era atlas content as trustworthy at the level Cycle 35's mass backfill established, with these three specific surfaces now permanently documented as exceptions.

The reason the pattern was narrow is structurally interesting: the false revival hypothesis arose from a real diagnostic ("40 models × 5% cap = 175% gross, that's clearly too much") that motivated a real fix-shaped suggestion ("add a portfolio gross cap"). The leap from "fix-shaped suggestion" to "fix is implemented + measured result" happened in three places that all flowed from the same wrong belief about what the fix would accomplish (i.e. that it would expose a hidden positive Sharpe). Cycle 36c's Sharpe-invariance finding refutes that belief at its root, which is why all three surfaces fall together.

### Secondary observation: a2202a7's commit-message authorship

The a2202a7 commit message body claims "Files added: engines/dex_scanner.py, engines/dex_quoter.py, engines/momentum_signals.py, scripts/run_dex_scanner.py, scripts/run_momentum.py". Those 5 files do exist in the repo and behave as the commit message describes -- but they were added in the **immediately preceding commit `0cddac0`** ("carry trade executor + paper trading pipeline"), not in a2202a7 itself. `git log --diff-filter=A` confirms.

This is a commit-message authorship error, not a fabrication of experimental content. The engines are real; the attribution is just wrong. Worth recording as a small data point on a2202a7's authoring care -- the same commit that conflated three syntactic surfaces of one false hypothesis also misattributed five file additions. The level of carelessness suggests a fast-moving authoring session rather than deliberate misrepresentation, but the practical effect on downstream cycles (Cycle 36a/36c needing to disentangle the leverage-cap narrative) is the same regardless of intent.

### Memory #19 status

Updated to: "swept; isolated to Exp 10 leverage-cap narrative across three syntactic surfaces (main block, Addendum, CRITICAL FINDING). Pattern is not spread elsewhere in a2202a7. Memory remains as historical reference but no further sweep is needed."

The original memory entry (`project_a2202a7_fabrication_sweep_queued.md`) is preserved as authored, with a "swept Cycle 37" status update line.

### Suspect-deferred entries: when to revisit

Exp 11, 12, 13, 14 P3-style verifiability gaps remain open as Cycle 38+ candidates if their results ever become operationally relevant:

- **Exp 13 (Funding Carry)**: highest priority if revisited, because it's the atlas's lone CONFIRMED POSITIVE result and downstream live deployment depends on the Sharpe +4-11 claim. The downstream live monitoring partially substitutes for the missing P3 artifact -- if the live signal generation has been generating the dashboard for 30+ cycles and the per-asset Sharpes documented in atlas were materially wrong, that would have surfaced operationally. Suspect-LOW.
- **Exp 14 (Grid Bot)**: catastrophic failure; the result claim is easier to motivate than for a positive result, but the asymmetric-payoff diagnosis ("requires ~70% WR, achieved 48%") is a falsifiable mechanical statement. Suspect-deferred.
- **Exp 11 / Exp 12 (Futures / FX TB)**: small-OOS-window caveat already documented in-line; the headline numbers (+2.93% Sharpe +1.528 for Exp 11; -0.39% Sharpe -0.471 for Exp 12) are within plausible ranges for 47-52 day windows. Suspect-deferred.

A future Cycle 38+ would re-run any of these through the universal_ta path (Exp 11/12) or the existing strategy adapters (Exp 13/14) following the Cycle 36c pattern. None are blocking the atlas's current ISOLATED verdict for a2202a7.

### Lessons for future cycles

Patterns worth preserving for Cycle 38+:
- **Defensive md_hash sweep before/after inter-experiment-scaffolding edits.** When editing a block that sits between parsed-experiment boundaries, snapshot all entries' md_hashes pre-edit, run atlas_sync post-edit, and confirm byte-identical hashes. This proved sufficient to verify Correction 1 didn't accidentally cross a parsed boundary, with no other safeguard needed.
- **Distinguishing "derived projection" from "free aspirational guess".** The user's wording refinement on Correction 1 captured a useful distinction: the Exp 10 Addendum's -27.95% projection at least had cleanly-derived structure (linear scaling that Cycle 36c later confirmed was correct), while the CRITICAL FINDING block's -12% didn't even match any leverage ratio applied to -83.78%. Both are fabrications, but the second is a more brazen variety. Future audits should preserve this distinction in retraction text -- it's diagnostic about authoring style and helps future readers calibrate confidence in other a2202a7-era claims.
- **Single-pause-point cycles for low-risk audits.** Cycle 37 used one pause point (between investigation report and corrections) rather than the two-pause-point structure of Cycles 36b/36c (which had separate pauses for results review and atlas-update approval). For markdown-only correction cycles with no compute, one pause point is sufficient.

---

## Open items / next cycle inputs

- **Cycle 38+ candidates if operationally relevant:** verify Exp 11/12/13/14 headline results via re-run (only if a downstream consumer surfaces a discrepancy; not blocking).
- **`atlas_search` engine-filter parameter** (deferred TODO from Cycle 35) -- still open.
- **PMA backfill** (separate cycle) -- still open.
- **Memory #19 file update**: `project_a2202a7_fabrication_sweep_queued.md` adds a "Status: SWEPT (Cycle 37) -- ISOLATED verdict" line at top. Index entry in MEMORY.md updated accordingly.
- **Info bars as a new experiment (Cycle 38+ candidate)**: gating depends on whether the TA-on-crypto thread closure motivates a pivot to fundamentally different signals. Reframed in Cycle 36c retro; remains queued.
- **Cycle 36b's audit of `--feature-mode` plumbing**: out of scope for 37; possible Cycle 38+ item if the feature-mode caching introduced in a2202a7's `engines/cpo_core.py` modification ever surfaces a regression.
