# Retro: Cycle 62B -- D1 Leveraged-Token Rebalance Mechanism (T4 landed; T1 gate: TRIGGERED, stopped)

**Brief:** `claude/handoffs/BRIEF_cycle62b_d1_mechanism_prereg.md` (v3, `e1bc2e4`)
**Date:** 2026-10-16
**Session:** one, about 13:20-15:15 EDT
**Status:** COMPLETE TO THE T1 GATE, and stopped there as designed. T4 is landed and tested. T1
establishes that Binance's leveraged tokens rebalance on a **drift trigger, not a schedule**, which
is the brief's stop condition. T2 (pre-registration) and T3 (the study) were not started, as the gate
and Jeff's ordering ("T4 first, then T1, then stop for Jeff's call") both require.

---

## Summary

**T4.** The Cycle 60 leveraged-token filter is now structural, the class is re-measured, and the
Cycle 60 verdict is unchanged. A name counts as a leveraged token only if its implied underlying
actually trades.
- `JUPUSDT` and `SYRUPUSDT` survive, as does any other spot asset whose name merely ends in
  UP/DOWN/BULL/BEAR.
- The brief described the defect as `in` vs `endswith`. That is not the defect: both names *end*
  with `UPUSDT`, so an `endswith` fix would have changed nothing.

**T1.** All 50 genuine tokens split into two product families with different mechanisms:
- **Binance's own UP/DOWN tokens (BLVT).** 40 tokens, 97.7% of the token-months. They rebalance when
  leverage drifts past a hidden, moving target, at any hour of the day, every two to five days.
- **FTX's BULL/BEAR tokens, listed by Binance.** 10 tokens, 2.3% of the token-months, 2020-Q1 only.
  These did have a daily reset. They serve here as the positive control.

Two independent routes agree, and the data route's power to detect a schedule is shown on that
control:
- **Documentation:** Binance's own text, most of it now deleted.
- **Data:** a slope-break detector on 1m market prices.

A drift-triggered rebalance inherits A2's endogeneity, so **the gate stops the cycle**. Separately,
T1 found three problems that any redesign must solve regardless:
- there is no NAV anywhere;
- there is no order book before 2023;
- the forced flow hits the perp, not spot.

**Confidence that BLVT is triggered and not scheduled: HIGH.**

---

## Hand-back

| # | Item | Status |
|---|---|---|
| 1 | T1 mechanism answer, route, confidence; granularity; NAV | **Delivered** -- next section |
| 2 | Pre-registration commit hash | Not created: the gate stopped the cycle before T2 |
| 3-7 | Pressure, profile, cost grid, event count used, branch | Not reached (gated). No pressure was measured, by design |
| 8 | T4 confirmed by test, class re-measured | **Delivered** -- `8962646` |

Stopping at T1 is the outcome the brief built the gate for. It is a complete result, not an
incomplete cycle.

---

## T1 -- the mechanism answer

### In one line

**Binance's leveraged tokens (BLVT) rebalance when their leverage drifts past a hidden, moving
target -- at any hour, every few days, always in the direction of the move that caused the drift.
There is no schedule. Confidence: HIGH.**

### The six sub-questions

| # | Question | Answer | Established by | Confidence |
|---|---|---|---|---|
| 1 | Scheduled, triggered, or both? | **BLVT (40 tokens): triggered.** FTX BULL/BEAR (10 tokens, 2020-Q1): both | Docs + data, positive control | **HIGH** |
| 2 | Trigger condition, target band | A hidden, moving target leverage, rebalanced "as needed". The documented band is 1.25-4x (1.5-3x at launch). In practice leverage sits much tighter: median 2.0-2.5x, q10-q90 about 1.5-3.6x before a rebalance | Docs (band, hidden target) + data (operating range) | Class HIGH; numbers MEDIUM; exact rule **unrecoverable** (hidden by design, no NAV) |
| 3 | Published rebalance time? | **None.** Binance says BLVTs are "not forced to rebalance periodically". The data show no fixed-hour component of any size | Docs + data | **HIGH** |
| 4 | Detectable in price/NAV? | **Yes, from market price alone**, as a break in the slope of token price on perp price. Placed to within about 30 min; large rebalances only | Data (synthetic power/null tests + real positive control) | HIGH for large; small ones invisible |
| 5 | Granularity | **1m, 1h, 1d klines and aggTrades exist for all 50 tokens over their whole lives**, and 1m spot and perp exist for every underlying. **Order book: none before 2023-01**, which covers only 12.7% of BLVT token-months | Archive listing | HIGH |
| 6 | NAV available? | **No.** It is not in the archive, and the historical NAV endpoint is gone (404). Market price only | Archive structure + API probes | HIGH |

### Route 1 -- documentation (Binance's own words, mostly deleted)

Binance's BLVT FAQ pages are gone. Fetched on 2026-10-16:
- `support/faq/360042957472` ("What Are Binance Leveraged Tokens") and
  `support/faq/what-is-binance-leveraged-tokens-rebalancing-360043387931` both return *"This article
  does not exist"*.
- The AMP copies return 404.
- WebFetch cannot reach web.archive.org. I did not work around that block.

What survives:

**1. Binance blog, "Your Essential Guide To Binance Leveraged Tokens", 2020-05-31 (live)**
(`binance.com/en/blog/futures/your-essential-guide-to-binance-leveraged-tokens-421499824684900580`).
Quotes as extracted by the fetch tool:
- "BLVTs will only rebalance the positions as needed to maximize profitability on upswings"
- "BLVTs are not forced to rebalance unless losses are extreme"
- "'normal' fluctuations in the market will not cause rebalances"
- "BLVTs maintain a variable target leverage range between 1.5x and 3x" (the launch range)
- Traditional leveraged tokens, by contrast, "are programmed to rebalance at pre-configured times (daily)"
- BLVTs "track the BTC perpetual contracts on Binance Futures"

**2. Binance Academy, "Beginner's Guide to Binance Leveraged Tokens (BLVT)"** (Binance Square post
288729, verified Binance Academy account, live):
- "BLVTs only rebalance in times of extremely high volatility and are not forced to rebalance
  periodically otherwise."
- "In the case of BTCUP and BTCDOWN, this range is between x1.25 and x4."
- Target leverage "is not constant, nor is it publicly visible."

**3. The deleted FAQ, as quoted in search-engine snippets** (not fetchable; treat as secondary):
- "Binance Leveraged Tokens rebalance on an as-needed basis only."
- "...attempt to maintain a variable target leverage range between 1.25x and 4x. However, do note
  that the real leverage may fluctuate beyond the target range in extreme market movements."
- "The true target leverage is constantly changing and hidden from outside observation. At all
  times, the token info page will always show the 'realized' target leverage post rebalancing event."
- "If the real leverage ratio decreases to 1.25x and below, the algorithm will decide on a new target
  leverage ratio and trigger the rebalancing."
- A snippet, paraphrased by the search tool rather than quoted, says the hiding is meant to make
  rebalancing harder to front-run.

**4. FTX BULL/BEAR (secondary sources only; FTX is defunct).** The tokens rebalanced daily at
00:02 UTC; one source says 02:00. They also rebalanced intraday when the underlying moved 11.5%
(BULL) or 6.7% (BEAR). Their basket was FTX perps, so their forced flow traded on FTX, not on
Binance.

**Route 1 verdict:** BLVT is triggered ("as needed", "not periodic", target hidden). The
documentation also claims rebalancing happens "only in extremely high volatility". The data
contradict that claim: rebalances come every few days.

### Route 2 -- the data

**The identity it rests on.** A leveraged token holds a fixed quantity `q` of the perp per token
between rebalances. So `NAV = a + q*F` is **linear in price levels**, and real leverage
`L = q*F/NAV` drifts mechanically with price. Only a rebalance changes `q`. A rebalance is therefore
a break in the slope of token price on perp price, and that break is visible in market prices
without NAV.

**The detector** is `engines/forced_trade/blvt_mechanism.py`:
1. Fit level regressions over 12 h windows either side of every hour boundary.
2. Place each candidate to the minute with a two-segment fit across the full window.
3. Give every jump a Newey-West SE (120-minute lags), since the premium is persistent.
4. Count an event as significant only if `|dlog q| >= 0.10` and `|z| >= 4`.

Prices are per-minute VWAP, so the 31.5% of BTCUP minutes with no trade become gaps rather than
stale closes. The debugging trail below records how each of these choices was forced.

**The timing test that decides scheduled vs triggered.** A scheduled reset at hour H must undo the
drift since the previous reset, so its size is predictable:
- long token: `d(H) ~ (L-1)*r_24h`
- short token: `d(H) ~ -(|L|+1)*r_24h`

For each UTC hour, regress the 3 h slope change on the prior 24 h perp move. A schedule shows up as
one hour with a large, correctly signed beta and near-zero elsewhere. A trigger shows up as small
betas spread over the whole day. Synthetic tests (9 tests, 67 in total across the affected suites):
- at FTX scale it recovers a planted daily reset at the right hour, with betas 1.9-2.1 against a
  theoretical 2, and -4.0 to -4.2 against -4;
- it stays silent (all |z| < 3) on triggered and null fixtures.

**Positive control (real data).** The four FTX tokens show exactly the scheduled signature:

| token | top hour | z | beta (theory) | homogeneity Q, df 23 | share of a daily reset at 00h / 01h | median L (design 3x) |
|---|---|---|---|---|---|---|
| BULLUSDT | 01 | +17.5 | +1.93 (+2) | 418, p=4e-74 | 0.72 / 0.85 | 3.05 |
| BEARUSDT | 01 | -6.9 | -1.98 (-4) | 178, p=7e-26 | 0.50 / 0.45 | 2.98 |
| ETHBULLUSDT | 01 | +13.7 | +1.49 (+2) | 234, p=7e-37 | 0.53 / 0.69 | 3.04 |
| ETHBEARUSDT | 00 | -8.3 | -2.48 (-4) | 161, p=1e-22 | 0.60 / 0.51 | 2.97 |

The method finds FTX's daily reset at 00:00-01:00 UTC, which fits the documented 00:02. Their reset
lands exactly back on 3x: post-rebalance median leverage is 2.94-3.04. The method also recovers the
products' designed leverage to within 2% from market prices alone.

**BLVT results** (8 tokens, the four longest-lived underlyings, 10,273 token-days, 37% of BLVT
token-months; `outputs/forced_trade/c62b/t1_mechanism_table.csv`):

| token | rebalances (sig.) | per yr | median gap, d | agree w/ drift | median L | Q (p) | right-signed hrs | share at 00h | share at 01h |
|---|---|---|---|---|---|---|---|---|---|
| BTCUP | 137 | 36 | 5.1 | 0.64 | 2.03 | 28.5 (0.20) | 21 | +0.04 +- 0.07 | +0.14 +- 0.07 |
| BTCDOWN | 283 | 75 | 2.7 | 0.76 | 2.32 | 20.8 (0.59) | 24 | +0.10 +- 0.06 | +0.02 +- 0.04 |
| ETHUP | 170 | 47 | 3.8 | 0.66 | 2.11 | 36.0 (0.04) | 22 | -0.02 +- 0.04 | +0.05 +- 0.04 |
| ETHDOWN | 326 | 90 | 2.8 | 0.84 | 2.45 | 26.1 (0.30) | 24 | -0.01 +- 0.02 | -0.00 +- 0.03 |
| BNBUP | 152 | 43 | 3.0 | 0.61 | 2.15 | 29.9 (0.15) | 23 | -0.05 +- 0.04 | +0.03 +- 0.04 |
| BNBDOWN | 286 | 80 | 2.0 | 0.79 | 2.29 | 68.1 (2e-6) | 23 | -0.07 +- 0.03 | -0.06 +- 0.03 |
| ADAUP | 173 | 56 | 4.1 | 0.65 | 2.19 | 23.4 (0.44) | 24 | +0.05 +- 0.04 | +0.07 +- 0.05 |
| ADADOWN | 372 | 121 | 1.9 | 0.83 | 2.26 | 30.8 (0.13) | 24 | -0.04 +- 0.02 | -0.01 +- 0.03 |

### Five independent signatures that BLVT is triggered

1. **No fixed hour.**
   - Every BLVT token responds to drift at nearly every hour of the day: 21-24 of 24 hours carry the
     right sign.
   - Where it is precisely estimated, the share of a daily reset at any fixed hour is at most about
     0.14, and mostly indistinguishable from zero. FTX's is 0.45-0.85.
   - Homogeneity across hours is not rejected at 5% for 6 of 8 tokens. ETHUP is marginal (p=0.04).
   - The one clear rejection, BNBDOWN, is a broad intraday activity pattern rather than a spike: betas
     are strong at 02-11h, 13-15h and 20-22h, weak elsewhere, and peak at hour 06, not 00.
   - These results hold with 6 h profile windows. BTCDOWN gives Q=6.0 (p=1.0) and BTCUP Q=16.9
     (p=0.81), while the controls stay sharply concentrated (Q=486 and 138). See
     `t1_sensitivity_profile6h.txt`.
2. **DOWN tokens rebalance about twice as often as their UP twins:** 75 vs 36, 90 vs 47, 80 vs 43
   and 121 vs 56 per year. A daily schedule gives both twins the same frequency. A drift trigger
   must favour DOWN, because short-token leverage drifts about 3x faster, as `(|L|+1)` against
   `(L-1)`.
3. **Rebalances undo the drift that preceded them.** Direction agrees with the move since the last
   rebalance 61-84% of the time, against 50% for noise.
4. **The target moves.** BLVT post-rebalance leverage wanders (q10-q90 about 1.5-3.0) where FTX's
   lands on 3x. That is the "constantly changing and hidden" target the FAQ describes.
5. **Irregular cadence.** Median gaps are 1.9-5.1 days, not one day.

### The residue -- a small 00:00 UTC excess, recorded rather than explained away

Pooled over all 1,899 significant BLVT rebalances (`t1_timing_pooled_stdout.txt`):
- The 00:00 UTC hour holds 127 against 79 expected under uniform timing (1.6x, Poisson p=5e-7).
- 41 of those 127 fall at minutes 00-09.
- That is **48 excess events, 2.5% of all rebalances.**

It is not the funding clock: 08:00 shows 1.04x. 16:00 is mildly elevated at 1.21x. The rest of the
timing structure follows market activity (busier around the US open), which is what a trigger
predicts. The regression puts the share of drift corrected at 00h at about zero for most tokens.

I have not established whether this residue is a daily operational check that rebalances *if* out
of tolerance, or day-boundary processing (fees, subscription/redemption limits). Either way it is
conditional on drift and small. It does not make the mechanism scheduled.

### Why this stops the cycle

Every BLVT rebalance trades **with** the move that triggered it:
- UP delevers by selling into a fall and relevers by buying into a rise;
- DOWN delevers by buying into a rise and relevers by selling into a fall.

The flow fires when the move pushes leverage past a hidden threshold. Timing and size are not
knowable in advance, because the target is hidden by design. This is the A2 structure Cycle 66
showed to be unidentifiable: the move triggers the forcing, so any pressure measured around these
events mixes cause with effect. The brief's whole justification for studying a dead instrument was
a *scheduled* rebalance. **That justification is gone, and T3 was not run.**

### Other T1 findings any redesign has to face

- **No NAV.** Neither the archive nor the API has it:
  - `data.binance.vision` holds only `spot/{aggTrades,klines,trades}`, futures
    `{aggTrades,bookTicker,fundingRate,*PriceKlines,klines,trades}` and daily
    `{bookDepth,metrics}`. Nothing BLVT- or NAV-shaped.
  - `fapi/v1/lvtKlines` returns 404 while `fapi/v1/time` and `fapi/v1/klines` return 200, so the
    endpoint is removed, not geo-blocked.
  - `sapi/v1/blvt/tokenInfo` returns 404.
  - Detection therefore falls back to the market-price slope proxy. The cost: hour-scale resolution
    in calm markets; 830 of 2,729 BLVT candidates (30%) fail significance; small rebalances are
    invisible; and the exact trigger rule cannot be recovered.
- **No order book before 2023.** Perp `bookDepth` (daily) starts 2023-01-01. Perp `bookTicker`
  covers 2023-05 to 2024-04. Spot has no book data at all. Only 118 of 928 BLVT token-months
  (12.7%) have any book data: BTC, ETH and BNB 14 each, ADA 8, XRP 5, LINK 4. Cycle 66's primary
  instrument, hit-side depth, does not exist for 87% of this sample.
- **The forced flow hits the perp, not spot.** BLVT baskets are USDT-M perpetuals ("track the BTC
  perpetual contracts on Binance Futures"). The brief's T3 assumed spot underlyings via the Cycle 60
  loader. Any redesign must measure on perps; the archive has perp klines, aggTrades and trades
  throughout.
- **There are not ~28,500 events.** That figure was Cycle 61's token-day count multiplied out on
  the `SCHEDULED` assumption. The data show about 68 detectable rebalances per token-year (1,899 in
  10,273 token-days). That suggests roughly 5,000 across the whole BLVT class (rough, and a lower
  bound on true rebalances), and none of them scheduled.
- **The FTX family is the only scheduled one, and it cannot carry a study.**
  - 10 tokens, about 74 days each, 2020-Q1 only (about 660 token-days, including the COVID crash).
  - Their flow traded on FTX's own perps, which are not in this archive. FTX is defunct.
  - They also had intraday triggers.
  - Docs: `docs/FORCED_TRADE_TAXONOMY.md` D1 row corrected to `TRIGGERED` for BLVT. The P1 basis is
    annotated as void; the triage itself is left to Chat.

### Scope guard -- what T1 did not do

T1 never measured underlying price displacement around a rebalance. It used no event-time price
profile, no depth, no imbalance and no reversion. The detector's only use of the underlying is as
the regressor that reveals `q`, plus the move since the previous rebalance, which *is* the trigger
state. The T2 pre-registration discipline is intact: no measurement code exists.

---

## T4 -- leveraged-pattern substring bug (`8962646`)

- **The fix.** `filter_symbol_names` now treats a name as a leveraged token only if it is
  `<BASE><UP|DOWN|BULL|BEAR><QUOTE>` **and** `<BASE><QUOTE>` is in the traded set. The logic is
  ported from `forced_trade/leveraged.py:split_leveraged`.
  - A bare BULL/BEAR maps to BTC, as in Cycle 61.
  - A bare UP/DOWN maps to nothing: no BLVT was ever bare, so such a name would be a spot asset.
  - `UniverseSpec.leveraged_patterns` became `leveraged_suffixes`.
  - `DEFAULT_LEVERAGED_PATTERNS` is kept as a commented LEGACY constant, so the frozen Cycle 61
    `run_audit t3` still reproduces (verified: 64 parse, 60 genuine, 2 false positives).
- **Correction to the brief.** The defect is not `in` vs `endswith`. `JUPUSDT` and `SYRUPUSDT` both
  end with `UPUSDT`, so a suffix rule deletes them too. A test pins this, so a future "fix" to
  `endswith` would fail it.
- **Tests:**
  - JUP and SYRUP survive.
  - Invented lookalikes survive (`MEGADOWNUSDT`, `TEDDYBEARUSDT`, `REDBULLUSDT`, bare `UPUSDT`).
  - The underlying is looked up in the traded set, so subset filtering works.
  - Every new assertion fails on the pre-fix rule, which was checked against the HEAD version.
  - The existing `test_name_filters_exclude_leveraged_and_stables` gained `ETHUSDT` in its input.
    A leveraged token is only recognised when its underlying trades, which it always did in the
    real archive. This is a fixture change, not a weakened assertion.
- **Re-measured on a fresh S3 listing** (3,738 symbols, all quotes; +48 since Cycle 61, -0; 755
  USDT):
  - The old rule catches **52** USDT names: **50** genuine leveraged tokens and **2** spot assets
    (JUPUSDT, SYRUPUSDT).
  - The spelling class across all quotes is **64**, with **4** lookalikes (JUP and SYRUP on USDT
    and USDC).
  - There is no drift from Cycle 61's genuine list.
  - So the class is a live trap, but today it contains exactly those names.
- **Cycle 60 impact.** JUP has 30 in-window months and SYRUP 14, at most 0.19% of panel symbol-bars
  and at most 0.46% of the end-of-window cross-section. Cycle 60 failed DSR decisively (best 0.337
  against 0.95). **The verdict does not change.**

---

## Changes Made

### Files

| File | Change | Lines |
|---|---|---|
| `engines/xsec_reversal/universe.py` | T4: structural leveraged-token rule (`leveraged_underlying`, `is_leveraged_token`, `filter_symbol_names(..., traded=)`), suffix constants, LEGACY pattern constant | 22-62, 131-200 |
| `engines/xsec_reversal/tests/test_xsec_reversal.py` | T4: fixture fix + 3 new tests | 21-24, 147-195 |
| `engines/forced_trade/leveraged_class.py` | T4: class re-measure CLI (new) | 1-159 |
| `engines/forced_trade/blvt_mechanism.py` | T1: rebalance detector, schedule regression, scheduled share, `run`/`batch`/`table` CLI (new) | 1-573 |
| `tests/test_blvt_mechanism.py` | T1: 9 synthetic power/null/schedule tests (new) | 1-146 |
| `engines/forced_trade/__init__.py` | Module index: Cycle 62B entries | 34-41, 50-52 |
| `docs/FORCED_TRADE_TAXONOMY.md` | D1 row corrected to TRIGGERED (BLVT); P1 basis annotated void; footer | 119-125, 179, end |
| `outputs/forced_trade/c62b/` | T4 json/stdout/raw listing; T1 per-token events, hour profiles, summaries, table, batch stdout, granularity/book/timing/sensitivity evidence, probe scripts as `.py.txt` | new |

Not committed: the per-token `window_slopes` series (about 1.7 MB each). They are regenerable with
`python -m engines.forced_trade.blvt_mechanism batch --keep-slopes`. The archive cache grew by about
686 MB of 1m klines (533 files) under the gitignored `data/external/binance_archive`.

### Key decisions

- **Ported** `split_leveraged` into `universe.py` rather than importing it. "Port" was the brief's
  word, and importing would make the lower-level package depend on `forced_trade`. The Cycle 61
  module stays frozen, as Cycle 66 established.
- **Detector design is driven by the mechanism, not tuned.** Every knob is in `DetectParams`.
  Sensitivity to the profile window is reported. Thresholds were fixed before the real runs, and the
  verdict does not depend on them: the threshold-free schedule regression decides it.
- **FTX tokens as the positive control.** Without them, "no schedule found" could have meant "test
  has no power". With them it cannot.
- **Stopped at T1** even though most of the T3 machinery could have been bolted on in an hour. That
  is what the brief and Jeff's instruction require.

---

## Test Results

### Passed
- `pytest tests/test_blvt_mechanism.py engines/xsec_reversal/tests/ tests/test_forced_trade_labels.py`
  gives **67 passed** (9 detector + 41 xsec + 17 Cycle 66 labels).
- The new T4 assertions fail on the pre-fix `universe.py` (HEAD), as checked against an importlib
  copy.
- Cycle 61 `run_audit t3-leveraged` still runs against the LEGACY constant on today's listing (into
  scratch, so Cycle 61's outputs were untouched).
- The FTX positive control is detected at p <= 1e-22 on all four tokens, with leverage recovered at
  2.97-3.05x.
- All new Python is ASCII and AST-clean.

### Failed / known limits
- **Resolution.** Rebalances closer than 12 h merge. Events within 12 h of a data edge are
  undetectable. Placement is within about 30 min (synthetic).
- **Power.** For a single year of a 2x *long* token with a daily reset, the schedule regression
  gives |z| of only about 2-5. Real tokens are run over 3-4 years, and their DOWN twins carry the
  strong signal. This is stated in the test docstring.
- **Harmless warnings.** `RuntimeWarning: invalid value encountered in log` comes from candidates
  whose before and after slopes have opposite signs. Those give NaN and are never significant.

---

## Failures & Debugging Trail

1. **cp1252 crash.** Printing the bucket's rejected names failed on non-ASCII bytes. Fixed with
   `ascii()`.
2. **Old-rule check via importlib** failed because a dataclass needs its module in `sys.modules`.
   Registered it first.
3. **Detector v1 (hourly slope from 1m differences + Dimson lags) failed its own null test.** It
   produced 40 false events in 30 synthetic days; hourly slope noise was about 0.13 in log terms.
   Minute differences are swamped by premium noise, which does not shrink over a minute. **Switched
   to level regressions:** the premium is bounded, so a 12 h price range identifies the slope.
4. **Mislocation by 2 h.** A level fit is dominated by the part of its window where price moved
   most, so `|d|` plateaus for hours around an event. The ±2 h refine window missed it. **Fixed by
   searching the split across the whole ±12 h span.**
5. **Real data contradicted the synthetic picture.** BTCUP v2 found 470 "events" with
   `L_pre ~ L_post ~ 1.9` and 2% median moves between them. That is noise, not band triggers.
   Diagnostics:
   - 31.5% of token minutes have no trade, so their closes are stale.
   - Residual premium is 10-35 bp with lag-1 autocorrelation of 0.6-0.9.
   - Calm 12 h windows have perp sd of 0.1-0.4%, leaving the slope unidentified; one window read
     L=1.07 between neighbours at 2.1.
   - **Fixed with per-minute VWAP (gaps, not stale closes), Newey-West SE, and the z >= 4 rule.**
     30% of BLVT candidates now fail significance.
6. **The mean-|d| hour profile had no power for small daily resets.** A synthetic year at 2x gave
   z = 0.3, 1.8 and 0.3 at the planted hour, so BTCUP's flat profile proved nothing. **Replaced as the
   decision statistic by the schedule regression**, which uses the predicted sign and size of a reset.
   Its betas match theory (+2 / -4 at 3x). The mean-|d| profile is kept as descriptive only.
7. **Rate denominator bug.** `rebalances_per_year` divided by minutes-with-prints, which are fewer
   than calendar days under VWAP. Fixed before the final batch; BTCUP went from 52.8 to 36.1 per
   year.
8. **Process slips:**
   - The CLI change broke the in-flight BNB/ADA job's analysis step. Its downloads were unaffected,
     and the final `batch` reran everything from the final code.
   - A shell `&` inside a background command returned at once, so I monitored the log instead.
   - One probe wrote to `/tmp` instead of the scratchpad.
9. **Documentation access.** Binance's FAQ is deleted and WebFetch refuses web.archive.org. I did not
   circumvent this. The route 1 verdict rests on two live Binance pages plus search-index snippets of
   the deleted FAQ, each labelled by provenance above.

---

## Commits

- `8962646` -- Cycle 62B T4: leveraged-token filter made structural - JUP/SYRUP restored, class re-measured
- *(this commit)* -- Cycle 62B T1: BLVT rebalance is drift-triggered, not scheduled - gate stops the cycle

Not pushed, per standing instruction.

---

## Open Items for Chat

1. **Re-triage D1.** The P1 basis ("scheduled, direction knowable in advance") is void for BLVT.
   The taxonomy row is corrected; the triage table is only annotated.
2. **If Jeff wants to look for salvage, these are the only candidates, and none restores the cycle's
   premise at the scale the brief assumed:**
   - (a) **The 00:00 UTC conditional residue.** About 48 excess rebalances over 8 token-lives,
     nature unestablished, mixed with ordinary triggered events at the same hour, no book data for
     most of it, and NAV absent. If it is a daily "check and rebalance if out of tolerance", it is
     the one piece of BLVT flow whose triggering drift precedes it.
   - (b) **FTX BULL/BEAR.** Scheduled, but about 660 token-days in one quarter, with the flow on a
     dead venue.
   - (c) **A live product with a genuine fixed-time reset.** The brief prohibits a live-product
     search this cycle (gated on branch C), so that is Jeff's call.
3. **Stale tokenized-equity list (found while re-measuring T4).** 31 new `*BUSDT` tokenized
   equities have listed since Cycle 60 (ADBEB, ASMLB, CRWDB, GMEB, NFLXB, SMCIB, SQQQB, ...). The
   explicit list in `universe.py` does not exclude them. This matters only if the Cycle 60 runner
   is re-run. A structural fix needs a design choice, since names alone cannot distinguish them.
4. **Duplicate parse logic** now lives in `xsec_reversal/universe.py` and the frozen
   `forced_trade/leveraged.py`. Consolidate if the Cycle 61 modules are ever unfrozen.
5. **Any future leveraged-token or rebalance study must measure on USDT-M perps, not spot,** and
   must plan for no NAV and, before 2023, no book.
6. `split_zip.py zip --repo-delta` anchors on the latest commit that *added* a brief. That is
   `9166f08` (the Cycle 66 brief), not the 62B brief, which was only modified in v2/v3. The delta is
   therefore mtime-selected from 2026-10-15.

---

## Artifacts

- `outputs/forced_trade/c62b/t1_mechanism_table.csv`: the 12-token T1 table.
- `outputs/forced_trade/c62b/t1_<TOKEN>_{events.csv,boundary_profile.csv,summary.json}`: per-token
  outputs.
- `outputs/forced_trade/c62b/t1_timing_pooled_stdout.txt`, `t1_sensitivity_profile6h.txt`: residue
  and robustness evidence.
- `outputs/forced_trade/c62b/t1_granularity_{tokens,underlyings}.csv`, `t1_book_availability.json`,
  `t1_*_probe.py.txt`: sub-Q5/Q6 evidence.
- `outputs/forced_trade/c62b/t4_leveraged_class.json`, `t4_run_stdout.txt`,
  `t4_archive_symbols_raw_2026-10-16.txt`: T4.
- Reproduce T1: `python -m engines.forced_trade.blvt_mechanism batch` (about 15 min, archive cache
  warm).
