# Cycle 59 Phase 2 -- Kibot GLD/GDX load + validation

**Mode:** data validation only. No backtest, no Sharpe, no signal layer, no
data committed. Code is uncommitted pending review of the schema decision.

**Headline:** the 9-column schema assumption **did not hold**. The bid/ask
files are 10 columns and carry **no volume and no trade price**. Everything
else validates, and the spreads are exactly in the band that justified buying
Kibot over FirstRate.

---

## 0. Premise correction -- Phase 1 did not exist

The brief states the loader + validator "were built and unit-tested in Phase 1
(`engines/chan_cpo/`)". That package was not in the tree, in any commit, or in
any branch or stash. The only `chan_cpo` artifacts are
`claude/handoffs/BRIEF_chan_cpo_recon.md` and
`claude/retros/RETRO_chan_cpo_recon.md` -- Cycle 38, an explicitly read-only
investigation that wrote no code. There is also no Cycle 59 anywhere; the
history goes Cycle 58 -> Cycle 60.

So there was no `KibotSchema` to make a "one-line change" to. I built the layer
this phase needed, against the real files rather than synthetic ones:

| File | Purpose |
|---|---|
| `engines/chan_cpo/data_loader.py` | `KibotSchema` + `load_kibot` + `inspect_file`, fatal guard on column-count mismatch |
| `engines/chan_cpo/validate_kibot_data.py` | the `--inspect` / full-validation CLI the brief specifies |
| `engines/chan_cpo/build_panel.py` | Step 4 parquet builder |

### Filename deviations from the brief

- GLD OHLCV files use a **dot**: `GLD.1m_adj.txt`, `GLD.1m_unadj.txt`.
  Everything else uses an underscore. `build_panel.resolve()` tries both
  separators so this cannot silently break later.
- There is no `GLD_1m_bidask_anadj.txt`. The delivered file is
  `GLD_1m_bidask_unadj.txt` -- the suspected typo is not present.

---

## 1. INSPECT -- raw first lines and column counts

**`--inspect` on the bid/ask pair (expected 9, observed 10):**

```
--- GLD: data\external\kibot\GLD_1m_bidask_adj.txt ---
  1| 09/28/2009,04:15,96.89,96.89,96.89,96.89,96.95,96.95,96.95,96.95
  2| 09/28/2009,04:30,96.98,96.98,96.98,96.98,97.04,97.04,97.04,97.04
  3| 09/28/2009,04:57,97.05,97.05,97.05,97.05,97.11,97.11,97.11,97.11
  4| 09/28/2009,07:11,97.25,97.25,97.25,97.25,97.3,97.3,97.3,97.3
  5| 09/28/2009,07:14,97.32,97.32,97.32,97.32,97.35,97.35,97.35,97.35
  observed column counts: [10, 10, 10, 10, 10]

--- GDX: data\external\kibot\GDX_1m_bidask_adj.txt ---
  1| 09/28/2009,08:01,37.28,37.28,37.28,37.28,37.52,37.52,37.52,37.52
  2| 09/28/2009,08:02,37.47,37.47,37.47,37.47,37.52,37.52,37.52,37.52
  3| 09/28/2009,08:20,37.28,37.28,37.28,37.28,37.52,37.52,37.52,37.52
  4| 09/28/2009,08:52,37.47,37.47,37.47,37.47,37.91,37.91,37.91,37.91
  5| 09/28/2009,08:56,37.49,37.49,37.49,37.49,37.65,37.65,37.65,37.65
  observed column counts: [10, 10, 10, 10, 10]
```

**`--inspect --ohlcv` on the plain pair (expected 7, observed 7):**

```
--- GLD: data\external\kibot\GLD.1m_adj.txt ---
  1| 01/02/2009,06:50,85.52,85.52,85.52,85.52,1500
  2| 01/02/2009,07:03,85.6,85.6,85.6,85.6,1000
  3| 01/02/2009,07:25,85.48,85.48,85.48,85.48,4950
  4| 01/02/2009,07:26,85.48,85.48,85.48,85.48,10000
  5| 01/02/2009,07:35,85.8,85.8,85.8,85.8,120
  observed column counts: [7, 7, 7, 7, 7]

--- GDX: data\external\kibot\GDX_1m_adj.txt ---
  1| 01/02/2009,07:07,29.597,29.597,29.597,29.597,5489
  2| 01/02/2009,08:00,29.009,29.009,28.862,28.862,693
  3| 01/02/2009,08:12,28.862,28.862,28.862,28.862,347
  4| 01/02/2009,08:41,28.871,28.871,28.871,28.871,116
  5| 01/02/2009,08:44,28.646,28.646,28.646,28.646,116
  observed column counts: [7, 7, 7, 7, 7]
```

### The corrected schema

```
BIDASK (10)  Date,Time,BidO,BidH,BidL,BidC,AskO,AskH,AskL,AskC
OHLCV  ( 7)  Date,Time,Open,High,Low,Close,Volume
```

Kibot delivers bid and ask as **separate OHLC quartets** -- the case the brief
flagged as possible. Proof it is bid-then-ask rather than anything else:

1. Both quartets independently satisfy OHLC bounds across the full 2.28M bars
   (0 violations each).
2. On a held-out day (2015-06-15/16), **853 of 866** trade closes from the
   OHLCV file fall strictly inside `[quartetA_close, quartetB_close]`, with
   A < B.
3. Across all 1.65M RTH bars the trade print sits at the mid on the typical
   bar (median |trade - mid| = 0.36 bps GLD, 1.17 bps GDX -- both *inside* the
   half-spread). Where the trade falls outside the quote it does so by a median
   of 0.80 bps GLD / 0.70 bps GDX, i.e. sub-tick last-trade-vs-last-quote
   timing offset, not a mapping error.
4. The implied spreads land in the expected band (below). A mis-map would not.

### Consequence for Phase 3 -- read this

**The bid/ask files have no volume column and no trade price at all.** Any
work needing both a traded price and a spread must join the two layouts on
timestamp. `build_panel.py` does this; nothing downstream should re-derive it.

---

## 2. Adjusted vs unadjusted -- proven from data, not filenames

All four adj/unadj pairs differ; none are byte-identical. Compared on the
first 5,000 common timestamps:

| Pair | Bars differing | Median adj/unadj ratio | Last bar identical |
|---|---|---|---|
| GLD OHLCV | 40.3% close, 66.2% volume | 1.000000 | yes |
| GDX OHLCV | 100% close, 100% volume | **0.865425** (vol 1.155500) | yes |
| GLD bid/ask | 88.8% | 1.000100 | yes |
| GDX bid/ask | 100% | **0.865425 / 0.865424** | yes |

Two independent confirmations:

- GDX's adjustment factor is **0.865425 derived from the OHLCV files and
  0.865425/0.865424 derived from the bid/ask files** -- the same number from
  two separately-parsed layouts, with volume moving inversely
  (1.155500 = 1/0.865425). That is a corporate-action adjustment behaving
  exactly as it should, and it independently corroborates the column mapping.
- Every pair converges to an identical final bar, i.e. the adjustment factor
  goes to 1 at the present. Correct by construction.

GLD's adjustment is real but ~1 bp (median ratio 1.0001) -- **smaller than its
own bid/ask spread**, so adj-vs-unadj is immaterial for GLD and material for
GDX (13.5% cumulative since 2009).

`GLD_1m_bidask_unadj.txt` is confirmed the unadjusted twin of
`GLD_1m_bidask_adj.txt`: identical timestamps, 88.8% differing prices in the
same direction as the OHLCV pair, identical tail.

---

## 3. Validation reports

### Bid/ask, adjusted (`-vv --expect-start 2009-01-01 --expect-end 2026-06-30`)

| | GLD | GDX |
|---|---|---|
| total bars | 2,277,398 | 2,218,924 |
| first timestamp (ET) | 2009-09-28 04:15 | 2009-09-28 08:01 |
| last timestamp (ET) | 2026-08-11 19:58 | 2026-08-11 19:58 |
| median RTH bars/day | **390 / 390** | **390 / 390** |
| days at full 390 | 3,625 (85.4%) | 3,952 (93.1%) |
| session days | 4,243 | 4,243 |
| earliest / latest time-of-day | 04:00 / 20:29 | 04:00 / 20:16 |
| extended-hours bars | 629,320 (27.6%) | 570,157 (25.7%) |
| intraday RTH gaps >1min | 1,710 (max 2h02m) | 1,130 (max 2h10m) |
| **crossed, whole bar** (bid_low>ask_high) | **917** (71 in RTH, 0.040%) | **207** (51 in RTH, 0.009%) |
| crossed at close only | 2,724 (403 in RTH) | 669 (267 in RTH) |
| non-positive prices | 6 | 1 |
| locked bars (bid==ask), RTH | 21,143 (1.28%) | 124,569 (7.56%) |
| overnight jumps >10% | 0 | 3 |

### SPREAD -- the headline number

| RTH, locked bars excluded | GLD | GDX |
|---|---|---|
| **median** | **0.79 bps** | **3.52 bps** |
| **p95** | **1.45 bps** | **5.58 bps** |
| p99 | 1.96 bps | 7.39 bps |
| one-way half-spread | 0.40 bps | 1.76 bps |

Both land exactly where the brief predicted (GLD 0.5-2 bps; GDX wider but
single-digit). Including locked bars the medians read 0.79 / 3.34 -- the
locked-excluded figures above are the tradeable ones, since a zero spread is
never actually crossable.

Median RTH spread is stable across all 17 years (GLD 0.47-0.93 bps; GDX
1.32-5.70, peaking 2015 and tightening to 1.32 by 2026). No regime break that
would invalidate a full-sample study.

**Measured round-trip cost to cross both legs of the pair: ~8.6 bps**
(GLD 1.58 + GDX 7.05). Flagging without acting on it: the atlas's 2026-04-02
NEGATIVE verdict assumed **4 bps** round-trip. The measured number is ~2x that.
No arithmetic done here -- that is Phase 3.

### OHLCV, adjusted

| | GLD | GDX |
|---|---|---|
| total bars | 2,374,736 | 2,300,357 |
| range (ET) | 2009-01-02 06:50 .. 2026-08-11 19:58 | 2009-01-02 07:07 .. 2026-08-11 19:58 |
| median RTH bars/day | 390 / 390 | 390 / 390 |
| session days | 4,428 | 4,428 |
| total volume | 39,756,785,389 | 142,669,191,342 |
| OHLC / non-positive / duplicate / ordering | all PASS | all PASS |

The OHLCV files are clean on every check.

### Pair alignment

| | bid/ask | OHLCV |
|---|---|---|
| common bars | **1,946,048** | 2,021,711 |
| common RTH bars | 1,646,747 | 1,719,258 |
| common session days | 4,243 | 4,428 |
| median common RTH bars/day | 390 | 390 |
| retained | 85.4% GLD / 87.7% GDX | 85.1% / 87.9% |

The ~14% loss is almost entirely extended-hours bars where only one symbol
quoted. **Inside RTH the pair is essentially fully aligned at 390 bars/day.**

### Both FAILs, characterised

The validator exits 1. Both failures are understood and neither blocks Phase 3:

1. **`coverage start`** -- real and material. The bid/ask files begin
   **2009-09-28**, not 2009-01. That is a genuine 270-day truncation relative
   to the brief's expectation, and it is a property of the delivery, not the
   parse. The OHLCV files *do* start 2009-01-02, so the extra 9 months exist
   for price-only work but there are no quotes for it.
   (On the OHLCV run this same check fails "1 day short" purely because
   2009-01-01 was a market holiday -- spurious.)

2. **`crossed whole bar`** -- 917 GLD / 207 GDX bars where bid_low > ask_high.
   0.04% and 0.009% of bars, only 71 and 51 of them inside RTH, concentrated
   in 2009-2015 and decaying since, median cross depth 0.85 / 4.33 bps. These
   are aggregation artifacts of Kibot building bid and ask bars independently,
   not locked markets. I added `--allow-crossed N` so Phase 3 can gate on a
   **documented** tolerance rather than deleting the check.

**Non-positive prices (7 bars total):** all are zero-ask prints at exactly
`04:00:00` (6 GLD, 1 GDX; 2016, 2017, 2024, 2025x3, 2017). Unambiguous vendor
corruption, all outside RTH, dropped by the panel builder.

**Overnight jumps -- no unhandled corporate actions.** Every GDX flag is a
genuine market event, and none is near the 2x/0.5x a missed split would give:
2020-03-24 +11.9%, 2020-03-16 -10.8%, 2020-03-12 -9.7% (COVID),
2014-12-03 +10.7% (gold-miner selloff), 2016-06-24 +8.4% (Brexit),
2026-06-15 +8.3%. GLD: zero.

---

## 4. Parquet

```
data/external/kibot/pair_gld_gdx_1m_adj.parquet     121.2 MB   1,946,045 x 36
data/external/kibot/pair_gld_gdx_1m_unadj.parquet   121.8 MB   1,946,045 x 36
```

Both built (adjusted is primary; unadjusted retained since GDX's 13.5%
adjustment makes real price levels and tick-size effects differ).

Join policy, since neither layout is self-sufficient:

- index = timestamps where **both** symbols have quotes (inner join) -- the
  pair is untradeable on a bar where one leg has no quote
- trade OHLCV **left**-joined onto that index, so a minute that quoted but
  never printed keeps its row with NaN trade fields. Sparsity is liquidity
  information, not missing data. In RTH, print coverage is 100.00%.
- the 7 corrupt bars dropped
- `{sym}_crossed` and `{sym}_locked` carried as boolean columns so Phase 3 can
  filter on them explicitly rather than rediscovering them

Verified: **0 mismatched values** against the source text on 20,000 randomly
sampled bars, and load time **0.22s vs ~14s** to parse the text.

Raw `.txt` files remain the source of truth. `data/` is gitignored at
`.gitignore:137`; `git status` sees no Kibot file as trackable. Nothing was
written to `crypto_data.db` (24.7 GB, untouched -- note it is now 24.7 GB, not
the 12+ GB the brief assumed).

---

## Confirmed usable range

| Purpose | Window |
|---|---|
| **spread-aware pair work (primary)** | **2009-09-28 08:01 -> 2026-08-11 19:58 ET**, 4,243 session days, 1,946,045 common bars |
| price/volume-only pair work | 2009-01-02 08:00 -> 2026-08-11 19:58 ET, 4,428 session days |

Coverage runs ~6 weeks *past* the brief's expected 2026-06-30 end, not short of
it. Data is US/Eastern wall clock, kept tz-naive on purpose (the only ambiguous
hour is the autumn DST repeat at 01:xx, far outside any session).

## Open decisions for Phase 3

1. **Replication window** -- the Chan replication starts 2009-09-28, not
   2009-01, if it needs spreads. Confirm that is acceptable.
2. **`--allow-crossed` tolerance** -- pick a documented value (917/207 is the
   observed count) or filter crossed bars in the signal layer.
3. **RTH-only vs extended hours** -- panel keeps all hours; ~26% of bars are
   extended-hours with far wider spreads (GLD all-hours p95 5.86 vs RTH 1.45).
4. **Commit** -- the three new modules are uncommitted pending your review of
   the schema decision. No data files will be committed.
