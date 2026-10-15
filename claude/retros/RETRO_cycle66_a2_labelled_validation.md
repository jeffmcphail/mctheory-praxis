# Retro: Cycle 66 -- A2 Validation Against Labelled Liquidations

**Brief:** `claude/handoffs/BRIEF_cycle66_a2_labelled_validation.md` (v2, `8dc0a2d`)
**Date:** 2026-10-15
**Sessions:** two -- 01:39-02:59 EDT (all computation finished by 02:57; the session was ended by a
Windows Update restart, not a crash), then recovery and completion the same afternoon
**Status:** COMPLETE. A2 refuted as a tradeable scenario; the Cycle 61 detector does not detect
cascades; the signature study is closed.

---

## Summary

Cycle 61 built an A2 flow-burst detector and could not compute a false-positive rate because no
ground truth existed. Cycle 62A's Bybit `allLiquidation` collector has since recorded 191,802
liquidations over 57 days. This cycle scored the detector against them, characterised what real
cascades look like in the spot tape, and sized the signature question. `crypto_data.db` was only
ever opened `mode=ro`; everything except two targeted reads runs from a parquet snapshot.

**The verdict leads with an oracle result: even a perfect detector does not make A2 pay.** T3
measures reversion from the labels themselves, so detection error is zero by construction. From the
first instant a cascade is knowably over, the reversion across the 210 BTC+ETH clusters >= $1M is
-0.6 bps at 5 minutes and -2.7 bps at 15 (day-clustered SE 3.0 and 4.6): zero, and 3.2 / 2.6 SE
below the cheapest taker cost on the estate. The only positive reversion in the study requires
look-ahead, and even that sits at the cost floor. Detector quality is not the binding constraint.

| Task | Question | Answer |
|---|---|---|
| T1 | What labelled data exists? | 191,802 rows, 6 symbols, 90.8% of 1,358 h observed; **210** BTC+ETH clusters >= $1M, **8** >= $10M |
| T2 | Does the Cycle 61 detector find cascades? | **No.** 0 of 8 >= $10M, 0 of both nine-figure manual cascades; it fails on its imbalance condition, which real cascades never meet |
| T3 | Do real cascades revert, exploitably? | **No.** A large move that does not come back; reversion from the first knowable instant is zero |
| T4 | Is a signature study viable? | **No, closed** -- a better detector buys access to a move that does not pay |

---

## Amendments -- recorded verbatim

The brief on disk (v1) was incomplete: five amendments were agreed in chat before any code was
written. They now live in brief v2 as a binding section. They are reproduced here as well, because
the retro records what was run. The text below is the agreement exactly as the pre-crash session
received it (transcript `c47f7a3d-7f35-4a28-bdc6-390f7429d725.jsonl`, line 261, 2026-10-15 06:20
UTC). It survived the interruption on disk. Brief v2's amendments (a)-(e) and the reconstruction
in the recovery prompt match it in substance.

```
Go. Both defaults accepted, with constraints and four additions.

AUG 19 CASCADE - include as flagged segment, report T2/T3 with and without, as you
proposed. One constraint: the dataset's first row IS the cascade (15:27). The collector
started at or during the event, so there is no pre-event label baseline from that session.
It can serve as an event; it must not contribute to any baseline or control computed from
the liquidations table - that is selection on the dependent variable. Price/flow/book
profiles around it are fine, since trades and order_book_snapshots are continuous and
independent of the liquidation collector.

SCAN TIMING - start now. Stop rule: if the scans are not finished by 02:40 local, kill
them and restart after 03:45. Do not let a scan run into the 03:00 mirror.

APPROVED AS YOU DESCRIBED, do not re-litigate:
- Mask built from logs/bybit_liquidation_collector.log, not collector_gaps.
- Retry chains merged into one outage. Your reading of :410-424 is right; "any gap still
  open" was my wording and it is wrong against this collector's behaviour.
- Precision reported next to its chance rate and lift, cluster size as a swept parameter.
- Full-span cache rebuild to a new path, with the Cycle 61 window reproducing 2092/229/26
  BTC and 1142/72/5 ETH as a regression check. That is a control that can fail. Good.
- T4 installs in a throwaway 3.12.10 venv, .venv untouched.

FOUR ADDITIONS:

1. MASK THE PERIODIC HANDOFF, NOT JUST LOGGED GAPS. Runs last 3550s on an hourly
   schedule, so there is a fixed ~50s blind spot at roughly :59:10-:00:00 every hour -
   about 17.7 hours total, and periodic rather than random. Cascades cluster by
   time-of-day, so a periodic blind spot can bias both event and control sets. Mask those
   windows explicitly. Free validation while you do: check whether the liquidations table
   shows a deficit in that minute-of-hour. If the mask is built correctly the deficit
   should be there; if it is not, the mask is wrong and I want to know.

2. VENUE INVENTORY, in T1, one line each. State which venue and instrument type trades
   and order_book_snapshots actually cover. Then check whether any Binance PERP trades
   table exists with coverage over the label span. If it does and a third scan fits inside
   the timing guard, run the unchanged detector against it as well - Bybit perp
   liquidations are far likelier to register in Binance perp than Binance spot, and that
   scoring run removes most of the venue confound. If no such table exists, say so in one
   line and move on. Do not build one.

3. RECALL CONDITIONED ON CLUSTER SIZE, and matching on INTERVAL OVERLAP. The venue
   mismatch you found means pooled recall is uninterpretable, so report recall bucketed by
   cluster notional and event count. A $180M cascade should register on any venue; if
   recall is near zero even in the top bucket, that is detector failure rather than a venue
   artifact, and that distinction is the whole point of T2. Separately, replace point
   tolerance with interval overlap: Cycle 61's detector fires on sub-minute bursts (229 at
   W=60s, 1 at W=300s) while labelled cascades run minutes, so score "did the detector fire
   anywhere inside the labelled cluster" alongside the strict match. That separates wrong
   timescale from no signal, and neither requires retuning the detector.

4. BANKRUPTCY PRICE - establish the direction, do not assert it. Compare Bybit's price
   against spot mid at the same timestamp, split by liquidation side, and report the
   observed sign and magnitude of the bias. Report event count and quantity x spot price
   alongside raw quote_qty as you planned.

RETRO: record both collector defects explicitly with file:line - the overwritten open-gap
pointer at engines/bybit_liquidation_collector.py:410-424, and the ~85 hourly runs that
never started (1273 starts against ~1357 hours). Neither is fixed this cycle. They need to
survive into a brief rather than live in this transcript.

Everything else as briefed.
```

| Brief v2 | Above | Where it was implemented | Outcome |
|---|---|---|---|
| (a) | AUG 19 CASCADE | `labels.py:607-614`, `label_inventory.py:574-580` (`segment="manual"`, censored by construction) | Events only. Never in a baseline, chance rate, control or mask validation. T2/T3 reported with and without it. |
| (b) | Addition 1 | `labels.py:506-550` (band measured from the runs), `label_inventory.py` `handoff_validation` + the two residual tests | Deficit present. The residual beyond the mask is settled as seasonality (T1.2). |
| (c) | Addition 2 | `run_labelled.py` T1 venue inventory | No Binance perp trades table exists. One line, no third scan. |
| (d) | Addition 3 | `label_scoring.py` `label_to_detector` + `run_labelled.py` T2.3 | Recall bucketed by notional and by count, overlap and strict, with null rates and lift. |
| (e) | Addition 4 | `label_inventory.py` `bankruptcy_bias`, `run_labelled.py` T1 label table | Sign and magnitude measured. Count, raw `quote_qty` and qty x spot reported together. |

The 02:40 stop rule never fired: both scans finished at 02:30:40.

---

## T1 -- Inventory of the labelled data

### Venue inventory (amendment c)

- `trades`: Binance **spot** BTC/USDT and ETH/USDT, via ccxt REST `fetch_trades` with a `fromId` cursor (`engines/crypto_data_collector.py:1156`). BTC and ETH only.
- `order_book_snapshots`: Binance **spot** top-10 REST snapshots every 10 s (`crypto_data_collector.py:1003`). BTC and ETH only.
- `liquidations`: Bybit USDT-linear **perp** `allLiquidation` for BTC, ETH, SOL, XRP, ADA and AVAX. Anything outside these six is unobserved, not quiet.
- **Binance perp trades: no table exists in `crypto_data.db` or any `data/*.db`.** No third scan was run, so the spot-vs-perp venue confound stands. Amendment (d) handles it.

### Label rows

| symbol | rows | manual (Aug 19) | forced-sell share | quote_raw (bankruptcy px x qty) | qty x Binance spot |
|---|---|---|---|---|---|
| BTCUSDT | 77,957 | 3,091 | 41.8% | $915.5M | $915.1M |
| ETHUSDT | 43,878 | 1,455 | 48.5% | $510.2M | $510.4M |
| SOLUSDT | 30,816 | 292 | 51.7% | $128.5M | n/a |
| XRPUSDT | 29,532 | 101 | 64.2% | $133.1M | n/a |
| ADAUSDT | 6,017 | 24 | 64.1% | $20.4M | n/a |
| AVAXUSDT | 3,602 | 10 | 62.7% | $5.4M | n/a |

The labels run from 2026-08-19 15:27:17 to 2026-10-15 06:06:50 UTC: 191,802 rows, of which 4,973
are in the manual 62A segment. The two notional bases agree to within 0.05% in aggregate. The
bankruptcy offset is symmetric (see below), and liquidated longs and shorts net out.

### Coverage mask

The mask is built from the collector log, not from `collector_gaps` (approved). It runs from the
first scheduled connect at 2026-08-19 16:00:02 to the analysis end at 2026-10-15 06:20:50, a span
of 1,358.3 h.
- **Observed time:** the log gives 1,237.6 h. After the hourly band is removed, 1,233.2 h (90.8%) remain.
- **Runs:** 1,274 in total: 1,252 `ok`, 15 killed or still running, 7 quiet hours.
- **Hourly slots:** 1,358, of which **85 had no run start, in 19 outages** (defect 2, below).
- **Handoff holes between runs:** 1,237 of them, totalling **17.65 h** (p50 51.2 s, p95 53.6 s, max 94.4 s).
- **`collector_gaps`:** 298 rows become 266 outages once retry chains are merged. That absorbs 32 permanently open rows (defect 1), leaving 0 unresolved. All 266 match a log outage, with 0 ms difference at both ends.
- **Pads:** observation starts 1 s after each `connected to` line and ends 60 s before each logged disconnect.

### Mask validation (amendment b) -- the deficit is there, and the residual is settled

- **Rows outside the log mask:** 1,014 of 186,829 scheduled rows.
  - 941 fall in the conservative pre-disconnect pad.
  - 55 are within 2 s of an edge.
  - The remaining 18 all fall at most 4.0 s before a logged (re)connect: pre-connect delivery on subscribe.
- **No row lands in time the mask calls unobserved for any other reason.**
- **Handoff band:** :59:06.9 to :00:10.4. It is measured from the runs as p01 of the window end minus a 2 s pad, and p99 of the first connect plus 3 s. It is applied to every hour. Inside it the table holds 866 rows against 4,573 expected at the out-of-band rate.
- **Deficit:** **81.1% observed against the 66.7% the log mask predicts.** The deficit the brief asked to see is present.
- **The excess** is the residual the brief says to settle or bound. Two tests, then the bounding rule:

**Test 1: leading-edge latency, which would mean the new run is dead longer than the log says. Rejected.**
- **Sample:** 1,248 on-schedule handoffs.
- **Connect line:** at p50 :00:02.0 and p99 :00:07.5.
- **First liquidation after the start line:** p01 :00:03.3, **p05 :00:08.6**, p10 :00:16.4, p50 :02:33. The p05 floor sits **1.8 s inside** the band's :00:10.4 edge, and 76 runs logged a row before that edge.
- **First row after the connect line:** as little as 0.67 s (p01). Data flows within a second of connecting.

**Test 2: a local baseline instead of the hourly mean. This settles it.**
- **Method:** expected band rows come from the seconds immediately before and after the band, on the same hours, at 1-second resolution. The 90% CIs resample whole UTC days (58).
- **Exact-second frame:** the band [3546 s, 11 s) has 4.82 log-exposed hours out of 24.52 calendar hours.

| unit | baseline | observed / expected given exposure | outgoing tail | incoming head |
|---|---|---|---|---|
| rows | global hourly mean | **0.70** [0.51, 0.91] | 0.48 [0.29, 0.69] | 0.87 [0.59, 1.22] |
| rows | local +-60 s | **0.95** [0.78, 1.17] | 0.85 [0.52, 1.24] | 1.01 [0.80, 1.30] |
| rows | local +-120 s | 0.84 [0.68, 1.06] | 0.88 [0.60, 1.15] | 0.82 [0.62, 1.13] |
| rows | local +-300 s | 0.85 [0.69, 1.02] | 0.91 [0.59, 1.20] | 0.82 [0.61, 1.06] |
| active seconds | global hourly mean | 1.01 [0.85, 1.19] | 0.61 [0.46, 0.76] | 1.33 [1.07, 1.61] |
| active seconds | local +-60 s | 1.14 [0.98, 1.31] | 0.92 [0.69, 1.17] | 1.24 [1.02, 1.49] |

How to read it:
- **Against the hourly mean**, the residual reproduces and its CI excludes 1.0.
- **Against the adjacent minutes**, every CI includes 1.0.
- **In active seconds** (a second with at least one row), which a handful of large clusters cannot dominate, the residual is absent even against the global mean.

The cause is market seasonality, visible in fully observed time:
- **Liquidations run at 0.3-0.6x** the hourly mean across :57:00-:58:59.
- **Binance spot shows the same lull** (0.69-0.89x), then a 1.65-1.75x spike at :00:00.

The outgoing tail sits in the lull and the incoming head in the spike. Rows are dominated by large
clusters, which amplifies both. **The mask is right.**

**Bounding rule, run as confirmation although Test 2 closed the residual.** The band was widened
to :59:00-:00:30 (`--band-soh 3540 30`, `sensitivity/band_bound_5900_0030/`). Observed time falls
by 9.3 h, and nothing material moves:
- clusters >= $1M stay at 139/71, and >= $10M at 4/4;
- precision and recall move at most 0.0015;
- book ratios move in the third decimal;
- every reversion figure moves less than 0.5 bps.

See the sensitivity table.

### Clusters (gap-based, per symbol; primary gap 30 s, pre-registered)

| gap | BTC clusters (sched, uncensored) | ETH | BTC >= $1M | ETH >= $1M | BTC >= $10M | ETH >= $10M |
|---|---|---|---|---|---|---|
| 10 s | 10,714 | 8,910 | 123 | 61 | 3 | 6 |
| **30 s** | **7,757** | **6,746** | **139** | **71** | **4** | **4** |
| 60 s | 6,055 | 5,418 | 127 | 72 | 3 | 4 |

- **The T3 gate at gap 30 s:** 1,640 BTC+ETH clusters at >= $100k, **210 at >= $1M (viable)**, and **8 at >= $10M (too small)**.
- **Cluster size:** p50 under $5k. p99 is 135 events (BTC) and 85 (ETH). The largest scheduled clusters are $23.4M (BTC, 1,730 events) and $33.8M (ETH, 1,682 events).
- **Other symbols:** SOL, XRP, ADA and AVAX have labels only (9, 9, 0 and 0 clusters >= $1M) and no flow or book data.

### Bankruptcy price (amendment e) -- measured, not asserted

Bybit's price was compared with the nearest Binance spot mid within 5 s:

| asset | liquidated | n | median bps vs mid | share below mid | offset (half the side difference) | common component |
|---|---|---|---|---|---|---|
| BTC | shorts (BUY) | 41,573 | **+31.2** | 0.2% | **34.7** | -3.5 |
| BTC | longs (SELL) | 31,296 | **-38.2** | 100.0% | | |
| ETH | shorts (BUY) | 20,851 | +32.3 | 1.7% | 35.8 | -3.5 |
| ETH | longs (SELL) | 20,309 | -39.2 | 98.7% | | |

The bias runs in the bankruptcy direction on both sides, about 35 bps each way, which is why raw
`quote_qty` overstates notional for liquidated shorts and understates it for longs. The -3.5 bps
common component is the perp-spot basis plus timing. The <= 1 s subset gives the same answer
(34.5 / 35.5).

### Overlap with flow and depth, and tape completeness

- **Trade buckets:** they cover the whole label span. BTC has every 10 s bucket; ETH has 99.96% (175 empty buckets, no gap over 60 s).
- **The book has its own gaps:** 1,286 gaps over 30 s, totalling 108.2 h, the longest 28.6 h. Within that, a **~52 s blind spot starting at :34-:35** recurs in 1,252 hours: the book collector runs the same 3550 s schedule, offset to :35. Book values in its outages read NaN, never calm.
- **Contiguity (DB, `mode=ro`, run at 14:34 EDT): 0 missing trade ids.** That covers 149 BTC and 75 ETH cluster windows (cluster +-60 s, 4.36M trades) and 400 random control windows. The spot tape is complete wherever the detector was asked to see a cascade.

---

## T2 -- The confusion matrix Cycle 61 could not compute

**Regression check:** today's code on Cycle 61's cache, and Cycle 61's window cut from the new
cache, both reproduce **2092/229/26 BTC and 1142/72/5 ETH exactly** (6/6). The single
value-mismatch bucket per asset is Cycle 61's last, partially filled bucket. The detector is
scored exactly as built (`cascade.py` mtime 2026-08-19).

### Window-level 2x2 (W = 60 s detector windows in observed time; cluster gap 30 s; tau 60 s)

| asset | setting | floor | TP | FP | FN | precision [90% CI] | chance | lift | recall | FPR |
|---|---|---|---|---|---|---|---|---|---|---|
| BTC | loose | $100k | 162 | 685 | 2,941 | 19.1% [16.0, 22.5] | 4.29% | 4.46 | 5.2% | 0.99% |
| BTC | base | $100k | 15 | 56 | 3,088 | 21.1% [14.3, 29.0] | 4.29% | 4.92 | 0.5% | 0.08% |
| BTC | strict | $100k | 0 | 2 | 3,103 | 0.0% | 4.29% | 0 | 0.0% | 0.003% |
| ETH | loose | $100k | 60 | 503 | 2,127 | 10.7% [7.9, 13.6] | 3.02% | 3.52 | 2.7% | 0.72% |
| ETH | base | $100k | 0 | 16 | 2,187 | 0.0% | 3.02% | 0 | 0.0% | 0.02% |
| ETH | strict | $100k | 0 | 0 | 2,187 | n/a | 3.02% | n/a | 0.0% | 0.00% |
| BTC | loose | $1M | 30 | 816 | 500 | 3.5% [2.2, 5.2] | 0.73% | 4.83 | 5.7% | 1.14% |
| ETH | loose | $1M | 2 | 559 | 276 | 0.4% | 0.38% | 0.93 | 0.7% | 0.78% |
| any | any | $10M | 0 | all | 22 / 28 | 0.0% | 0.03-0.04% | 0 | 0.0% | -- |

Seen from the detector's side (each event, tau 60 s, floor $100k):
- **BTC loose:** 808 events. 151 match a cluster: **18.7% against a 4.4% chance rate (lift 4.2)**, with 92% side agreement.
- **BTC base:** 70 events, **21.4%** (lift 4.8).
- **ETH loose:** 547 events, 10.2% (lift 3.4).
- **ETH base:** 0 of 16.

So the detector is better than random at landing near small liquidation clusters. Four of every
five of its events still have no liquidation cluster nearby at all.

### Label -> detector: recall by cluster size (amendment d), pooled BTC+ETH

| setting | size | n | recall overlap | null | lift | recall strict | null |
|---|---|---|---|---|---|---|---|
| loose | < $100k | 12,863 | 3.4% | 1.1% | 3.2 | 4.4% | 1.9% |
| loose | $100k-1M | 1,430 | 9.7% | 1.5% | 6.6 | 9.9% | 1.8% |
| loose | $1M-10M | 202 | 11.9% | 2.0% | 6.1 | 10.4% | 1.8% |
| loose | **>= $10M** | **8** | **0 / 8** | -- | -- | **0 / 8** | -- |
| base | $1M-10M | 202 | 0.0% | 0.1% | 0 | 0.0% | 0.1% |
| base | >= $10M | 8 | 0 / 8 | -- | -- | 0 / 8 | -- |
| strict | all | 14,503 | 0.0% | 0.0% | -- | 0.0% | 0.0% |

- **The manual 62A cascades:** **BTC $144.0M (245 s) and ETH $33.2M (54 s) were not detected** at any setting, by overlap or by strict match.
- **By event count** (base): **0 of the 174 clusters with 100 or more liquidations**.
- **Reading:** the top size bucket is where the venue confound cannot hide, since a nine-figure cascade registers on any venue. Recall there is zero. **That is detector failure, not a venue artifact.**
- **Timescale is not the explanation either.** Interval-overlap recall is no better than strict match, and Cycle 61's own window sweep confirms it. At W = 300 s the base detector fires 0 times (BTC) and once (ETH) inside the label window; at W = 900 s, never.

### Why it fails: the imbalance condition

The base rule needs K > 5 (flow burst), I > 0.8 (aggressor imbalance) and M > 3 (move z) in one
60 s window. Inside clusters >= $1M:

| asset | burst (null) | K > 5 | \|imbalance\| (null) | I > 0.8 | \|move z\| (null) | M > 3 |
|---|---|---|---|---|---|---|
| BTC $1M-10M (135) | 19.0 (1.3) | 95% | **0.44 (0.49)** | **0%** | 3.9 (0.65) | 67% |
| ETH $1M-10M (67) | 39.0 (1.2) | 99% | **0.34 (0.44)** | **0%** | 5.7 (0.61) | 91% |
| BTC >= $10M (4) | 37.3 | 100% | 0.41 | 0% | 8.2 | 100% |
| ETH >= $10M (4) | 112.7 | 100% | 0.26 | 0% | 15.9 | 100% |

Every one of the 15 largest clusters meets K and M; none meets I. Real cascades show a huge flow
burst and a large move on Binance spot, but the aggressive flow is **two-sided**. Imbalance inside
large clusters is *lower* than at random times. The forced flow on Bybit perp gets absorbed and
arbitraged on spot, so a minute is never 80% one-sided. The detector's distinguishing condition is
the one cascades never produce.

### Order-book stress at labelled events (vs matched control)

| set | n | depth | hit-side depth | spread |
|---|---|---|---|---|
| BTC labels >= $100k | 963 | **0.92** | **0.88** | 1.00 |
| BTC labels $1M-10M | 132 | 0.88 | 0.84 | 0.99 |
| ETH labels >= $100k | 667 | 0.91 | 1.04 | 1.00 |
| ETH labels >= $10M | 4 | 0.53 | 0.71 | 5.7 |
| BTC detector base events | 70 | 0.95 | 0.62 | 1.00 |

These are span means against each span's own trailing 24 h median, divided by the same for its
matched control. Cycle 61 found depth 1.02 on detector events. Labelled cascades show **real but
moderate depth thinning**, about 8-12% at >= $100k. The four largest ETH cascades show genuine
stress (depth roughly halved, spread 5.7x). The book collector's :34-:35 gaps read as missing.

---

## T3 -- What a labelled cascade looks like (210 clusters >= $1M; 20 matched controls each)

- **Events:** 139 BTC and 71 ETH, from the scheduled segment, uncensored, at gap 30 s.
- **Controls:** matched on UTC hour and trailing 60-minute volatility quintile, drawn from observed time at least 15 minutes from any cluster >= $100k. There are 2,780 and 1,420, and none comes from a thin cell.
- **Size:** p50 $1.6M / $1.7M.
- **Duration:** p50 54 s / 55 s, p90 137 s / 117 s.
- **Forced-sell share:** 41% / 56%.

### Event-time profile (medians, oriented to the forced flow; event / control)

| offset from start | BTC price bps | ETH price bps | BTC hit-side depth | ETH hit-side depth |
|---|---|---|---|---|
| -600 s | -27.3 / +0.4 | -28.8 / +0.1 | 0.92 / 1.37 | 0.93 / 1.28 |
| -60 s (reference) | 0 / 0 | 0 / 0 | 0.90 / 1.48 | 0.79 / 1.22 |
| +0 s | +13.0 / -0.1 | +16.8 / -0.2 | **0.66 / 1.40** | **0.62 / 1.30** |
| +60 s | +29.2 / -0.1 | +56.3 / 0.0 | 0.73 / 1.38 | 0.69 / 1.25 |
| +300 s | +26.1 / +0.3 | +52.9 / -0.2 | 0.91 / 1.45 | 0.68 / 1.31 |
| +900 s | +26.7 / 0.0 | +57.9 / 0.0 | 0.83 / 1.43 | 0.93 / 1.25 |
| +3600 s | +25.4 / -0.8 | +56.6 / -3.3 | 0.87 / 1.51 | 0.84 / 1.28 |

Three things stand out:
- **The move starts before the first forced fill.** Price runs about 27 bps in the forced direction over the nine minutes before the cluster begins (-600 s to -60 s). Price moves trigger these liquidations at least as much as the liquidations move price.
- **The book's hit side thins to roughly half its control level at onset.**
- **The move is permanent.** One hour later price is still +25 bps (BTC) and +57 bps (ETH) from the reference, against a control at zero.

### Size, move and reversion -- gross bps first

All figures are the excess over matched controls. The control means are at most 0.4 bps at
horizons up to 30 minutes, so the adjustment is immaterial. SD is per cluster; SE is
day-clustered; CIs are 90% day-block bootstrap.

**Knowable entry**, measured from the end + 30 s, the first instant the label stream itself can say a cluster is over:

| | n | 1 m | 5 m | 15 m | 30 m |
|---|---|---|---|---|---|
| BTC | 139 | -1.3 (SE 1.3) | **-0.5** (SE 3.1) [-5.3, +4.9] | **+0.9** (SE 3.5) [-4.8, +7.1] | +0.6 (SE 5.9) |
| ETH | 71 | -4.2 (SE 4.3) | **-1.0** (SE 4.9) [-8.2, +7.6] | **-9.6** (SE 9.4) [-23.4, +6.6] | -7.7 (SE 12.4) |
| **BTC+ETH** | **210** | -2.3 (SE 1.7) | **-0.6** (SD 50, SE 3.0) [-5.1, +4.8] | **-2.7** (SD 72, SE 4.6) [-9.9, +5.8] | -2.2 (SE 6.9) |

**Look-ahead reference**, measured from the cluster's last forced fill. This is not knowable at the time; the code labels it `_end`:

| | 1 m | 5 m | 15 m |
|---|---|---|---|
| BTC | +5.3 (SE 1.4) [3.1, 7.5] | +4.9 (SE 2.9) [0.5, 10.1] | +7.1 (SE 3.6) [1.3, 13.3] |
| ETH | +7.8 (SE 4.1) [1.4, 14.6] | +10.9 (SE 5.5) [2.8, 20.6] | +0.7 (SE 10.5) [-14.5, 18.7] |
| BTC+ETH | +6.1 (SE 1.6) [3.5, 8.8] | +6.9 (SE 3.1) [2.4, 12.6] | +5.0 (SE 4.5) [-2.1, 13.5] |

- **The move itself** (start - 60 s to end): BTC +38.6 bps [35.0, 43.4], ETH +79.6 [65.4, 97.2], pooled +52.5. 99% of clusters move with the forced flow.
- **Hit rates** at knowable entry are 49-55%.
- **Dispersion:** a typical cluster's 5-minute reversion is +-50 bps around a mean of zero. The SE (3.0) is a third of the cheapest taker cost.
- **The manual cascades** change no knowable-entry cell by more than 1.3 bps (ETH 5 m), and no pooled cell by more than 0.4. Look at the "with manual" rows in `t3_reversion.csv`.

### Net of a round-trip cost grid (event mean, knowable entry, pooled BTC+ETH)

| horizon | gross | 2 bps (maker ~1/side) | 9 bps (HL taker) | 20 bps (Binance spot taker) | 30 bps (Cycle 55 real-money) |
|---|---|---|---|---|---|
| 1 m | -2.5 | -4.5 | -11.5 | -22.5 | -32.5 |
| 5 m | -0.9 | -2.9 | -9.9 | -20.9 | -30.9 |
| 15 m | -2.6 | -4.6 | -11.6 | -22.6 | -32.6 |
| 60 m | -9.0 | -11.0 | -18.0 | -29.0 | -39.0 |

- **No knowable-entry cell is positive at any cost tier,** except BTC alone at 240 minutes (+14.4 gross, +5.4 at 9 bps). That cell's excess CI is [-11.6, +33.9], ETH's 240-minute figure is -3.2, and it is not a signal.
- **What is in the 9 bps:** it is Cycle 57's Hyperliquid perp taker round trip (`CYCLE57_REAL_SHARPE.md:101-102`, "HL perp RT (~9 bps taker / ~3 bps maker)"), roughly 4.5 bps a side.
- **It is fees only:** no spread, slippage, impact, funding or adverse selection. It is also a perp-venue fee applied to a move measured on Binance spot last-trade prices.
- **It is the floor of the taker grid, not an estimate.** Binance spot taker is 20 bps; Cycle 55's real-money estimate is 30.

### Sensitivity

| variant | observed h | >= $1M BTC/ETH | >= $10M | BTC loose precision vs chance | overlap recall $1M-10M | recall >= $10M | BTC depth vs matched | BTC rev 5m knowable / look-ahead | ETH rev 5m knowable / look-ahead | BTC rev 15m knowable / look-ahead |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1233.2 | 139/71 | 4/4 | 18.7% vs 4.4% | 11.9% | 0/8 | 0.919 | -0.5 / +4.9 | -1.0 / +10.9 | +0.9 / +7.1 |
| band +60 s pre | 1212.2 | 138/71 | 4/4 | 18.8% vs 4.7% | 11.9% | 0/8 | 0.920 | -2.1 / +4.2 | -1.2 / +10.4 | -0.4 / +5.9 |
| bound :59:00-:00:30 | 1223.9 | 139/71 | 4/4 | 18.7% vs 4.5% | 11.9% | 0/8 | 0.922 | -0.5 / +4.9 | -1.2 / +10.4 | +0.9 / +7.1 |
| disconnect pad 0 | 1237.6 | 139/71 | 4/4 | 18.6% vs 4.4% | 11.9% | 0/8 | 0.921 | -0.4 / +4.9 | -1.0 / +10.9 | +1.1 / +7.3 |
| cluster gap 10 s | 1233.2 | 123/61 | 3/6 | 16.7% vs 3.9% | 8.6% | 0/9 | 0.911 | -0.3 / +2.8 | -0.7 / +2.6 | +5.2 / +7.4 |
| cluster gap 60 s | 1233.2 | 127/72 | 3/4 | 19.9% vs 5.1% | 15.6% | 0/7 | 0.941 | +0.2 / +8.4 | -1.9 / +9.6 | -0.4 / +7.3 |

- **Knowable-entry reversion is zero in every variant.** With a 10 s gap, entry comes 20 s sooner and still captures nothing at 1-5 minutes.
- **The look-ahead bounce grows with the gap**, since a longer silence selects stronger turns. That is the conditioning at work.
- Each variant's printed report is in its folder as `run_stdout.txt`.

---

## T4 -- Sizing the path-signature question (no implementation)

- **Supervised sample** (BTC+ETH, gap 30 s, uncensored): 1,640 clusters at >= $100k, **210 at >= $1M** (139 BTC, 71 ETH), and **8 at >= $10M**.
- **Channels:**
  - price and signed aggressive flow come from Binance spot on a 10 s grid from 2026-04-29;
  - top-10 depth and spread come from 10 s REST snapshots, with 108 h of gaps including the :34-:35 blind spot;
  - labels exist only from 2026-08-19, so the supervised window is about 57 days, BTC and ETH only.

| channels | level 2 | level 3 | level 4 | level 4 vs 210 (>= $1M) | level 4 vs 8 (>= $10M) |
|---|---|---|---|---|---|
| 4 (price, flow, depth, spread) | 20 | 84 | 340 | 1.6x | 42.5x |
| 5 (+ time augmentation) | 30 | 155 | **780** | **3.7x** | **97.5x** |
| 10 (lead-lag of the 5) | 110 | 1,110 | 11,110 | 52.9x | 1,388.8x |

Against the brief's own criterion, features outnumbering clusters by an order of magnitude:
- **Met** at >= $10M, the bucket the detector question is actually about (97.5x), and by the lead-lag specification at >= $1M (52.9x).
- **Not met** at the reference specification against the 210 clusters >= $1M (3.7x). A level-3 signature (155 terms) would fit inside that sample.

So dimensionality alone does not rule out a modest study at >= $1M. **The oracle result closes it
anyway.** A signature study exists to build a better detector, and a better detector buys access
to a move that does not pay even when detection is perfect.

**Tooling** on Python 3.12.10 / Windows 11 26200, with Smart App Control on and no MSVC. Each
package was tested in a throwaway venv; `.venv` was untouched.
- **`roughpy` 0.3.0** installs and matches the reference.
- **`signax` 0.2.1** computes. It is float32 by default (error 3.7e-5) and matches to 8.5e-14 with `JAX_ENABLE_X64=1`.
- **`esig` 1.0.0** installs, but `stream2sig` raises a RoughPy ValueError. With RoughPy 0.2.0, Smart App Control blocks the DLL.
- **`pysiglib` 4.0.0** installs, but Application Control blocks its DLL (WinError 4551).
- **`iisignature` 0.24** has no py3.12 Windows wheel and needs MSVC to build.
- **`signatory`** is unmaintained (it targets torch 1.9) and fails to build.

The probe and its raw output are archived as `t4_probe_sig_tooling.py.txt`,
`t4_tooling_pass2.txt`, `t4_tooling_pass3.txt` and `t4_tooling_results.json`.

---

## Verdict on A2

**Even a perfect detector does not make A2 pay.**

T3 measures reversion from the labels, not from the detector, so detection error is zero by
construction. Every cascade is known, and known to be over, at the first instant the label stream
itself can say so: 30 s after the last forced fill.
- **The numbers:** from that instant, reversion across the 210 BTC+ETH clusters >= $1M is **-0.6 bps at 5 minutes and -2.7 bps at 15** (day-clustered SE 3.0 and 4.6; 90% CIs [-5.1, +4.8] and [-9.9, +5.8]).
- **Statistically:** that is indistinguishable from zero, and 3.2 and 2.6 SE below the 9 bps Hyperliquid taker fee, the cheapest taker cost on the estate.
- **Robustness:** it holds at cluster gaps of 10 s and 60 s and under every mask variant.
- **This oracle is not hypothetical.** Bybit publishes these labels live, so it is what a trader subscribed to `allLiquidation` would actually have.

The only positive reversion in the study needs look-ahead: entering at the exact last forced fill,
which nobody can know is the last until the silence that defines it has passed.
- **The ceiling** is +6.9 bps at 5 minutes and +5.0 at 15, pooled (BTC +4.9 / +7.1, ETH +10.9 / +0.7).
- **Part of it is manufactured by the conditioning:** a cluster ends because price stopped running against the liquidated side.
- **It still only reaches the cost floor:** within one SE of the 9 bps fee-only taker tier, and 4.3 SE below Binance spot taker.

What cascades do show is a large move, starting before the first forced fill, that does not come
back. **This is a decisive negative, not "the detector needs work".** Detection quality is not the
binding constraint, because perfect detection buys a move that does not cover costs.

1. **Does the Cycle 61 detector work? No.** It never fires on the cascades that matter: 0 of 8
   clusters >= $10M, 0 of the $144M and $33M manual cascades, and 0 of 174 clusters with 100 or
   more liquidations. It fails on its imbalance condition, because real cascades are two-sided on
   spot. Its loose setting does land near small liquidation clusters 4x more often than chance,
   but four of five of its events have no cluster nearby. Chat's pre-stated expectation, "volume
   bursts, not liquidation cascades", is confirmed. That closes the open "cannot be determined"
   in the atlas.
2. **Do real cascades show exploitable reversion? No.** The 0-of-8 top-bucket recall and the
   0.92x book depth are corroborating, not load-bearing. The oracle result above stands on its own.
3. **Is the signature study worth a cycle? No, closed.** Feature counts outnumber the >= $10M
   sample by 97.5x. More fundamentally, the economics are dead for any detector.

**Where this sits in the program:**
- **The same place as Exp 1, Cycle 60 and Chan:** a real effect whose tradeable part is below cost. Here it is so far below that the realizable gross is zero. This is the TC-bound shape the Cycle 62B brief names as **branch B** (`BRIEF_cycle62b_d1_mechanism_prereg.md:111`).
- **Four independent instances,** across different mechanisms, venues and data, make this the most robust finding the program has.
- **It is branch B, not branch A.** Cascades show real pressure: a 39-80 bps move, and the hit side of the book thinned to about half its control level at onset. So this is not evidence that forced flow produces no pressure.
- **Why 62B is worth running:** liquidation cascades are endogenous (price moves trigger them), so this cycle cannot separate pressure caused by forced flow from the move that caused the forcing. Cycle 62B's scheduled-rebalance design can separate them, which makes it worth running rather than redundant.

---

## Open items -- recorded, NOT fixed this cycle

**1. Collector defect: the open-gap pointer is overwritten on every retry**
(`engines/bybit_liquidation_collector.py:410-424`).
- **What happens:** every failed reconnect inserts a new open `collector_gaps` row and overwrites `stats["open_gap_start"]`. The eventual successful connect closes only the last row (`:265-270`), and the earlier rows of the chain stay open forever.
- **Measured here:** 298 rows collapse to 266 outages, with 32 permanently open rows. Read literally, each one claims "unobserved from here to now".
- **Workaround in this cycle:** `label_inventory.merge_gap_chains`. The fix belongs in the collector: open a gap only when none is open, and close that one.

**2. Collector defect: 85 hourly runs never started.**
- **Scale:** 1,358 hourly slots, 1,274 run starts, 85 slots with no start, in 19 outages (`t1_missing_run_outages.csv`). The task is registered with `-StartWhenAvailable` (`services/register_bybit_liquidation_task.ps1:44`), a 65-minute `ExecutionTimeLimit` (`:45`), `-MultipleInstances IgnoreNew` (`:46`) and a 1-hour repetition (`:36`). Every run is a fixed `--duration 3550`.
- **Mechanism:**
  - **74 slot-hours (8 outages):** each follows a run that never printed an exit line (`killed_or_running`), and lasts 1 to 28 h. A hung instance makes `IgnoreNew` drop every trigger while it lives.
  - **9 single-slot outages:** a run was still holding the task at the next trigger. Eight were catch-up starts at odd minutes (e.g. 13:35:59, 16:38:25, 20:40:31) that the fixed 3550 s carried past the next :00. One run overran to two hours (2026-08-28, 06:00-07:59).
  - **2 single-slot outages are unexplained by the log:** 2026-10-06 18:00 and 2026-10-08 00:00. The previous runs ended normally at :59:12 and :59:15.
- **Fix direction:** end each run at the next :59:10 rather than after a fixed 3550 s, and find out why runs hang without exiting.

**3. `ping_interval=None` (`engines/bybit_liquidation_collector.py:260`) -- assessed.**
- **What it disables:** the websockets library's RFC 6455 keepalive (by default a 20 s ping with a 20 s pong timeout).
- **What detects a dead peer instead:** only TCP retransmission timeout on the application-level `{"op":"ping"}` sent every 20 s (`:285-291`). That pinger swallows its own send failure and exits silently (`:290-291`), and pongs are never checked (`:322-329`).
- **Effect in this dataset: bounded, and already priced into the mask.**
  - The time since the last row before a logged disconnect is p25/p50/p75 83/275/636 s, against 59/205/546 s at random observed instants. That is consistent with tens of seconds of detection latency, not minutes of silent death.
  - 941 rows land inside the 60 s pre-disconnect pad, so rows usually kept arriving until shortly before the logged disconnect.
  - The pad (298 disconnects) absorbs the latency, and the `disconnect pad 0` sensitivity run moves nothing.
- **Residual risk:** a stall that never errors, where the server stops pushing while the socket and pongs stay alive. Neither the library keepalive nor the application ping would catch it.
- **What would catch it:** a real-time watchdog on the kline control topic, which is currently only counted at the end of a run. That watchdog matters more than the keepalive flag. It belongs in the same brief as items 1 and 2.

**4. Order-book collector blind spot.** About 52 s at :34-:35 in 1,252 of 1,358 hours, plus 108.2 h
of gaps over 30 s in the label window (the longest 28.6 h). It is the same fixed-duration pattern
as item 2, on the book collector's own :35 schedule. Its depth and spread values read NaN in
outages, so it costs coverage, not bias.

**5. The local mirror has failed every night since 2026-09-04. Open item for Chat.**
- **Evidence:**
  - `McTheoryMirror` exited 0x1 at 03:00:01 today. The script maps robocopy >= 8 to exit 1 (`C:\Data\Backups\scripts\backup_mctheory.bat:68-86`).
  - There is no `mirror_*.log` in `C:\Data\Backups\logs` within the 30-day retention.
  - The newest files in `C:\Backups\McTheory_mirror\praxis\logs` are from **2026-09-04 09:05**, so the last successful copy was that morning's run.
  - The script was last modified at 2026-09-04 19:17; its predecessor `.pre20260904.bak` dates from 2026-08-05.
- **Hypothesis:**
  - The 09-04 edit replaced `%DATE%` slicing with `wmic os get LocalDateTime` (`backup_mctheory.bat:39-44`).
  - WMIC is not on this machine. `where wmic` finds nothing and there is no `C:\Windows\System32\wbem\WMIC.exe`; on Windows 11 24H2+ it is a deprecated Feature on Demand, and this is build 26200.
  - So `LDT` is never set, `TIMESTAMP` expands to garbage, `/LOG+:"%LOGFILE%"` is malformed, and robocopy rejects its parameters before copying anything or writing a log.
  - That matches every symptom.
- **What would confirm it:**
  1. Run lines 42-44 alone in `cmd` and echo `%LOGFILE%`.
  2. Run `robocopy /L` with that `/LOG+` value; expect "Invalid Parameter" and exit 16.
  3. Date the first 0x1 to 2026-09-05 03:00 in the TaskScheduler history.
  4. Run `DISM /Online /Get-Capabilities | findstr /i wmic`.
- **A correction to the framing that accompanied brief v2: the mirror's failure does not touch `crypto_data.db`.** The mirror excludes `*.db`, `*.db-wal` and `*.db-shm` by design (`backup_mctheory.bat:19-25` and `:64`, the same 09-04 change), so it has not been one of the database's copies since. The database's backup path is item 6, and that is where the labels are exposed.

**6. Large-database backup race: `crypto_data.db` has not been backed up since 2026-09-27. Open item for Chat; time-sensitive.** This was found while checking item 5.
- **How the path works:** on full days (Sundays), `backup_to_dropbox.ps1` (03:30) takes `VACUUM INTO` snapshots of the large databases, and `backup_large_dbs.ps1` (03:45) writes whatever is staged to restic. The Sunday 7-Zip archive does not carry it: it scans 3,334 MiB in total.
- **The race:**

  | Sunday | `crypto_data.db` snapshot finished | restic started | backed up? |
  |---|---|---|---|
  | 09-27 | 03:44:39 | 03:45:03 | yes, with 24 s to spare; only 2 of 3 copies (peer MCRMINI2 silent: "NOT mirroring") |
  | 10-04 | 03:50:52 | 03:45:05 | no |
  | 10-11 | 03:54:23 | 03:45:06 | no |

- **Silent on full days:** both misses logged "Nothing over 100MB staged. On a delta day the large databases are deferred, so this is expected." That message was printed on a full day, so a missed staging is indistinguishable from a deferral.
- **It will miss again:** the snapshot grows about 1.3 GB a week, so **Sunday 2026-10-18 will miss too.**
- **Exposure:** **50,252 of the 191,802 Bybit label rows (26%, 32% of the span) exist only in the live database.** The one other copy is this cycle's input snapshot (`data/external/forced_trade/c66/liquidations_bybit.parquet`, rows through 2026-10-15 06:06 UTC). It is gitignored and on the same drive.
- **Why it matters:** these labels cannot be re-fetched; no backfill exists on any path (Cycle 62A).

---

## Process notes

- **The interruption was a Windows Update restart.**
  - What happened: event 1074 from `MoUsoCoreWorker.exe` at 02:59:15 EDT, a clean shutdown at 02:59:33, and a boot at 02:59:40. It was neither a PyCharm crash nor a host crash; there was no 6008 or 41 overnight.
  - What survived: every computation had finished by 02:57:16. Nothing on disk was truncated: 16 modules parsed and 17 JSON, 104 CSV and 6 parquet files read in full. The cache passed coverage checks and the regression re-run (6/6).
  - What was at risk: the sensitivity runs' printed reports lived only in a Temp scratchpad. They were copied into their output folders first thing and hash-verified.
- **Recovery re-run:** the baseline re-run with the pre-crash code reproduced 23 output files byte for byte. It also added one diagnostic column, `KM_same`, which is never scored.
- **This session's additions, all additive:**
  - the two residual tests;
  - `--band-soh`;
  - the label table with qty x spot;
  - the T3 SD/SE columns and pooled BTC+ETH rows;
  - four tests, bringing the suite to 17, all passing.
- **Additivity check:** every pre-existing output is byte-identical, and `t3_reversion.csv` is a strict superset.
- **A mistake of mine, caught by the output, not the exit code.**
  - What happened: the first final run crashed in T3 because a new local named `pool` shadowed the control pool.
  - Why the exit code missed it: the wrapper reported exit 0 because its last command was `date`; only the traceback at the end of the run report showed it.
  - The fix: rename the local and propagate Python's own exit code. The clean re-run is the one committed.
- **The 03:00 and 03:30 windows were kept clear.** The only database reads (snapshot, cache build, contiguity) ran at 02:20-02:31 and 14:34 EDT.

## Files

- **New:** `engines/forced_trade/labels.py`, `label_inventory.py`, `label_scoring.py`, `cascade_profile.py`, `signature_sizing.py`, `run_labelled.py`, and `tests/test_forced_trade_labels.py` (17 tests).
- **Modified:** `engines/forced_trade/__init__.py` (module list). The Cycle 61 modules are imported, never edited.
- **Outputs:** `outputs/forced_trade/c66/` holds T1-T4, `contiguity_trade_ids.csv`, `run_stdout.txt` and `contiguity_stdout.txt`. `sensitivity/` holds five variants, each with its own `run_stdout.txt`.
- **Inputs (gitignored):** `data/external/forced_trade/c66/` holds the snapshot, the 10 s trade buckets, the book, the log snapshot, the manifest, and row-level tables under `rows/`.

Reproduce, read-only and from the snapshot unless noted:

```
.venv\Scripts\python -m engines.forced_trade.run_labelled extract        # DB, mode=ro (done once)
.venv\Scripts\python -m engines.forced_trade.run_labelled --tooling-results outputs\forced_trade\c66\t4_tooling_results.json all
.venv\Scripts\python -m engines.forced_trade.run_labelled contiguity     # DB, mode=ro
.venv\Scripts\python -m engines.forced_trade.run_labelled --band-soh 3540 30 --out-dir outputs\forced_trade\c66\sensitivity\band_bound_5900_0030 all
.venv\Scripts\python -m pytest tests\test_forced_trade_labels.py -q
```
