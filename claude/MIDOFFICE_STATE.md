# Mid-office state

**Updated:** 2026-11-10, at the end of midoffice07 (stopped at its gate). Brought up to date at the
end of every piece, in the same commit as its retro (Rule 43). Not a history: the state a new
session needs to carry on. The detail is in the retros named below.

---

## Start here

1. Read `claude/CLAUDE_CODE_RULES.md` (Rule 43 above all), then this file, then the latest retro:
   `claude/retros/RETRO_midoffice07_rebuild.md`.
2. **How the work runs.**
   - **The brief.** Each piece, `midofficeNN`, is a brief written in Chat and pasted into the Code
     session. Code transcribes it verbatim to `claude/handoffs/BRIEF_midofficeNN_<slug>.md`; its
     Source line says it was written in Chat and pasted in.
   - **The retro.** The piece ends with a full retro, `claude/retros/RETRO_midofficeNN_<slug>.md`,
     whose last item is the exact `git add`. The terminal gets a short summary and the retro's
     path, then Code stops.
   - **The commit.** Jeff commits with Chat's message; Code never commits or pushes. The next
     brief opens with "committed" (check it in `git log`) and Jeff's rulings on the decisions the
     last retro left.
3. **The discipline every brief restates.**
   - Changes go in the order given, each decomposed against the one before.
   - A gate means stop and report before building anything.
   - Nothing in `data/` is overwritten or deleted without Jeff.
   - Scratch goes to `claude/scratch/midofficeNN/` (gitignored), its outputs copied into the retro.
   - Python is ASCII and keeps the file's line endings (most `.py` here are CRLF; the `claude/`
     docs LF).
   - The suite runs as `python -m pytest tests --ignore=tests/test_market_data.py -q`: 2,279 passed
     and 1 skipped at midoffice07. That file looks for `../core_repo/src`, which isn't on this
     machine.

---

## Where the work stands

**The model's foundations** (generic, before the S&P 500 data):

- cb19460 (2026-11-01): the time-series (Fama-French) factor-model builder.
- 232b582: covariance estimators -- exponential weighting, Ledoit-Wolf shrinkage.
- 449ac95 (11-02): the cross-sectional (Barra) estimator and split-horizon covariance.
- 4fef869: the specific risk forecast from characteristics (Barra EQ 3-12 to 3-14).
- 7b160b5: the specific risk model wired into the cross-sectional estimator.
- 6dc0a99: Newey-West, Bartlett weights.
- 8a271d1: USE3's descriptors, the risk indices, the end-to-end pipeline.

Under them, not the model's: the risk core (98e7fdd to 260759b, 2026-10-21/22) and the value
layer (c92d343 to e859b4e, 2026-11-01).

**The S&P 500 model:**

- dc94fa6 (2026-10-16): point-in-time membership from Wikipedia's tables (`refdata/sp500.py`).
- 666599b (11-04): real data -- the Polygon loader, the 506-ticker daily panel, the model run.
- b9071d3: unit-basis exposures for readability, the covariance moved to match.
- c1275ab: industry members counted over the regression universe; 7cfe835, the SIC date
  fallback it needed.
- 17a588d: specific risk estimated over the regression universe; fallback forecasts labelled.
- f72310b (11-05): each period's regression held to that day's index members.
- 8cdd037: **midoffice01** -- exposures standardised over the estimation universe.
- f0ae2a9: Rule 43. 6e29b2c: its correction, the briefs written in Chat.
- 956331e (11-06): **midoffice02** -- each regression reads its inputs from t-1; an empty window
  gives no value.
- 08087bf (11-09): midoffice01's brief provenance corrected.
- 992f11a: **midoffice03** -- min_coverage 0.5; SERDP in no blend; the blend guard.
- 8c76c59: **midoffice04** -- USE4's market model; one blend per factor; USE4's Size
  Nonlinearity; the residual spread by the MAD.
- 1b9d43b: **midoffice05**, first change -- the size pair's standing line;
  `--size-nonlinearity none`.
- 701794e: **midoffice05** -- the identity table: members as securities, not tickers.
- 208ca80 (11-10): **midoffice06** -- the rebuild's fetch, the split classification, the
  distributions; stopped at its gate.
- Next: **midoffice07** -- the identity corrections, the lag rule and the price-gap test; stopped
  at its gate. Jeff commits it after 208ca80, and this file after it (`git log --oneline -3`).

**The current piece: midoffice07, STOPPED AT THE GATE (2026-11-10).**

- **Item 1 is done.** The corrections were made in 21 requests, and the plan regenerated: 600
  securities, 628 spans, 617 tickers.
- **The gate:**
  - **Identity:** 627 of 628 spans pass.
  - **Splits:** 55 events and no contradiction -- 47 splits, every one gapping by its ratio, and 8
    price-only.
  - **COHR fails.** Its record of 2022-09-08 names BBG000GS6QB4, which is not old Coherent's FIGI
    (BBG000BG1DH3). Its share-class FIGI is Coherent Corp.'s.
- **Waiting on the gate:** items 2 to 5 -- the panel, total returns, the model comparison, the
  docs.
- **The model still runs on the old panel:** 506 tickers, price returns, split-adjusted volumes,
  and share counts sampled yearly.

---

## What is settled -- do not reopen

**The retros.** "NN" is `claude/retros/RETRO_midofficeNN_*.md`; "NN ruling k" is Jeff's ruling k
at the top of the brief for the piece after NN. Pre-series commits have no retro; their detail is
in `docs/FACTOR_MODEL.md`.

**Membership and the estimation universe**

- The estimation universe on day t is that day's members passing the data test. It alone sets
  every scale: clip bounds, cap-weighted mean, equal-weighted sd. Outsiders get exposures on the
  members' scale. One membership array goes to every consumer. (01)
- Each period's regression holds to that day's members. Industry counts and specific risk are
  taken over the regression universe; a name in no regression gets a fallback forecast, labelled.
  (f72310b, c1275ab, 17a588d)
- membership():
  - reads the constituent list brought up to the change table (current_stale);
  - with dated renames from the identity table, reads the change table in today's tickers --
    ECHO's phantom days go. (05, 07)
- The rebuild reads membership through the renames and maps it to securities; the run prints
  current_stale. (05 ruling 3; not yet built)

**Time and coverage**

- **The lag.** Period t's regression reads its exposures, their standardisation, its weights and
  its membership from t-1, and only its returns from t; the forecast exposures are the last
  date's. The estimator's lag defaults to 1, and test_lookahead's repricing guard fails at 0.
  (02)
- **Coverage.** A window with no valid observation gives no value. min_coverage is 0.5: a window
  descriptor needs half its window valid; 0 stays runnable. (02, 03)
- **Share counts** are carried forward from the last sample, never back-filled. (02)
- **fit_index_weights** is lagged and whole-sample, and the run does not call it. (03)

**Definitions: USE3's by default, USE4's where USE3's choice measures badly, each switchable (04)**

- **The market model** (BTSG, HALPHA, SERDP, VLVR's residual sd): on a daily panel USE4's, 252
  trading days at half-life 63; on other panels USE3's 60 months. `--beta-half-life` takes inf
  for equal weights. (04; 04 ruling 3)
- **Size Nonlinearity**, USE4's: Size cubed, standardised and clipped at +/-4, orthogonalised
  against Size by the regression weights over the members, standardised again. It stays in the
  model, with "use3" and "none" runnable; a guard stops the run unless it is orthogonal. (04; 04
  rulings 1-2)
- **One definition per factor:** a period is estimated only if every index's blend has exactly
  the forecast's descriptors. It is asserted; 938 periods from 2023-02-08; `--mixed-blends`
  overrides. (04)
- **SERDP** is in no blend. A descriptor in the forecast and in no estimated period stops the run.
  (03)
- **RSTR** is USE3's twelve months, skip 0; `--rstr-skip` cuts the academic 12-1. USE4's lag was
  measured and not built. (03, 04)
- **EQ 3-14's residual spread** is 1.4826 x the MAD; `--residual-spread sd` stays runnable. (04;
  04 ruling 4)
- **Weights and industries.** Regression weights are sqrt(cap), and the standardisation's mean
  is cap-weighted. Industries come from SIC, groups under 15 merged into their division, with no
  market factor beside them. Newey-West takes 10 lags at half-life 90. (666599b; docs)
- **Noise shares** -- Size Nonlinearity 41.8%, Construction 52%, Wholesale 40% -- get one
  model-wide answer after the rebuild, never a deletion per factor. (04 ruling 1)

**The identity table (05, 06, 07)**

- **Keys.** A security is a share class, keyed by composite FIGI, else CIK; the CIK is the
  cross-check. A record with neither is no evidence. (05; 06 ruling 1)
- **Sources.** The table is built from the panel fetch's own records (its 252-day grid), Polygon's
  ticker events and the panel's prices. The rebuild's share samples check it and never build it.
  (05, 07)
- **The endpoint.** Member-days neither source names go to the reference endpoint on public-record
  hypotheses, and only what it confirms enters. WBD is its own security. (05 ruling 2; 06; 06
  ruling 3)
- **Splices and mergers.** A splice is two securities. A merger's predecessor (WestRock,
  Paramount) is another security, linked for membership. (05, 06)
- **A reorganisation** (a new identifier, the same ticker, the prices unbroken) is one security
  with one return series. It is keyed by the successor, the predecessor linked, and its boundary
  is dated by the key's identifier: Bunge by CIK, Vylor by FIGI. (05 ruling 1; 06 ruling 1)
- **Spans:**
  - a span starts at its first price under its ticker;
  - a span with no price goes, and so does one its records refute: S600, CEGV, CRHC;
  - CRH's ADR history before 2023-09-25 stays out;
  - II-VI's IIVI history, 2022-01-03 to 2022-09-07, is Coherent Corp.'s. (06 ruling 1; 07)
- **The identity check** reads each span's first sample with an identifier. It reads past a
  change-day record naming the ticker's previous holder (code since 07). (05 ruling 5; 06, 07)
- **The fetch** covers the whole window for every member security. A span-plus-look-back cut is
  to be a parameter for future panels; it is not built. (05 ruling 4)

**Share counts and splits (06, 07)**

- **Sampling.** Counts are sampled inside each security's own spans: each span's first and last
  day, and a 63-day grid kept in its phase.
- **Reorganisations' counts.** A successor carries its predecessor's count until its own first
  sample, never back-filled: Carnival's 1.1054. OKE's and XOM's carry 1:1, flagged. (06 ruling 4)
- **Split events** are classified by the ratio -- "whole" means no term over 100 in lowest terms
  -- and two tests within 10%: the counts either side, and the raw opening gap on the execution
  date against 1/ratio. (06 and its two amendments; 06 ruling 2)
  - **A split** is whole, and the counts follow it (or, where the counts cannot speak, the price
    gaps by it). It is applied to the carried count on its execution date.
  - **Price-only** is fractional with the counts unchanged: a spin-off, carried by the adjusted
    prices.
  - **A contradiction** -- a test disagreeing, or a split whose price does not gap -- stops the
    gate.
- **The lag rule.** A sample on or after a split still at the pre-split count is stale and does
  not reset the carried count; the first that follows the ratio does. (06 ruling 2)
- **The split guard** (in the build): no member-day pairs a post-split price with a pre-split
  count, exactly. (asked in midoffice03's and midoffice06's briefs)

**Prices and returns**

- **Price returns so far.** Polygon's adjusted bars are split-adjusted only, so every run so far
  is on price returns. The market series is SPY. (06)
- **Total returns:**
  - every cash distribution, specials included, put on the prices' split basis by the ex-date's
    adjusted over raw close;
  - TEL's duplicate of 2024-11-22 kept once; SPY's distributions too;
  - in the comparison, identity first, then dividends. (06's second amendment; 06 ruling 5;
    07's brief)
- **LEN's 2025-01-21 adjustment** is left as Polygon has it, flagged. (06 ruling 6)

---

## The data

All under `data/`, which is gitignored. Nothing there is overwritten or deleted without Jeff.

- **The old panel:** `data/external/polygon/sp500_daily_2022-01-03_2026-11-04.npz`.
  - 506 tickers -- the index on 2022-01-03 -- over 1,214 days to 2026-11-03, share counts
    sampled every 252 days.
  - Regenerate: `python scripts/fetch_sp500_panel.py --start 2022-01-03 --end 2026-11-04
    --shares-every 252`. From the cache it reproduces every array exactly (checked 2026-11-10).
- **The membership snapshot:** `data/external/sp500/sp500_20261104T033741Z_660dc8159b0a.json`.
  - Runs use the newest snapshot in the directory.
  - Validate: `python -m praxis.midoffice.refdata.sp500 check --dir data/external/sp500`.
  - `... fetch --out data/external/sp500` writes a new one, which every run would then use: only
    to roll membership forward.
- **The Polygon cache:** `data/external/polygon/cache/`, about 15,250 answers, found or not:
  `bars-`, `ref-`, `events-`, `splits-`, `dividends-`.
  - Every script reads it first, so reruns ask nothing.
  - It holds the evidence; it can only be rebuilt by asking again.
- **The identity table and fetch plan, current:**
  - `data/external/polygon/sp500_identity_2022-01-03_2026-11-03_corrected*`: the table `.csv`,
    `.json`, `_members.csv`, `_fetch.csv` (the plan), `_links.csv` (reorganisations and mergers),
    `_membership.csv`.
  - Regenerate: `python scripts/build_sp500_identity.py
    data/external/polygon/sp500_daily_2022-01-03_2026-11-04.npz --resolve --correct --offline`.
    Without `--correct` it writes midoffice06's `_resolved` files; without either,
    midoffice05's.
- **The rebuild's fetch and gate:** `python scripts/fetch_sp500_rebuild.py --offline`.
  - The default plan is the corrected one; without `--offline` it fetches what is missing.
  - Report: `data/external/polygon/sp500_rebuild_gate_2022-01-03_2026-11-03_corrected.json`.
    midoffice06's is the unsuffixed one.
- **The new panel:** not built.

---

## The run

```
python scripts/run_sp500_model.py data/external/polygon/sp500_daily_2022-01-03_2026-11-04.npz --industries --membership
```

**At today's defaults:**

- **Time:** about 90 s. `--industries` reads SIC codes from the cache but needs `.env`'s
  `POLYGON_API_KEY`.
- **Headline:** mean R-squared 0.298076 over 938 periods from 2023-02-08.
- **The size pair:** 11.94% for the smallest 85, 0.38% for the largest 50, 0.01% for the index.

**Its standing lines, and what each is there to catch:**

- **The snapshot.** Its hash and coverage, and the members in the panel on the first and last
  days. A change membership() does not reproduce is the change table disagreeing with itself. A
  date the snapshot cannot answer stops the run.
- **The market model's window and half-life:** which beta the model is using.
- **Coverage:** values from an empty window, or from under min_coverage, counted against each
  descriptor's own inputs independently. It must read "none".
- **Outsiders at the members' bounds:** the estimation universe setting the scale.
- **Each index's blend over the estimated periods.** Guard: it stops if a descriptor feeds the
  forecast and no estimated period, or if an estimated period's blend is not the forecast's.
- **Style exposures over the members on every date read:** cap-weighted mean zero and sd one, to
  rounding. The index at zero exposure; `!` if not.
- **Size Nonlinearity's regression-weighted covariance with Size.** Guard: it stops past
  rounding.
- **The size pair's portfolio line** -- smallest 85, largest 50, the index: where size risk sits.
- **Same-day membership:** the security-periods that move against it. The lag in effect.
- **The industry universe against the regression's:** counts taken over the wrong set; `!` if
  they differ.
- **Securities per estimated period** -- members, in the panel, in the regression: what keeps
  members out.
- **The factor table's noise share, and its exactly-1 and none columns:** thin and one-member
  factors.
- **Specific risk:**
  - the percentiles;
  - the shares of V-squared;
  - the residual spread, both MAD and sd, and its shares;
  - the characteristics' weight in a blended forecast.
  Together: who drives the specific risk model.
- **Qualifications:** dropped periods, excluded securities, fallback forecasts.

**The other guards:**

- `build_sp500_identity.py` exits non-zero on a failed guard:
  - each priced ticker-date belongs to one security;
  - no security trades under two tickers on one date;
  - no ticker names two securities on one date;
  - every member-day is mapped or named.
- `fetch_sp500_rebuild.py` exits non-zero when its gate stops.

---

## What is next

1. **Jeff's rulings on midoffice07** (its retro, "Decisions left for Jeff"):
   - **COHR's change-day record:**
     - **(a)** a record whose composite FIGI the table does not expect, but whose share-class FIGI
       is the expected security's, names the expected security -- it accounts for all seven such
       records;
     - **(b)** or a one-off, reading COHR at its second sample.
     - Either way, add Coherent, Inc.'s row under COHR: BBG000BG1DH3, 2022-01-03 to 2022-07-01.
   - **Share counts the records do not carry** (696 member-days, spin-offs' first quarters, all of
     FDXF's and Vylor's):
     - leave them out, the default;
     - or search for each first count, about 70 requests;
     - or take weighted counts where the share-class count is missing.
   - **Volumes:** build the new panel on traded volumes? The old panel's are split-adjusted:
     24,386 member-day cells off by 0.2 to 50. The basis would be measured as its own step.
2. **The gate again**, offline. Nothing but COHR stops it now.
3. **midoffice07's items 2 to 5:**
   - **the new panel** -- securities as columns; prices stitched by the table, splices separated,
     reorganisations continuous; counts per span with splits on their execution dates;
     membership through the renames; current_stale printed; the identity and split guards on the
     built panel;
   - **total returns;**
   - **the model in two steps** -- old panel against new on price returns, then the new panel's
     price returns against its total returns, the factor and industry means especially;
   - **the docs** -- `docs/FACTOR_MODEL.md`'s split paragraph (lines 777-786) rewritten for the new
     rule, with the old panel's 4,196 member-days over 37 splits, and a section on the identity
     table and the rebuild.
4. **After the rebuild:** residuals for every covered name, not only regression members; then the
   model-wide noise-share question.

**Open, not queued:**

- fit_index_weights is whole-sample, though unused;
- VLVR runs the market model a second time;
- the span-plus-look-back cut;
- `tests/test_market_data.py`'s missing `../core_repo/src`.
