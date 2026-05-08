# Cycle 33 -- Atlas schema extension + COMPUTATIONAL_ENGINES.md + 2 example backfills

**Predecessor:** Cycle 32 (`2c98a22`, atlas hygiene closeout).
**Mode:** Hybrid. Files-included delta zip with the new doc and
the migration script; Code applies markdown edits, parser
updates, runs the migration, runs atlas_sync.
**Risk:** medium. New DB columns are additive; parser changes
must not regress the 15 existing experiment rows; new markdown
sections must be parseable cleanly. Backfill is intentionally
limited to 2 experiments in this cycle; mass backfill is Cycle
35.

## Why now

The Atlas DB shipped in Cycle 12 captures `signal_type`,
`asset_class`, `framework`, `result_class`, `result_summary`,
`key_findings`, and `atlas_principle` per experiment. That's
enough for semantic similarity search but not enough for
navigating revival hypotheses ("under what conditions might
this dead experiment become alive?"), engine kinship ("show me
all Engine 1 experiments"), or regime sensitivity ("what
regimes were active during this test?").

The user's reframing at end-of-Cycle-31 made the gap concrete:
Rule 35 is a contract not a guideline; the Atlas similarly
should be contractually structured rather than free-text. This
cycle adds the structure.

The cycle also writes `docs/COMPUTATIONAL_ENGINES.md` --
documenting the 7-engine taxonomy that has lived only in chat
archive (praxis_main_1_cpo_atlas, March 24 2026) and partly in
code (`engines/base.py` ABC hierarchy, `engines/context/`,
`engines/adapters/`, `engines/model.py`) without a primary
written reference. The doc captures the framework, lists the 7
engines with their math + strategies, and provides the
mapping from atlas experiments to engines.

## What

Five deliverables:

1. **DB schema migration**: 4 new columns on
   `atlas_experiments` (`test_conditions`,
   `revival_hypotheses`, `regime_state_at_test`,
   `computational_engine`). Migration script provided in the
   delta zip.
2. **`docs/COMPUTATIONAL_ENGINES.md`**: full content provided
   in the delta zip; Code copies it into `docs/`.
3. **Markdown template design**: three new structured sections
   per experiment in `TRADING_ATLAS.md`:
   `**Test conditions:**`, `**Active regimes during test:**`,
   `**Revival hypotheses:**`. Plus a `Computational engine`
   row added to each experiment's existing attribute table.
4. **`engines/atlas_sync.py` parser update**: learn to extract
   the three new structured sections + the
   `Computational engine` attribute. Populate the four new DB
   columns.
5. **Backfill 2 experiments** (Exp 1 mean-reversion equity +
   Exp 13 funding carry) as the design validation. The
   remaining 13 backfills are Cycle 35.

No live collector touched. No production query path affected
(MCP tools `atlas_search` and `atlas_get` automatically expose
the new columns since they return full records).

## Out of scope

- Backfilling the other 13 experiments (Cycle 35).
- Changing `PREDICTION_MARKET_ATLAS.md` structure (it's a
  different beast -- categories of strategies rather than
  specific experiments with results).
- Running the regime engine retroactively over historical
  test windows to populate `regime_state_at_test` rigorously.
  For now, that field is best-effort: NULL or
  `"not_measured"` for most experiments; populated only when
  the experiment naturally documented regime state (e.g., Exp
  13 funding carry has rich regime context because the
  strategy *trades on* regime F).
- Changes to `regime_classes` / `strategy_regime_relevance`
  tables (unaffected).
- Changes to embeddings (unaffected -- the new columns are
  not part of the embedding text; embeddings stay derived
  from `full_markdown`).

## Specifics for Code

### Step 1: Apply schema migration

```powershell
cd C:\Data\Development\Python\McTheoryApps\praxis
python scripts\migrations\cycle33_atlas_schema_extension.py
```

The script is idempotent (detects already-applied state) and
uses `BEGIN`/`COMMIT` for atomicity. Expected output: 4 columns
added, 35 existing rows unchanged (all NULL in new columns).

### Step 2: Drop in `docs/COMPUTATIONAL_ENGINES.md`

The delta zip contains the file ready to copy. Just place it
at `docs/COMPUTATIONAL_ENGINES.md` and commit.

### Step 3: Update `engines/atlas_sync.py` parser

The parser already handles attribute tables and section
extraction. It needs three additions:

**3a. Extract `Computational engine` from the attribute table.**

Current attribute table parsing reads rows like:

```
| **Date** | 2026-03-22 |
| **Framework** | ... |
```

Extend to also capture:

```
| **Computational engine** | 1 (Cointegration) -- secondary: 3 (Allocation) |
```

Parse the integer at the start of the cell. If multi-engine
("1 + 7"), keep the FIRST integer in the
`computational_engine` column (it's the primary engine);
acknowledge the secondary in `test_conditions` JSON if the
prose mentions it. NULL if absent.

**3b. Extract the three new structured sections.**

Each section is delimited by a bold header line. Find the
header, then capture content until the next bold header (`**`)
or section divider (`---`). Parse content into a JSON-shaped
Python dict / list, then `json.dumps()` for storage.

**`**Test conditions:**`** -- table or bullet list parsed into
a flat dict. Key is the leftmost cell (lowercased,
spaces-to-underscores). Value is the rightmost cell, kept as
free text. Example:

```markdown
**Test conditions:**
| Aspect | Value |
|---|---|
| Bar type | hourly time bars |
| Frequency | 1h |
| TC | 4 bps round-trip |
| Pre-filter | yes (drops worst 3 of 8 TA types) |
| Risk management | equal weight, P>0.50 gate, 5% per-model cap |
| Feature set | 8 TA types per asset |
| Computational engine | Engine 2 (Momentum/Trend) |
```

becomes (after `json.dumps()`):

```json
{"bar_type": "hourly time bars",
 "frequency": "1h",
 "tc": "4 bps round-trip",
 "pre_filter": "yes (drops worst 3 of 8 TA types)",
 "risk_management": "equal weight, P>0.50 gate, 5% per-model cap",
 "feature_set": "8 TA types per asset",
 "computational_engine": "Engine 2 (Momentum/Trend)"}
```

**`**Active regimes during test:**`** -- bullet list parsed
into a flat dict. Each bullet must start with a regime class
letter or class name. Example:

```markdown
**Active regimes during test:**
- A (Trend): not measured
- D (Serial correlation): not measured
- F (Funding): regime-relevant; ablation lift +12.4% AUC; states present during 2024 included +1, +2 (positive funding sustained)
```

becomes:

```json
{"A_trend": "not measured",
 "D_serial_correlation": "not measured",
 "F_funding": "regime-relevant; ablation lift +12.4% AUC; ..."}
```

If the experiment has no `**Active regimes during test:**`
section at all, store `null` (not an empty dict). That signals
"the experiment didn't document this" vs. "documented but
nothing relevant."

**`**Revival hypotheses:**`** -- numbered list parsed into a
JSON list of dicts. Each numbered item is a hypothesis. Bold
phrase is the title; "likelihood: low/medium/high" is the
likelihood; rest is description. Example:

```markdown
**Revival hypotheses:**
1. **Switch to dollar bars** -- likelihood: medium. Each bar represents equal economic activity; eliminates "Sunday 3am bar = no signal". Test cost: rebuild data pipeline (Cycle 34 deliverable). Predicted lift: AUC +2-5%, calibration monotonicity strengthens.
2. **Add regime D filter** -- likelihood: medium. Only run TA when serial-correlation regime D is in trending state. Test cost: small (regime detection built post-Cycle-12). Predicted lift: filters worst regimes contributing to negative average alpha.
```

becomes:

```json
[
  {"title": "Switch to dollar bars",
   "likelihood": "medium",
   "description": "Each bar represents equal economic activity; eliminates ..."},
  {"title": "Add regime D filter",
   "likelihood": "medium",
   "description": "Only run TA when serial-correlation regime D ..."}
]
```

**3c. Markdown hash sensitivity.**

The new sections become part of `full_markdown` (since the
parser captures the full experiment body). This means
`md_hash` will change for any experiment that gains the new
sections. Re-embedding will fire for those experiments only.
That's expected and correct.

### Step 4: Update `TRADING_ATLAS.md` -- backfill 2 examples

#### Example A: Experiment 1 (MEAN_REVERSION x EQUITY_US)

Current attribute table at line 49 looks like:

```markdown
| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-22 (updated 2026-04-02) |
| **Framework** | Burgess pair discovery -> Chan CPO intraday execution |
| **Signal** | Cointegration-based spread z-score, single-asset execution |
| **Data** | Polygon.io minute bars, 38 deduplicated pairs |
| **Training** | 2025 (240 configs x 247 days x 38 pairs = 2.3M strategy-days) |
| **OOS** | 2026-01-01 -> 2026-03-20 (53 trading days, 45 with active models) |
| **TC** | 2 bps/leg x 2 legs = 4 bps round-trip |
| **RF AUC** | 0.855-0.896 (mean 0.873) -- minute-frequency features per Chan paper |
| **Result** | **NEGATIVE after TC** |
```

Add a row:

```markdown
| **Computational engine** | 1 (Cointegration/Mean-Reversion); secondary 3 (Allocation) |
```

After the existing `**Risk management lessons:**` block (around
line 100-105), before the `---` separator at line 108, INSERT
the three new sections:

```markdown

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | minute time bars |
| Frequency | per-minute, 9:30-16:00 ET equity hours |
| Universe | 38 deduplicated SP500 cointegrated pairs (Burgess discovery) |
| TC | 4 bps round-trip per pair |
| Feature set | 112 minute-frequency features (Chan paper spec; 7 indicators x 16 lookback combinations) |
| Pre-filter | none -- all pairs from Burgess discovery feed CPO directly |
| Risk management | equal weight, P>0.65 gate, notional_capital normalization, spread_history warmup |
| Computational engine | Engine 1 (Cointegration); composes with Engine 3 (Allocation) for portfolio construction |

**Active regimes during test:**

Regime ablation was run during this experiment (rare
feature -- most experiments don't have this data). Top
predictive classes for pairs MR:

- A (Trend): +7.25% AUC lift -- ADX tells you when NOT to mean-revert
- D (Serial correlation): +5.65% lift -- Hurst confirms MR vs momentum regime
- G (Liquidity): +4.75% lift -- can you execute the spread?
- I (Volume): +3.98% lift -- is there participation?
- E (Microstructure): +2.65% lift -- OFI adds moderate value
- B, C, F, H, J, K: <=0.4% -- noise for equities
- Full additive: +10.39% lift across all classes

The OOS test window (2026-01-01 to 2026-03-20) was a
medium-trend, low-vol period for SPX with abundant liquidity.

**Revival hypotheses:**

1. **Lower-TC venue or different asset class** -- likelihood: high. The signal is real (57.6% gross win rate, +5.3 bps/trade) but TC of 4 bps RT doubles the gross alpha. A market with TC < 2 bps RT (e.g., direct broker access at institutional rates, or moving to FX major pairs) could flip this NEGATIVE -> POSITIVE.
2. **Add ADX trend filter (Class A)** -- likelihood: medium. Class A produced the largest single-class lift (+7.25%). Only trade when ADX < 25 (no trend). Reduces trades, may improve net by avoiding adverse trend regimes.
3. **Switch to dollar bars** -- likelihood: low. Pairs MR's bar selection isn't the constraint here; the constraint is gross alpha vs. TC. Dollar bars wouldn't change the TC arithmetic.
4. **Re-run with intraday minute features but daily holding period** -- likelihood: low. The current setup already does this; the 2.7 trades/day count isn't the cost driver.
```

Now the parser-relevant attribute is also added to the
attribute table.

#### Example B: Experiment 13 (MICROSTRUCTURE x CRYPTO Funding Rate Carry)

This is the single confirmed POSITIVE in the Atlas. Its
revival hypotheses are a different shape -- they are about
*scaling* the win, not reviving from dead.

Locate Exp 13 (around line 606 of TRADING_ATLAS.md). Add a
`Computational engine` row to the attribute table:

```markdown
| **Computational engine** | 7 (Event/Signal); composes with Engine 3 (Allocation) |
```

After the existing closing prose (just before the next
experiment), insert:

```markdown

**Test conditions:**

| Aspect | Value |
|---|---|
| Bar type | 8h funding-rate cadence + daily OHLCV |
| Frequency | per-funding-event (8h on Binance perps) |
| Universe | BTC + ETH perps |
| TC | 4 bps round-trip per leg |
| Feature set | 11 hand-crafted features (annualized funding, percentile rank, sustained-positive flag, basis, OI change, volatility level, trend strength, etc.) |
| Pre-filter | none; gate is P > 0.70 from RF |
| Risk management | Kelly-sized within configurable max-leverage; long-only structural exposure |
| Computational engine | Engine 7 (Event/Signal -- funding rate IS the signal); composes with Engine 3 (Allocation/Kelly) |

**Active regimes during test:**

The strategy is regime F-conditioned by design (Class F:
Funding/positioning). Behavior across regime states:

- F = +1, +2 (positive funding sustained): trades nearly
  every day; this is the alpha-bearing regime
- F = 0 (flat funding): mostly does not trade
- F = -1, -2 (negative funding): does not trade (correctly --
  the structural carry is absent)
- A (Trend): not directly conditioned but observed alignment;
  positive funding tends to coincide with positive trend in
  crypto bull regimes
- B (Vol level): no strong dependence; the carry mechanism is
  vol-agnostic when long-only

The OOS validation period (2024 + 2025 H1) included two
distinct funding regimes (Q1-Q3 2024 sustained positive;
Feb-Aug 2025 flat-to-negative). Strategy traded in the first,
sat out in the second. P&L profile: monotone-positive in
favorable regimes, flat in unfavorable -- never bleeds.

**Revival hypotheses:**

For a POSITIVE experiment, "revival" reframes as
"scaling/improving":

1. **Add cross-exchange funding spread (Bybit, OKX, Hyperliquid)** -- likelihood: high. Same engine; adds breadth. Each new venue is +30-50% effective universe size. Test cost: small (CCXT supports all named venues).
2. **Add term structure feature (Class J)** -- likelihood: medium. Funding term slope (8h vs longer-dated basis) is class J in the regime matrix; tested-in but limited weight. Could improve entry timing.
3. **Bear-market validation needed** -- likelihood: not a revival but a confirmation. Strategy hasn't been tested in a sustained negative-funding bear regime. The behavior should be "sit out cleanly" but real-world execution has slippage / withdrawal risk.
4. **LSTM v2 architecture for non-funding alpha** -- likelihood: low for THIS engine. Engine 7 + funding-rate features is mechanistically tied to the carry P&L; replacing the classifier with an LSTM might add modest lift but the structural edge is the funding mechanism itself, not the model class.
```

### Step 5: Re-run atlas_sync

```powershell
python -m engines.atlas_sync
```

Expected output:

- TRADING_ATLAS.md: 0 added / 2 updated (Exps 1 and 13) /
  13 unchanged.
- Embeddings: 2 regenerated (because `full_markdown` changed
  for Exps 1 and 13 due to the new sections becoming part of
  the captured body).
- New columns populated for Exps 1 and 13:
  `test_conditions`, `revival_hypotheses`,
  `regime_state_at_test`, `computational_engine`.
- Other 13 experiments still NULL in the new columns
  (expected; backfill comes Cycle 35).

### Step 6: Verify with MCP

```
praxis:atlas_get(1)   # MEAN_REVERSION x EQUITY_US
praxis:atlas_get(11)  # MICROSTRUCTURE x CRYPTO Funding Carry (DB id 11)
```

Expected: response includes `test_conditions`,
`revival_hypotheses`, `regime_state_at_test`,
`computational_engine` keys, populated with the structured
content. The other entries (atlas_get(2), atlas_get(3), etc.)
should show those keys as NULL.

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | Migration script applied; `PRAGMA table_info(atlas_experiments)` shows the 4 new columns |
| 2 | `docs/COMPUTATIONAL_ENGINES.md` committed to repo |
| 3 | TRADING_ATLAS.md: Exp 1 attribute table includes Computational engine row + new structured sections (Test conditions, Active regimes during test, Revival hypotheses) parse cleanly |
| 4 | TRADING_ATLAS.md: Exp 13 attribute table includes Computational engine row + same three new sections (with content tuned for a POSITIVE experiment) |
| 5 | `engines/atlas_sync.py` parser updated; py_compile clean |
| 6 | atlas_sync re-run reports 2 updated TRADING entries (Exps 1, 13) + 0 added + 13 unchanged |
| 7 | atlas_sync regenerates embeddings for the 2 updated entries; 33 skipped |
| 8 | `praxis:atlas_get(1)` returns populated `test_conditions`, `revival_hypotheses`, `regime_state_at_test`, `computational_engine` |
| 9 | `praxis:atlas_get(11)` (Funding Carry) returns same with content reflecting POSITIVE experiment shape |
| 10 | `praxis:atlas_get(2)` (TA crypto, not backfilled) returns NULL for the 4 new columns -- proves backfill is per-experiment, not blanket |
| 11 | `praxis:atlas_search("MOMENTUM crypto", top_k=3)` still returns Exps 17, 9, 8 in same order (embeddings stable for non-backfilled entries) |

## Step ordering (load-bearing)

Code MUST execute in this order:

1. Apply schema migration (script).
2. Update parser (`engines/atlas_sync.py`).
3. Test parser on a CHEAP path: `python -m engines.atlas_sync
   --validate` if that flag exists, otherwise dry-read the
   markdown and verify section extraction works without
   writing.
4. Add `docs/COMPUTATIONAL_ENGINES.md` (no parsing impact;
   just a new file).
5. Update `TRADING_ATLAS.md` for Exps 1 and 13.
6. Run atlas_sync for real.
7. Verify via MCP `atlas_get`.

If steps 1-3 fail, no markdown work happens (markdown rolls
forward only after parser is proven).

## Commit messages

### Commit 1: schema + script + doc + parser

```
Cycle 33: Atlas schema extension + COMPUTATIONAL_ENGINES.md + parser

Adds four new columns to atlas_experiments for structured
experiment metadata beyond the original signal_type / asset_class
/ result_class / key_findings / atlas_principle:

- test_conditions: JSON dict capturing bar type, frequency,
  TC, pre-filter, risk management, feature set, computational
  engine. Parsed from a new "**Test conditions:**" section in
  each experiment's markdown.

- revival_hypotheses: JSON list capturing what modifications
  could flip a NEGATIVE/INCONCLUSIVE experiment's verdict, or
  scale a POSITIVE experiment. Parsed from a new "**Revival
  hypotheses:**" section.

- regime_state_at_test: JSON dict capturing which of the 12
  regime classes (REGIME_MATRIX.md) were active during the
  test window. Often NULL or "not_measured" because most
  pre-Cycle-17 experiments didn't have regime detection
  built. Parsed from "**Active regimes during test:**".

- computational_engine: integer 1-7 referencing
  COMPUTATIONAL_ENGINES.md. Parsed from a new
  "Computational engine" row in each experiment's attribute
  table.

Adds docs/COMPUTATIONAL_ENGINES.md documenting the 7-engine
taxonomy (Cointegration/MR, Momentum/Trend, Allocation,
Vol/Options, Microstructure, On-Chain/DeFi, Event/Signal)
that has lived in chat archive since March 2026 and partly
in code (engines/base.py, engines/context/, engines/adapters/,
engines/model.py). Includes the experiment-to-engine mapping
for all 15 atlas experiments.

Parser update in engines/atlas_sync.py extracts the new
sections and populates the new columns. md_hash sensitivity
preserved: experiments that gain the new sections will
re-embed (since full_markdown changes); experiments without
them stay NULL on the new columns.

Migration script at scripts/migrations/cycle33_atlas_schema_
extension.py is idempotent; runs once cleanly, exits cleanly
on re-run. Existing 35 atlas_experiments rows preserved with
NULL in the 4 new columns.

This commit is infrastructure only. Mass backfill of the
remaining 13 experiments is Cycle 35 (after info bars land in
Cycle 34).
```

### Commit 2: backfill 2 experiments + re-sync + verify

```
Cycle 33 step 2: backfill Exp 1 + Exp 13 with structured fields

Adds Computational engine attribute row + Test conditions /
Active regimes during test / Revival hypotheses sections to
the markdown for two reference experiments:

- Exp 1 (MEAN_REVERSION x EQUITY_US): NEGATIVE result.
  Computational engine: 1 (Cointegration) + 3 (Allocation).
  Revival hypotheses focus on TC reduction and regime A trend
  filtering. The pre-existing regime ablation results are
  formalized into the regime_state_at_test field.

- Exp 13 (MICROSTRUCTURE x CRYPTO Funding Rate Carry): the
  single POSITIVE in the Atlas. Computational engine: 7
  (Event/Signal) + 3 (Allocation/Kelly). "Revival" reframed
  as "scaling/improving" since this experiment isn't dead.
  Hypotheses cover cross-venue funding, term structure
  features, bear-market validation needs.

These two are the design validation: a NEGATIVE experiment
and a POSITIVE experiment, both well-documented, both with
non-trivial revival narratives. If the structure works for
both, it should work for the remaining 13 (backfilled in
Cycle 35).

atlas_sync re-run: 2 updated, 0 added, 13 unchanged on
TRADING_ATLAS.md side. 2 embeddings regenerated, 33 skipped.

Verified via praxis:atlas_get(1) and atlas_get(11):
both return the new fields populated; atlas_get(2) and
others return NULL for new columns.
atlas_search("MOMENTUM crypto", top_k=3) returns same Exps
17, 9, 8 in same order (non-backfilled embeddings stable).
```

## Post-cycle: what comes next

- **Cycle 34**: Info Bars v0.1 (dollar bars, volume bars,
  volume-imbalance bars per Lopez AFML Ch. 2). Built against
  `crypto_data.trades`. Standalone work; doesn't touch the
  Atlas.
- **Cycle 35**: Atlas mass backfill -- 13 remaining
  experiments get their structured fields populated. With
  info bars now available (Cycle 34), the
  `revival_hypotheses` for several Engine 2 experiments can
  reference info bars concretely.
- **Cycle 36+**: First info-bar revival re-runs targeting the
  highest-likelihood revival hypotheses from the Atlas.
