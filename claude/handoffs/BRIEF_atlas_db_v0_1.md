# Implementation Brief: Atlas DB v0.1 — Markdown-as-source / DB-as-derived

**Series:** praxis
**Cycle:** 8
**Priority:** P2 (infrastructure for future agent work; immediate value as machine-queryable Atlas)
**Mode:** A (offline migration tool + new sidecar DB + new MCP tool; no live execution paths touched, no edits to existing Atlas markdown content)

**Estimated Scope:** L (3-4 hours: parser, schema, embedding pipeline, MCP tool, smoke test, docs)
**Estimated Cost:** $0.50-2.00 one-time (embedding generation for ~25 atlas entries via Voyage or OpenAI ada-002; falls within sandbox budget)
**Estimated Data Volume:** ~25 atlas entries → ~25 SQLite rows + ~25 embedding vectors. Total `praxis_meta.db` size < 5 MB. No writes to `crypto_data.db` or any other existing DB.
**Kill switch:** **No edits to TRADING_ATLAS.md, PREDICTION_MARKET_ATLAS.md, or REGIME_MATRIX.md content.** The migration tool reads them and never writes them. The MCP tool exposes db reads only. If at any point the parser would need to disambiguate by changing markdown structure, stop and write a retro flagging the ambiguity for human resolution — do not "normalize" the markdown.

Reference: rules 9-15 (progress reporting), rule 27 (md-only-when-needed), `WORKFLOW_MODES_PRAXIS.md` Mode A.

---

## Context

Praxis has three reference documents that constitute the strategic memory of the project:

- `TRADING_ATLAS.md` (1001 lines, 17 numbered experiments, well-structured per-entry schema with attribute tables)
- `PREDICTION_MARKET_ATLAS.md` (526 lines, narrative + ranked strategy categories, less rigidly structured)
- `docs/REGIME_MATRIX.md` (254 lines, 12 regime classes with definitions, conceptual reference not experiments)

Today these are read-only-by-human. As the project matures toward the multi-agent vision (Discovery → Research → Data → Backtest → Manager pipeline), the Atlas needs to be machine-queryable — specifically, the Research Agent will need to do **semantic similarity search** against past experiments to filter incoming trade ideas. Keyword grep on markdown does not solve this; "BTC mean-reversion at 1-min timescale" must match against an existing experiment indexed under "intraday equities pairs trading at minute frequency" via embedding similarity, even though the words don't overlap.

Independent of agents, the same capability has immediate value for our manual workflow. When Jeff sends an article from the gym, Chat currently reasons against the Atlas from memory. With a queryable Atlas, Chat can do `atlas_search("ease of movement indicator backtest")` and get back the relevant TA_STANDARD × CRYPTO experiment with its actual quantitative result, instead of paraphrasing.

### Architectural decision: MD is source of truth, DB is derived + augmented

Three sync models were considered:
1. **Bidirectional sync** — both editable, conflict resolution. **Rejected** as too much engineering.
2. **DB authoritative, MD generated** — clean, but loses what makes md valuable: free-form authored prose. **Rejected**.
3. **MD authoritative, DB derived + augmented** — humans author markdown as today; migration tool parses to db; db adds machine-only fields (embeddings, normalized enums, cross-references) that md does not contain. **Selected.**

**Workflow rule:** "the md and db should always be updated simultaneously" is operationalized as: edit md → run `python -m engines.atlas_sync` → review printed diff → commit both files. No git pre-commit hook is added in this Brief — workflow lock-in should follow workflow validation, not lead it.

### Why a sidecar DB rather than a table in `crypto_data.db`?

`crypto_data.db` is touched by live collectors at high cadence. The Atlas is metadata, written rarely (one row added per cycle in steady state, less than weekly). Mixing them creates failure-mode entanglement: a corrupted Atlas migration could lock the live collector DB; a collector schema change could orphan Atlas rows. Separation of concerns. New file: `data/praxis_meta.db`.

---

## Objective

Deliver three things, in order:

1. **`engines/atlas_sync.py`** — a CLI tool that parses the three Atlas markdown files into a structured SQLite DB at `data/praxis_meta.db`, generates embedding vectors per experiment, and reports a diff of what changed since the last sync.
2. **`servers/praxis_mcp/tools/atlas.py`** — a new MCP tool module exposing two read-only tools: `atlas_search(query, top_k)` for semantic similarity search and `atlas_get(entry_id)` for full-detail retrieval of a single experiment. Wired into `servers/praxis_mcp/server.py`.
3. **One-page docs update** — `docs/ATLAS_DB.md` describing schema, sync workflow, and the md-authoritative rule. Linked from both Atlas markdown files at the top so future humans (and future Claudes) know the relationship.

---

## Detailed Spec

### Phase 0 — Schema design (45 min, write code only after this is correct)

**Schema for `data/praxis_meta.db`:**

```sql
-- Experiments parsed from TRADING_ATLAS.md and PREDICTION_MARKET_ATLAS.md
CREATE TABLE atlas_experiments (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file         TEXT NOT NULL,         -- 'TRADING_ATLAS.md' | 'PREDICTION_MARKET_ATLAS.md'
    source_section      TEXT NOT NULL,         -- e.g. '13. MICROSTRUCTURE × CRYPTO — Funding Rate Carry'
    source_line_start   INTEGER NOT NULL,      -- line number in source md (for citation in MCP tool)
    source_line_end     INTEGER NOT NULL,
    -- Parsed structured fields (nullable; not every entry has every field)
    signal_type         TEXT,                  -- 'TA_STANDARD' | 'MEAN_REVERSION' | 'MICROSTRUCTURE' | etc. (free text; do not enum-enforce)
    asset_class         TEXT,                  -- 'CRYPTO' | 'EQUITY_US' | 'FUTURES_INDEX' | 'FX_G10' | etc.
    framework           TEXT,                  -- 'CPO' | 'Triple Barrier' | 'NegRisk arb' | etc.
    date_run            TEXT,                  -- ISO date string; null if undateable
    result_class        TEXT,                  -- 'POSITIVE' | 'NEGATIVE' | 'INCONCLUSIVE' | 'PARTIAL'
    result_summary      TEXT,                  -- one-paragraph plain-language summary
    -- Free-text capture
    full_markdown       TEXT NOT NULL,         -- the raw markdown of the entry, for full-detail retrieval
    key_findings        TEXT,                  -- bulleted findings as a concatenated string (multi-line)
    atlas_principle     TEXT,                  -- the "Atlas principle established" block if present
    -- Sync metadata
    md_hash             TEXT NOT NULL,         -- sha256 of full_markdown; changes when md is edited
    synced_at           TEXT NOT NULL,         -- ISO timestamp of last sync
    UNIQUE(source_file, source_section)
);

-- Embeddings stored separately (BLOB column would balloon the experiments table)
CREATE TABLE atlas_embeddings (
    experiment_id       INTEGER PRIMARY KEY,
    embedding_model     TEXT NOT NULL,         -- 'voyage-3-lite' | 'text-embedding-3-small' | etc.
    embedding_dim       INTEGER NOT NULL,
    embedding           BLOB NOT NULL,         -- numpy float32 array, .tobytes()
    embedded_at         TEXT NOT NULL,         -- ISO timestamp; if md_hash changed, re-embed
    FOREIGN KEY (experiment_id) REFERENCES atlas_experiments(id) ON DELETE CASCADE
);

-- Regime matrix reference data (not experiments)
CREATE TABLE regime_classes (
    class_letter        TEXT PRIMARY KEY,      -- 'A' | 'B' | ... | 'L'
    class_name          TEXT NOT NULL,         -- 'Trend' | 'Vol level' | etc.
    states              TEXT NOT NULL,         -- '-2/-1/0/+1/+2' literal as in md
    detection_method    TEXT NOT NULL,         -- 'ADX + multi-period return alignment + MA stack'
    key_data            TEXT NOT NULL          -- 'OHLCV daily' etc.
);

-- Strategy × regime relevance matrix (parsed from REGIME_MATRIX.md)
CREATE TABLE strategy_regime_relevance (
    strategy_name       TEXT NOT NULL,         -- 'Crypto TA' | 'Momentum' | etc.
    class_letter        TEXT NOT NULL,         -- references regime_classes.class_letter
    relevance_dots      INTEGER NOT NULL,      -- 1 (●), 2 (●●), 3 (●●●)
    PRIMARY KEY (strategy_name, class_letter)
);

-- Sync log (so we can see history of what changed)
CREATE TABLE sync_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    synced_at           TEXT NOT NULL,
    source_file         TEXT NOT NULL,
    entries_added       INTEGER NOT NULL,
    entries_updated     INTEGER NOT NULL,
    entries_unchanged   INTEGER NOT NULL,
    notes               TEXT
);
```

**Design notes:**

- Per memory rule "everything is a parameter" — `signal_type`, `asset_class`, etc. are free-text TEXT columns, not enums. The Atlas markdown evolves; new categories will appear; we don't want a schema migration every time. Validation, if needed later, lives in queries not the schema.
- `full_markdown` is duplicated from the source file. This is deliberate — it makes `atlas_get` self-contained without re-parsing markdown at query time, and the duplication is naturally re-synced on every run.
- `md_hash` enables incremental sync: if the md content of an entry hasn't changed, skip re-embedding (saves API cost on every run).
- Embeddings are stored as raw bytes of a numpy float32 array. Use `np.frombuffer(blob, dtype=np.float32)` to recover. Don't use SQLite's vector extensions (sqlite-vss etc.) — adds a build dependency, and at ~25 entries cosine similarity in numpy is sub-millisecond.

### Phase 1 — Markdown parser (75 min)

Build `engines/atlas_sync.py` with these responsibilities:

```python
# Pseudocode
def parse_trading_atlas(md_path: Path) -> list[ExperimentRecord]:
    """
    Parse TRADING_ATLAS.md. Numbered experiments (### N. NAME) are records.
    Sections like 'Data pipeline', 'RF architecture', etc. are NOT experiments — skip them.
    Records run from a numbered ### heading until the next ### heading.

    For each record, extract the attribute table if present:
    | **Date** | 2026-03-22 |
    | **Framework** | ... |
    etc. Map keys to schema columns case-insensitively, allowing for variants
    ('Date' / 'Date Run' / 'Run Date').
    """

def parse_prediction_market_atlas(md_path: Path) -> list[ExperimentRecord]:
    """
    PMA structure is different — strategy categories with sub-strategies.
    For v0.1 we capture each ranked strategy (Rank 1, Rank 2, etc.) as an entry,
    plus each sub-section under "Category A/B/C/D" as separate entries.
    Most fields will be sparser than TRADING_ATLAS entries — that's fine,
    the schema allows nulls.
    """

def parse_regime_matrix(md_path: Path) -> tuple[list[RegimeClass], list[RelevanceRow]]:
    """
    Parse the 'Regime Classes' table (regime_classes table) and the
    'Relevance Matrix' table (strategy_regime_relevance table).
    Convert the dot count: ●●● = 3, ●● = 2, ● = 1.
    """
```

**Parser robustness rules (rule 16 maximalist validation):**

- The parser is the part most likely to silently corrupt data, because markdown is freeform. Three defenses:
  1. **`--validate` flag** — runs the parser, reports per-entry parsing decisions to stdout, but does NOT write the DB. Default off.
  2. **`--verbose` flag** — for each entry, log which fields were extracted and which were null. Tells us when a regex misses a field that should have been captured.
  3. **Round-trip test** — for each parsed entry, log: "Entry N. NAME — captured signal_type=X, asset_class=Y, result_class=Z. Did NOT capture: [list]." Lets a human spot-check that nothing important was missed.
- If the parser raises on a specific entry, log the entry name and skip it (do not crash the whole run). Add a `--strict` flag that does crash on parse error for use in CI later.
- Parser should be deterministic: two runs over the same markdown produce identical DB rows (modulo `synced_at` and `embedded_at` timestamps).

### Phase 2 — Embedding generation (45 min)

**Provider choice:** Use **Voyage AI's `voyage-3-lite`** if `VOYAGE_API_KEY` is in `.env`, else fall back to **OpenAI `text-embedding-3-small`** if `OPENAI_API_KEY` is set, else **fail with a clear error** ("set VOYAGE_API_KEY or OPENAI_API_KEY in .env to enable embeddings; you can run the parser without embeddings via --no-embed for schema validation"). Both providers cost cents for the entire Atlas; voyage-3-lite is cheaper and produces higher-quality retrieval embeddings on technical content.

**What to embed:** The concatenation of `source_section` + `result_summary` + `key_findings` + `atlas_principle`, NOT the full markdown. Reasoning: the full markdown contains attribute tables (`| Date | ... |`) that pollute embeddings with structural tokens. The semantic content is in the prose fields.

**Skip-if-unchanged logic:** Before calling the embedding API for an entry, check if `md_hash` matches the current `atlas_embeddings.embedded_at` row's hash. If yes, skip. This makes re-runs of the sync tool cheap.

**Use `load_dotenv()`** before reading any API keys (memory rule #20).

### Phase 3 — Diff reporting (30 min)

The sync tool prints a structured diff at the end of each run:

```
Atlas sync — 2026-04-23 21:14:33 UTC
  TRADING_ATLAS.md
    + 0 added
    ~ 1 updated:   #13 MICROSTRUCTURE × CRYPTO — Funding Rate Carry (md_hash changed)
    = 16 unchanged
  PREDICTION_MARKET_ATLAS.md
    + 0 added
    ~ 0 updated
    = 8 unchanged
  REGIME_MATRIX.md
    = 12 regime classes unchanged
    = 5 strategy relevance rows unchanged
  Embeddings: 1 regenerated (Voyage), 24 skipped (md unchanged), $0.001 total
  Total entries in atlas_experiments: 25
  Run logged to sync_log table.
```

This output is the user-visible artifact of the workflow. Make it scannable. No emojis (Windows cp1252 — memory rule).

### Phase 4 — MCP tool exposure (45 min)

Create `servers/praxis_mcp/tools/atlas.py` with two tools:

```python
@mcp.tool()
def atlas_search(query: str, top_k: int = 5) -> dict:
    """Semantic similarity search across Praxis Atlas experiments.

    Embeds the query using the same embedding model used at sync time,
    then returns the top_k most similar experiments by cosine similarity.

    Use this when triaging a new trading idea against accumulated experimental
    evidence. Returns experiment metadata and result_summary; for full detail
    on a specific entry call atlas_get(entry_id).

    Returns:
        Dict with:
          query: the input query (echoed)
          model: embedding model used
          results: list of {id, source_file, source_section, signal_type,
                            asset_class, result_class, result_summary,
                            similarity_score} ordered by similarity desc
    """

@mcp.tool()
def atlas_get(entry_id: int) -> dict:
    """Retrieve full details for a single Atlas entry by id.

    Returns the complete parsed structure plus the original markdown,
    plus a citation pointer (source_file:line_start-line_end) suitable
    for human verification.
    """
```

**Key principles:**

- Both tools read from `data/praxis_meta.db` via the existing `connect_ro()` helper used by the rest of the MCP server. Read-only enforcement is automatic.
- `atlas_search` requires the same embedding provider be available at query time. If `VOYAGE_API_KEY` is missing at query time but the DB was synced with Voyage, return a clear error instead of falling back silently to a different model (cosine similarity across model families is meaningless).
- Wire the new tool module in `servers/praxis_mcp/server.py` following the existing pattern in `tools/meta.py`, `tools/ohlcv.py`, etc.
- Update `servers/praxis_mcp/README.md` with the two new tools in the same style as the existing entries.

### Phase 5 — Markdown linkage (15 min)

Add a one-line note at the top of `TRADING_ATLAS.md`, `PREDICTION_MARKET_ATLAS.md`, and `REGIME_MATRIX.md`:

```markdown
> **Sync state:** This file is the source of truth. After editing, run
> `python -m engines.atlas_sync` to update the queryable DB at `data/praxis_meta.db`.
> See `docs/ATLAS_DB.md` for details.
```

**This is the only edit to the existing Atlas markdown.** Pure header note, no content changes anywhere else.

Create `docs/ATLAS_DB.md` documenting:
- The md-authoritative / db-derived rule
- Schema overview (table list, link to atlas_sync.py for source of truth on column definitions)
- Sync workflow: edit md → run sync → review diff → commit both
- How to query interactively (MCP tools available in Chat; sqlite3 CLI for ad-hoc)
- How embeddings are stored and which provider is used
- Failure modes and recovery (re-run sync; corrupt DB → delete and re-create from md)

### Phase 6 — Smoke test (30 min)

Create `tests/test_atlas_sync.py` with these tests at minimum:

1. **Parser determinism.** Parse `TRADING_ATLAS.md` twice. Assert identical extracted records.
2. **Round-trip stability.** Sync to a temp DB, re-sync. Assert `entries_unchanged == total_entries` on the second run.
3. **md_hash sensitivity.** Modify one entry's content in a copy of the md, re-sync, assert exactly that entry is in the `entries_updated` count.
4. **Schema validation.** After sync, query `atlas_experiments` and assert all rows have non-null `id`, `source_file`, `source_section`, `full_markdown`, `md_hash`, `synced_at`.
5. **Regime matrix counts.** Assert exactly 12 rows in `regime_classes` and exactly 5 strategy rows in `strategy_regime_relevance` (per current REGIME_MATRIX.md).

If a test discovers the parser is wrong about a real Atlas entry, **fix the parser, do not edit the markdown to match the parser**. The md is canonical.

### Phase 7 — Retro

Standard retro at `claude/retros/RETRO_atlas_db_v0_1.md`. Include:

- Schema as actually shipped (in case Phase 0 design drifted during implementation)
- Per-Atlas-file count of entries successfully parsed
- Embedding cost (actual, in cents)
- Any entries that the parser couldn't fully extract — list them as known gaps for a follow-up Brief, not as bugs to fix in this cycle
- `atlas_search` smoke-query output: run `atlas_search("funding rate carry")` after sync and report the top 3 results — should include experiment 13 with high similarity score; if not, the embedding pipeline is broken.

---

## Acceptance Criteria

1. `data/praxis_meta.db` exists with all five tables populated. At least 17 rows in `atlas_experiments` from TRADING_ATLAS.md, plus PMA entries (target ≥5), plus 12 regime_classes, plus 5 strategy_regime_relevance.
2. `python -m engines.atlas_sync --validate` runs without errors, prints per-entry capture report, does NOT write the DB.
3. `python -m engines.atlas_sync` runs to completion, prints the diff summary, writes the DB. Re-running immediately reports zero updates.
4. The two MCP tools (`atlas_search`, `atlas_get`) appear in `tool_search` results from a fresh Claude Desktop relaunch.
5. `atlas_search("funding rate carry")` returns experiment 13 (MICROSTRUCTURE × CRYPTO Funding Rate Carry) within the top 3 results.
6. `docs/ATLAS_DB.md` exists and accurately describes the workflow.
7. Each of the three Atlas markdown files has the sync-state header note and no other modifications.
8. All tests in `tests/test_atlas_sync.py` pass.
9. ASCII-only across all new files (rule 19).
10. No edits to live execution paths, no edits to `crypto_data.db`, no edits to existing collector code.

---

## Known Pitfalls

- **PMA parser will be lossier than TA parser.** PREDICTION_MARKET_ATLAS.md has narrative structure rather than per-entry attribute tables. Don't try to force the same schema fit. Capture what's parseable cleanly (rank, category, strategy_name, summary) and let other fields be null. A future Brief can deepen PMA parsing once we know what queries we want.
- **Don't pollute the DB with Atlas section dividers.** Lines like `## How to use this atlas`, `## Landscape Matrix`, `### Signal types (rows)`, `### Data pipeline`, `### RF architecture` are reference material, not experiments. The parser must skip these. Numbered `### 1. NAME × CLASS — DESCRIPTION` headings are the experiments. Numbered or letter-prefixed sub-sections inside an experiment (e.g. "Addendum: Experiment 10") are part of the parent experiment, not separate.
- **Embedding pollution from attribute tables.** The full markdown has lots of `| Field | Value |` table rows that look like noise to an embedding model. Embed prose fields only (`source_section + result_summary + key_findings + atlas_principle`).
- **Voyage and OpenAI embedding dimensions differ.** Voyage 3-lite is 512-d, OpenAI 3-small is 1536-d. Store `embedding_dim` per row. At query time, verify the query embedding's dim matches the stored embeddings' dim — if not, error clearly.
- **Cosine similarity, not L2.** Use cosine similarity (or equivalently, L2 on L2-normalized embeddings) for retrieval. Both providers return non-normalized vectors by default; either normalize before storing or normalize at query time. Be consistent.
- **The MCP server caches its module imports.** After adding `tools/atlas.py` and modifying `server.py`, Claude Desktop must be fully quit (not just window-closed) and relaunched to pick up the new tools. Note this in the retro so Jeff knows the verification step.
- **Don't include `data/praxis_meta.db` in delta zips going forward** (memory delta-zip rule, no DB files). It's regeneratable from md + a sync run; senders include the migration tool + md, recipients run sync to recreate.
- **One-shot embeddings are not free, but they're cheap.** ~25 entries × ~500 tokens each × Voyage 3-lite price = pennies. But re-running every cycle multiplies. The skip-if-unchanged logic in Phase 2 is non-optional, not an optimization.
- **Don't add a git pre-commit hook.** Tempting because it would enforce sync. But it locks the workflow before we've validated it, and breaks single-file commits where Jeff intentionally edits md without immediately re-syncing. Workflow lock-in follows workflow validation; not in this Brief.

---

## What this Brief deliberately does NOT do

- No agent design. No Discovery / Research / Backtest / Manager Agent stubs.
- No promotion/demotion schema for live strategies. Field set in `atlas_experiments` is descriptive only — no `live_status`, no `kelly_weight`, no `current_sharpe`. These come later when there's a Manager Agent to consume them.
- No NautilusTrader integration. Separate cycle.
- No edits to existing live execution code (`engines/funding_rate_strategy.py`, `engines/lstm_predictor.py`, etc.).
- No new MCP tools beyond the two atlas-specific ones. Resist scope creep — the MCP server has 12 tools after this; that's enough for one cycle.
- No bidirectional sync. MD is canonical; DB is derived.
- No git hook. Workflow rule, not a technical mechanism.

---

## References

- `claude/WORKFLOW_MODES_PRAXIS.md` — Mode A (offline tools, no live execution)
- `claude/CLAUDE_CODE_RULES.md` — rules 9-15 (progress reporting), rule 16 (max validation), rule 19 (ASCII), rule 20 (load_dotenv)
- `TRADING_ATLAS.md` — primary source data (1001 lines, 17 numbered experiments)
- `PREDICTION_MARKET_ATLAS.md` — secondary source (526 lines, narrative structure)
- `docs/REGIME_MATRIX.md` — reference data (12 regime classes, 5 strategy relevance rows)
- `servers/praxis_mcp/tools/meta.py` — pattern reference for new MCP tool module (post-Cycle 7 patched version)
- `servers/praxis_mcp/server.py` — wire-in point for new tool module
- Voyage AI embeddings docs: https://docs.voyageai.com/docs/embeddings (voyage-3-lite, 512-d, free tier covers our usage)
- OpenAI embeddings docs: https://platform.openai.com/docs/guides/embeddings (text-embedding-3-small fallback)
