# Atlas DB

Queryable mirror of the Praxis Atlas markdown files, plus embeddings for
semantic search. Lives at `data/praxis_meta.db` (separate from the live
collector DB at `data/crypto_data.db`).

## The rule

**Markdown is the source of truth. The DB is derived.**

The three Atlas markdown files are authored by humans:

- `TRADING_ATLAS.md`
- `PREDICTION_MARKET_ATLAS.md`
- `docs/REGIME_MATRIX.md`

The DB at `data/praxis_meta.db` is regenerated from these by
`python -m engines.atlas_sync`. Edit the markdown, then run sync; never the
other way around. Nothing in the DB is canonical -- if the DB and markdown
disagree, the markdown wins and the DB needs re-syncing.

## Sync workflow

```bash
# Edit one or more atlas markdown files in your editor.
# Then:
python -m engines.atlas_sync

# Review the printed diff (added / updated / unchanged per file).
# Commit both the markdown change and any DB-side artifacts you care about.
```

The sync tool is idempotent: re-running with no markdown changes prints
`+0 added, ~0 updated, =N unchanged` and skips the embedding step entirely
(via `md_hash` comparison). You can run it any time without spending API
credits.

### Flags

- `--validate` -- parse only; print per-entry capture report; do not write
  the DB. Use this when you change parser logic and want to see what the
  parser would produce before committing to a write.
- `--verbose` -- alongside `--validate`, prints which fields each entry
  captured / missed. Helps debug new markdown patterns the parser doesn't
  recognize yet.
- `--no-embed` -- skip the embedding step. Useful for fast iteration on
  parser changes without burning API quota.
- `--strict` -- crash on parse errors instead of logging and continuing.
  Intended for CI; not the default because the markdown is freeform and
  occasional partial extraction is preferable to a hard fail.
- `--db-path PATH` -- write to a different DB (used by tests).

## Schema overview

Five tables:

- `atlas_experiments` -- one row per parsed experiment / strategy entry.
  Source-of-truth fields: `source_file`, `source_section`,
  `source_line_start`, `source_line_end`, `full_markdown`, `md_hash`.
  Parsed fields (best-effort, may be null): `signal_type`, `asset_class`,
  `framework`, `date_run`, `result_class`, `result_summary`, `key_findings`,
  `atlas_principle`. Sync metadata: `synced_at`. Unique on
  `(source_file, source_section)`.
- `atlas_embeddings` -- one row per experiment, holding a numpy float32
  embedding vector as a BLOB. The vector is L2-normalized at write time so
  cosine similarity at query time is a plain dot product. Includes
  `embedding_model`, `embedding_dim`, `md_hash` (for skip-if-unchanged
  logic), and `embedded_at`.
- `regime_classes` -- 12 rows mirroring the "Regime Classes" table in
  `REGIME_MATRIX.md`. Columns: `class_letter` (PK), `class_name`, `states`,
  `detection_method`, `key_data`.
- `strategy_regime_relevance` -- bridge table mirroring the "Relevance
  Matrix". One row per (`strategy_name`, `class_letter`) pair with a
  `relevance_dots` integer (1-3). Roughly 5 strategies x 12 classes = 60
  rows.
- `sync_log` -- append-only log of sync runs with per-file
  added/updated/unchanged counts. Useful for archaeology; not used by any
  tool.

The schema is defined in `engines/atlas_sync.py` (DDL string at the top of
the file). Treat that as the source of truth for column definitions.

## Querying

### Through Claude Desktop (preferred)

The Praxis MCP server exposes two read-only tools backed by this DB:

- `atlas_search(query, top_k=5)` -- semantic similarity search.
- `atlas_get(entry_id)` -- full detail + citation.

Both tools live in `servers/praxis_mcp/tools/atlas.py`. They use the same
embedding model that synced the DB; if the env var for that provider is
missing at query time, they return a clear error rather than silently
falling back to a different model.

### Ad-hoc via sqlite3 CLI

```bash
sqlite3 data/praxis_meta.db
sqlite> .schema atlas_experiments
sqlite> SELECT id, source_file, source_section, result_class
        FROM atlas_experiments
        WHERE result_class = 'POSITIVE';
```

## Embeddings

Voyage `voyage-3-lite` (512-dim) is the default if `VOYAGE_API_KEY` is in
`.env`. Falls back to OpenAI `text-embedding-3-small` (1536-dim) if
`OPENAI_API_KEY` is set. Errors clearly if neither key is present (use
`--no-embed` to skip).

What gets embedded: `source_section + result_summary + key_findings +
atlas_principle`, NOT the full markdown. Attribute tables in the full
markdown contain table-formatting tokens (`| Field | Value |`) that
pollute embeddings; the prose-only blob retrieves much better.

Vectors are L2-normalized at write time. Cosine similarity at query time
is therefore a plain dot product against a normalized query vector.

The DB stores `embedding_model` per row. If you switch models (e.g.
Voyage -> OpenAI), the next sync re-embeds everything because the stored
model differs from the current provider.

## Failure modes and recovery

- **Parse error on one entry.** Default behavior: log a warning and skip
  the entry's structured fields (full_markdown is still preserved). Use
  `--strict` to crash instead.
- **Embedding API outage.** Sync logs the error and writes a `notes`
  field in `sync_log` indicating the embedding step failed. Re-run later;
  unchanged-since-last-sync entries are skipped automatically.
- **Corrupt DB.** Delete `data/praxis_meta.db` and re-run sync. The DB is
  fully derived; nothing is lost.
- **Markdown and DB out of sync (e.g. someone edited markdown without
  running sync).** Just re-run sync. The diff report shows what changed.

## What this DB does NOT contain

- Anything from `data/crypto_data.db`. The two are intentionally separate
  -- a corrupt Atlas migration must not be able to lock the live collector
  DB, and a collector schema change must not orphan Atlas rows.
- Live trading state, positions, P&L, agent decisions. Those belong
  elsewhere when they exist.
- Bidirectional state. The DB is a read mirror of the markdown plus
  embeddings; nothing flows back.
