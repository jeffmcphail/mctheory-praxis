# Cycle 33.5 -- MCP atlas tools: expose the 4 new structured columns

**Predecessor:** Cycle 33 (`6ab289d`, atlas schema extension).
**Mode:** Hybrid. Tiny brief; ~15-line edit to a single file.
**Risk:** very low. Read-only tool; additive SELECT columns;
JSON parsing wrapped in defensive try/except.

## Why now

Cycle 33's MCP-side verification surfaced a gap:
`praxis:atlas_get(N)` returns the existing 16 columns but not
the 4 new structured columns added in Cycle 33
(`computational_engine`, `test_conditions`,
`revival_hypotheses`, `regime_state_at_test`). The data is
correctly in `data/praxis_meta.db` (verified by direct
sqlite3 query). The gap is purely in
`servers/praxis_mcp/tools/atlas.py`, which hand-codes a
`SELECT` rather than `SELECT *`.

Until this is fixed, the Research Agent and any MCP client
can't navigate the new structured fields via the live tool;
they'd have to fall back to `raw_query` against the DB, which
defeats the purpose of having structured columns. Also blocks
clean Cycle 35 design (mass backfill is the use case that
makes structured fields valuable, and that value only
materializes via MCP access).

## What

Single file edit: `servers/praxis_mcp/tools/atlas.py`. Two
small functional changes:

1. **`atlas_get`**: extend the `SELECT` to include the 4 new
   columns. Parse the three JSON-encoded columns
   (`test_conditions`, `revival_hypotheses`,
   `regime_state_at_test`) back into Python dicts/lists with
   defensive try/except, so MCP consumers receive structured
   objects rather than raw JSON strings.
2. **`atlas_search`**: extend the per-result dict to include
   `computational_engine` (integer or null). Keep the big
   JSON fields out of search results -- search payloads
   should remain lightweight.

No DB schema changes. No parser changes. No markdown changes.
No re-sync needed (the data is already in the DB from Cycle
33).

## Specifics for Code

### File: `servers/praxis_mcp/tools/atlas.py`

**Edit 1 -- `atlas_search`** (around line 145-171):

The SELECT for the per-result fields needs `x.computational_engine`
added:

```python
cursor.execute(
    """
    SELECT e.experiment_id, e.embedding,
           x.source_file, x.source_section, x.signal_type,
           x.asset_class, x.result_class, x.result_summary,
           x.computational_engine
    FROM atlas_embeddings e
    JOIN atlas_experiments x ON x.id = e.experiment_id
    """
)
```

And the per-row dict construction (around line 160-170)
needs a new key:

```python
scored.append(
    {
        "id": row["experiment_id"],
        "source_file": row["source_file"],
        "source_section": row["source_section"],
        "signal_type": row["signal_type"],
        "asset_class": row["asset_class"],
        "result_class": row["result_class"],
        "result_summary": row["result_summary"],
        "computational_engine": row["computational_engine"],
        "similarity_score": round(score, 4),
    }
)
```

Also update the docstring's "Returns" section to mention the
new field:

```
results: list ordered by descending similarity, each containing
  {id, source_file, source_section, signal_type, asset_class,
   result_class, result_summary, computational_engine,
   similarity_score}
```

**Edit 2 -- `atlas_get`** (around line 207-227):

Extend the SELECT to include the 4 new columns:

```python
cursor.execute(
    """
    SELECT id, source_file, source_section, source_line_start,
           source_line_end, signal_type, asset_class, framework,
           date_run, result_class, result_summary, full_markdown,
           key_findings, atlas_principle,
           test_conditions, revival_hypotheses,
           regime_state_at_test, computational_engine,
           md_hash, synced_at
    FROM atlas_experiments WHERE id = ?
    """,
    (entry_id,),
)
```

Then, **after** the existing `result = {k: row[k] for k in
row.keys()}` line, decode the three JSON columns back into
structured objects. Use defensive parsing -- if any column is
NULL, leave it NULL; if it fails to parse as JSON, leave the
raw string but include a `_parse_error` note so debugging
is possible:

```python
import json  # add to file imports at the top if not already there

result = {k: row[k] for k in row.keys()}

for json_field in ("test_conditions", "revival_hypotheses",
                   "regime_state_at_test"):
    raw = result.get(json_field)
    if raw is None:
        continue
    try:
        result[json_field] = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Leave as-is; signal the issue but don't break the tool
        result[json_field + "_parse_error"] = (
            "stored value was not valid JSON"
        )

result["citation"] = (
    f"{row['source_file']}:lines "
    f"{row['source_line_start']}-{row['source_line_end']}"
)
return result
```

Also update the docstring's "Returns" section to enumerate
the new fields clearly:

```
Returns:
    Dict with all fields from atlas_experiments plus a `citation`
    field formatted as 'TRADING_ATLAS.md:lines 602-749'. The
    three JSON-typed fields (test_conditions,
    revival_hypotheses, regime_state_at_test) are decoded back
    into dicts/lists before return -- consumers don't need to
    json.loads() them. computational_engine is an integer
    (1-7) or null.
```

### Validation

Run `python -m py_compile servers/praxis_mcp/tools/atlas.py`
after the edit. There's no straightforward way for Code to
unit-test the MCP tool without the MCP server running, but
Code CAN verify the SQL changes work by running a manual
sqlite query that mirrors the new SELECT:

```python
import sqlite3, json
conn = sqlite3.connect("data/praxis_meta.db")
conn.row_factory = sqlite3.Row
row = conn.execute("""
    SELECT id, source_file, source_section, source_line_start,
           source_line_end, signal_type, asset_class, framework,
           date_run, result_class, result_summary, full_markdown,
           key_findings, atlas_principle,
           test_conditions, revival_hypotheses,
           regime_state_at_test, computational_engine,
           md_hash, synced_at
    FROM atlas_experiments WHERE id = 1
""").fetchone()
assert row is not None
assert row["computational_engine"] == 1
tc = json.loads(row["test_conditions"])
assert isinstance(tc, dict)
assert "bar_type" in tc
rh = json.loads(row["revival_hypotheses"])
assert isinstance(rh, list)
assert len(rh) == 4
print("OK")
```

That confirms the SELECT works against the actual data and the
JSON columns parse cleanly. The same test against entry_id=11
(Exp 13) should yield `computational_engine == 7`. Against
entry_id=2 (TA crypto, not backfilled) should yield
`computational_engine is None` and `test_conditions is None`.

After Code commits + pushes, I'll restart the MCP server
(Claude Desktop quit/relaunch -- same pattern as Cycle 27.5)
and verify via live `praxis:atlas_get(1)`, `atlas_get(11)`,
`atlas_get(2)`, and `atlas_search("MOMENTUM crypto", top_k=3)`
that the new fields surface correctly.

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | `servers/praxis_mcp/tools/atlas.py` py_compile clean after edit |
| 2 | `atlas_get` SELECT lists the 4 new columns explicitly |
| 3 | `atlas_get` decodes 3 JSON columns to structured objects with defensive try/except |
| 4 | `atlas_search` SELECT includes `computational_engine` |
| 5 | `atlas_search` per-result dicts include `computational_engine` key |
| 6 | Docstrings updated in both tools to enumerate new return fields |
| 7 | Local sqlite SELECT test (see above) passes against Exps 1, 11, 2 |

## Out of scope

- DB schema changes (Cycle 33 already done).
- Parser changes in `engines/atlas_sync.py` (Cycle 33 already done).
- Markdown backfill of remaining 13 experiments (Cycle 35).
- Filter parameters like `?engine=N` for `atlas_search`. That's
  a deferred TODO already in claude/TODO.md; comes after
  Cycle 35 mass backfill when the column is fully populated.
- Any other MCP tools (this cycle only touches atlas tools).

## Commit message (use verbatim)

```
Cycle 33.5: expose Cycle 33 structured fields via atlas MCP tools

Closes the MCP-layer gap from Cycle 33. The 4 new columns added
to atlas_experiments (computational_engine, test_conditions,
revival_hypotheses, regime_state_at_test) are correctly stored
in data/praxis_meta.db and correctly populated for Exps 1 and 13
by Cycle 33's atlas_sync run -- but the praxis_mcp server's
atlas_get and atlas_search tools hand-code their SELECT
statements and weren't extended to include the new columns.

Two functional changes to servers/praxis_mcp/tools/atlas.py:

1. atlas_get: SELECT now includes the 4 new columns. The 3
   JSON-typed columns (test_conditions, revival_hypotheses,
   regime_state_at_test) are decoded back to dicts/lists before
   return so MCP consumers receive structured objects rather
   than raw JSON strings. Defensive parsing: NULL stays NULL;
   JSON decode failure leaves the raw string and adds a
   _parse_error note rather than breaking the tool.

2. atlas_search: per-result dicts now include
   computational_engine (integer or null). The bigger JSON
   fields stay out of search results -- search payloads remain
   lightweight; full structured navigation goes through
   atlas_get.

Docstrings updated. No DB or parser changes. Backward-compatible:
all existing fields still returned with same names and types;
new fields are additive. MCP server restart (Claude Desktop
quit/relaunch) required to pick up the change, same pattern as
Cycle 27.5.
```

## Post-cycle: status

After Cycle 33.5 lands and MCP restart, the Atlas DB schema
extension chain is fully usable end-to-end:

- [x] DB schema extended (Cycle 33)
- [x] Parser populates the new columns (Cycle 33)
- [x] 2 reference experiments backfilled (Cycle 33)
- [x] MCP tools expose the new columns (Cycle 33.5)
- [ ] Mass backfill of remaining 13 experiments (Cycle 35)
- [ ] `atlas_search` engine filter (deferred TODO after Cycle 35)

Cycle 34 (Info Bars v0.1) is the next *substantive* cycle after
this small fix.
