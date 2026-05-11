# Retro: Cycle 33.5 -- MCP atlas tools serialization fix

**Brief:** `claude/handoffs/BRIEF_atlas_mcp_serialization.md`
**Date:** 2026-05-11
**Mode:** Hybrid (Claude drafted brief; Code applied the small
edit; user did MCP restart + live verification)
**Status:** DONE
**Predecessor:** Cycle 33 (`6ab289d`, atlas schema extension)
**Commit:** `522d22d`

---

## Summary

Closes the MCP serialization gap from Cycle 33. The 4 new
columns added in Cycle 33 are now exposed by the
`praxis:atlas_get` and (for `computational_engine` only)
`praxis:atlas_search` tools. Single-file edit to
`servers/praxis_mcp/tools/atlas.py` adding the columns to the
SELECT statements and decoding the 3 JSON-typed columns back
to structured objects for MCP consumers.

Net change: +33/-4 lines in
`servers/praxis_mcp/tools/atlas.py` (per `git show 522d22d --stat`).

---

## Why this came up

Cycle 33's brief asserted that the MCP tools "automatically
expose the new columns since they return full records." That
assumption was wrong: the atlas tools cherry-pick fields with
hand-coded SELECT statements rather than `SELECT *`. The DB had
the data correctly (Code verified directly with sqlite3 in
Cycle 33), but MCP consumers couldn't access the structured
fields via the live tool -- only via `full_markdown` (text) or
`raw_query` (DB-direct).

The brief drafting error was Claude's, not Code's. Memory entry
#15 added post-Cycle-33 to guard against repeating it: read
`servers/*/tools/*.py` source before asserting that DB changes
auto-flow to MCP tools.

---

## Execution log

### Step 1: Edit `servers/praxis_mcp/tools/atlas.py`

Two changes:
- `atlas_search`: SELECT extended with `x.computational_engine`;
  per-result dict gained the same key.
- `atlas_get`: SELECT extended with all 4 new columns. After
  the existing `{k: row[k] for k in row.keys()}` line,
  defensive `json.loads()` decoding of the 3 JSON columns
  (test_conditions, revival_hypotheses, regime_state_at_test)
  wrapped in try/except so JSON parse failures fall back to
  raw string + `_parse_error` note.

Docstrings updated to enumerate the new return fields.

### Step 2: py_compile validation

```powershell
python -m py_compile servers/praxis_mcp/tools/atlas.py
```

Output: clean.

### Step 3: Local sqlite validation (per brief)

```python
# (see brief Section "Validation" for the full snippet)
```

Output (verbatim from the local run):

```
id=1 OK ce=1 tc_keys=8 rh_len=4 rs_keys=7
id=11 OK ce=7 tc_keys=8 rh_len=4 rs_keys=5
id=2 OK ce=None tc=NULL rh=NULL rs=NULL
```

Confirmed Exp 1's `computational_engine = 1`, `test_conditions`
parses to 8-key dict, `revival_hypotheses` parses to 4-item list,
`regime_state_at_test` parses to 7-key dict. Exp 11 (Funding Carry,
Exp 13 in markdown) has `computational_engine = 7` and a 5-key
`regime_state_at_test`. Exp 2 (TA crypto, not backfilled) has all
4 new columns NULL.

### Step 4: Commit + push

Committed as `522d22d` and pushed.

### Step 5: MCP server restart (by user)

Claude Desktop quit/relaunch, same procedure as Cycle 27.5
(MCP servers don't hot-reload code changes; restart required
to pick up the new SELECT statements).

### Step 6: Live MCP verification (by user)

- `praxis:atlas_get(1)`: All 4 new fields populated.
  `computational_engine = 1`. `test_conditions` decoded to 8-key
  dict (`bar_type`, `frequency`, `universe`, `tc`, `feature_set`,
  `prefilter`, `risk_management`, `computational_engine`).
  `revival_hypotheses` decoded to 4-item list of
  `{title, likelihood, description}` dicts.
  `regime_state_at_test` decoded to 7-key dict
  (`A_trend`, `D_serial_correlation`, `G_liquidity`, `I_volume`,
  `E_microstructure`, `b,_c,_f,_h,_j,_k`, `full_additive`).
- `praxis:atlas_get(11)`: All 4 new fields populated.
  `computational_engine = 7`. `test_conditions` 8-key dict.
  `revival_hypotheses` 4-item list. `regime_state_at_test`
  5-key dict.
- `praxis:atlas_get(2)`: All 4 new fields present as `null`
  (Exp 2 not backfilled). Defensive None-handling working
  correctly -- no spurious `_parse_error` keys.
- `praxis:atlas_search("MOMENTUM crypto", top_k=3)`: returns
  ids 15, 7, 6 (= Exps 17, 9, 8) with similarity_scores
  0.5349 / 0.4925 / 0.4547 -- identical to Cycle 32 baseline.
  Each result dict includes a `computational_engine` key
  (all `null` since non-backfilled). Embeddings stable for
  non-backfilled entries as expected.

---

## Notes

### Why JSON decoding on the server side

Two reasonable options:
- (A) Pass JSON columns through as raw strings; let consumers
  `json.loads()` themselves.
- (B) Decode on the server side; consumers receive structured
  dicts/lists.

Chose (B) because:
1. The storage-format-as-string is an implementation detail
   consumers shouldn't have to know about.
2. The Research Agent (the primary intended consumer) shouldn't
   have boilerplate around every Atlas read.
3. The defensive try/except costs ~5 lines of code and
   guarantees the tool never breaks if a future migration
   introduces malformed JSON.

### Why `computational_engine` in `atlas_search` but not the JSON fields

`atlas_search` results are intentionally lightweight summaries
for triage; full detail comes from `atlas_get`. The single
integer engine column is cheap to include and immediately
useful for navigation ("show me all Engine 7 hits in this
search"). The JSON fields can be 1-5 KB each per row and would
bloat search payloads significantly when most consumers only
care about the headline result_summary.

### `_parse_error` annotation, not exception

If `test_conditions` stored value fails JSON decoding (which
shouldn't happen given the parser writes via `json.dumps()`,
but might if a future migration corrupts a row), the tool
silently substitutes a `_parse_error` key into the response
and returns the raw string. This is preferable to raising
because (a) the rest of the entry is still useful, (b) the
error is visible enough for debugging, (c) the MCP layer
shouldn't be the place where data integrity issues hard-fail.

### Cosmetic: Exp 13's `regime_state_at_test` key shapes

Surfaced during MCP verification: Exp 13's
`regime_state_at_test` keys came out in the parser's
sanitized-fallback form (e.g.
`"f_=_+1,_+2_(positive_funding_sustained)"`) because that
experiment's regime bullets used compound state notation
`F = +1, +2` instead of the canonical `F (Funding)` form. The
parser's fallback path correctly produced valid dict keys --
the dict is fully usable -- but they're cosmetically rough
compared to Exp 1's clean `A_trend`, `G_liquidity`, etc.

Two options for Cycle 35 mass backfill:
1. Tighten the markdown template to enforce canonical regime
   labels (`F (Funding)`, then describe the state separately).
2. Leave the fallback path as-is for authoring flexibility.

Not blocking; flagged here as input to the Cycle 35 design
conversation.

---

## Open items / next cycle inputs

- **Cycle 34**: Info Bars v0.1 (dollar bars, volume bars,
  volume-imbalance bars per Lopez AFML Ch. 2). Independent of
  this work.
- **Cycle 35**: Atlas mass backfill of remaining 13
  experiments. Now genuinely useful because the structured
  fields are exposed end-to-end.
- **`atlas_search` `engine` filter** (deferred TODO): once
  `computational_engine` is populated for all experiments,
  add a server-side filter parameter so search can be scoped
  to a specific engine. Not in scope here.
