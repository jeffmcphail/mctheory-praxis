# McTheory Praxis — `praxis_main` Series Transcript

> **Purpose:** Chronological log of key decisions, discoveries, and milestones across all `praxis_main` chats.

---

## praxis_main_current (2026-02-12 — present)

### Session: Phase 0 Spikes + Phase 1.1

**Context:** First implementation chat. Spec v9.3.1 and execution plan v1.2 frozen. Reference files (dataUtilities.py, statsUtilities.py, main.py) clarified as domain logic references only — NOT code to port.

**Key Decisions:**

1. **[DECISION] Reference files are NOT code to port.** They inform what models do and how math works. All implementations are fresh builds in Polars/modern patterns. Validated against reference outputs, not translated line-by-line.

2. **[SPIKE 1 — PASS] vt2_ View Performance validated.** 100K rows, all queries 5-200× under thresholds. Temporal view architecture proceeds as spec'd.
   - **[FINDING]** xxHash64 `intdigest()` returns uint64 but DuckDB BIGINT is int64. Must convert: `if unsigned >= (1 << 63): return unsigned - (1 << 64)`. Baked into production `generate_base_id()`.
   - **[FINDING]** DuckDB indexes made <5% difference at 100K scale. Columnar storage handles it. Indexes may matter at 1M+.
   - **[FINDING]** Materialized views take 129ms to refresh and yield 5-50× speedup. Available fallback but not needed at current scale.

3. **[SPIKE 2 — PASS] DuckDB STRUCT Handling validated.** 18/18 tests, max 87ms. Full model definition schema works.
   - **[FINDING]** `values` is a reserved word in DuckDB. Cannot use as UNNEST column alias. Use LATERAL subquery pattern.
   - **[FINDING]** Parameterized `?::JSON` binding is slow (~700ms). Inline `'...'::JSON` casting is fast (~41ms). Always use inline.
   - **[FINDING]** INSERT-AS-SELECT for STRUCT versioning is verbose (must spell all fields) but correct. Python helpers should generate this SQL.
   - **[FINDING]** UNNEST of array-of-STRUCT needs LATERAL pattern, not simple `AS alias(cols)`.

4. **[DELIVERED] Phase 1.1: _bpk/_base_id infrastructure.** 63 tests passing.
   - `generate_base_id()` with signed conversion
   - `generate_hist_id()` with UTC enforcement
   - `validate_bpk()` with entity-specific format rules
   - `EntityKeys` immutable key triple (create, new_version, to_dict, to_tuple)
   - `build_security_bpk()` with SecIdType hierarchy walk
   - DuckDB round-trip integration tests including negative base_ids

5. **[NEXT] Phase 1.2: Diagnostic Logger** — singleton, tag system, routing matrix, terminal + file + database adapters. Must exist before any other code writes a log statement.

---

*v1.0 — Created 2026-02-12 (Chat: praxis_main_current)*
