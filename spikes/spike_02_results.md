# Spike 2 Results: DuckDB STRUCT Handling

**Date:** 2026-02-12  
**Verdict:** ✅ PASS (18/18 tests, max 87.1ms < 100ms threshold)  
**Decision D2:** PROCEED with full STRUCTs (Option A — native DuckDB STRUCTs, not JSON)

## Results Summary

| Test | Time | Verdict |
|------|------|---------|
| Create full schema (10 STRUCT columns, 4+ nesting levels) | 3.9ms | ✅ |
| Insert Chan CPO (complex, all fields populated) | 66.4ms | ✅ |
| Insert SMA (simple, mostly NULLs) | 62.1ms | ✅ |
| Top-level STRUCT field query | 1.9ms | ✅ |
| Nested STRUCT field filter | 1.8ms | ✅ |
| Array-in-STRUCT (list_contains) | 4.1ms | ✅ |
| Array-of-STRUCT indexed access [1].param_name | 1.7ms | ✅ |
| Array-of-STRUCT [2].values | 2.1ms | ✅ |
| 4-level nesting (risk.stress_tests.custom_shocks[1].factor) | 1.8ms | ✅ |
| NULL STRUCT handling (IS NULL) | 1.2ms | ✅ |
| NULL nested STRUCT field | 3.4ms | ✅ |
| Array-of-STRUCT with nested params (.params.windows) | 4.2ms | ✅ |
| Multi-STRUCT path query (4 paths) | 6.1ms | ✅ |
| Version update (INSERT-AS-SELECT) | 87.1ms | ✅ |
| JSON escape hatch (round-trip) | 41.0ms | ✅ |
| UNNEST array-of-STRUCT | 7.0ms | ✅ |
| Nested boolean query | 1.7ms | ✅ |

## Key Findings

### 1. UNNEST Syntax for Array-of-STRUCT
DuckDB's UNNEST of struct arrays does NOT support the simple `UNNEST(col) AS alias(field1, field2)` form for all column names. In particular, `values` is a reserved word. The working pattern uses LATERAL subquery:
```sql
SELECT u.param_name, u.vals
FROM (
    SELECT UNNEST(cpo_params.parameter_grid) AS grid_entry
    FROM fact_model_definition WHERE ...
) t,
LATERAL (SELECT t.grid_entry.param_name AS param_name, 
                t.grid_entry.values AS vals) u
```
**Action:** Document this pattern for all array-of-STRUCT access in Praxis.

### 2. JSON Parameter Binding Is SLOW
Using `?::JSON` parameterized binding: ~700ms. Using inline `'...'::JSON`: ~41ms. The parameterized path hits a slow code path in DuckDB's type coercion.
**Action:** Always use inline JSON casting for inserts. Build JSON string in Python, embed in SQL.

### 3. INSERT-AS-SELECT for Versioning Works But Is Verbose
To create a new version with one modified STRUCT field, you must spell out ALL fields of that STRUCT in the INSERT-AS-SELECT. This is verbose but correct.
**Action:** Build Python helpers that generate the SQL for STRUCT version updates. The Pydantic model → SQL STRUCT bridge should handle this automatically.

### 4. INSERT Time ~60ms for Complex STRUCTs
Both the Chan CPO (all fields) and SMA (mostly NULLs) inserts take ~60ms. This is the DuckDB STRUCT type validation cost — acceptable for model definition inserts (rare operations, not hot path).

### 5. All Query Patterns < 10ms
Once data is inserted, all query patterns (filtering, nested access, array indexing, NULL checks, multi-path) are well under 10ms. STRUCT queries are fast reads.

## Confirmed Capabilities

| Capability | Status |
|-----------|--------|
| 4+ levels of STRUCT nesting | ✅ Works |
| Array-of-STRUCT with indexed access | ✅ Works (1-based indexing) |
| UNNEST for array-of-STRUCT iteration | ✅ Works (with LATERAL pattern) |
| NULL at any nesting level | ✅ Works (IS NULL, nested NULL) |
| JSON escape hatch for arbitrary data | ✅ Works (inline casting) |
| INSERT-AS-SELECT for versioning | ✅ Works (verbose but functional) |
| Multi-STRUCT path filtering | ✅ Works |
| list_contains on arrays inside STRUCTs | ✅ Works |

## Next Spike

Spike 3: Workflow Executor Library Fit (Prefect/Dagster) — Week 3-4, not blocking.

Phase 1 implementation can begin: _bpk/_base_id infrastructure + DuckDB init.
