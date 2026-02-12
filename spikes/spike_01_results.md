# Spike 1 Results: vt2_ View Performance

**Date:** 2026-02-12  
**Verdict:** ✅ PASS  
**Decision D1:** PROCEED as spec'd (Option A — derived views, no materialization needed)

## Performance Results (100K rows: 10,000 securities × 10 versions)

| Test | Median | Threshold | Margin | Verdict |
|------|--------|-----------|--------|---------|
| vt2 AS-IS: single security | 9.9ms | 50ms | **5.1×** | ✅ |
| vt2 AS-WAS: single security | 9.6ms | 50ms | **5.2×** | ✅ |
| vt2 AS-IS: 1,000-security batch | 96.7ms | 2,000ms | **20.7×** | ✅ |
| vt2 AS-WAS: 1,000-security batch | 104.4ms | 2,000ms | **19.2×** | ✅ |
| vt2 AS-IS: full scan | 50.3ms | 10,000ms | **199×** | ✅ |
| vt2 AS-WAS: full scan | 50.2ms | 10,000ms | **199×** | ✅ |
| vt2: full materialization | 49.9ms | 10,000ms | **200×** | ✅ |

## Correctness Validation

- ✅ vew_security: 10,000 rows (one per security)
- ✅ vt2_security: 86,858 rows (bucketing correctly removed 13,142 same-day dupes)
- ✅ All start_date <= end_date (0 invalid ranges)
- ✅ No overlapping date ranges (0 overlaps)
- ✅ All 10,000 securities have sentinel end_date (9999-12-31)
- ✅ AS-IS vs AS-WAS correctly return different versions

## Key Findings

### 1. xxHash64 uint64 → int64 Conversion Required
`xxhash.xxh64().intdigest()` returns unsigned 64-bit values (0 to 2^64-1), but DuckDB's `BIGINT` is signed (-2^63 to 2^63-1). Must reinterpret unsigned as signed:
```python
unsigned = xxhash.xxh64(bpk.encode()).intdigest()
if unsigned >= (1 << 63):
    return unsigned - (1 << 64)
return unsigned
```
**Action:** Bake this into the production `generate_base_id()` utility.

### 2. DuckDB Indexes Are Minimal Impact
Indexes on `security_base_id` and `(security_base_id, security_hist_id)` made <5% difference. DuckDB's columnar storage and in-memory execution already handle these patterns well at this scale. Indexes may matter more at 1M+ rows.

### 3. Materialized Views: Available But Not Needed
Materialization takes 129ms and yields 5-50× speedup over derived views. At current scale, derived views are well within thresholds. Materialization is a valid fallback for future scale — note it takes only 129ms to fully refresh.

### 4. Same-Day Bucketing Works Correctly
With 15% probability of same-day updates in test data, the `ROW_NUMBER() OVER (PARTITION BY base_id, CAST(hist_id AS DATE) ORDER BY hist_id DESC)` pattern correctly keeps only the latest intra-day record. 13,142 duplicates removed from 100,000 rows.

### 5. AS-IS/AS-WAS Query Pattern Confirmed
Both query patterns work correctly on the same view. The only difference is the date used in the `BETWEEN start_date AND end_date` filter — `CURRENT_DATE` for AS-IS, historical date for AS-WAS.

## Scale Projections

At 5.1× margin on single-security and 20× on batch, the view-based approach should handle:
- ~50K securities × 10 versions (500K rows) before approaching single-security threshold
- ~200K securities × 10 versions (2M rows) before batch approaches threshold

For Phase 4+ (live trading, millions of rows), materialized views with periodic refresh become the optimization path.

## Next Spike

Spike 2: DuckDB STRUCT Handling — validate nested model definition schemas.
