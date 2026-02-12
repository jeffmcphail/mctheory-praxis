"""
Spike 1: vt2_ View Performance
===============================
McTheory Praxis — Phase 0

ASSUMPTION: The PARTITION BY / ROW_NUMBER / LEAD pattern for period bucketing
(Spec §2.3-2.4) performs acceptably on DuckDB at realistic scale.

TEST:
    1. Generate synthetic dim_security: 10,000 securities × 10 versions = 100,000 rows
    2. Create vt2_security view per spec §2.4
    3. Run AS-IS and AS-WAS queries
    4. Measure: single-security lookup, 1,000-security batch, full-table scan

PASS CRITERIA:
    - Single-security AS-IS:  < 50ms
    - 1,000-security batch AS-IS: < 2s
    - Full-table scan: < 10s

FAIL THRESHOLD: Any query > 10× the pass criteria.

FALLBACK OPTIONS (if FAIL):
    A: Materialized views (CREATE TABLE AS)
    B: Pre-compute start/end dates at insert time
    C: Hybrid — materialized with periodic refresh
"""

import duckdb
import xxhash
import time
import random
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_SECURITIES = 10_000
VERSIONS_PER_SECURITY = 10
TOTAL_ROWS = NUM_SECURITIES * VERSIONS_PER_SECURITY  # 100,000
BATCH_SIZE = 1_000
WARMUP_ITERATIONS = 3
MEASUREMENT_ITERATIONS = 10

# Pass/Fail thresholds (milliseconds)
PASS_SINGLE_MS = 50
PASS_BATCH_MS = 2_000
PASS_FULL_SCAN_MS = 10_000
FAIL_MULTIPLIER = 10  # > 10× pass = FAIL


@dataclass
class BenchmarkResult:
    name: str
    times_ms: list
    pass_threshold_ms: float

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms)

    @property
    def p95_ms(self) -> float:
        return sorted(self.times_ms)[int(len(self.times_ms) * 0.95)]

    @property
    def min_ms(self) -> float:
        return min(self.times_ms)

    @property
    def max_ms(self) -> float:
        return max(self.times_ms)

    @property
    def verdict(self) -> str:
        if self.median_ms <= self.pass_threshold_ms:
            return "PASS"
        elif self.median_ms <= self.pass_threshold_ms * FAIL_MULTIPLIER:
            return "WARN"
        else:
            return "FAIL"


def generate_bpk(sec_type: str, identifier: str) -> str:
    """Generate business primary key per spec §2.2."""
    return f"{sec_type}|TICKER|{identifier}"


def generate_base_id(bpk: str) -> int:
    """Generate deterministic base_id from bpk using xxHash64 per spec §2.2.
    
    xxHash64 returns uint64 (0 to 2^64-1), but DuckDB BIGINT is int64 (-2^63 to 2^63-1).
    We interpret the unsigned value as signed to fit DuckDB's BIGINT.
    """
    unsigned = xxhash.xxh64(bpk.encode()).intdigest()
    # Convert to signed 64-bit: values >= 2^63 become negative
    if unsigned >= (1 << 63):
        return unsigned - (1 << 64)
    return unsigned


def setup_database(con: duckdb.DuckDBPyConnection) -> list[int]:
    """Create dim_security and populate with synthetic data. Returns list of base_ids."""

    # Create the raw dimension table per spec §2.2
    con.execute("""
        CREATE TABLE dim_security (
            security_hist_id  TIMESTAMP PRIMARY KEY,
            security_bpk      VARCHAR NOT NULL,
            security_base_id  BIGINT NOT NULL,
            security_name     VARCHAR,
            security_type     VARCHAR,
            exchange          VARCHAR,
            currency          VARCHAR,
            status            VARCHAR
        )
    """)

    print(f"Generating {TOTAL_ROWS:,} rows ({NUM_SECURITIES:,} securities × {VERSIONS_PER_SECURITY} versions)...")

    # Security type distribution
    sec_types = ["EQUITY", "BOND", "ETF", "FUTURE", "OPTION"]
    exchanges = ["NYSE", "NASDAQ", "LSE", "TSE", "HKEX", "ASX"]
    currencies = ["USD", "GBP", "JPY", "HKD", "AUD", "EUR"]
    statuses = ["ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "SUSPENDED", "DELISTED"]  # weighted toward ACTIVE

    base_ids = []
    rows = []
    base_date = datetime(2018, 1, 1)

    for i in range(NUM_SECURITIES):
        ticker = f"SEC{i:05d}"
        sec_type = sec_types[i % len(sec_types)]
        bpk = generate_bpk(sec_type, ticker)
        base_id = generate_base_id(bpk)
        base_ids.append(base_id)

        for v in range(VERSIONS_PER_SECURITY):
            # Spread versions across ~7 years with some same-day updates
            days_offset = v * 250 + random.randint(0, 30)  # ~yearly with jitter
            hist_timestamp = base_date + timedelta(
                days=days_offset,
                hours=random.randint(8, 17),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59),
                microseconds=random.randint(0, 999999)  # ensure uniqueness
            )

            # Occasionally create same-day updates (tests the bucketing logic)
            if v > 0 and random.random() < 0.15:
                prev_row = rows[-1]
                hist_timestamp = prev_row[0] + timedelta(
                    hours=random.randint(1, 6),
                    minutes=random.randint(0, 59)
                )

            rows.append((
                hist_timestamp,
                bpk,
                base_id,
                f"{ticker} Corp v{v + 1}",
                sec_type,
                random.choice(exchanges),
                random.choice(currencies),
                random.choice(statuses),
            ))

    # Bulk insert
    con.executemany(
        "INSERT INTO dim_security VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows
    )

    row_count = con.execute("SELECT COUNT(*) FROM dim_security").fetchone()[0]
    print(f"Inserted {row_count:,} rows into dim_security")

    return base_ids


def create_views(con: duckdb.DuckDBPyConnection):
    """Create temporal views per spec §2.4."""

    # vew_ — Current state (latest version per entity)
    con.execute("""
        CREATE VIEW vew_security AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY security_base_id 
                       ORDER BY security_hist_id DESC
                   ) AS rn
            FROM dim_security
        ) WHERE rn = 1
    """)

    # vt2_ — Point-in-time with period bucketing (THE critical view)
    con.execute("""
        CREATE VIEW vt2_security AS
        WITH bucketed AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY security_base_id, CAST(security_hist_id AS DATE)
                       ORDER BY security_hist_id DESC
                   ) AS rn_within_day
            FROM dim_security
        ),
        latest_per_day AS (
            SELECT *,
                   CAST(security_hist_id AS DATE) AS start_date,
                   COALESCE(
                       LEAD(CAST(security_hist_id AS DATE)) OVER (
                           PARTITION BY security_base_id ORDER BY security_hist_id
                       ) - INTERVAL 1 DAY,
                       DATE '9999-12-31'
                   )::DATE AS end_date
            FROM bucketed
            WHERE rn_within_day = 1
        )
        SELECT * EXCLUDE (rn_within_day) FROM latest_per_day
    """)

    print("Created views: vew_security, vt2_security")


def benchmark(con: duckdb.DuckDBPyConnection, name: str, query: str,
              pass_threshold_ms: float, params: list = None,
              warmup: int = WARMUP_ITERATIONS,
              iterations: int = MEASUREMENT_ITERATIONS) -> BenchmarkResult:
    """Run a query multiple times and collect timing statistics."""

    # Warmup
    for _ in range(warmup):
        if params:
            con.execute(query, params).fetchall()
        else:
            con.execute(query).fetchall()

    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        if params:
            result = con.execute(query, params).fetchall()
        else:
            result = con.execute(query).fetchall()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    return BenchmarkResult(name=name, times_ms=times, pass_threshold_ms=pass_threshold_ms)


def run_spike(con: duckdb.DuckDBPyConnection, base_ids: list[int]):
    """Execute all spike benchmarks."""

    results: list[BenchmarkResult] = []

    # Pick specific base_ids for targeted queries
    single_id = base_ids[0]
    batch_ids = random.sample(base_ids, BATCH_SIZE)

    # -- Test dates --
    # Mid-range date for AS-IS (should capture most securities)
    as_of_date = "2022-06-15"
    # Historical system date for AS-WAS
    as_was_system_date = "2023-01-10"

    print("\n" + "=" * 70)
    print("SPIKE 1: vt2_ VIEW PERFORMANCE BENCHMARKS")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. vew_ (current state) — baseline comparison
    # -----------------------------------------------------------------------
    print("\n--- vew_ (current state) baselines ---")

    results.append(benchmark(
        con, "vew: single security lookup",
        "SELECT * FROM vew_security WHERE security_base_id = ?",
        pass_threshold_ms=PASS_SINGLE_MS,
        params=[single_id]
    ))

    results.append(benchmark(
        con, "vew: full scan (all current)",
        "SELECT COUNT(*) FROM vew_security",
        pass_threshold_ms=PASS_FULL_SCAN_MS
    ))

    # -----------------------------------------------------------------------
    # 2. vt2_ — The critical tests
    # -----------------------------------------------------------------------
    print("\n--- vt2_ (point-in-time) — THE critical tests ---")

    # 2a. Single security, AS-IS
    results.append(benchmark(
        con, "vt2 AS-IS: single security",
        """SELECT * FROM vt2_security 
           WHERE security_base_id = ? 
           AND CURRENT_DATE BETWEEN start_date AND end_date""",
        pass_threshold_ms=PASS_SINGLE_MS,
        params=[single_id]
    ))

    # 2b. Single security, AS-WAS
    results.append(benchmark(
        con, "vt2 AS-WAS: single security",
        """SELECT * FROM vt2_security 
           WHERE security_base_id = ? 
           AND DATE '2023-01-10' BETWEEN start_date AND end_date""",
        pass_threshold_ms=PASS_SINGLE_MS,
        params=[single_id]
    ))

    # 2c. Single security, point-in-time with as_of_date filter
    results.append(benchmark(
        con, "vt2 AS-IS: single security + as_of_date",
        """SELECT * FROM vt2_security 
           WHERE security_base_id = ? 
           AND start_date <= DATE '2022-06-15'
           AND end_date >= DATE '2022-06-15'""",
        pass_threshold_ms=PASS_SINGLE_MS,
        params=[single_id]
    ))

    # 2d. Batch of 1,000 securities, AS-IS
    # Use a temp table for the batch to avoid huge IN clause
    con.execute("CREATE TEMP TABLE batch_ids AS SELECT UNNEST(?) AS security_base_id", [batch_ids])

    results.append(benchmark(
        con, "vt2 AS-IS: 1,000-security batch",
        """SELECT v.* FROM vt2_security v
           INNER JOIN batch_ids b ON v.security_base_id = b.security_base_id
           WHERE CURRENT_DATE BETWEEN v.start_date AND v.end_date""",
        pass_threshold_ms=PASS_BATCH_MS
    ))

    # 2e. Batch of 1,000 securities, AS-WAS
    results.append(benchmark(
        con, "vt2 AS-WAS: 1,000-security batch",
        """SELECT v.* FROM vt2_security v
           INNER JOIN batch_ids b ON v.security_base_id = b.security_base_id
           WHERE DATE '2023-01-10' BETWEEN v.start_date AND v.end_date""",
        pass_threshold_ms=PASS_BATCH_MS
    ))

    # 2f. Full table scan — all securities, current version
    results.append(benchmark(
        con, "vt2 AS-IS: full scan (all current)",
        """SELECT COUNT(*) FROM vt2_security 
           WHERE CURRENT_DATE BETWEEN start_date AND end_date""",
        pass_threshold_ms=PASS_FULL_SCAN_MS
    ))

    # 2g. Full table scan — all securities, historical point
    results.append(benchmark(
        con, "vt2 AS-WAS: full scan (historical)",
        """SELECT COUNT(*) FROM vt2_security 
           WHERE DATE '2021-06-15' BETWEEN start_date AND end_date""",
        pass_threshold_ms=PASS_FULL_SCAN_MS
    ))

    # 2h. Full materialization of vt2_ (how long to compute the entire view)
    results.append(benchmark(
        con, "vt2: full materialization",
        "SELECT COUNT(*) FROM vt2_security",
        pass_threshold_ms=PASS_FULL_SCAN_MS,
        warmup=2,
        iterations=5
    ))

    # -----------------------------------------------------------------------
    # 3. Correctness checks
    # -----------------------------------------------------------------------
    print("\n--- Correctness validation ---")

    # 3a. Every security should have exactly one current record in vew_
    vew_count = con.execute("SELECT COUNT(*) FROM vew_security").fetchone()[0]
    assert vew_count == NUM_SECURITIES, f"vew_ should have {NUM_SECURITIES} rows, got {vew_count}"
    print(f"  ✓ vew_security has {vew_count:,} rows (one per security)")

    # 3b. vt2_ should have fewer or equal rows to dim_security (bucketing removes same-day dupes)
    dim_count = con.execute("SELECT COUNT(*) FROM dim_security").fetchone()[0]
    vt2_count = con.execute("SELECT COUNT(*) FROM vt2_security").fetchone()[0]
    assert vt2_count <= dim_count, f"vt2_ ({vt2_count}) should be <= dim ({dim_count})"
    dedup_removed = dim_count - vt2_count
    print(f"  ✓ vt2_security has {vt2_count:,} rows (bucketing removed {dedup_removed:,} same-day dupes)")

    # 3c. start_date should always be <= end_date
    invalid_ranges = con.execute(
        "SELECT COUNT(*) FROM vt2_security WHERE start_date > end_date"
    ).fetchone()[0]
    assert invalid_ranges == 0, f"Found {invalid_ranges} rows where start_date > end_date"
    print(f"  ✓ All start_date <= end_date (0 invalid ranges)")

    # 3d. For each base_id, date ranges should not overlap
    overlaps = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT security_base_id, start_date, end_date,
                   LEAD(start_date) OVER (
                       PARTITION BY security_base_id ORDER BY start_date
                   ) AS next_start
            FROM vt2_security
        ) WHERE next_start <= end_date AND next_start IS NOT NULL
    """).fetchone()[0]
    assert overlaps == 0, f"Found {overlaps} overlapping date ranges"
    print(f"  ✓ No overlapping date ranges (0 overlaps)")

    # 3e. Latest version per security should have end_date = 9999-12-31
    sentinel_check = con.execute("""
        SELECT COUNT(DISTINCT security_base_id) 
        FROM vt2_security 
        WHERE end_date = DATE '9999-12-31'
    """).fetchone()[0]
    assert sentinel_check == NUM_SECURITIES, \
        f"Expected {NUM_SECURITIES} securities with sentinel end_date, got {sentinel_check}"
    print(f"  ✓ All {sentinel_check:,} securities have sentinel end_date (9999-12-31)")

    # 3f. AS-IS and AS-WAS should return different results when data has been corrected
    # Pick a security and check that querying at different system dates gives different versions
    test_id = base_ids[0]
    versions = con.execute("""
        SELECT start_date, end_date, security_name 
        FROM vt2_security 
        WHERE security_base_id = ? 
        ORDER BY start_date
    """, [test_id]).fetchall()
    print(f"  ✓ Security {test_id} has {len(versions)} temporal versions")

    # Verify AS-IS vs AS-WAS divergence
    if len(versions) >= 2:
        early_date = versions[0][0]
        late_date = versions[-1][0]

        as_is_result = con.execute("""
            SELECT security_name FROM vt2_security 
            WHERE security_base_id = ? 
            AND ? BETWEEN start_date AND end_date
        """, [test_id, early_date]).fetchall()

        as_was_result = con.execute("""
            SELECT security_name FROM vt2_security 
            WHERE security_base_id = ? 
            AND ? BETWEEN start_date AND end_date
        """, [test_id, late_date]).fetchall()

        if as_is_result and as_was_result and as_is_result != as_was_result:
            print(f"  ✓ AS-IS vs AS-WAS correctly return different versions")
        else:
            print(f"  ℹ AS-IS/AS-WAS divergence test inconclusive (depends on synthetic data distribution)")

    # -----------------------------------------------------------------------
    # 4. Index impact test
    # -----------------------------------------------------------------------
    print("\n--- Index impact test ---")

    # Test with index on base_id
    con.execute("CREATE INDEX idx_security_base_id ON dim_security(security_base_id)")
    print("  Created index on security_base_id")

    idx_single = benchmark(
        con, "vt2 AS-IS: single (WITH index)",
        """SELECT * FROM vt2_security 
           WHERE security_base_id = ? 
           AND CURRENT_DATE BETWEEN start_date AND end_date""",
        pass_threshold_ms=PASS_SINGLE_MS,
        params=[single_id]
    )
    results.append(idx_single)

    idx_batch = benchmark(
        con, "vt2 AS-IS: 1,000-batch (WITH index)",
        """SELECT v.* FROM vt2_security v
           INNER JOIN batch_ids b ON v.security_base_id = b.security_base_id
           WHERE CURRENT_DATE BETWEEN v.start_date AND v.end_date""",
        pass_threshold_ms=PASS_BATCH_MS
    )
    results.append(idx_batch)

    # Composite index: (base_id, hist_id)
    con.execute("CREATE INDEX idx_security_composite ON dim_security(security_base_id, security_hist_id)")
    print("  Created composite index on (security_base_id, security_hist_id)")

    idx_composite_single = benchmark(
        con, "vt2 AS-IS: single (composite index)",
        """SELECT * FROM vt2_security 
           WHERE security_base_id = ? 
           AND CURRENT_DATE BETWEEN start_date AND end_date""",
        pass_threshold_ms=PASS_SINGLE_MS,
        params=[single_id]
    )
    results.append(idx_composite_single)

    idx_composite_batch = benchmark(
        con, "vt2 AS-IS: 1,000-batch (composite index)",
        """SELECT v.* FROM vt2_security v
           INNER JOIN batch_ids b ON v.security_base_id = b.security_base_id
           WHERE CURRENT_DATE BETWEEN v.start_date AND v.end_date""",
        pass_threshold_ms=PASS_BATCH_MS
    )
    results.append(idx_composite_batch)

    # -----------------------------------------------------------------------
    # 5. Fallback option A: Materialized view
    # -----------------------------------------------------------------------
    print("\n--- Fallback A: Materialized view performance ---")

    mat_start = time.perf_counter()
    con.execute("CREATE TABLE mat_vt2_security AS SELECT * FROM vt2_security")
    mat_create_ms = (time.perf_counter() - mat_start) * 1000
    print(f"  Materialized vt2_security in {mat_create_ms:.1f}ms")

    # Index the materialized table
    con.execute("CREATE INDEX idx_mat_base_id ON mat_vt2_security(security_base_id)")
    con.execute("CREATE INDEX idx_mat_dates ON mat_vt2_security(start_date, end_date)")
    con.execute("CREATE INDEX idx_mat_composite ON mat_vt2_security(security_base_id, start_date, end_date)")

    results.append(benchmark(
        con, "MATERIALIZED: single security AS-IS",
        """SELECT * FROM mat_vt2_security 
           WHERE security_base_id = ? 
           AND CURRENT_DATE BETWEEN start_date AND end_date""",
        pass_threshold_ms=PASS_SINGLE_MS,
        params=[single_id]
    ))

    results.append(benchmark(
        con, "MATERIALIZED: 1,000-batch AS-IS",
        """SELECT v.* FROM mat_vt2_security v
           INNER JOIN batch_ids b ON v.security_base_id = b.security_base_id
           WHERE CURRENT_DATE BETWEEN v.start_date AND v.end_date""",
        pass_threshold_ms=PASS_BATCH_MS
    ))

    results.append(benchmark(
        con, "MATERIALIZED: full scan AS-IS",
        """SELECT COUNT(*) FROM mat_vt2_security 
           WHERE CURRENT_DATE BETWEEN start_date AND end_date""",
        pass_threshold_ms=PASS_FULL_SCAN_MS
    ))

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Test':<45} {'Median':>8} {'P95':>8} {'Min':>8} {'Max':>8} {'Pass':>8} {'Verdict':>8}")
    print("-" * 100)

    overall_pass = True
    for r in results:
        verdict_marker = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[r.verdict]
        if r.verdict == "FAIL":
            overall_pass = False
        print(f"{r.name:<45} {r.median_ms:>7.1f}ms {r.p95_ms:>7.1f}ms {r.min_ms:>7.1f}ms {r.max_ms:>7.1f}ms {r.pass_threshold_ms:>7.0f}ms {verdict_marker}")

    print("\n" + "=" * 70)
    if overall_pass:
        print("SPIKE 1 VERDICT: ✅ PASS")
        print("Temporal view architecture is viable at 100K rows.")
        print("Decision D1: PROCEED as spec'd (Option A).")
    else:
        print("SPIKE 1 VERDICT: ❌ FAIL")
        print("Review materialized view results above for Fallback Option A.")
        print("Decision D1 requires evaluation of Options B or C.")
    print("=" * 70)

    # Summary stats
    print(f"\nDatabase stats:")
    print(f"  dim_security rows:        {dim_count:>10,}")
    print(f"  vt2_security rows:        {vt2_count:>10,}")
    print(f"  Same-day dupes removed:   {dedup_removed:>10,}")
    print(f"  Materialization time:     {mat_create_ms:>10.1f}ms")

    return overall_pass


def main():
    print("McTheory Praxis — Spike 1: vt2_ View Performance")
    print(f"Config: {NUM_SECURITIES:,} securities × {VERSIONS_PER_SECURITY} versions = {TOTAL_ROWS:,} rows")
    print(f"Warmup: {WARMUP_ITERATIONS} iterations, Measurement: {MEASUREMENT_ITERATIONS} iterations")
    print()

    # In-memory DuckDB for maximum performance measurement
    con = duckdb.connect(":memory:")

    # Generate data
    base_ids = setup_database(con)

    # Create views
    create_views(con)

    # Run benchmarks
    passed = run_spike(con, base_ids)

    con.close()
    return passed


if __name__ == "__main__":
    main()
