"""
Burgess Stat-Arb Pipeline — Full S&P 500, Real Market Data
============================================================

Production-grade pipeline:
1. Pull S&P 500 constituents via SP500MembershipProvider (survivorship-
   bias-free point-in-time membership from Wikipedia change history)
2. Fetch security T&Cs from yfinance .info (split into two source tables
   simulating multi-vendor coverage: yf_securities + pg_securities)
3. T&C filters via SQL DataView (market cap, sector, SIC code)
4. Batch price download via yf.download() — view-driven cascade
5. Liquidity filter (avg daily dollar volume, price floor)
6. Burgess cointegration scan on filtered universe

Architecture — single collect() triggers full cascade:

    price_view.collect(params={"universe_id": universe_key})
        └─ yf_prices.fill_from_view()
            └─ filtered_universe_view.collect()
                ├─ sp500_members ← SP500MembershipProvider
                ├─ yf_securities ← yfinance .info (sector, industry, mcap)
                ├─ pg_securities ← yfinance .info (sic, exchange, float)
                └─ SQL: COALESCE + WHERE filters → filtered tickers
            └─ yf.download([filtered tickers]) → batch price fetch
        └─ SQL: SELECT * FROM yf_prices → full price matrix

Usage:
    python validate_burgess_cascade.py
    python validate_burgess_cascade.py --max-tickers 100   # limit for testing
    python validate_burgess_cascade.py --period 2y         # shorter history
    python validate_burgess_cascade.py --as-of-date 2023-01-15  # survivorship-bias-free
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
for candidate in [
    SCRIPT_DIR.parent / "core" / "src",
    SCRIPT_DIR.parent / "core_repo" / "src",
    SCRIPT_DIR / "core_repo" / "src",
    Path(r"C:\Data\Development\Python\McTheoryApps\core\src"),
    Path(r"C:\Data\Development\Python\McTheoryApps\mctheory-core\src"),
]:
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

sys.path.insert(0, str(SCRIPT_DIR))

import polars as pl
from mctheory.core.datastore import DataStore, DataTable, DataView, BusinessEntity
from market_data.universe import SP500MembershipProvider


# ═════════════════════════════════════════════════════════════════════════════
# REAL DATA FETCHERS
# ═════════════════════════════════════════════════════════════════════════════

class YFinanceRowFetcher:
    """
    Row-level fetch callback for has_row_filtered().

    Delegates universe membership to SP500MembershipProvider (survivorship-
    bias-free). Handles security T&Cs via yfinance .info, split across two
    tables to simulate multi-vendor coverage.

    Handles:
        sp500_members  — delegates to SP500MembershipProvider
        yf_securities  — yfinance .info: name, sector, industry, mcap, currency
        pg_securities  — yfinance .info: sic, exchange, shares, float
    """

    def __init__(
        self,
        membership: SP500MembershipProvider,
        max_tickers: int | None = None,
    ):
        self._membership = membership
        self._max_tickers = max_tickers
        self._membership_callback = membership.fetch_callback()
        self._info_cache: dict[str, dict] = {}
        self.call_counts: dict[str, int] = {}

    def __call__(self, table_name: str, filter_col: str, filter_val: Any) -> pl.DataFrame | None:
        self.call_counts[table_name] = self.call_counts.get(table_name, 0) + 1

        if table_name == "sp500_members":
            result = self._membership_callback(table_name, filter_col, filter_val)
            if result is not None and self._max_tickers:
                result = result.head(self._max_tickers)
            return result
        elif table_name == "yf_securities":
            return self._fetch_yf_security(filter_val)
        elif table_name == "pg_securities":
            return self._fetch_pg_security(filter_val)
        return None

    def _get_info(self, ticker: str) -> dict:
        """Fetch and cache yfinance .info for a ticker."""
        if ticker in self._info_cache:
            return self._info_cache[ticker]
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            self._info_cache[ticker] = info
            return info
        except Exception:
            self._info_cache[ticker] = {}
            return {}

    def _fetch_yf_security(self, ticker: str) -> pl.DataFrame | None:
        """YFinance source: name, sector, industry, market_cap, currency."""
        info = self._get_info(ticker)
        if not info:
            return None
        return pl.DataFrame({
            "security_id": [ticker],
            "name": [info.get("longName") or info.get("shortName") or ticker],
            "sector": [info.get("sector") or ""],
            "industry": [info.get("industry") or ""],
            "market_cap": [float(info.get("marketCap") or 0)],
            "currency": [info.get("currency") or "USD"],
        })

    def _fetch_pg_security(self, ticker: str) -> pl.DataFrame | None:
        """Polygon-style source: sic, exchange, shares, float."""
        info = self._get_info(ticker)
        if not info:
            return None
        return pl.DataFrame({
            "security_id": [ticker],
            "name": [info.get("longName") or ticker],
            "sic_code": [str(info.get("sic") or "")],
            "market_cap": [float(info.get("marketCap") or 0)],
            "shares_outstanding": [float(info.get("sharesOutstanding") or 0)],
            "exchange": [info.get("exchange") or ""],
            "float_shares": [float(info.get("floatShares") or 0)],
        })


class YFinancePriceVendor:
    """
    View-driven batch price fetcher using yf.download().

    Called by DataTable.fill_from_view() with the filtered ticker list.
    Uses yfinance's batch download API — one HTTP call for all tickers.
    """

    def __init__(self, period: str = "3y"):
        self.period = period
        self.tickers_fetched: list[str] = []
        self.fetch_time: float = 0.0

    def __call__(self, tickers: list[str]) -> pl.DataFrame:
        """Batch download OHLCV for all tickers."""
        import yfinance as yf

        self.tickers_fetched = list(tickers)
        print(f"\n  📡 yf.download({len(tickers)} tickers, period='{self.period}')...")
        t0 = time.time()

        # yfinance batch download — single API call
        raw = yf.download(
            tickers,
            period=self.period,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=True,
        )
        self.fetch_time = time.time() - t0
        print(f"  ⏱ Download complete: {self.fetch_time:.1f}s")

        if raw.empty:
            print("  ⚠ No price data returned")
            return pl.DataFrame()

        # Convert multi-index DataFrame to long format
        frames = []
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    # Single ticker: no multi-level columns
                    ticker_data = raw
                else:
                    ticker_data = raw[ticker]

                if ticker_data.empty or ticker_data.dropna(how="all").empty:
                    continue

                df = ticker_data.dropna(subset=["Close"]).reset_index()

                # Handle both Timestamp and Date index types
                dates = df["Date"] if "Date" in df.columns else df.index
                dates_list = [d.date() if hasattr(d, "date") else d for d in dates]

                n = len(df)
                frames.append(pl.DataFrame({
                    "security_id": [ticker] * n,
                    "date": pl.Series(dates_list).cast(pl.Date),
                    "open": df["Open"].values.tolist(),
                    "high": df["High"].values.tolist(),
                    "low": df["Low"].values.tolist(),
                    "close": df["Close"].values.tolist(),
                    "volume": df["Volume"].astype(float).values.tolist(),
                    "adj_close": df["Close"].values.tolist(),
                }))
            except Exception as e:
                print(f"  ⚠ Skipped {ticker}: {e}")
                continue

        if not frames:
            return pl.DataFrame()

        result = pl.concat(frames)
        n_tickers = result["security_id"].n_unique()
        n_rows = len(result)
        print(f"  ✓ Loaded {n_tickers} tickers, {n_rows:,} price rows")
        return result


# ═════════════════════════════════════════════════════════════════════════════
# DATASTORE WIRING — CASCADE GRAPH
# ═════════════════════════════════════════════════════════════════════════════

def create_cascade_datastore(
    row_fetcher: YFinanceRowFetcher,
    price_vendor: YFinancePriceVendor,
) -> DataStore:
    """
    Wire the full dependency cascade:

        sp500_members          ← Wikipedia
            ↓ FK cascade
        yf_securities          ← yfinance .info (per-ticker)
        pg_securities          ← yfinance .info (per-ticker, different fields)
            ↓ SQL COALESCE JOIN + WHERE
        filtered_universe_view ← T&C filters
            ↓ view-driven batch fetch
        yf_prices              ← yf.download([filtered tickers])
            ↓ SQL SELECT
        price_view             ← final output
    """
    DataStore.reset_instance()
    ds = DataStore.get_instance()

    # ── Raw tables ──────────────────────────────────────────────────────

    table_defs = {
        "sp500_members": {
            "schema": {"universe_id": pl.Utf8, "security_id": pl.Utf8},
            "pk": ["universe_id", "security_id"],
            "filter_col": "universe_id",
        },
        "yf_securities": {
            "schema": {
                "security_id": pl.Utf8, "name": pl.Utf8, "sector": pl.Utf8,
                "industry": pl.Utf8, "market_cap": pl.Float64, "currency": pl.Utf8,
            },
            "pk": ["security_id"], "filter_col": "security_id",
        },
        "pg_securities": {
            "schema": {
                "security_id": pl.Utf8, "name": pl.Utf8, "sic_code": pl.Utf8,
                "market_cap": pl.Float64, "shares_outstanding": pl.Float64,
                "exchange": pl.Utf8, "float_shares": pl.Float64,
            },
            "pk": ["security_id"], "filter_col": "security_id",
        },
        "yf_prices": {
            "schema": {
                "security_id": pl.Utf8, "date": pl.Date, "open": pl.Float64,
                "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
                "volume": pl.Float64, "adj_close": pl.Float64,
            },
            "pk": ["security_id", "date"], "filter_col": "security_id",
        },
    }

    for name, defn in table_defs.items():
        table = DataTable(
            data=pl.DataFrame(schema=defn["schema"]),
            name=name, primary_key=defn["pk"],
        )
        table.set_filter_column(defn["filter_col"])
        if name != "yf_prices":
            table.set_fetch_callback(row_fetcher)
        ds.register_table(name, table)

    # ── Filtered universe view: COALESCE merge + T&C filters ────────────
    #
    # Filters:
    #   market_cap > $10B         — exclude micro/small caps
    #   sector NOT IN (Utilities, Real Estate)  — low vol / regulated
    #   SIC code NOT starting with '6'          — exclude financials
    #
    # Note: V and MA have SIC 7372 (software services), so they pass through.
    #       UNH has SIC 6324, so it's correctly excluded as a financial.

    filtered_universe_view = DataView(
        name="filtered_universe_view",
        sql_query="""
            SELECT
                COALESCE(yf.security_id, pg.security_id) AS security_id,
                COALESCE(yf.name, pg.name) AS name,
                yf.sector,
                yf.industry,
                pg.sic_code,
                COALESCE(yf.market_cap, pg.market_cap) AS market_cap,
                pg.exchange
            FROM yf_securities yf
            FULL OUTER JOIN pg_securities pg
                ON yf.security_id = pg.security_id
            WHERE COALESCE(yf.market_cap, pg.market_cap) > 10000000000
              AND yf.sector NOT IN ('Utilities', 'Real Estate')
              AND (pg.sic_code NOT LIKE '6%' OR pg.sic_code IS NULL)
        """,
        datastore=ds,
    )
    filtered_universe_view.register_fk_chain(
        "sp500_members", "security_id", "yf_securities", "security_id"
    )
    filtered_universe_view.register_fk_chain(
        "sp500_members", "security_id", "pg_securities", "security_id"
    )

    # ── Security master view (unfiltered, for inspection) ───────────────

    security_master_view = DataView(
        name="security_master_view",
        sql_query="""
            SELECT
                COALESCE(yf.security_id, pg.security_id) AS security_id,
                COALESCE(yf.name, pg.name) AS name,
                yf.sector,
                yf.industry,
                pg.sic_code,
                COALESCE(yf.market_cap, pg.market_cap) AS market_cap,
                pg.shares_outstanding,
                pg.float_shares,
                pg.exchange,
                yf.currency
            FROM yf_securities yf
            FULL OUTER JOIN pg_securities pg
                ON yf.security_id = pg.security_id
        """,
        datastore=ds,
    )

    # ── Wire prices with view-driven batch fetch ────────────────────────

    prices_table = ds.get_table("yf_prices")
    prices_table.set_view_driven_fetch(
        source_view_name="filtered_universe_view",
        extract_column="security_id",
        vendor_fetcher=price_vendor,
    )

    # ── Price view — the top-level entry point ──────────────────────────

    price_view = DataView(
        name="price_view",
        sql_query="""
            SELECT security_id, date, open, high, low, close, volume, adj_close
            FROM yf_prices
            ORDER BY security_id, date
        """,
        datastore=ds,
    )

    return ds


# ═════════════════════════════════════════════════════════════════════════════
# BUSINESS ENTITIES
# ═════════════════════════════════════════════════════════════════════════════

class SecurityMaster(BusinessEntity):
    _primary_key_fields: ClassVar[list[str]] = ["security_id"]
    _data_view_name: ClassVar[str] = "security_master_view"

    @property
    def name(self) -> str: return self.get_field("name", "")
    @property
    def sector(self) -> str: return self.get_field("sector", "")
    @property
    def market_cap(self) -> float: return self.get_field("market_cap", 0.0)
    @property
    def sic_code(self) -> str: return self.get_field("sic_code", "")
    @property
    def exchange(self) -> str: return self.get_field("exchange", "")


class PriceHistory(BusinessEntity):
    _primary_key_fields: ClassVar[list[str]] = ["security_id"]
    _data_view_name: ClassVar[str] = "price_view"


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Burgess Stat-Arb — Full S&P 500, Real Market Data            ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    # ── Membership provider (survivorship-bias-free) ──────────────────
    membership = SP500MembershipProvider()

    # Determine universe key (supports point-in-time via colon syntax)
    if args.as_of_date:
        universe_key = f"SP500:{args.as_of_date}"
    else:
        universe_key = "SP500"

    row_fetcher = YFinanceRowFetcher(
        membership=membership,
        max_tickers=args.max_tickers,
    )
    price_vendor = YFinancePriceVendor(period=args.period)

    ds = create_cascade_datastore(row_fetcher, price_vendor)

    # ── Membership summary ──────────────────────────────────────────────
    print(f"\n─── MEMBERSHIP ───────────────────────────────────────────────")
    print(f"  Current members:   {membership.n_current}")
    print(f"  All-time members:  {membership.n_all_time}")
    print(f"  History periods:   {membership.n_periods}")
    if args.as_of_date:
        pit = membership.members(args.as_of_date)
        print(f"  As-of {args.as_of_date}:  {len(pit)} members")
    if args.max_tickers:
        print(f"  Capped at:         {args.max_tickers} tickers")

    # ── Verify everything is empty ──────────────────────────────────────
    print("\n─── BEFORE ───────────────────────────────────────────────────")
    for name in ["sp500_members", "yf_securities", "pg_securities", "yf_prices"]:
        print(f"  {name:<20} {len(ds.get_table(name)):>8} rows")

    # ════════════════════════════════════════════════════════════════════
    # THE SINGLE CALL — triggers the entire cascade
    # ════════════════════════════════════════════════════════════════════
    print(f"\n═══ TRIGGERING: price_view.collect(params={{'universe_id': '{universe_key}'}}) ═══")
    print(f"  This will:")
    print(f"    1. Fetch S&P 500 members from Wikipedia")
    print(f"    2. Fetch T&Cs from yfinance .info (×2 per ticker)")
    print(f"    3. Apply T&C filters via SQL")
    print(f"    4. Batch download prices via yf.download()")
    print(f"  Expect ~5-15 min for T&C fetches on full S&P 500...\n")

    t0 = time.time()
    price_view = ds.get_view("price_view")
    all_prices = price_view.collect(params={"universe_id": universe_key})
    total_cascade = time.time() - t0

    print(f"\n  ⏱ Total cascade time: {total_cascade:.1f}s")

    # ── Guard: empty cascade ────────────────────────────────────────────
    if len(all_prices) == 0:
        print("\n  ✗ CASCADE RETURNED NO DATA")
        print("    Likely causes:")
        print("      - lxml not installed (pip install lxml)")
        print("      - Wikipedia/yfinance network blocked")
        print("      - All tickers filtered out by T&C rules")
        for name in ["sp500_members", "yf_securities", "pg_securities", "yf_prices"]:
            n = len(ds.get_table(name))
            print(f"      {name:<20} {n:>6} rows {'← problem here' if n == 0 else ''}")
        return 1

    # ── Post-cascade state ──────────────────────────────────────────────
    print("\n─── AFTER ────────────────────────────────────────────────────")
    for name in ["sp500_members", "yf_securities", "pg_securities", "yf_prices"]:
        print(f"  {name:<20} {len(ds.get_table(name)):>8} rows")

    # ── Cascade audit ───────────────────────────────────────────────────
    print(f"\n─── CASCADE AUDIT ────────────────────────────────────────────")
    n_members = len(ds.get_table("sp500_members"))
    n_fetched = len(price_vendor.tickers_fetched)
    n_excluded = n_members - n_fetched

    print(f"  Universe members:    {n_members}")
    print(f"  After T&C filter:    {n_fetched}")
    print(f"  Excluded:            {n_excluded}")
    print(f"  Price download:      1 batch call, {price_vendor.fetch_time:.1f}s")

    # Show T&C filter breakdown
    sec_master = ds.get_view("security_master_view").collect()
    filtered = ds.get_view("filtered_universe_view").collect()

    print(f"\n  T&C Filter Breakdown:")
    # Market cap filter
    has_mcap = sec_master.filter(pl.col("market_cap") > 10e9)
    print(f"    market_cap > $10B:    {len(has_mcap):>4} / {len(sec_master)}")

    # Sector filter (applied after mcap)
    excluded_sectors = ["Utilities", "Real Estate"]
    no_util_re = has_mcap.filter(~pl.col("sector").is_in(excluded_sectors))
    print(f"    excl Utilities/RE:    {len(no_util_re):>4} / {len(has_mcap)}")

    # SIC filter
    print(f"    excl SIC starts '6':  {len(filtered):>4} / {len(no_util_re)}")

    # Sector distribution of filtered universe
    sectors = filtered.group_by("sector").len().sort("len", descending=True)
    print(f"\n  Filtered Universe by Sector:")
    for row in sectors.iter_rows(named=True):
        print(f"    {row['sector']:<30} {row['len']:>4}")

    # Show excluded tickers sample
    all_members = set(sec_master["security_id"].to_list())
    kept = set(filtered["security_id"].to_list())
    excluded_tickers = sorted(all_members - kept)

    if excluded_tickers:
        print(f"\n  Sample Excluded ({len(excluded_tickers)} total):")
        excluded_df = sec_master.filter(pl.col("security_id").is_in(excluded_tickers))
        sample = excluded_df.sort("market_cap", descending=True).head(15)
        for row in sample.iter_rows(named=True):
            mcap = row.get("market_cap") or 0
            mcap_str = f"${mcap/1e9:.0f}B" if mcap > 0 else "N/A"
            sic = row.get("sic_code") or "?"
            sector = row.get("sector") or "?"
            print(f"    {row['security_id']:<6} {sector:<25} sic={sic:<6} mcap={mcap_str}")

    # ── Price data stats ────────────────────────────────────────────────
    print(f"\n─── PRICE DATA ───────────────────────────────────────────────")
    n_tickers = all_prices["security_id"].n_unique()
    n_dates = all_prices["date"].n_unique()
    print(f"  Tickers:  {n_tickers}")
    print(f"  Dates:    {n_dates}")
    print(f"  Rows:     {len(all_prices):,}")
    print(f"  Range:    {all_prices['date'].min()} → {all_prices['date'].max()}")

    # ── Liquidity filter ────────────────────────────────────────────────
    print(f"\n─── LIQUIDITY FILTER ─────────────────────────────────────────")
    print(f"  Criteria: avg_daily_dollar_volume > $20M, last_close > $5, <5% missing days")

    liq = all_prices.group_by("security_id").agg([
        (pl.col("close") * pl.col("volume")).mean().alias("avg_dollar_vol"),
        pl.col("close").last().alias("last_close"),
        pl.col("date").count().alias("n_days"),
    ])

    liquid = liq.filter(
        (pl.col("avg_dollar_vol") > 20e6) &
        (pl.col("last_close") > 5.0) &
        (pl.col("n_days") > n_dates * 0.95)
    ).sort("avg_dollar_vol", descending=True)

    n_illiquid = n_tickers - len(liquid)
    print(f"  Before:   {n_tickers}")
    print(f"  After:    {len(liquid)}  (removed {n_illiquid})")

    if len(liquid) > 0:
        print(f"\n  Top 10 by avg daily dollar volume:")
        for row in liquid.head(10).iter_rows(named=True):
            print(f"    {row['security_id']:<6} ${row['avg_dollar_vol']/1e6:>10.1f}M  "
                  f"last=${row['last_close']:.2f}  days={row['n_days']}")

        if len(liquid) > 10:
            print(f"\n  Bottom 5:")
            for row in liquid.tail(5).iter_rows(named=True):
                print(f"    {row['security_id']:<6} ${row['avg_dollar_vol']/1e6:>10.1f}M  "
                      f"last=${row['last_close']:.2f}  days={row['n_days']}")

    final_tickers = sorted(liquid["security_id"].to_list())

    # ── Build price matrix ──────────────────────────────────────────────
    print(f"\n─── PRICE MATRIX ─────────────────────────────────────────────")

    wide = all_prices.filter(
        pl.col("security_id").is_in(final_tickers)
    ).select(
        ["security_id", "date", "adj_close"]
    ).sort("date").pivot(
        on="security_id", index="date", values="adj_close"
    ).sort("date")

    available = [t for t in final_tickers if t in wide.columns]
    matrix = wide.select(available).to_numpy()

    # Forward-fill NaN
    nan_before = np.isnan(matrix).sum()
    for col in range(matrix.shape[1]):
        for row in range(1, matrix.shape[0]):
            if np.isnan(matrix[row, col]):
                matrix[row, col] = matrix[row - 1, col]
    nan_after = np.isnan(matrix).sum()

    print(f"  Shape:     {matrix.shape} (dates × assets)")
    print(f"  NaN fill:  {nan_before} → {nan_after}")

    n_pairs = len(available) * (len(available) - 1) // 2
    print(f"  Pairs:     {n_pairs:,}")

    if len(available) < 2:
        print("\n  ✗ Fewer than 2 assets after filtering — cannot run Burgess scan")
        return 1

    # ── Burgess stat-arb scan ───────────────────────────────────────────
    print(f"\n─── BURGESS STAT-ARB SCAN ────────────────────────────────────")

    from engines.cointegration import StatArbEngine, StatArbParams, StatArbInput

    engine = StatArbEngine()

    for label, params in [
        ("Classic (ADF)", StatArbParams(
            max_candidates=50, n_per_basket=2, entry_threshold=2.0,
            zscore_lookback=63, regression_method="ols",
            scoring_mode="classic", max_hurst=1.1,
        )),
        ("Composite", StatArbParams(
            max_candidates=50, n_per_basket=2, entry_threshold=1.5,
            zscore_lookback=63, regression_method="ridge",
            scoring_mode="composite", max_hurst=1.1,
        )),
    ]:
        t1 = time.time()
        result = engine.compute(
            StatArbInput(prices=matrix, asset_names=available), params
        )
        dt = time.time() - t1

        status = "✓" if result.ok else "⚠"
        print(f"\n  {status} {label} ({dt:.1f}s):")
        print(f"    Scanned: {result.n_candidates_scanned}  "
              f"Stationary: {result.n_passed_stationarity}  "
              f"Baskets: {len(result.top_baskets)}")

        for i, basket in enumerate(result.top_baskets[:10]):
            idx = [basket.target_idx] + basket.basket_indices
            assets = [available[j] for j in idx]
            print(f"    {i+1:>2}. {'/'.join(assets):<25} "
                  f"score={basket.composite_score:>8.4f}  "
                  f"hurst={basket.stationarity.hurst_exponent:.3f}  "
                  f"hl={basket.stationarity.half_life:>5.0f}d  "
                  f"adf_p={basket.stationarity.adf_pvalue:.4f}")

        if result.signals:
            print(f"\n    Active signals:")
            for sig in result.signals[:10]:
                idx = sig.basket_indices
                assets = [available[j] for j in idx]
                print(f"      {'/'.join(assets):<25} "
                      f"z={sig.current_zscore:>7.2f}  "
                      f"signal={sig.signal:<6}  "
                      f"entry@{sig.entry_level}  exit@{sig.exit_level}")

    # ── Cache verification ──────────────────────────────────────────────
    print(f"\n─── CACHE VERIFICATION ───────────────────────────────────────")
    t2 = time.time()
    cached = price_view.collect(params={"universe_id": universe_key})
    cache_time = time.time() - t2
    print(f"  Second collect: {cache_time*1000:.2f}ms (cached={len(cached):,} rows)")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 64}")
    print(f"  SUMMARY")
    print(f"{'═' * 64}")
    print(f"  Data source:       YFinance (real market data)")
    print(f"  Universe:          S&P 500 ({n_members} members"
          f"{', as-of ' + args.as_of_date if args.as_of_date else ''})")
    print(f"  After T&C filter:  {n_fetched} tickers")
    print(f"  After liquidity:   {len(available)} tickers")
    print(f"  Price matrix:      {matrix.shape}")
    print(f"  Pairs scanned:     {n_pairs:,}")
    print(f"  Price period:      {args.period}")
    print(f"  Cascade time:      {total_cascade:.1f}s")
    print(f"    T&C fetches:     {sum(row_fetcher.call_counts.values())} calls")
    print(f"    Price download:  {price_vendor.fetch_time:.1f}s (1 batch)")
    print(f"\n  ✓ PIPELINE COMPLETE")

    return 0


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Burgess Stat-Arb — Full S&P 500, Real Market Data"
    )
    parser.add_argument(
        "--max-tickers", type=int, default=None,
        help="Limit number of S&P 500 members (for faster testing)"
    )
    parser.add_argument(
        "--period", type=str, default="3y",
        help="Price history period (yfinance format: 1y, 2y, 3y, 5y, max)"
    )
    parser.add_argument(
        "--as-of-date", type=str, default=None,
        help="Point-in-time S&P 500 membership date (YYYY-MM-DD). "
             "Uses current membership if not specified."
    )
    args = parser.parse_args()

    if args.max_tickers:
        print(f"⚠ Limited to {args.max_tickers} tickers (--max-tickers)")
    if args.as_of_date:
        print(f"📅 Point-in-time membership: {args.as_of_date} (survivorship-bias-free)")

    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
