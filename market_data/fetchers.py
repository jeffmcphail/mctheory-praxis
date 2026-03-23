"""
Market data fetchers — pluggable callbacks for DataTable.set_fetch_callback().

Each fetcher is a callable with signature:
    (table_name: str, filter_col: str, filter_val: Any) -> pl.DataFrame | None

The DataTable calls this when has_row(fill_missing=True) misses cache.
The fetcher returns a DataFrame matching the table's schema, or None.

Fetchers:
    YFinanceFetcher     — Downloads from Yahoo Finance (requires network)
    MockMarketDataFetcher — Generates realistic synthetic data (for testing)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl


# ═════════════════════════════════════════════════════════════════════════════
# Yahoo Finance Fetcher
# ═════════════════════════════════════════════════════════════════════════════

class YFinanceFetcher:
    """
    Fetches market data from Yahoo Finance via yfinance.

    Handles both the 'securities' table (ticker info) and the 'prices'
    table (OHLCV history). Other tables return None (populated manually
    or by other means).

    Args:
        default_period: History period for price fetches (e.g. "2y", "5y", "max")
        start_date: Explicit start date (overrides default_period)
        end_date: Explicit end date (defaults to today)
    """

    def __init__(
        self,
        default_period: str = "5y",
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ):
        self.default_period = default_period
        self.start_date = str(start_date) if start_date else None
        self.end_date = str(end_date) if end_date else None

    def __call__(
        self, table_name: str, filter_col: str, filter_val: Any
    ) -> pl.DataFrame | None:
        """Dispatch to the appropriate table handler."""
        if table_name == "securities":
            return self._fetch_security(filter_val)
        elif table_name == "prices":
            return self._fetch_prices(filter_val)
        return None

    def _fetch_security(self, ticker: str) -> pl.DataFrame | None:
        """Fetch security master info for a single ticker."""
        try:
            import yfinance as yf

            info = yf.Ticker(ticker).info
            return pl.DataFrame(
                {
                    "security_id": [ticker],
                    "name": [info.get("longName") or info.get("shortName") or ticker],
                    "asset_class": [self._infer_asset_class(info)],
                    "exchange": [info.get("exchange", "")],
                    "currency": [info.get("currency", "USD")],
                    "sector": [info.get("sector", "")],
                    "industry": [info.get("industry", "")],
                }
            )
        except Exception:
            # Return a minimal row so the pipeline doesn't break
            return pl.DataFrame(
                {
                    "security_id": [ticker],
                    "name": [ticker],
                    "asset_class": ["equity"],
                    "exchange": [""],
                    "currency": ["USD"],
                    "sector": [""],
                    "industry": [""],
                }
            )

    def _fetch_prices(self, ticker: str) -> pl.DataFrame | None:
        """Fetch OHLCV price history for a single ticker."""
        try:
            import yfinance as yf

            t = yf.Ticker(ticker)
            if self.start_date:
                hist = t.history(start=self.start_date, end=self.end_date)
            else:
                hist = t.history(period=self.default_period)

            if hist.empty:
                return None

            # yfinance returns pandas with Date index
            hist = hist.reset_index()

            return pl.DataFrame(
                {
                    "security_id": [ticker] * len(hist),
                    "date": pl.Series(hist["Date"].dt.date.tolist()).cast(pl.Date),
                    "open": hist["Open"].values.tolist(),
                    "high": hist["High"].values.tolist(),
                    "low": hist["Low"].values.tolist(),
                    "close": hist["Close"].values.tolist(),
                    "volume": hist["Volume"].values.astype(float).tolist(),
                    "adj_close": hist["Close"].values.tolist(),  # yfinance v1 uses adjusted Close
                }
            )
        except Exception:
            return None

    @staticmethod
    def _infer_asset_class(info: dict) -> str:
        """Infer asset class from yfinance info dict."""
        qtype = info.get("quoteType", "").upper()
        if qtype == "ETF":
            return "equity"  # ETFs trade as equity
        elif qtype == "CRYPTOCURRENCY":
            return "crypto"
        elif qtype in ("FUTURE", "FUTURES"):
            return "futures"
        elif qtype == "CURRENCY":
            return "fx"
        return "equity"


# ═════════════════════════════════════════════════════════════════════════════
# Mock Market Data Fetcher (for testing without network)
# ═════════════════════════════════════════════════════════════════════════════

class MockMarketDataFetcher:
    """
    Generates realistic synthetic market data for testing.

    Produces correlated price series with configurable properties:
    - Annual volatility per asset class
    - Drift (expected return)
    - Intra-universe correlation structure
    - Realistic OHLCV bars from close prices

    Args:
        n_days: Number of trading days to generate
        end_date: Last date in the series (defaults to today)
        seed: Random seed for reproducibility
        security_profiles: Dict of {ticker: {name, asset_class, sector, ...}}
                           If a requested ticker isn't here, a default equity profile is used.
    """

    # Default volatility by asset class (annualized)
    VOL_BY_CLASS = {
        "equity": 0.20,
        "fixed_income": 0.05,
        "commodity": 0.15,
        "fx": 0.08,
        "crypto": 0.60,
        "futures": 0.18,
    }

    # Default drift by asset class (annualized)
    DRIFT_BY_CLASS = {
        "equity": 0.08,
        "fixed_income": 0.03,
        "commodity": 0.02,
        "fx": 0.00,
        "crypto": 0.15,
        "futures": 0.04,
    }

    def __init__(
        self,
        n_days: int = 756,
        end_date: date | None = None,
        seed: int = 42,
        security_profiles: dict[str, dict[str, str]] | None = None,
    ):
        self.n_days = n_days
        self.end_date = end_date or date.today()
        self.rng = np.random.default_rng(seed)
        self.security_profiles = security_profiles or {}

        # Cache generated prices so repeated fetches are consistent
        self._price_cache: dict[str, pl.DataFrame] = {}
        self._security_cache: dict[str, pl.DataFrame] = {}

        # Cointegrated pairs: {ticker: (leader_ticker, beta, noise_vol)}
        # GDX tracks GLD with beta ~1.5 and mean-reverting spread
        self._cointegrated_pairs: dict[str, tuple[str, float, float]] = {
            "GDX": ("GLD", 1.5, 0.03),
        }

    def __call__(
        self, table_name: str, filter_col: str, filter_val: Any
    ) -> pl.DataFrame | None:
        if table_name == "securities":
            return self._fetch_security(filter_val)
        elif table_name == "prices":
            return self._fetch_prices(filter_val)
        elif table_name == "universes":
            return self._fetch_universe(filter_val)
        elif table_name == "universe_members":
            return self._fetch_universe_members(filter_val)
        return None

    def _get_profile(self, ticker: str) -> dict[str, str]:
        """Get or generate a security profile."""
        if ticker in self.security_profiles:
            return self.security_profiles[ticker]
        # Default: equity
        return {
            "name": ticker,
            "asset_class": "equity",
            "exchange": "NYSE",
            "currency": "USD",
            "sector": "Unknown",
            "industry": "Unknown",
        }

    def _fetch_security(self, ticker: str) -> pl.DataFrame:
        if ticker in self._security_cache:
            return self._security_cache[ticker]

        profile = self._get_profile(ticker)
        df = pl.DataFrame(
            {
                "security_id": [ticker],
                "name": [profile.get("name", ticker)],
                "asset_class": [profile.get("asset_class", "equity")],
                "exchange": [profile.get("exchange", "NYSE")],
                "currency": [profile.get("currency", "USD")],
                "sector": [profile.get("sector", "")],
                "industry": [profile.get("industry", "")],
            }
        )
        self._security_cache[ticker] = df
        return df

    def _fetch_prices(self, ticker: str) -> pl.DataFrame:
        if ticker in self._price_cache:
            return self._price_cache[ticker]

        # Check if this ticker is a cointegrated follower
        if ticker in self._cointegrated_pairs:
            return self._generate_cointegrated_prices(ticker)

        profile = self._get_profile(ticker)
        asset_class = profile.get("asset_class", "equity")

        vol = self.VOL_BY_CLASS.get(asset_class, 0.20)
        drift = self.DRIFT_BY_CLASS.get(asset_class, 0.08)

        # Generate GBM close prices
        daily_vol = vol / np.sqrt(252)
        daily_drift = drift / 252
        log_returns = self.rng.normal(daily_drift, daily_vol, self.n_days)
        close_prices = 100.0 * np.exp(np.cumsum(log_returns))

        # Generate realistic OHLCV from closes
        intraday_vol = daily_vol * 0.6
        high_prices = close_prices * (1 + np.abs(self.rng.normal(0, intraday_vol, self.n_days)))
        low_prices = close_prices * (1 - np.abs(self.rng.normal(0, intraday_vol, self.n_days)))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]  # first day
        volume = np.abs(self.rng.normal(1e6, 3e5, self.n_days))

        # Generate business day dates
        dates = []
        d = self.end_date - timedelta(days=int(self.n_days * 1.45))  # overshoot to get enough bdays
        while len(dates) < self.n_days:
            if d.weekday() < 5:  # Mon-Fri
                dates.append(d)
            d += timedelta(days=1)
        dates = dates[-self.n_days:]  # take last n_days

        df = pl.DataFrame(
            {
                "security_id": [ticker] * self.n_days,
                "date": pl.Series(dates).cast(pl.Date),
                "open": open_prices.tolist(),
                "high": high_prices.tolist(),
                "low": low_prices.tolist(),
                "close": close_prices.tolist(),
                "volume": volume.tolist(),
                "adj_close": close_prices.tolist(),
            }
        )
        self._price_cache[ticker] = df
        return df

    def _generate_cointegrated_prices(self, ticker: str) -> pl.DataFrame:
        """Generate prices that are cointegrated with the leader ticker."""
        leader_ticker, beta, noise_vol = self._cointegrated_pairs[ticker]

        # Ensure leader is generated first
        leader_prices = self._fetch_prices(leader_ticker)
        leader_close = leader_prices["close"].to_numpy()

        # Follower = beta * leader + OU(mean-reverting noise)
        n = len(leader_close)
        spread = np.zeros(n)
        theta = 0.05  # mean reversion speed
        for i in range(1, n):
            spread[i] = spread[i - 1] * (1 - theta) + self.rng.normal(0, noise_vol)

        close_prices = beta * leader_close + spread * leader_close
        close_prices = np.maximum(close_prices, 1.0)  # floor at $1

        intraday_vol = noise_vol * 0.6
        high_prices = close_prices * (1 + np.abs(self.rng.normal(0, intraday_vol, n)))
        low_prices = close_prices * (1 - np.abs(self.rng.normal(0, intraday_vol, n)))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        volume = np.abs(self.rng.normal(1e6, 3e5, n))

        dates = leader_prices["date"].to_list()

        df = pl.DataFrame(
            {
                "security_id": [ticker] * n,
                "date": pl.Series(dates).cast(pl.Date),
                "open": open_prices.tolist(),
                "high": high_prices.tolist(),
                "low": low_prices.tolist(),
                "close": close_prices.tolist(),
                "volume": volume.tolist(),
                "adj_close": close_prices.tolist(),
            }
        )
        self._price_cache[ticker] = df
        return df

    def _fetch_universe(self, universe_id: str) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "universe_id": [universe_id],
                "name": [universe_id],
                "description": [f"Auto-generated universe: {universe_id}"],
            }
        )

    def _fetch_universe_members(self, universe_id: str) -> pl.DataFrame | None:
        """Return members for known universe definitions."""
        # Predefined universes for testing
        known = {
            "PAIRS_GLD_GDX": ["GLD", "GDX"],
            "TAA_5": ["SPY", "EFA", "AGG", "GLD", "VNQ"],
            "SECTOR_ETFS": [
                "SPY", "XLK", "XLF", "XLE", "XLV", "XLI",
                "XLP", "XLY", "XLU", "XLB", "XLRE",
            ],
            "SP500_MEGA": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                "BRK-B", "JPM", "JNJ", "V", "PG", "UNH",
                "HD", "MA", "DIS", "PYPL", "BAC", "INTC",
                "CMCSA", "NFLX",
            ],
            "GLOBAL_FUTURES": [
                "ES=F", "NQ=F", "YM=F", "RTY=F",  # US equity
                "ZB=F", "ZN=F",                     # US bonds
                "GC=F", "SI=F", "CL=F",             # commodities
                "EURUSD=X", "GBPUSD=X", "JPYUSD=X", # FX
            ],
        }

        tickers = known.get(universe_id)
        if tickers is None:
            return None

        return pl.DataFrame(
            {
                "universe_id": [universe_id] * len(tickers),
                "security_id": tickers,
            }
        )

    # ─────────────────────────────────────────────────────────────────────
    # Convenience: register profiles for known universes
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def with_standard_profiles(cls, **kwargs) -> "MockMarketDataFetcher":
        """Create fetcher with profiles for standard test universes."""
        profiles = {
            # TAA universe
            "SPY": {"name": "SPDR S&P 500", "asset_class": "equity", "sector": "Broad Market"},
            "EFA": {"name": "iShares MSCI EAFE", "asset_class": "equity", "sector": "International"},
            "AGG": {"name": "iShares Core US Aggregate Bond", "asset_class": "fixed_income", "sector": "Bonds"},
            "GLD": {"name": "SPDR Gold Shares", "asset_class": "commodity", "sector": "Precious Metals"},
            "VNQ": {"name": "Vanguard Real Estate", "asset_class": "equity", "sector": "Real Estate"},
            # Pairs
            "GDX": {"name": "VanEck Gold Miners", "asset_class": "equity", "sector": "Gold Miners"},
            # Sector ETFs
            "XLK": {"name": "Technology Select Sector", "asset_class": "equity", "sector": "Technology"},
            "XLF": {"name": "Financial Select Sector", "asset_class": "equity", "sector": "Financials"},
            "XLE": {"name": "Energy Select Sector", "asset_class": "equity", "sector": "Energy"},
            "XLV": {"name": "Health Care Select Sector", "asset_class": "equity", "sector": "Health Care"},
            "XLI": {"name": "Industrial Select Sector", "asset_class": "equity", "sector": "Industrials"},
            "XLP": {"name": "Consumer Staples Select Sector", "asset_class": "equity", "sector": "Consumer Staples"},
            "XLY": {"name": "Consumer Discretionary Select Sector", "asset_class": "equity", "sector": "Consumer Discretionary"},
            "XLU": {"name": "Utilities Select Sector", "asset_class": "equity", "sector": "Utilities"},
            "XLB": {"name": "Materials Select Sector", "asset_class": "equity", "sector": "Materials"},
            "XLRE": {"name": "Real Estate Select Sector", "asset_class": "equity", "sector": "Real Estate"},
            # Futures (mock as their asset class)
            "ES=F": {"name": "E-mini S&P 500", "asset_class": "futures", "sector": "Equity Index"},
            "NQ=F": {"name": "E-mini NASDAQ 100", "asset_class": "futures", "sector": "Equity Index"},
            "YM=F": {"name": "E-mini Dow", "asset_class": "futures", "sector": "Equity Index"},
            "RTY=F": {"name": "E-mini Russell 2000", "asset_class": "futures", "sector": "Equity Index"},
            "ZB=F": {"name": "US Treasury Bond", "asset_class": "fixed_income", "sector": "Government Bonds"},
            "ZN=F": {"name": "10-Year T-Note", "asset_class": "fixed_income", "sector": "Government Bonds"},
            "GC=F": {"name": "Gold Futures", "asset_class": "commodity", "sector": "Precious Metals"},
            "SI=F": {"name": "Silver Futures", "asset_class": "commodity", "sector": "Precious Metals"},
            "CL=F": {"name": "Crude Oil Futures", "asset_class": "commodity", "sector": "Energy"},
            "EURUSD=X": {"name": "EUR/USD", "asset_class": "fx", "sector": "Major Pairs"},
            "GBPUSD=X": {"name": "GBP/USD", "asset_class": "fx", "sector": "Major Pairs"},
            "JPYUSD=X": {"name": "JPY/USD", "asset_class": "fx", "sector": "Major Pairs"},
        }
        return cls(security_profiles=profiles, **kwargs)
