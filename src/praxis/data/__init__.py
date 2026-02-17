"""
Simple Data Feed (Phase 1.10).

Direct yfinance pull → Polars DataFrame. No security master,
no loader lifecycle, no archive. Ephemeral-mode compatible.

Phase 2 replaces this with the full loader infrastructure.

Usage:
    from praxis.data import fetch_prices, PriceData

    # Single ticker
    prices = fetch_prices("AAPL", start="2023-01-01", end="2024-01-01")

    # Multiple tickers
    prices = fetch_prices(["AAPL", "MSFT"], start="2023-01-01")

    # From config
    prices = PriceData.from_config(model_config)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import polars as pl

from praxis.config import ModelConfig
from praxis.logger.core import PraxisLogger

# Optional dependency - imported at module level for mockability
try:
    import yfinance as yf
except ImportError:
    yf = None


def fetch_prices(
    tickers: str | list[str],
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    provider: str = "yfinance",
) -> pl.DataFrame:
    """
    Fetch OHLCV price data.

    Args:
        tickers: Single ticker or list of tickers.
        start: Start date (ISO string). Default: 1 year ago.
        end: End date (ISO string). Default: today.
        interval: Data frequency ('1d', '1h', '5m', etc.).
        provider: Data source ('yfinance').

    Returns:
        Polars DataFrame with columns:
        [date, open, high, low, close, volume]
        For multiple tickers: [date, ticker, open, high, low, close, volume]

    Raises:
        ImportError: If yfinance not installed.
        ValueError: If provider not supported or no data returned.
    """
    log = PraxisLogger.instance()

    if isinstance(tickers, str):
        tickers = [tickers]

    if start is None:
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    log.info(
        f"Fetching prices: {', '.join(tickers)} "
        f"({start} → {end}, {interval})",
        tags={"data", "trade_cycle"},
        tickers=tickers,
        provider=provider,
    )

    if provider == "yfinance":
        return _fetch_yfinance(tickers, start, end, interval)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'yfinance'.")


def _fetch_yfinance(
    tickers: list[str],
    start: str,
    end: str,
    interval: str,
) -> pl.DataFrame:
    """Fetch from yfinance, return Polars DataFrame."""
    log = PraxisLogger.instance()

    if yf is None:
        raise ImportError(
            "yfinance required for data fetching. Install: pip install yfinance"
        )

    from praxis.logger.vendor_capture import vendor_capture

    frames = []

    for ticker in tickers:
        log.debug(f"Downloading {ticker}...", tags={"data"})

        try:
            df_pd = yf.download(
                ticker, start=start, end=end,
                interval=interval, progress=False, auto_adjust=True,
            )

            # ── Vendor raw capture (dormant unless tag active) ──
            if df_pd is not None and not df_pd.empty:
                vendor_capture(
                    vendor="yfinance",
                    endpoint="download",
                    ticker=ticker,
                    params={"start": start, "end": end, "interval": interval, "auto_adjust": "true"},
                    raw_payload=df_pd.to_csv(float_format="%.10g"),
                )

            if df_pd.empty:
                log.warning(f"No data returned for {ticker}", tags={"data"})
                continue

            # Flatten MultiIndex columns if present (yfinance quirk)
            if hasattr(df_pd.columns, 'levels'):
                df_pd.columns = [
                    col[0] if isinstance(col, tuple) else col
                    for col in df_pd.columns
                ]

            # Standardize column names
            df_pd = df_pd.reset_index()
            col_map = {}
            for col in df_pd.columns:
                lower = col.lower().strip()
                if lower in ("date", "datetime"):
                    col_map[col] = "date"
                elif lower == "open":
                    col_map[col] = "open"
                elif lower == "high":
                    col_map[col] = "high"
                elif lower == "low":
                    col_map[col] = "low"
                elif lower == "close":
                    col_map[col] = "close"
                elif lower == "volume":
                    col_map[col] = "volume"
            df_pd = df_pd.rename(columns=col_map)

            # Convert to Polars
            df = pl.from_pandas(df_pd)

            # Select only standard OHLCV columns
            available = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
            df = df.select(available)

            if len(tickers) > 1:
                df = df.with_columns(pl.lit(ticker).alias("ticker"))

            frames.append(df)

            log.debug(
                f"{ticker}: {len(df)} bars fetched",
                tags={"data"},
            )

        except Exception as e:
            log.error(f"Failed to fetch {ticker}: {e}", tags={"data"})
            continue

    if not frames:
        raise ValueError(f"No data returned for any ticker: {tickers}")

    if len(frames) == 1:
        result = frames[0]
    else:
        result = pl.concat(frames)

    log.info(
        f"Fetched {len(result)} total bars for {len(frames)} tickers",
        tags={"data", "trade_cycle"},
    )

    return result


class PriceData:
    """
    Convenience class for loading prices from a ModelConfig.

    Reads construction.universe.instruments and construction.data_source
    from the config, or falls back to sensible defaults.
    """

    @staticmethod
    def from_config(
        config: ModelConfig,
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pl.DataFrame:
        """
        Load price data based on model config.

        Priority:
        1. Explicit tickers/start/end args
        2. Config construction.universe.instruments
        3. Config construction.data_source settings

        Args:
            config: ModelConfig to extract data settings from.
            tickers: Override tickers (ignores config).
            start: Override start date.
            end: Override end date.

        Returns:
            Polars DataFrame with OHLCV columns.
        """
        # Extract from config if not overridden
        if tickers is None and config.construction:
            if config.construction.universe:
                tickers = config.construction.universe.instruments

        if tickers is None:
            raise ValueError(
                "No tickers specified. Provide tickers or "
                "set construction.universe.instruments in config."
            )

        # Data source settings
        provider = "yfinance"
        lookback = 365

        if config.construction and config.construction.data_source:
            ds = config.construction.data_source
            if ds.provider:
                provider = ds.provider
            if ds.lookback_days:
                lookback = ds.lookback_days

        if start is None:
            start = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")

        return fetch_prices(tickers, start=start, end=end, provider=provider)


def generate_synthetic_prices(
    n_bars: int = 252,
    initial_price: float = 100.0,
    drift: float = 0.0005,
    volatility: float = 0.02,
    seed: int | None = None,
    ticker: str | None = None,
) -> pl.DataFrame:
    """
    Generate synthetic OHLCV data for testing.

    Uses geometric Brownian motion: dS = μSdt + σSdW

    Args:
        n_bars: Number of bars to generate.
        initial_price: Starting price.
        drift: Daily drift (mean return).
        volatility: Daily volatility (std of returns).
        seed: Random seed for reproducibility.
        ticker: Optional ticker name column.

    Returns:
        Polars DataFrame with [date, open, high, low, close, volume].
    """
    import numpy as np
    from datetime import date, timedelta

    if seed is not None:
        np.random.seed(seed)

    # Generate returns
    returns = np.random.randn(n_bars) * volatility + drift
    prices = initial_price * np.exp(np.cumsum(returns))

    # Synthetic OHLCV from close
    close = prices
    open_ = np.roll(close, 1)
    open_[0] = initial_price
    high = np.maximum(open_, close) * (1 + np.abs(np.random.randn(n_bars) * 0.005))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.randn(n_bars) * 0.005))
    volume = (np.random.exponential(1e6, n_bars)).astype(int)

    # Generate dates (business days approximation)
    start = date(2023, 1, 3)  # First business day 2023
    dates = []
    d = start
    for _ in range(n_bars):
        while d.weekday() >= 5:  # Skip weekends
            d += timedelta(days=1)
        dates.append(d)
        d += timedelta(days=1)

    data = {
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }

    if ticker:
        data["ticker"] = [ticker] * n_bars

    return pl.DataFrame(data)
