"""
Universal TA Strategy — Asset-class agnostic TA trading via config.

Supports: crypto (CCXT), futures (yfinance), FX (yfinance), equities (yfinance).
All share the same 8 TA models, param grid, features, and execution.
Only the data source, asset universe, session hours, and TC differ.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engines.cpo_core import ModelSpec, ConfigSpec
from engines.ta_models import (
    TAModelSpec, TAConfig, TA_MODEL_TYPES, TA_FEATURES,
    generate_ta_param_grid, run_ta_single_day, compute_ta_features,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# ASSET CLASS CONFIGURATIONS
# ═════════════════════════════════════════════════════════════════════════════

ASSET_CLASSES = {
    "crypto": {
        "name": "Crypto",
        "source": "ccxt",
        "exchange": "binance",
        "assets": {
            "BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT",
            "BNB": "BNB/USDT", "XRP": "XRP/USDT", "ADA": "ADA/USDT",
            "AVAX": "AVAX/USDT", "DOGE": "DOGE/USDT",
        },
        "benchmark": "BTC",
        "tc_bps": 2.0,
        "session": "24/7",  # no session filter
        "min_bars_per_day": 12,
        "timeframe": "1h",
    },
    "futures": {
        "name": "US Futures",
        "source": "yfinance",
        "assets": {
            "ES": "ES=F",     # E-mini S&P 500
            "NQ": "NQ=F",     # E-mini Nasdaq 100
            "YM": "YM=F",     # E-mini Dow
            "RTY": "RTY=F",   # E-mini Russell 2000
            "CL": "CL=F",     # Crude Oil
            "GC": "GC=F",     # Gold
            "SI": "SI=F",     # Silver
            "NG": "NG=F",     # Natural Gas
        },
        "benchmark": "ES",
        "tc_bps": 1.0,       # futures are very cheap to trade
        "session": "23h",     # nearly 24h with 1h maintenance
        "min_bars_per_day": 12,
        "timeframe": "1h",
    },
    "fx": {
        "name": "FX G10",
        "source": "yfinance",
        "assets": {
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X",
            "USDJPY": "USDJPY=X",
            "AUDUSD": "AUDUSD=X",
            "USDCAD": "USDCAD=X",
            "USDCHF": "USDCHF=X",
            "NZDUSD": "NZDUSD=X",
            "EURGBP": "EURGBP=X",
        },
        "benchmark": "EURUSD",
        "tc_bps": 0.5,       # FX has tightest spreads
        "session": "24/5",
        "min_bars_per_day": 12,
        "timeframe": "1h",
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_hourly_yfinance(ticker: str, start: str, end: str,
                           cache_dir: Path) -> pd.DataFrame:
    """Fetch hourly bars via yfinance (futures, FX, equities)."""
    import yfinance as yf

    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.replace("/", "_").replace("=", "_")
    cache_path = cache_dir / f"{safe_ticker}_{start}_{end}_1h.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        t = yf.Ticker(ticker)
        # yfinance max for hourly is ~730 days; fetch in chunks if needed
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        all_dfs = []

        current = start_dt
        while current < end_dt:
            chunk_end = min(current + timedelta(days=59), end_dt)
            df = t.history(
                start=current.strftime("%Y-%m-%d"),
                end=(chunk_end + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1h",
            )
            if not df.empty:
                all_dfs.append(df)
            current = chunk_end + timedelta(days=1)
            time.sleep(0.5)

        if not all_dfs:
            return pd.DataFrame()

        df = pd.concat(all_dfs)
        df = df[~df.index.duplicated(keep="last")]

        # Normalize columns
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0.0

        df = df[["open", "high", "low", "close", "volume"]]

        # Remove timezone info for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        if not df.empty:
            df.to_parquet(cache_path)
        return df

    except Exception as e:
        logger.warning(f"yfinance fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def _fetch_hourly_ccxt(asset: str, start: str, end: str,
                       cache_dir: Path, symbol: str,
                       exchange_id: str = "binance") -> pd.DataFrame:
    """Fetch hourly bars via CCXT."""
    import ccxt

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{asset}_{start}_{end}_1h_{exchange_id}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    try:
        exchange.load_markets()
        if symbol not in exchange.markets:
            logger.warning(f"{symbol} not on {exchange_id}")
            return pd.DataFrame()
    except Exception as e:
        logger.warning(f"CCXT markets failed: {e}")
        return pd.DataFrame()

    all_bars = []
    since = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)

    while since < end_ms:
        try:
            bars = exchange.fetch_ohlcv(symbol, "1h", since=since, limit=1000)
            if not bars:
                break
            all_bars.extend(bars)
            since = bars[-1][0] + 3600000
            if len(bars) < 1000:
                break
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            logger.warning(f"CCXT error {symbol}: {e}")
            break

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="last")]
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    if not df.empty:
        df.to_parquet(cache_path)
    return df


def fetch_all_hourly(asset_config: dict, start: str, end: str,
                     cache_dir: Path) -> dict[str, pd.DataFrame]:
    """Fetch hourly data for all assets in config. Auto-selects data source."""
    source = asset_config["source"]
    assets = asset_config["assets"]
    exchange = asset_config.get("exchange", "binance")

    print(f"  Fetching hourly data for {len(assets)} assets: {start} -> {end}")
    print(f"  Source: {source}" + (f" ({exchange})" if source == "ccxt" else ""))

    data = {}
    for i, (name, ticker) in enumerate(assets.items()):
        if source == "ccxt":
            df = _fetch_hourly_ccxt(name, start, end, cache_dir, ticker, exchange)
        else:
            df = _fetch_hourly_yfinance(ticker, start, end, cache_dir)

        if not df.empty:
            data[name] = df
        if (i + 1) % 4 == 0 or i == len(assets) - 1:
            n = len(df) if not df.empty else 0
            print(f"    {i+1}/{len(assets)}: {name} ({n} bars)")

    print(f"  Loaded {len(data)}/{len(assets)} assets")
    return data


# ═════════════════════════════════════════════════════════════════════════════
# UNIVERSAL TA STRATEGY
# ═════════════════════════════════════════════════════════════════════════════

class UniversalTAStrategy:
    """
    TA strategy for any asset class. Configured via ASSET_CLASSES dict.

    Usage:
        strategy = UniversalTAStrategy("crypto")
        strategy = UniversalTAStrategy("futures")
        strategy = UniversalTAStrategy("fx")
    """

    def __init__(self, asset_class: str,
                 assets: list[str] | None = None,
                 ta_types: list[str] | None = None,
                 cache_dir: str | Path = "data/ta_cache",
                 training_start: str = "2025-01-01",
                 training_end: str = "2025-12-31",
                 **kwargs):
        if asset_class not in ASSET_CLASSES:
            raise ValueError(f"Unknown asset class: {asset_class}. "
                             f"Available: {list(ASSET_CLASSES.keys())}")

        self.config = ASSET_CLASSES[asset_class].copy()
        self.asset_class = asset_class
        self.ta_types = ta_types or TA_MODEL_TYPES
        self.cache_dir = Path(cache_dir) / asset_class
        self.training_start = training_start
        self.training_end = training_end

        # Override specific assets if provided
        if assets:
            self.config["assets"] = {a: self.config["assets"][a]
                                     for a in assets if a in self.config["assets"]}

        self.tc_bps = self.config["tc_bps"]
        self.benchmark = self.config["benchmark"]

        # Build model universe
        self._models = []
        for asset in self.config["assets"]:
            for ta_type in self.ta_types:
                self._models.append(TAModelSpec(
                    model_id=f"{asset}_{ta_type}",
                    asset=asset, ta_type=ta_type,
                    ticker=self.config["assets"][asset],
                ))

        self._grids_by_type = generate_ta_param_grid()
        self._all_configs = []
        for ta_type in self.ta_types:
            self._all_configs.extend(self._grids_by_type.get(ta_type, []))

        print(f"  {self.config['name']} TA: {len(self._models)} models "
              f"({len(self.config['assets'])} assets × {len(self.ta_types)} TA types)")
        print(f"  TC: {self.tc_bps} bps/leg, Benchmark: {self.benchmark}")
        print(f"  Total configs: {len(self._all_configs)}")

    # ── Protocol implementation ───────────────────────────────────

    def get_models(self) -> list[TAModelSpec]:
        return self._models

    def get_param_grid(self) -> list[TAConfig]:
        return self._all_configs

    @staticmethod
    def model_group_fn(model_id: str) -> str:
        return model_id.split("_", 1)[1]

    def daily_feature_names(self) -> list[str]:
        return list(TA_FEATURES)

    def config_param_names(self) -> list[str]:
        return TAConfig.param_names()

    def config_to_features(self, config: TAConfig) -> list[float]:
        return config.to_feature_vector()

    def compute_features(self, model: TAModelSpec, as_of_date: str,
                         data: dict) -> np.ndarray | None:
        hourly = data.get("hourly_data", {})
        bars = hourly.get(model.asset)
        if bars is None:
            return None
        bench_bars = hourly.get(self.benchmark) if model.asset != self.benchmark else None
        return compute_ta_features(bars, bench_bars, as_of_date)

    def run_single_day(self, model: TAModelSpec, config: TAConfig,
                       day: Any, data: dict) -> dict:
        hourly = data.get("hourly_data", {})
        bars = hourly.get(model.asset)
        if bars is None:
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

        day_start = pd.Timestamp(day)
        day_end = day_start + timedelta(hours=24)
        bars_day = bars[(bars.index >= day_start) & (bars.index < day_end)]

        if len(bars_day) < 6:
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

        hist_start = day_start - timedelta(days=7)
        bars_hist = bars[(bars.index >= hist_start) & (bars.index < day_start)]

        return run_ta_single_day(bars_day, config, bars_hist, self.tc_bps)

    def run_model_year(self, model: TAModelSpec, data: dict,
                       param_grid: list[TAConfig]) -> pd.DataFrame:
        hourly = data.get("hourly_data", {})
        bars = hourly.get(model.asset)
        if bars is None:
            return pd.DataFrame()

        model_configs = [c for c in param_grid if c.ta_type == model.ta_type]
        if not model_configs:
            return pd.DataFrame()

        trading_days = sorted(bars.index.normalize().unique())
        results = []

        for day_idx, day in enumerate(trading_days):
            day_start = pd.Timestamp(day)
            day_end = day_start + timedelta(hours=24)
            bars_day = bars[(bars.index >= day_start) & (bars.index < day_end)]

            if len(bars_day) < self.config["min_bars_per_day"]:
                continue

            hist_start = day_start - timedelta(days=7)
            bars_hist = bars[(bars.index >= hist_start) & (bars.index < day_start)]

            for config in model_configs:
                result = run_ta_single_day(bars_day, config, bars_hist, self.tc_bps)
                results.append({
                    "model_id": model.model_id,
                    "date": day.strftime("%Y-%m-%d"),
                    "config_id": config.config_id,
                    "daily_return": result["daily_return"],
                    "gross_return": result["gross_return"],
                    "n_trades": result["n_trades"],
                })

            if (day_idx + 1) % 50 == 0:
                print(f"    {model.model_id}: {day_idx+1}/{len(trading_days)} days")

        return pd.DataFrame(results)

    def fetch_training_data(self, models, start, end) -> dict:
        hourly = fetch_all_hourly(
            self.config, self.training_start, self.training_end, self.cache_dir)
        return {"hourly_data": hourly}

    def fetch_oos_data(self, models, start, end) -> dict:
        hourly = fetch_all_hourly(self.config, start, end, self.cache_dir)
        return {"hourly_data": hourly}

    def fetch_warmup_daily(self, models, start, end) -> dict[str, pd.Series]:
        hourly = fetch_all_hourly(self.config, start, end, self.cache_dir)
        daily = {}
        for asset, bars in hourly.items():
            d = bars["close"].resample("D").last().dropna()
            d.index = d.index.date
            daily[asset] = d
        return daily

    def get_daily_prices(self, data, models) -> dict[str, pd.Series]:
        hourly = data.get("hourly_data", {})
        daily = {}
        for asset, bars in hourly.items():
            d = bars["close"].resample("D").last().dropna()
            d.index = d.index.date
            daily[asset] = d
        return daily

    def get_trading_days(self, data) -> list:
        hourly = data.get("hourly_data", {})
        if not hourly:
            return []
        first_bars = next(iter(hourly.values()))
        days = sorted(first_bars.index.normalize().unique())
        min_bars = self.config["min_bars_per_day"]
        trading_days = []
        for day in days:
            day_start = pd.Timestamp(day)
            day_end = day_start + timedelta(hours=24)
            n = len(first_bars[(first_bars.index >= day_start) & (first_bars.index < day_end)])
            if n >= min_bars:
                trading_days.append(day)
        return trading_days

    def prepare_warmup(self, daily_prices, warmup_daily):
        for asset in daily_prices:
            if asset in warmup_daily:
                warmup = warmup_daily[asset]
                oos = daily_prices[asset]
                warmup_only = warmup[warmup.index < oos.index.min()]
                daily_prices[asset] = pd.concat([warmup_only, oos])
        return daily_prices
