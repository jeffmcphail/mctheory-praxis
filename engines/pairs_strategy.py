"""
Pairs Trading Strategy — Implementation of CPO TradingStrategy protocol.

Burgess pair discovery → Chan CPO intraday execution.
Single-asset trading (target only, hedge is signal).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engines.cpo_core import ModelSpec, ConfigSpec, TradingStrategy

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PairSpec(ModelSpec):
    """A cointegrated pair from Burgess Phase 1."""
    target: str = ""
    hedge: str = ""
    hedge_ratio: float = 0.0
    adf_t: float = 0.0
    adf_p: float = 1.0
    hurst: float = 0.5
    half_life: float = 10.0
    composite_score: float = 0.0


@dataclass
class PairConfig(ConfigSpec):
    """One parameter configuration for intraday trading."""
    lookback_minutes: int = 390
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0

    # Normalization constants
    _NORMS = {"lookback_minutes": 1950.0, "entry_z": 2.5, "exit_z": 0.75, "stop_z": 5.0}

    def to_feature_vector(self) -> list[float]:
        return [
            self.lookback_minutes / self._NORMS["lookback_minutes"],
            self.entry_z / self._NORMS["entry_z"],
            self.exit_z / self._NORMS["exit_z"],
            self.stop_z / self._NORMS["stop_z"],
        ]

    @staticmethod
    def param_names() -> list[str]:
        return ["lookback_minutes", "entry_z", "exit_z", "stop_z"]


# Daily lagged features
DAILY_FEATURES = [
    "spread_z_eod", "spread_vol_5d", "vol_ratio",
    "hurst_20d", "vr_5", "corr_20d",
    "spread_ret_5d", "intraday_vol",
]

# Known dual share class pairs to filter
STRUCTURAL_PAIRS = {
    frozenset({"GOOGL", "GOOG"}),
    frozenset({"FOXA", "FOX"}),
    frozenset({"NWSA", "NWS"}),
    frozenset({"BRK-A", "BRK-B"}),
    frozenset({"LBRDA", "LBRDK"}),
}


# ═════════════════════════════════════════════════════════════════════════════
# PAIRS STRATEGY
# ═════════════════════════════════════════════════════════════════════════════

class PairsStrategy:
    """
    Pairs trading strategy implementing the CPO TradingStrategy protocol.

    Pair discovery via Burgess (Phase 1, external).
    Intraday mean-reversion on minute bars, single-asset execution.
    """

    def __init__(self, pairs_json: str | Path, api_key: str,
                 cache_dir: str | Path = "data/minute_cache",
                 top_n: int = 50,
                 tc_bps: float = 2.0,
                 training_start: str = "2025-01-01",
                 training_end: str = "2025-12-31"):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.top_n = top_n
        self.tc_bps = tc_bps
        self.training_start = training_start
        self.training_end = training_end

        self._pairs = self._load_pairs(pairs_json)
        self._param_grid = self._generate_param_grid()
        self._data_cache = {}

    def _load_pairs(self, json_path: str | Path) -> list[PairSpec]:
        """Load and deduplicate pairs from Burgess output."""
        with open(json_path) as f:
            data = json.load(f)

        raw = []
        for b in data["ranked_baskets"][:self.top_n * 3]:
            if len(b["basket"]) != 1:
                continue
            raw.append(PairSpec(
                model_id=f"{b['target']}_{b['basket'][0]}",
                target=b["target"],
                hedge=b["basket"][0],
                hedge_ratio=b["hedge_ratio"][0],
                adf_t=b["adf_t"], adf_p=b["adf_p"],
                hurst=b["hurst"], half_life=b["half_life"],
                composite_score=b["composite_score"],
            ))

        # Filter structural arb
        n_struct = 0
        filtered = []
        for p in raw:
            if frozenset({p.target, p.hedge}) in STRUCTURAL_PAIRS:
                n_struct += 1
                continue
            filtered.append(p)

        # Deduplicate
        seen = {}
        n_dedup = 0
        for p in filtered:
            key = tuple(sorted([p.target, p.hedge]))
            if key in seen:
                if p.adf_t < seen[key].adf_t:
                    seen[key] = p
                n_dedup += 1
            else:
                seen[key] = p

        pairs = list(seen.values())[:self.top_n]
        if n_struct > 0 or n_dedup > 0:
            print(f"  Pair filtering: {n_struct} structural arb removed, "
                  f"{n_dedup} duplicates removed, {len(pairs)} unique pairs")
        return pairs

    def _generate_param_grid(self) -> list[PairConfig]:
        lookbacks = [195, 390, 780, 1170, 1950]
        entries = [1.0, 1.5, 2.0, 2.5]
        exits = [0.0, 0.25, 0.5, 0.75]
        stops = [3.0, 4.0, 5.0]
        configs = []
        cid = 0
        for lb in lookbacks:
            for entry in entries:
                for exit_ in exits:
                    if exit_ >= entry:
                        continue
                    for stop in stops:
                        if stop <= entry:
                            continue
                        configs.append(PairConfig(
                            config_id=cid, lookback_minutes=lb,
                            entry_z=entry, exit_z=exit_, stop_z=stop,
                        ))
                        cid += 1
        return configs

    # ── Protocol implementation ───────────────────────────────────

    def get_models(self) -> list[PairSpec]:
        return self._pairs

    def get_param_grid(self) -> list[PairConfig]:
        return self._param_grid

    def daily_feature_names(self) -> list[str]:
        return list(DAILY_FEATURES)

    def config_param_names(self) -> list[str]:
        return PairConfig.param_names()

    def config_to_features(self, config: PairConfig) -> list[float]:
        return config.to_feature_vector()

    def compute_features(self, model: PairSpec, as_of_date: str,
                         data: dict) -> np.ndarray | None:
        """Compute lagged daily features for a pair."""
        daily_prices = data.get("daily_prices", {})
        minute_data = data.get("minute_data", {})

        tgt = daily_prices.get(model.target)
        hdg = daily_prices.get(model.hedge)
        if tgt is None or hdg is None:
            return None

        return _compute_pair_features(model, tgt, hdg, minute_data, as_of_date)

    def run_single_day(self, model: PairSpec, config: PairConfig,
                       day: Any, data: dict) -> dict:
        """Execute one pair+config for one day on minute bars."""
        minute_data = data.get("minute_data", {})
        target_bars = minute_data.get(model.target)
        hedge_bars = minute_data.get(model.hedge)
        if target_bars is None or hedge_bars is None:
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

        spread_df = _construct_spread(target_bars, hedge_bars, model.hedge_ratio)
        spread = spread_df["spread"]

        day_start = day.replace(hour=9, minute=30)
        day_end = day.replace(hour=16, minute=0)
        spread_day = spread[(spread.index >= day_start) & (spread.index <= day_end)]

        if len(spread_day) < 30:
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

        # Notional
        tgt_open = spread_df["target_close"][(spread_df.index >= day_start) & (spread_df.index <= day_end)]
        hdg_open = spread_df["hedge_close"][(spread_df.index >= day_start) & (spread_df.index <= day_end)]
        if len(tgt_open) > 0 and len(hdg_open) > 0:
            notional = float(tgt_open.iloc[0]) + abs(model.hedge_ratio) * float(hdg_open.iloc[0])
        else:
            notional = 1.0

        # Lookback for z-score warmup
        hist_start = day - timedelta(days=20)
        spread_hist = spread[(spread.index >= hist_start) & (spread.index < day_start)]

        return _run_intraday(spread_day, config, spread_hist, notional, self.tc_bps)

    def run_model_year(self, model: PairSpec, data: dict,
                       param_grid: list[PairConfig]) -> pd.DataFrame:
        """Run all configs for all days for one pair."""
        minute_data = data.get("minute_data", {})
        target_bars = minute_data.get(model.target)
        hedge_bars = minute_data.get(model.hedge)
        if target_bars is None or hedge_bars is None:
            return pd.DataFrame()

        spread_df = _construct_spread(target_bars, hedge_bars, model.hedge_ratio)
        spread = spread_df["spread"]
        target_close = spread_df["target_close"]
        hedge_close = spread_df["hedge_close"]

        trading_days = sorted(spread.index.normalize().unique())
        results = []

        for day_idx, day in enumerate(trading_days):
            day_start = day.replace(hour=9, minute=30)
            day_end = day.replace(hour=16, minute=0)
            spread_day = spread[(spread.index >= day_start) & (spread.index <= day_end)]

            if len(spread_day) < 30:
                continue

            tgt_open = target_close[(target_close.index >= day_start) & (target_close.index <= day_end)]
            hdg_open = hedge_close[(hedge_close.index >= day_start) & (hedge_close.index <= day_end)]
            if len(tgt_open) > 0 and len(hdg_open) > 0:
                notional = float(tgt_open.iloc[0]) + abs(model.hedge_ratio) * float(hdg_open.iloc[0])
            else:
                notional = 1.0

            hist_start = day - timedelta(days=20)
            spread_hist = spread[(spread.index >= hist_start) & (spread.index < day_start)]

            for config in param_grid:
                result = _run_intraday(spread_day, config, spread_hist, notional, self.tc_bps)
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
        """Fetch minute data for training period."""
        minute_data = _fetch_all_minute(
            self._pairs, self.training_start, self.training_end,
            self.api_key, self.cache_dir
        )
        daily_prices = _aggregate_daily(minute_data)
        return {"minute_data": minute_data, "daily_prices": daily_prices}

    def fetch_oos_data(self, models, start, end) -> dict:
        minute_data = _fetch_all_minute(
            self._pairs, start, end, self.api_key, self.cache_dir
        )
        daily_prices = _aggregate_daily(minute_data)
        return {"minute_data": minute_data, "daily_prices": daily_prices}

    def fetch_warmup_daily(self, models, start, end) -> dict[str, pd.Series]:
        """Fetch daily close prices for warmup."""
        import requests
        tickers = set()
        for p in self._pairs:
            tickers.add(p.target)
            tickers.add(p.hedge)

        daily = {}
        for i, ticker in enumerate(sorted(tickers)):
            try:
                url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
                       f"{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}")
                resp = requests.get(url, timeout=15)
                data = resp.json()
                results = data.get("results", [])
                if results:
                    dates = [pd.Timestamp(r["t"], unit="ms").date() for r in results]
                    closes = [r["c"] for r in results]
                    daily[ticker] = pd.Series(closes, index=dates, name=ticker)
            except Exception:
                pass
            if (i + 1) % 50 == 0:
                print(f"    Warmup: {i+1}/{len(tickers)} loaded")
            time.sleep(0.12)
        return daily

    def get_daily_prices(self, data: dict, models) -> dict[str, pd.Series]:
        return data.get("daily_prices", {})

    def get_trading_days(self, data: dict) -> list:
        minute_data = data.get("minute_data", {})
        all_days = sorted(set().union(*(
            set(bars.index.normalize().unique()) for bars in minute_data.values()
        )))
        trading_days = []
        tickers = list(minute_data.keys())[:5]
        for day in all_days:
            if day.weekday() >= 5:
                continue
            for t in tickers:
                day_start = day.replace(hour=9, minute=30)
                day_end = day.replace(hour=16, minute=0)
                n = len(minute_data[t][(minute_data[t].index >= day_start) &
                                       (minute_data[t].index <= day_end)])
                if n >= 100:
                    trading_days.append(day)
                    break
        return trading_days

    def prepare_warmup(self, daily_prices: dict, warmup_daily: dict) -> dict:
        for ticker in daily_prices:
            if ticker in warmup_daily:
                warmup = warmup_daily[ticker]
                oos = daily_prices[ticker]
                warmup_only = warmup[warmup.index < oos.index.min()]
                daily_prices[ticker] = pd.concat([warmup_only, oos])
        return daily_prices


# ═════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_minute_polygon(ticker, start, end, api_key, cache_dir):
    """Fetch minute bars with UTC→Eastern conversion and caching."""
    import requests
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{ticker}_{start}_{end}_1min.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            if df.index.tz is not None:
                df.index = df.index.tz_convert("US/Eastern").tz_localize(None)
            else:
                df.index = (pd.DatetimeIndex(df.index).tz_localize("UTC")
                           .tz_convert("US/Eastern").tz_localize(None))
            return df

    bars = []
    current_start = pd.Timestamp(start)
    final_end = pd.Timestamp(end)
    while current_start < final_end:
        chunk_end = min(current_start + timedelta(days=60), final_end)
        req_url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/"
                   f"{current_start.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}"
                   f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}")
        resp = requests.get(req_url, timeout=30)
        data = resp.json()
        if data.get("results"):
            bars.extend(data["results"])
        current_start = chunk_end + timedelta(days=1)
        time.sleep(0.25)

    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["datetime"] = df["datetime"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume"})
    df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="last")]

    if cache_dir:
        df.to_parquet(cache_path)
    return df


def _fetch_all_minute(pairs, start, end, api_key, cache_dir):
    """Fetch minute bars for all unique tickers across pairs."""
    tickers = set()
    for p in pairs:
        tickers.add(p.target)
        tickers.add(p.hedge)
    tickers = sorted(tickers)

    print(f"  Fetching minute data for {len(tickers)} tickers: {start} -> {end}")
    data = {}
    for i, ticker in enumerate(tickers):
        df = _fetch_minute_polygon(ticker, start, end, api_key, cache_dir)
        if not df.empty:
            data[ticker] = df
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(tickers)} fetched ({len(df)} bars for {ticker})")
    print(f"  Loaded minute data for {len(data)}/{len(tickers)} tickers")
    return data


def _aggregate_daily(minute_data):
    """Build daily close prices from minute bars."""
    daily = {}
    for ticker, bars in minute_data.items():
        d = bars["close"].resample("D").last().dropna()
        d.index = d.index.date
        daily[ticker] = d
    return daily


def _construct_spread(target_bars, hedge_bars, hedge_ratio):
    """Build spread = target - HR × hedge on minute bars."""
    common = target_bars.index.intersection(hedge_bars.index)
    t_close = target_bars.loc[common, "close"]
    h_close = hedge_bars.loc[common, "close"]
    return pd.DataFrame({
        "target_close": t_close,
        "hedge_close": h_close,
        "spread": t_close - hedge_ratio * h_close,
    }, index=common)


def _run_intraday(spread_day, config, spread_history, notional, tc_bps):
    """Run one config on one day of minute-bar spread data."""
    tc_per_trade = 2 * tc_bps / 10000.0 * notional

    if spread_history is not None and len(spread_history) > 0:
        full_spread = pd.concat([spread_history, spread_day])
    else:
        full_spread = spread_day

    lb = config.lookback_minutes
    rolling_mean = full_spread.rolling(lb, min_periods=max(lb // 2, 30)).mean()
    rolling_std = full_spread.rolling(lb, min_periods=max(lb // 2, 30)).std()
    z_scores = (full_spread - rolling_mean) / (rolling_std + 1e-10)

    today_mask = full_spread.index.isin(spread_day.index)
    z_today = z_scores[today_mask].values
    spread_today = full_spread[today_mask].values

    if len(z_today) < 10:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    position = 0
    entry_price = 0.0
    trades = []
    pnl = 0.0
    gross_pnl = 0.0

    for i in range(len(z_today)):
        z = z_today[i]
        price = spread_today[i]
        if np.isnan(z):
            continue

        if position == 0:
            if z <= -config.entry_z:
                position = 1
                entry_price = price
            elif z >= config.entry_z:
                position = -1
                entry_price = price
        elif position == 1:
            close = False
            if z >= -config.exit_z:
                close = True
            elif z <= -config.stop_z:
                close = True
            if close:
                gross = price - entry_price
                pnl += gross - tc_per_trade
                gross_pnl += gross
                trades.append(gross - tc_per_trade)
                position = 0
        elif position == -1:
            close = False
            if z <= config.exit_z:
                close = True
            elif z >= config.stop_z:
                close = True
            if close:
                gross = entry_price - price
                pnl += gross - tc_per_trade
                gross_pnl += gross
                trades.append(gross - tc_per_trade)
                position = 0

    if position != 0:
        price = spread_today[-1]
        gross = (price - entry_price) if position == 1 else (entry_price - price)
        pnl += gross - tc_per_trade
        gross_pnl += gross
        trades.append(gross - tc_per_trade)

    return {
        "daily_return": pnl / notional if notional > 0 else 0.0,
        "gross_return": gross_pnl / notional if notional > 0 else 0.0,
        "n_trades": len(trades),
    }


def _compute_pair_features(pair, daily_target, daily_hedge, minute_data, as_of_date):
    """Compute lagged features for one pair as of a given date."""
    dt = pd.Timestamp(as_of_date).date()

    tgt = daily_target[daily_target.index <= dt]
    hdg = daily_hedge[daily_hedge.index <= dt]
    if len(tgt) < 25 or len(hdg) < 25:
        return None

    common = tgt.index.intersection(hdg.index)
    tgt_c = tgt.loc[common].values
    hdg_c = hdg.loc[common].values
    spread = tgt_c - pair.hedge_ratio * hdg_c

    if len(spread) < 25:
        return None

    s20 = spread[-20:]
    spread_z_eod = (spread[-1] - np.mean(s20)) / (np.std(s20) + 1e-10)

    spread_ret = np.diff(spread) / (np.abs(spread[:-1]) + 1e-10)
    vol_5d = np.std(spread_ret[-5:]) if len(spread_ret) >= 5 else 0
    vol_20d = np.std(spread_ret[-20:]) if len(spread_ret) >= 20 else vol_5d
    vol_ratio = vol_5d / (vol_20d + 1e-10)

    # Hurst
    hurst_20d = _rolling_hurst(spread[-20:])

    # Variance ratio
    vr_5 = _variance_ratio(spread[-30:], 5) if len(spread) >= 30 else 1.0

    # Correlation
    if len(common) >= 20:
        tgt_ret = np.diff(tgt_c[-21:]) / (np.abs(tgt_c[-21:-1]) + 1e-10)
        hdg_ret = np.diff(hdg_c[-21:]) / (np.abs(hdg_c[-21:-1]) + 1e-10)
        corr_20d = np.corrcoef(tgt_ret, hdg_ret)[0, 1] if len(tgt_ret) == len(hdg_ret) else 0
    else:
        corr_20d = 0

    # Spread momentum
    spread_ret_5d = (spread[-1] - spread[-6]) / (np.std(s20) + 1e-10) if len(spread) >= 6 else 0

    # Intraday features
    intraday_vol = 0.0
    target_min = minute_data.get(pair.target)
    hedge_min = minute_data.get(pair.hedge)
    if target_min is not None and hedge_min is not None:
        for offset in range(1, 5):
            check = dt - timedelta(days=offset)
            check_ts = pd.Timestamp(check)
            t_day = target_min[target_min.index.normalize() == check_ts]
            if len(t_day) > 30:
                h_day = hedge_min[hedge_min.index.normalize() == check_ts]
                common_min = t_day.index.intersection(h_day.index)
                if len(common_min) > 30:
                    intra = (t_day.loc[common_min, "close"].values -
                             pair.hedge_ratio * h_day.loc[common_min, "close"].values)
                    intraday_vol = np.std(np.diff(intra)) / (np.mean(np.abs(intra)) + 1e-10)
                break

    return np.array([
        float(np.nan_to_num(spread_z_eod)),
        float(np.nan_to_num(vol_5d)),
        float(np.nan_to_num(vol_ratio)),
        float(np.nan_to_num(hurst_20d)),
        float(np.nan_to_num(vr_5)),
        float(np.nan_to_num(corr_20d)),
        float(np.nan_to_num(spread_ret_5d)),
        float(np.nan_to_num(intraday_vol)),
    ])


def _variance_ratio(series, lag):
    if len(series) < lag * 2:
        return 1.0
    returns = np.diff(series)
    if len(returns) < lag:
        return 1.0
    var1 = np.var(returns)
    if var1 < 1e-12:
        return 1.0
    returns_lag = series[lag:] - series[:-lag]
    return np.var(returns_lag) / (lag * var1)


def _rolling_hurst(series, max_lag=20):
    n = len(series)
    if n < max_lag * 2:
        return 0.5
    lags = range(2, min(max_lag, n // 2))
    rs_values = []
    for lag in lags:
        rs_list = []
        for start in range(0, n - lag, lag):
            chunk = series[start:start + lag]
            if len(chunk) < 2:
                continue
            mean_adj = chunk - np.mean(chunk)
            cumdev = np.cumsum(mean_adj)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(chunk, ddof=1)
            if S > 1e-10:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append((np.log(lag), np.log(np.mean(rs_list))))
    if len(rs_values) < 3:
        return 0.5
    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])
    slope = np.polyfit(x, y, 1)[0]
    if np.isnan(slope):
        return 0.5
    return np.clip(slope, 0.0, 1.0)
