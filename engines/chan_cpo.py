"""
ENGINE 2: Chan Conditional Portfolio Optimization — Minute-Bar Trading.

Combines Burgess pair discovery with Chan CPO framework:
    Phase 1 — Pair discovery via Burgess (daily, n=2, ADF + VR eigen2)
    Phase 2 — Parameter grid search on minute data (2025 training year)
    Phase 3 — Random Forest: lagged features → optimal param config
    Phase 4 — Kelly Vector portfolio of RF-selected models (2026 OOS)

Each "model" = (pair, param_config). RF learns which config to use each
day based on previous day's lagged features. Kelly allocates capital
across the active models.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PairSpec:
    """A cointegrated pair from Burgess Phase 1."""
    pair_id: str            # e.g. "RSG_WM"
    target: str             # target ticker (laggard)
    hedge: str              # hedge ticker
    hedge_ratio: float      # beta from cointegration regression
    adf_t: float
    adf_p: float
    hurst: float
    half_life: float
    composite_score: float


@dataclass
class ParamConfig:
    """One parameter configuration for intraday trading."""
    config_id: int
    lookback_minutes: int   # z-score rolling window (in minutes)
    entry_z: float          # enter when |z| > entry_z
    exit_z: float           # exit when |z| < exit_z
    stop_z: float           # stop loss when |z| > stop_z

    def __repr__(self):
        return (f"Config({self.config_id}: lb={self.lookback_minutes}, "
                f"entry={self.entry_z}, exit={self.exit_z}, stop={self.stop_z})")


@dataclass
class DailyFeatures:
    """Lagged features from previous day — inputs to the Random Forest."""
    pair_id: str
    date: str
    # Spread features
    spread_z_eod: float         # end-of-day spread z-score
    spread_vol_5d: float        # 5-day spread volatility (daily returns)
    spread_vol_20d: float       # 20-day spread volatility
    vol_ratio: float            # 5d / 20d vol ratio (vol regime)
    spread_ma_dev: float        # spread deviation from 20d MA
    # Mean-reversion features
    hurst_20d: float            # rolling 20-day Hurst estimate
    vr_5: float                 # variance ratio lag 5 (daily)
    vr_10: float                # variance ratio lag 10 (daily)
    # Correlation / relationship
    corr_20d: float             # 20-day rolling correlation
    corr_5d: float              # 5-day rolling correlation
    beta_20d: float             # rolling 20-day hedge ratio
    # Momentum
    target_ret_5d: float        # 5-day target return
    hedge_ret_5d: float         # 5-day hedge return
    spread_ret_5d: float        # 5-day spread return
    # Intraday features (from previous day's minute bars)
    intraday_vol: float         # previous day intraday spread vol
    intraday_range: float       # (max - min) spread / std
    intraday_trades_est: int    # estimated number of z-crossings

    def to_array(self) -> np.ndarray:
        """Convert to feature vector for RF."""
        return np.array([
            self.spread_z_eod, self.spread_vol_5d, self.spread_vol_20d,
            self.vol_ratio, self.spread_ma_dev,
            self.hurst_20d, self.vr_5, self.vr_10,
            self.corr_20d, self.corr_5d, self.beta_20d,
            self.target_ret_5d, self.hedge_ret_5d, self.spread_ret_5d,
            self.intraday_vol, self.intraday_range, self.intraday_trades_est,
        ])

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "spread_z_eod", "spread_vol_5d", "spread_vol_20d",
            "vol_ratio", "spread_ma_dev",
            "hurst_20d", "vr_5", "vr_10",
            "corr_20d", "corr_5d", "beta_20d",
            "target_ret_5d", "hedge_ret_5d", "spread_ret_5d",
            "intraday_vol", "intraday_range", "intraday_trades_est",
        ]


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETER GRID
# ═════════════════════════════════════════════════════════════════════════════

def generate_param_grid() -> list[ParamConfig]:
    """
    Generate all parameter combinations for intraday trading.

    Lookback in minutes: ~1 day to ~10 days of market time (390 min/day).
    Entry/exit z-scores: standard mean-reversion ranges.
    """
    lookbacks = [195, 390, 780, 1170, 1950]   # 0.5d, 1d, 2d, 3d, 5d
    entries   = [1.0, 1.5, 2.0, 2.5]
    exits     = [0.0, 0.25, 0.5, 0.75]
    stops     = [3.0, 4.0, 5.0]

    configs = []
    cid = 0
    for lb in lookbacks:
        for entry in entries:
            for exit_ in exits:
                if exit_ >= entry:
                    continue  # exit must be inside entry
                for stop in stops:
                    if stop <= entry:
                        continue  # stop must be outside entry
                    configs.append(ParamConfig(
                        config_id=cid,
                        lookback_minutes=lb,
                        entry_z=entry,
                        exit_z=exit_,
                        stop_z=stop,
                    ))
                    cid += 1
    return configs


# ═════════════════════════════════════════════════════════════════════════════
# MINUTE DATA PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def fetch_minute_bars_polygon(ticker: str, start: str, end: str,
                              api_key: str, cache_dir: Path | None = None
                              ) -> pd.DataFrame:
    """
    Fetch minute bars from Polygon.io. Caches to parquet per ticker.

    Returns DataFrame with DatetimeIndex and columns:
        open, high, low, close, volume
    """
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{ticker}_{start}_{end}_1min.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            # All Polygon timestamps are UTC; convert cached naive-UTC to Eastern
            if df.index.tz is not None:
                df.index = df.index.tz_convert("US/Eastern").tz_localize(None)
            else:
                df.index = (pd.DatetimeIndex(df.index).tz_localize("UTC")
                            .tz_convert("US/Eastern").tz_localize(None))
            return df

    import requests
    bars = []
    url = "https://api.polygon.io/v2/aggs/ticker"
    # Polygon allows up to 50,000 bars per request
    # 1 year of minute bars ≈ 252 * 390 ≈ 98,280 → need pagination
    current_start = pd.Timestamp(start)
    final_end = pd.Timestamp(end)

    while current_start < final_end:
        chunk_end = min(current_start + timedelta(days=60), final_end)
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": api_key,
        }
        req_url = (f"{url}/{ticker}/range/1/minute/"
                   f"{current_start.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}")
        resp = requests.get(req_url, params=params, timeout=30)
        data = resp.json()
        if data.get("results"):
            bars.extend(data["results"])
        current_start = chunk_end + timedelta(days=1)
        time.sleep(0.25)  # rate limiting

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


def fetch_all_minute_data(pairs: list[PairSpec], start: str, end: str,
                          api_key: str, cache_dir: Path
                          ) -> dict[str, pd.DataFrame]:
    """Fetch minute bars for all unique tickers across all pairs."""
    tickers = set()
    for p in pairs:
        tickers.add(p.target)
        tickers.add(p.hedge)
    tickers = sorted(tickers)

    print(f"  Fetching minute data for {len(tickers)} tickers: {start} -> {end}")
    data = {}
    for i, ticker in enumerate(tickers):
        df = fetch_minute_bars_polygon(ticker, start, end, api_key, cache_dir)
        if not df.empty:
            data[ticker] = df
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(tickers)} fetched ({len(df)} bars for {ticker})")

    print(f"  Loaded minute data for {len(data)}/{len(tickers)} tickers")
    return data


# ═════════════════════════════════════════════════════════════════════════════
# SPREAD CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════

def construct_minute_spread(target_bars: pd.DataFrame,
                            hedge_bars: pd.DataFrame,
                            hedge_ratio: float) -> pd.DataFrame:
    """
    Build spread = target_close - hedge_ratio * hedge_close on minute bars.
    Aligns on shared timestamps.
    """
    # Align on common timestamps
    common_idx = target_bars.index.intersection(hedge_bars.index)
    t_close = target_bars.loc[common_idx, "close"]
    h_close = hedge_bars.loc[common_idx, "close"]
    spread = t_close - hedge_ratio * h_close

    df = pd.DataFrame({
        "target_close": t_close,
        "hedge_close": h_close,
        "spread": spread,
    }, index=common_idx)
    return df


def construct_daily_spread(target_daily: pd.Series,
                           hedge_daily: pd.Series,
                           hedge_ratio: float) -> pd.Series:
    """Build daily spread from daily close prices."""
    common = target_daily.index.intersection(hedge_daily.index)
    return target_daily.loc[common] - hedge_ratio * hedge_daily.loc[common]


# ═════════════════════════════════════════════════════════════════════════════
# INTRADAY TRADING ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def run_intraday_single_day(spread_day: pd.Series, config: ParamConfig,
                            spread_history: pd.Series | None = None,
                            notional_capital: float = 1.0,
                            tc_bps: float = 2.0,
                            ) -> dict:
    """
    Run one parameter config on one day of minute-bar spread data.

    Uses rolling z-score for signals. If spread_history is provided,
    it's prepended for lookback warmup.

    notional_capital: gross capital per unit of spread
        = target_price + |hedge_ratio| * hedge_price at day open.
    tc_bps: one-way transaction cost in basis points per leg.
        Default 2.0 bps for liquid S&P 500 names (IBKR-style):
          ~0.5 bps commission + ~1.5 bps half-spread.
        Single-asset execution: trade target only (hedge is signal only).
        Each round-trip = 2 leg executions (entry + exit) × tc_bps.

    Returns dict with daily_return (as % of notional), n_trades, etc.
    """
    tc_per_trade = 2 * tc_bps / 10000.0 * notional_capital  # 2 legs per round trip
    if spread_history is not None and len(spread_history) > 0:
        full_spread = pd.concat([spread_history, spread_day])
    else:
        full_spread = spread_day

    # Rolling z-score
    lb = config.lookback_minutes
    rolling_mean = full_spread.rolling(lb, min_periods=max(lb // 2, 30)).mean()
    rolling_std = full_spread.rolling(lb, min_periods=max(lb // 2, 30)).std()
    z_scores = (full_spread - rolling_mean) / (rolling_std + 1e-10)

    # Only trade during today's window
    today_mask = full_spread.index.isin(spread_day.index)
    z_today = z_scores[today_mask].values
    spread_today = full_spread[today_mask].values

    if len(z_today) < 10:
        return {"daily_return": 0.0, "n_trades": 0, "max_position": 0}

    # Trading simulation
    position = 0  # +1 long spread, -1 short spread, 0 flat
    entry_price = 0.0
    trades = []
    pnl = 0.0           # net (post-TC)
    gross_pnl = 0.0     # gross (pre-TC)
    total_costs = 0.0

    for i in range(len(z_today)):
        z = z_today[i]
        price = spread_today[i]

        if np.isnan(z):
            continue

        # Entry
        if position == 0:
            if z <= -config.entry_z:
                position = 1  # long spread (cheap)
                entry_price = price
            elif z >= config.entry_z:
                position = -1  # short spread (expensive)
                entry_price = price

        # Exit
        elif position == 1:
            if z >= -config.exit_z:  # mean reverted
                gross_trade = price - entry_price
                trade_pnl = gross_trade - tc_per_trade
                pnl += trade_pnl
                gross_pnl += gross_trade
                trades.append(trade_pnl)
                total_costs += tc_per_trade
                position = 0
            elif z <= -config.stop_z:  # stop loss
                gross_trade = price - entry_price
                trade_pnl = gross_trade - tc_per_trade
                pnl += trade_pnl
                gross_pnl += gross_trade
                trades.append(trade_pnl)
                total_costs += tc_per_trade
                position = 0

        elif position == -1:
            if z <= config.exit_z:  # mean reverted
                gross_trade = entry_price - price
                trade_pnl = gross_trade - tc_per_trade
                pnl += trade_pnl
                gross_pnl += gross_trade
                trades.append(trade_pnl)
                total_costs += tc_per_trade
                position = 0
            elif z >= config.stop_z:  # stop loss
                gross_trade = entry_price - price
                trade_pnl = gross_trade - tc_per_trade
                pnl += trade_pnl
                gross_pnl += gross_trade
                trades.append(trade_pnl)
                total_costs += tc_per_trade
                position = 0

    # Force close at end of day (no overnight)
    if position != 0:
        price = spread_today[-1]
        if position == 1:
            gross_trade = price - entry_price
        else:
            gross_trade = entry_price - price
        trade_pnl = gross_trade - tc_per_trade
        pnl += trade_pnl
        gross_pnl += gross_trade
        trades.append(trade_pnl)
        total_costs += tc_per_trade

    # Normalize returns as percentage of notional capital
    daily_return = pnl / notional_capital if notional_capital > 0 else 0.0
    gross_return = gross_pnl / notional_capital if notional_capital > 0 else 0.0

    return {
        "daily_return": daily_return,    # net (post-TC) % return
        "gross_return": gross_return,    # gross (pre-TC) % return
        "raw_pnl": pnl,
        "n_trades": len(trades),
        "total_costs": total_costs,
        "max_position": 1 if trades else 0,
        "trades": trades,
    }


def run_pair_year(pair: PairSpec, minute_data: dict[str, pd.DataFrame],
                  param_grid: list[ParamConfig],
                  lookback_days: int = 10
                  ) -> pd.DataFrame:
    """
    Run all parameter configs for all trading days of a pair.

    Returns DataFrame: rows = (date, config_id), cols = metrics.
    """
    target_bars = minute_data.get(pair.target)
    hedge_bars = minute_data.get(pair.hedge)
    if target_bars is None or hedge_bars is None:
        return pd.DataFrame()

    # Build full minute spread
    spread_df = construct_minute_spread(target_bars, hedge_bars, pair.hedge_ratio)
    spread = spread_df["spread"]
    target_close = spread_df["target_close"]
    hedge_close = spread_df["hedge_close"]

    # Get unique trading days
    trading_days = sorted(spread.index.normalize().unique())

    results = []
    for day_idx, day in enumerate(trading_days):
        day_start = day.replace(hour=9, minute=30)
        day_end = day.replace(hour=16, minute=0)
        spread_day = spread[(spread.index >= day_start) & (spread.index <= day_end)]

        if len(spread_day) < 30:
            continue

        # Notional capital = target_price + |HR| * hedge_price at day open
        tgt_open = target_close[(target_close.index >= day_start) & (target_close.index <= day_end)]
        hdg_open = hedge_close[(hedge_close.index >= day_start) & (hedge_close.index <= day_end)]
        if len(tgt_open) > 0 and len(hdg_open) > 0:
            notional = float(tgt_open.iloc[0]) + abs(pair.hedge_ratio) * float(hdg_open.iloc[0])
        else:
            notional = 1.0

        # Lookback history for z-score warmup
        hist_start = day - timedelta(days=lookback_days * 2)  # extra buffer for weekends
        spread_hist = spread[(spread.index >= hist_start) & (spread.index < day_start)]

        for config in param_grid:
            result = run_intraday_single_day(spread_day, config, spread_hist, notional)
            results.append({
                "pair_id": pair.pair_id,
                "date": day.strftime("%Y-%m-%d"),
                "config_id": config.config_id,
                "daily_return": result["daily_return"],
                "gross_return": result["gross_return"],
                "raw_pnl": result["raw_pnl"],
                "n_trades": result["n_trades"],
            })

        if (day_idx + 1) % 50 == 0:
            print(f"    {pair.pair_id}: {day_idx+1}/{len(trading_days)} days")

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════════════
# DAILY FEATURE COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def _variance_ratio(series: np.ndarray, lag: int) -> float:
    """Variance ratio test statistic."""
    if len(series) < lag * 2:
        return 1.0
    returns = np.diff(series)
    if len(returns) < lag:
        return 1.0
    var1 = np.var(returns)
    if var1 < 1e-12:
        return 1.0
    returns_lag = series[lag:] - series[:-lag]
    var_lag = np.var(returns_lag)
    return (var_lag / (lag * var1))


def _rolling_hurst(series: np.ndarray, max_lag: int = 20) -> float:
    """Simple Hurst exponent via R/S method."""
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


def compute_daily_features(pair: PairSpec,
                           daily_target: pd.Series,
                           daily_hedge: pd.Series,
                           minute_data: dict[str, pd.DataFrame],
                           as_of_date: str) -> DailyFeatures | None:
    """
    Compute all lagged features as of end-of-day for a given date.
    Uses data STRICTLY before as_of_date (no lookahead).
    """
    dt = pd.Timestamp(as_of_date).date()  # compare as date, not Timestamp

    # Daily prices up to (and including) as_of_date
    tgt = daily_target[daily_target.index <= dt]
    hdg = daily_hedge[daily_hedge.index <= dt]
    if len(tgt) < 25 or len(hdg) < 25:
        return None

    # Daily spread
    common = tgt.index.intersection(hdg.index)
    tgt_c = tgt.loc[common].values
    hdg_c = hdg.loc[common].values
    spread_daily = tgt_c - pair.hedge_ratio * hdg_c

    if len(spread_daily) < 25:
        return None

    # Spread z-score (20-day)
    s20 = spread_daily[-20:]
    spread_z_eod = ((spread_daily[-1] - np.mean(s20)) /
                    (np.std(s20) + 1e-10))

    # Spread volatility
    spread_ret = np.diff(spread_daily) / (np.abs(spread_daily[:-1]) + 1e-10)
    vol_5d = np.std(spread_ret[-5:]) if len(spread_ret) >= 5 else 0
    vol_20d = np.std(spread_ret[-20:]) if len(spread_ret) >= 20 else vol_5d
    vol_ratio = vol_5d / (vol_20d + 1e-10)

    # Spread MA deviation
    ma_20 = np.mean(spread_daily[-20:])
    spread_ma_dev = (spread_daily[-1] - ma_20) / (np.std(spread_daily[-20:]) + 1e-10)

    # Hurst (rolling 20-day)
    hurst_20d = _rolling_hurst(spread_daily[-20:])

    # Variance ratios
    vr_5 = _variance_ratio(spread_daily[-30:], 5) if len(spread_daily) >= 30 else 1.0
    vr_10 = _variance_ratio(spread_daily[-30:], 10) if len(spread_daily) >= 30 else 1.0

    # Rolling correlation
    if len(common) >= 20:
        tgt_ret = np.diff(tgt_c[-21:]) / (np.abs(tgt_c[-21:-1]) + 1e-10)
        hdg_ret = np.diff(hdg_c[-21:]) / (np.abs(hdg_c[-21:-1]) + 1e-10)
        corr_20d = np.corrcoef(tgt_ret, hdg_ret)[0, 1] if len(tgt_ret) == len(hdg_ret) else 0
    else:
        corr_20d = 0

    if len(common) >= 5:
        tgt_r5 = np.diff(tgt_c[-6:]) / (np.abs(tgt_c[-6:-1]) + 1e-10)
        hdg_r5 = np.diff(hdg_c[-6:]) / (np.abs(hdg_c[-6:-1]) + 1e-10)
        corr_5d = np.corrcoef(tgt_r5, hdg_r5)[0, 1] if len(tgt_r5) == len(hdg_r5) else 0
    else:
        corr_5d = 0

    # Rolling beta
    if len(common) >= 20:
        cov = np.cov(tgt_ret, hdg_ret)
        beta_20d = cov[0, 1] / (cov[1, 1] + 1e-10) if cov.shape == (2, 2) else pair.hedge_ratio
    else:
        beta_20d = pair.hedge_ratio

    # Momentum
    target_ret_5d = (tgt_c[-1] / tgt_c[-6] - 1) if len(tgt_c) >= 6 else 0
    hedge_ret_5d = (hdg_c[-1] / hdg_c[-6] - 1) if len(hdg_c) >= 6 else 0
    spread_ret_5d = spread_daily[-1] - spread_daily[-6] if len(spread_daily) >= 6 else 0
    spread_ret_5d /= (np.std(spread_daily[-20:]) + 1e-10)  # normalize

    # Intraday features from previous day's minute bars
    intraday_vol = 0.0
    intraday_range = 0.0
    intraday_trades_est = 0

    target_min = minute_data.get(pair.target)
    hedge_min = minute_data.get(pair.hedge)
    if target_min is not None and hedge_min is not None:
        prev_day = dt - timedelta(days=1)
        # Find actual previous trading day
        for offset in range(1, 5):
            check = dt - timedelta(days=offset)
            check_ts = pd.Timestamp(check)
            t_day = target_min[target_min.index.normalize() == check_ts]
            if len(t_day) > 30:
                prev_day = check
                break
            t_day = pd.DataFrame()

        if len(t_day) > 30:
            prev_ts = pd.Timestamp(prev_day)
            h_day = hedge_min[hedge_min.index.normalize() == prev_ts]
            common_min = t_day.index.intersection(h_day.index)
            if len(common_min) > 30:
                intra_spread = (t_day.loc[common_min, "close"].values -
                               pair.hedge_ratio * h_day.loc[common_min, "close"].values)
                intra_ret = np.diff(intra_spread)
                intraday_vol = np.std(intra_ret) / (np.mean(np.abs(intra_spread)) + 1e-10)
                intraday_range = (np.max(intra_spread) - np.min(intra_spread)) / (np.std(intra_spread) + 1e-10)
                # Count z-crossings (proxy for trade opportunity)
                intra_z = (intra_spread - np.mean(intra_spread)) / (np.std(intra_spread) + 1e-10)
                crossings = np.sum(np.abs(np.diff(np.sign(intra_z))) > 0)
                intraday_trades_est = int(crossings)

    return DailyFeatures(
        pair_id=pair.pair_id, date=as_of_date,
        spread_z_eod=float(spread_z_eod),
        spread_vol_5d=float(vol_5d), spread_vol_20d=float(vol_20d),
        vol_ratio=float(vol_ratio), spread_ma_dev=float(spread_ma_dev),
        hurst_20d=float(hurst_20d), vr_5=float(vr_5), vr_10=float(vr_10),
        corr_20d=float(np.nan_to_num(corr_20d)),
        corr_5d=float(np.nan_to_num(corr_5d)),
        beta_20d=float(np.nan_to_num(beta_20d, nan=pair.hedge_ratio)),
        target_ret_5d=float(target_ret_5d), hedge_ret_5d=float(hedge_ret_5d),
        spread_ret_5d=float(spread_ret_5d),
        intraday_vol=float(intraday_vol), intraday_range=float(intraday_range),
        intraday_trades_est=int(intraday_trades_est),
    )


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3: RANDOM FOREST — CONDITIONAL PARAMETER SELECTION
# ═════════════════════════════════════════════════════════════════════════════


# Config parameters used as RF features alongside daily lagged features
CONFIG_PARAM_NAMES = ["lookback_minutes", "entry_z", "exit_z", "stop_z"]

# Lagged daily features
DAILY_FEATURES = [
    "spread_z_eod",     # where is spread now
    "spread_vol_5d",    # recent volatility
    "vol_ratio",        # vol regime (5d/20d)
    "hurst_20d",        # mean-reversion strength
    "vr_5",             # variance ratio
    "corr_20d",         # pair relationship stability
    "spread_ret_5d",    # recent momentum
    "intraday_vol",     # previous day's intraday activity
]

# Normalization constants for config params (so RF sees similar scales)
_CONFIG_NORMS = {
    "lookback_minutes": 1950.0,  # max lookback
    "entry_z": 2.5,
    "exit_z": 0.75,
    "stop_z": 5.0,
}


def _config_to_features(config) -> list[float]:
    """Convert a ParamConfig to normalized feature values."""
    return [
        config.lookback_minutes / _CONFIG_NORMS["lookback_minutes"],
        config.entry_z / _CONFIG_NORMS["entry_z"],
        config.exit_z / _CONFIG_NORMS["exit_z"],
        config.stop_z / _CONFIG_NORMS["stop_z"],
    ]


def train_conditional_model(features_df: pd.DataFrame,
                            returns_df: pd.DataFrame,
                            pair_id: str,
                            param_grid: list | None = None,
                            n_estimators: int = 200,
                            max_depth: int = 5,
                            ) -> dict:
    """
    Train a Random Forest CLASSIFIER for one pair.

    Each training sample = (daily_features, config_params) -> profitable? (1/0)
    This gives ~225 days x 240 configs = 54,000 samples per pair.

    The RF learns the INTERACTION between market conditions and parameter
    choice: "when vol is high and z is extended, a tight entry profits 75%
    of the time, but a wide entry only 40%."

    At prediction time: sweep all 240 configs, pick the one with highest
    P(profitable). That P is the model's trading signal.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    pair_features = features_df[features_df["pair_id"] == pair_id].copy()
    pair_returns = returns_df[returns_df["pair_id"] == pair_id].copy()

    if pair_features.empty or pair_returns.empty:
        return {"model": None, "error": "no data"}

    if param_grid is None:
        param_grid = generate_param_grid()

    # Build config lookup: config_id -> normalized feature vector
    config_lookup = {c.config_id: _config_to_features(c) for c in param_grid}

    # Daily features
    feature_cols = [f for f in DAILY_FEATURES if f in features_df.columns]
    if not feature_cols:
        feature_cols = DailyFeatures.feature_names()

    pair_features = pair_features.set_index("date")
    trading_dates = sorted(pair_returns["date"].unique())
    feature_dates = set(pair_features.index)

    # Build training matrix: each row = [daily_features..., config_params...] -> profitable
    X_rows = []
    y_rows = []

    for i in range(len(trading_dates) - 1):
        feat_date = trading_dates[i]      # features from this day
        trade_date = trading_dates[i + 1]  # predict profitability for next day

        if feat_date not in feature_dates:
            continue

        daily_feats = pair_features.loc[feat_date, feature_cols]
        if hasattr(daily_feats, 'values'):
            daily_vec = daily_feats.values.astype(float)
        else:
            daily_vec = np.array([float(daily_feats)])

        if not np.all(np.isfinite(daily_vec)):
            continue

        # Get returns for ALL configs on the trade date
        day_returns = pair_returns[pair_returns["date"] == trade_date]
        for _, row in day_returns.iterrows():
            cid = row["config_id"]
            # Classify on GROSS (pre-TC) profitability — more signal, higher base rate
            # The RF's job: "is this a good mean-reversion setup?"
            # TC filtering is handled by the probability threshold + Kelly sizing
            gross_ret = row.get("gross_return", row["daily_return"])
            if not np.isfinite(gross_ret) or cid not in config_lookup:
                continue
            config_vec = config_lookup[cid]
            X_rows.append(np.concatenate([daily_vec, config_vec]))
            y_rows.append(1 if gross_ret > 0 else 0)

    X = np.array(X_rows)
    y = np.array(y_rows)

    # Clean any remaining NaN/Inf
    valid = np.all(np.isfinite(X), axis=1)
    X, y = X[valid], y[valid]

    if len(X) < 100:
        return {"model": None, "error": f"only {len(X)} valid samples (need 100+)"}

    n_pos = int(y.sum())
    base_rate = n_pos / len(y)

    # Feature names for importance tracking
    all_feature_names = feature_cols + [f"cfg_{n}" for n in CONFIG_PARAM_NAMES]

    # Train RF Classifier
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=20, min_samples_split=40,
        class_weight="balanced",
        random_state=42, n_jobs=1,
    )
    rf.fit(X, y)

    # In-sample metrics
    proba_train = rf.predict_proba(X)[:, 1]
    try:
        auc = roc_auc_score(y, proba_train)
    except ValueError:
        auc = 0.5

    # Mean return by outcome (for Kelly expected return sizing)
    # Use the actual returns to compute E[return | profitable] etc.
    all_rets = pair_returns.set_index(["date", "config_id"])["daily_return"]
    ret_vals = np.array([r["daily_return"] for _, r in pair_returns.iterrows()
                        if np.isfinite(r["daily_return"])])
    if len(ret_vals) > 0:
        mean_win = float(np.mean(ret_vals[ret_vals > 0])) if (ret_vals > 0).any() else 0.0
        mean_loss = float(np.mean(ret_vals[ret_vals <= 0])) if (ret_vals <= 0).any() else 0.0
    else:
        mean_win, mean_loss = 0.01, -0.01

    importance = dict(zip(all_feature_names, rf.feature_importances_))

    return {
        "model": rf,
        "config_models": {},
        "feature_cols": feature_cols,
        "feature_importance": importance,
        "train_score": auc,
        "base_rate": base_rate,
        "mean_win": mean_win,
        "mean_loss": mean_loss,
        "n_samples": len(X),
        "n_days": len(trading_dates),
    }


def predict_model(trained_model: dict, features: np.ndarray,
                   param_grid: list[ParamConfig]
                   ) -> tuple[ParamConfig, float, float]:
    """
    Predict P(profitable) for all configs, pick the best.

    Sweeps all 240 configs through the RF with today's features.
    Returns the config with highest P(profitable) and that probability.
    """
    model = trained_model.get("model")
    if model is None:
        return param_grid[0], 0.0, 0.0

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Map to core features if needed
    feature_cols = trained_model.get("feature_cols")
    if feature_cols and len(features) > len(feature_cols):
        all_names = DailyFeatures.feature_names()
        indices = [all_names.index(f) for f in feature_cols if f in all_names]
        features = features[indices]

    # Build feature matrix: one row per config
    X_pred = []
    for config in param_grid:
        config_vec = _config_to_features(config)
        X_pred.append(np.concatenate([features, config_vec]))

    X_pred = np.array(X_pred)
    X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict P(profitable) for all configs at once
    try:
        probas = model.predict_proba(X_pred)[:, 1]
    except Exception:
        return param_grid[0], 0.5, 0.0

    # Pick config with highest P(profitable)
    best_idx = int(np.argmax(probas))
    best_config = param_grid[best_idx]
    p_profitable = float(probas[best_idx])

    # Expected return from probability × historical win/loss sizes
    mean_win = trained_model.get("mean_win", 0.01)
    mean_loss = trained_model.get("mean_loss", -0.01)
    expected_return = p_profitable * mean_win + (1 - p_profitable) * mean_loss

    return best_config, p_profitable, float(expected_return)


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4: KELLY VECTOR PORTFOLIO OPTIMIZATION
# ═════════════════════════════════════════════════════════════════════════════

def _ledoit_wolf_shrink(cov: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward scaled identity."""
    n = cov.shape[0]
    if n <= 1:
        return cov + 1e-6 * np.eye(n)
    trace = np.trace(cov)
    mu_target = trace / n
    target = mu_target * np.eye(n)
    delta = cov - target
    delta_sq_sum = np.sum(delta ** 2)
    if delta_sq_sum < 1e-12:
        return cov + 1e-6 * np.eye(n)
    alpha = np.clip(n / (n + delta_sq_sum / (trace ** 2 / n)), 0.1, 0.9)
    return alpha * target + (1 - alpha) * cov + 1e-6 * np.eye(n)


def kelly_vector(expected_returns: np.ndarray,
                 cov_matrix: np.ndarray,
                 max_leverage: float = 2.0,
                 kelly_fraction: float = 0.5) -> np.ndarray:
    """Half-Kelly vector with Ledoit-Wolf shrinkage."""
    n = len(expected_returns)
    if n == 0:
        return np.array([])
    mu = np.maximum(expected_returns, 0)
    if mu.sum() < 1e-10:
        return np.zeros(n)
    cov = _ledoit_wolf_shrink(cov_matrix)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    weights = kelly_fraction * (cov_inv @ mu)
    weights = np.maximum(weights, 0)
    total = weights.sum()
    if total > max_leverage:
        weights *= (max_leverage / total)
    return weights


def compute_kelly_allocation(pair_predictions: list[dict],
                             returns_history: pd.DataFrame,
                             max_leverage: float = 2.0,
                             prob_threshold: float = 0.65,
                             corr_threshold: float = 0.85,
                             max_weight_per_model: float = 0.15,
                             lookback_days: int = 60,
                             min_model_days: int = 5,
                             ) -> dict[str, float]:
    """
    Three-stage Kelly allocation:

    Stage 1 — GATE: only models with P(profitable) > prob_threshold are
              candidates. This is the conditional part of CPO.
    Stage 2 — CORRELATION DEDUP: among candidates, if two models have
              pairwise return correlation > corr_threshold, keep the one
              with higher P(profitable). This prevents over-concentrating
              in correlated trades.
    Stage 3 — ALLOCATE: half-Kelly on survivors using expected returns
              and Ledoit-Wolf covariance. Leverage ramps linearly from
              10% to 100% of max_leverage over the first 20 days.

    min_model_days: a model must have traded at least this many days
        in returns_history before it can receive non-trivial Kelly weight.
    """
    # Stage 1: Probability gate
    candidates = [p for p in pair_predictions if p.get("p_profitable", 0) > prob_threshold]

    if not candidates:
        return {}

    # Filter by minimum model history (exempt during bootstrap)
    if not returns_history.empty and "pair_id" in returns_history.columns:
        n_history_days = returns_history["date"].nunique()
        if n_history_days >= min_model_days:  # only enforce after bootstrap
            model_day_counts = returns_history.groupby("pair_id")["date"].nunique()
            candidates = [p for p in candidates
                          if model_day_counts.get(p["pair_id"], 0) >= min_model_days]

    if not candidates:
        return {}

    pair_ids = [p["pair_id"] for p in candidates]
    probs = np.array([p["p_profitable"] for p in candidates])
    mu = np.array([p["expected_return"] for p in candidates])

    # Stage 2: Correlation dedup (requires return history)
    if (not returns_history.empty and "date" in returns_history.columns
            and returns_history["date"].nunique() >= 10):

        recent = returns_history[returns_history["date"].isin(
            returns_history["date"].unique()[-lookback_days:]
        )]
        pivot = recent.pivot_table(index="date", columns="pair_id",
                                   values="daily_return", aggfunc="first")

        available_ids = [p for p in pair_ids if p in pivot.columns]
        if len(available_ids) >= 2:
            corr_matrix = pivot[available_ids].corr().values

            # Greedy dedup: iterate by descending P, drop correlated duplicates
            sorted_idx = np.argsort(-probs[:len(available_ids)])
            keep_mask = np.ones(len(available_ids), dtype=bool)

            for i in range(len(sorted_idx)):
                idx_i = sorted_idx[i]
                if not keep_mask[idx_i]:
                    continue
                for j in range(i + 1, len(sorted_idx)):
                    idx_j = sorted_idx[j]
                    if not keep_mask[idx_j]:
                        continue
                    if idx_i < corr_matrix.shape[0] and idx_j < corr_matrix.shape[1]:
                        if abs(corr_matrix[idx_i, idx_j]) > corr_threshold:
                            keep_mask[idx_j] = False  # drop the lower-P model

            # Rebuild candidate list
            survived_ids = [available_ids[i] for i in range(len(available_ids)) if keep_mask[i]]
            n_dropped = len(available_ids) - len(survived_ids)

            # Remap to survived
            pair_ids = survived_ids
            candidates = [p for p in candidates if p["pair_id"] in survived_ids]
            probs = np.array([p["p_profitable"] for p in candidates])
            mu = np.array([p["expected_return"] for p in candidates])

    if not candidates:
        return {}

    n = len(candidates)

    # Stage 3: Kelly allocation
    # Leverage ramp: linearly increase from 10% to 100% of max over first 20 days
    if (returns_history.empty or "date" not in returns_history.columns):
        n_history_days = 0
    else:
        n_history_days = returns_history["date"].nunique()

    ramp_days = 20
    ramp_factor = np.clip(n_history_days / ramp_days, 0.10, 1.0)
    effective_max_leverage = max_leverage * ramp_factor

    # Cold start: equal weight scaled by half-Kelly
    if n_history_days < 10:
        mu_pos = np.maximum(mu, 0)
        total = min(mu_pos.sum() * 0.5, effective_max_leverage)
        if total < 1e-10:
            return {}
        equal_w = min(total / n, max_weight_per_model)
        return {pid: float(equal_w) for pid in pair_ids}

    # Enough history: use covariance-based Kelly
    recent = returns_history[returns_history["date"].isin(
        returns_history["date"].unique()[-lookback_days:]
    )]
    pivot = recent.pivot_table(index="date", columns="pair_id",
                               values="daily_return", aggfunc="first")

    available = [p for p in pair_ids if p in pivot.columns]
    if len(available) < 1:
        return {}

    mu_avail = np.array([mu[pair_ids.index(p)] for p in available])
    cov_matrix = pivot[available].cov().values

    if cov_matrix.shape[0] != len(available):
        return {}

    weights = kelly_vector(mu_avail, cov_matrix, effective_max_leverage)

    # Cap per-model weight to prevent concentration
    weights = np.minimum(weights, max_weight_per_model)

    allocation = {}
    for pid, w in zip(available, weights):
        if w > 0.0001:
            allocation[pid] = float(w)

    return allocation


# ═════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════════

def load_pairs_from_burgess(json_path: str | Path, top_n: int = 50) -> list[PairSpec]:
    """
    Load top pairs from Burgess Phase 1 output (n_vars=2).

    Applies two filters:
        1. Deduplication: A_B and B_A are the same trade. Keep the one with
           better (more negative) ADF t-stat.
        2. Structural arbitrage: Dual share classes (GOOGL/GOOG, FOXA/FOX)
           mean-revert by construction. They inflate results without testing
           actual strategy edge. Excluded.
    """
    # Known dual share class pairs and structural near-equivalents
    STRUCTURAL_PAIRS = {
        frozenset({"GOOGL", "GOOG"}),
        frozenset({"FOXA", "FOX"}),
        frozenset({"NWSA", "NWS"}),
        frozenset({"BRK-A", "BRK-B"}),
        frozenset({"LBRDA", "LBRDK"}),
    }

    with open(json_path) as f:
        data = json.load(f)

    raw_pairs = []
    for b in data["ranked_baskets"][:top_n * 3]:  # scan extra to fill after filtering
        if len(b["basket"]) != 1:
            logger.warning(f"Expected n_vars=2 (1 hedge), got {len(b['basket'])} for {b['target']}")
            continue
        raw_pairs.append(PairSpec(
            pair_id=f"{b['target']}_{b['basket'][0]}",
            target=b["target"],
            hedge=b["basket"][0],
            hedge_ratio=b["hedge_ratio"][0],
            adf_t=b["adf_t"],
            adf_p=b["adf_p"],
            hurst=b["hurst"],
            half_life=b["half_life"],
            composite_score=b["composite_score"],
        ))

    # Filter structural arbitrage
    n_structural = 0
    filtered = []
    for p in raw_pairs:
        pair_set = frozenset({p.target, p.hedge})
        if pair_set in STRUCTURAL_PAIRS:
            n_structural += 1
            continue
        filtered.append(p)

    # Deduplicate: canonical key = sorted(target, hedge), keep best ADF
    seen = {}  # canonical_key -> PairSpec
    n_dedup = 0
    for p in filtered:
        key = tuple(sorted([p.target, p.hedge]))
        if key in seen:
            # Keep the one with more negative ADF t-stat (stronger stationarity)
            if p.adf_t < seen[key].adf_t:
                seen[key] = p
            n_dedup += 1
        else:
            seen[key] = p

    pairs = list(seen.values())[:top_n]

    if n_structural > 0 or n_dedup > 0:
        print(f"  Pair filtering: {n_structural} structural arb removed, "
              f"{n_dedup} duplicates removed, {len(pairs)} unique pairs")

    return pairs


def aggregate_daily_returns(minute_data: dict[str, pd.DataFrame],
                            pairs: list[PairSpec]) -> dict[str, pd.Series]:
    """Build daily close prices from minute bars for each ticker."""
    daily = {}
    for ticker, bars in minute_data.items():
        # Get last close per day
        d = bars["close"].resample("D").last().dropna()
        d.index = d.index.date
        daily[ticker] = d
    return daily


def run_phase2_training(pairs: list[PairSpec],
                        minute_data: dict[str, pd.DataFrame],
                        param_grid: list[ParamConfig],
                        output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phase 2: Run all configs on all pairs for training year.
    Returns (returns_df, features_df).
    """
    print(f"\n{'='*70}")
    print("PHASE 2: Parameter Grid Search — Minute Data")
    print(f"{'='*70}")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Configs: {len(param_grid)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    returns_path = output_dir / "phase2_returns.parquet"

    # Skip grid search if returns already cached
    if returns_path.exists():
        print(f"\n  Returns already cached: {returns_path}")
        returns_df = pd.read_parquet(returns_path)
        print(f"  Loaded {len(returns_df)} rows")
    else:
        # Run param grid for each pair
        all_returns = []
        for i, pair in enumerate(pairs):
            print(f"\n  [{i+1}/{len(pairs)}] {pair.pair_id} (HR={pair.hedge_ratio:.4f})")
            pair_returns = run_pair_year(pair, minute_data, param_grid)
            if not pair_returns.empty:
                all_returns.append(pair_returns)
                print(f"    {len(pair_returns)} strategy-days")

        returns_df = pd.concat(all_returns, ignore_index=True)
        returns_df.to_parquet(returns_path)
        print(f"\n  Saved: {returns_path} ({len(returns_df)} rows)")

    # Compute daily features (also cached)
    features_path = output_dir / "phase2_features.parquet"
    if features_path.exists():
        print(f"\n  Features already cached: {features_path}")
        features_df = pd.read_parquet(features_path)
        print(f"  Loaded {len(features_df)} rows")
    else:
        print(f"\n  Computing daily features...")
        daily_prices = aggregate_daily_returns(minute_data, pairs)
        trading_days = sorted(returns_df["date"].unique())

        all_features = []
        for pair in pairs:
            tgt_daily = daily_prices.get(pair.target)
            hdg_daily = daily_prices.get(pair.hedge)
            if tgt_daily is None or hdg_daily is None:
                continue
            for day in trading_days:
                feat = compute_daily_features(pair, tgt_daily, hdg_daily, minute_data, day)
                if feat is not None:
                    row = {"pair_id": feat.pair_id, "date": feat.date}
                    for fn in DailyFeatures.feature_names():
                        row[fn] = getattr(feat, fn)
                    all_features.append(row)

        features_df = pd.DataFrame(all_features)
        features_df.to_parquet(features_path)
        print(f"  Saved: {features_path}")

    return returns_df, features_df


def run_phase3_training(pairs: list[PairSpec],
                        returns_df: pd.DataFrame,
                        features_df: pd.DataFrame,
                        output_dir: Path,
                        param_grid: list[ParamConfig] | None = None,
                        ) -> dict[str, dict]:
    """
    Phase 3: Train Random Forest for each pair.
    Returns dict: pair_id → trained model dict.
    """
    if param_grid is None:
        param_grid = generate_param_grid()

    print(f"\n{'='*70}")
    print("PHASE 3: Random Forest Training")
    print(f"  Samples per pair: ~{len(param_grid)} configs × ~{features_df['date'].nunique()} days")
    print(f"{'='*70}")

    models = {}
    for i, pair in enumerate(pairs):
        result = train_conditional_model(features_df, returns_df, pair.pair_id, param_grid)
        models[pair.pair_id] = result
        status = (f"AUC={result['train_score']:.4f}, base_rate={result['base_rate']:.1%}, n={result['n_samples']}"
                  if result["model"] else result.get("error", "failed"))
        print(f"  [{i+1}/{len(pairs)}] {pair.pair_id}: {status}")

    # Save feature importances
    importance_summary = {}
    for pid, m in models.items():
        if m.get("feature_importance"):
            importance_summary[pid] = m["feature_importance"]

    with open(output_dir / "phase3_importances.json", "w") as f:
        json.dump(importance_summary, f, indent=2)

    return models


def run_phase4_oos(pairs: list[PairSpec],
                   models: dict[str, dict],
                   minute_data_oos: dict[str, pd.DataFrame],
                   param_grid: list[ParamConfig],
                   output_dir: Path,
                   max_leverage: float = 2.0,
                   warmup_daily: dict[str, pd.Series] | None = None,
                   ) -> pd.DataFrame:
    """
    Phase 4: OOS portfolio trading with Kelly allocation.

    warmup_daily: optional dict of {ticker: daily_close_series} from training
        period to seed feature computation so trading can start day 1.

    Returns daily portfolio P&L DataFrame.
    """
    print(f"\n{'='*70}")
    print("PHASE 4: OOS Portfolio Trading (Kelly Vector)")
    print(f"{'='*70}")

    daily_prices = aggregate_daily_returns(minute_data_oos, pairs)

    # Seed with warmup daily prices (prepend training period)
    if warmup_daily:
        for ticker in daily_prices:
            if ticker in warmup_daily:
                warmup = warmup_daily[ticker]
                oos = daily_prices[ticker]
                # Remove any overlap
                warmup_only = warmup[warmup.index < oos.index.min()]
                daily_prices[ticker] = pd.concat([warmup_only, oos])
        n_warmup = len(warmup_daily.get(pairs[0].target, []))
        print(f"  Warmup: {n_warmup} daily prices prepended per ticker")

    # Build trading days from minute data — filter to actual market days
    # (must be weekday and have sufficient bars for most tickers)
    all_days = sorted(set().union(*(
        set(minute_data_oos[t].index.normalize().unique())
        for t in minute_data_oos
    )))

    trading_days = []
    for day in all_days:
        # Must be weekday (Mon=0 .. Fri=4)
        if day.weekday() >= 5:
            continue
        # Check at least one ticker has 100+ bars (full trading day)
        has_bars = False
        for t in list(minute_data_oos.keys())[:5]:  # spot check first 5
            day_start = day.replace(hour=9, minute=30)
            day_end = day.replace(hour=16, minute=0)
            bars = minute_data_oos[t]
            n_bars = len(bars[(bars.index >= day_start) & (bars.index <= day_end)])
            if n_bars >= 100:
                has_bars = True
                break
        if has_bars:
            trading_days.append(day)

    print(f"  Trading days: {len(trading_days)} (filtered from {len(all_days)} raw dates)")

    # Accumulate returns for Kelly covariance estimation
    returns_history = []
    portfolio_pnl = []

    # Per-model tracking for performance breakdown
    model_returns = {p.pair_id: [] for p in pairs}
    model_predictions = {p.pair_id: [] for p in pairs}  # (predicted, actual) for OOS R²

    for day_idx, day in enumerate(trading_days):
        day_str = day.strftime("%Y-%m-%d")

        # Step 1: Compute features for each pair (from previous days)
        predictions = []
        n_no_model = 0
        n_no_daily = 0
        n_no_feat = 0
        for pair in pairs:
            model = models.get(pair.pair_id, {})
            if model.get("model") is None:
                n_no_model += 1
                continue

            tgt_daily = daily_prices.get(pair.target)
            hdg_daily = daily_prices.get(pair.hedge)
            if tgt_daily is None or hdg_daily is None:
                n_no_daily += 1
                continue

            feat = compute_daily_features(pair, tgt_daily, hdg_daily,
                                         minute_data_oos, day_str)
            if feat is None:
                n_no_feat += 1
                continue

            config, p_profitable, expected_ret = predict_model(
                model, feat.to_array(), param_grid
            )
            predictions.append({
                "pair_id": pair.pair_id,
                "pair": pair,
                "config": config,
                "p_profitable": p_profitable,
                "expected_return": expected_ret,
                "features": feat,
            })

        # Diagnostics — always for first 5 days, then every 10
        if day_idx < 5 or day_idx % 10 == 0:
            print(f"  Day {day_idx+1} ({day_str}): "
                  f"no_model={n_no_model} no_daily={n_no_daily} "
                  f"no_feat={n_no_feat} predicted={len(predictions)}")
            if predictions:
                probs = [p["p_profitable"] for p in predictions]
                n_above = sum(1 for p in probs if p > 0.55)
                print(f"    P(profit) range: [{min(probs):.3f}, {max(probs):.3f}], "
                      f"{n_above}/{len(probs)} above 0.55")
            if day_idx == 0:
                sp = pairs[0]
                td = daily_prices.get(sp.target)
                print(f"    Sample: {sp.pair_id}, target daily len="
                      f"{len(td) if td is not None else 'None'}, "
                      f"idx type={type(td.index[0]).__name__ if td is not None and len(td) > 0 else 'N/A'}")
                if td is not None and len(td) > 0:
                    print(f"    Range: {td.index[0]} -> {td.index[-1]}")
                print(f"    day_str={day_str}, as_date={pd.Timestamp(day_str).date()}")

        if not predictions:
            continue

        # Step 2: Kelly allocation
        returns_hist_df = pd.DataFrame(returns_history) if returns_history else pd.DataFrame()
        allocation = compute_kelly_allocation(
            predictions, returns_hist_df,
            max_leverage=max_leverage, lookback_days=60
        )

        # Diagnostics — allocation
        if day_idx < 5 or (day_idx < 35 and day_idx % 5 == 0):
            n_alloc = len(allocation)
            alloc_sum = sum(allocation.values()) if allocation else 0
            print(f"    Allocation: {n_alloc} models, total_weight={alloc_sum:.4f}")
            if not allocation:
                probs = [p["p_profitable"] for p in predictions]
                print(f"    WARNING: empty allocation! P(profit) range: "
                      f"[{min(probs):.3f}, {max(probs):.3f}]")

        # Step 3: Execute trades for allocated models
        day_pnl = 0.0
        day_details = []
        n_weight_skip = 0
        n_no_bars = 0
        n_short_spread = 0
        n_executed = 0
        for pred in predictions:
            pid = pred["pair_id"]
            weight = allocation.get(pid, 0)
            if weight < 0.0001:  # essentially zero
                n_weight_skip += 1
                continue

            # Run intraday
            pair = pred["pair"]
            config = pred["config"]
            target_bars = minute_data_oos.get(pair.target)
            hedge_bars = minute_data_oos.get(pair.hedge)
            if target_bars is None or hedge_bars is None:
                n_no_bars += 1
                continue

            spread_df = construct_minute_spread(target_bars, hedge_bars, pair.hedge_ratio)
            spread = spread_df["spread"]

            day_start = day.replace(hour=9, minute=30)
            day_end = day.replace(hour=16, minute=0)
            spread_day = spread[(spread.index >= day_start) & (spread.index <= day_end)]

            if len(spread_day) < 30:
                n_short_spread += 1
                # Print once on first failure
                if n_short_spread == 1 and day_idx < 35:
                    # Show what's actually in the spread index for this day
                    day_bars = spread[spread.index.normalize() == day]
                    print(f"    SPREAD DEBUG {pid}: day={day}, "
                          f"day_start={day_start}, day_end={day_end}")
                    print(f"      spread_day len={len(spread_day)}, "
                          f"bars on this date={len(day_bars)}")
                    if len(day_bars) > 0:
                        print(f"      actual bar range: {day_bars.index[0]} -> {day_bars.index[-1]}")
                    if len(spread) > 0:
                        print(f"      full spread range: {spread.index[0]} -> {spread.index[-1]}")
                continue

            # Notional capital at day open
            tgt_open = spread_df["target_close"][(spread_df.index >= day_start) & (spread_df.index <= day_end)]
            hdg_open = spread_df["hedge_close"][(spread_df.index >= day_start) & (spread_df.index <= day_end)]
            if len(tgt_open) > 0 and len(hdg_open) > 0:
                notional = float(tgt_open.iloc[0]) + abs(pair.hedge_ratio) * float(hdg_open.iloc[0])
            else:
                notional = 1.0

            # Lookback for z-score warmup
            hist_start = day - timedelta(days=20)
            spread_hist = spread[(spread.index >= hist_start) & (spread.index < day_start)]

            result = run_intraday_single_day(spread_day, config, spread_hist, notional)
            weighted_ret = result["daily_return"] * weight
            day_pnl += weighted_ret

            returns_history.append({
                "date": day_str, "pair_id": pid,
                "daily_return": result["daily_return"],
            })

            # Track per-model returns and OOS prediction accuracy
            model_returns[pid].append(result["daily_return"])
            # Track (P(profitable), actually_profitable) for calibration
            actually_profitable = 1.0 if result["daily_return"] > 0 else 0.0
            model_predictions[pid].append((pred["p_profitable"], actually_profitable))

            day_details.append({
                "pair_id": pid, "config_id": config.config_id,
                "weight": weight, "return": result["daily_return"],
                "n_trades": result["n_trades"],
            })
            n_executed += 1

        # Execution summary for early days
        if day_idx < 5 or (day_idx < 35 and day_idx % 5 == 0):
            print(f"    Execution: weight_skip={n_weight_skip} no_bars={n_no_bars} "
                  f"short_spread={n_short_spread} executed={n_executed}")

        portfolio_pnl.append({
            "date": day_str,
            "portfolio_return": day_pnl,
            "n_models_active": len([d for d in day_details if d["weight"] > 0]),
            "total_weight": sum(d["weight"] for d in day_details),
            "details": day_details,
        })

        if (day_idx + 1) % 20 == 0:
            cum_ret = sum(p["portfolio_return"] for p in portfolio_pnl)
            n_active = portfolio_pnl[-1]["n_models_active"]
            print(f"  Day {day_idx+1}/{len(trading_days)}: "
                  f"cum={cum_ret:+.4f}, active={n_active}")

    # Summary
    pnl_df = pd.DataFrame(portfolio_pnl)
    if not pnl_df.empty:
        rets = pnl_df["portfolio_return"].values
        n_days = len(rets)
        daily_mean = np.mean(rets)
        daily_std = np.std(rets) + 1e-10
        sr = daily_mean / daily_std * np.sqrt(252)
        ann_ret = daily_mean * 252
        ann_vol = daily_std * np.sqrt(252)
        cum = np.cumsum(rets)
        max_dd = np.min(cum - np.maximum.accumulate(cum))
        win_days = (rets > 0).sum()

        print(f"\n  OOS Results:")
        print(f"    Trading days:     {n_days}")
        print(f"    Cumulative ret:   {cum[-1]:+.4f}  ({cum[-1]*100:+.2f}%)")
        print(f"    Ann. return:      {ann_ret:+.4f}  ({ann_ret*100:+.1f}%)")
        print(f"    Ann. volatility:  {ann_vol:.4f}   ({ann_vol*100:.1f}%)")
        print(f"    Sharpe ratio:     {sr:+.4f}")
        print(f"    Max drawdown:     {max_dd:+.4f}  ({max_dd*100:+.2f}%)")
        print(f"    Win days:         {win_days}/{n_days} ({win_days/n_days:.1%})")
        print(f"    Avg models/day:   {pnl_df['n_models_active'].mean():.1f}")

        # ── OOS R² — predicted vs actual returns ──────────────────────
        print(f"\n  OOS Prediction Calibration (RF P(profitable) vs actual):")
        all_pred = []
        all_actual = []
        for pid in model_predictions:
            for (pred_p, actual_win) in model_predictions[pid]:
                if np.isfinite(pred_p) and np.isfinite(actual_win):
                    all_pred.append(pred_p)
                    all_actual.append(actual_win)

        if len(all_pred) > 10:
            all_pred = np.array(all_pred)
            all_actual = np.array(all_actual)

            # Overall accuracy
            pred_class = (all_pred > 0.5).astype(int)
            accuracy = np.mean(pred_class == all_actual)
            actual_rate = np.mean(all_actual)

            # Calibration by probability bin
            print(f"    Samples:          {len(all_pred)}")
            print(f"    Base win rate:    {actual_rate:.1%}")
            print(f"    Classification accuracy: {accuracy:.1%}")
            print(f"    Brier score:      {np.mean((all_pred - all_actual)**2):.4f}")
            print(f"\n    Calibration (predicted P bin → actual win rate):")
            print(f"    {'P(profit) bin':>15s}  {'n':>5s}  {'Actual WR':>9s}  {'Lift':>6s}")
            for lo, hi in [(0.0, 0.4), (0.4, 0.5), (0.5, 0.55), (0.55, 0.6),
                           (0.6, 0.65), (0.65, 0.7), (0.7, 0.8), (0.8, 1.01)]:
                mask = (all_pred >= lo) & (all_pred < hi)
                if mask.sum() >= 3:
                    bin_wr = np.mean(all_actual[mask])
                    lift = bin_wr - actual_rate
                    print(f"    [{lo:.2f}, {hi:.2f})  {mask.sum():5d}  "
                          f"{bin_wr:.1%}  {lift:+.1%}")

        # ── Per-model performance breakdown ───────────────────────────
        print(f"\n  Per-Model Performance (top 20 by Sharpe):")
        print(f"    {'Pair':>20s}  {'Days':>5s}  {'MeanRet':>9s}  {'Sharpe':>7s}  "
              f"{'CumRet':>8s}  {'WinRate':>7s}  {'AvgP':>6s}  {'Calib':>6s}")

        model_stats = []
        for pair in pairs:
            pid = pair.pair_id
            rets_m = np.array(model_returns.get(pid, []))
            preds_m = model_predictions.get(pid, [])
            if len(rets_m) < 3:
                continue
            m_mean = np.mean(rets_m)
            m_std = np.std(rets_m) + 1e-10
            m_sr = m_mean / m_std * np.sqrt(252)
            m_cum = np.sum(rets_m)
            m_wr = np.mean(rets_m > 0)

            # Per-model calibration: avg predicted P vs actual win rate
            if len(preds_m) > 3:
                avg_p = np.mean([p[0] for p in preds_m])
                actual_wr = np.mean([p[1] for p in preds_m])
                calib_err = abs(avg_p - actual_wr)  # lower = better calibrated
            else:
                avg_p = 0.5
                calib_err = 1.0

            model_stats.append({
                "pair_id": pid, "n_days": len(rets_m),
                "mean_ret": m_mean, "sharpe": m_sr,
                "cum_ret": m_cum, "win_rate": m_wr,
                "avg_p": avg_p, "calib_err": calib_err,
            })

        model_stats.sort(key=lambda x: x["sharpe"], reverse=True)
        for ms in model_stats[:20]:
            print(f"    {ms['pair_id']:>20s}  {ms['n_days']:5d}  "
                  f"{ms['mean_ret']:+.6f}  {ms['sharpe']:+.3f}  "
                  f"{ms['cum_ret']:+.5f}  {ms['win_rate']:.1%}  "
                  f"{ms['avg_p']:.3f}  {ms['calib_err']:.3f}")

        # Bottom 5
        if len(model_stats) > 5:
            print(f"\n  Bottom 5 by Sharpe:")
            for ms in model_stats[-5:]:
                print(f"    {ms['pair_id']:>20s}  {ms['n_days']:5d}  "
                      f"{ms['mean_ret']:+.6f}  {ms['sharpe']:+.3f}  "
                      f"{ms['cum_ret']:+.5f}  {ms['win_rate']:.1%}  "
                      f"{ms['avg_p']:.3f}  {ms['calib_err']:.3f}")

        # Summary stats across models
        if model_stats:
            sharpes = [ms["sharpe"] for ms in model_stats]
            print(f"\n  Model Sharpe Distribution (n={len(model_stats)}):")
            print(f"    Mean: {np.mean(sharpes):+.3f}  Median: {np.median(sharpes):+.3f}")
            print(f"    Positive: {sum(1 for s in sharpes if s > 0)}/{len(sharpes)}")

        # Save model stats
        with open(output_dir / "phase4_model_stats.json", "w") as f:
            json.dump(model_stats, f, indent=2)

    pnl_df.to_parquet(output_dir / "phase4_portfolio.parquet")
    return pnl_df
