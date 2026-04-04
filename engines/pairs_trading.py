"""
Pairs Trading Engine — Core data structures, simulation, and portfolio allocation.

Contains everything needed to:
    - Define pairs (PairSpec) and trading configs (ParamConfig)
    - Fetch and cache minute-bar data from Polygon.io
    - Construct minute-bar spreads from target + hedge
    - Simulate intraday z-score mean-reversion (run_intraday_single_day)
    - Run full-year grid search across all configs (run_pair_year)
    - Allocate capital via Kelly criterion (compute_kelly_allocation)
    - Load pairs from Burgess Phase 1 output

Does NOT contain feature computation or ML training — those are in
engines/cpo_training.py and engines/minute_features.py.
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

# Config parameters used as RF features alongside daily lagged features
CONFIG_PARAM_NAMES = ["lookback_minutes", "entry_z", "exit_z", "stop_z"]

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



# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2: RETURNS GRID SEARCH
# ═════════════════════════════════════════════════════════════════════════════

def run_phase2_returns(pairs: list[PairSpec],
                       minute_data: dict[str, pd.DataFrame],
                       param_grid: list[ParamConfig],
                       output_dir: Path) -> pd.DataFrame:
    """
    Phase 2: Run all configs on all pairs for the training year.

    Simulates the intraday z-score strategy for each (pair, config, day)
    and records gross and net daily P&L. These returns are the LABELS
    for RF training in Phase 3 — they answer "was this config profitable
    on this day?"

    Returns a DataFrame of strategy returns. Caches to parquet.
    """
    print(f"\n{'='*70}")
    print("PHASE 2: Strategy Grid Search — Minute Data")
    print(f"{'='*70}")
    print(f"  Pairs: {len(pairs)}, Configs: {len(param_grid)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    returns_path = output_dir / "phase2_returns.parquet"

    if returns_path.exists():
        print(f"  Returns cached: {returns_path}")
        returns_df = pd.read_parquet(returns_path)
        print(f"  Loaded {len(returns_df)} rows")
        return returns_df

    all_returns = []
    for i, pair in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] {pair.pair_id} (HR={pair.hedge_ratio:.4f})")
        pair_returns = run_pair_year(pair, minute_data, param_grid)
        if not pair_returns.empty:
            all_returns.append(pair_returns)
            print(f"    {len(pair_returns)} strategy-days")

    returns_df = pd.concat(all_returns, ignore_index=True)
    returns_df.to_parquet(returns_path)
    print(f"\n  Saved: {returns_path} ({len(returns_df)} rows)")

    return returns_df
