"""
engines/grid_bot_strategy.py
==============================
Grid bot trading strategy as a CPO TradingStrategy.

MECHANISM:
    Define a price range [lower, upper] centered on the current price.
    Place buy limit orders at each grid line going down, sell limit orders
    at each grid line going up. Every completed buy→sell cycle at adjacent
    grid lines earns: grid_spacing_pct - 2*TC.

    Profit condition: price oscillates within the range (choppy/sideways).
    Loss condition: price trends strongly out of the range (breakout).

    This is structurally the inverse of momentum — it profits from the
    ABSENCE of trend, making it a natural diversifier.

CPO FIT:
    RF question: "given today's market conditions, will a grid bot on
    this asset at this spacing be profitable over the next N days?"

    Key features:
        - Short-term realized vol (oscillation intensity)
        - Momentum (trending = bad for grids)
        - Mean-reversion autocorrelation (negative autocorr = ideal)
        - Range stability (price distance from N-day mean)
        - Volume profile (thin volume = less oscillation)

    Base rate: ~40-55% (grids work in sideways markets, which occur
    roughly half the time in crypto)

CONFIG DIMENSIONS:
    grid_spacing_pct: 0.3%, 0.5%, 1.0%, 2.0%  (profit per cycle)
    range_width_pct:  5%, 10%, 20%             (breakout threshold)
    hold_days:        3, 7, 14                  (evaluation window)

TC MATH (favorable):
    Profit per cycle = grid_spacing - 2*TC
    At 0.5% spacing, 4bps TC: 0.5% - 0.08% = 0.42% per cycle
    In a choppy day: 3-8 cycles → 1.3-3.4% daily gross

SIMULATION:
    Approximate the order book by tracking grid line crossings
    in hourly bars. Each crossing generates a buy or sell.
    Matched pairs (buy→sell at adjacent lines) = completed cycles.
    Unmatched inventory at hold end = closed at market (loss if trended).
    Range breakout = all inventory closed at breakout price.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engines.triple_barrier import BarrierConfig

logger = logging.getLogger(__name__)

# ── Universe ──────────────────────────────────────────────────────────────────

DEFAULT_ASSETS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE"
]

GRID_FEATURES = [
    "ret_1h",            # latest 1h return (momentum signal)
    "ret_4h",            # 4h return
    "ret_24h",           # 24h return (trend direction)
    "vol_1h_ann",        # 1h realized vol annualized (oscillation intensity)
    "vol_4h_ann",        # 4h realized vol
    "vol_24h_ann",       # 24h realized vol
    "autocorr_1h",       # 1h return autocorrelation (negative = mean-reverting)
    "autocorr_4h",       # 4h return autocorrelation
    "range_position",    # (price - 30d_low) / (30d_high - 30d_low)
    "price_vs_ma24",     # (price - 24h_MA) / price  (deviation from mean)
    "price_vs_ma7d",     # (price - 7d_MA) / price
    "vol_regime",        # today_vol / 30d_avg_vol
    "volume_surge",      # today volume / 7d avg volume
    "cycle_rate_est",    # estimated grid cycles per day at this spacing
]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class GridModelSpec:
    model_id: str
    asset: str


@dataclass
class GridConfig:
    config_id: str
    grid_spacing_pct: float    # distance between grid lines as fraction (e.g. 0.005 = 0.5%)
    range_width_pct:  float    # total range width as fraction (e.g. 0.10 = 10%)
    hold_days:        int      # evaluation window in days

    @property
    def n_grids(self) -> int:
        """Number of grid lines within the range."""
        return max(2, int(self.range_width_pct / self.grid_spacing_pct))

    @property
    def profit_per_cycle_gross(self) -> float:
        """Gross profit per completed buy→sell cycle (before TC)."""
        return self.grid_spacing_pct

    def profit_per_cycle_net(self, tc_pct: float) -> float:
        """Net profit per cycle after TC."""
        return self.grid_spacing_pct - 2 * tc_pct

    @staticmethod
    def param_names() -> list[str]:
        return [
            "grid_spacing_norm",  # grid_spacing_pct / 0.02
            "range_width_norm",   # range_width_pct / 0.20
            "hold_days_norm",     # hold_days / 14
            "n_grids_norm",       # n_grids / 50
        ]

    def to_feature_vector(self) -> list[float]:
        return [
            min(self.grid_spacing_pct / 0.02, 1.0),
            min(self.range_width_pct  / 0.20, 1.0),
            self.hold_days / 14.0,
            min(self.n_grids / 50.0,  1.0),
        ]


# ── Parameter grid ────────────────────────────────────────────────────────────

def generate_grid_param_grid() -> list[GridConfig]:
    """
    Generate grid bot parameter grid.

    4 spacings × 3 range widths × 3 hold durations = 36 configs.

    Only include configs where profit_per_cycle > TC overhead:
        spacing >= 0.3% means gross >= 0.3% per cycle
        At 4bps TC: net = 0.3% - 0.08% = 0.22% per cycle → viable
    """
    configs = []
    cid = 0
    for spacing, width, hold in itertools.product(
        [0.003, 0.005, 0.010, 0.020],   # 0.3%, 0.5%, 1.0%, 2.0%
        [0.05,  0.10,  0.20],            # 5%, 10%, 20% range
        [3,     7,     14],              # days
    ):
        # Skip configs where spacing > range (degenerate)
        if spacing >= width:
            continue
        configs.append(GridConfig(
            config_id=f"grid_{cid:04d}",
            grid_spacing_pct=spacing,
            range_width_pct=width,
            hold_days=hold,
        ))
        cid += 1
    return configs


# ── Grid simulation ───────────────────────────────────────────────────────────

def simulate_grid_hold(
    prices:   np.ndarray,    # hourly prices for the hold window
    config:   GridConfig,
    tc_pct:   float = 0.0004,  # one-way TC
) -> dict:
    """
    Simulate grid bot P&L over a hold window.

    Mechanics:
        1. Set center = prices[0], define range [lower, upper]
        2. Compute grid lines within range
        3. Walk hourly prices, detect grid line crossings
        4. Each downward crossing = BUY at that grid price
        5. Each upward crossing = SELL at that grid price
        6. Match buys and sells at adjacent levels = completed cycles
        7. Range breakout = close all inventory at current price
        8. End of hold = close remaining inventory at last price

    Returns dict with net_return, gross_return, n_cycles, breakout, profitable
    """
    if len(prices) < 2:
        return {"net_return": 0.0, "gross_return": 0.0,
                "n_cycles": 0, "breakout": False, "profitable": False}

    center    = prices[0]
    half      = config.range_width_pct / 2
    lower     = center * (1 - half)
    upper     = center * (1 + half)
    spacing   = config.grid_spacing_pct

    # Grid lines (prices at each level)
    n_lines   = config.n_grids + 1
    grid_lines = np.array([lower + i * (upper - lower) / n_lines
                           for i in range(n_lines + 1)])

    # State: inventory[level_idx] = number of units held from that level
    inventory = np.zeros(len(grid_lines), dtype=float)
    unit_size = 1.0 / len(grid_lines)  # equal allocation per level

    gross_pnl = 0.0
    n_cycles  = 0
    breakout  = False
    prev_price = prices[0]

    for price in prices[1:]:
        # Check for range breakout
        if price < lower or price > upper:
            # Close all inventory at current price
            for idx, units in enumerate(inventory):
                if units > 0:
                    buy_price   = grid_lines[idx]
                    sell_price  = price
                    gross       = (sell_price - buy_price) / buy_price * units
                    gross_pnl  += gross
                    # TC already paid on entry, pay exit TC
                    gross_pnl  -= tc_pct * units
                    inventory[idx] = 0
            breakout = True
            break

        # Detect grid line crossings (price moved down → buy, up → sell)
        for idx, gl in enumerate(grid_lines):
            # Price crossed this line downward (buy)
            if prev_price > gl >= price:
                # Buy at this grid line
                inventory[idx] += unit_size
                gross_pnl -= tc_pct * unit_size  # entry TC

            # Price crossed this line upward (sell)
            elif prev_price < gl <= price:
                # Find lowest buy to match against
                buy_idx = None
                for j in range(idx - 1, -1, -1):
                    if inventory[j] > 0:
                        buy_idx = j
                        break

                if buy_idx is not None:
                    units = min(inventory[buy_idx], unit_size)
                    buy_price   = grid_lines[buy_idx]
                    sell_price  = gl
                    gross       = (sell_price - buy_price) / buy_price * units
                    gross_pnl  += gross
                    gross_pnl  -= tc_pct * units  # exit TC
                    inventory[buy_idx] -= units
                    n_cycles  += 1

        prev_price = price

    # Close remaining inventory at last price
    if not breakout:
        last_price = prices[-1]
        for idx, units in enumerate(inventory):
            if units > 0:
                buy_price  = grid_lines[idx]
                gross      = (last_price - buy_price) / buy_price * units
                gross_pnl += gross
                gross_pnl -= tc_pct * units  # exit TC
                inventory[idx] = 0

    net = float(np.clip(gross_pnl - tc_pct, -0.99, 5.0))  # entry TC once
    gross = float(np.clip(gross_pnl, -0.99, 5.0))

    return {
        "net_return":  net,
        "gross_return": gross,
        "n_cycles":    n_cycles,
        "breakout":    breakout,
        "profitable":  net > 0,
    }


def run_grid_single_day(
    spot_all:  pd.DataFrame,    # all spot bars (history + hold window)
    config:    GridConfig,
    tc_bps:    float = 4.0,
    as_of:     pd.Timestamp | None = None,
) -> dict:
    """
    For one evaluation day (as_of = day_start):
        - Check if conditions are appropriate to start a grid
        - Simulate grid hold for hold_days from as_of

    Returns {"daily_return", "gross_return", "n_trades"}
    """
    tc_pct = tc_bps / 10000.0

    if spot_all.empty:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    if as_of is None:
        as_of = spot_all.index[0].normalize()
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")

    hold_end  = as_of + timedelta(days=config.hold_days)
    hold_bars = spot_all[
        (spot_all.index >= as_of) & (spot_all.index <= hold_end)
    ]["close"].values

    if len(hold_bars) < 4:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    result = simulate_grid_hold(hold_bars, config, tc_pct)

    if not np.isfinite(result["net_return"]):
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    return {
        "daily_return": float(result["net_return"]),
        "gross_return": float(result["gross_return"]),
        "n_trades":     result["n_cycles"],
    }


# ── Daily feature computation ─────────────────────────────────────────────────

def _compute_grid_features(
    spot_bars: pd.DataFrame,
    as_of_date: str,
    config: GridConfig | None = None,
) -> np.ndarray | None:
    """Compute grid-bot-relevant daily features from hourly bars."""
    as_of = pd.Timestamp(as_of_date)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")

    hist_start = as_of - timedelta(days=35)
    hist = spot_bars[(spot_bars.index >= hist_start) & (spot_bars.index < as_of)]

    if len(hist) < 48:
        return None

    close = hist["close"]

    def safe_ret(n_bars: int) -> float:
        if len(close) < n_bars + 1:
            return 0.0
        return float((close.iloc[-1] - close.iloc[-n_bars-1]) /
                     (close.iloc[-n_bars-1] + 1e-10))

    ret_1h  = safe_ret(1)
    ret_4h  = safe_ret(4)
    ret_24h = safe_ret(24)

    # Realized volatilities at multiple horizons
    rets_1h  = close.pct_change().dropna()
    rets_4h  = close.resample("4h").last().pct_change().dropna()

    vol_1h   = float(rets_1h.iloc[-24:].std()  * np.sqrt(24  * 365)) if len(rets_1h)  >= 4 else 0.01
    vol_4h   = float(rets_4h.iloc[-42:].std()  * np.sqrt(6   * 365)) if len(rets_4h)  >= 4 else 0.01
    vol_24h  = float(rets_1h.iloc[-24:].std()  * np.sqrt(24  * 365)) if len(rets_1h)  >= 24 else 0.01
    vol_30d  = float(rets_1h.std() * np.sqrt(24 * 365)) if len(rets_1h) >= 5 else 0.01
    vol_regime = vol_24h / (vol_30d + 1e-10)

    # Autocorrelation (negative = mean-reverting = good for grids)
    def autocorr(rets: pd.Series, lag: int = 1) -> float:
        if len(rets) < lag + 4:
            return 0.0
        r = float(rets.autocorr(lag=lag))
        return r if np.isfinite(r) else 0.0

    ac_1h = autocorr(rets_1h.iloc[-48:], lag=1)
    ac_4h = autocorr(rets_4h.iloc[-14:], lag=1) if len(rets_4h) >= 4 else 0.0

    # Range position (where in 30d range is current price)
    low_30d  = float(close.min())
    high_30d = float(close.max())
    rng      = high_30d - low_30d
    range_pos = (float(close.iloc[-1]) - low_30d) / (rng + 1e-10)

    # Deviation from moving averages
    ma_24h = float(close.iloc[-24:].mean()) if len(close) >= 24 else float(close.mean())
    ma_7d  = float(close.iloc[-168:].mean()) if len(close) >= 168 else float(close.mean())
    px     = float(close.iloc[-1])
    dev_24h = (px - ma_24h) / (px + 1e-10)
    dev_7d  = (px - ma_7d)  / (px + 1e-10)

    # Volume surge
    vol_series = hist["volume"] if "volume" in hist.columns else pd.Series([1.0])
    vol_24h_sum = float(vol_series.iloc[-24:].sum()) if len(vol_series) >= 24 else 0.0
    vol_7d_avg  = float(vol_series.iloc[-168:].mean()) * 24 if len(vol_series) >= 168 else vol_24h_sum
    vol_surge   = vol_24h_sum / (vol_7d_avg + 1e-10)

    # Estimated cycle rate: how many grid crossings per day given this vol
    # Rough estimate: cycles ≈ vol_1h * sqrt(24) / grid_spacing
    grid_spacing = config.grid_spacing_pct if config else 0.005
    cycle_est = min((vol_1h / np.sqrt(365) / (grid_spacing + 1e-10)), 20.0)

    features = np.array([
        np.clip(ret_1h,    -0.1, 0.1),
        np.clip(ret_4h,    -0.2, 0.2),
        np.clip(ret_24h,   -0.3, 0.3),
        np.clip(vol_1h,     0,   5),
        np.clip(vol_4h,     0,   5),
        np.clip(vol_24h,    0,   5),
        np.clip(ac_1h,     -1,   1),
        np.clip(ac_4h,     -1,   1),
        np.clip(range_pos,  0,   1),
        np.clip(dev_24h,  -0.1,  0.1),
        np.clip(dev_7d,   -0.2,  0.2),
        np.clip(vol_regime, 0,   5),
        np.clip(vol_surge,  0,   5),
        np.clip(cycle_est,  0,   20),
    ], dtype=np.float32)

    return np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_spot(
    assets:    list[str],
    start:     str,
    end:       str,
    quote:     str = "USDT",
    cache_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch spot hourly bars via CCXT Binance."""
    import ccxt
    from datetime import datetime, timezone

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    exchange = ccxt.binance({"enableRateLimit": True})
    results  = {}

    for asset in assets:
        symbol     = f"{asset}/{quote}"
        cache_path = cache_dir / f"grid_spot_{asset}_{start}_{end}.parquet" if cache_dir else None

        if cache_path and cache_path.exists():
            try:
                results[asset] = pd.read_parquet(cache_path)
                logger.info(f"  Cache hit: {asset}")
                continue
            except Exception:
                pass

        logger.info(f"  Fetching {symbol} {start}→{end}")
        try:
            since_ms = int(datetime.strptime(start, "%Y-%m-%d")
                          .replace(tzinfo=timezone.utc).timestamp() * 1000)
            end_ms   = int(datetime.strptime(end, "%Y-%m-%d")
                          .replace(tzinfo=timezone.utc).timestamp() * 1000)

            all_bars, cursor = [], since_ms
            while cursor < end_ms:
                bars = exchange.fetch_ohlcv(symbol, "1h", since=cursor, limit=1000)
                if not bars:
                    break
                bars = [b for b in bars if b[0] < end_ms]
                if not bars:
                    break
                all_bars.extend(bars)
                last_ts = bars[-1][0]
                if last_ts <= cursor:
                    break
                cursor = last_ts + 1

            if all_bars:
                df = pd.DataFrame(all_bars,
                    columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
                df = df[df.index < pd.Timestamp(end, tz="UTC")]
                if cache_path:
                    df.to_parquet(cache_path)
                results[asset] = df
                logger.info(f"    {asset}: {len(df)} bars")

        except Exception as e:
            logger.warning(f"  Failed {asset}: {e}")

    return results


# ── TradingStrategy implementation ────────────────────────────────────────────

class GridBotStrategy:
    """
    Portfolio of grid bots as a CPO TradingStrategy.

    Each model = one asset.
    Configs = grid spacing × range width × hold duration (36 combos).
    RF learns: which market regimes (choppy vs trending) favor grid bots
    at each spacing/range combination.

    Expected CPO fit:
        - Base rate ~40-55% (sideways markets ~half the time in crypto)
        - RF learns: negative autocorr + high vol + low momentum = grid profits
        - AUC target: 0.70-0.85
    """

    def __init__(
        self,
        assets:         list[str] | None = None,
        cache_dir:      str | Path = "data/grid_cache",
        tc_bps:         float = 4.0,
        training_start: str = "2024-01-01",
        training_end:   str = "2024-12-31",
        quote:          str = "USDT",
    ):
        self.assets         = assets or DEFAULT_ASSETS
        self.cache_dir      = Path(cache_dir)
        self.tc_bps         = tc_bps
        self.training_start = training_start
        self.training_end   = training_end
        self.quote          = quote

        self._models  = [
            GridModelSpec(model_id=f"{a}_GRID", asset=a)
            for a in self.assets
        ]
        self._configs = generate_grid_param_grid()

        print(f"  Grid Bot CPO: {len(self._models)} models ({len(self.assets)} assets)")
        print(f"  Total configs: {len(self._configs)}")
        print(f"  TC: {tc_bps} bps one-way | Grid spacings: 0.3/0.5/1.0/2.0%")
        print(f"  Range widths: 5/10/20% | Hold: 3/7/14 days")

    # ── Protocol ──────────────────────────────────────────────────

    def get_models(self):     return self._models
    def get_param_grid(self): return self._configs

    def daily_feature_names(self) -> list[str]:
        return list(GRID_FEATURES)

    def config_param_names(self) -> list[str]:
        return GridConfig.param_names()

    def config_to_features(self, config: GridConfig) -> list[float]:
        return config.to_feature_vector()

    def compute_features(self, model: GridModelSpec, as_of_date: str,
                          data: dict) -> np.ndarray | None:
        spot = data.get("spot", {}).get(model.asset)
        if spot is None or len(spot) < 48:
            return None
        return _compute_grid_features(spot, as_of_date)

    def run_single_day(self, model: GridModelSpec, config: GridConfig,
                        day: Any, data: dict) -> dict:
        spot = data.get("spot", {}).get(model.asset)
        if spot is None:
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

        day_start = pd.Timestamp(day)
        if day_start.tzinfo is None:
            day_start = day_start.tz_localize("UTC")

        hold_end   = day_start + timedelta(days=config.hold_days + 1)
        spot_window = spot[spot.index <= hold_end]

        return run_grid_single_day(spot_window, config, self.tc_bps, as_of=day_start)

    def run_model_year(self, model: GridModelSpec, data: dict,
                        param_grid: list[GridConfig]) -> pd.DataFrame:
        spot = data.get("spot", {}).get(model.asset)
        if spot is None:
            return pd.DataFrame()

        trading_days = sorted(spot.index.normalize().unique())
        results = []

        for day_idx, day in enumerate(trading_days):
            day_start = pd.Timestamp(day)
            if day_start.tzinfo is None:
                day_start = day_start.tz_localize("UTC")

            # Include up to 14 days ahead for hold window
            hold_end    = day_start + timedelta(days=15)
            spot_window = spot[spot.index <= hold_end]

            if len(spot_window) < 4:
                continue

            for config in param_grid:
                result = run_grid_single_day(
                    spot_window, config, self.tc_bps, as_of=day_start
                )
                dr, gr = result["daily_return"], result["gross_return"]
                if not (np.isfinite(dr) and np.isfinite(gr)):
                    continue
                results.append({
                    "model_id":     model.model_id,
                    "date":         day.strftime("%Y-%m-%d"),
                    "config_id":    config.config_id,
                    "daily_return": float(dr),
                    "gross_return": float(gr),
                    "n_trades":     result["n_trades"],
                })

            if (day_idx + 1) % 30 == 0:
                print(f"    {model.model_id}: {day_idx+1}/{len(trading_days)} days")

        return pd.DataFrame(results)

    # ── Data fetching protocol ─────────────────────────────────────

    def fetch_training_data(self, models, start, end) -> dict:
        assets = list({m.asset for m in models})
        spot   = _fetch_spot(assets, self.training_start, self.training_end,
                             self.quote, self.cache_dir)
        return {"spot": spot}

    def fetch_oos_data(self, models, start, end) -> dict:
        assets = list({m.asset for m in models})
        spot   = _fetch_spot(assets, start, end, self.quote, self.cache_dir)
        return {"spot": spot}

    def fetch_warmup_daily(self, models, start, end) -> dict[str, pd.Series]:
        assets = list({m.asset for m in models})
        spot   = _fetch_spot(assets, start, end, self.quote, self.cache_dir)
        return {a: b["close"].resample("D").last().dropna() for a, b in spot.items()}

    def get_daily_prices(self, oos_data: dict, models) -> dict[str, pd.Series]:
        spot = oos_data.get("spot", {})
        return {a: b["close"].resample("D").last().dropna() for a, b in spot.items()}

    def get_trading_days(self, data: dict) -> list:
        spot = data.get("spot", {})
        if not spot:
            return []
        all_days: set = set()
        for bars in spot.values():
            all_days.update(bars.index.normalize().unique())
        return sorted(all_days)

    def prepare_warmup(self, daily_prices: dict, warmup_daily: dict) -> dict:
        result = {}
        for asset in daily_prices:
            warm = warmup_daily.get(asset, pd.Series(dtype=float))
            curr = daily_prices[asset]
            if not warm.empty:
                combined = pd.concat([warm, curr])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                result[asset] = combined
            else:
                result[asset] = curr
        return result
