"""
engines/vol_strategy.py
=========================
Volatility Risk Premium (VRP) strategy as a CPO TradingStrategy.

MECHANISM:
    Compare implied vol (Deribit DVOL) to GARCH-forecasted realized vol.
    VRP = IV - E[RV]  (in annualized vol points)

    When VRP > 0 (IV > expected RV):
        → Vol is "expensive" → SELL vol (short straddle / short vega)
        → Collect theta, pay if vol is higher than expected

    When VRP < 0 (IV < expected RV):
        → Vol is "cheap" → BUY vol (long straddle / gamma scalp)
        → Pay theta, profit if vol is higher than expected

P&L APPROXIMATION (delta-hedged straddle):
    For a delta-neutral position, the daily P&L is approximately:
        P&L_day ≈ ½ * Γ * S² * (σ²_realized - σ²_implied) * dt

    Integrated over the hold period:
        P&L_hold = K * (σ²_realized_hold - σ²_entry_IV²) * hold_days/365

    where K is a notional scaling factor (we normalize to 1.0).

    For SHORT vol: multiply by -1.
    TC: ~10 bps on entry (options bid-ask wider than spot/perp).

CPO FIT:
    Models: BTC_7d, BTC_30d, ETH_7d, ETH_30d (asset × expiry tenor)
    Configs: vrp_entry_threshold × hold_days × direction
    RF question: "Will selling vol be profitable over the next N days?"
    Base rate: ~55-60% (VRP tends to be positive — vol sellers usually win)
    AUC target: 0.65-0.75 (weaker than funding carry — vol regime harder to predict)

DATA:
    IV:   Deribit DVOL index (daily, free public API)
    RV:   Computed from hourly spot prices (same source as other strategies)
    GARCH: Fit on daily returns, forecast at 7d and 30d horizons
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ASSETS = ["BTC", "ETH"]
DEFAULT_TENORS = [7, 30]       # days — expiry tenors to model separately

VOL_FEATURES = [
    "iv_atm_30d",          # current DVOL (30d IV index)
    "vrp_7d",              # IV_7d - GARCH_7d forecast (vol points)
    "vrp_30d",             # IV_30d - GARCH_30d forecast
    "iv_pct_rank",         # IV percentile rank (0=low, 1=high)
    "iv_trend",            # IV change last 21 days
    "vov_30d",             # vol-of-vol (std of daily IV changes)
    "garch_persistence",   # GARCH alpha+beta
    "garch_vol_7d",        # GARCH 7d forecast
    "garch_vol_30d",       # GARCH 30d forecast
    "realized_vol_30d",    # realized vol last 30 days
    "rv_iv_ratio_30d",     # RV/IV ratio (1 = fairly priced, <1 = expensive IV)
]


@dataclass
class VolModelSpec:
    model_id: str
    asset:    str
    tenor:    int        # days (7 or 30)


@dataclass
class VolConfig:
    config_id:          str
    min_vrp_sell:       float   # min VRP (IV - GARCH) to sell vol (vol pts, e.g. 5.0)
    max_vrp_buy:        float   # max VRP to buy vol (negative, e.g. -5.0)
    hold_days:          int     # position hold duration
    direction:          str     # "sell", "buy", "best" (RF picks)

    @staticmethod
    def param_names() -> list[str]:
        return [
            "min_vrp_norm",     # min_vrp_sell / 50
            "max_vrp_norm",     # (max_vrp_buy + 50) / 50
            "hold_days_norm",   # hold_days / 30
            "direction_enc",    # 0=sell, 1=buy
        ]

    def to_feature_vector(self) -> list[float]:
        dir_enc = {"sell": 0.0, "buy": 1.0, "best": 0.5}
        return [
            min(self.min_vrp_sell / 50.0, 1.0),
            min((self.max_vrp_buy + 50.0) / 50.0, 1.0),
            self.hold_days / 30.0,
            dir_enc.get(self.direction, 0.5),
        ]


def generate_vol_param_grid() -> list[VolConfig]:
    """
    Generate vol strategy parameter grid.

    For SELL vol (VRP > threshold):
        min_vrp_sell: 0, 3, 7, 12 vol points above GARCH forecast

    For BUY vol (VRP < threshold):
        max_vrp_buy: 0, -3, -7 vol points below GARCH forecast

    Hold periods: 7, 14, 30 days

    Total: (4 sell + 3 buy) × 3 holds = 21 configs per model.
    """
    configs = []
    cid     = 0

    # Short vol configs
    for vrp_min in [0.0, 3.0, 7.0, 12.0]:
        for hold in [7, 14, 30]:
            configs.append(VolConfig(
                config_id    = f"vol_{cid:04d}",
                min_vrp_sell = vrp_min,
                max_vrp_buy  = 999.0,   # not used
                hold_days    = hold,
                direction    = "sell",
            ))
            cid += 1

    # Long vol configs
    for vrp_max in [0.0, -3.0, -7.0]:
        for hold in [7, 14, 30]:
            configs.append(VolConfig(
                config_id    = f"vol_{cid:04d}",
                min_vrp_sell = -999.0,  # not used
                max_vrp_buy  = vrp_max,
                hold_days    = hold,
                direction    = "buy",
            ))
            cid += 1

    return configs


# ── P&L simulation ────────────────────────────────────────────────────────────

def simulate_vol_trade(
    entry_iv:   float,        # ATM IV at entry (annualized decimal, e.g. 0.75)
    rv_hold:    float,        # realized vol over hold period (annualized decimal)
    hold_days:  int,
    direction:  str,          # "sell" or "buy"
    tc_bps:     float = 10.0, # one-way TC (options wider than spot)
) -> dict:
    """
    Simulate P&L of a delta-hedged ATM straddle using variance swap approximation.

    P&L ≈ ½ * (IV² - RV²) * hold_days/365   [for short vol]
         ≈ ½ * (RV² - IV²) * hold_days/365   [for long vol]

    This is the continuous-time limit of daily delta-hedging.
    We normalize notional to 1.0 (percent of portfolio).

    TC: paid at entry and exit. Options have wider spreads than spot.
    """
    tc_pct = tc_bps / 10000.0
    t      = hold_days / 365.0

    # Variance swap P&L (per unit notional)
    iv_var = entry_iv**2
    rv_var = rv_hold**2

    if direction == "sell":
        gross = 0.5 * (iv_var - rv_var) * t
    else:  # buy
        gross = 0.5 * (rv_var - iv_var) * t

    net = gross - tc_pct * 2   # entry + exit
    net   = float(np.clip(net,   -0.99, 5.0))
    gross = float(np.clip(gross, -0.99, 5.0))

    return {
        "net_return":   net,
        "gross_return": gross,
        "profitable":   net > 0,
        "vrp":          (entry_iv - rv_hold) * 100,  # vol points
    }


def run_vol_single_day(
    spot_all:   pd.DataFrame,
    dvol_hist:  pd.Series,
    garch_fc:   pd.DataFrame,    # rolling GARCH forecast DataFrame
    config:     VolConfig,
    as_of:      pd.Timestamp,
    tc_bps:     float = 10.0,
) -> dict:
    """
    Evaluate one day's vol trade.

    Entry conditions:
        SELL vol: VRP = IV - GARCH_forecast > config.min_vrp_sell
        BUY vol:  VRP = IV - GARCH_forecast < config.max_vrp_buy

    P&L: realized vol over hold_days vs entry IV.
    """
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")

    # Entry IV from DVOL (or nearest available)
    dvol_to_date = dvol_hist[dvol_hist.index <= as_of]
    if dvol_to_date.empty:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}
    entry_iv = float(dvol_to_date.iloc[-1])

    # GARCH forecast for today
    garch_today = garch_fc[garch_fc.index <= as_of]
    if garch_today.empty:
        garch_rv = entry_iv  # fallback: assume IV = GARCH
    else:
        col = "vol_7d" if config.hold_days <= 10 else "vol_30d"
        garch_rv = float(garch_today[col].iloc[-1])

    # Compute VRP
    vrp = (entry_iv - garch_rv) * 100   # vol points

    # Check entry conditions
    if config.direction == "sell" and vrp < config.min_vrp_sell:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}
    if config.direction == "buy" and vrp > config.max_vrp_buy:
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    # Realized vol over hold period
    hold_end = as_of + timedelta(days=config.hold_days)
    from engines.garch_model import compute_realized_vol
    rv_hold = compute_realized_vol(spot_all["close"], as_of, hold_end)

    if not np.isfinite(rv_hold):
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    result = simulate_vol_trade(entry_iv, rv_hold, config.hold_days,
                                config.direction, tc_bps)

    if not np.isfinite(result["net_return"]):
        return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

    return {
        "daily_return": float(result["net_return"]),
        "gross_return": float(result["gross_return"]),
        "n_trades":     1,
    }


# ── Feature computation ───────────────────────────────────────────────────────

def _compute_vol_features(
    spot_bars:  pd.DataFrame,
    dvol_hist:  pd.Series,
    garch_fc:   pd.DataFrame,
    as_of_date: str,
) -> np.ndarray | None:
    """Compute vol strategy features as of a given date."""
    as_of = pd.Timestamp(as_of_date)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")

    dvol_to_date = dvol_hist[dvol_hist.index <= as_of]
    if len(dvol_to_date) < 30:
        return None

    iv_30d = float(dvol_to_date.iloc[-1])

    # GARCH forecasts
    garch_today = garch_fc[garch_fc.index <= as_of]
    if garch_today.empty:
        return None
    garch_7d   = float(garch_today["vol_7d"].iloc[-1])
    garch_30d  = float(garch_today["vol_30d"].iloc[-1])
    persist    = float(garch_today["persistence"].iloc[-1]) if "persistence" in garch_today else 0.85

    vrp_7d  = (iv_30d - garch_7d)  * 100   # vol points
    vrp_30d = (iv_30d - garch_30d) * 100

    # Vol-of-vol
    dvol_30d = dvol_to_date.iloc[-30:]
    dvol_chg = dvol_30d.diff().dropna()
    vov      = float(dvol_chg.std()) if len(dvol_chg) > 2 else 0.05

    # IV percentile rank
    dvol_252 = dvol_to_date.iloc[-252:] if len(dvol_to_date) >= 252 else dvol_to_date
    iv_rank  = float((dvol_252 < iv_30d).mean()) if len(dvol_252) > 10 else 0.5

    # IV trend (21d)
    if len(dvol_to_date) >= 22:
        iv_trend = (iv_30d - float(dvol_to_date.iloc[-22])) / (float(dvol_to_date.iloc[-22]) + 1e-6)
    else:
        iv_trend = 0.0

    # Realized vol last 30d
    spot_30d = spot_bars[spot_bars.index >= (as_of - timedelta(days=31))]
    spot_30d = spot_30d[spot_30d.index <= as_of]
    from engines.garch_model import compute_realized_vol
    rv_30d = compute_realized_vol(spot_30d["close"], spot_30d.index[0] if not spot_30d.empty else as_of - timedelta(days=30), as_of) if len(spot_30d) > 24 else iv_30d
    if not np.isfinite(rv_30d):
        rv_30d = iv_30d

    rv_iv_ratio = rv_30d / (iv_30d + 1e-6)

    features = np.array([
        np.clip(iv_30d,       0,   5),
        np.clip(vrp_7d,     -50,  50),
        np.clip(vrp_30d,    -50,  50),
        np.clip(iv_rank,      0,   1),
        np.clip(iv_trend,    -1,   1),
        np.clip(vov,          0,   0.5),
        np.clip(persist,      0,   1),
        np.clip(garch_7d,     0,   5),
        np.clip(garch_30d,    0,   5),
        np.clip(rv_30d,       0,   5),
        np.clip(rv_iv_ratio,  0,   3),
    ], dtype=np.float32)

    return np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_spot(assets, start, end, quote="USDT", cache_dir=None):
    """Reuse spot fetch logic (same as other strategies)."""
    import ccxt
    from datetime import datetime, timezone

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    exchange = ccxt.binance({"enableRateLimit": True})
    results  = {}

    for asset in assets:
        symbol = f"{asset}/{quote}"
        cp     = cache_dir / f"vol_spot_{asset}_{start}_{end}.parquet" if cache_dir else None

        if cp and cp.exists():
            try:
                results[asset] = pd.read_parquet(cp)
                continue
            except Exception:
                pass

        try:
            s_ms = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
            e_ms = int(datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
            all_bars, cur = [], s_ms
            while cur < e_ms:
                bars = exchange.fetch_ohlcv(symbol, "1h", since=cur, limit=1000)
                if not bars: break
                bars = [b for b in bars if b[0] < e_ms]
                if not bars: break
                all_bars.extend(bars)
                last = bars[-1][0]
                if last <= cur: break
                cur = last + 1
            if all_bars:
                df = pd.DataFrame(all_bars, columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
                df = df[df.index < pd.Timestamp(end, tz="UTC")]
                if cp: df.to_parquet(cp)
                results[asset] = df
                logger.info(f"  {asset} spot: {len(df)} bars")
        except Exception as e:
            logger.warning(f"  Spot failed {asset}: {e}")

    return results


# ── TradingStrategy implementation ────────────────────────────────────────────

class VolStrategy:
    """
    Volatility Risk Premium strategy as a CPO TradingStrategy.

    Models:  BTC_7d, BTC_30d, ETH_7d, ETH_30d
    Configs: VRP threshold × direction × hold duration (21 configs)
    Label:   +1 if selling vol was profitable, 0 if buying vol was profitable

    Data pipeline:
        1. Fetch DVOL history (Deribit)
        2. Fit rolling GARCH on hourly spot → daily vol forecasts
        3. Compute VRP = IV - GARCH on each day
        4. Simulate straddle P&L vs actual RV over hold period
        5. RF learns: which VRP levels / regimes → profitable vol selling
    """

    def __init__(
        self,
        assets:         list[str] | None = None,
        tenors:         list[int] | None = None,
        cache_dir:      str | Path = "data/vol_cache",
        tc_bps:         float = 10.0,
        training_start: str = "2023-01-01",
        training_end:   str = "2023-12-31",
        quote:          str = "USDT",
    ):
        self.assets         = assets or DEFAULT_ASSETS
        self.tenors         = tenors or DEFAULT_TENORS
        self.cache_dir      = Path(cache_dir)
        self.tc_bps         = tc_bps
        self.training_start = training_start
        self.training_end   = training_end
        self.quote          = quote

        self._models = [
            VolModelSpec(f"{a}_{t}d_VOL", a, t)
            for a in self.assets
            for t in self.tenors
        ]
        self._configs = generate_vol_param_grid()

        print(f"  Vol Strategy CPO: {len(self._models)} models "
              f"({len(self.assets)} assets × {len(self.tenors)} tenors)")
        print(f"  Total configs: {len(self._configs)}")
        print(f"  TC: {tc_bps} bps (options wider than spot)")
        print(f"  Models: {[m.model_id for m in self._models]}")

    def get_models(self):     return self._models
    def get_param_grid(self): return self._configs

    def daily_feature_names(self):  return list(VOL_FEATURES)
    def config_param_names(self):   return VolConfig.param_names()
    def config_to_features(self, c): return c.to_feature_vector()

    def compute_features(self, model, as_of_date, data):
        spot     = data.get("spot", {}).get(model.asset)
        dvol     = data.get("dvol", {}).get(model.asset)
        garch_fc = data.get("garch_fc", {}).get(model.asset)
        if spot is None or dvol is None or garch_fc is None:
            return None
        return _compute_vol_features(spot, dvol, garch_fc, as_of_date)

    def run_single_day(self, model, config, day, data):
        spot     = data.get("spot",     {}).get(model.asset)
        dvol     = data.get("dvol",     {}).get(model.asset)
        garch_fc = data.get("garch_fc", {}).get(model.asset)
        if spot is None or dvol is None or garch_fc is None:
            return {"daily_return": 0.0, "gross_return": 0.0, "n_trades": 0}

        day_start = pd.Timestamp(day)
        if day_start.tzinfo is None:
            day_start = day_start.tz_localize("UTC")

        hold_end    = day_start + timedelta(days=config.hold_days + 1)
        spot_window = spot[spot.index <= hold_end]

        return run_vol_single_day(
            spot_window, dvol, garch_fc, config, day_start, self.tc_bps
        )

    def run_model_year(self, model, data, param_grid):
        spot     = data.get("spot",     {}).get(model.asset)
        dvol     = data.get("dvol",     {}).get(model.asset)
        garch_fc = data.get("garch_fc", {}).get(model.asset)

        if spot is None or dvol is None or garch_fc is None:
            logger.warning(f"  {model.model_id}: missing data, skipping")
            return pd.DataFrame()

        trading_days = sorted(spot.index.normalize().unique())
        results      = []

        for day_idx, day in enumerate(trading_days):
            day_start   = pd.Timestamp(day)
            if day_start.tzinfo is None:
                day_start = day_start.tz_localize("UTC")

            hold_end    = day_start + timedelta(days=31)
            spot_window = spot[spot.index <= hold_end]

            for config in param_grid:
                result = run_vol_single_day(
                    spot_window, dvol, garch_fc, config, day_start, self.tc_bps
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

    def _prepare_data(self, assets, start, end):
        """Fetch spot, DVOL, and compute rolling GARCH forecasts."""
        from engines.garch_model import rolling_garch_forecasts

        # Extend start back for GARCH warmup
        garch_start = (pd.Timestamp(start) - timedelta(days=200)).strftime("%Y-%m-%d")
        dvol_start  = (pd.Timestamp(start) - timedelta(days=252)).strftime("%Y-%m-%d")

        spot_data = _fetch_spot(assets, garch_start, end, self.quote, self.cache_dir)
        dvol_data, garch_data = {}, {}

        for asset in assets:
            # Fetch DVOL (with synthetic fallback if API unavailable)
            from engines.vol_surface import fetch_dvol_history_with_fallback
            spot_for_fallback = spot_data.get(asset)
            spot_prices_for_fallback = (spot_for_fallback["close"]
                                        if spot_for_fallback is not None else None)
            print(f"  Fetching DVOL {asset}...")
            dvol = fetch_dvol_history_with_fallback(
                asset, dvol_start, end,
                hourly_prices=spot_prices_for_fallback,
                cache_dir=self.cache_dir,
            )
            if dvol.empty:
                print(f"    {asset}: no DVOL data (live or synthetic) — skipping")
                continue
            dvol_data[asset] = dvol
            print(f"    {asset} DVOL: {len(dvol)} days ✓")

            # Rolling GARCH on spot
            if asset in spot_data:
                print(f"  Fitting rolling GARCH {asset}...")
                fc = rolling_garch_forecasts(
                    spot_data[asset]["close"],
                    lookback_days=180,
                    refit_freq=7,
                )
                if not fc.empty:
                    garch_data[asset] = fc
                    print(f"    {asset} GARCH: {len(fc)} days ✓")

        return {"spot": spot_data, "dvol": dvol_data, "garch_fc": garch_data}

    def fetch_training_data(self, models, start, end):
        assets = list({m.asset for m in models})
        return self._prepare_data(assets, self.training_start, self.training_end)

    def fetch_oos_data(self, models, start, end):
        assets = list({m.asset for m in models})
        return self._prepare_data(assets, start, end)

    def fetch_warmup_daily(self, models, start, end):
        assets = list({m.asset for m in models})
        spot   = _fetch_spot(assets, start, end, self.quote, self.cache_dir)
        return {a: b["close"].resample("D").last().dropna() for a, b in spot.items()}

    def get_daily_prices(self, oos_data, models):
        spot = oos_data.get("spot", {})
        return {a: b["close"].resample("D").last().dropna() for a, b in spot.items()}

    def get_trading_days(self, data):
        spot = data.get("spot", {})
        all_days: set = set()
        for bars in spot.values():
            all_days.update(bars.index.normalize().unique())
        return sorted(all_days)

    def prepare_warmup(self, daily_prices, warmup_daily):
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
