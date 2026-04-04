"""
engines/momentum_signals.py — Short-Term Momentum Signal Detector

Detects high-confidence momentum conditions on crypto assets using
real-time 1-minute candle data. Designed for the flash loan looping
strategy: signals trigger leveraged position entry/exit via Aave.

Key design principle: we want SELECTIVE, HIGH-CONFIDENCE signals.
Not every minute produces a trade. We're looking for conditions where
multiple indicators align, suggesting a short-term move is already
underway with enough conviction to ride for 5-30 minutes at leverage.

Signals:
    1. Volume Spike — current volume >> recent average
    2. Price Velocity — rate of change over short windows
    3. Consecutive Candles — N same-direction candles in a row
    4. Range Breakout — price exceeding N-period high/low with volume
    5. Order Flow Imbalance — buy vs sell volume asymmetry

Usage:
    from engines.momentum_signals import MomentumDetector
    detector = MomentumDetector()
    detector.load_candles("BTC/USDT", lookback_minutes=120)
    signals = detector.compute_signals("BTC/USDT")
    if signals.composite_score > 0.7:
        print(f"LONG signal: {signals}")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MomentumSignal:
    """A single momentum signal from one indicator."""
    name: str
    direction: int       # +1 = bullish, -1 = bearish, 0 = neutral
    strength: float      # 0.0 to 1.0 (confidence)
    value: float         # raw indicator value
    threshold: float     # threshold used for signal generation
    description: str     # human-readable explanation

@dataclass
class CompositeSignal:
    """Combined signal from all indicators for one asset."""
    timestamp: datetime
    asset: str
    direction: int          # +1 = LONG, -1 = SHORT, 0 = NO TRADE
    composite_score: float  # -1.0 to +1.0 (negative = short, positive = long)
    confidence: float       # 0.0 to 1.0 (abs of composite_score)
    signals: list[MomentumSignal]
    current_price: float
    price_5m_ago: float
    price_15m_ago: float
    vol_ratio: float        # current vol / avg vol

    @property
    def is_actionable(self) -> bool:
        """True if signal is strong enough to trade."""
        return self.confidence >= 0.6 and self.direction != 0

    def summary(self) -> str:
        dir_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}[self.direction]
        return (f"{self.asset} {dir_str} score={self.composite_score:+.2f} "
                f"conf={self.confidence:.0%} price=${self.current_price:,.2f} "
                f"vol={self.vol_ratio:.1f}x")


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SignalParams:
    """Tunable parameters for signal generation."""
    # Volume spike
    vol_window: int = 30           # minutes for average volume baseline
    vol_spike_threshold: float = 2.5  # vol/avg ratio to trigger

    # Price velocity
    velocity_windows: list[int] = field(default_factory=lambda: [3, 5, 10])
    velocity_threshold_bps: float = 15.0  # min move in bps to trigger

    # Consecutive candles
    consec_min: int = 3            # minimum consecutive same-direction candles
    consec_vol_confirm: bool = True  # require increasing volume

    # Range breakout
    range_window: int = 60         # minutes for range computation
    range_breakout_pct: float = 0.3  # how far past range edge (as % of range)
    range_vol_confirm_ratio: float = 2.0  # volume must be Nx average

    # Order flow imbalance
    flow_window: int = 5           # minutes for flow computation
    flow_imbalance_threshold: float = 0.65  # buy_vol / total_vol ratio

    # Composite
    weights: dict[str, float] = field(default_factory=lambda: {
        "volume_spike": 0.15,
        "velocity": 0.30,
        "consecutive": 0.20,
        "breakout": 0.20,
        "flow": 0.15,
    })

    # Trade management
    min_composite_score: float = 0.6   # minimum |score| to trigger entry
    take_profit_bps: float = 50.0      # TP in bps on the underlying
    stop_loss_bps: float = 30.0        # SL in bps on the underlying
    max_hold_minutes: int = 30         # max hold before forced exit


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_volume_spike(df: pd.DataFrame, params: SignalParams) -> MomentumSignal:
    """
    Detect volume spikes relative to recent average.

    High volume = something is happening. Combined with direction,
    it suggests conviction behind the move.
    """
    if len(df) < params.vol_window + 1:
        return MomentumSignal("volume_spike", 0, 0.0, 0.0, 0.0, "insufficient data")

    avg_vol = df["volume"].iloc[-(params.vol_window + 1):-1].mean()
    current_vol = df["volume"].iloc[-1]

    if avg_vol <= 0:
        return MomentumSignal("volume_spike", 0, 0.0, 0.0, 0.0, "zero avg volume")

    ratio = current_vol / avg_vol

    if ratio >= params.vol_spike_threshold:
        # Direction from current candle
        candle_dir = 1 if df["close"].iloc[-1] > df["open"].iloc[-1] else -1
        # Strength scales from 0 at threshold to 1 at 2x threshold
        strength = min(1.0, (ratio - params.vol_spike_threshold) / params.vol_spike_threshold)
        return MomentumSignal(
            "volume_spike", candle_dir, strength, ratio,
            params.vol_spike_threshold,
            f"vol {ratio:.1f}x avg ({'bullish' if candle_dir > 0 else 'bearish'} candle)"
        )

    return MomentumSignal("volume_spike", 0, 0.0, ratio, params.vol_spike_threshold,
                          f"vol {ratio:.1f}x avg (below threshold)")


def compute_velocity(df: pd.DataFrame, params: SignalParams) -> MomentumSignal:
    """
    Price velocity — rate of change over multiple short windows.

    Looks at 3, 5, and 10-minute returns. Strong signal when all
    windows agree on direction AND magnitude exceeds threshold.
    """
    if len(df) < max(params.velocity_windows) + 1:
        return MomentumSignal("velocity", 0, 0.0, 0.0, 0.0, "insufficient data")

    current_price = df["close"].iloc[-1]
    velocities = []

    for w in params.velocity_windows:
        past_price = df["close"].iloc[-(w + 1)]
        if past_price > 0:
            change_bps = (current_price - past_price) / past_price * 10000
            velocities.append(change_bps)
        else:
            velocities.append(0.0)

    # All windows must agree on direction
    all_positive = all(v > 0 for v in velocities)
    all_negative = all(v < 0 for v in velocities)

    if not (all_positive or all_negative):
        avg_v = np.mean(velocities)
        return MomentumSignal("velocity", 0, 0.0, avg_v, params.velocity_threshold_bps,
                              f"mixed signals: {[f'{v:.1f}bps' for v in velocities]}")

    # Use the median velocity as the signal value
    median_v = float(np.median(velocities))
    direction = 1 if median_v > 0 else -1
    abs_v = abs(median_v)

    if abs_v >= params.velocity_threshold_bps:
        # Strength: 0 at threshold, 1 at 3x threshold
        strength = min(1.0, (abs_v - params.velocity_threshold_bps) /
                       (2 * params.velocity_threshold_bps))
        return MomentumSignal(
            "velocity", direction, strength, median_v,
            params.velocity_threshold_bps,
            f"{'up' if direction > 0 else 'down'} {abs_v:.1f}bps "
            f"across {params.velocity_windows}min windows"
        )

    return MomentumSignal("velocity", 0, 0.0, median_v, params.velocity_threshold_bps,
                          f"{abs_v:.1f}bps (below threshold)")


def compute_consecutive(df: pd.DataFrame, params: SignalParams) -> MomentumSignal:
    """
    Count consecutive same-direction candles.

    3+ green candles in a row with increasing volume = strong bullish.
    """
    if len(df) < params.consec_min + 1:
        return MomentumSignal("consecutive", 0, 0.0, 0.0, 0.0, "insufficient data")

    # Compute candle directions for recent history
    directions = np.sign(df["close"].iloc[-20:].values - df["open"].iloc[-20:].values)
    volumes = df["volume"].iloc[-20:].values

    # Count consecutive from the end
    if len(directions) == 0 or directions[-1] == 0:
        return MomentumSignal("consecutive", 0, 0.0, 0.0, float(params.consec_min),
                              "doji candle")

    current_dir = int(directions[-1])
    count = 0
    vol_increasing = True
    for i in range(len(directions) - 1, -1, -1):
        if directions[i] == current_dir:
            count += 1
            if count >= 2 and volumes[i] > volumes[i - 1] if i > 0 else True:
                pass  # volume still increasing
            elif count >= 2:
                vol_increasing = False
        else:
            break

    if count >= params.consec_min:
        vol_ok = not params.consec_vol_confirm or vol_increasing
        strength = min(1.0, (count - params.consec_min) / 3.0)
        if vol_ok:
            strength = min(1.0, strength + 0.3)

        return MomentumSignal(
            "consecutive", current_dir, strength, float(count),
            float(params.consec_min),
            f"{count} consecutive {'green' if current_dir > 0 else 'red'} candles"
            f"{' (vol confirming)' if vol_ok else ' (vol not confirming)'}"
        )

    return MomentumSignal("consecutive", 0, 0.0, float(count),
                          float(params.consec_min),
                          f"only {count} consecutive (need {params.consec_min})")


def compute_breakout(df: pd.DataFrame, params: SignalParams) -> MomentumSignal:
    """
    Range breakout — price exceeding N-period high/low with volume.

    A breakout beyond the recent range with above-average volume
    suggests a directional move is starting.
    """
    window = min(params.range_window, len(df) - 1)
    if window < 10:
        return MomentumSignal("breakout", 0, 0.0, 0.0, 0.0, "insufficient data")

    lookback = df.iloc[-(window + 1):-1]
    current = df.iloc[-1]

    high = lookback["high"].max()
    low = lookback["low"].min()
    range_size = high - low

    if range_size <= 0:
        return MomentumSignal("breakout", 0, 0.0, 0.0, 0.0, "zero range")

    current_price = current["close"]
    avg_vol = lookback["volume"].mean()
    current_vol = current["volume"]

    vol_confirmed = (current_vol / avg_vol >= params.range_vol_confirm_ratio
                     if avg_vol > 0 else False)

    breakout_threshold = range_size * params.range_breakout_pct

    if current_price > high + breakout_threshold:
        excess = (current_price - high) / range_size
        strength = min(1.0, excess) * (1.0 if vol_confirmed else 0.5)
        return MomentumSignal(
            "breakout", 1, strength, current_price,
            high + breakout_threshold,
            f"broke above {window}min high ${high:.2f} by {excess:.0%} of range"
            f"{' (vol confirmed)' if vol_confirmed else ''}"
        )

    if current_price < low - breakout_threshold:
        excess = (low - current_price) / range_size
        strength = min(1.0, excess) * (1.0 if vol_confirmed else 0.5)
        return MomentumSignal(
            "breakout", -1, strength, current_price,
            low - breakout_threshold,
            f"broke below {window}min low ${low:.2f} by {excess:.0%} of range"
            f"{' (vol confirmed)' if vol_confirmed else ''}"
        )

    return MomentumSignal("breakout", 0, 0.0, current_price, 0.0,
                          f"within range [${low:.2f}, ${high:.2f}]")


def compute_flow_imbalance(df: pd.DataFrame, params: SignalParams) -> MomentumSignal:
    """
    Order flow imbalance — buy vs sell volume asymmetry.

    Approximated from candle data: volume on green candles is "buy volume",
    volume on red candles is "sell volume". Not perfect but useful as a
    signal component.
    """
    window = min(params.flow_window, len(df))
    if window < 2:
        return MomentumSignal("flow", 0, 0.0, 0.5, 0.0, "insufficient data")

    recent = df.iloc[-window:]
    buy_vol = recent.loc[recent["close"] >= recent["open"], "volume"].sum()
    sell_vol = recent.loc[recent["close"] < recent["open"], "volume"].sum()
    total_vol = buy_vol + sell_vol

    if total_vol <= 0:
        return MomentumSignal("flow", 0, 0.0, 0.5, params.flow_imbalance_threshold,
                              "zero volume")

    buy_ratio = buy_vol / total_vol

    if buy_ratio >= params.flow_imbalance_threshold:
        strength = min(1.0, (buy_ratio - params.flow_imbalance_threshold) /
                       (1.0 - params.flow_imbalance_threshold))
        return MomentumSignal(
            "flow", 1, strength, buy_ratio, params.flow_imbalance_threshold,
            f"buy flow {buy_ratio:.0%} over {window}min"
        )

    if (1 - buy_ratio) >= params.flow_imbalance_threshold:
        sell_ratio = 1 - buy_ratio
        strength = min(1.0, (sell_ratio - params.flow_imbalance_threshold) /
                       (1.0 - params.flow_imbalance_threshold))
        return MomentumSignal(
            "flow", -1, strength, buy_ratio, params.flow_imbalance_threshold,
            f"sell flow {sell_ratio:.0%} over {window}min"
        )

    return MomentumSignal("flow", 0, 0.0, buy_ratio,
                          params.flow_imbalance_threshold,
                          f"balanced flow {buy_ratio:.0%} buy")


# ═════════════════════════════════════════════════════════════════════════════
# DETECTOR
# ═════════════════════════════════════════════════════════════════════════════

class MomentumDetector:
    """
    Real-time momentum signal detector.

    Pulls 1-minute candle data, computes multiple indicators,
    and produces composite signals with confidence scores.
    """

    def __init__(self, params: SignalParams | None = None,
                 exchange: str = "binance"):
        self.params = params or SignalParams()
        self.exchange_name = exchange
        self._exchange = None
        self.candle_cache: dict[str, pd.DataFrame] = {}

    def _get_exchange(self):
        """Lazy-init exchange connection."""
        if self._exchange is None:
            import ccxt
            exchange_class = getattr(ccxt, self.exchange_name)
            self._exchange = exchange_class({"enableRateLimit": True})
        return self._exchange

    def load_candles(self, symbol: str,
                     lookback_minutes: int = 120,
                     timeframe: str = "1m",
                     drop_incomplete: bool = True) -> pd.DataFrame | None:
        """
        Fetch recent candles from exchange.

        Args:
            symbol: Trading pair (e.g. "BTC/USDT")
            lookback_minutes: How many minutes of history to fetch
            timeframe: Candle timeframe (default: 1 minute)
            drop_incomplete: Drop the last candle (current incomplete).
                             Set False for backtest where all candles are complete.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        exchange = self._get_exchange()

        try:
            since = int((time.time() - lookback_minutes * 60) * 1000)
            # Request 1 extra to compensate for dropping incomplete
            limit = lookback_minutes + (1 if drop_incomplete else 0)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since,
                                         limit=limit)

            if not ohlcv:
                logger.warning(f"No candles returned for {symbol}")
                return None

            df = pd.DataFrame(ohlcv,
                              columns=["timestamp", "open", "high", "low",
                                       "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Drop the last candle — it's the current incomplete candle
            # from the exchange. Its volume is near-zero because the
            # minute just started, which corrupts volume-based signals.
            if drop_incomplete and len(df) > 1:
                df = df.iloc[:-1].reset_index(drop=True)

            self.candle_cache[symbol] = df
            return df

        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol}: {e}")
            return None

    def compute_signals(self, symbol: str) -> CompositeSignal | None:
        """
        Compute all signals for an asset and produce a composite score.

        Returns CompositeSignal with direction, confidence, and individual
        signal details.
        """
        df = self.candle_cache.get(symbol)
        if df is None or len(df) < 20:
            return None

        # Compute individual signals
        signals = [
            compute_volume_spike(df, self.params),
            compute_velocity(df, self.params),
            compute_consecutive(df, self.params),
            compute_breakout(df, self.params),
            compute_flow_imbalance(df, self.params),
        ]

        # Compute weighted composite score
        weights = self.params.weights
        total_weight = 0
        weighted_score = 0

        for sig in signals:
            w = weights.get(sig.name, 0.0)
            # Score = direction * strength * weight
            weighted_score += sig.direction * sig.strength * w
            total_weight += w

        # Normalize to [-1, 1]
        if total_weight > 0:
            composite_score = weighted_score / total_weight
        else:
            composite_score = 0.0

        # Direction: requires a minimum score
        if composite_score > 0.2:
            direction = 1
        elif composite_score < -0.2:
            direction = -1
        else:
            direction = 0

        # Context data
        current_price = float(df["close"].iloc[-1])
        price_5m = float(df["close"].iloc[-6]) if len(df) > 5 else current_price
        price_15m = float(df["close"].iloc[-16]) if len(df) > 15 else current_price

        avg_vol = df["volume"].iloc[-31:-1].mean() if len(df) > 30 else df["volume"].mean()
        current_vol = float(df["volume"].iloc[-1])
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0

        return CompositeSignal(
            timestamp=datetime.now(timezone.utc),
            asset=symbol,
            direction=direction,
            composite_score=composite_score,
            confidence=abs(composite_score),
            signals=signals,
            current_price=current_price,
            price_5m_ago=price_5m,
            price_15m_ago=price_15m,
            vol_ratio=vol_ratio,
        )

    def scan_assets(self, symbols: list[str],
                    lookback_minutes: int = 120) -> list[CompositeSignal]:
        """
        Scan multiple assets and return signals sorted by confidence.
        """
        results = []
        for symbol in symbols:
            df = self.load_candles(symbol, lookback_minutes)
            if df is None:
                continue
            sig = self.compute_signals(symbol)
            if sig is not None:
                results.append(sig)

        # Sort by absolute score (highest confidence first)
        results.sort(key=lambda s: abs(s.composite_score), reverse=True)
        return results

    def monitor_loop(self, symbols: list[str],
                     interval_seconds: float = 60.0,
                     duration_minutes: float = 60.0,
                     lookback_minutes: int = 120):
        """
        Continuous monitoring loop — print signals as they appear.

        This is the paper-trading mode: watch signals in real time,
        track what would have happened.
        """
        end_time = time.time() + duration_minutes * 60
        scan_count = 0
        trade_log: list[dict] = []

        print(f"\n{'='*80}")
        print(f"MOMENTUM MONITOR — Live Mode")
        print(f"  Assets: {', '.join(symbols)}")
        print(f"  Interval: {interval_seconds}s")
        print(f"  Duration: {duration_minutes} min")
        print(f"  Min score: {self.params.min_composite_score}")
        print(f"  TP/SL: {self.params.take_profit_bps}bps / {self.params.stop_loss_bps}bps")
        print(f"{'='*80}\n")

        # Track open paper trades
        open_trades: dict[str, dict] = {}

        try:
            while time.time() < end_time:
                t0 = time.time()
                scan_count += 1
                now = datetime.now(timezone.utc)

                signals = self.scan_assets(symbols, lookback_minutes)

                for sig in signals:
                    # Check open trades for exit conditions
                    if sig.asset in open_trades:
                        trade = open_trades[sig.asset]
                        entry_price = trade["entry_price"]
                        direction = trade["direction"]
                        pnl_bps = ((sig.current_price - entry_price) / entry_price
                                   * 10000 * direction)
                        hold_mins = (time.time() - trade["entry_time"]) / 60

                        # Check exit conditions
                        exit_reason = None
                        if pnl_bps >= self.params.take_profit_bps:
                            exit_reason = "TP"
                        elif pnl_bps <= -self.params.stop_loss_bps:
                            exit_reason = "SL"
                        elif hold_mins >= self.params.max_hold_minutes:
                            exit_reason = "TIMEOUT"
                        elif (sig.direction != 0
                              and sig.direction != direction
                              and sig.confidence >= 0.5):
                            exit_reason = "REVERSAL"

                        if exit_reason:
                            dir_str = "LONG" if direction > 0 else "SHORT"
                            icon = "✅" if pnl_bps > 0 else "❌"
                            print(f"  [{now.strftime('%H:%M:%S')}] EXIT "
                                  f"{sig.asset} {dir_str} "
                                  f"pnl={pnl_bps:+.1f}bps "
                                  f"hold={hold_mins:.0f}min "
                                  f"reason={exit_reason} {icon}")
                            trade_log.append({
                                "asset": sig.asset,
                                "direction": dir_str,
                                "entry_price": entry_price,
                                "exit_price": sig.current_price,
                                "pnl_bps": pnl_bps,
                                "hold_minutes": hold_mins,
                                "exit_reason": exit_reason,
                            })
                            del open_trades[sig.asset]
                        continue

                    # Check for new entry
                    if sig.is_actionable and sig.asset not in open_trades:
                        dir_str = "LONG" if sig.direction > 0 else "SHORT"
                        print(f"  [{now.strftime('%H:%M:%S')}] ENTRY "
                              f"{sig.asset} {dir_str} "
                              f"score={sig.composite_score:+.2f} "
                              f"price=${sig.current_price:,.2f} "
                              f"vol={sig.vol_ratio:.1f}x")

                        # Print individual signals
                        for s in sig.signals:
                            if s.direction != 0:
                                print(f"    {s.name}: {s.description}")

                        open_trades[sig.asset] = {
                            "direction": sig.direction,
                            "entry_price": sig.current_price,
                            "entry_time": time.time(),
                            "entry_score": sig.composite_score,
                        }

                # Status update every 10 scans
                if scan_count % 10 == 0:
                    open_str = (", ".join(f"{k}({'LONG' if v['direction']>0 else 'SHORT'})"
                                         for k, v in open_trades.items())
                                or "none")
                    print(f"  --- Scan #{scan_count}: {len(trade_log)} completed trades, "
                          f"open: {open_str} ---")

                elapsed = time.time() - t0
                wait = max(0, interval_seconds - elapsed)
                if wait > 0:
                    time.sleep(wait)

        except KeyboardInterrupt:
            print("\n  Monitor stopped by user.")

        # Close any remaining open trades at last price
        for asset, trade in open_trades.items():
            df = self.candle_cache.get(asset)
            if df is not None:
                last_price = float(df["close"].iloc[-1])
                pnl_bps = ((last_price - trade["entry_price"]) / trade["entry_price"]
                           * 10000 * trade["direction"])
                hold_mins = (time.time() - trade["entry_time"]) / 60
                trade_log.append({
                    "asset": asset,
                    "direction": "LONG" if trade["direction"] > 0 else "SHORT",
                    "entry_price": trade["entry_price"],
                    "exit_price": last_price,
                    "pnl_bps": pnl_bps,
                    "hold_minutes": hold_mins,
                    "exit_reason": "SESSION_END",
                })

        # Summary
        _print_trade_summary(trade_log)
        return trade_log


def _print_trade_summary(trade_log: list[dict]):
    """Print paper trading summary statistics."""
    print(f"\n{'='*80}")
    print("PAPER TRADING SUMMARY")
    print(f"{'='*80}")

    if not trade_log:
        print("  No trades executed.")
        return

    df = pd.DataFrame(trade_log)
    n = len(df)
    wins = (df["pnl_bps"] > 0).sum()
    losses = (df["pnl_bps"] <= 0).sum()
    wr = wins / n * 100 if n > 0 else 0

    print(f"  Total trades:  {n}")
    print(f"  Win rate:      {wr:.0f}% ({wins}W / {losses}L)")
    print(f"  Avg P&L:       {df['pnl_bps'].mean():+.1f} bps")
    print(f"  Median P&L:    {df['pnl_bps'].median():+.1f} bps")
    print(f"  Best trade:    {df['pnl_bps'].max():+.1f} bps")
    print(f"  Worst trade:   {df['pnl_bps'].min():+.1f} bps")
    print(f"  Total P&L:     {df['pnl_bps'].sum():+.1f} bps")
    print(f"  Avg hold:      {df['hold_minutes'].mean():.1f} min")

    print(f"\n  By exit reason:")
    for reason, grp in df.groupby("exit_reason"):
        print(f"    {reason}: {len(grp)} trades, "
              f"avg={grp['pnl_bps'].mean():+.1f}bps")

    print(f"\n  By asset:")
    for asset, grp in df.groupby("asset"):
        print(f"    {asset}: {len(grp)} trades, "
              f"WR={( grp['pnl_bps'] > 0).mean():.0%}, "
              f"total={grp['pnl_bps'].sum():+.1f}bps")

    print(f"\n  At 3x leverage, total P&L would be: "
          f"{df['pnl_bps'].sum() * 3:+.1f} bps on capital")
    print(f"{'='*80}")


# ═════════════════════════════════════════════════════════════════════════════
# BACKTESTER — validate signals on historical data
# ═════════════════════════════════════════════════════════════════════════════

def backtest_signals(
    symbol: str,
    lookback_hours: int = 24,
    params: SignalParams | None = None,
    exchange: str = "binance",
) -> list[dict]:
    """
    Backtest momentum signals on recent historical data.

    Walks through the data minute by minute, computing signals at each
    point using only past data (no look-ahead), and simulates trades
    with TP/SL/timeout exits.

    Args:
        symbol: Trading pair (e.g. "BTC/USDT")
        lookback_hours: Hours of history to test on
        params: Signal parameters (uses defaults if None)
        exchange: Exchange for data

    Returns:
        List of trade dicts with P&L
    """
    params = params or SignalParams()
    detector = MomentumDetector(params, exchange)

    # Fetch extended history — paginate if needed (Binance max 1000/call)
    total_minutes = lookback_hours * 60
    warmup = 120  # need 120 minutes warmup for indicators
    needed = total_minutes + warmup

    ex = detector._get_exchange()
    all_candles = []
    fetch_since = int((time.time() - needed * 60) * 1000)
    batch_size = 1000

    while True:
        try:
            batch = ex.fetch_ohlcv(symbol, "1m", since=fetch_since,
                                   limit=batch_size)
        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol}: {e}")
            break

        if not batch:
            break

        all_candles.extend(batch)
        # Move cursor past last candle
        fetch_since = batch[-1][0] + 60_000  # +1 minute in ms

        if len(batch) < batch_size:
            break  # no more data

        time.sleep(0.5)  # rate limit

    if len(all_candles) < warmup + 60:
        print(f"  Insufficient data for {symbol}: got {len(all_candles)} candles")
        return []

    df = pd.DataFrame(all_candles,
                      columns=["timestamp", "open", "high", "low",
                               "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    # Drop last candle (likely incomplete)
    if len(df) > 1:
        df = df.iloc[:-1].reset_index(drop=True)

    print(f"\n  Backtesting {symbol}: {len(df)} candles ({lookback_hours}h + warmup)")
    print(f"  Params: TP={params.take_profit_bps}bps, SL={params.stop_loss_bps}bps, "
          f"timeout={params.max_hold_minutes}min, min_score={params.min_composite_score}")

    trade_log = []
    open_trade = None

    # Walk forward through data, starting after warmup
    for i in range(warmup, len(df)):
        window = df.iloc[:i + 1].copy()
        detector.candle_cache[symbol] = window
        sig = detector.compute_signals(symbol)

        if sig is None:
            continue

        current_price = sig.current_price
        current_time = df["timestamp"].iloc[i]

        # Check open trade
        if open_trade is not None:
            pnl_bps = ((current_price - open_trade["entry_price"])
                       / open_trade["entry_price"] * 10000 * open_trade["direction"])
            bars_held = i - open_trade["entry_bar"]

            exit_reason = None
            if pnl_bps >= params.take_profit_bps:
                exit_reason = "TP"
            elif pnl_bps <= -params.stop_loss_bps:
                exit_reason = "SL"
            elif bars_held >= params.max_hold_minutes:
                exit_reason = "TIMEOUT"
            elif (sig.direction != 0
                  and sig.direction != open_trade["direction"]
                  and sig.confidence >= 0.5):
                exit_reason = "REVERSAL"

            if exit_reason:
                trade_log.append({
                    "asset": symbol,
                    "direction": "LONG" if open_trade["direction"] > 0 else "SHORT",
                    "entry_price": open_trade["entry_price"],
                    "exit_price": current_price,
                    "entry_time": str(open_trade["entry_time"]),
                    "exit_time": str(current_time),
                    "pnl_bps": pnl_bps,
                    "hold_minutes": bars_held,
                    "exit_reason": exit_reason,
                    "entry_score": open_trade["entry_score"],
                })
                open_trade = None
            continue

        # Check for new entry
        if sig.is_actionable and open_trade is None:
            open_trade = {
                "direction": sig.direction,
                "entry_price": current_price,
                "entry_time": current_time,
                "entry_bar": i,
                "entry_score": sig.composite_score,
            }

    _print_trade_summary(trade_log)
    return trade_log
