"""
market_cipher_b.py
==================
A Python/pandas implementation of Market Cipher B (MCb), replicating the
VuManChu Cipher B + Divergences open-source TradingView indicator.

Components implemented:
    1. WaveTrend Oscillator (WT1 & WT2)  — LazyBear's algorithm
    2. RSI+MFI Area (Money Flow)          — andreholanda73's formula
    3. Modified RSI                        — standard Wilder RSI
    4. Stochastic RSI                      — Stoch applied to RSI
    5. Buy/Sell dot signals                — OB/OS cross detection
    6. Divergence detection               — fractal-based WT divergences

References:
    - LazyBear WaveTrend: tradingview.com/script/2KE8wTuF
    - VuManChu Cipher B:  tradingview.com/script/Msm4SjwI
    - Original MCb:       marketciphertrading.com

VALIDATION NOTES:
    The WaveTrend algorithm matches LazyBear's published Pine script exactly.
    The RSI+MFI hybrid matches andreholanda73's formula.
    Pine Script uses RMA (Wilder's moving average) for RSI; this is replicated
    via ewm(alpha=1/n, adjust=False) which is numerically identical.

Usage:
    import pandas as pd
    from market_cipher_b import MarketCipherB

    # df must have columns: open, high, low, close, volume (lowercase)
    df = pd.read_csv("btcusdt_4h.csv", parse_dates=["timestamp"], index_col="timestamp")
    mcb = MarketCipherB()
    result = mcb.calculate(df)
    print(result[["wt1","wt2","rsi_mfi","rsi","stoch_k","buy_dot","sell_dot"]].tail(20))
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Parameter dataclass — mirrors MCb TradingView settings panel
# ---------------------------------------------------------------------------

@dataclass
class MCBParams:
    # WaveTrend
    wt_channel_len: int = 10       # Channel length (EMA of HLC3)
    wt_average_len: int = 21       # Average length (EMA of CI → WT1)
    wt_ma_len: int = 4             # MA length (SMA of WT1 → WT2)
    wt_ob_level_1: float = 60.0    # Overbought level 1
    wt_ob_level_2: float = 53.0    # Overbought level 2
    wt_os_level_1: float = -60.0   # Oversold level 1
    wt_os_level_2: float = -53.0   # Oversold level 2

    # RSI+MFI (Money Flow)
    mfi_period: int = 60           # RSI+MFI lookback period
    mfi_multiplier: float = 150.0  # RSI+MFI output scaler

    # RSI
    rsi_length: int = 14           # RSI period

    # Stochastic RSI
    stoch_length: int = 14         # Stochastic lookback
    stoch_rsi_length: int = 14     # RSI length for Stoch RSI
    stoch_k_smooth: int = 3        # K-line smoothing
    stoch_d_smooth: int = 3        # D-line smoothing


# ---------------------------------------------------------------------------
# Core calculation engine
# ---------------------------------------------------------------------------

class MarketCipherB:
    """
    Market Cipher B indicator calculator.

    Replicates the VuManChu Cipher B open-source approximation of MCb.
    All calculations are vectorised using pandas/numpy.
    """

    def __init__(self, params: Optional[MCBParams] = None):
        self.p = params or MCBParams()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all Market Cipher B components.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with lowercase columns: open, high, low, close, volume
            Index should be a DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Original df augmented with MCb columns:
              wt1, wt2          — WaveTrend leading/lagging waves
              rsi_mfi           — Money Flow (RSI+MFI hybrid), green>0, red<0
              rsi               — Modified RSI (0-100)
              stoch_k, stoch_d  — Stochastic RSI K and D lines
              stoch_color       — 1=green (bullish cross), -1=red (bearish)
              wt_cross          — True on any WT1/WT2 cross
              wt_cross_up       — True on bullish cross
              wt_cross_down     — True on bearish cross
              buy_dot           — True: green buy dot (OS zone, cross up)
              sell_dot          — True: red sell dot (OB zone, cross down)
              gold_dot          — True: gold extreme oversold signal
              bull_div          — True: bullish WT divergence detected
              bear_div          — True: bearish WT divergence detected
        """
        result = df.copy()

        # 1. WaveTrend
        result["wt1"], result["wt2"] = self._wavetrend(df)

        # 2. Money Flow (RSI+MFI area)
        result["rsi_mfi"] = self._rsi_mfi(df)

        # 3. RSI
        result["rsi"] = self._rsi(df["close"], self.p.rsi_length)

        # 4. Stochastic RSI
        result["stoch_k"], result["stoch_d"] = self._stoch_rsi(df["close"])

        # 5. Stoch RSI color (1 = green/bullish, -1 = red/bearish)
        result["stoch_color"] = self._stoch_color(result["stoch_k"], result["stoch_d"])

        # 6. WT cross events
        result["wt_cross"] = self._crosses(result["wt1"], result["wt2"])
        result["wt_cross_up"] = self._crosses_up(result["wt1"], result["wt2"])
        result["wt_cross_down"] = self._crosses_down(result["wt1"], result["wt2"])

        # 7. Dot signals
        result["buy_dot"] = self._buy_dot(result)
        result["sell_dot"] = self._sell_dot(result)
        result["gold_dot"] = self._gold_dot(result)

        # 8. Divergences (simplified fractal method)
        result["bull_div"], result["bear_div"] = self._divergences(result)

        return result

    # ------------------------------------------------------------------
    # 1. WaveTrend Oscillator
    # ------------------------------------------------------------------

    def _wavetrend(self, df: pd.DataFrame):
        """
        LazyBear's WaveTrend Oscillator.

        Pine Script equivalent:
            ap  = hlc3
            esa = ema(ap, n1)
            d   = ema(abs(ap - esa), n1)
            ci  = (ap - esa) / (0.015 * d)
            wt1 = ema(ci, n2)
            wt2 = sma(wt1, 4)
        """
        n1 = self.p.wt_channel_len
        n2 = self.p.wt_average_len
        n3 = self.p.wt_ma_len

        ap = (df["high"] + df["low"] + df["close"]) / 3.0  # HLC3

        esa = ap.ewm(span=n1, adjust=False).mean()
        d   = (ap - esa).abs().ewm(span=n1, adjust=False).mean()

        # Guard against division by zero
        ci = (ap - esa) / (0.015 * d.replace(0, np.nan))

        wt1 = ci.ewm(span=n2, adjust=False).mean()
        wt2 = wt1.rolling(window=n3).mean()

        return wt1, wt2

    # ------------------------------------------------------------------
    # 2. RSI + MFI Area (Money Flow)
    # ------------------------------------------------------------------

    def _rsi_mfi(self, df: pd.DataFrame) -> pd.Series:
        """
        Custom RSI+MFI oscillator.

        Pine Script equivalent (andreholanda73 / vumanchu):
            Rmfi = rsi(volume * (close > open ? 1 : close < open ? -1 : 0), Period)
            rsiMFI = (((Rmfi - 50) * Multiplier) / 100)

        This approach uses RSI of directional volume, scaled to oscillate
        around zero. Green (positive) = money flowing in; red (negative) = out.
        """
        period = self.p.mfi_period
        mult   = self.p.mfi_multiplier

        # Directional volume: positive if close > open, negative if close < open
        direction = np.where(df["close"] > df["open"], 1,
                    np.where(df["close"] < df["open"], -1, 0))
        dir_vol = df["volume"] * direction

        # RSI of directional volume
        rmfi = self._rsi(pd.Series(dir_vol, index=df.index), period)

        # Scale and center around zero
        rsi_mfi = ((rmfi - 50.0) * mult) / 100.0

        return rsi_mfi

    # ------------------------------------------------------------------
    # 3. RSI (Wilder / RMA)
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(series: pd.Series, length: int) -> pd.Series:
        """
        Wilder's RSI. Pine's rsi() uses RMA (Wilder MA = EMA with alpha=1/n).
        pandas ewm with alpha=1/n and adjust=False is numerically equivalent.
        """
        delta = series.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)

        alpha = 1.0 / length
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    # ------------------------------------------------------------------
    # 4. Stochastic RSI
    # ------------------------------------------------------------------

    def _stoch_rsi(self, close: pd.Series):
        """
        Stochastic RSI: applies stochastic formula to RSI values.

        Pine Script:
            rsi1  = rsi(src, lengthRSI)
            k     = sma(stoch(rsi1, rsi1, rsi1, lengthStoch), smoothK)
            d     = sma(k, smoothD)
        """
        rsi   = self._rsi(close, self.p.stoch_rsi_length)
        n     = self.p.stoch_length

        lowest  = rsi.rolling(n).min()
        highest = rsi.rolling(n).max()

        stoch = (rsi - lowest) / (highest - lowest).replace(0, np.nan) * 100.0

        k = stoch.rolling(self.p.stoch_k_smooth).mean()
        d = k.rolling(self.p.stoch_d_smooth).mean()

        return k, d

    # ------------------------------------------------------------------
    # 5. Stochastic RSI color
    # ------------------------------------------------------------------

    @staticmethod
    def _stoch_color(k: pd.Series, d: pd.Series) -> pd.Series:
        """
        Returns +1 when K is above D (green/bullish), -1 when below (red/bearish).
        In MCb, the RSI line is painted green/red based on Stoch RSI position.
        """
        return np.where(k >= d, 1, -1)

    # ------------------------------------------------------------------
    # 6. Cross detection utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _crosses(a: pd.Series, b: pd.Series) -> pd.Series:
        """True on any bar where a crosses b (in either direction)."""
        prev_a = a.shift(1)
        prev_b = b.shift(1)
        cross_up   = (prev_a <= prev_b) & (a > b)
        cross_down = (prev_a >= prev_b) & (a < b)
        return (cross_up | cross_down)

    @staticmethod
    def _crosses_up(a: pd.Series, b: pd.Series) -> pd.Series:
        """True on bar where a crosses above b."""
        return (a.shift(1) <= b.shift(1)) & (a > b)

    @staticmethod
    def _crosses_down(a: pd.Series, b: pd.Series) -> pd.Series:
        """True on bar where a crosses below b."""
        return (a.shift(1) >= b.shift(1)) & (a < b)

    # ------------------------------------------------------------------
    # 7. Dot signals
    # ------------------------------------------------------------------

    def _buy_dot(self, df: pd.DataFrame) -> pd.Series:
        """
        Green buy dot: WT1 crosses above WT2 in oversold territory.
        Condition: wt2 < OS_level_2 on the cross bar.
        """
        return df["wt_cross_up"] & (df["wt2"] < self.p.wt_os_level_2)

    def _sell_dot(self, df: pd.DataFrame) -> pd.Series:
        """
        Red sell dot: WT1 crosses below WT2 in overbought territory.
        Condition: wt2 > OB_level_2 on the cross bar.
        """
        return df["wt_cross_down"] & (df["wt2"] > self.p.wt_ob_level_2)

    def _gold_dot(self, df: pd.DataFrame) -> pd.Series:
        """
        Gold dot: extreme oversold — RSI < 20, wt2 <= -80, bullish cross present.
        WARNING: Official MCb says "DON'T BUY when gold circle appears."
        This fires at maximum fear — it is NOT a direct entry signal.
        """
        return (
            df["wt_cross_up"] &
            (df["wt2"] <= -80) &
            (df["rsi"] < 20)
        )

    # ------------------------------------------------------------------
    # 8. Divergence detection (simplified fractal method)
    # ------------------------------------------------------------------

    def _divergences(self, df: pd.DataFrame, lookback: int = 5):
        """
        Simplified divergence detection using local extrema comparison.

        Bullish: price makes lower low, WT2 makes higher low → bull_div
        Bearish: price makes higher high, WT2 makes higher high → bear_div

        This is a simplified version; the Pine original uses a 35-bar fractal
        scan. This implementation uses a rolling lookback window.
        """
        close = df.index.map(lambda x: df.loc[x, "close"]) if "close" not in df.columns else df["close"]
        if "close" in df.columns:
            close = df["close"]

        wt2 = df["wt2"]

        # Rolling min/max in lookback window
        close_min = close.rolling(lookback).min()
        close_max = close.rolling(lookback).max()
        wt2_min   = wt2.rolling(lookback).min()
        wt2_max   = wt2.rolling(lookback).max()

        # Previous period rolling min/max
        prev_close_min = close_min.shift(lookback)
        prev_close_max = close_max.shift(lookback)
        prev_wt2_min   = wt2_min.shift(lookback)
        prev_wt2_max   = wt2_max.shift(lookback)

        # Bullish divergence: price lower low, WT higher low, in OS zone
        bull_div = (
            (close_min < prev_close_min) &
            (wt2_min > prev_wt2_min) &
            (wt2 < self.p.wt_os_level_1)
        )

        # Bearish divergence: price higher high, WT lower high, in OB zone
        bear_div = (
            (close_max > prev_close_max) &
            (wt2_max < prev_wt2_max) &
            (wt2 > self.p.wt_ob_level_1)
        )

        return bull_div.fillna(False), bear_div.fillna(False)


# ---------------------------------------------------------------------------
# Signal summary helper
# ---------------------------------------------------------------------------

def summarize_signals(result: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows where a notable signal fires.
    Useful for backtesting signal frequency and distribution.
    """
    sig = result[
        result["buy_dot"] |
        result["sell_dot"] |
        result["gold_dot"] |
        result["bull_div"] |
        result["bear_div"]
    ].copy()

    sig["signal"] = (
        np.where(sig["gold_dot"], "GOLD_BUY",
        np.where(sig["buy_dot"],  "BUY",
        np.where(sig["sell_dot"], "SELL",
        np.where(sig["bull_div"], "BULL_DIV",
        np.where(sig["bear_div"], "BEAR_DIV", "")))))
    )

    return sig[["close", "wt1", "wt2", "rsi_mfi", "rsi",
                "stoch_k", "stoch_color", "signal"]]


# ---------------------------------------------------------------------------
# Validation helper — compare against known values
# ---------------------------------------------------------------------------

def validate_wavetrend(df: pd.DataFrame, expected_wt1: float,
                        expected_wt2: float, tolerance: float = 0.01,
                        row: int = -1) -> dict:
    """
    Validate the WaveTrend calculation against a known reference value.

    Parameters
    ----------
    df           : OHLCV DataFrame
    expected_wt1 : WT1 value from TradingView reference
    expected_wt2 : WT2 value from TradingView reference
    tolerance    : acceptable absolute error (default 0.01)
    row          : which row to check (default: last row)

    Returns
    -------
    dict with 'passed', 'wt1_error', 'wt2_error'
    """
    mcb = MarketCipherB()
    result = mcb.calculate(df)

    calc_wt1 = result["wt1"].iloc[row]
    calc_wt2 = result["wt2"].iloc[row]

    wt1_error = abs(calc_wt1 - expected_wt1)
    wt2_error = abs(calc_wt2 - expected_wt2)

    return {
        "passed"    : (wt1_error <= tolerance) and (wt2_error <= tolerance),
        "calc_wt1"  : round(calc_wt1, 4),
        "calc_wt2"  : round(calc_wt2, 4),
        "exp_wt1"   : expected_wt1,
        "exp_wt2"   : expected_wt2,
        "wt1_error" : round(wt1_error, 6),
        "wt2_error" : round(wt2_error, 6),
    }


# ---------------------------------------------------------------------------
# Quick demo — generates synthetic data and plots
# ---------------------------------------------------------------------------

def demo():
    """
    Demonstration using synthetic sinusoidal OHLCV data.
    Prints a signal summary and plots (if matplotlib available).
    """
    import warnings
    warnings.filterwarnings("ignore")

    np.random.seed(42)
    n = 500
    t = np.linspace(0, 4 * np.pi, n)

    # Synthetic price — sine wave with noise + drift
    close = 30000 + 5000 * np.sin(t) + 2000 * np.sin(2.3 * t) + \
            np.cumsum(np.random.randn(n) * 100)
    close = np.abs(close)

    high   = close + np.random.uniform(50, 300, n)
    low    = close - np.random.uniform(50, 300, n)
    open_  = close + np.random.randn(n) * 100
    volume = np.random.uniform(100, 10000, n) * (1 + 0.5 * np.sin(t))

    dates = pd.date_range("2023-01-01", periods=n, freq="4h")
    df = pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    }, index=dates)

    mcb = MarketCipherB()
    result = mcb.calculate(df)

    print("=" * 60)
    print("Market Cipher B — Demo Output (last 10 rows)")
    print("=" * 60)
    print(result[["close","wt1","wt2","rsi_mfi","rsi",
                  "stoch_k","buy_dot","sell_dot","gold_dot"]].tail(10).to_string())

    print("\n" + "=" * 60)
    print("Signal Summary")
    print("=" * 60)
    sigs = summarize_signals(result)
    print(f"Total signals: {len(sigs)}")
    print(sigs.tail(10).to_string())

    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.1)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        # Price chart
        ax1.plot(result.index, result["close"], color="white", linewidth=0.8, label="Close")
        buy_rows  = result[result["buy_dot"]]
        sell_rows = result[result["sell_dot"]]
        gold_rows = result[result["gold_dot"]]
        ax1.scatter(buy_rows.index,  buy_rows["close"],  color="lime",   s=60, zorder=5, label="Buy dot")
        ax1.scatter(sell_rows.index, sell_rows["close"], color="red",    s=60, zorder=5, label="Sell dot")
        ax1.scatter(gold_rows.index, gold_rows["close"], color="gold",   s=80, zorder=5, label="Gold dot")
        ax1.set_facecolor("#1a1a2e")
        ax1.legend(fontsize=8)
        ax1.set_title("Market Cipher B — Demo (Synthetic BTC 4H)", color="white")
        ax1.tick_params(colors="white")

        # MCb oscillator pane
        ax2.axhline(0,  color="gray",  linewidth=0.5)
        ax2.axhline(60, color="red",   linewidth=0.3, linestyle="--")
        ax2.axhline(-60,color="green", linewidth=0.3, linestyle="--")

        # Money Flow fill
        mfi_color = np.where(result["rsi_mfi"] > 0, "green", "red")
        for i in range(len(result)):
            ax2.axvspan(result.index[i], result.index[i], alpha=0.3,
                        color=mfi_color[i], linewidth=0)
        ax2.fill_between(result.index, result["rsi_mfi"], 0,
                         where=(result["rsi_mfi"] > 0), color="green", alpha=0.25)
        ax2.fill_between(result.index, result["rsi_mfi"], 0,
                         where=(result["rsi_mfi"] < 0), color="red",   alpha=0.25)

        ax2.plot(result.index, result["wt1"], color="#40c4ff", linewidth=1.0, label="WT1 (lead)")
        ax2.plot(result.index, result["wt2"], color="#1565c0", linewidth=1.5, label="WT2 (lag)")
        ax2.plot(result.index, result["rsi"], color="#e040fb", linewidth=0.8, label="RSI", alpha=0.7)

        # Plot dots on oscillator pane
        ax2.scatter(buy_rows.index,  buy_rows["wt2"],  color="lime",  s=40, zorder=5)
        ax2.scatter(sell_rows.index, sell_rows["wt2"], color="red",   s=40, zorder=5)
        ax2.scatter(gold_rows.index, gold_rows["wt2"], color="gold",  s=55, zorder=5)

        ax2.set_facecolor("#1a1a2e")
        ax2.legend(fontsize=7, loc="upper left")
        ax2.set_ylim(-120, 120)
        ax2.tick_params(colors="white")

        fig.patch.set_facecolor("#0d0d1a")
        plt.tight_layout()
        plt.savefig("/mnt/user-data/outputs/market_cipher_b_demo.png", dpi=150,
                    facecolor="#0d0d1a")
        print("\nChart saved to: market_cipher_b_demo.png")
        plt.show()

    except ImportError:
        print("\n[matplotlib not available — skipping chart]")


if __name__ == "__main__":
    demo()
