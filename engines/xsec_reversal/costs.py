"""
engines/xsec_reversal/costs.py

Transaction-cost model for the tiered cross-sectional reversal test.

WHY THIS IS THE LOAD-BEARING MODULE
-----------------------------------
The hypothesis is that gross alpha rises as you go down the liquidity tiers --
but so does the spread. The whole experiment is a race between those two
curves, so applying a FLAT cost assumption across tiers would either
manufacture the result (cost too low in the illiquid tier) or destroy it (cost
too high in the liquid tier). Costs MUST vary by symbol and period.

HONEST LIMITATION (pre-registered)
----------------------------------
Binance kline archives contain OHLCV only -- NO quotes. So unlike the Chan/GLD
replication (where per-bar bid/ask was purchased), spreads here are ESTIMATED
from high/low/close, not measured. We use two standard estimators from the
microstructure literature and require agreement between them as a robustness
condition:

  * Corwin & Schultz (2012): effective spread from two consecutive high-low
    ranges. The insight is that the high-low RANGE reflects both volatility
    (which scales with time) and the spread (which does not), so comparing
    single-period ranges with the two-period range separates them.

  * Abdi & Ranaldo (2017): uses the gap between the close and the mid-range
    of the high-low band across consecutive periods. Generally more robust in
    illiquid samples, which is exactly the tier we care about most.

Both estimate the EFFECTIVE (round-trip) proportional spread S. A one-way
crossing therefore costs S/2, plus fees, plus any assumed extra slippage.

Both estimators are noisy per-observation and are meant to be AVERAGED over a
window; single-bar values are not meaningful. Negative estimates (a known
artifact) are floored at zero before averaging, per the original papers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("xsec.costs")

_K = 3.0 - 2.0 * np.sqrt(2.0)  # Corwin-Schultz constant


@dataclass(frozen=True)
class CostSpec:
    """All cost assumptions. Everything a parameter; nothing hard-coded.

    spread_model:
        'corwin_schultz' | 'abdi_ranaldo' | 'fixed' | 'max_of_estimators'
    fee_bps_per_side: exchange taker fee, one way (Binance spot taker ~ 10 bps
        at base tier, ~ 7.5 with BNB; set explicitly per pre-registration).
    extra_slippage_bps_per_side: conservatism knob for market impact beyond the
        quoted spread. Default 0 so the base case is spread+fee only.
    fixed_spread_bps: used when spread_model == 'fixed'.
    spread_window_bars: averaging window for the estimators.
    min_spread_bps / max_spread_bps: sanity clamps on estimated spreads.
    """
    spread_model: str = "max_of_estimators"
    fee_bps_per_side: float = 10.0
    extra_slippage_bps_per_side: float = 0.0
    fixed_spread_bps: float = 10.0
    spread_window_bars: int = 180
    min_spread_bps: float = 0.5
    max_spread_bps: float = 500.0


def corwin_schultz_spread(high: pd.Series, low: pd.Series,
                          window: int = 180) -> pd.Series:
    """Rolling Corwin-Schultz effective proportional spread (as a fraction).

    Returns a series aligned to `high`/`low`, causal (uses only trailing data).
    """
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    valid = (h > 0) & (l > 0)
    h = h.where(valid)
    l = l.where(valid)

    log_hl = np.log(h / l)
    beta = (log_hl ** 2) + (log_hl ** 2).shift(1)

    h2 = pd.concat([h, h.shift(1)], axis=1).max(axis=1)
    l2 = pd.concat([l, l.shift(1)], axis=1).min(axis=1)
    gamma = np.log(h2 / l2) ** 2

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.clip(lower=0.0)  # negative estimates -> 0 (per the paper)
    return s.rolling(window, min_periods=max(5, window // 6)).mean()


def abdi_ranaldo_spread(high: pd.Series, low: pd.Series, close: pd.Series,
                        window: int = 180) -> pd.Series:
    """Rolling Abdi-Ranaldo (2017) effective proportional spread (fraction)."""
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    valid = (h > 0) & (l > 0) & (c > 0)
    h, l, c = h.where(valid), l.where(valid), c.where(valid)

    eta = (np.log(h) + np.log(l)) / 2.0
    lc = np.log(c)
    prod = (lc - eta) * (lc - eta.shift(-1))
    prod = prod.replace([np.inf, -np.inf], np.nan)
    # Causal average: shift by 1 so bar t only uses information through t.
    mean_prod = prod.shift(1).rolling(window,
                                      min_periods=max(5, window // 6)).mean()
    s2 = (4.0 * mean_prod).clip(lower=0.0)
    return np.sqrt(s2)


def estimate_spread_bps(
    panel_high: pd.DataFrame,
    panel_low: pd.DataFrame,
    panel_close: pd.DataFrame,
    spec: CostSpec = CostSpec(),
) -> pd.DataFrame:
    """Per-symbol, per-bar estimated EFFECTIVE spread in basis points.

    Returns a wide [time x symbol] panel. Causal by construction.
    """
    if spec.spread_model == "fixed":
        out = pd.DataFrame(spec.fixed_spread_bps,
                           index=panel_close.index, columns=panel_close.columns)
        return out.where(panel_close.notna())

    cs_frames, ar_frames = {}, {}
    for sym in panel_close.columns:
        if spec.spread_model in ("corwin_schultz", "max_of_estimators"):
            cs_frames[sym] = corwin_schultz_spread(
                panel_high[sym], panel_low[sym], spec.spread_window_bars)
        if spec.spread_model in ("abdi_ranaldo", "max_of_estimators"):
            ar_frames[sym] = abdi_ranaldo_spread(
                panel_high[sym], panel_low[sym], panel_close[sym],
                spec.spread_window_bars)

    if spec.spread_model == "corwin_schultz":
        est = pd.DataFrame(cs_frames)
    elif spec.spread_model == "abdi_ranaldo":
        est = pd.DataFrame(ar_frames)
    elif spec.spread_model == "max_of_estimators":
        cs = pd.DataFrame(cs_frames)
        ar = pd.DataFrame(ar_frames)
        # Conservative: take the larger of the two estimates where both exist.
        est = pd.concat([cs, ar]).groupby(level=0).max()
        est = est.reindex(index=panel_close.index, columns=panel_close.columns)
    else:
        raise ValueError(f"unknown spread_model '{spec.spread_model}'")

    bps = (est * 1e4).clip(lower=spec.min_spread_bps, upper=spec.max_spread_bps)
    bps = bps.where(panel_close.notna())
    logger.info("estimate_spread_bps[%s]: median=%.1f bps, p95=%.1f bps",
                spec.spread_model,
                float(np.nanmedian(bps.values)) if bps.notna().any().any() else float("nan"),
                float(np.nanpercentile(bps.values[~np.isnan(bps.values)], 95))
                if bps.notna().any().any() else float("nan"))
    return bps


def one_way_cost_bps(spread_bps: pd.DataFrame, spec: CostSpec) -> pd.DataFrame:
    """Total one-way cost in bps: half the effective spread + fee + slippage."""
    return spread_bps / 2.0 + spec.fee_bps_per_side + spec.extra_slippage_bps_per_side
