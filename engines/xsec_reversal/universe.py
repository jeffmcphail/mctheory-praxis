"""
engines/xsec_reversal/universe.py

Point-in-time, survivorship-bias-free universe construction with causal
liquidity tiering.

THE TWO BIASES THIS MODULE EXISTS TO KILL
-----------------------------------------
1. SURVIVORSHIP. Handled upstream in archive.py (enumerate from the S3 bucket,
   not the API) AND here: a symbol is in the universe at time t iff it actually
   had trading data in the trailing window ending at t. Symbols that later got
   delisted remain in the universe for every period they DID trade, and simply
   drop out when their data stops -- which is what really happened to a holder.

2. LOOK-AHEAD IN TIER ASSIGNMENT. Liquidity tiers are assigned from TRAILING
   dollar ADV computed strictly on data at or before the formation timestamp.
   Ranking on full-sample ADV would leak the future (a coin that later became
   liquid would be pre-promoted into the liquid tier), and because the
   hypothesis is *specifically about tier membership*, that leak would corrupt
   the headline result rather than a side statistic.

EXCLUSIONS (all parameters, none hard-coded)
   - stablecoin pairs (no cross-sectional dispersion to trade)
   - leveraged tokens (UP/DOWN/BULL/BEAR): rebalancing mechanics create
     artificial mean reversion that is a product artifact, not a market effect.
     Leaving these in would be a fake-positive generator.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("xsec.universe")

# Substrings marking Binance leveraged tokens (artificial mean reversion).
DEFAULT_LEVERAGED_PATTERNS = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
# Stable/pegged bases -- no dispersion, pure noise in a reversal ranking.
DEFAULT_STABLE_BASES = (
    "BUSD", "USDC", "TUSD", "USDP", "DAI", "FDUSD", "SUSD", "EUR", "GBP",
    "AUD", "USDS", "USDSB", "PAX", "UST", "USTC",
)


@dataclass(frozen=True)
class TierSpec:
    """Liquidity tiering by trailing dollar ADV.

    n_tiers: number of equal-count tiers (1 = liquid ... n = illiquid), OR
    explicit adv_breakpoints_usd (e.g. (1e8, 1e7, 1e6)) for fixed cuts.
    """
    n_tiers: int = 3
    adv_breakpoints_usd: Optional[tuple] = None
    tier_labels: Optional[tuple] = None

    def labels(self) -> tuple:
        if self.tier_labels:
            return self.tier_labels
        n = len(self.adv_breakpoints_usd) + 1 if self.adv_breakpoints_usd else self.n_tiers
        if n == 3:
            return ("T1_liquid", "T2_mid", "T3_illiquid")
        return tuple(f"T{i+1}" for i in range(n))


@dataclass(frozen=True)
class UniverseSpec:
    """Point-in-time universe filters. Everything is a parameter."""
    quote: str = "USDT"
    adv_lookback_bars: int = 180          # trailing bars for ADV + liveness
    min_adv_usd: float = 50_000.0         # floor: below this it is not tradeable
    min_history_bars: int = 200           # need this much trailing data to enter
    max_symbols: Optional[int] = None     # cap universe size (by ADV) if set
    exclude_leveraged: bool = True
    exclude_stables: bool = True
    leveraged_patterns: tuple = DEFAULT_LEVERAGED_PATTERNS
    stable_bases: tuple = DEFAULT_STABLE_BASES
    require_full_window: bool = False     # if True, symbol must have EVERY bar


def filter_symbol_names(symbols, spec: UniverseSpec = UniverseSpec()) -> list[str]:
    """Apply name-based exclusions (leveraged tokens, stablecoin pairs)."""
    out = []
    for s in symbols:
        su = s.upper()
        if spec.quote and not su.endswith(spec.quote):
            continue
        if spec.exclude_leveraged and any(p in su for p in spec.leveraged_patterns):
            continue
        if spec.exclude_stables:
            base = su[: -len(spec.quote)] if spec.quote else su
            if base in spec.stable_bases:
                continue
        out.append(s)
    dropped = len(list(symbols)) - len(out)
    logger.info("filter_symbol_names: kept %d, dropped %d (leveraged/stable/quote)",
                len(out), dropped)
    return out


def build_point_in_time_universe(
    panel_close: pd.DataFrame,
    panel_dollar_vol: pd.DataFrame,
    rebalance_index: pd.DatetimeIndex,
    spec: UniverseSpec = UniverseSpec(),
) -> pd.DataFrame:
    """Determine universe membership and trailing ADV at each rebalance time.

    Parameters
    ----------
    panel_close : wide DataFrame [time x symbol] of close prices (NaN where the
        symbol was not trading -- that NaN pattern IS the listing history).
    panel_dollar_vol : wide DataFrame [time x symbol] of per-bar quote-asset
        (USD) volume, same shape/index.
    rebalance_index : timestamps at which the universe is re-formed.

    Returns
    -------
    Long DataFrame with columns [dt, symbol, adv_usd, n_obs, eligible].
    Computed strictly from data at or BEFORE each dt (causal).
    """
    if not panel_close.index.equals(panel_dollar_vol.index):
        raise ValueError("close and dollar-volume panels must share an index")
    if list(panel_close.columns) != list(panel_dollar_vol.columns):
        raise ValueError("close and dollar-volume panels must share columns")

    rows = []
    lb = spec.adv_lookback_bars
    for dt in rebalance_index:
        # STRICTLY causal: only bars at or before dt.
        hist_c = panel_close.loc[:dt].tail(lb)
        hist_v = panel_dollar_vol.loc[:dt].tail(lb)
        if hist_c.empty:
            continue

        n_obs = hist_c.notna().sum()
        adv = hist_v.mean(skipna=True)
        # Must be trading AT dt (not just sometime in the window).
        live_now = panel_close.loc[dt].notna() if dt in panel_close.index else \
            pd.Series(False, index=panel_close.columns)

        eligible = (
            live_now
            & (n_obs >= spec.min_history_bars)
            & (adv >= spec.min_adv_usd)
            & adv.notna()
        )
        if spec.require_full_window:
            eligible &= (n_obs >= lb)

        sub = pd.DataFrame({
            "dt": dt,
            "symbol": panel_close.columns,
            "adv_usd": adv.values,
            "n_obs": n_obs.values,
            "eligible": eligible.values,
        })

        if spec.max_symbols:
            elig = sub[sub["eligible"]].nlargest(spec.max_symbols, "adv_usd")
            sub["eligible"] = sub["symbol"].isin(set(elig["symbol"]))

        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=["dt", "symbol", "adv_usd", "n_obs", "eligible"])
    out = pd.concat(rows, ignore_index=True)
    n_elig = out.groupby("dt")["eligible"].sum()
    logger.info(
        "point-in-time universe: %d rebalances, eligible/rebalance min=%d med=%d max=%d",
        len(rebalance_index), int(n_elig.min()), int(n_elig.median()), int(n_elig.max()),
    )
    return out


def assign_tiers(universe: pd.DataFrame, tier: TierSpec = TierSpec()) -> pd.DataFrame:
    """Assign a liquidity tier per (dt, symbol) from trailing ADV.

    Tiers are formed WITHIN each rebalance date across eligible symbols only, so
    membership adapts as the market's liquidity distribution shifts. Tier 1 is
    the most liquid.
    """
    labels = tier.labels()
    out = universe.copy()
    out["tier"] = pd.Series([pd.NA] * len(out), dtype="object")

    for dt, grp in out.groupby("dt"):
        elig = grp[grp["eligible"]]
        if elig.empty:
            continue
        adv = elig["adv_usd"]

        if tier.adv_breakpoints_usd:
            bps = sorted(tier.adv_breakpoints_usd, reverse=True)
            t_idx = np.zeros(len(elig), dtype=int)
            for i, bp in enumerate(bps):
                t_idx = np.where(adv.values < bp, i + 1, t_idx)
        else:
            n = tier.n_tiers
            if len(elig) < n:
                logger.warning("dt=%s only %d eligible symbols for %d tiers; "
                               "assigning all to T1", dt, len(elig), n)
                t_idx = np.zeros(len(elig), dtype=int)
            else:
                # rank 0 = most liquid
                ranks = adv.rank(ascending=False, method="first").values - 1
                t_idx = np.floor(ranks / (len(elig) / n)).astype(int)
                t_idx = np.clip(t_idx, 0, n - 1)

        out.loc[elig.index, "tier"] = [labels[i] for i in t_idx]

    counts = out[out["eligible"]].groupby("tier").size().to_dict()
    logger.info("assign_tiers: symbol-periods per tier: %s", counts)
    return out


def panels_from_long(
    df: pd.DataFrame,
    value_col: str,
    time_col: str = "dt",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Pivot a long frame into a wide [time x symbol] panel."""
    wide = df.pivot_table(index=time_col, columns=symbol_col,
                          values=value_col, aggfunc="last")
    return wide.sort_index()
