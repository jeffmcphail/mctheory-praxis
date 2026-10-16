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
     Leaving these in would be a fake-positive generator. Identified
     STRUCTURALLY (see is_leveraged_token), not by name pattern.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("xsec.universe")

# Binance leveraged tokens are recognised STRUCTURALLY: <BASE> + one of these
# suffixes + <QUOTE>, AND <BASE><QUOTE> must itself trade. Ported from
# engines/forced_trade/leveraged.py:split_leveraged (Cycle 61), Cycle 62B T4.
#
# *** A name pattern CANNOT do this job, and neither can endswith. ***
# JUPUSDT (Jupiter) and SYRUPUSDT (Maple) both END with "UPUSDT", exactly like
# ETHUPUSDT. Any base name ending in UP/DOWN/BULL/BEAR is caught by a pattern,
# so the Cycle 60 substring rule silently deleted real spot assets from the
# study universe. What separates a leveraged token from a coincidental spelling
# is its underlying: ETHUPUSDT -> ETHUSDT trades; JUPUSDT -> JUSDT does not.
DEFAULT_LEVERAGED_SUFFIXES = ("DOWN", "BULL", "BEAR", "UP")
# Quotes tried when UniverseSpec.quote is empty (the Cycle 61 set).
DEFAULT_LEVERAGED_QUOTES = ("USDT", "BUSD", "USDC")
# LEGACY -- the Cycle 60 substring rule. NOT used for filtering any more. Kept
# only so the frozen Cycle 61 audit (forced_trade/run_audit.py t3) can still
# reproduce what this rule deleted.
DEFAULT_LEVERAGED_PATTERNS = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
# Stable/pegged bases -- no dispersion, pure noise in a reversal ranking.
# Binance tokenized EQUITIES / ETFs (symbol pattern <TICKER>B + USDT).
# These are STOCKS, not crypto: their returns are driven by equity factors and
# US market hours, so leaving them in a crypto cross-sectional study is
# contamination. They also carry regulatory-delisting events unrelated to any
# market mechanism.
#
# *** An explicit list is REQUIRED -- a naive endswith("BUSDT") rule would also
# delete BNBUSDT, SHIBUSDT, ARBUSDT, TRBUSDT, AMBUSDT, VIBUSDT and BBUSDT,
# which are real (and major) crypto assets. ***
DEFAULT_TOKENIZED_EQUITY_SYMBOLS = (
    "AAPLBUSDT", "AAOIBUSDT", "AMATBUSDT", "AMDBUSDT", "AMZNBUSDT", "ARMBUSDT",
    "AVGOBUSDT", "AXTIBUSDT", "BABABUSDT", "CBRSBUSDT", "COINBUSDT", "CRCLBUSDT",
    "CRWVBUSDT", "DELLBUSDT", "DRAMBUSDT", "EWYBUSDT", "FLNCBUSDT", "GLWBUSDT",
    "GOOGLBUSDT", "GSBUSDT", "HOODBUSDT", "IBMBUSDT", "INTCBUSDT", "INTWBUSDT",
    "KORUBUSDT", "LITEBUSDT", "METABUSDT", "MRVLBUSDT", "MSFTBUSDT", "MSTRBUSDT",
    "MUBUSDT", "MUUBUSDT", "MVLLBUSDT", "NBISBUSDT", "NOKBUSDT", "NVDABUSDT",
    "ORCLBUSDT", "PLTRBUSDT", "PYPLBUSDT", "QCOMBUSDT", "QNTBUSDT", "QQQBUSDT",
    "RKLBBUSDT", "SKHYBUSDT", "SMHBUSDT", "SNDKBUSDT", "SNXXBUSDT", "SOXLBUSDT",
    "SOXSBUSDT", "SPCXBUSDT", "SPYBUSDT", "TQQQBUSDT", "TSLABUSDT", "TSMBUSDT",
    "WDCBUSDT",
)

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
    """Point-in-time universe filters. Everything is a parameter.

    TWO DISTINCT HISTORY REQUIREMENTS (they were conflated in the first
    version, which silently made the universe EMPTY -- see __post_init__):

      * min_obs_in_window -- DATA QUALITY. How many of the trailing
        `adv_lookback_bars` must be non-missing. Capped by the window size, so
        it MUST be <= adv_lookback_bars.

      * min_total_history_bars -- MATURITY. Total observed bars since the
        symbol's first print, up to this rebalance. Unbounded by the window;
        this is what keeps 3-bar listings and freshly-launched tokens out.
    """
    quote: str = "USDT"
    adv_lookback_bars: int = 180          # trailing bars for ADV + liveness
    min_adv_usd: float = 50_000.0         # floor: below this it is not tradeable
    min_obs_in_window: int = 144          # 80% coverage of the ADV window
    min_total_history_bars: int = 200     # maturity since listing
    max_symbols: Optional[int] = None     # cap universe size (by ADV) if set
    exclude_leveraged: bool = True
    exclude_stables: bool = True
    exclude_tokenized_equities: bool = True
    leveraged_suffixes: tuple = DEFAULT_LEVERAGED_SUFFIXES
    stable_bases: tuple = DEFAULT_STABLE_BASES
    tokenized_equity_symbols: tuple = DEFAULT_TOKENIZED_EQUITY_SYMBOLS
    require_full_window: bool = False     # if True, symbol must have EVERY bar

    def __post_init__(self):
        # Loud failure beats a silently empty universe.
        if self.min_obs_in_window > self.adv_lookback_bars:
            raise ValueError(
                f"min_obs_in_window ({self.min_obs_in_window}) exceeds "
                f"adv_lookback_bars ({self.adv_lookback_bars}). Coverage is "
                f"counted WITHIN the trailing window, so this can never be "
                f"satisfied and the universe would always be empty. Use "
                f"min_total_history_bars for a maturity requirement instead."
            )


def leveraged_underlying(symbol: str, spec: UniverseSpec = UniverseSpec()) -> Optional[str]:
    """The underlying a leveraged-token SPELLING implies, or None.

    Spelling only: JUPUSDT parses (J + UP + USDT -> JUSDT). Whether that
    underlying trades is what is_leveraged_token decides.
    """
    su = symbol.upper()
    quotes = (spec.quote,) if spec.quote else DEFAULT_LEVERAGED_QUOTES
    for q in quotes:
        if not su.endswith(q):
            continue
        stem = su[: -len(q)]
        for suf in spec.leveraged_suffixes:
            if stem.endswith(suf):
                base = stem[: -len(suf)]
                if base:
                    return f"{base}{q}"
                # Bare BULL/BEAR were the Binance-listed FTX 3x BTC tokens
                # (Cycle 61 maps them to BTC). No BLVT was ever a bare UP/DOWN,
                # so a bare UPUSDT would be some new spot asset, not leveraged.
                return f"BTC{q}" if suf in ("BULL", "BEAR") else None
    return None


def is_leveraged_token(symbol: str, traded, spec: UniverseSpec = UniverseSpec()) -> bool:
    """True iff `symbol` is spelled <BASE><UP|DOWN|BULL|BEAR><QUOTE> AND its
    implied underlying is in `traded` (a set of upper-case symbols)."""
    und = leveraged_underlying(symbol, spec)
    return und is not None and und in traded


def filter_symbol_names(symbols, spec: UniverseSpec = UniverseSpec(),
                        traded=None) -> list[str]:
    """Apply name-based exclusions (leveraged tokens, stablecoin pairs).

    `traded` is the symbol set that confirms a leveraged token's underlying
    trades. It defaults to `symbols`, which is right for the full archive
    listing (cmd_symbols). When filtering a SUBSET, pass the full listing as
    `traded`, or a genuine leveraged token whose underlying is not in the
    subset is kept.
    """
    symbols = list(symbols)
    traded_set = {s.upper() for s in (symbols if traded is None else traded)}
    out, lookalikes = [], []
    for s in symbols:
        su = s.upper()
        if spec.quote and not su.endswith(spec.quote):
            continue
        lookalike = False
        if spec.exclude_leveraged:
            if is_leveraged_token(su, traded_set, spec):
                continue
            lookalike = leveraged_underlying(su, spec) is not None
        if spec.exclude_stables:
            base = su[: -len(spec.quote)] if spec.quote else su
            if base in spec.stable_bases:
                continue
        if spec.exclude_tokenized_equities and su in spec.tokenized_equity_symbols:
            continue
        out.append(s)
        if lookalike:
            lookalikes.append(s)
    if lookalikes:
        logger.info("filter_symbol_names: kept %d leveraged-looking name(s) whose "
                    "implied underlying does not trade (spot assets): %s",
                    len(lookalikes), lookalikes)
    logger.info("filter_symbol_names: kept %d, dropped %d (leveraged/stable/quote)",
                len(out), len(symbols) - len(out))
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
    # Cumulative observed bars per symbol, computed once (maturity criterion).
    cum_obs = panel_close.notna().cumsum()
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

        total_obs = (cum_obs.loc[dt] if dt in cum_obs.index
                     else cum_obs.loc[:dt].iloc[-1])
        eligible = (
            live_now
            & (n_obs >= spec.min_obs_in_window)
            & (total_obs >= spec.min_total_history_bars)
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
            "total_obs": total_obs.values,
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
