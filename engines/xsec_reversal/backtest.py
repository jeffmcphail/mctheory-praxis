"""
engines/xsec_reversal/backtest.py

Tiered cross-sectional reversal backtest engine.

STRATEGY (per pre-registration)
-------------------------------
At each rebalance time t, within each liquidity tier:
  1. formation return  r_form(i) = close_i(t)/close_i(t - L) - 1   [causal]
  2. residualize       rr(i) = r_form(i) - cross-sectional mean    (or beta-adj)
  3. rank rr ascending; LONG the bottom quantile (biggest losers),
     SHORT the top quantile (biggest winners) -- reversal
  4. dollar-neutral, equal-weight within each leg
  5. enter after `execution_lag_bars`, hold H bars, then rebalance

NO-LOOKAHEAD GUARANTEES (each has a test in tests/)
  * formation returns use only bars <= t
  * tier/ADV assignment uses only bars <= t (see universe.py)
  * spread estimates are trailing rolling means (see costs.py)
  * positions are shifted forward by execution_lag_bars before being multiplied
    by forward returns, so a signal computed at t can never earn the return of
    the bar that produced it
  * forward returns span (t+lag) -> (t+lag+H), strictly after the signal

WHY DOLLAR-NEUTRAL AND WHY RESIDUALIZE
--------------------------------------
Crypto returns are dominated by one common factor (roughly "BTC beta"). A raw
long-short ranked on total return is mostly a bet on beta dispersion, not a
relative-value signal. Residualizing against the cross-sectional mean (an
equal-weight index proxy) leaves the idiosyncratic component the reversal
hypothesis is actually about.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .costs import CostSpec, one_way_cost_bps

logger = logging.getLogger("xsec.backtest")


@dataclass(frozen=True)
class SignalSpec:
    """Cross-sectional reversal signal parameters."""
    formation_bars: int = 6           # lookback L for the formation return
    holding_bars: int = 6             # holding horizon H
    execution_lag_bars: int = 1       # bars between signal and entry (>=1 safe)
    quantile: float = 0.2             # top/bottom fraction traded per leg
    min_symbols_per_tier: int = 10    # skip a tier-period with fewer names
    residualize_mode: str = "demean"  # 'demean' | 'beta' | 'none'
    beta_lookback_bars: int = 180     # for residualize_mode='beta'
    winsorize_pct: float = 0.01       # clip extreme formation returns


@dataclass(frozen=True)
class BacktestSpec:
    """Backtest execution and accounting parameters."""
    periods_per_year: float = 2190.0  # 4h bars: 6/day * 365
    gross_leverage: float = 1.0       # total book size (long + short = 1.0)
    apply_costs: bool = True
    rebalance_every_bars: Optional[int] = None  # default: holding_bars


@dataclass
class BacktestResult:
    tier: str
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    cost_bps: pd.Series
    n_names: pd.Series
    ic: pd.Series
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        return (
            f"[{self.tier}] periods={m.get('n_periods',0)} "
            f"avg_names={m.get('avg_names',0):.1f} | "
            f"GROSS sharpe={m.get('gross_sharpe',float('nan')):.3f} "
            f"ann={m.get('gross_ann_return',float('nan'))*100:.2f}% | "
            f"NET sharpe={m.get('net_sharpe',float('nan')):.3f} "
            f"ann={m.get('net_ann_return',float('nan'))*100:.2f}% | "
            f"turnover={m.get('avg_turnover',float('nan')):.2f} "
            f"cost={m.get('avg_cost_bps',float('nan')):.1f}bps/reb | "
            f"IC={m.get('mean_ic',float('nan')):+.4f} "
            f"maxDD={m.get('net_max_drawdown',float('nan'))*100:.1f}%"
        )


# --------------------------------------------------------------------------- #
# Signal construction
# --------------------------------------------------------------------------- #
def compute_formation_returns(panel_close: pd.DataFrame,
                              formation_bars: int) -> pd.DataFrame:
    """r(t) = close(t)/close(t-L) - 1, causal (no future bars)."""
    return panel_close / panel_close.shift(formation_bars) - 1.0


def residualize(form_ret: pd.DataFrame,
                panel_close: Optional[pd.DataFrame] = None,
                spec: SignalSpec = SignalSpec(),
                market_proxy: Optional[pd.Series] = None) -> pd.DataFrame:
    """Remove the common factor from formation returns.

    'demean' : subtract the cross-sectional mean at each timestamp.

               *** WARNING: THIS IS A NO-OP FOR RANK-BASED POSITIONS. ***
               Demeaning subtracts the SAME scalar from every symbol at a given
               timestamp, and build_positions ranks the signal. Ranking is
               invariant to a constant shift, so 'demean' and 'none' produce
               IDENTICAL positions and identical P&L. Measured in Cycle 60:
               best and median net Sharpe matched to 3 decimals in all three
               tiers. Use 'beta' if you need an actual market-neutrality
               control -- only a PER-SYMBOL adjustment changes the ranking.
    'beta'   : subtract beta_i * market_return, with beta estimated on a
               TRAILING window (causal). More precise, adds estimation noise.
    'none'   : raw returns (kept as a pre-registered control -- it tests
               whether any apparent edge is really just beta dispersion).
    """
    mode = spec.residualize_mode
    if mode == "none":
        out = form_ret.copy()
    elif mode == "demean":
        out = form_ret.sub(form_ret.mean(axis=1, skipna=True), axis=0)
    elif mode == "beta":
        if panel_close is None:
            raise ValueError("residualize_mode='beta' requires panel_close")
        bar_ret = panel_close.pct_change()
        mkt = market_proxy if market_proxy is not None else bar_ret.mean(axis=1, skipna=True)
        lb = spec.beta_lookback_bars
        cov = bar_ret.rolling(lb, min_periods=lb // 3).cov(mkt)
        var = mkt.rolling(lb, min_periods=lb // 3).var()
        beta = cov.div(var, axis=0)
        mkt_form = (1.0 + mkt).rolling(spec.formation_bars).apply(
            np.prod, raw=True) - 1.0
        out = form_ret.sub(beta.mul(mkt_form, axis=0))
    else:
        raise ValueError(f"unknown residualize_mode '{mode}'")

    if spec.winsorize_pct and spec.winsorize_pct > 0:
        lo = out.quantile(spec.winsorize_pct, axis=1)
        hi = out.quantile(1.0 - spec.winsorize_pct, axis=1)
        out = out.clip(lower=lo, upper=hi, axis=0)
    return out


def build_positions(signal: pd.DataFrame,
                    eligible_mask: pd.DataFrame,
                    spec: SignalSpec = SignalSpec(),
                    gross_leverage: float = 1.0) -> pd.DataFrame:
    """Dollar-neutral reversal positions from a cross-sectional signal.

    LONG the most negative residual returns (losers), SHORT the most positive
    (winners). Each leg is equal-weighted and sums to gross_leverage/2, so the
    book is dollar-neutral with total gross exposure = gross_leverage.
    """
    sig = signal.where(eligible_mask)
    pos = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)

    for t in sig.index:
        row = sig.loc[t].dropna()
        n = len(row)
        if n < spec.min_symbols_per_tier:
            continue
        k = max(1, int(np.floor(n * spec.quantile)))
        ranked = row.sort_values()
        longs = ranked.index[:k]        # biggest losers -> buy
        shorts = ranked.index[-k:]      # biggest winners -> sell
        if len(set(longs) & set(shorts)):
            continue                     # degenerate overlap; skip period
        w = gross_leverage / 2.0 / k
        pos.loc[t, longs] = w
        pos.loc[t, shorts] = -w
    return pos


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _rank_ic(signal_row: pd.Series, fwd_row: pd.Series) -> float:
    """Spearman rank IC between signal and forward return.

    NOTE the sign convention: for a reversal effect we EXPECT a negative
    correlation between the formation return and the forward return. We report
    the raw correlation, so a working reversal signal shows NEGATIVE IC here.
    """
    both = pd.concat([signal_row, fwd_row], axis=1).dropna()
    if len(both) < 5:
        return float("nan")
    return float(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman"))


def run_backtest(
    panel_close: pd.DataFrame,
    universe_tiers: pd.DataFrame,
    spread_bps: Optional[pd.DataFrame] = None,
    signal_spec: SignalSpec = SignalSpec(),
    bt_spec: BacktestSpec = BacktestSpec(),
    cost_spec: CostSpec = CostSpec(),
    tiers: Optional[list] = None,
) -> dict:
    """Run the tiered cross-sectional reversal backtest.

    Returns {tier_label: BacktestResult}. Each tier is run INDEPENDENTLY -- the
    per-tier alpha-vs-cost comparison is the experiment's headline output, so a
    blended number is never produced.
    """
    form = compute_formation_returns(panel_close, signal_spec.formation_bars)
    sig = residualize(form, panel_close, signal_spec)

    reb_every = bt_spec.rebalance_every_bars or signal_spec.holding_bars
    lag = signal_spec.execution_lag_bars
    H = signal_spec.holding_bars

    # Forward return realised over the holding window, entered after the lag.
    entry = panel_close.shift(-lag)
    exit_ = panel_close.shift(-(lag + H))
    fwd = exit_ / entry - 1.0

    if tiers is None:
        tiers = [t for t in universe_tiers["tier"].dropna().unique()]
    results: dict = {}

    for tier in sorted(tiers):
        tsub = universe_tiers[(universe_tiers["tier"] == tier)
                              & universe_tiers["eligible"]]
        if tsub.empty:
            logger.warning("tier %s: no eligible symbol-periods; skipping", tier)
            continue

        mask = (tsub.assign(v=True)
                    .pivot_table(index="dt", columns="symbol", values="v",
                                 aggfunc="first")
                    .reindex(index=panel_close.index, columns=panel_close.columns)
                    .fillna(False).astype(bool))

        reb_idx = panel_close.index[::reb_every]
        mask_reb = mask.loc[mask.index.isin(reb_idx)]
        sig_reb = sig.reindex(mask_reb.index)
        fwd_reb = fwd.reindex(mask_reb.index)

        pos = build_positions(sig_reb, mask_reb, signal_spec,
                              bt_spec.gross_leverage)

        gross = (pos * fwd_reb).sum(axis=1, skipna=True)
        traded = pos.abs().sum(axis=1)
        # Turnover: full round trip each rebalance (enter + exit the book).
        prev = pos.shift(1).fillna(0.0)
        turnover = (pos - prev).abs().sum(axis=1)

        if bt_spec.apply_costs and spread_bps is not None:
            cost_one_way = one_way_cost_bps(
                spread_bps.reindex(index=mask_reb.index,
                                   columns=panel_close.columns), cost_spec)
            # Cost = |weight change| * one-way cost, per symbol, per rebalance.
            dpos = (pos - prev).abs()
            cost_frac = (dpos * cost_one_way / 1e4).sum(axis=1, skipna=True)
        elif bt_spec.apply_costs:
            flat = (cost_spec.fixed_spread_bps / 2.0
                    + cost_spec.fee_bps_per_side
                    + cost_spec.extra_slippage_bps_per_side)
            cost_frac = turnover * flat / 1e4
        else:
            cost_frac = pd.Series(0.0, index=pos.index)

        net = gross - cost_frac

        ic = pd.Series(
            {t: _rank_ic(sig_reb.loc[t].where(mask_reb.loc[t]), fwd_reb.loc[t])
             for t in mask_reb.index}, dtype=float)

        valid = gross.notna() & (traded > 0)
        g, n = gross[valid], net[valid]
        pen = float(bt_spec.periods_per_year) / float(reb_every)

        def _sharpe(x: pd.Series) -> float:
            if len(x) < 2 or x.std(ddof=1) == 0 or np.isnan(x.std(ddof=1)):
                return float("nan")
            return float(x.mean() / x.std(ddof=1) * np.sqrt(pen))

        metrics = {
            "n_periods": int(valid.sum()),
            "avg_names": float((pos != 0).sum(axis=1)[valid].mean()) if valid.any() else 0.0,
            "gross_sharpe": _sharpe(g),
            "net_sharpe": _sharpe(n),
            "gross_mean_bps": float(g.mean() * 1e4) if len(g) else float("nan"),
            "net_mean_bps": float(n.mean() * 1e4) if len(n) else float("nan"),
            "gross_ann_return": float(g.mean() * pen) if len(g) else float("nan"),
            "net_ann_return": float(n.mean() * pen) if len(n) else float("nan"),
            "avg_turnover": float(turnover[valid].mean()) if valid.any() else float("nan"),
            "avg_cost_bps": float(cost_frac[valid].mean() * 1e4) if valid.any() else float("nan"),
            "mean_ic": float(ic.mean(skipna=True)) if len(ic) else float("nan"),
            "net_max_drawdown": _max_drawdown((1.0 + n).cumprod()) if len(n) else float("nan"),
            "hit_rate": float((n > 0).mean()) if len(n) else float("nan"),
        }
        res = BacktestResult(tier=tier, gross_returns=g, net_returns=n,
                             turnover=turnover[valid], cost_bps=cost_frac[valid] * 1e4,
                             n_names=(pos != 0).sum(axis=1)[valid], ic=ic,
                             metrics=metrics)
        logger.info(res.summary())
        results[tier] = res

    return results


def capacity_analysis(
    universe_tiers: pd.DataFrame,
    signal_spec: SignalSpec = SignalSpec(),
    participation_rate: float = 0.01,
    bars_per_rebalance: Optional[int] = None,
) -> pd.DataFrame:
    """Max deployable capital per tier at a given ADV participation rate.

    A tier whose edge only exists at $2k positions is a NEGATIVE result for
    practical purposes; this is pre-registered as a first-class output, not a
    footnote.
    """
    bars = bars_per_rebalance or signal_spec.holding_bars
    rows = []
    for tier, grp in universe_tiers[universe_tiers["eligible"]].groupby("tier"):
        per_dt = grp.groupby("dt")["adv_usd"]
        median_adv = float(per_dt.median().median())
        n_names = float(grp.groupby("dt").size().median())
        k = max(1, int(np.floor(n_names * signal_spec.quantile)))
        # Per-name notional at the participation cap, over the holding window.
        per_name = median_adv * participation_rate * bars
        book = per_name * k * 2.0  # both legs
        rows.append({
            "tier": tier,
            "median_adv_usd": median_adv,
            "median_names": n_names,
            "names_per_leg": k,
            "max_per_name_usd": per_name,
            "max_book_usd": book,
            "participation_rate": participation_rate,
        })
    out = pd.DataFrame(rows).sort_values("median_adv_usd", ascending=False)
    logger.info("capacity: %s", out.to_dict("records"))
    return out
