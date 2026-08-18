"""Performance metrics and the prereg section 7 benchmarks.

Metric definitions are fixed here, once, so that ambiguity A2 (simple vs log
round-trip aggregation) changes only how a DAY's return is assembled and never
how a day series is scored.  A log-mode daily series is converted to simple
(exp(r)-1) before scoring; every number below is therefore a simple-return
statistic.

Note on Chan's own target table (prereg section 2): 17.29% annual and 73%
three-year cumulative are not mutually consistent under any standard
convention -- compounding 17.29% for three years gives 61.4%, and 73% over
three years implies a 20.07% CAGR.  Geometric can never exceed arithmetic on
the same series, so the pair cannot both be right.  Both an arithmetic
annualisation and a CAGR are therefore reported, and the comparison is stated
against each rather than picking the flattering one.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


def to_simple(daily: pd.Series, return_mode: str) -> pd.Series:
    """Normalise a daily series to simple returns for scoring."""
    if return_mode == "log":
        return np.expm1(daily)
    return daily


@dataclass
class Performance:
    n_days: int
    cumulative: float          # compounded over the whole window
    annual_arithmetic: float   # mean(daily) * 252
    annual_cagr: float         # (1+cum)^(252/n) - 1
    vol_annual: float
    sharpe: float
    max_drawdown: float
    calmar_cagr: float
    calmar_arithmetic: float
    hit_rate: float
    best_day: float
    worst_day: float

    def as_dict(self) -> dict:
        return asdict(self)


def performance(daily: pd.Series, return_mode: str = "simple") -> Performance:
    r = to_simple(daily, return_mode).to_numpy(dtype=np.float64)
    n = r.size
    if n == 0:
        raise ValueError("empty daily return series")

    equity = np.cumprod(1.0 + r)
    cumulative = float(equity[-1] - 1.0)
    mean, sd = float(r.mean()), float(r.std(ddof=1)) if n > 1 else 0.0

    cagr = float(equity[-1] ** (TRADING_DAYS / n) - 1.0) if equity[-1] > 0 else float("nan")
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max(1.0 - equity / peak)) if n else 0.0
    traded = r[r != 0.0]

    return Performance(
        n_days=n,
        cumulative=cumulative,
        annual_arithmetic=mean * TRADING_DAYS,
        annual_cagr=cagr,
        vol_annual=sd * np.sqrt(TRADING_DAYS),
        sharpe=(mean / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
        max_drawdown=max_dd,
        calmar_cagr=(cagr / max_dd) if max_dd > 0 else float("nan"),
        calmar_arithmetic=(mean * TRADING_DAYS / max_dd) if max_dd > 0 else float("nan"),
        hit_rate=float((traded > 0).mean()) if traded.size else float("nan"),
        best_day=float(r.max()),
        worst_day=float(r.min()),
    )


# --------------------------------------------------------------------------
# prereg section 7 benchmarks -- MANDATORY, the strategy is UNHEDGED
# --------------------------------------------------------------------------

@dataclass
class Benchmarks:
    buy_and_hold: Performance          # (a) close-to-close incl. overnight
    intraday_long: Performance         # (b) 09:30 -> 16:00 every session
    buy_and_hold_total: float
    intraday_long_total: float
    correlation_to_intraday_long: float
    long_bar_share: float              # of in-market bar-time
    short_bar_share: float
    time_in_market: float
    long_trip_share: float             # of round trips
    beats_intraday_long: bool

    def as_dict(self) -> dict:
        d = asdict(self)
        d["buy_and_hold"] = self.buy_and_hold.as_dict()
        d["intraday_long"] = self.intraday_long.as_dict()
        return d


def benchmark_series(price: np.ndarray, day_code: np.ndarray,
                     day_start: np.ndarray, day_end: np.ndarray
                     ) -> tuple[pd.Series, pd.Series]:
    """Daily returns for (a) buy-and-hold GLD and (b) intraday-only long GLD.

    (a) closes at the session close and re-marks at the next session close, so
        it carries the overnight gap -- that is the point of the comparison.
    (b) enters at the 09:30 bar close and exits at the 16:00 force-liquidation
        bar, i.e. exactly the exposure window the strategy is allowed.
    """
    days = np.unique(day_code)
    idx = pd.DatetimeIndex(days.astype("datetime64[ns]"), name="session")

    opens = price[day_start]
    closes = price[day_end]

    intraday = closes / opens - 1.0
    bh = np.empty_like(closes)
    bh[0] = closes[0] / opens[0] - 1.0     # first session bought at its open
    bh[1:] = closes[1:] / closes[:-1] - 1.0

    return (pd.Series(bh, index=idx, name="buy_and_hold"),
            pd.Series(intraday, index=idx, name="intraday_long"))


def benchmarks(strategy_daily: pd.Series, return_mode: str,
               price: np.ndarray, day_code: np.ndarray,
               day_start: np.ndarray, day_end: np.ndarray,
               position: np.ndarray, side: np.ndarray) -> Benchmarks:
    bh, intraday = benchmark_series(price, day_code, day_start, day_end)
    strat = to_simple(strategy_daily, return_mode)

    common = strat.index.intersection(intraday.index)
    corr = float(np.corrcoef(strat.loc[common].to_numpy(),
                             intraday.loc[common].to_numpy())[0, 1])

    live = position != 0
    n_live = int(live.sum())
    strat_perf = performance(strategy_daily, return_mode)
    intraday_perf = performance(intraday)

    return Benchmarks(
        buy_and_hold=performance(bh),
        intraday_long=intraday_perf,
        buy_and_hold_total=float(np.prod(1.0 + bh.to_numpy()) - 1.0),
        intraday_long_total=float(np.prod(1.0 + intraday.to_numpy()) - 1.0),
        correlation_to_intraday_long=corr,
        long_bar_share=float((position == 1).sum() / n_live) if n_live else float("nan"),
        short_bar_share=float((position == -1).sum() / n_live) if n_live else float("nan"),
        time_in_market=float(n_live / position.size) if position.size else 0.0,
        long_trip_share=float((side > 0).mean()) if side.size else float("nan"),
        beats_intraday_long=bool(strat_perf.sharpe > intraday_perf.sharpe
                                 and strat_perf.cumulative > intraday_perf.cumulative),
    )


def round_trip_stats(trades_per_day: pd.Series) -> dict:
    """Round trips per day -- prereg section 6 makes this the headline number."""
    c = trades_per_day.to_numpy(dtype=np.float64)
    active = c[c > 0]
    return {
        "sessions": int(c.size),
        "total_round_trips": int(c.sum()),
        "mean_per_day": float(c.mean()) if c.size else 0.0,
        "median_per_day": float(np.median(c)) if c.size else 0.0,
        "p95_per_day": float(np.percentile(c, 95)) if c.size else 0.0,
        "max_per_day": float(c.max()) if c.size else 0.0,
        "sessions_with_a_trade": int(active.size),
        "share_of_sessions_traded": float(active.size / c.size) if c.size else 0.0,
        "mean_per_active_day": float(active.mean()) if active.size else 0.0,
    }


def trip_economics(trip_return: np.ndarray, entry_idx: np.ndarray,
                   exit_idx: np.ndarray, return_mode: str = "simple") -> dict:
    """Gross return per round trip, in bps, and how long each one is held.

    This is the number prereg section 6 is really about.  Cost is charged per
    round trip, so the comparison that decides the experiment is mean GROSS
    bps per round trip against the pre-registered 1.26 bps round-trip cost --
    not the annual return against the annual drag.  `mean_gross_bps` doubles
    as the breakeven cost: at that round-trip cost the strategy nets zero.
    """
    r = np.expm1(trip_return) if return_mode == "log" else trip_return
    bps = 1e4 * np.asarray(r, dtype=np.float64)
    hold = np.asarray(exit_idx - entry_idx, dtype=np.float64)
    if bps.size == 0:
        return {"trips": 0, "mean_gross_bps": float("nan")}
    return {
        "trips": int(bps.size),
        "mean_gross_bps": float(bps.mean()),
        "median_gross_bps": float(np.median(bps)),
        "std_gross_bps": float(bps.std(ddof=1)) if bps.size > 1 else float("nan"),
        "win_rate": float((bps > 0).mean()),
        "mean_hold_bars": float(hold.mean()),
        "median_hold_bars": float(np.median(hold)),
        "breakeven_cost_bps_per_round_trip": float(bps.mean()),
    }


def cost_drag_bps(mean_round_trips: float, bps_per_round_trip: float = 1.26) -> dict:
    """Annualised drag implied by the prereg's 1.26 bps round-trip cost.

    Phase 4 owns the cost overlay; this is the arithmetic prereg section 6
    demands be shown alongside the trade count, not a net result.
    """
    per_day = mean_round_trips * bps_per_round_trip
    return {
        "bps_per_round_trip": bps_per_round_trip,
        "bps_per_day": per_day,
        "annualised_drag": per_day * TRADING_DAYS / 1e4,
    }
