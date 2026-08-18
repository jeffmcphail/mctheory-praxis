"""Chan CPO Layer A -- signal construction and intraday round-trip simulation.

Implements prereg section 3 verbatim:

    spread(t)      = GLD_close(t) - GDX_close(t) * GDX_weight
    Spread_EMA(0)  = Spread(0)
    Spread_EMA(t+1)= a*Spread(t+1)      + (1-a)*Spread_EMA(t)
    Spread_VAR(t+1)= a*dev(t+1)^2       + (1-a)*Spread_VAR(t)     dev = S - S_EMA
    Z(t)           = (Spread(t) - Spread_EMA(t)) / denominator     a = 2/lookback

with `denominator` a parameter (ambiguity A1: VAR as the paper writes it, or
STD as a conventional Bollinger z).

Everything here is strictly causal by construction: both recursions are
first-order IIR filters over the bar series, so the value at bar t is a
function of bars <= t only.  `leakage_checks.py` asserts that empirically
rather than trusting the claim.

TIMING CONVENTION (one place, stated once)
------------------------------------------
`pos[t]` is the position held FROM the close of bar t TO the close of bar
t+1, decided by Z(t) which is known at the close of bar t.  A round trip that
occupies bars [a, b] therefore enters at close[a] and exits at close[b+1].
`pos` is forced to 0 on the last bar of every session, which IS the
force-liquidation at 16:00 (Kibot stamps bars at their start, so the 15:59 bar
closes at 16:00), and is never carried in across a session boundary.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

# The prereg grid (section 3).  Nothing downstream hard-codes these.
GRID_GDX_WEIGHTS = (2.0, 2.5, 3.0, 3.5, 4.0)
GRID_ENTRY_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.25, 1.5, 2.0, 2.5)
GRID_LOOKBACKS = (30, 60, 90, 120, 180, 240, 360, 720)

EXIT_RATIO = -0.6  # exit_threshold = -0.6 * entry_threshold (prereg section 3)

SESSION_START = dt.time(9, 30)
SESSION_END = dt.time(15, 59)

DATA_DIR = Path("data/external/kibot")


@dataclass(frozen=True)
class Combination:
    """One point of the 400-combination grid."""

    gdx_weight: float
    entry_threshold: float
    lookback: int

    def as_dict(self) -> dict:
        return {
            "gdx_weight": self.gdx_weight,
            "entry_threshold": self.entry_threshold,
            "lookback": self.lookback,
        }

    def __str__(self) -> str:
        return (f"w={self.gdx_weight:g} entry={self.entry_threshold:g} "
                f"lookback={self.lookback:d}m")


@dataclass(frozen=True)
class Ambiguities:
    """The pre-registered interpretation gaps (prereg section 8).

    A3 (the unnamed eighth technical indicator) is a Layer B input only and is
    carried here purely so the record of what was run is complete.
    """

    zscore_denominator: str = "var"   # A1: var | std
    return_mode: str = "simple"       # A2: simple | log
    adjustment: str = "unadj"         # A4: adj | unadj
    eighth_indicator: str = "n/a"     # A3: Layer B only

    def label(self) -> str:
        return (f"A1={self.zscore_denominator} A2={self.return_mode} "
                f"A4={self.adjustment}")


@dataclass(frozen=True)
class Engineering:
    """Decisions the prereg does not specify, exposed rather than resolved silently.

    These are NOT ambiguities A1-A4 and MUST NOT be tuned to reach the target
    (prereg section 11).  They are fixed at their most faithful reading for the
    primary result; any variant run is reported as an out-of-protocol
    robustness diagnostic, never as the headline.
    """

    price_source: str = "trade"     # trade close (a "1-minute bar close") | quote mid
    bar_universe: str = "rth"       # EWMA over RTH bars only | over every delivered bar
    ewma_reset: str = "none"        # continuous recursion | restart each session
    flat_days: str = "include"      # zero-trade sessions count as 0.0 returns | dropped
    execution_lag: int = 0          # bars between the deciding close and the fill

    def label(self) -> str:
        return (f"price={self.price_source} bars={self.bar_universe} "
                f"ewma_reset={self.ewma_reset} flat_days={self.flat_days} "
                f"lag={self.execution_lag}")


@dataclass
class Bars:
    """Bar series ready for the signal, plus the session bookkeeping."""

    index: pd.DatetimeIndex      # every bar the EWMA consumes
    gld: np.ndarray              # aligned with index
    gdx: np.ndarray
    signal_mask: np.ndarray      # bars eligible to carry a position (RTH)
    day_code: np.ndarray         # session id, on the SIGNAL bars only
    day_start: np.ndarray        # bool, on the SIGNAL bars only
    day_end: np.ndarray          # bool, on the SIGNAL bars only
    dropped_price_bars: int = 0

    @property
    def n_signal_bars(self) -> int:
        return int(self.signal_mask.sum())

    @property
    def session_days(self) -> np.ndarray:
        return np.unique(self.day_code)


@dataclass
class Simulation:
    """Result of running one combination over the prepared bars."""

    combination: Combination
    position: np.ndarray         # int8, on signal bars
    entry_idx: np.ndarray        # round-trip entry bar (signal-bar space)
    exit_idx: np.ndarray         # round-trip exit bar  = entry block end + 1
    side: np.ndarray             # +1 long / -1 short
    trip_return: np.ndarray      # per round trip, in the A2 return mode
    trip_day: np.ndarray         # session id of each round trip
    daily: pd.Series             # session-indexed daily return, A2 mode
    trades_per_day: pd.Series    # session-indexed round-trip count


# --------------------------------------------------------------------------
# bar preparation
# --------------------------------------------------------------------------

def panel_path(adjustment: str, data_dir: Path = DATA_DIR) -> Path:
    return Path(data_dir) / f"pair_gld_gdx_1m_{adjustment}.parquet"


def day_codes(index: pd.DatetimeIndex) -> np.ndarray:
    """Session identifier per bar: midnight of its calendar day, as int64 ns."""
    return index.normalize().to_numpy().astype("datetime64[ns]").astype("int64")


def prepare_bars(
    panel: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    session_start: dt.time = SESSION_START,
    session_end: dt.time = SESSION_END,
    engineering: Engineering = Engineering(),
) -> Bars:
    """Slice, price and session-tag the panel.

    `end` is a hard wall: Phase 3 must not be able to see walk-forward bars at
    all, so they are cut here rather than merely ignored downstream.
    """
    panel = panel.loc[(panel.index >= start) & (panel.index <= end)]

    if engineering.price_source == "trade":
        gld_col, gdx_col = "gld_trade_close", "gdx_trade_close"
    elif engineering.price_source == "mid":
        gld_col, gdx_col = "gld_mid_close", "gdx_mid_close"
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown price_source {engineering.price_source!r}")

    times = panel.index.time
    rth = (times >= session_start) & (times <= session_end)

    if engineering.bar_universe == "rth":
        panel = panel.loc[rth]
        rth = np.ones(len(panel), dtype=bool)
    elif engineering.bar_universe != "all":  # pragma: no cover
        raise ValueError(f"unknown bar_universe {engineering.bar_universe!r}")

    px = panel[[gld_col, gdx_col]].copy()
    px.columns = ["gld", "gdx"]
    if px.empty:
        raise ValueError(f"no bars in window {start} .. {end}")

    # ~0.001% of RTH bars quote but never print.  Fill forward WITHIN a session
    # (causal); anything still missing is a session's leading bars, which are
    # dropped rather than back-filled -- back-filling would be lookahead.
    px = px.groupby(day_codes(px.index), sort=False).ffill()
    good = px["gld"].notna().to_numpy() & px["gdx"].notna().to_numpy()
    dropped = int((~good).sum())
    px = px.loc[good]
    rth = rth[good]

    gld = px["gld"].to_numpy(dtype=np.float64)
    gdx = px["gdx"].to_numpy(dtype=np.float64)

    sig_index = px.index[rth]
    day_code = day_codes(sig_index)
    day_start = np.ones(day_code.size, dtype=bool)
    day_end = np.ones(day_code.size, dtype=bool)
    if day_code.size:
        day_start[1:] = day_code[1:] != day_code[:-1]
        day_end[:-1] = day_code[1:] != day_code[:-1]

    return Bars(
        index=px.index,
        gld=gld,
        gdx=gdx,
        signal_mask=rth,
        day_code=day_code,
        day_start=day_start,
        day_end=day_end,
        dropped_price_bars=dropped,
    )


# --------------------------------------------------------------------------
# the recursions
# --------------------------------------------------------------------------

def ewma(x: np.ndarray, alpha: float, *, reset: np.ndarray | None = None) -> np.ndarray:
    """y[0] = x[0];  y[t] = alpha*x[t] + (1-alpha)*y[t-1].

    Exactly the prereg recursion.  `lfilter` runs it as a first-order IIR in C;
    the initial condition zi = (1-alpha)*x[0] is what makes y[0] == x[0].
    `reset` (bool mask) restarts the recursion at each marked bar and is only
    used by the out-of-protocol `--ewma-reset daily` diagnostic.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    if reset is None or not reset.any():
        y, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x,
                       zi=np.array([(1.0 - alpha) * x[0]]))
        return y

    out = np.empty_like(x)
    edges = np.flatnonzero(reset)
    edges = np.concatenate(([0], edges[edges > 0], [x.size]))
    edges = np.unique(edges)
    for lo, hi in zip(edges[:-1], edges[1:]):
        seg = x[lo:hi]
        y, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], seg,
                       zi=np.array([(1.0 - alpha) * seg[0]]))
        out[lo:hi] = y
    return out


def spread_series(gld: np.ndarray, gdx: np.ndarray, gdx_weight: float) -> np.ndarray:
    return gld - gdx * gdx_weight


def zscore(
    spread: np.ndarray,
    lookback: int,
    *,
    denominator: str = "var",
    reset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (z, finite_mask).

    VAR(0) is 0 by construction (dev(0) = spread(0) - EMA(0) = 0), so the first
    bars divide by ~0.  Those bars are marked non-finite and are held FLAT by
    `positions`; combined with the `lookback`-bar warm-up there is no live bar
    whose z is an artefact of initialisation.
    """
    alpha = 2.0 / float(lookback)
    ema = ewma(spread, alpha, reset=reset)
    dev = spread - ema
    var = ewma(dev * dev, alpha, reset=reset)

    if denominator == "var":
        den = var
    elif denominator == "std":
        den = np.sqrt(var)
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown zscore denominator {denominator!r}")

    with np.errstate(divide="ignore", invalid="ignore"):
        z = dev / den
    return z, np.isfinite(z)


# --------------------------------------------------------------------------
# the position state machine, vectorised
# --------------------------------------------------------------------------

def positions(
    z: np.ndarray,
    entry_threshold: float,
    *,
    valid: np.ndarray,
    day_start: np.ndarray,
    day_end: np.ndarray,
) -> np.ndarray:
    """Bollinger state machine (exit-then-enter), vectorised exactly.

    With E = entry_threshold and inner = 0.6E, every bar falls in exactly one
    of five bands, and each band's effect on the state is fixed:

        z < -E            DECISIVE long   -> +1 whatever the state was
        -E <= z <= -inner carry long band -> +1 only if already long, else 0
        |z| < inner       DECISIVE flat   ->  0 (both exit tests fire)
        inner <= z <= E   carry short band-> -1 only if already short, else 0
        z > E             DECISIVE short  -> -1 whatever the state was

    The naive ffill is WRONG here: a long sitting in the SHORT carry band must
    exit (its exit test z > -inner is satisfied) even though that band is a
    "hold" band for a short.  So instead of filling forward from the last
    non-carry bar, the state is read off two facts, both cheap prefix maxima:

        s          = index of the last DECISIVE bar at or before t
        last carry = index of the last opposite-band carry bar at or before t

    and the position is +1 iff bar s was a decisive long AND no short-band
    carry bar has occurred since (symmetrically for short).

    Bars with an undefined z (warm-up, VAR ~ 0) and the carry bars that open a
    session are folded into "decisive flat", which is what makes the
    force-liquidation structural: a run of carry bars can never reach back
    across a session boundary.
    """
    n = z.size
    inner = -EXIT_RATIO * entry_threshold          # 0.6 * E, positive

    with np.errstate(invalid="ignore"):
        dec_long = (z < -entry_threshold) & valid
        dec_short = (z > entry_threshold) & valid
        carry = valid & ~dec_long & ~dec_short & (np.abs(z) >= inner)
    carry_long = carry & (z < 0.0)
    carry_short = carry & ~carry_long

    # decisive-flat absorbs: the inner band, undefined z, and any carry bar
    # that opens a session (nothing may be carried in across the boundary).
    opening_carry = day_start & carry
    decisive = dec_long | dec_short | ~(carry & ~opening_carry)

    idx = np.arange(n)
    last_dec = np.where(decisive, idx, -1)
    np.maximum.accumulate(last_dec, out=last_dec)

    live_carry_long = carry_long & ~decisive
    live_carry_short = carry_short & ~decisive
    last_cl = np.where(live_carry_long, idx, -1)
    last_cs = np.where(live_carry_short, idx, -1)
    np.maximum.accumulate(last_cl, out=last_cl)
    np.maximum.accumulate(last_cs, out=last_cs)

    s = last_dec
    is_long = dec_long[s] & (last_cs < s)
    is_short = dec_short[s] & (last_cl < s)

    pos = np.where(is_long, 1, np.where(is_short, -1, 0)).astype(np.int8)
    pos[day_end] = 0                                # force-liquidate at 16:00
    return pos


def apply_execution_lag(pos: np.ndarray, lag: int, day_start: np.ndarray,
                        day_end: np.ndarray) -> np.ndarray:
    """Fill `lag` bars after the close that decided the position.

    The paper fills at the very close whose z produced the signal, which is the
    standard and standardly optimistic assumption -- at one-minute resolution
    that close is a single print sitting on one side of the quote, so a
    mean-reversion rule can appear to buy at the bid and sell at the ask
    without paying anything.  lag=1 is the diagnostic for that; lag=0 is what
    the prereg specifies and remains the primary.
    """
    if lag <= 0:
        return pos
    out = np.zeros_like(pos)
    out[lag:] = pos[:-lag]

    idx = np.arange(pos.size)
    session_open = np.where(day_start, idx, 0)
    np.maximum.accumulate(session_open, out=session_open)
    out[(idx - session_open) < lag] = 0        # no decision leaks across a session
    out[day_end] = 0
    return out


def extract_trades(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Round trips as (entry_idx, exit_idx, side).

    A round trip is a maximal run of a constant non-zero position.  Entry fills
    at close[entry_idx]; exit fills at close[exit_idx] where exit_idx is one bar
    past the run.  That bar always exists inside the same session because
    `positions` forces the session's last bar flat.
    """
    prev = np.empty_like(pos)
    prev[0] = 0
    prev[1:] = pos[:-1]
    nxt = np.empty_like(pos)
    nxt[-1] = 0
    nxt[:-1] = pos[1:]

    live = pos != 0
    entry_idx = np.flatnonzero(live & (pos != prev))
    end_idx = np.flatnonzero(live & (pos != nxt))
    return entry_idx, end_idx + 1, pos[entry_idx].astype(np.int64)


def simulate(
    bars: Bars,
    combination: Combination,
    ambiguities: Ambiguities,
    engineering: Engineering,
    *,
    z_cache: dict | None = None,
) -> Simulation:
    """Run one grid combination end to end over the prepared bars."""
    key = (combination.gdx_weight, combination.lookback,
           ambiguities.zscore_denominator, engineering.ewma_reset)
    if z_cache is not None and key in z_cache:
        z_sig, valid_sig = z_cache[key]
    else:
        reset = None
        if engineering.ewma_reset == "daily":
            reset = np.zeros(bars.index.size, dtype=bool)
            reset[np.flatnonzero(bars.signal_mask)[bars.day_start]] = True
        spread = spread_series(bars.gld, bars.gdx, combination.gdx_weight)
        z, finite = zscore(spread, combination.lookback,
                           denominator=ambiguities.zscore_denominator,
                           reset=reset)
        warm = np.arange(z.size) >= combination.lookback
        z_sig = z[bars.signal_mask]
        valid_sig = (finite & warm)[bars.signal_mask]
        if z_cache is not None:
            z_cache[key] = (z_sig, valid_sig)

    pos = positions(z_sig, combination.entry_threshold, valid=valid_sig,
                    day_start=bars.day_start, day_end=bars.day_end)
    pos = apply_execution_lag(pos, engineering.execution_lag,
                              bars.day_start, bars.day_end)
    entry_idx, exit_idx, side = extract_trades(pos)

    price = bars.gld[bars.signal_mask]
    if ambiguities.return_mode == "simple":
        trip = side * (price[exit_idx] / price[entry_idx] - 1.0)
    elif ambiguities.return_mode == "log":
        trip = side * (np.log(price[exit_idx]) - np.log(price[entry_idx]))
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown return_mode {ambiguities.return_mode!r}")

    days = np.unique(bars.day_code)
    trip_day = bars.day_code[entry_idx]
    slot = np.searchsorted(days, trip_day)

    daily = np.zeros(days.size, dtype=np.float64)
    np.add.at(daily, slot, trip)                       # sum of round-trip returns
    counts = np.zeros(days.size, dtype=np.int64)
    np.add.at(counts, slot, 1)

    day_index = pd.DatetimeIndex(days.astype("datetime64[ns]"), name="session")
    return Simulation(
        combination=combination,
        position=pos,
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        side=side,
        trip_return=trip,
        trip_day=trip_day,
        daily=pd.Series(daily, index=day_index, name="ret"),
        trades_per_day=pd.Series(counts, index=day_index, name="round_trips"),
    )
