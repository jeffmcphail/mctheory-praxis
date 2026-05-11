"""Information-driven bar builders (AFML Ch. 2 sections 2.3.2.1 - 2.3.2.3).

Four bar types, all single-asset, all stateful (push trades in
order; emit a ClosedBar when the threshold is crossed).

Trade-order convention: caller must push trades in (timestamp ASC,
trade_id ASC) order. Builders do not re-sort.

Aggressor convention (verified from engines/crypto_data_collector.py):
    side='buy'  -> taker bought (sign +1, BUY-side aggression)
    side='sell' -> taker sold   (sign -1, SELL-side aggression)

Closed-bar OHLC follows AFML's convention: the bar that closes
contains every trade up to AND INCLUDING the trade that triggered
the threshold cross. Any threshold overshoot stays attributed to
that bar; the next bar starts fresh on the next pushed trade.

ClosedBar.bar_index is left as -1 here; the writer assigns it
based on what is already persisted for that slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Trade:
    """A single trade from the trades table."""
    trade_id: int
    timestamp_ms: int
    price: float
    amount: float          # base-asset units
    quote_amount: float    # USD-equivalent (price * amount)
    side: str              # 'buy' or 'sell' (taker direction)


@dataclass
class ClosedBar:
    """A closed information-driven bar ready for INSERT."""
    asset: str
    bar_type: str
    threshold_value: float
    bar_index: int                  # -1 until writer assigns
    start_timestamp: int            # ms UTC
    end_timestamp: int              # ms UTC
    start_datetime: str             # ISO 8601 +00:00
    end_datetime: str               # ISO 8601 +00:00
    open: float
    high: float
    low: float
    close: float
    base_volume: float
    quote_volume: float
    tick_count: int
    buy_quote: float                # aggressor-buy USD
    sell_quote: float               # aggressor-sell USD
    imbalance_quote: float          # buy_quote - sell_quote


def _iso_utc(ts_ms: int) -> str:
    """ms epoch -> ISO 8601 string with +00:00 suffix (Rule 35)."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def _signed_quote(trade: Trade) -> float:
    """+quote_amount if taker bought, -quote_amount if taker sold."""
    return trade.quote_amount if trade.side == "buy" else -trade.quote_amount


@dataclass
class _OpenState:
    """Mutable accumulator for the bar currently being built."""
    tick_count: int = 0
    start_ts: int = 0
    end_ts: int = 0
    open_px: float = 0.0
    high_px: float = float("-inf")
    low_px: float = float("inf")
    close_px: float = 0.0
    base_volume: float = 0.0
    quote_volume: float = 0.0
    buy_quote: float = 0.0
    sell_quote: float = 0.0
    signed_quote_sum: float = 0.0   # for VIB

    def absorb(self, t: Trade) -> None:
        if self.tick_count == 0:
            self.start_ts = t.timestamp_ms
            self.open_px = t.price
        self.tick_count += 1
        self.end_ts = t.timestamp_ms
        self.close_px = t.price
        if t.price > self.high_px:
            self.high_px = t.price
        if t.price < self.low_px:
            self.low_px = t.price
        self.base_volume += t.amount
        self.quote_volume += t.quote_amount
        if t.side == "buy":
            self.buy_quote += t.quote_amount
            self.signed_quote_sum += t.quote_amount
        else:
            self.sell_quote += t.quote_amount
            self.signed_quote_sum -= t.quote_amount


class BarBuilder:
    """Common base. Subclasses implement `_should_close()`."""

    bar_type = ""  # subclasses set

    def __init__(self, asset: str, threshold_value: float):
        self.asset = asset
        self.threshold_value = float(threshold_value)
        self._state = _OpenState()

    def push(self, trade: Trade) -> Optional[ClosedBar]:
        """Push a trade in chronological order.

        Returns a ClosedBar if this trade pushed the bar past
        threshold; otherwise None. The bar contains every trade
        absorbed since the last close.
        """
        self._state.absorb(trade)
        if self._should_close():
            bar = self._finalize()
            self._state = _OpenState()
            return bar
        return None

    def push_all(self, trades: Iterable[Trade]) -> List[ClosedBar]:
        """Convenience: feed an iterable of trades, return all closed bars."""
        out: List[ClosedBar] = []
        for t in trades:
            bar = self.push(t)
            if bar is not None:
                out.append(bar)
        return out

    def flush_partial(self) -> Optional[ClosedBar]:
        """Return the current partial state as a ClosedBar (NOT for
        persistence -- partial bars are never written). Returns None
        if no trades have accumulated."""
        if self._state.tick_count == 0:
            return None
        return self._finalize()

    def _should_close(self) -> bool:
        raise NotImplementedError

    def _finalize(self) -> ClosedBar:
        s = self._state
        return ClosedBar(
            asset=self.asset,
            bar_type=self.bar_type,
            threshold_value=self.threshold_value,
            bar_index=-1,
            start_timestamp=s.start_ts,
            end_timestamp=s.end_ts,
            start_datetime=_iso_utc(s.start_ts),
            end_datetime=_iso_utc(s.end_ts),
            open=s.open_px,
            high=s.high_px,
            low=s.low_px,
            close=s.close_px,
            base_volume=s.base_volume,
            quote_volume=s.quote_volume,
            tick_count=s.tick_count,
            buy_quote=s.buy_quote,
            sell_quote=s.sell_quote,
            imbalance_quote=s.buy_quote - s.sell_quote,
        )


class DollarBars(BarBuilder):
    """Closes when cumulative quote_amount (USD) >= threshold."""
    bar_type = "dollar"

    def _should_close(self) -> bool:
        return self._state.quote_volume >= self.threshold_value


class VolumeBars(BarBuilder):
    """Closes when cumulative base-asset volume >= threshold."""
    bar_type = "volume"

    def _should_close(self) -> bool:
        return self._state.base_volume >= self.threshold_value


class VolumeImbalanceBars(BarBuilder):
    """Closes when |signed cumulative quote| crosses +/- threshold.

    AFML Ch. 2 Section 2.3.2.2 (theta_T). The bar's
    imbalance_quote (= buy_quote - sell_quote) preserves the
    direction sign; consumers can read sign(imbalance_quote) to
    recover whether the close was on the buy or sell side.
    """
    bar_type = "vib"

    def _should_close(self) -> bool:
        return abs(self._state.signed_quote_sum) >= self.threshold_value


class VolumeRunBars(BarBuilder):
    """Closes when max(cum buy_quote, cum sell_quote) >= threshold.

    AFML Ch. 2 Section 2.3.2.3 (one-sided run). v0.1 uses the
    simple "max-side" definition rather than AFML's adaptive
    expected-run estimator; that refinement is deferred to v0.2.
    """
    bar_type = "vrb"

    def _should_close(self) -> bool:
        s = self._state
        return max(s.buy_quote, s.sell_quote) >= self.threshold_value


_BUILDER_BY_TYPE = {
    "dollar": DollarBars,
    "volume": VolumeBars,
    "vib": VolumeImbalanceBars,
    "vrb": VolumeRunBars,
}


def build_for(bar_type: str, asset: str, threshold_value: float) -> BarBuilder:
    """Factory: bar_type string -> BarBuilder instance.

    Raises ValueError on unknown bar_type.
    """
    try:
        cls = _BUILDER_BY_TYPE[bar_type]
    except KeyError as e:
        raise ValueError(
            f"unknown bar_type {bar_type!r}; "
            f"expected one of {sorted(_BUILDER_BY_TYPE)}"
        ) from e
    return cls(asset=asset, threshold_value=threshold_value)
