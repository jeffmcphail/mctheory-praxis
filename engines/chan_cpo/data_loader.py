"""Kibot minute-bar file loading for the Chan CPO replication.

Kibot delivers two distinct layouts for these symbols, and NEITHER one
carries both trade prices and quotes:

    OHLCV   (7 cols)  Date,Time,Open,High,Low,Close,Volume
    BIDASK  (10 cols) Date,Time,BidO,BidH,BidL,BidC,AskO,AskH,AskL,AskC

The BIDASK layout has no volume column and no trade price at all -- bid and
ask each get a full OHLC quartet.  Anything that needs a traded price *and* a
spread must join the two files on timestamp; see `load_pair_panel`.

Two spread columns are derived, and the distinction is load-bearing:
`spread`/`spread_bps` are SIGNED and go negative on the bars Kibot delivers
crossed; `cost_spread`/`cost_spread_bps` are floored at zero and are the only
ones a cost calculation may consume.  See `floor_spread_cost`.

Timestamps are US/Eastern wall clock as delivered.  They are kept tz-naive on
purpose: the only ambiguous wall-clock hour is the autumn DST repeat at 01:xx,
which is far outside any session we trade, and carrying naive ET keeps both
legs of the pair on one clock without DST localisation failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class KibotSchemaError(ValueError):
    """Raised when a file's column count does not match the expected schema.

    Deliberately fatal: silently mapping 10 delivered columns onto a 9-column
    schema would put ask prices into a volume field and go unnoticed.
    """


@dataclass(frozen=True)
class KibotSchema:
    """Column layout of a Kibot delivery file."""

    name: str
    columns: tuple[str, ...]
    date_format: str = "%m/%d/%Y"
    time_format: str = "%H:%M"
    timezone: str = "US/Eastern"

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    @property
    def datetime_format(self) -> str:
        return f"{self.date_format} {self.time_format}"


# Verified against the 2026-08 delivery for GLD/GDX (see Cycle 59 Phase 2).
DEFAULT_OHLCV_SCHEMA = KibotSchema(
    name="ohlcv",
    columns=("date", "time", "open", "high", "low", "close", "volume"),
)

DEFAULT_BIDASK_SCHEMA = KibotSchema(
    name="bidask",
    columns=(
        "date",
        "time",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    ),
)


@dataclass
class InspectResult:
    """Raw head of a file plus its observed column counts."""

    path: Path
    raw_lines: list[str]
    column_counts: list[int]

    @property
    def n_columns(self) -> int | None:
        """The single observed column count, or None if the head is ragged."""
        uniq = set(self.column_counts)
        return uniq.pop() if len(uniq) == 1 else None

    def matches(self, schema: KibotSchema) -> bool:
        return self.n_columns == schema.n_columns


def inspect_file(path: str | Path, n_lines: int = 5) -> InspectResult:
    """Read the first `n_lines` of a Kibot file without parsing it."""
    path = Path(path)
    raw: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for _ in range(n_lines):
            line = fh.readline()
            if not line:
                break
            raw.append(line.rstrip("\r\n"))
    return InspectResult(path, raw, [len(line.split(",")) for line in raw])


def _observed_columns(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    if not first:
        raise KibotSchemaError(f"{path} is empty")
    return len(first.rstrip("\r\n").split(","))


def floor_spread_cost(spread):
    """Floor a raw ask-minus-bid spread at zero for cost purposes.

    Kibot aggregates each bar's bid and ask quartets independently, so a small
    fraction of bars (~0.04% GLD / ~0.009% GDX, 71 and 51 of them inside RTH)
    come back CROSSED -- bid above ask.  Subtracting a crossed quote gives a
    NEGATIVE spread, and any cost model that consumes it raw is *paid* to
    trade on those bars: a sign error small enough to survive review and
    directional enough to manufacture edge.

    Every cost calculation must go through this floor.  `spread` and
    `spread_bps` stay signed on purpose -- the validator's crossed-bar checks
    key off that sign, so clipping at the source would hide the defect
    instead of neutralising it.

    Accepts a scalar, ndarray, or Series and preserves the input type.
    """
    return np.maximum(spread, 0.0)


def load_kibot(
    path: str | Path,
    schema: KibotSchema = DEFAULT_BIDASK_SCHEMA,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load a Kibot file into a timestamp-indexed frame.

    Raises KibotSchemaError if the delivered column count differs from
    `schema` rather than mis-mapping columns.
    """
    path = Path(path)
    observed = _observed_columns(path)
    if observed != schema.n_columns:
        raise KibotSchemaError(
            f"{path.name}: delivered {observed} columns but schema "
            f"'{schema.name}' expects {schema.n_columns} "
            f"({', '.join(schema.columns)}). Refusing to guess the mapping."
        )

    df = pd.read_csv(
        path,
        header=None,
        names=list(schema.columns),
        nrows=nrows,
        dtype={"date": str, "time": str},
    )
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"], format=schema.datetime_format
    )
    df = df.drop(columns=["date", "time"]).set_index("timestamp")

    if "bid_close" in df.columns:
        mid = (df["bid_close"] + df["ask_close"]) / 2.0
        df["mid_close"] = mid
        # Signed -- diagnostics only; goes negative on crossed bars by design.
        df["spread"] = df["ask_close"] - df["bid_close"]
        df["spread_bps"] = 1e4 * df["spread"] / mid
        # Floored -- the ONLY spread a cost calculation may consume.
        df["cost_spread"] = floor_spread_cost(df["spread"])
        df["cost_spread_bps"] = floor_spread_cost(df["spread_bps"])
    return df


def load_pair_panel(
    gld_path: str | Path,
    gdx_path: str | Path,
    schema: KibotSchema = DEFAULT_BIDASK_SCHEMA,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Inner-join GLD and GDX on timestamp into one column-prefixed panel."""
    gld = load_kibot(gld_path, schema, nrows=nrows).add_prefix("gld_")
    gdx = load_kibot(gdx_path, schema, nrows=nrows).add_prefix("gdx_")
    return gld.join(gdx, how="inner")
