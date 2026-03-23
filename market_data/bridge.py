"""
Bridge: DataStore → Praxis DataProvider.

DataStoreDataProvider implements the Praxis DataProvider protocol
(fetch_prices, asset_names) backed by a core.datastore DataStore.

The engine sees a numpy array. The DataStore handles all fetching,
caching, and dependency resolution underneath. Zero I/O leaks
into engine or business code.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from mctheory.core.datastore import DataStore

from engines.context.model_context import UniverseSpec, TemporalSpec, Frequency


class DataStoreDataProvider:
    """
    Praxis DataProvider backed by core.datastore.

    Translates UniverseSpec + TemporalSpec into DataStore queries,
    returns numpy arrays that engines consume directly.

    The DataStore does all the heavy lifting:
    - prices table has_row(fill_missing=True) triggers fetch callbacks
    - Securities are auto-populated on first access
    - Subsequent calls hit cache

    Args:
        datastore: Configured DataStore with market data tables registered.
    """

    def __init__(self, datastore: "DataStore"):
        self._ds = datastore
        self._resolved_names: list[str] | None = None
        self._last_universe: UniverseSpec | None = None

    # ─────────────────────────────────────────────────────────────────────
    # Praxis DataProvider protocol
    # ─────────────────────────────────────────────────────────────────────

    def fetch_prices(
        self,
        universe: UniverseSpec,
        temporal: TemporalSpec,
    ) -> np.ndarray:
        """
        Fetch price matrix for the given universe and time window.

        Returns:
            np.ndarray of shape (n_dates, n_assets) with adjusted close prices.
            Dates are sorted ascending. Assets are in ticker order.
        """
        tickers = self._resolve_tickers(universe)
        self._resolved_names = tickers
        self._last_universe = universe

        if not tickers:
            raise ValueError(f"Universe '{universe.name}' resolved to zero tickers")

        # Ensure all securities and prices are in the DataStore.
        # This triggers the has_row → fetch_callback cascade.
        self._ensure_data_loaded(tickers)

        # Build date range
        start, end = self._resolve_dates(temporal)

        # Query prices from DataStore
        prices_table = self._ds.get_table("prices")
        all_prices = prices_table.data

        # Filter to our tickers and date range
        mask = (
            all_prices.lazy()
            .filter(pl.col("security_id").is_in(tickers))
            .filter(pl.col("date") >= start)
            .filter(pl.col("date") <= end)
            .select(["security_id", "date", "adj_close"])
            .sort("date")
            .collect()
        )

        if len(mask) == 0:
            raise ValueError(
                f"No price data found for {tickers} between {start} and {end}"
            )

        # Pivot to wide format: rows=dates, cols=tickers
        wide = mask.pivot(
            on="security_id",
            index="date",
            values="adj_close",
        ).sort("date")

        # Ensure column order matches tickers list
        # (some tickers might be missing if fetch failed)
        available = [t for t in tickers if t in wide.columns]
        if not available:
            raise ValueError(f"No price columns found. Available: {wide.columns}")

        matrix = wide.select(available).to_numpy()

        # Forward-fill any NaN gaps (weekends, holidays between sources)
        matrix = self._forward_fill(matrix)

        # Update resolved names to only what we actually got
        self._resolved_names = available

        return matrix

    def asset_names(self) -> list[str]:
        """Return ticker list from the most recent fetch_prices call."""
        return (self._resolved_names or []).copy()

    # ─────────────────────────────────────────────────────────────────────
    # Extended interface (beyond minimal DataProvider protocol)
    # ─────────────────────────────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        tickers: list[str],
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pl.DataFrame:
        """
        Fetch full OHLCV data as a Polars DataFrame.

        Useful for strategies that need intraday range (high, low)
        or volume data beyond what the numpy price matrix provides.
        """
        self._ensure_data_loaded(tickers)

        prices_table = self._ds.get_table("prices")
        lf = prices_table.data.lazy().filter(
            pl.col("security_id").is_in(tickers)
        )
        if start:
            lf = lf.filter(pl.col("date") >= self._to_date(start))
        if end:
            lf = lf.filter(pl.col("date") <= self._to_date(end))

        return lf.sort(["security_id", "date"]).collect()

    def get_security_info(self, ticker: str) -> dict[str, Any]:
        """Get security master info for a single ticker."""
        sec_table = self._ds.get_table("securities")
        sec_table.has_row((ticker,), fill_missing=True)
        row = sec_table.data.filter(pl.col("security_id") == ticker)
        if len(row) == 0:
            return {"security_id": ticker}
        return row.row(0, named=True)

    def get_universe_members(self, universe_id: str) -> list[str]:
        """Get tickers in a named universe."""
        members_table = self._ds.get_table("universe_members")
        members_table.has_row_filtered("universe_id", universe_id, fill_missing=True)
        rows = members_table.data.filter(pl.col("universe_id") == universe_id)
        return rows["security_id"].to_list()

    # ─────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────

    def _resolve_tickers(self, universe: UniverseSpec) -> list[str]:
        """
        Resolve a UniverseSpec to a concrete ticker list.

        If tickers are provided directly, use those.
        If a universe name is given, look it up in the universe_members table.
        Apply exclusions after resolution.
        """
        if len(universe.tickers) > 0:
            tickers = list(universe.tickers)
        elif universe.name:
            tickers = self.get_universe_members(universe.name)
        else:
            raise ValueError("UniverseSpec must have either tickers or name")

        # Apply exclusions
        if universe.exclusions:
            tickers = [t for t in tickers if t not in universe.exclusions]

        return tickers

    def _ensure_data_loaded(self, tickers: list[str]) -> None:
        """
        Ensure securities and prices are loaded for all tickers.

        This is where the DataStore's lazy-loading magic happens.
        has_row(fill_missing=True) triggers the fetch callback for any
        ticker not yet in cache. Subsequent calls are instant.
        """
        sec_table = self._ds.get_table("securities")
        prices_table = self._ds.get_table("prices")

        for ticker in tickers:
            # Ensure security master row exists
            sec_table.has_row_filtered("security_id", ticker, fill_missing=True)
            # Ensure price history exists
            prices_table.has_row_filtered("security_id", ticker, fill_missing=True)

    def _resolve_dates(self, temporal: TemporalSpec) -> tuple[date, date]:
        """Convert TemporalSpec to concrete start/end dates."""
        if temporal.end:
            end = self._to_date(temporal.end)
        else:
            end = date.today()

        if temporal.start:
            start = self._to_date(temporal.start)
        else:
            start = end - timedelta(days=int(temporal.lookback_days * 1.45))

        return start, end

    @staticmethod
    def _to_date(d: str | date) -> date:
        """Convert string or date to date."""
        if isinstance(d, date):
            return d
        from datetime import datetime
        return datetime.strptime(d, "%Y-%m-%d").date()

    @staticmethod
    def _forward_fill(matrix: np.ndarray) -> np.ndarray:
        """Forward-fill NaN values column-wise."""
        result = matrix.copy()
        for col in range(result.shape[1]):
            for row in range(1, result.shape[0]):
                if np.isnan(result[row, col]):
                    result[row, col] = result[row - 1, col]
        return result
