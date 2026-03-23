"""
Market data schema — DataTable definitions for the quant domain.

Tables:
    securities    — Security master (ticker, name, asset_class, exchange)
    prices        — OHLCV time series (security_id, date, open, high, low, close, volume, adj_close)
    universes     — Named universe definitions (universe_id, description)
    universe_members — Universe membership (universe_id, security_id)

All tables start empty. Data is fetched lazily via has_row(fill_missing=True)
when the DataView dependency cascade triggers during collect().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import polars as pl

if TYPE_CHECKING:
    from mctheory.core.datastore import DataStore

# ─────────────────────────────────────────────────────────────────────────────
# Schema definitions: (table_name, columns_schema, primary_key, filter_column)
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_TABLES = {
    "securities": {
        "schema": {
            "security_id": pl.Utf8,
            "name": pl.Utf8,
            "asset_class": pl.Utf8,       # equity, futures, fx, crypto, fixed_income, commodity
            "exchange": pl.Utf8,
            "currency": pl.Utf8,
            "sector": pl.Utf8,
            "industry": pl.Utf8,
        },
        "primary_key": ["security_id"],
        "filter_column": "security_id",
    },
    "prices": {
        "schema": {
            "security_id": pl.Utf8,
            "date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "adj_close": pl.Float64,
        },
        "primary_key": ["security_id", "date"],
        "filter_column": "security_id",   # fetch by ticker, get all dates
    },
    "universes": {
        "schema": {
            "universe_id": pl.Utf8,
            "name": pl.Utf8,
            "description": pl.Utf8,
        },
        "primary_key": ["universe_id"],
        "filter_column": "universe_id",
    },
    "universe_members": {
        "schema": {
            "universe_id": pl.Utf8,
            "security_id": pl.Utf8,
        },
        "primary_key": ["universe_id", "security_id"],
        "filter_column": "universe_id",   # fetch by universe, get all members
    },
}


def _create_empty_table(table_name: str, table_def: dict) -> pl.DataFrame:
    """Create an empty DataFrame with the correct schema."""
    return pl.DataFrame(schema=table_def["schema"])


def create_market_datastore(
    fetcher: Callable[[str, str, Any], pl.DataFrame | None] | None = None,
    datastore_class: type | None = None,
) -> "DataStore":
    """
    Create and return a DataStore populated with empty market data tables.

    Tables are registered with the given fetcher callback so that
    has_row(fill_missing=True) triggers on-demand data loading.

    Args:
        fetcher: Callback matching DataTable.set_fetch_callback signature:
                 (table_name, filter_column, filter_value) -> DataFrame | None
                 If None, tables start empty with no auto-fetch.
        datastore_class: DataStore class to use (for testing/injection).
                         Defaults to mctheory.core.datastore.DataStore.

    Returns:
        Configured DataStore with all market data tables registered.
    """
    from mctheory.core.datastore import DataStore, DataTable

    if datastore_class is not None:
        ds = datastore_class.get_instance()
    else:
        DataStore.reset_instance()
        ds = DataStore.get_instance()

    for table_name, table_def in SCHEMA_TABLES.items():
        empty_df = _create_empty_table(table_name, table_def)
        table = DataTable(
            data=empty_df,
            name=table_name,
            primary_key=table_def["primary_key"],
        )
        # Set the column used for fetch lookups (may differ from full PK)
        table.set_filter_column(table_def["filter_column"])

        if fetcher is not None:
            table.set_fetch_callback(fetcher)

        ds.register_table(table_name, table)

    return ds
