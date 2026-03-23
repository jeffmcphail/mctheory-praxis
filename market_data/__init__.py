"""
Market Data Layer — bridges core.datastore to Praxis engines.

The market data layer provides:
- Schema definitions for quant domain tables (securities, prices, universes)
- Pluggable market data fetchers (yfinance, mock, CSV, etc.)
- DataStoreDataProvider: implements Praxis DataProvider protocol backed by DataStore

Business code never touches I/O. Engines receive numpy arrays.
All fetching, caching, and dependency resolution is invisible.

Example:
    >>> from market_data import create_market_datastore, DataStoreDataProvider
    >>> from market_data.fetchers import YFinanceFetcher
    >>>
    >>> ds = create_market_datastore(fetcher=YFinanceFetcher())
    >>> provider = DataStoreDataProvider(ds)
    >>> prices = provider.fetch_prices(universe, temporal)  # numpy array
"""

from .schema import create_market_datastore, SCHEMA_TABLES
from .bridge import DataStoreDataProvider
from .universe import (
    SP500MembershipProvider,
    IndexMembershipProvider,
    normalize_ticker,
)

__all__ = [
    "create_market_datastore",
    "DataStoreDataProvider",
    "SCHEMA_TABLES",
    "SP500MembershipProvider",
    "IndexMembershipProvider",
    "normalize_ticker",
]
