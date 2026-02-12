"""
SecIdType Hierarchies (§3.3).

The preferred identifier depends on the security type.
This hierarchy determines which SecIdType is used to form the security_bpk.

The hierarchy is APPEND-ONLY: you can add new SecIdTypes at the end,
but you can never reorder existing ones.
"""

from __future__ import annotations

from enum import Enum


class SecType(str, Enum):
    """Security types supported by the platform."""
    EQUITY = "EQUITY"
    BOND = "BOND"
    ETF = "ETF"
    CRYPTO = "CRYPTO"
    FX = "FX"
    FUTURE = "FUTURE"
    OPTION = "OPTION"
    INDEX = "INDEX"


class SecIdType(str, Enum):
    """Identifier types in the hierarchy."""
    ISIN = "ISIN"
    CUSIP = "CUSIP"
    FIGI = "FIGI"
    SEDOL = "SEDOL"
    TICKER = "TICKER"
    RIC = "RIC"
    BLOOMBERG_ID = "BLOOMBERG_ID"
    PERMID = "PERMID"
    CIK = "CIK"
    LEI = "LEI"
    CONTRACT_ADDRESS = "CONTRACT_ADDRESS"
    SYMBOL = "SYMBOL"
    OCC_SYMBOL = "OCC_SYMBOL"
    EXCHANGE_CODE = "EXCHANGE_CODE"
    ISO_PAIR = "ISO_PAIR"
    INDEX_CODE = "INDEX_CODE"


# ═══════════════════════════════════════════════════════════════════
#  §3.3 SecIdType Hierarchies — append-only, never reorder
# ═══════════════════════════════════════════════════════════════════

SECID_HIERARCHY: dict[SecType, list[SecIdType]] = {
    SecType.EQUITY:  [SecIdType.ISIN, SecIdType.CUSIP, SecIdType.FIGI, SecIdType.SEDOL, SecIdType.TICKER],
    SecType.BOND:    [SecIdType.ISIN, SecIdType.CUSIP, SecIdType.FIGI],
    SecType.ETF:     [SecIdType.ISIN, SecIdType.CUSIP, SecIdType.FIGI, SecIdType.TICKER],
    SecType.CRYPTO:  [SecIdType.SYMBOL, SecIdType.CONTRACT_ADDRESS],
    SecType.FX:      [SecIdType.ISO_PAIR],
    SecType.FUTURE:  [SecIdType.EXCHANGE_CODE, SecIdType.ISIN],
    SecType.OPTION:  [SecIdType.OCC_SYMBOL, SecIdType.ISIN],
    SecType.INDEX:   [SecIdType.INDEX_CODE, SecIdType.ISIN],
}


# Map SecIdType to dim_security column name
SECID_TO_COLUMN: dict[SecIdType, str] = {
    SecIdType.ISIN: "isin",
    SecIdType.CUSIP: "cusip",
    SecIdType.FIGI: "figi",
    SecIdType.SEDOL: "sedol",
    SecIdType.TICKER: "ticker",
    SecIdType.RIC: "ric",
    SecIdType.BLOOMBERG_ID: "bloomberg_id",
    SecIdType.PERMID: "permid",
    SecIdType.CIK: "cik",
    SecIdType.LEI: "lei",
    SecIdType.CONTRACT_ADDRESS: "contract_address",
    SecIdType.SYMBOL: "symbol",
    SecIdType.OCC_SYMBOL: "occ_symbol",
    SecIdType.EXCHANGE_CODE: "exchange_code",
    SecIdType.ISO_PAIR: "symbol",       # FX pairs stored in symbol column
    SecIdType.INDEX_CODE: "symbol",     # Index codes stored in symbol column
}


def get_preferred_bpk(sec_type: str | SecType, identifiers: dict[str, str]) -> tuple[str, SecIdType]:
    """
    Walk the hierarchy and return (security_bpk, preferred_secid_type).

    Args:
        sec_type: Security type (string or SecType enum).
        identifiers: {SecIdType_name: value} e.g., {"TICKER": "AAPL", "ISIN": "US0378331005"}

    Returns:
        (security_bpk, SecIdType) e.g., ("EQUITY|ISIN|US0378331005", SecIdType.ISIN)

    Raises:
        ValueError: If sec_type unknown or no valid identifier in hierarchy.
    """
    if isinstance(sec_type, str):
        sec_type = SecType(sec_type)

    hierarchy = SECID_HIERARCHY.get(sec_type)
    if hierarchy is None:
        raise ValueError(f"Unknown sec_type: {sec_type}. Valid: {[s.value for s in SecType]}")

    # Normalize identifier keys to uppercase
    norm = {k.upper(): v for k, v in identifiers.items()}

    for secid_type in hierarchy:
        value = norm.get(secid_type.value)
        if value:
            bpk = f"{sec_type.value}|{secid_type.value}|{value}"
            return bpk, secid_type

    raise ValueError(
        f"No valid identifier for {sec_type.value}. "
        f"Need one of: {[s.value for s in hierarchy]}. "
        f"Got: {list(identifiers.keys())}"
    )
