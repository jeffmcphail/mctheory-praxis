"""
Universal _bpk / _base_id / _hist_id Infrastructure
=====================================================
McTheory Praxis — Phase 1, Deliverable 1.1

Spec Reference: §2.2

Every dimension table in the system follows the same pattern. No exceptions.
This module provides the key generation utilities used by ALL tables.

Key semantics:
    _bpk:     VARCHAR   Human-readable business primary key (IMMUTABLE once set)
    _base_id: BIGINT    xxHash64 of _bpk — join-optimized (derived from _bpk)
    _hist_id: TIMESTAMP Record creation timestamp — IS the primary key AND creation time
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import xxhash


def generate_base_id(bpk: str) -> int:
    """Generate deterministic base_id from business primary key using xxHash64.
    
    xxHash64 returns uint64 (0 to 2^64-1), but DuckDB BIGINT is int64 (-2^63 to 2^63-1).
    We interpret the unsigned value as signed to fit DuckDB's BIGINT.
    
    Spike 1 finding: This conversion is mandatory — raw xxhash intdigest() values
    exceed BIGINT range for ~50% of inputs.
    
    Args:
        bpk: Business primary key string (e.g., 'EQUITY|ISIN|US0378331005')
        
    Returns:
        Signed 64-bit integer suitable for DuckDB BIGINT storage.
        
    Raises:
        ValueError: If bpk is empty or None.
    """
    if not bpk:
        raise ValueError("Business primary key (bpk) cannot be empty or None")
    
    unsigned = xxhash.xxh64(bpk.encode("utf-8")).intdigest()
    
    # Convert uint64 → int64: values >= 2^63 become negative
    if unsigned >= (1 << 63):
        return unsigned - (1 << 64)
    return unsigned


def generate_hist_id(timestamp: Optional[datetime] = None) -> datetime:
    """Generate a hist_id timestamp for a new record version.
    
    The hist_id IS the record creation timestamp. It serves double duty:
    primary key AND 'when was this record created.' There is NO separate
    created_timestamp column on any dimension table.
    
    Uses a monotonic counter to guarantee uniqueness even when
    multiple records are created within the same clock tick
    (common on Windows where timer resolution is ~15.6ms).
    
    Args:
        timestamp: Explicit timestamp (for testing/replay). 
                   If None, uses current UTC time with microsecond precision.
                   
    Returns:
        UTC datetime with microsecond precision, guaranteed unique.
    """
    if timestamp is not None:
        # Ensure UTC
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)
    
    now = datetime.now(timezone.utc)
    
    # Ensure monotonically increasing: if clock hasn't advanced since
    # last call, bump by 1 microsecond from the previous value.
    last = getattr(generate_hist_id, "_last", None)
    if last is not None and now <= last:
        now = last + timedelta(microseconds=1)
    generate_hist_id._last = now
    
    return now


def validate_bpk(bpk: str, entity_type: str = "") -> bool:
    """Validate business primary key format.
    
    BPK format varies by entity type but must always be non-empty.
    Known patterns:
        dim_security:              '{sec_type}|{secid_type}|{secid_value}'
        dim_model_definition:      '{model_name}|{version}'
        dim_exchange:              '{exchange_code}'
        dim_signal_type:           '{signal_method_name}'
        dim_strategy_type:         '{strategy_family}'
        dim_index:                 '{index_code}'
        dim_calendar:              '{exchange}|{date}'
        dim_security_exchange:     '{security_base_id}|{exchange_code}'
        
    Args:
        bpk: Business primary key to validate.
        entity_type: Optional entity type for format-specific validation.
        
    Returns:
        True if valid.
        
    Raises:
        ValueError: If bpk is invalid, with descriptive message.
    """
    if not bpk or not isinstance(bpk, str):
        raise ValueError(f"BPK must be a non-empty string, got: {bpk!r}")
    
    if bpk != bpk.strip():
        raise ValueError(f"BPK must not have leading/trailing whitespace: {bpk!r}")
    
    # Entity-specific validation
    if entity_type == "dim_security":
        parts = bpk.split("|")
        if len(parts) != 3:
            raise ValueError(
                f"Security BPK must be 'sec_type|secid_type|secid_value', "
                f"got {len(parts)} parts: {bpk!r}"
            )
        sec_type, secid_type, secid_value = parts
        
        valid_sec_types = {
            "EQUITY", "BOND", "ETF", "FUTURE", "OPTION", 
            "CRYPTO", "FX", "INDEX"
        }
        if sec_type not in valid_sec_types:
            raise ValueError(
                f"Unknown sec_type '{sec_type}' in BPK. "
                f"Valid types: {sorted(valid_sec_types)}"
            )
        
        if not secid_value:
            raise ValueError(f"secid_value cannot be empty in BPK: {bpk!r}")
    
    elif entity_type == "dim_model_definition":
        parts = bpk.split("|")
        if len(parts) != 2:
            raise ValueError(
                f"Model definition BPK must be 'model_name|version', "
                f"got {len(parts)} parts: {bpk!r}"
            )
    
    return True


class EntityKeys:
    """Immutable key triple for a dimension entity.
    
    Once created, the bpk and base_id are frozen. hist_id is set at creation
    and represents this specific version of the entity.
    
    Usage:
        keys = EntityKeys.create("EQUITY|ISIN|US0378331005")
        # keys.bpk = 'EQUITY|ISIN|US0378331005'
        # keys.base_id = xxhash64 of bpk (signed)
        # keys.hist_id = current UTC timestamp
    """
    
    __slots__ = ("_bpk", "_base_id", "_hist_id")
    
    def __init__(self, bpk: str, base_id: int, hist_id: datetime):
        object.__setattr__(self, "_bpk", bpk)
        object.__setattr__(self, "_base_id", base_id)
        object.__setattr__(self, "_hist_id", hist_id)
    
    def __setattr__(self, name, value):
        raise AttributeError("EntityKeys are immutable after creation")
    
    @classmethod
    def create(
        cls, 
        bpk: str, 
        entity_type: str = "",
        timestamp: Optional[datetime] = None
    ) -> "EntityKeys":
        """Create a new EntityKeys triple.
        
        Args:
            bpk: Business primary key.
            entity_type: Optional entity type for bpk format validation.
            timestamp: Optional explicit timestamp (for testing/replay).
            
        Returns:
            Immutable EntityKeys with bpk, base_id, and hist_id set.
        """
        validate_bpk(bpk, entity_type)
        base_id = generate_base_id(bpk)
        hist_id = generate_hist_id(timestamp)
        return cls(bpk, base_id, hist_id)
    
    @classmethod
    def new_version(
        cls,
        bpk: str,
        base_id: int,
        timestamp: Optional[datetime] = None
    ) -> "EntityKeys":
        """Create keys for a new version of an existing entity.
        
        Same bpk and base_id, new hist_id. Used when updating a dimension
        record (inserting a new version row).
        
        Args:
            bpk: Existing business primary key.
            base_id: Existing base_id (must match xxhash64 of bpk).
            timestamp: Optional explicit timestamp.
            
        Returns:
            EntityKeys with same bpk/base_id, new hist_id.
            
        Raises:
            ValueError: If base_id doesn't match regenerated hash of bpk.
        """
        expected_base_id = generate_base_id(bpk)
        if base_id != expected_base_id:
            raise ValueError(
                f"base_id mismatch: provided {base_id}, "
                f"but xxhash64('{bpk}') = {expected_base_id}. "
                f"base_id must always be derived from bpk."
            )
        hist_id = generate_hist_id(timestamp)
        return cls(bpk, base_id, hist_id)
    
    @property
    def bpk(self) -> str:
        return self._bpk
    
    @property
    def base_id(self) -> int:
        return self._base_id
    
    @property
    def hist_id(self) -> datetime:
        return self._hist_id
    
    def to_dict(self) -> dict:
        """Return keys as a dict for DuckDB insertion."""
        return {
            "bpk": self._bpk,
            "base_id": self._base_id,
            "hist_id": self._hist_id,
        }
    
    def to_tuple(self) -> tuple:
        """Return keys as (hist_id, bpk, base_id) tuple — DuckDB column order."""
        return (self._hist_id, self._bpk, self._base_id)
    
    def __repr__(self) -> str:
        return (
            f"EntityKeys(bpk={self._bpk!r}, "
            f"base_id={self._base_id}, "
            f"hist_id={self._hist_id.isoformat()})"
        )
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, EntityKeys):
            return NotImplemented
        return (
            self._bpk == other._bpk 
            and self._base_id == other._base_id 
            and self._hist_id == other._hist_id
        )
    
    def __hash__(self) -> int:
        return hash((self._bpk, self._base_id, self._hist_id))


# ---------------------------------------------------------------------------
# Security-specific BPK construction
# ---------------------------------------------------------------------------

# SecIdType hierarchy per spec §3.3 — append-only, never reorder
SECID_HIERARCHY: dict[str, list[str]] = {
    "EQUITY":  ["ISIN", "CUSIP", "FIGI", "SEDOL", "TICKER"],
    "BOND":    ["ISIN", "CUSIP", "FIGI"],
    "CRYPTO":  ["SYMBOL", "CONTRACT_ADDRESS"],
    "FX":      ["ISO_PAIR"],
    "FUTURE":  ["EXCHANGE_CODE", "ISIN"],
    "OPTION":  ["OCC_SYMBOL", "ISIN"],
    "ETF":     ["ISIN", "CUSIP", "FIGI", "TICKER"],
    "INDEX":   ["INDEX_CODE", "ISIN"],
}


def build_security_bpk(sec_type: str, identifiers: dict[str, str]) -> str:
    """Build a security BPK from sec_type and available identifiers.
    
    Walks the SecIdType hierarchy for the given sec_type and uses the
    first available identifier to form the BPK.
    
    Per spec §3.3: 'The hierarchy is append-only. You can add new SecIdTypes 
    at the end, but you can never reorder existing ones.'
    
    Args:
        sec_type: Security type (EQUITY, BOND, CRYPTO, etc.)
        identifiers: Available identifiers {secid_type: value}
        
    Returns:
        BPK string: '{sec_type}|{secid_type}|{secid_value}'
        
    Raises:
        ValueError: If sec_type unknown or no identifier matches hierarchy.
    """
    sec_type = sec_type.upper()
    
    if sec_type not in SECID_HIERARCHY:
        raise ValueError(
            f"Unknown sec_type: '{sec_type}'. "
            f"Valid types: {sorted(SECID_HIERARCHY.keys())}"
        )
    
    # Normalize identifier keys to uppercase
    normalized = {k.upper(): v for k, v in identifiers.items()}
    
    hierarchy = SECID_HIERARCHY[sec_type]
    for secid_type in hierarchy:
        if secid_type in normalized and normalized[secid_type]:
            return f"{sec_type}|{secid_type}|{normalized[secid_type]}"
    
    raise ValueError(
        f"No matching identifier for {sec_type}. "
        f"Need one of: {hierarchy}. "
        f"Got: {list(identifiers.keys())}"
    )
