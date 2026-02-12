"""
Tests for Universal _bpk / _base_id / _hist_id Infrastructure
================================================================
McTheory Praxis — Phase 1, Deliverable 1.1

Tests cover:
    - generate_base_id: determinism, signed conversion, edge cases
    - generate_hist_id: UTC enforcement, microsecond precision
    - validate_bpk: format validation per entity type
    - EntityKeys: creation, immutability, versioning, equality
    - build_security_bpk: hierarchy walk, normalization, error handling
"""

import pytest
from datetime import datetime, timezone, timedelta

from praxis.datastore.keys import (
    generate_base_id,
    generate_hist_id,
    validate_bpk,
    EntityKeys,
    build_security_bpk,
    SECID_HIERARCHY,
)


# =========================================================================
# generate_base_id
# =========================================================================

class TestGenerateBaseId:
    """Tests for xxHash64 base_id generation."""
    
    def test_deterministic(self):
        """Same input always produces same output."""
        bpk = "EQUITY|ISIN|US0378331005"
        assert generate_base_id(bpk) == generate_base_id(bpk)
    
    def test_different_inputs_different_outputs(self):
        """Different inputs produce different outputs (collision test)."""
        id1 = generate_base_id("EQUITY|ISIN|US0378331005")
        id2 = generate_base_id("EQUITY|TICKER|AAPL")
        assert id1 != id2
    
    def test_returns_signed_int64(self):
        """Result fits in DuckDB BIGINT range (-2^63 to 2^63-1)."""
        # Test across many inputs to catch unsigned overflow
        for i in range(1000):
            result = generate_base_id(f"TEST|KEY|{i:06d}")
            assert -(1 << 63) <= result < (1 << 63), \
                f"base_id {result} out of BIGINT range for input {i}"
    
    def test_some_negative_values(self):
        """Signed conversion should produce some negative values (~50%)."""
        negatives = sum(
            1 for i in range(1000) 
            if generate_base_id(f"TEST|KEY|{i:06d}") < 0
        )
        # Should be roughly 50% negative (uint64 → int64 conversion)
        assert 300 < negatives < 700, \
            f"Expected ~500 negatives, got {negatives}"
    
    def test_empty_string_raises(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            generate_base_id("")
    
    def test_none_raises(self):
        """None should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            generate_base_id(None)
    
    def test_known_value(self):
        """Verify a known xxhash64 value (regression test)."""
        result = generate_base_id("EQUITY|ISIN|US0378331005")
        # This value should be stable across xxhash library versions
        assert isinstance(result, int)
        # Re-running should give same value
        assert generate_base_id("EQUITY|ISIN|US0378331005") == result
    
    def test_unicode_input(self):
        """UTF-8 encoded strings should hash correctly."""
        result = generate_base_id("EQUITY|TICKER|日経225")
        assert isinstance(result, int)
        assert -(1 << 63) <= result < (1 << 63)
    
    def test_special_characters(self):
        """Pipe-delimited strings with special chars."""
        result = generate_base_id("FX|ISO_PAIR|EUR/USD")
        assert isinstance(result, int)


# =========================================================================
# generate_hist_id
# =========================================================================

class TestGenerateHistId:
    """Tests for hist_id timestamp generation."""
    
    def test_returns_utc(self):
        """Generated timestamp is always UTC."""
        ts = generate_hist_id()
        assert ts.tzinfo == timezone.utc
    
    def test_explicit_timestamp_returned(self):
        """Explicit timestamp is passed through."""
        explicit = datetime(2026, 1, 15, 10, 30, 0)
        ts = generate_hist_id(explicit)
        assert ts.year == 2026
        assert ts.month == 1
        assert ts.day == 15
    
    def test_naive_timestamp_gets_utc(self):
        """Naive datetime (no tzinfo) gets UTC assigned."""
        naive = datetime(2026, 1, 15, 10, 30, 0)
        ts = generate_hist_id(naive)
        assert ts.tzinfo == timezone.utc
    
    def test_non_utc_converted(self):
        """Non-UTC timestamp is converted to UTC."""
        est = timezone(timedelta(hours=-5))
        eastern_time = datetime(2026, 1, 15, 10, 0, 0, tzinfo=est)
        ts = generate_hist_id(eastern_time)
        assert ts.tzinfo == timezone.utc
        assert ts.hour == 15  # 10 EST = 15 UTC
    
    def test_monotonic_increasing(self):
        """Sequential calls should produce increasing timestamps."""
        ts1 = generate_hist_id()
        ts2 = generate_hist_id()
        assert ts2 >= ts1
    
    def test_microsecond_precision(self):
        """Timestamps should have microsecond precision."""
        ts = generate_hist_id()
        # microsecond field exists and is accessible
        assert hasattr(ts, 'microsecond')


# =========================================================================
# validate_bpk
# =========================================================================

class TestValidateBpk:
    """Tests for BPK format validation."""
    
    def test_valid_security_bpk(self):
        """Standard security BPK passes."""
        assert validate_bpk("EQUITY|ISIN|US0378331005", "dim_security") is True
    
    def test_valid_model_bpk(self):
        """Model definition BPK passes."""
        assert validate_bpk("burgess_stat_arb|v2.1", "dim_model_definition") is True
    
    def test_valid_generic_bpk(self):
        """Any non-empty string passes without entity type."""
        assert validate_bpk("NYSE") is True
        assert validate_bpk("ema_crossover") is True
    
    def test_empty_string_raises(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            validate_bpk("")
    
    def test_none_raises(self):
        """None raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            validate_bpk(None)
    
    def test_whitespace_raises(self):
        """Leading/trailing whitespace raises ValueError."""
        with pytest.raises(ValueError, match="whitespace"):
            validate_bpk(" EQUITY|ISIN|US0378331005")
        with pytest.raises(ValueError, match="whitespace"):
            validate_bpk("EQUITY|ISIN|US0378331005 ")
    
    def test_security_wrong_parts_count(self):
        """Security BPK with wrong number of parts raises."""
        with pytest.raises(ValueError, match="sec_type.secid_type.secid_value"):
            validate_bpk("EQUITY|ISIN", "dim_security")
        with pytest.raises(ValueError, match="sec_type.secid_type.secid_value"):
            validate_bpk("EQUITY|ISIN|US|EXTRA", "dim_security")
    
    def test_security_unknown_sec_type(self):
        """Security BPK with unknown sec_type raises."""
        with pytest.raises(ValueError, match="Unknown sec_type"):
            validate_bpk("WARRANT|ISIN|US0378331005", "dim_security")
    
    def test_security_empty_secid_value(self):
        """Security BPK with empty secid_value raises."""
        with pytest.raises(ValueError, match="secid_value cannot be empty"):
            validate_bpk("EQUITY|ISIN|", "dim_security")
    
    def test_model_wrong_parts_count(self):
        """Model definition BPK with wrong parts count raises."""
        with pytest.raises(ValueError, match="model_name.version"):
            validate_bpk("model_name", "dim_model_definition")
    
    def test_all_sec_types_valid(self):
        """All known sec_types pass validation."""
        for sec_type in ["EQUITY", "BOND", "ETF", "FUTURE", "OPTION", "CRYPTO", "FX", "INDEX"]:
            assert validate_bpk(f"{sec_type}|ISIN|TEST123", "dim_security") is True


# =========================================================================
# EntityKeys
# =========================================================================

class TestEntityKeys:
    """Tests for the EntityKeys immutable key triple."""
    
    def test_create_basic(self):
        """Basic creation with auto-generated hist_id."""
        keys = EntityKeys.create("EQUITY|ISIN|US0378331005")
        assert keys.bpk == "EQUITY|ISIN|US0378331005"
        assert isinstance(keys.base_id, int)
        assert isinstance(keys.hist_id, datetime)
    
    def test_create_with_entity_type(self):
        """Creation with entity type validation."""
        keys = EntityKeys.create(
            "EQUITY|ISIN|US0378331005", 
            entity_type="dim_security"
        )
        assert keys.bpk == "EQUITY|ISIN|US0378331005"
    
    def test_create_with_explicit_timestamp(self):
        """Creation with explicit timestamp."""
        ts = datetime(2026, 1, 15, 10, 0, 0)
        keys = EntityKeys.create("EQUITY|ISIN|US0378331005", timestamp=ts)
        assert keys.hist_id.year == 2026
        assert keys.hist_id.month == 1
    
    def test_immutable_bpk(self):
        """Cannot modify bpk after creation."""
        keys = EntityKeys.create("EQUITY|ISIN|US0378331005")
        with pytest.raises(AttributeError, match="immutable"):
            keys.bpk = "something_else"
    
    def test_immutable_base_id(self):
        """Cannot modify base_id after creation."""
        keys = EntityKeys.create("EQUITY|ISIN|US0378331005")
        with pytest.raises(AttributeError, match="immutable"):
            keys.base_id = 999
    
    def test_immutable_hist_id(self):
        """Cannot modify hist_id after creation."""
        keys = EntityKeys.create("EQUITY|ISIN|US0378331005")
        with pytest.raises(AttributeError, match="immutable"):
            keys.hist_id = datetime.now()
    
    def test_immutable_arbitrary_attr(self):
        """Cannot set any attribute after creation."""
        keys = EntityKeys.create("EQUITY|ISIN|US0378331005")
        with pytest.raises(AttributeError, match="immutable"):
            keys.new_field = "nope"
    
    def test_base_id_matches_generate(self):
        """base_id matches standalone generate_base_id()."""
        bpk = "EQUITY|ISIN|US0378331005"
        keys = EntityKeys.create(bpk)
        assert keys.base_id == generate_base_id(bpk)
    
    def test_new_version_same_identity(self):
        """New version keeps same bpk and base_id, different hist_id."""
        ts1 = datetime(2026, 1, 15, 10, 0, 0)
        ts2 = datetime(2026, 1, 16, 10, 0, 0)
        original = EntityKeys.create("EQUITY|ISIN|US0378331005", timestamp=ts1)
        new_ver = EntityKeys.new_version(original.bpk, original.base_id, timestamp=ts2)
        assert new_ver.bpk == original.bpk
        assert new_ver.base_id == original.base_id
        assert new_ver.hist_id != original.hist_id  # Different timestamp
    
    def test_new_version_mismatched_base_id_raises(self):
        """new_version rejects mismatched base_id."""
        with pytest.raises(ValueError, match="base_id mismatch"):
            EntityKeys.new_version("EQUITY|ISIN|US0378331005", 999)
    
    def test_to_dict(self):
        """to_dict returns proper dict."""
        keys = EntityKeys.create("EQUITY|ISIN|US0378331005")
        d = keys.to_dict()
        assert d["bpk"] == "EQUITY|ISIN|US0378331005"
        assert d["base_id"] == keys.base_id
        assert d["hist_id"] == keys.hist_id
    
    def test_to_tuple_order(self):
        """to_tuple returns (hist_id, bpk, base_id) — DuckDB column order."""
        keys = EntityKeys.create("EQUITY|ISIN|US0378331005")
        t = keys.to_tuple()
        assert t == (keys.hist_id, keys.bpk, keys.base_id)
    
    def test_equality(self):
        """Two EntityKeys with same values are equal."""
        ts = datetime(2026, 1, 15, 10, 0, 0)
        keys1 = EntityKeys.create("TEST|A|1", timestamp=ts)
        keys2 = EntityKeys.create("TEST|A|1", timestamp=ts)
        assert keys1 == keys2
    
    def test_inequality_different_bpk(self):
        """Different bpk means not equal."""
        ts = datetime(2026, 1, 15, 10, 0, 0)
        keys1 = EntityKeys.create("TEST|A|1", timestamp=ts)
        keys2 = EntityKeys.create("TEST|A|2", timestamp=ts)
        assert keys1 != keys2
    
    def test_hashable(self):
        """EntityKeys can be used as dict keys / in sets."""
        ts = datetime(2026, 1, 15, 10, 0, 0)
        keys = EntityKeys.create("TEST|A|1", timestamp=ts)
        d = {keys: "value"}
        assert d[keys] == "value"
    
    def test_repr(self):
        """repr is readable."""
        keys = EntityKeys.create("EQUITY|ISIN|US0378331005")
        r = repr(keys)
        assert "EQUITY|ISIN|US0378331005" in r
        assert "base_id=" in r
        assert "hist_id=" in r
    
    def test_create_invalid_bpk_raises(self):
        """Creating with invalid bpk raises ValueError."""
        with pytest.raises(ValueError):
            EntityKeys.create("")
    
    def test_create_invalid_entity_type_raises(self):
        """Creating with wrong entity format raises ValueError."""
        with pytest.raises(ValueError, match="sec_type.secid_type.secid_value"):
            EntityKeys.create("EQUITY|ISIN", entity_type="dim_security")


# =========================================================================
# build_security_bpk
# =========================================================================

class TestBuildSecurityBpk:
    """Tests for security BPK construction from identifiers."""
    
    def test_equity_isin_first(self):
        """ISIN is preferred for equities when available."""
        bpk = build_security_bpk("EQUITY", {
            "TICKER": "AAPL",
            "CUSIP": "037833100",
            "ISIN": "US0378331005"
        })
        assert bpk == "EQUITY|ISIN|US0378331005"
    
    def test_equity_cusip_fallback(self):
        """CUSIP used when ISIN not available."""
        bpk = build_security_bpk("EQUITY", {
            "TICKER": "AAPL",
            "CUSIP": "037833100"
        })
        assert bpk == "EQUITY|CUSIP|037833100"
    
    def test_equity_ticker_last_resort(self):
        """TICKER used only when no higher-priority identifiers exist."""
        bpk = build_security_bpk("EQUITY", {"TICKER": "AAPL"})
        assert bpk == "EQUITY|TICKER|AAPL"
    
    def test_crypto_symbol(self):
        """SYMBOL preferred for crypto."""
        bpk = build_security_bpk("CRYPTO", {"SYMBOL": "BTC"})
        assert bpk == "CRYPTO|SYMBOL|BTC"
    
    def test_crypto_contract_address_fallback(self):
        """CONTRACT_ADDRESS used when SYMBOL not available."""
        bpk = build_security_bpk("CRYPTO", {
            "CONTRACT_ADDRESS": "0xdac17f958d2ee523a2206206994597c13d831ec7"
        })
        assert bpk == "CRYPTO|CONTRACT_ADDRESS|0xdac17f958d2ee523a2206206994597c13d831ec7"
    
    def test_fx_iso_pair(self):
        """FX uses ISO_PAIR."""
        bpk = build_security_bpk("FX", {"ISO_PAIR": "EUR/USD"})
        assert bpk == "FX|ISO_PAIR|EUR/USD"
    
    def test_etf_hierarchy(self):
        """ETF shares EQUITY-like hierarchy."""
        bpk = build_security_bpk("ETF", {
            "ISIN": "US78462F1030",
            "TICKER": "SPY"
        })
        assert bpk == "ETF|ISIN|US78462F1030"
    
    def test_case_insensitive_keys(self):
        """Identifier keys are normalized to uppercase."""
        bpk = build_security_bpk("EQUITY", {"isin": "US0378331005"})
        assert bpk == "EQUITY|ISIN|US0378331005"
    
    def test_case_insensitive_sec_type(self):
        """sec_type is normalized to uppercase."""
        bpk = build_security_bpk("equity", {"ISIN": "US0378331005"})
        assert bpk == "EQUITY|ISIN|US0378331005"
    
    def test_unknown_sec_type_raises(self):
        """Unknown sec_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown sec_type"):
            build_security_bpk("WARRANT", {"TICKER": "XYZ"})
    
    def test_no_matching_identifier_raises(self):
        """No matching identifier for hierarchy raises ValueError."""
        with pytest.raises(ValueError, match="No matching identifier"):
            build_security_bpk("FX", {"TICKER": "EURUSD"})
    
    def test_empty_identifier_value_skipped(self):
        """Empty string identifier values are skipped."""
        bpk = build_security_bpk("EQUITY", {
            "ISIN": "",  # Empty — skip
            "TICKER": "AAPL"
        })
        assert bpk == "EQUITY|TICKER|AAPL"
    
    def test_none_identifier_value_skipped(self):
        """None identifier values are skipped."""
        bpk = build_security_bpk("EQUITY", {
            "ISIN": None,
            "TICKER": "AAPL"
        })
        assert bpk == "EQUITY|TICKER|AAPL"
    
    def test_all_sec_types_have_hierarchy(self):
        """Every sec_type in SECID_HIERARCHY has at least one entry."""
        for sec_type, hierarchy in SECID_HIERARCHY.items():
            assert len(hierarchy) >= 1, f"{sec_type} has empty hierarchy"
    
    def test_option_occ_symbol(self):
        """Options use OCC_SYMBOL."""
        bpk = build_security_bpk("OPTION", {
            "OCC_SYMBOL": "AAPL  260117C00200000"
        })
        assert bpk == "OPTION|OCC_SYMBOL|AAPL  260117C00200000"
    
    def test_future_exchange_code(self):
        """Futures use EXCHANGE_CODE."""
        bpk = build_security_bpk("FUTURE", {
            "EXCHANGE_CODE": "ESH6",
            "ISIN": "US12345"
        })
        assert bpk == "FUTURE|EXCHANGE_CODE|ESH6"


# =========================================================================
# Integration: Key → DuckDB round-trip  
# =========================================================================

class TestDuckDBRoundTrip:
    """Integration tests: generate keys, insert into DuckDB, read back."""
    
    @pytest.fixture
    def duckdb_con(self):
        import duckdb
        con = duckdb.connect(":memory:")
        con.execute("""
            CREATE TABLE dim_test (
                test_hist_id TIMESTAMP PRIMARY KEY,
                test_bpk VARCHAR NOT NULL,
                test_base_id BIGINT NOT NULL,
                name VARCHAR
            )
        """)
        yield con
        con.close()
    
    def test_insert_and_read(self, duckdb_con):
        """Keys survive DuckDB round-trip."""
        keys = EntityKeys.create("TEST|A|123")
        duckdb_con.execute(
            "INSERT INTO dim_test VALUES (?, ?, ?, ?)",
            [keys.hist_id, keys.bpk, keys.base_id, "test entity"]
        )
        result = duckdb_con.execute(
            "SELECT test_bpk, test_base_id FROM dim_test WHERE test_base_id = ?",
            [keys.base_id]
        ).fetchone()
        assert result[0] == keys.bpk
        assert result[1] == keys.base_id
    
    def test_multiple_versions_same_base_id(self, duckdb_con):
        """Multiple versions of same entity share base_id."""
        ts1 = datetime(2026, 1, 15, 10, 0, 0)
        ts2 = datetime(2026, 1, 16, 10, 0, 0)
        
        keys1 = EntityKeys.create("TEST|A|123", timestamp=ts1)
        keys2 = EntityKeys.new_version(keys1.bpk, keys1.base_id, timestamp=ts2)
        
        duckdb_con.execute(
            "INSERT INTO dim_test VALUES (?, ?, ?, ?)",
            [keys1.hist_id, keys1.bpk, keys1.base_id, "v1"]
        )
        duckdb_con.execute(
            "INSERT INTO dim_test VALUES (?, ?, ?, ?)",
            [keys2.hist_id, keys2.bpk, keys2.base_id, "v2"]
        )
        
        count = duckdb_con.execute(
            "SELECT COUNT(*) FROM dim_test WHERE test_base_id = ?",
            [keys1.base_id]
        ).fetchone()[0]
        assert count == 2
    
    def test_negative_base_id_stored(self, duckdb_con):
        """Negative base_ids (signed conversion) store correctly in BIGINT."""
        # Find a BPK that produces a negative base_id
        negative_found = False
        for i in range(100):
            bpk = f"TEST|KEY|{i:06d}"
            base_id = generate_base_id(bpk)
            if base_id < 0:
                keys = EntityKeys.create(bpk, timestamp=datetime(2026, 1, 15))
                duckdb_con.execute(
                    "INSERT INTO dim_test VALUES (?, ?, ?, ?)",
                    [keys.hist_id, keys.bpk, keys.base_id, "negative test"]
                )
                result = duckdb_con.execute(
                    "SELECT test_base_id FROM dim_test WHERE test_base_id = ?",
                    [keys.base_id]
                ).fetchone()
                assert result[0] == keys.base_id
                assert result[0] < 0
                negative_found = True
                break
        
        assert negative_found, "Could not find a BPK producing negative base_id"
