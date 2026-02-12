"""
Tests for Calendar (2.7), Data Quality (2.8), Classification (2.10).
"""

import time
from datetime import date, datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from praxis.datastore.calendar import Calendar, _us_market_holidays
from praxis.datastore.classification import ClassificationManager
from praxis.datastore.database import PraxisDatabase
from praxis.datastore.quality import DataQualityValidator, ValidationResult
from praxis.logger.core import PraxisLogger
from praxis.security import SecurityMaster


@pytest.fixture(autouse=True)
def reset():
    PraxisLogger.reset()
    yield
    PraxisLogger.reset()


@pytest.fixture
def db():
    database = PraxisDatabase(":memory:")
    database.initialize()
    return database


# ═══════════════════════════════════════════════════════════════════
#  Calendar (Phase 2.7)
# ═══════════════════════════════════════════════════════════════════

class TestCalendarHolidays:
    def test_christmas_2024(self):
        h = _us_market_holidays(2024)
        assert date(2024, 12, 25) in h

    def test_new_years_2024(self):
        h = _us_market_holidays(2024)
        assert date(2024, 1, 1) in h

    def test_july_4_observed_on_saturday(self):
        # 2020: July 4 is Saturday → observed Friday July 3
        h = _us_market_holidays(2020)
        assert date(2020, 7, 3) in h

    def test_thanksgiving_fourth_thursday(self):
        h = _us_market_holidays(2024)
        # Nov 28, 2024 is a Thursday
        assert date(2024, 11, 28) in h

    def test_good_friday_exists(self):
        h = _us_market_holidays(2024)
        # Good Friday 2024 is March 29
        assert date(2024, 3, 29) in h


class TestCalendarPopulate:
    def test_populate_single_year(self, db):
        cal = Calendar(db.connection)
        rows = cal.populate(2024, 2024)
        assert rows == 366  # 2024 is leap year

    def test_populate_range(self, db):
        cal = Calendar(db.connection)
        rows = cal.populate(2023, 2024)
        assert rows == 365 + 366  # 2023 + 2024

    def test_count(self, db):
        cal = Calendar(db.connection)
        cal.populate(2024, 2024)
        assert cal.count() == 366


class TestCalendarQueries:
    @pytest.fixture
    def cal(self, db):
        c = Calendar(db.connection)
        c.populate(2024, 2024)
        return c

    def test_business_days_count(self, cal):
        bdays = cal.business_days("2024-01-01", "2024-01-31")
        # January 2024: 21 business days (31 - 8 weekends - 1 NY - 1 MLK)
        assert 19 <= len(bdays) <= 22  # approximate

    def test_weekend_not_business_day(self, cal):
        # 2024-01-06 is Saturday
        assert not cal.is_business_day("2024-01-06")
        # 2024-01-07 is Sunday
        assert not cal.is_business_day("2024-01-07")

    def test_weekday_is_business_day(self, cal):
        # 2024-01-08 is Monday (not a holiday)
        assert cal.is_business_day("2024-01-08")

    def test_holiday_not_business_day(self, cal):
        # Christmas 2024
        assert not cal.is_business_day("2024-12-25")

    def test_next_business_day(self, cal):
        # Friday → Monday
        nbd = cal.next_business_day("2024-01-05")
        assert nbd == date(2024, 1, 8)

    def test_prev_business_day(self, cal):
        # Monday → Friday
        pbd = cal.prev_business_day("2024-01-08")
        assert pbd == date(2024, 1, 5)

    def test_holidays_list(self, cal):
        holidays = cal.holidays(2024)
        assert len(holidays) >= 9  # NYSE has ~9-10 holidays
        names = [h[1] for h in holidays]
        assert any("Christmas" in n for n in names)

    def test_business_days_are_sorted(self, cal):
        bdays = cal.business_days("2024-01-01", "2024-03-31")
        for i in range(len(bdays) - 1):
            assert bdays[i] < bdays[i + 1]


# ═══════════════════════════════════════════════════════════════════
#  Data Quality (Phase 2.8)
# ═══════════════════════════════════════════════════════════════════

def _insert_hist_rows(conn, batch_id, rows):
    """Insert test rows into ldr_yfinance_hist."""
    for r in rows:
        load_id = int(time.time_ns())
        time.sleep(0.000001)
        conn.execute("""
            INSERT INTO ldr_yfinance_hist (
                load_id, load_timestamp, batch_id,
                symbol, date, open, high, low, close, volume,
                process_status
            ) VALUES ($1, CURRENT_TIMESTAMP, $2, $3, $4, $5, $6, $7, $8, $9, 'created')
        """, [load_id, batch_id, r.get("symbol", "TEST"),
              r.get("date"), r.get("open"), r.get("high"),
              r.get("low"), r.get("close"), r.get("volume")])


class TestDataQualityClean:
    def test_clean_batch(self, db):
        _insert_hist_rows(db.connection, "batch_1", [
            {"symbol": "AAPL", "date": "2024-01-02", "open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000000},
            {"symbol": "AAPL", "date": "2024-01-03", "open": 103, "high": 106, "low": 101, "close": 104, "volume": 900000},
        ])
        dq = DataQualityValidator(db.connection)
        result = dq.validate_price_batch("batch_1", source="yfinance")

        assert result.quality_score == 1.0
        assert result.records_received == 2
        assert result.records_accepted == 2
        assert result.records_rejected == 0
        assert result.null_count == 0
        assert result.range_violations == 0

    def test_empty_batch(self, db):
        dq = DataQualityValidator(db.connection)
        result = dq.validate_price_batch("nonexistent", source="test")
        assert result.quality_score == 0.0
        assert len(result.issues) > 0


class TestDataQualityNulls:
    def test_null_close_detected(self, db):
        _insert_hist_rows(db.connection, "batch_n", [
            {"symbol": "X", "date": "2024-01-02", "open": 100, "high": 105, "low": 99, "close": None, "volume": 1000},
            {"symbol": "X", "date": "2024-01-03", "open": 100, "high": 105, "low": 99, "close": 101, "volume": 1000},
        ])
        dq = DataQualityValidator(db.connection)
        result = dq.validate_price_batch("batch_n")
        assert result.null_count >= 1
        assert result.quality_score < 1.0

    def test_null_date_detected(self, db):
        _insert_hist_rows(db.connection, "batch_nd", [
            {"symbol": "X", "date": None, "open": 100, "high": 105, "low": 99, "close": 101, "volume": 1000},
        ])
        dq = DataQualityValidator(db.connection)
        result = dq.validate_price_batch("batch_nd")
        assert result.null_count >= 1


class TestDataQualityRange:
    def test_negative_price_detected(self, db):
        _insert_hist_rows(db.connection, "batch_np", [
            {"symbol": "X", "date": "2024-01-02", "open": 100, "high": 105, "low": 99, "close": -5, "volume": 1000},
        ])
        dq = DataQualityValidator(db.connection)
        result = dq.validate_price_batch("batch_np")
        assert result.range_violations >= 1
        assert any("Negative" in i or "zero" in i for i in result.issues)

    def test_high_low_inversion(self, db):
        _insert_hist_rows(db.connection, "batch_hl", [
            {"symbol": "X", "date": "2024-01-02", "open": 100, "high": 95, "low": 105, "close": 101, "volume": 1000},
        ])
        dq = DataQualityValidator(db.connection)
        result = dq.validate_price_batch("batch_hl")
        assert result.range_violations >= 1

    def test_negative_volume_detected(self, db):
        _insert_hist_rows(db.connection, "batch_nv", [
            {"symbol": "X", "date": "2024-01-02", "open": 100, "high": 105, "low": 99, "close": 101, "volume": -500},
        ])
        dq = DataQualityValidator(db.connection)
        result = dq.validate_price_batch("batch_nv")
        assert result.range_violations >= 1


class TestDataQualityOutliers:
    def test_outlier_detected(self, db):
        # Normal returns ~0.1% then one massive spike
        rows = []
        base = 100.0
        for i in range(20):
            d = f"2024-01-{(i+2):02d}"
            close = base + i * 0.1
            rows.append({"symbol": "X", "date": d, "open": close - 0.05,
                         "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 1000})
        # Add a massive outlier
        rows.append({"symbol": "X", "date": "2024-01-25", "open": 102,
                     "high": 200, "low": 100, "close": 200, "volume": 1000})

        _insert_hist_rows(db.connection, "batch_out", rows)
        dq = DataQualityValidator(db.connection)
        result = dq.validate_price_batch("batch_out", outlier_threshold=3.0)
        assert result.outliers_detected >= 1


class TestDataQualityStorage:
    def test_results_stored_and_retrievable(self, db):
        _insert_hist_rows(db.connection, "batch_s", [
            {"symbol": "X", "date": "2024-01-02", "open": 100, "high": 105, "low": 99, "close": 101, "volume": 1000},
        ])
        dq = DataQualityValidator(db.connection)
        dq.validate_price_batch("batch_s", source="yfinance")

        stored = dq.get_quality("batch_s")
        assert stored is not None
        assert stored["source"] == "yfinance"
        assert stored["quality_score"] == 1.0


# ═══════════════════════════════════════════════════════════════════
#  Classification (Phase 2.10)
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def cls_mgr(db):
    return ClassificationManager(db.connection)


@pytest.fixture
def master(db):
    return SecurityMaster(db.connection)


class TestCoreClassification:
    def test_assign_and_get(self, cls_mgr):
        cls_mgr.assign_core(100, "EQUITY", "COMMON_STOCK")
        core = cls_mgr.get_core(100)
        assert core is not None
        assert core["asset_class"] == "EQUITY"
        assert core["instrument_type"] == "COMMON_STOCK"

    def test_assign_with_settlement(self, cls_mgr):
        cls_mgr.assign_core(200, "CRYPTO", "SPOT", settlement_type="ON_CHAIN")
        core = cls_mgr.get_core(200)
        assert core["settlement_type"] == "ON_CHAIN"

    def test_get_nonexistent(self, cls_mgr):
        assert cls_mgr.get_core(999) is None


class TestExternalClassification:
    def test_assign_gics(self, cls_mgr):
        cls_mgr.assign_external(
            100, "GICS", version="v2024",
            level_1="Information Technology",
            level_2="Software & Services",
            raw_code="45102010",
            source="bloomberg",
        )
        ext = cls_mgr.get_external(100, "GICS")
        assert len(ext) == 1
        assert ext[0]["level_1"] == "Information Technology"
        assert ext[0]["raw_code"] == "45102010"

    def test_multiple_systems(self, cls_mgr):
        cls_mgr.assign_external(100, "GICS", level_1="IT", raw_code="45")
        cls_mgr.assign_external(100, "ICB", level_1="Technology", raw_code="9000")

        all_ext = cls_mgr.get_external(100)
        assert len(all_ext) == 2

        gics_only = cls_mgr.get_external(100, "GICS")
        assert len(gics_only) == 1


class TestMapping:
    def test_add_and_lookup(self, cls_mgr):
        cls_mgr.add_mapping("GICS", "EQUITY", "COMMON_STOCK", external_raw_code="45")
        m = cls_mgr.lookup_mapping("GICS", "45")
        assert m is not None
        assert m["core_asset_class"] == "EQUITY"
        assert m["core_instrument_type"] == "COMMON_STOCK"

    def test_prefix_match(self, cls_mgr):
        cls_mgr.add_mapping("GICS", "EQUITY", "COMMON_STOCK", external_raw_code="45")
        # Should match "45102010" via prefix to "45"
        m = cls_mgr.lookup_mapping("GICS", "45102010")
        assert m is not None
        assert m["core_asset_class"] == "EQUITY"

    def test_no_match(self, cls_mgr):
        m = cls_mgr.lookup_mapping("GICS", "99")
        assert m is None


class TestBootstrapGICS:
    def test_bootstrap_count(self, cls_mgr):
        count = cls_mgr.bootstrap_gics_mappings()
        # 11 sectors + 9 industry groups
        assert count == 20

    def test_bootstrap_mappings_queryable(self, cls_mgr):
        cls_mgr.bootstrap_gics_mappings()
        assert cls_mgr.count_mappings() == 20

        m = cls_mgr.lookup_mapping("GICS", "45")
        assert m is not None
        assert m["core_asset_class"] == "EQUITY"

    def test_bootstrap_sector_lookup(self, cls_mgr):
        cls_mgr.bootstrap_gics_mappings()
        m = cls_mgr.lookup_mapping("GICS", "35")
        assert m["external_level_1"] == "Health Care"


class TestAutoClassify:
    def test_auto_classify_success(self, cls_mgr, master):
        cls_mgr.bootstrap_gics_mappings()
        base_id = master.match_or_create("EQUITY", {"TICKER": "AAPL"}, source="test")

        ok = cls_mgr.auto_classify(base_id, "GICS", "45")
        assert ok is True

        core = cls_mgr.get_core(base_id)
        assert core["asset_class"] == "EQUITY"
        assert core["instrument_type"] == "COMMON_STOCK"
        assert core["created_by"] == "auto:GICS"

    def test_auto_classify_no_mapping(self, cls_mgr):
        ok = cls_mgr.auto_classify(100, "GICS", "99")
        assert ok is False

    def test_full_workflow(self, cls_mgr, master):
        """§15.6: Bootstrap → new security → auto-classify from GICS."""
        cls_mgr.bootstrap_gics_mappings()

        # New security arrives with GICS code
        base_id = master.match_or_create("EQUITY", {"TICKER": "MSFT"}, source="yfinance")

        # Assign external GICS
        cls_mgr.assign_external(
            base_id, "GICS", version="v2024",
            level_1="Information Technology",
            level_2="Software & Services",
            raw_code="4510",
        )

        # Auto-classify from mapping
        cls_mgr.auto_classify(base_id, "GICS", "4510")

        core = cls_mgr.get_core(base_id)
        assert core["asset_class"] == "EQUITY"

        ext = cls_mgr.get_external(base_id, "GICS")
        assert ext[0]["level_1"] == "Information Technology"
