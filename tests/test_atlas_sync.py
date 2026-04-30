"""Tests for engines/atlas_sync.py.

These tests run the parser against the real Atlas markdown files at the
repo root. They do NOT call the embedding API -- the sync writes a temp
DB with --no-embed semantics so tests are deterministic and free.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engines.atlas_sync import (  # noqa: E402
    init_db,
    parse_prediction_market_atlas,
    parse_regime_matrix,
    parse_trading_atlas,
    run_sync,
)

TRADING_ATLAS = REPO_ROOT / "TRADING_ATLAS.md"
PMA_PATH = REPO_ROOT / "PREDICTION_MARKET_ATLAS.md"
REGIME_MATRIX = REPO_ROOT / "docs" / "REGIME_MATRIX.md"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_parser_determinism_trading_atlas():
    """Parsing the same file twice yields identical structured records."""
    text = _read(TRADING_ATLAS)
    a = parse_trading_atlas(text)
    b = parse_trading_atlas(text)
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra.source_section == rb.source_section
        assert ra.md_hash == rb.md_hash
        assert ra.signal_type == rb.signal_type
        assert ra.asset_class == rb.asset_class
        assert ra.result_class == rb.result_class


def test_parser_determinism_pma():
    text = _read(PMA_PATH)
    a = parse_prediction_market_atlas(text)
    b = parse_prediction_market_atlas(text)
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra.source_section == rb.source_section
        assert ra.md_hash == rb.md_hash


def test_round_trip_stability(tmp_path: Path):
    """Sync to a temp DB, sync again. Second sync reports zero updates."""
    db = tmp_path / "praxis_meta.db"
    rc = run_sync(no_embed=True, db_path=db)
    assert rc == 0

    # Capture entries from first sync
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    n1 = conn.execute("SELECT COUNT(*) AS n FROM atlas_experiments").fetchone()["n"]
    conn.close()

    # Re-sync; everything should be unchanged
    rc = run_sync(no_embed=True, db_path=db)
    assert rc == 0

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    n2 = conn.execute("SELECT COUNT(*) AS n FROM atlas_experiments").fetchone()["n"]
    # The most recent sync_log rows should show all entries unchanged.
    rows = conn.execute(
        "SELECT source_file, entries_added, entries_updated, entries_unchanged "
        "FROM sync_log ORDER BY id DESC LIMIT 3"
    ).fetchall()
    conn.close()
    assert n1 == n2
    for row in rows:
        if row["source_file"] in ("TRADING_ATLAS.md", "PREDICTION_MARKET_ATLAS.md"):
            assert row["entries_added"] == 0, row["source_file"]
            assert row["entries_updated"] == 0, row["source_file"]


def test_md_hash_sensitivity(tmp_path: Path):
    """Sync to temp DB, then write a modified TA file with one entry's
    content tweaked, run a partial parse, and verify hash differs only for
    that entry.
    """
    text = _read(TRADING_ATLAS)
    records_a = parse_trading_atlas(text)
    # Modify the body of the funding-rate-carry experiment
    target = "MICROSTRUCTURE"
    found = False
    for rec in records_a:
        if target in rec.source_section:
            found = True
            target_section = rec.source_section
            break
    assert found, "Could not locate a microstructure experiment to mutate"

    # Build an edited copy of the markdown. Inject an extra line into the
    # body of experiment 13 (microstructure / funding carry) without touching
    # the heading or other entries.
    needle = "Why N-day hold (not 8h):"
    assert needle in text, "expected anchor text inside experiment 13 not found"
    edited = text.replace(
        needle, needle + "\n_Test edit for hash sensitivity test._\n", 1
    )
    assert edited != text
    records_b = parse_trading_atlas(edited)

    a_hashes = {r.source_section: r.md_hash for r in records_a}
    b_hashes = {r.source_section: r.md_hash for r in records_b}

    diffs = [
        sec for sec in a_hashes if sec in b_hashes and a_hashes[sec] != b_hashes[sec]
    ]
    assert any("MICROSTRUCTURE" in d and "13" in d for d in diffs), diffs
    unchanged = [
        sec
        for sec in a_hashes
        if sec in b_hashes
        and a_hashes[sec] == b_hashes[sec]
        and "MICROSTRUCTURE" not in sec
    ]
    assert len(unchanged) >= 10


def test_schema_validation(tmp_path: Path):
    """After a sync, every row has the required non-null fields."""
    db = tmp_path / "praxis_meta.db"
    rc = run_sync(no_embed=True, db_path=db)
    assert rc == 0

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, source_file, source_section, full_markdown, md_hash, "
        "synced_at FROM atlas_experiments"
    ).fetchall()
    conn.close()
    assert rows, "no atlas_experiments rows found"
    for row in rows:
        assert row["id"] is not None
        assert row["source_file"]
        assert row["source_section"]
        assert row["full_markdown"]
        assert row["md_hash"]
        assert row["synced_at"]


def test_regime_matrix_counts():
    """Regime matrix parses to exactly 12 classes and 5 distinct strategies."""
    classes, relevance = parse_regime_matrix(_read(REGIME_MATRIX))
    assert len(classes) == 12, f"expected 12 regime classes, got {len(classes)}"
    distinct_strategies = {r.strategy_name for r in relevance}
    assert len(distinct_strategies) == 5, (
        f"expected 5 distinct strategies, got {sorted(distinct_strategies)}"
    )
    # Every strategy should have at most one row per class letter
    for s in distinct_strategies:
        letters = [r.class_letter for r in relevance if r.strategy_name == s]
        assert len(letters) == len(set(letters)), (
            f"duplicate class_letter rows for strategy {s}"
        )


def test_trading_atlas_skips_pending():
    """The 'Pending experiments' placeholders must not appear in records."""
    records = parse_trading_atlas(_read(TRADING_ATLAS))
    sections = [r.source_section for r in records]
    # The pending placeholders are short; the real entries have attribute
    # tables. Heuristic check: every captured TA experiment has a non-empty
    # full_markdown that includes either '| **Date**' or another table row.
    for rec in records:
        assert "|" in rec.full_markdown, f"no table content in {rec.source_section}"
    # Also: no entry should have signal_type 'TA_STANDARD' AND
    # asset_class 'FUTURES_INDEX' (that was the pending placeholder; the
    # real one is 'FUTURES (...)').
    for rec in records:
        if rec.signal_type == "TA_STANDARD" and rec.asset_class == "FUTURES_INDEX":
            pytest.fail(
                f"pending placeholder leaked into records: {rec.source_section}"
            )
