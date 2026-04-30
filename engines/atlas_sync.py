"""
Atlas sync -- parse Praxis Atlas markdown into a queryable SQLite DB.

Source files (markdown is the source of truth):
  - TRADING_ATLAS.md (numbered experiments)
  - PREDICTION_MARKET_ATLAS.md (lettered strategies under categories)
  - docs/REGIME_MATRIX.md (12 regime classes + strategy x regime relevance)

The DB at data/praxis_meta.db is fully derived. Run after editing any source
file to re-sync; embeddings are skipped for entries whose md_hash is unchanged
since the last sync.

Usage:
  python -m engines.atlas_sync                  # full sync (writes DB + embeds)
  python -m engines.atlas_sync --validate       # parse-only, no DB write
  python -m engines.atlas_sync --verbose        # log per-entry capture report
  python -m engines.atlas_sync --no-embed       # skip embeddings (cheap test)
  python -m engines.atlas_sync --strict         # crash on any parse error

Markdown is canonical. This script never writes to the source markdown files.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "praxis_meta.db"
TRADING_ATLAS = REPO_ROOT / "TRADING_ATLAS.md"
PMA_PATH = REPO_ROOT / "PREDICTION_MARKET_ATLAS.md"
REGIME_MATRIX = REPO_ROOT / "docs" / "REGIME_MATRIX.md"

VOYAGE_MODEL = "voyage-3-lite"
VOYAGE_DIM = 512
OPENAI_MODEL = "text-embedding-3-small"
OPENAI_DIM = 1536

# Unicode characters used in the source markdown. Defined as escape sequences
# so this source file remains ASCII per project rule 19.
EM_DASH = "\u2014"
EN_DASH = "\u2013"
MULT_SIGN = "\u00d7"
DOT = "\u25cf"

# Diagnostic logging to stderr (stdout is used for the sync diff report).
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[atlas-sync] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


DDL = """
CREATE TABLE IF NOT EXISTS atlas_experiments (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file         TEXT NOT NULL,
    source_section      TEXT NOT NULL,
    source_line_start   INTEGER NOT NULL,
    source_line_end     INTEGER NOT NULL,
    signal_type         TEXT,
    asset_class         TEXT,
    framework           TEXT,
    date_run            TEXT,
    result_class        TEXT,
    result_summary      TEXT,
    full_markdown       TEXT NOT NULL,
    key_findings        TEXT,
    atlas_principle     TEXT,
    md_hash             TEXT NOT NULL,
    synced_at           TEXT NOT NULL,
    UNIQUE(source_file, source_section)
);

CREATE TABLE IF NOT EXISTS atlas_embeddings (
    experiment_id       INTEGER PRIMARY KEY,
    embedding_model     TEXT NOT NULL,
    embedding_dim       INTEGER NOT NULL,
    embedding           BLOB NOT NULL,
    md_hash             TEXT NOT NULL,
    embedded_at         TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES atlas_experiments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS regime_classes (
    class_letter        TEXT PRIMARY KEY,
    class_name          TEXT NOT NULL,
    states              TEXT NOT NULL,
    detection_method    TEXT NOT NULL,
    key_data            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_regime_relevance (
    strategy_name       TEXT NOT NULL,
    class_letter        TEXT NOT NULL,
    relevance_dots      INTEGER NOT NULL,
    PRIMARY KEY (strategy_name, class_letter)
);

CREATE TABLE IF NOT EXISTS sync_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    synced_at           TEXT NOT NULL,
    source_file         TEXT NOT NULL,
    entries_added       INTEGER NOT NULL,
    entries_updated     INTEGER NOT NULL,
    entries_unchanged   INTEGER NOT NULL,
    notes               TEXT
);
"""


@dataclass
class ExperimentRecord:
    source_file: str
    source_section: str
    source_line_start: int
    source_line_end: int
    full_markdown: str
    md_hash: str
    signal_type: Optional[str] = None
    asset_class: Optional[str] = None
    framework: Optional[str] = None
    date_run: Optional[str] = None
    result_class: Optional[str] = None
    result_summary: Optional[str] = None
    key_findings: Optional[str] = None
    atlas_principle: Optional[str] = None
    captured_fields: list[str] = field(default_factory=list)
    missed_fields: list[str] = field(default_factory=list)


@dataclass
class RegimeClass:
    class_letter: str
    class_name: str
    states: str
    detection_method: str
    key_data: str


@dataclass
class RelevanceRow:
    strategy_name: str
    class_letter: str
    relevance_dots: int


# ----------------------------- Trading Atlas ------------------------------

EXPERIMENT_HEADING = re.compile(r"^### (\d+)\.\s+(.+)$")
ATTR_ROW = re.compile(r"^\|\s*\*\*([^*|]+)\*\*\s*\|\s*(.+?)\s*\|\s*$")

SKIP_PARENT_SECTIONS = {
    "pending experiments",
}


def parse_trading_atlas(text: str, *, strict: bool = False) -> list[ExperimentRecord]:
    """Parse TRADING_ATLAS.md into experiment records.

    An experiment runs from `### N. ...` until the next `### N. ...` heading
    or until the next `## ` major section heading, whichever comes first.
    Experiments under skip-listed parent sections (e.g. 'Pending experiments')
    are intentionally excluded -- they are placeholders, not real entries.
    """
    lines = text.splitlines()
    n = len(lines)
    records: list[ExperimentRecord] = []
    current_parent: str = ""

    i = 0
    while i < n:
        line = lines[i]
        if line.startswith("## "):
            current_parent = line[3:].strip().lower()
            current_parent = current_parent.replace(EM_DASH, "--").replace(EN_DASH, "--")
            i += 1
            continue

        m = EXPERIMENT_HEADING.match(line)
        if not m:
            i += 1
            continue

        if any(current_parent.startswith(skip) for skip in SKIP_PARENT_SECTIONS):
            i += 1
            continue

        section_title = lines[i][4:].strip()
        start_idx = i

        j = i + 1
        while j < n:
            if EXPERIMENT_HEADING.match(lines[j]):
                break
            if lines[j].startswith("## "):
                break
            j += 1

        block_lines = lines[start_idx:j]
        while block_lines and (
            block_lines[-1].strip() == "" or block_lines[-1].strip() == "---"
        ):
            block_lines.pop()

        block = "\n".join(block_lines)
        rec = ExperimentRecord(
            source_file="TRADING_ATLAS.md",
            source_section=section_title,
            source_line_start=start_idx + 1,
            source_line_end=start_idx + len(block_lines),
            full_markdown=block,
            md_hash=_sha256(block),
        )

        try:
            _extract_signal_and_asset(rec, section_title)
            _extract_attribute_table_ta(rec, block)
            _extract_result_class(rec, block)
            _extract_result_summary(rec, block)
            _extract_key_findings(rec, block)
            _extract_atlas_principle(rec, block)
        except Exception as e:
            if strict:
                raise
            log.warning("Failed to fully parse '%s': %s", section_title, e)

        _record_capture(rec)
        records.append(rec)
        i = j

    return records


def _extract_signal_and_asset(rec: ExperimentRecord, title: str) -> None:
    """Parse signal_type and asset_class from a heading like
    'N. SIGNAL <MULT> ASSET_CLASS -- Description'.
    """
    m = re.match(r"^\d+\.\s+(.+)$", title)
    if not m:
        return
    body = m.group(1)
    parts = re.split(r"\s+[xX" + MULT_SIGN + r"]\s+", body, maxsplit=1)
    if len(parts) == 2:
        rec.signal_type = parts[0].strip()
        asset_part = parts[1].strip()
        # Trim trailing description after dash or paren
        asset = re.split(
            r"\s+[" + EN_DASH + EM_DASH + r"-]+\s+|\s*\(",
            asset_part,
            maxsplit=1,
        )[0]
        rec.asset_class = asset.strip()


def _extract_attribute_table_ta(rec: ExperimentRecord, block: str) -> None:
    """Extract the leading '| **Field** | value |' attribute table common to
    most TA experiments. Maps recognized field names to schema columns.
    """
    for line in block.splitlines():
        m = ATTR_ROW.match(line)
        if not m:
            continue
        key = m.group(1).strip().lower()
        value = m.group(2).strip()
        if key in ("date", "date run", "run date"):
            d = re.search(r"\d{4}-\d{2}-\d{2}", value)
            if d:
                rec.date_run = d.group(0)
        elif key == "framework":
            rec.framework = value
        elif key == "result":
            cls = _classify_result_token(value)
            if cls:
                rec.result_class = cls


_VERDICT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"verdict[^.]*confirmed positive", re.I), "POSITIVE"),
    (re.compile(r"verdict[^.]*confirmed[^.]*no edge", re.I), "NEGATIVE"),
    (re.compile(r"confirmed positive", re.I), "POSITIVE"),
    (re.compile(r"\bweak positive\b", re.I), "PARTIAL"),
    (re.compile(r"\bpromising\b", re.I), "PARTIAL"),
    (re.compile(r"\binconclusive\b", re.I), "INCONCLUSIVE"),
    (re.compile(r"\bblocked\b", re.I), "INCONCLUSIVE"),
    (re.compile(r"\bno edge\b", re.I), "NEGATIVE"),
]


def _extract_result_class(rec: ExperimentRecord, block: str) -> None:
    if rec.result_class:
        return
    for pat, cls in _VERDICT_PATTERNS:
        if pat.search(block):
            rec.result_class = cls
            return


def _classify_result_token(value: str) -> Optional[str]:
    upper = value.upper()
    if "POSITIVE" in upper and "NEGATIVE" not in upper:
        return "POSITIVE"
    if "NEGATIVE" in upper:
        return "NEGATIVE"
    if "INCONCLUSIVE" in upper or "BLOCKED" in upper:
        return "INCONCLUSIVE"
    if "WEAK" in upper or "PROMISING" in upper or "PARTIAL" in upper:
        return "PARTIAL"
    return None


def _extract_result_summary(rec: ExperimentRecord, block: str) -> None:
    """Best-effort prose summary extraction.

    Handles both forms of bold: text after the closing ** OR the entire
    phrase enclosed by **...**.
    """
    for pat in (
        r"\*\*Result:\s*([^*]+?)\*\*",
        r"\*\*Verdict:?\s*([^*]+?)\*\*",
        r"\*\*Atlas verdict[^*]*\*\*:?\s*(.+)",
        r"\*\*Result\*\*:\s*(.+)",
        r"\*\*Verdict[^*]*\*\*:?\s*(.+)",
        r"Atlas verdict:\s*(.+)",
        r"Verdict:\s*(.+)",
    ):
        m = re.search(pat, block)
        if m:
            line = m.group(1).strip()
            line = re.sub(r"\*\*", "", line)
            rec.result_summary = line[:400]
            return

    in_table = False
    for line in block.splitlines()[1:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("|"):
            in_table = True
            continue
        if in_table and not s.startswith("|"):
            in_table = False
        if s.startswith("#") or s.startswith("---"):
            continue
        if s.startswith("|"):
            continue
        rec.result_summary = re.sub(r"\*\*", "", s)[:400]
        return


def _extract_key_findings(rec: ExperimentRecord, block: str) -> None:
    lines = block.splitlines()
    out: list[str] = []
    capturing = False
    for line in lines:
        s = line.strip()
        if re.match(r"^\*\*Key findings.*\*\*", s, re.I):
            capturing = True
            continue
        if capturing:
            if re.match(r"^\*\*[^*]+\*\*\s*$|^#{1,4}\s|^---\s*$", s):
                break
            if s:
                out.append(s)
    if out:
        rec.key_findings = "\n".join(out)[:4000]


def _extract_atlas_principle(rec: ExperimentRecord, block: str) -> None:
    lines = block.splitlines()
    out: list[str] = []
    capturing = False
    for line in lines:
        s = line.strip()
        if re.match(r"^\*\*Atlas principle.*\*\*", s, re.I):
            capturing = True
            continue
        if capturing:
            if re.match(r"^\*\*[^*]+\*\*\s*$|^#{1,4}\s|^---\s*$", s):
                break
            if s:
                out.append(s)
    if out:
        rec.atlas_principle = "\n".join(out)[:4000]


# ------------------------- Prediction Market Atlas -------------------------

PMA_STRATEGY_HEADING = re.compile(r"^####\s+([A-D]\d+\w*)\.\s+(.+)$")
PMA_RANK_HEADING = re.compile(r"^###\s+Rank\s+(\d+):\s*(.+)$")

PMA_CATEGORY_NAMES = {
    "A": "PURE_ARBITRAGE",
    "B": "STATISTICAL_EDGE",
    "C": "INFORMATION_EDGE",
    "D": "MARKET_MAKING",
}


def parse_prediction_market_atlas(
    text: str, *, strict: bool = False
) -> list[ExperimentRecord]:
    lines = text.splitlines()
    n = len(lines)
    records: list[ExperimentRecord] = []

    i = 0
    while i < n:
        m_strat = PMA_STRATEGY_HEADING.match(lines[i])
        m_rank = PMA_RANK_HEADING.match(lines[i])
        if not (m_strat or m_rank):
            i += 1
            continue

        section_title = lines[i].lstrip("#").strip()
        start_idx = i
        j = i + 1
        while j < n:
            if PMA_STRATEGY_HEADING.match(lines[j]):
                break
            if PMA_RANK_HEADING.match(lines[j]):
                break
            if lines[j].startswith("## "):
                break
            if lines[j].startswith("### "):
                break
            j += 1

        block_lines = lines[start_idx:j]
        while block_lines and (
            block_lines[-1].strip() == "" or block_lines[-1].strip() == "---"
        ):
            block_lines.pop()
        block = "\n".join(block_lines)

        rec = ExperimentRecord(
            source_file="PREDICTION_MARKET_ATLAS.md",
            source_section=section_title,
            source_line_start=start_idx + 1,
            source_line_end=start_idx + len(block_lines),
            full_markdown=block,
            md_hash=_sha256(block),
            asset_class="PREDICTION_MARKET",
        )

        try:
            if m_strat:
                code = m_strat.group(1)
                rec.signal_type = PMA_CATEGORY_NAMES.get(code[0], code)
            else:
                rec.signal_type = "PRIORITIZED_STRATEGY"

            _extract_pma_status(rec, block)
            _extract_pma_summary(rec, block)
        except Exception as e:
            if strict:
                raise
            log.warning("Failed to fully parse PMA '%s': %s", section_title, e)

        _record_capture(rec)
        records.append(rec)
        i = j

    return records


def _extract_pma_status(rec: ExperimentRecord, block: str) -> None:
    """Map the '**Current status: ...**' phrase to a result_class.

    Handles both forms:
      **Current status:** ALIVE -- ...     (bold ends after colon)
      **Current status: ALIVE -- ...**     (bold spans whole phrase)
    """
    text: Optional[str] = None
    m = re.search(r"\*\*Current status:\s*([^*]+?)\*\*", block, re.I)
    if m:
        text = m.group(1).strip()
    else:
        m = re.search(r"\*\*Current status:\*\*\s*([^\n]+)", block, re.I)
        if m:
            text = m.group(1).strip()
    if text is None:
        m = re.search(r"\*\*Current state:\s*([^*]+?)\*\*", block, re.I)
        if m:
            text = m.group(1).strip()
    if text is None:
        return
    rec.result_summary = re.sub(r"\*\*", "", text)[:400]
    upper = text.upper()
    if "DEAD" in upper:
        rec.result_class = "NEGATIVE"
    elif "ALIVE" in upper:
        rec.result_class = "POSITIVE"
    elif "DYING" in upper or "EMERGING" in upper:
        rec.result_class = "PARTIAL"
    elif "UNDER-RESEARCHED" in upper or "RESEARCH" in upper:
        rec.result_class = "INCONCLUSIVE"


def _extract_pma_summary(rec: ExperimentRecord, block: str) -> None:
    if rec.result_summary:
        m = re.search(r"\*\*Mechanism:\*\*\s*([^\n]+)", block, re.I)
        if m:
            extra = re.sub(r"\*\*", "", m.group(1).strip())
            rec.result_summary = (rec.result_summary + " | " + extra)[:600]
        return
    for pat in (
        r"\*\*Mechanism:\*\*\s*([^\n]+)",
        r"\*\*Why:\*\*\s*([^\n]+)",
        r"\*\*Implementation:\*\*\s*([^\n]+)",
    ):
        m = re.search(pat, block, re.I)
        if m:
            rec.result_summary = re.sub(r"\*\*", "", m.group(1).strip())[:400]
            return


# ------------------------------ Regime Matrix ------------------------------


def parse_regime_matrix(
    text: str,
) -> tuple[list[RegimeClass], list[RelevanceRow]]:
    classes = _parse_regime_classes_table(text)
    relevance = _parse_relevance_table(text)
    return classes, relevance


def _parse_regime_classes_table(text: str) -> list[RegimeClass]:
    lines = text.splitlines()
    in_section = False
    out: list[RegimeClass] = []
    for line in lines:
        if line.startswith("## Regime Classes"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section:
            continue
        s = line.strip()
        if s.startswith("|") and not s.startswith("|---") and "---" not in s:
            cells = [c.strip() for c in s.strip("|").split("|")]
            if len(cells) >= 5 and cells[0] != "#" and re.match(r"^[A-Z]$", cells[0]):
                out.append(
                    RegimeClass(
                        class_letter=cells[0],
                        class_name=cells[1],
                        states=cells[2],
                        detection_method=cells[3],
                        key_data=cells[4],
                    )
                )
    return out


def _parse_relevance_table(text: str) -> list[RelevanceRow]:
    """Parse the '## Relevance Matrix' table; counts dot characters per cell."""
    lines = text.splitlines()
    in_section = False
    headers: list[str] = []
    out: list[RelevanceRow] = []
    for line in lines:
        if line.startswith("## Relevance Matrix"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section:
            continue
        s = line.strip()
        if not s.startswith("|"):
            continue
        if "---" in s:
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if not cells:
            continue
        if cells[0].lower() == "strategy":
            headers = cells
            continue
        if not headers or len(cells) != len(headers):
            continue
        strategy_name = cells[0]
        for hdr, val in zip(headers[1:], cells[1:]):
            if not val:
                continue
            dots = val.count(DOT)
            if dots == 0:
                continue
            out.append(
                RelevanceRow(
                    strategy_name=strategy_name,
                    class_letter=hdr,
                    relevance_dots=dots,
                )
            )
    return out


# ------------------------------- Embeddings --------------------------------


def embedding_text(rec: ExperimentRecord) -> str:
    parts = [rec.source_section]
    if rec.result_summary:
        parts.append(rec.result_summary)
    if rec.key_findings:
        parts.append(rec.key_findings)
    if rec.atlas_principle:
        parts.append(rec.atlas_principle)
    return "\n\n".join(parts)


def detect_embedding_provider() -> tuple[Optional[str], Optional[str], Optional[int]]:
    if os.getenv("VOYAGE_API_KEY"):
        return "voyage", VOYAGE_MODEL, VOYAGE_DIM
    if os.getenv("OPENAI_API_KEY"):
        return "openai", OPENAI_MODEL, OPENAI_DIM
    return None, None, None


def embed_batch(provider: str, model: str, texts: list[str]) -> list[np.ndarray]:
    import requests

    if provider == "voyage":
        key = os.getenv("VOYAGE_API_KEY")
        url = "https://api.voyageai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = {"input": texts, "model": model, "input_type": "document"}
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()["data"]
        return [np.asarray(d["embedding"], dtype=np.float32) for d in data]

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = {"input": texts, "model": model}
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()["data"]
        return [np.asarray(d["embedding"], dtype=np.float32) for d in data]

    raise ValueError(f"Unknown provider: {provider}")


def embed_query(provider: str, model: str, query: str) -> np.ndarray:
    """Embed a single query string. Voyage uses input_type=query for retrieval."""
    import requests

    if provider == "voyage":
        key = os.getenv("VOYAGE_API_KEY")
        url = "https://api.voyageai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = {"input": [query], "model": model, "input_type": "query"}
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        return np.asarray(r.json()["data"][0]["embedding"], dtype=np.float32)

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = {"input": [query], "model": model}
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        return np.asarray(r.json()["data"][0]["embedding"], dtype=np.float32)

    raise ValueError(f"Unknown provider: {provider}")


# ----------------------------- DB operations ------------------------------


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    conn.commit()
    return conn


def upsert_experiments(
    conn: sqlite3.Connection,
    records: list[ExperimentRecord],
    source_file: str,
    synced_at: str,
) -> tuple[int, int, int, list[ExperimentRecord]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT id, source_section, md_hash FROM atlas_experiments "
        "WHERE source_file = ?",
        (source_file,),
    )
    existing = {row["source_section"]: (row["id"], row["md_hash"]) for row in cur.fetchall()}

    added = updated = unchanged = 0
    changed: list[ExperimentRecord] = []
    seen_sections: set[str] = set()

    for rec in records:
        seen_sections.add(rec.source_section)
        prior = existing.get(rec.source_section)
        if prior is None:
            cur.execute(
                """
                INSERT INTO atlas_experiments (
                    source_file, source_section, source_line_start, source_line_end,
                    signal_type, asset_class, framework, date_run,
                    result_class, result_summary, full_markdown, key_findings,
                    atlas_principle, md_hash, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.source_file, rec.source_section, rec.source_line_start,
                    rec.source_line_end, rec.signal_type, rec.asset_class,
                    rec.framework, rec.date_run, rec.result_class,
                    rec.result_summary, rec.full_markdown, rec.key_findings,
                    rec.atlas_principle, rec.md_hash, synced_at,
                ),
            )
            added += 1
            changed.append(rec)
        elif prior[1] != rec.md_hash:
            cur.execute(
                """
                UPDATE atlas_experiments SET
                    source_line_start = ?, source_line_end = ?,
                    signal_type = ?, asset_class = ?, framework = ?,
                    date_run = ?, result_class = ?, result_summary = ?,
                    full_markdown = ?, key_findings = ?, atlas_principle = ?,
                    md_hash = ?, synced_at = ?
                WHERE id = ?
                """,
                (
                    rec.source_line_start, rec.source_line_end,
                    rec.signal_type, rec.asset_class, rec.framework,
                    rec.date_run, rec.result_class, rec.result_summary,
                    rec.full_markdown, rec.key_findings, rec.atlas_principle,
                    rec.md_hash, synced_at, prior[0],
                ),
            )
            updated += 1
            changed.append(rec)
        else:
            cur.execute(
                "UPDATE atlas_experiments SET source_line_start = ?, "
                "source_line_end = ? WHERE id = ?",
                (rec.source_line_start, rec.source_line_end, prior[0]),
            )
            unchanged += 1

    stale = set(existing.keys()) - seen_sections
    for section in stale:
        cur.execute(
            "DELETE FROM atlas_experiments WHERE source_file = ? AND source_section = ?",
            (source_file, section),
        )

    conn.commit()
    return added, updated, unchanged, changed


def upsert_regime_classes(
    conn: sqlite3.Connection, classes: list[RegimeClass]
) -> int:
    cur = conn.cursor()
    cur.execute("DELETE FROM regime_classes")
    for c in classes:
        cur.execute(
            """
            INSERT INTO regime_classes (
                class_letter, class_name, states, detection_method, key_data
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (c.class_letter, c.class_name, c.states, c.detection_method, c.key_data),
        )
    conn.commit()
    return len(classes)


def upsert_strategy_relevance(
    conn: sqlite3.Connection, rows: list[RelevanceRow]
) -> int:
    cur = conn.cursor()
    cur.execute("DELETE FROM strategy_regime_relevance")
    for r in rows:
        cur.execute(
            """
            INSERT INTO strategy_regime_relevance (
                strategy_name, class_letter, relevance_dots
            ) VALUES (?, ?, ?)
            """,
            (r.strategy_name, r.class_letter, r.relevance_dots),
        )
    conn.commit()
    return len(rows)


def write_embeddings(
    conn: sqlite3.Connection,
    records_by_section: dict[tuple[str, str], ExperimentRecord],
    provider: str,
    model: str,
    dim: int,
    embedded_at: str,
) -> tuple[int, int]:
    if not records_by_section:
        return 0, 0

    cur = conn.cursor()
    to_embed: list[tuple[int, ExperimentRecord]] = []
    skipped = 0

    for (sf, sec), rec in records_by_section.items():
        cur.execute(
            "SELECT id FROM atlas_experiments WHERE source_file = ? AND source_section = ?",
            (sf, sec),
        )
        row = cur.fetchone()
        if row is None:
            continue
        exp_id = row["id"]

        cur.execute(
            "SELECT md_hash, embedding_model FROM atlas_embeddings WHERE experiment_id = ?",
            (exp_id,),
        )
        emb_row = cur.fetchone()
        if emb_row is not None and emb_row["md_hash"] == rec.md_hash and emb_row["embedding_model"] == model:
            skipped += 1
            continue
        to_embed.append((exp_id, rec))

    if not to_embed:
        return 0, skipped

    log.info("Embedding %d entries via %s", len(to_embed), model)
    texts = [embedding_text(rec) for _, rec in to_embed]

    BATCH = 64
    vectors: list[np.ndarray] = []
    for k in range(0, len(texts), BATCH):
        batch = texts[k : k + BATCH]
        vectors.extend(embed_batch(provider, model, batch))

    for (exp_id, rec), vec in zip(to_embed, vectors):
        if vec.shape[0] != dim:
            raise RuntimeError(
                f"Embedding dim mismatch: expected {dim}, got {vec.shape[0]}"
            )
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        blob = vec.astype(np.float32).tobytes()
        cur.execute(
            """
            INSERT INTO atlas_embeddings (
                experiment_id, embedding_model, embedding_dim, embedding,
                md_hash, embedded_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(experiment_id) DO UPDATE SET
                embedding_model = excluded.embedding_model,
                embedding_dim   = excluded.embedding_dim,
                embedding       = excluded.embedding,
                md_hash         = excluded.md_hash,
                embedded_at     = excluded.embedded_at
            """,
            (exp_id, model, dim, blob, rec.md_hash, embedded_at),
        )

    conn.commit()
    return len(to_embed), skipped


def write_sync_log(
    conn: sqlite3.Connection,
    synced_at: str,
    counts_by_file: dict[str, tuple[int, int, int]],
    notes: Optional[str] = None,
) -> None:
    cur = conn.cursor()
    for sf, (added, updated, unchanged) in counts_by_file.items():
        cur.execute(
            """
            INSERT INTO sync_log (
                synced_at, source_file, entries_added, entries_updated,
                entries_unchanged, notes
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (synced_at, sf, added, updated, unchanged, notes),
        )
    conn.commit()


# ------------------------------ Helpers -----------------------------------


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _record_capture(rec: ExperimentRecord) -> None:
    fields = [
        ("signal_type", rec.signal_type),
        ("asset_class", rec.asset_class),
        ("framework", rec.framework),
        ("date_run", rec.date_run),
        ("result_class", rec.result_class),
        ("result_summary", rec.result_summary),
        ("key_findings", rec.key_findings),
        ("atlas_principle", rec.atlas_principle),
    ]
    for name, val in fields:
        if val:
            rec.captured_fields.append(name)
        else:
            rec.missed_fields.append(name)


# ------------------------------ Main flow ---------------------------------


def run_sync(
    *,
    validate: bool = False,
    verbose: bool = False,
    no_embed: bool = False,
    strict: bool = False,
    db_path: Path = DB_PATH,
) -> int:
    load_dotenv(REPO_ROOT / ".env")
    synced_at = datetime.now(tz=timezone.utc).isoformat()

    print(f"Atlas sync -- {synced_at}")

    if not TRADING_ATLAS.exists():
        log.error("Missing source: %s", TRADING_ATLAS)
        return 1
    if not PMA_PATH.exists():
        log.error("Missing source: %s", PMA_PATH)
        return 1
    if not REGIME_MATRIX.exists():
        log.error("Missing source: %s", REGIME_MATRIX)
        return 1

    ta_text = TRADING_ATLAS.read_text(encoding="utf-8")
    pma_text = PMA_PATH.read_text(encoding="utf-8")
    rm_text = REGIME_MATRIX.read_text(encoding="utf-8")

    ta_records = parse_trading_atlas(ta_text, strict=strict)
    pma_records = parse_prediction_market_atlas(pma_text, strict=strict)
    regime_classes, relevance_rows = parse_regime_matrix(rm_text)

    print(f"  Parsed TRADING_ATLAS.md: {len(ta_records)} experiments")
    print(f"  Parsed PREDICTION_MARKET_ATLAS.md: {len(pma_records)} entries")
    print(f"  Parsed REGIME_MATRIX.md: {len(regime_classes)} classes, {len(relevance_rows)} relevance rows")

    if verbose:
        print("\n--- Per-entry capture report ---")
        for rec in ta_records + pma_records:
            print(
                f"  [{rec.source_file}:{rec.source_line_start}] {rec.source_section}"
            )
            print(f"    captured: {','.join(rec.captured_fields) or '(none)'}")
            print(f"    missed:   {','.join(rec.missed_fields) or '(none)'}")

    if validate:
        print("\n--validate: skipping DB write.")
        return 0

    conn = init_db(db_path)

    ta_added, ta_updated, ta_unchanged, _ = upsert_experiments(
        conn, ta_records, "TRADING_ATLAS.md", synced_at
    )
    pma_added, pma_updated, pma_unchanged, _ = upsert_experiments(
        conn, pma_records, "PREDICTION_MARKET_ATLAS.md", synced_at
    )
    rc_count = upsert_regime_classes(conn, regime_classes)
    rr_count = upsert_strategy_relevance(conn, relevance_rows)

    print("\n--- Sync summary ---")
    print("  TRADING_ATLAS.md")
    print(f"    + {ta_added} added")
    print(f"    ~ {ta_updated} updated")
    print(f"    = {ta_unchanged} unchanged")
    print("  PREDICTION_MARKET_ATLAS.md")
    print(f"    + {pma_added} added")
    print(f"    ~ {pma_updated} updated")
    print(f"    = {pma_unchanged} unchanged")
    print("  REGIME_MATRIX.md")
    print(f"    = {rc_count} regime classes (full replace)")
    print(f"    = {rr_count} strategy relevance rows (full replace)")

    embed_note = None
    if no_embed:
        print("\n  Embeddings: --no-embed flag set, skipping embedding step.")
    else:
        provider, model, dim = detect_embedding_provider()
        if provider is None:
            print(
                "\n  Embeddings: no API key found (set VOYAGE_API_KEY or OPENAI_API_KEY in .env)"
            )
            print("  Re-run after setting a key, or use --no-embed to silence this.")
            embed_note = "no_api_key"
        else:
            records_by_section: dict[tuple[str, str], ExperimentRecord] = {}
            for rec in ta_records:
                records_by_section[(rec.source_file, rec.source_section)] = rec
            for rec in pma_records:
                records_by_section[(rec.source_file, rec.source_section)] = rec
            try:
                regenerated, skipped = write_embeddings(
                    conn, records_by_section, provider, model, dim, synced_at
                )
                print(
                    f"\n  Embeddings ({model}): {regenerated} regenerated, {skipped} skipped"
                )
            except Exception as e:
                log.error("Embedding step failed: %s", e)
                if strict:
                    raise
                embed_note = f"embedding_error:{e}"

    counts = {
        "TRADING_ATLAS.md": (ta_added, ta_updated, ta_unchanged),
        "PREDICTION_MARKET_ATLAS.md": (pma_added, pma_updated, pma_unchanged),
        "REGIME_MATRIX.md": (0, 0, len(regime_classes) + len(relevance_rows)),
    }
    write_sync_log(conn, synced_at, counts, notes=embed_note)

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM atlas_experiments")
    total = cur.fetchone()["n"]
    print(f"\n  Total entries in atlas_experiments: {total}")
    print(f"  DB: {db_path}")
    print("  Run logged to sync_log table.")

    conn.close()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sync Praxis Atlas markdown to DB.")
    parser.add_argument("--validate", action="store_true", help="Parse only; do not write DB.")
    parser.add_argument("--verbose", action="store_true", help="Per-entry capture report.")
    parser.add_argument("--no-embed", action="store_true", help="Skip embedding step.")
    parser.add_argument("--strict", action="store_true", help="Crash on parse error.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="Override DB path.")
    args = parser.parse_args(argv)

    return run_sync(
        validate=args.validate,
        verbose=args.verbose,
        no_embed=args.no_embed,
        strict=args.strict,
        db_path=args.db_path,
    )


if __name__ == "__main__":
    sys.exit(main())
