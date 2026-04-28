"""Read-only SQLite connection helper.

The URI `file:{path}?mode=ro` instructs SQLite to open in read-only mode.
Any attempted write raises sqlite3.OperationalError: attempt to write a
readonly database. This is enforced at the SQLite layer, below anything
Python can accidentally bypass.
"""

import sqlite3
from pathlib import Path


def connect_ro(db_path: Path) -> sqlite3.Connection:
    """Open a read-only connection to the given SQLite DB."""
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    uri = f"file:{db_path}?mode=ro"
    # timeout=5.0 so the connection will wait briefly if collectors are
    # mid-commit rather than erroring out on any momentary contention.
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn
