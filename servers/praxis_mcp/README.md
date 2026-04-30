# Praxis MCP Server v0.1

A read-only Model Context Protocol server exposing `data/crypto_data.db` as
tools any MCP client (primarily Claude Desktop) can call. Lets Chat directly
inspect collected microstructure, OHLCV, funding, and sentiment data without
needing Code round-trips through Jeff.

Scope is deliberately narrow for v0.1: `data/crypto_data.db` only. Future
versions will add `smart_money.db` and analysis-layer tools.

---

## Install

From the praxis repo root, install the `mcp` optional dependency:

```bash
pip install "mcp>=1.2.0"
```

Or, if the repo's `.venv` is active, use the optional-dependencies extra:

```bash
pip install -e ".[mcp]"
```

Python 3.10+ required. (The repo pyproject requires 3.11+, which satisfies
this.)

## Smoke test

Before wiring the server into Claude Desktop, verify everything loads:

```bash
python -m servers.praxis_mcp.test_smoke
```

Expected output: server name, DB path, row counts per table, "OK: write
rejected" (confirms read-only enforcement), a registered-tools count, and
"Smoke test PASSED." at the end.

If this fails, stop and resolve before touching the Claude Desktop config
-- debugging inside the MCP stdio channel is harder than debugging the
imports directly.

## Claude Desktop config

Add the following to `%APPDATA%\Claude\claude_desktop_config.json` (Windows)
or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS).
If the file or `mcpServers` block doesn't exist yet, create it:

```json
{
  "mcpServers": {
    "praxis": {
      "command": "python",
      "args": [
        "-m",
        "servers.praxis_mcp.server"
      ],
      "cwd": "C:\\Data\\Development\\Python\\McTheoryApps\\praxis",
      "env": {
        "PYTHONPATH": "C:\\Data\\Development\\Python\\McTheoryApps\\praxis"
      }
    }
  }
}
```

**Replace `cwd` and `PYTHONPATH`** with your actual repo path if different.

If your interpreter isn't on PATH as plain `python` (e.g. you use the repo's
`.venv`), point `command` directly at the interpreter:

```json
"command": "C:\\Data\\Development\\Python\\McTheoryApps\\praxis\\.venv\\Scripts\\python.exe"
```

## Verification

1. Fully quit Claude Desktop (not just close the window -- quit from the
   tray / menu bar).
2. Relaunch Claude Desktop.
3. Look for the MCP indicator (hammer/plug icon) in a new chat. "praxis"
   should appear in the list of connected servers.
4. Test with: "Call `list_tables` on the praxis server."

A successful response enumerates all 10 tables in `crypto_data.db` with
their schemas.

## Troubleshooting

If the `praxis` server doesn't appear:

- **Check MCP logs:** `%APPDATA%\Claude\logs\mcp*.log` on Windows, or
  `~/Library/Logs/Claude/mcp*.log` on macOS. Look for `[praxis-mcp]`
  stderr output.
- **Wrong Python:** Claude Desktop launches `python` from its own PATH,
  which may not match your venv. If the smoke test works but the server
  doesn't appear in Claude Desktop, use the absolute-path `command` form
  shown above.
- **Wrong `cwd`:** the server imports `servers.praxis_mcp.tools.*`, which
  requires the working directory to be the repo root. A wrong `cwd` gives
  `ModuleNotFoundError: No module named 'servers'`.
- **`mcp` not installed in the interpreter Claude Desktop launched:** if
  you pip-installed in one venv but Claude Desktop launches a different
  one, the import fails silently. Use the absolute-path interpreter in
  `command`.
- **Config JSON malformed:** Claude Desktop silently ignores a broken
  config. Validate your JSON before saving.
- **Restart required:** Claude Desktop only reads `mcpServers` at app
  startup. After any config change, fully quit and relaunch.

If the server appears but tool calls error: check the log for Python
tracebacks. All tools return `{"error": "..."}` on expected failures
rather than crashing the server, so a missing error-key in the response
means the failure is upstream of the tool function.

## Tools

Twelve tools across seven modules.

### Meta (`tools/meta.py`)

- `list_tables()` -- enumerate all tables with their schemas. Call this
  first when you need to know what data is available.
- `table_stats(table_name)` -- row count, date range, per-asset breakdown.
  Detects timestamp unit (s vs ms vs ISO string).
- `get_collector_health()` -- snapshot of monitored tables (`trades`,
  `order_book_snapshots`, `ohlcv_1m`) with per-table staleness thresholds
  matching each collector's natural cadence. Orphan tables (those
  populated only by manual CLI runs -- currently `funding_rates` and
  `fear_greed`) are reported under `unmonitored` rather than alarmed on.
  See `claude/retros/RETRO_praxis_collector_outage_triage.md` for the
  investigation that established this split.

### OHLCV (`tools/ohlcv.py`)

- `get_recent_ohlcv(asset, lookback_bars=60)` -- most recent N 1-minute bars,
  oldest-first. Cap 1440 (1 day).

### Order book (`tools/order_book.py`)

- `get_order_book_snapshot(asset, at_timestamp_ms=None)` -- snapshot nearest
  to a given timestamp (or the most recent if None). Returns all 10 bid +
  10 ask levels, spread, and pre-computed aggregates.
- `get_order_book_range(asset, start_ts_ms, end_ts_ms, max_rows=200)` --
  snapshots across a range; evenly sampled if the range contains more than
  `max_rows` rows.

### Trade flow (`tools/trades.py`)

- `get_recent_trades(asset, lookback_minutes=10, min_quote_amount=0,
  max_rows=500)` -- recent trades, optionally filtered by dollar size.
  Returns `largest_trade` as a convenience. Use `min_quote_amount=100000`
  to see $100k+ trades only.
- `get_trade_flow_summary(asset, window_minutes=10)` -- aggregate
  statistics: total volume, buy/sell split, aggressor imbalance, avg and
  max trade size.

### Funding (`tools/funding.py`)

- `get_funding_rate_history(asset="BTC", lookback_days=30, max_rows=500)`
  -- funding rates with mean/min/max/positive_share stats.

### Raw (`tools/raw.py`)

- `raw_query(sql, max_rows=100)` -- escape hatch for ad-hoc SELECTs.
  Rejects INSERT/UPDATE/DELETE/DROP/etc. via keyword denylist. SQLite's
  read-only mode is the belt; this denylist is the suspenders.

**Prefer targeted tools over `raw_query`.** The targeted tools encode the
correct filters and time-range semantics; raw queries are easy to get
subtly wrong.

### Atlas (`tools/atlas.py`)

Semantic search over the Praxis Atlas (TRADING_ATLAS.md, PREDICTION_MARKET_
ATLAS.md, REGIME_MATRIX.md). Reads the sidecar `data/praxis_meta.db`
populated by `python -m engines.atlas_sync`. See `docs/ATLAS_DB.md`.

- `atlas_search(query, top_k=5)` -- find the `top_k` Atlas experiments most
  similar to a natural-language query, ranked by cosine similarity over
  Voyage / OpenAI embeddings. Use this when triaging a new trading idea
  against accumulated experimental evidence.
- `atlas_get(entry_id)` -- retrieve full markdown + parsed structured
  fields + a `source_file:lines` citation for a single experiment.

## Safety

Three layers protect the data:

1. **SQLite read-only mode:** the connection is opened with URI
   `file:{path}?mode=ro`. Any attempted write raises `OperationalError:
   attempt to write a readonly database`. Enforced at the SQLite layer,
   below anything Python can bypass.
2. **No writing tools:** none of the 10 tools call INSERT/UPDATE/DELETE.
3. **Keyword denylist in `raw_query`:** defense in depth for ad-hoc
   queries.

The smoke test verifies layer 1 actively (attempts a `CREATE TABLE` and
asserts it's rejected).

## Design notes

- **Stdio transport.** Server is launched as a subprocess by Claude Desktop
  when needed. No persistent daemon, no port binding.
- **Never writes to stdout.** Stdout is the MCP JSON-RPC channel. All
  diagnostic output goes to stderr via the `logging` module.
- **Tools in a submodule.** Each tool file is ~80 lines max. Adding new
  tools (v0.2 smart_money, analysis-layer helpers) means dropping a new
  file into `tools/` and adding one `register()` call in `server.py`.
- **Timestamp unit detection.** Some tables use seconds, some ms, some ISO
  strings. Meta tools detect the unit heuristically; targeted tools hard-
  code the unit they know each table uses.
- **Concurrent with collectors.** The Praxis collectors write to the DB
  continuously. SQLite WAL mode handles this; the read-only connection
  has a 5-second `timeout` so queries wait briefly on contention rather
  than erroring.

## What's out of scope for v0.1

- `smart_money.db` and any other Praxis DB (v0.2).
- Analysis-layer tools like "run whale detector for me and return the
  result" (v0.3).
- Streamable HTTP transport for remote mobile access (maybe v0.4).
- Write tools of any kind. The server is structurally read-only.
