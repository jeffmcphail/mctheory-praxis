# Retro: Praxis MCP Server v0.1 (COMPLETE)

**Date:** 2026-04-23
**Status:** COMPLETE -- all 10 acceptance criteria met, smoke test green, server verified starting + reading DB + blocking writes
**Brief:** `claude/handoffs/BRIEF_praxis_mcp_server_v0_1.md`
**Companion to (not superseding):** prior retros (intrabar confluence series, microstructure utilities v0.1). MCP server is a new module; no existing retro was touched.

---

## 1. TL;DR

V0.1 of the Praxis MCP server ships: 10 read-only tools exposing `data/crypto_data.db` across six modules (meta, ohlcv, order_book, trades, funding, raw). FastMCP 1.27 stdio transport, triple-layer safety (SQLite `mode=ro` + no-write-tools + keyword denylist on `raw_query`). Smoke test passes; the read-only enforcement is actively verified (attempts `CREATE TABLE`, asserts `OperationalError`). Server starts cleanly and streams `[praxis-mcp]` diagnostics to stderr only (critical for the JSON-RPC stdout channel). Total code ~300 lines across 12 files, plus 7.7 KB of install / troubleshooting docs. Ready for Claude Desktop config -- Jeff adds the JSON snippet from the README to `%APPDATA%\Claude\claude_desktop_config.json` and restarts Claude Desktop to wire it in.

---

## 2. What landed

### 2.1 Directory structure

```
servers/
  __init__.py
  praxis_mcp/
    __init__.py
    README.md              (7,726 B)
    server.py              (1,959 B)   -- FastMCP entry, tool registration
    db.py                  (  826 B)   -- read-only connection helper
    test_smoke.py          (2,663 B)   -- offline sanity check
    tools/
      __init__.py
      meta.py              (8,655 B)   -- 3 tools: list_tables, table_stats,
                                          get_collector_health
      ohlcv.py             (1,468 B)   -- 1 tool: get_recent_ohlcv
      order_book.py        (4,297 B)   -- 2 tools: get_order_book_snapshot,
                                          get_order_book_range (with
                                          ROW_NUMBER() sampling)
      trades.py            (4,773 B)   -- 2 tools: get_recent_trades,
                                          get_trade_flow_summary
      funding.py           (3,107 B)   -- 1 tool: get_funding_rate_history
      raw.py               (2,347 B)   -- 1 tool: raw_query (escape hatch)
```

All 12 Python files AST-clean and ASCII-only.

### 2.2 Dependencies

Added `mcp>=1.2.0` to `pyproject.toml` as an optional-dependency group (`[project.optional-dependencies].mcp`), not main deps. Rationale: nothing in `engines/` depends on MCP; anyone not running Claude Desktop does not need it installed. Actual install: `pip install "mcp>=1.2.0"` resolved to `mcp-1.27.0` with transitive deps `httpx-sse`, `pydantic-settings`, `pyjwt`, `sse-starlette`.

### 2.3 Tools (10 total)

| Module | Tool | One-line purpose |
|---|---|---|
| meta | `list_tables` | Enumerate all tables with schemas |
| meta | `table_stats` | Row count + date range + per-asset breakdown |
| meta | `get_collector_health` | Staleness check across live collectors |
| ohlcv | `get_recent_ohlcv` | Recent N 1-min bars (oldest-first, cap 1440) |
| order_book | `get_order_book_snapshot` | Snapshot nearest to a timestamp |
| order_book | `get_order_book_range` | Range query with even sampling if over cap |
| trades | `get_recent_trades` | Recent trades with optional `min_quote_amount` filter |
| trades | `get_trade_flow_summary` | Aggregate stats: volume split, aggressor imbalance |
| funding | `get_funding_rate_history` | Funding rates + mean/min/max/positive_share |
| raw | `raw_query` | Ad-hoc SELECT with keyword denylist + row cap |

### 2.4 Safety -- three layers, all verified

1. **SQLite `file:{path}?mode=ro`**: smoke test attempts `CREATE TABLE mcp_smoke_test_xyz` and confirms `OperationalError: attempt to write a readonly database`. SQLite enforces this below Python, so no code path can bypass.
2. **No write tools exist.** None of the 10 tools call INSERT/UPDATE/DELETE/DROP. The server is structurally read-only.
3. **`raw_query` keyword denylist:** 12 forbidden keywords (`INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, REPLACE, ATTACH, DETACH, PRAGMA, VACUUM`). Whole-word match with space padding. Direct test confirmed: `raw_query("DROP TABLE trades")` returns `{"error": "Forbidden keyword: DROP"}` without even touching SQLite.

### 2.5 Verified per-tool behavior

Direct invocation via `FastMCP._tool_manager._tools` (bypassing stdio):

- `raw_query("DROP TABLE trades")` -> `{"error": "Forbidden keyword: DROP"}`
- `raw_query("SELECT COUNT(*) AS n FROM trades")` -> `{"columns": ["n"], "count": 1, "rows": [{"n": 719794}], "truncated": False}`

Smoke test output at run time:

```
Server name: praxis
DB path: C:\Data\Development\Python\McTheoryApps\praxis\data\crypto_data.db
DB exists: True

Tables found: 10
  fear_greed:           900 rows
  funding_rates:      2,190 rows
  market_data:            2 rows
  ohlcv_1m:         520,230 rows
  ohlcv_4h:           2,160 rows
  ohlcv_daily:        1,800 rows
  onchain_btc:          365 rows
  order_book_snapshots: 4,894 rows
  sqlite_sequence:        9 rows
  trades:           719,794 rows

Testing read-only enforcement...
  OK: write rejected: OperationalError: attempt to write a readonly database

Raw query forbidden keywords: 12
  OK: denylist contains DROP, INSERT

Registered tools: 10

Smoke test PASSED.
```

### 2.6 Stdio protocol hygiene verified

`timeout 3 python -m servers.praxis_mcp.server` produced exactly one line, on stderr:

```
[praxis-mcp] INFO Starting Praxis MCP server (db=...crypto_data.db)
```

Zero stdout writes. The `logging.basicConfig(stream=sys.stderr, ...)` in `server.py` plus the explicit `format="[praxis-mcp] ..."` prefix keeps all diagnostics out of the JSON-RPC channel. This is the #1 silent-failure mode for stdio MCP servers and this deployment doesn't have it.

### 2.7 Brief-flagged schema fix

The brief's `get_collector_health` stub referenced a table `fear_greed_index`. The actual table in this DB is `fear_greed` (verified via sqlite_master). The final `meta.py` uses the correct name plus a graceful fallback for `date`-only tables (fear_greed has `date` not `timestamp`), reporting `row_count` and `latest` with a `note` explaining staleness isn't computed. The brief's "don't hard-fail" guidance was honored.

Other schema discovery: `funding_rates` uses a 10-second timestamp format (`1744531200` -> 2025-04-13) matching the brief's `ms_mode = ts_sample > 1e12` detection heuristic. Column `funding_rate` exists as assumed. No adjustment needed to `funding.py` beyond the heuristic.

## 3. Acceptance criteria -- all 10 green

- [x] `servers/praxis_mcp/` directory structure per Step 1 (12 files, verified)
- [x] `mcp>=1.2.0` added to pyproject.toml (optional-deps group `mcp`)
- [x] `python -m servers.praxis_mcp.server` starts without error, waits on stdio
- [x] `python -m servers.praxis_mcp.test_smoke` passes with "Smoke test PASSED"
- [x] Read-only enforcement verified: `CREATE TABLE` on RO connection raises `OperationalError`
- [x] `raw_query` rejects forbidden keywords: `DROP TABLE trades` -> `{"error": "Forbidden keyword: DROP"}`
- [x] Every tool returns a dict; no uncaught exceptions on happy-path inputs (verified via tool-manager invocation for representative cases)
- [x] `README.md` covers install, smoke test, Claude Desktop config JSON, verification, troubleshooting, tool list
- [x] AST parse + ASCII check pass on all 12 new Python files
- [x] Nothing writes to stdout -- only stderr logging (verified by running server with stdout capture)

## 4. Deployment path for Jeff

Pre-Claude-Desktop checklist:
1. `pip install "mcp>=1.2.0"` (done for this repo)
2. `python -m servers.praxis_mcp.test_smoke` -- confirm "Smoke test PASSED"

Wire into Claude Desktop:
1. Edit `%APPDATA%\Claude\claude_desktop_config.json`, add the `mcpServers.praxis` block per the README (with absolute `cwd` and `PYTHONPATH` matching Jeff's repo location)
2. **Fully quit** Claude Desktop (system tray, not just close window)
3. Relaunch Claude Desktop
4. Test in a new chat: "Use the praxis server's `list_tables` tool."

If it doesn't appear: `%APPDATA%\Claude\logs\mcp*.log` will have stderr output from the server process including the `[praxis-mcp]` log lines. Most likely failure mode per the README's troubleshooting section: Claude Desktop's default `python` on PATH differs from the repo venv, in which case use the absolute-path form `"command": "C:\\...\\praxis\\.venv\\Scripts\\python.exe"`.

## 5. Files changed in working tree

**New (all uncommitted):**
- `servers/__init__.py`
- `servers/praxis_mcp/__init__.py`
- `servers/praxis_mcp/README.md`
- `servers/praxis_mcp/server.py`
- `servers/praxis_mcp/db.py`
- `servers/praxis_mcp/test_smoke.py`
- `servers/praxis_mcp/tools/__init__.py`
- `servers/praxis_mcp/tools/meta.py`
- `servers/praxis_mcp/tools/ohlcv.py`
- `servers/praxis_mcp/tools/order_book.py`
- `servers/praxis_mcp/tools/trades.py`
- `servers/praxis_mcp/tools/funding.py`
- `servers/praxis_mcp/tools/raw.py`
- `claude/retros/RETRO_praxis_mcp_server.md` (this file)

**Modified (uncommitted):**
- `pyproject.toml` -- added `mcp = ["mcp>=1.2.0"]` to optional-dependencies

## 6. State at session end

- **Processes:** PraxisOrderBookCollector and PraxisTradesCollector both Running throughout the session. DB grew noticeably (trades went from ~127k to ~720k rows between session start and smoke test) -- confirms MCP reads happen concurrent with live writes without contention.
- **DB:** unchanged schema; all MCP work is read-only. Smoke test's attempted write was correctly blocked.
- **Git:** working tree has new `servers/` tree + modified `pyproject.toml` + new retro. Not committed per series convention.
- **Python env:** `mcp-1.27.0` installed in the repo's `.venv`. If Jeff has multiple venvs or Claude Desktop launches a different Python, the install may need to be repeated there.

## 7. Future work (informational, from brief §"Future work")

- **v0.2:** `smart_money.db` tools (leaderboard, positions, wallet history). Estimated 60-min Brief.
- **v0.2:** Polymarket live collector data if it lives in a Praxis-managed DB.
- **v0.3:** Analysis-layer tools -- invoke `whale_detector` through MCP, compute rolling aggressor imbalance over arbitrary window, etc. Turns MCP into an analysis layer not just a data layer.
- **v0.4:** Remote transport (Streamable HTTP) if Jeff ever wants Chat-on-mobile query access. Out of scope for local dev.
- **Alpaca / QuantConnect MCP wrappers:** separate architecture (external APIs), separate Brief.

**Immediate next-step candidate for Chat:** once the Claude Desktop wire-up is confirmed working end-to-end, the v0.2 smart_money tools are the natural follow-on -- smart_money.db is already being populated by the existing PraxisSmartMoney scheduled task, so the data foundation is there.
