# McTheory Praxis — `praxis_main` Series Overview

> **Purpose:** Comprehensive overview of the McTheory Praxis project and the `praxis_main` chat series. New chats in this series should read this document to get up to speed.

---

## Project Genesis

McTheory Praxis was born from the specification work done across the `praxis_spec` chat series (iterations 1-9), which produced a 3,457-line specification (v9.3.1) and an 882-line execution plan (v1.2). The specification covers 23 Parts spanning data architecture, temporal patterns, model parameterization, backtest engines, scheduling, diagnostic logging, and GUI/dashboard design.

The `praxis_main` series is the **implementation** series — where the spec becomes code.

---

## Architecture Summary

### The Two-User Test

Praxis must serve two users from the same codebase:
1. **High schooler:** `praxis run sma_crossover.yaml` — simple config, works out of the box
2. **Quant:** Chan CPO with ML parameter optimization, bi-temporal backtesting, regime-conditional sizing — all from YAML

### Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Config | YAML + Pydantic | Model definitions |
| Storage | DuckDB (dev), PostgreSQL + TimescaleDB (live) | Analytical + transactional |
| Compute | Polars + native | Vectorized operations |
| Temporal | vew_, vt2_, rpt_, ved_ views | Bi-temporal data access |
| Keys | _bpk / _base_id / _hist_id | Universal entity identity |

### Core Design Decisions

1. **Bi-temporal with derived dates** — start/end computed from _hist_id sequence, not stored
2. **DuckDB STRUCTs over JSON** — typed, queryable nested model configs
3. **Function registry** — YAML method names resolve to Python classes at runtime
4. **Coreify touchpoints** — 14 documented points where Praxis patterns become Core abstractions
5. **Best-of-breed adapter framework** — one interface, many vendor implementations, context-dependent selection

---

## Implementation Phases

| Phase | Goal | Days | Status |
|-------|------|------|--------|
| 0 | Spike validation + scaffolding | ~5 | ✅ Complete |
| 1 | Thin vertical slice: `praxis run sma_crossover.yaml` | 26-33 | 🔧 In progress |
| 2 | DataStore & Security Master | 36-46 | ⬜ |
| 3 | Chan CPO — architecture validation | 51-63 | ⬜ |
| 4 | Burgess Stat Arb + GUI Dashboard | 93-122 | ⬜ |
| 5 | Live trading + advanced features | 73-105 | ⬜ |

---

## Key Documents

| Document | Location | Purpose |
|----------|----------|---------|
| Specification v9.3.1 | Claude Project files | Full 23-part spec |
| Execution Plan v1.2 | Claude Project files | Phased implementation with dependency chain |
| Spike Results | `spikes/spike_*_results.md` | Assumption validation outcomes |
| This file | `docs/praxis_main_series.md` | Series overview (you're reading it) |
| Transcript | `docs/praxis_main_series_transcript.md` | Chronological decision log |

---

## Chat History

| Chat Name | Status | Focus |
|-----------|--------|-------|
| `praxis_main_current` | **Active** | Phase 0 spikes + Phase 1.1 keys infrastructure |

---

## Spike Results (Phase 0)

### Spike 1: vt2_ View Performance ✅ PASS
- 100K rows (10K securities × 10 versions)
- Single-security AS-IS: 9.9ms (threshold 50ms, 5× margin)
- 1,000-security batch: 97ms (threshold 2s, 20× margin)
- Full scan: 50ms (threshold 10s, 200× margin)
- **Finding:** xxHash64 returns uint64, DuckDB BIGINT is int64 — signed conversion required

### Spike 2: DuckDB STRUCT Handling ✅ PASS
- 18/18 tests, max 87.1ms (threshold 100ms)
- 4+ nesting levels, array-of-STRUCT, UNNEST, NULL handling: all work
- **Findings:** `values` is reserved word; parameterized `?::JSON` is slow (use inline); UNNEST needs LATERAL subquery

---

## Coreify Touchpoints (14 total)

Concrete patterns discovered in Praxis that should become Core abstractions:

| # | Milestone | What Becomes Core |
|---|-----------|-------------------|
| 1 | M2 | DuckDB initialization → `core.datastore.duckdb` |
| 2 | M2 | Temporal view generator → `core.datastore.temporal` |
| 3 | M2 | Loader lifecycle → `core.datastore.loader` |
| 4 | M2 | Data quality framework → `core.datastore.quality` |
| 5 | M2 | Needs filter → `core.datastore.needs` |
| 6 | M2 | Lifecycle state machine → `core.utils.lifecycle` |
| 7 | M3 | Backtest engine patterns → `core.compute.backtest` |
| 8 | M3 | Function registry → `core.platform.registry` |
| 9 | M3 | Config schema patterns → `core.platform.config` |
| 10 | M3 | Diagnostic logger → `core.utils.logger` |
| 11 | M3 | Best-of-breed selection → `core.compute.providers` |
| 12 | M3 | Scheduler → `core.utils.scheduler` |
| 13 | M3 | Reconciliation engine → `core.compute.reconciliation` |
| 14 | M4+ | GUI adapters → Praxis CONSUMES Core (unique direction) |

---

*v1.0 — Created 2026-02-12 (Chat: praxis_main_current)*
