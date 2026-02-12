# McTheory Praxis

**Universal Trading Platform — Quantitative Finance Infrastructure**

McTheory Praxis is a YAML-driven trading platform built on bi-temporal data architecture (DuckDB + Polars), declarative model parameterization, and a function registry that routes configuration to code. It implements the "Two-User Test": a high schooler can run an SMA crossover from YAML, and a quant can configure Chan CPO with ML-based parameter optimization — same platform, different config.

## Architecture

```
YAML Config → Pydantic Validation → Function Registry → Model Executor → DuckDB Storage
                                                              ↓
                                                    Backtest Engine (vectorized / event-driven)
```

### Core Principles

- **Config-driven, not code-driven.** Models are YAML documents, not Python scripts.
- **Bi-temporal everything.** AS-IS (current knowledge) and AS-WAS (historical state) queries on all dimension data.
- **DuckDB STRUCTs over JSON.** Typed, queryable, nested configuration at the storage level.
- **Temporal views, never direct table access.** `vew_` (current), `vt2_` (point-in-time), `rpt_` (reports).
- **Universal key pattern.** Every dimension table uses `_bpk` / `_base_id` / `_hist_id`.

### McTheory Ecosystem

| Project | Purpose | Dependency |
|---------|---------|------------|
| **McTheory Core** | Universal abstraction layers (datastore, compute, render, platform) | Foundation |
| **McTheory Praxis** | Trading platform (this repo) | Builds on Core |
| **AI Agent Factory** | Self-evolving AI agent system | Builds on Core |

Praxis builds concrete implementations first, then abstracts successful patterns into Core through a documented "Coreify" process with 14 touchpoints.

## Current Status

**Phase 0: Spikes & Scaffolding** ✅ Complete

| Spike | Result | Key Finding |
|-------|--------|-------------|
| vt2_ View Performance | ✅ PASS (5-200× margin) | xxHash64 needs uint64→int64 signed conversion |
| DuckDB STRUCT Handling | ✅ PASS (18/18 tests, <100ms) | UNNEST needs LATERAL pattern; inline JSON casting |

**Phase 1: Thin Vertical Slice** 🔧 In Progress
- Deliverable 1.1: _bpk/_base_id infrastructure ✅ (63 tests)
- Deliverable 1.2: Diagnostic Logger — next

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/mctheory-praxis.git
cd mctheory-praxis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run spikes (validation tests)
python spikes/spike_01_vt2_performance.py
python spikes/spike_02_struct_handling.py
```

## Project Structure

```
mctheory-praxis/
├── src/praxis/                  # Main source
│   ├── datastore/               # DuckDB, temporal views, keys
│   │   └── keys.py              # Universal _bpk/_base_id/_hist_id
│   └── utils/                   # Shared utilities
├── spikes/                      # Assumption validation scripts
│   ├── spike_01_vt2_performance.py
│   ├── spike_01_results.md
│   ├── spike_02_struct_handling.py
│   └── spike_02_results.md
├── tests/                       # Test suite
│   └── test_keys.py             # 63 tests for key infrastructure
├── docs/                        # Project documentation
│   ├── praxis_main_series.md    # Chat series overview
│   └── praxis_main_series_transcript.md
├── claude/chat/                 # Claude chat infrastructure
├── pyproject.toml               # Project metadata & dependencies
└── README.md
```

## Development Phases

| Phase | Goal | Status |
|-------|------|--------|
| 0 | Spike validation + scaffolding | ✅ |
| 1 | `praxis run sma_crossover.yaml` end-to-end | 🔧 |
| 2 | DataStore & Security Master | ⬜ |
| 3 | Chan CPO — full architecture validation | ⬜ |
| 4 | Burgess Stat Arb & GUI | ⬜ |
| 5 | Live trading & advanced features | ⬜ |

## Specification

The full specification (v9.3.1, 3,457 lines, 23 Parts) and execution plan (v1.2, 883 lines) are maintained in the `praxis_main` chat series docs.

## License

MIT

---

*v0.1.0 — Phase 0 complete, Phase 1.1 delivered*
