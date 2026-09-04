# RETRO — Cycle 65: Environment Reproducibility

**Date:** 2026-09-04
**Mode:** Code (audit + manifest repair + clean-build verification)
**Brief:** `claude/handoffs/BRIEF_cycle65_env_reproducibility.md` (`416a295`)
**Constraints honoured:** MCRMINI1's `.venv` untouched (the test venv was built in the
scratchpad); no collector, schema or scheduled-task changes; `crypto_data.db` and
`smart_money.db` not written; runs at ~17:00–17:30, far from the 03:00/03:30 windows.

---

## Headline

**Yes — a clean install now reproduces the environment.** A throwaway venv built from
`pyproject` alone collects **1425 tests and passes 1425**, identical to MCRMINI1, with the
single pre-existing `mctheory` collection error on both. Before this cycle that same clean
build was impossible: `python-dotenv` was in no manifest group at all.

Two things the brief expected turned out to be wrong, and both are worth stating plainly
because they would otherwise have been "fixed" without effect:

1. **pytest 9.0.3 vs 9.1.1 did not cause the test-count gap.**
2. **The plugin divergence (anyio + Faker + cov vs cov alone) did not either.**

The entire 1425-vs-1418 gap was the missing `python-dotenv`. Measured, not assumed — see T3.

---

## T1 — Every unmanifested dependency

Method: AST over every `.py` in `engines/ scripts/ services/ src/ servers/ tests/`, plus
**`gui/`**, which the brief's directory list omits — without it `fastapi`, `uvicorn` and
`streamlit` are falsely reported as declared-but-unused, since the Funding Studio and MCB
Studio backends are the only things importing them. Import names were mapped to
distribution names via `importlib.metadata.packages_distributions()`, and each import was
classified as module-level ("hard") or function/try-scoped ("lazy").

### P0 — undeclared, hard, and imported by SCHEDULED collectors

These are why **a clean install of Praxis was broken today.**

| Package | Sites | Scheduled collectors affected |
|---|---|---|
| `python-dotenv` | 51 (43 module-level) | `crypto_data_collector`, `funding_executor`, `live_collector`, `smart_money`, `funding_monitor`, `funding_regime_alert` |
| `requests` | 51 (36 module-level) | `crypto_data_collector`, `live_collector`, `smart_money` |
| `joblib` | 22 | `funding_monitor` (loads its models) |
| `websockets` | 2 | `liquidation_collector`, `bybit_liquidation_collector` |

`joblib` and `websockets` are function-level imports, but both sit in the execution path of
a scheduled task, so their absence breaks the task rather than degrading a feature. All four
are now **main** dependencies.

### P3 — undeclared but lazy: a feature degrades, the machine does not break

`torch`, `xgboost`, `arch`, `matplotlib`, `openai`, `anthropic`, `eth-account`, `psycopg2`.
Every one is function-level or try-guarded. Corroborating that they are genuinely optional:
**`torch`, `xgboost`, `arch` and `psycopg2` are not installed on MCRMINI1 at all**, so those
paths have never been exercised here. Now declared in new `ml` / `ai` / `viz` / `postgres`
groups, with `eth-account` added to `onchain`.

`psycopg2` deserves a note: `src/praxis/live/__init__.py` already catches the `ImportError`
and tells the operator to `pip install psycopg2-binary`. That was correct behaviour, not a
bug — it is declared now so the advice is satisfiable from the manifest.

### Test-only

Only two, and **neither is a real missing dependency** — see T4.

### False positives worth recording

`data_feed`, `controller`, `batch_engine` and `market_data` first appeared as undeclared
third-party imports. They are all **local**: the first three are sibling modules inside
`gui/*/backend/`, and `market_data/` is a git-tracked package at the repo root. An audit
that trusts `packages_distributions()` alone will invent PyPI names for these
(`dist=data-feed`), so each was checked against the filesystem before being reported.

### Declared but never imported

`pyarrow` (main), `pytest-cov`, `ruff` (dev), `python-multipart` (backend). All legitimate:
`pyarrow` is used through polars/pandas rather than imported directly, and the other three
are tooling or framework plumbing that is never `import`ed by name. **No removals made.**

---

## T2 — Pinning policy

**Runtime dependencies: lower bound plus a major cap** — `">=X.Y,<MAJOR+1"`. Patch and
minor upgrades stay routine; a breaking major cannot arrive unannounced. Pinning runtime
deps to exact versions would make routine maintenance a merge conflict, which the brief
explicitly warned against.

**Test tooling: pinned to the minor** — `pytest>=9.1,<9.2`, `pytest-cov>=7.0,<8`. Two
machines collecting different test counts is the failure this cycle exists to remove, and a
pytest *minor* is exactly the kind of change that can alter collection.

**One honest caveat, recorded in the manifest itself so the pin is not mistaken for a fix:**
the observed count gap was *not* caused by pytest or the plugins. This pin is precautionary
reproducibility, not the remedy for the symptom that prompted it.

**On the plugin divergence specifically:** `anyio` and `Faker` are transitive, not direct,
dependencies, and they were measured to be inert (T3). They were therefore **not** added as
explicit test dependencies. Adding them purely to make two `pip list` outputs match would be
cargo-culting a difference that provably changes nothing. The clean venv resolves to
`cov` only — the same plugin set as MCRMINI2 — and still collects 1425.

`pytest>=9.1` was chosen deliberately over MCRMINI1's current 9.0.3 so the estate converges
on the *newer* pytest rather than forcing MCRMINI2 to downgrade. The clean build then
verified that 9.1.1 collects and passes exactly the same 1425 tests.

---

## T3 — The 7-test gap: named, and explanation (b)

Both candidate explanations were tested rather than argued.

**Hypothesis (a), plugin-driven collection — disproved.** Disabling both plugins MCRMINI2
lacks changes nothing:

```
baseline                          : 1425 collected, 1 error
-p no:anyio -p no:faker           : 1425 collected, 1 error
```

**Hypothesis (b), tests that genuinely do not run — confirmed.** Simulating MCRMINI2 by
making `dotenv` unimportable reproduces its number exactly:

```
dotenv blocked : 1418 collected, 2 errors
                 ERROR tests/test_atlas_sync.py     <- new
                 ERROR tests/test_market_data.py    <- pre-existing
```

1425 − 1418 = 7, and all seven live in `tests/test_atlas_sync.py`:

1. `test_parser_determinism_trading_atlas`
2. `test_parser_determinism_pma`
3. `test_round_trip_stability`
4. `test_md_hash_sensitivity`
5. `test_schema_validation`
6. `test_regime_matrix_counts`
7. `test_trading_atlas_skips_pending`

Chain: `tests/test_atlas_sync.py:19` → `from engines.atlas_sync import …` →
`engines/atlas_sync.py:39` → `from dotenv import load_dotenv` (hard, module-level).

**This is a real coverage hole, not an artifact.** MCRMINI2 was silently running seven fewer
tests — and they cover atlas parser determinism and schema validation, so the hole was not
in a trivial area. All seven now pass in the clean venv.

---

## T4 — `mctheory`: a stale cross-project reference, not a missing package

`tests/test_market_data.py` lines 24–26:

```python
CORE_SRC = Path(__file__).parent.parent.parent / "core_repo" / "src"
sys.path.insert(0, str(CORE_SRC))
from mctheory.core.datastore import DataStore, DataTable, DataView
```

- `mctheory` is **not** a PyPI package and **not** installed anywhere.
- The `sys.path` target resolves to `McTheoryApps/core_repo/src`, and **`core_repo` does not
  exist on this machine.** The siblings of `praxis/` are `ai-agent-factory`,
  `ai-agent-factory_github`, and `praxis.fresh.backup` — no `core_repo`.

So this is a **stale reference to a separate McTheory repository that is not present**,
reached by `sys.path` manipulation rather than by any declared dependency. That is why it
fails identically on MCRMINI1, MCRMINI2 and the clean venv — it is not a handover issue.

Per the brief, this should be **removed, not satisfied**: a Praxis test reaching into
another project's source tree is a cross-project import, and installing something to silence
it would make Praxis's test suite depend on a repo that is not checked out. **Nothing was
installed and nothing was changed** — the file is left failing, visibly, pending a decision
on whether `test_market_data.py` belongs in Praxis at all. Note it is only the `mctheory`
line that fails; the `market_data.*` imports beneath it are local and fine, so if the
`mctheory`-dependent tests were split out, the rest of the module would collect.

---

## T5 — Clean build: the acting layer

```
python -m venv <scratch>/cleanvenv        # MCRMINI1 .venv untouched
pip install -e ".[dev]"                   # from pyproject alone
```

Resolved: **pytest 9.1.1**, plugins **`cov` only** — i.e. MCRMINI2's pytest version and
MCRMINI2's plugin set.

| | collected | passed | failed | errors |
|---|---|---|---|---|
| MCRMINI1 `.venv` (pytest 9.0.3, anyio+Faker+cov) | 1425 | **1425** | 0 | 1 (`mctheory`) |
| Clean venv (pytest 9.1.1, cov only) | **1425** | **1425** | 0 | 1 (`mctheory`) |
| MCRMINI2 (reported, pre-fix) | 1418 | — | — | — |

The clean venv reaches MCRMINI1's numbers **while running MCRMINI2's pytest and plugin
set**, which is what closes the question: neither the version nor the plugins were ever the
cause.

Direct confirmation of the P0 fix in the clean venv:

```
from dotenv import load_dotenv; import requests, joblib, websockets   -> all OK
tests/test_atlas_sync.py                                             -> 7 passed
```

### A flaky test found by running the suite twice

The **first** clean-venv full run reported `1 failed, 1424 passed` —
`tests/test_battle_round3.py::TestSurfaceIntegration::test_surface_interpolated_query`. The
second run passed 1425. Reporting only the second would have been convenient and wrong, so
the cause was traced.

`src/praxis/stats/surface.py:695–700`:

```python
current_seed = seed
for ...:
    universe = factory.create(n_obs, n_assets, seed=current_seed)
    current_seed = None          # Only seed the first batch
```

**Only the first batch is seeded; every later batch draws from the global RNG.** So
`compute()` is not deterministic despite `seed=42`, and its result depends on whatever ran
before it — which is why the test passes in isolation but can fail inside the full suite.
Demonstrated by varying only the *preceding* global state, with `seed=42` passed identically
every time:

```
 global      mid_5       avg   spread  assert
      0    -5.3431   -5.3080   0.2101  pass
      3    -5.3329   -5.2920   0.1435  pass
      4    -5.1027   -5.2664   0.0397  FAIL
      9    -5.2571   -5.3389   0.0441  pass
                                        -> 1 failure in 12
```

Two compounding defects: the seeding above, and a tolerance of `spread * 3.0` that collapses
when `spread` is itself small by chance (0.0397 at global=4 gives a bar of 0.119 against an
error of 0.164).

**Not fixed here.** Changing the seeding alters numerical output for every consumer of the
surface module, which is a behaviour change well outside an environment-reproducibility
brief and deserves its own decision. It is a **pre-existing defect, independent of this
cycle's work** — but it does bound the claim: test *counts* are now reproducible across
machines; test *results* carry a ~8% flake on this one test until the seeding is fixed.

---

## Files changed

| File | Change |
|---|---|
| `pyproject.toml` | 4 undeclared runtime deps added to main; new `ml` / `ai` / `viz` / `postgres` groups; `eth-account` added to `onchain`; major caps on all runtime deps; pytest pinned to the minor; pinning policy documented inline |
| `claude/handoffs/BRIEF_cycle65_env_reproducibility.md` | Brief (committed standalone, `416a295`) |

No source file was modified. No collector, schema or scheduled task was touched.

---

## Carried forward

1. **Decide what to do with `tests/test_market_data.py`.** It imports another repo's source
   via `sys.path`. Either drop the `mctheory`-dependent tests, or move the module to the
   project that owns `mctheory`. It has been failing on every machine for some time.
2. **Fix `surface.py` seeding** (`current_seed = None` after the first batch) and the
   `spread * 3.0` tolerance. Until then one test flakes at roughly 8%.
3. **`market_data/` is not packaged.** `[tool.setuptools.packages.find] where = ["src"]`
   covers `src/` only, so the repo-root `market_data/` package resolves today purely because
   the install is **editable**. A non-editable install would omit it. Not changed here —
   altering the packaging layout risks the working editable install and is not a dependency
   fix — but it is a genuine reproducibility gap.
4. Consider a lockfile (`uv.lock` / `pip-tools`) if bit-identical environments are wanted.
   The current policy makes environments *compatible*, not *identical*; minor upgrades still
   differ between machines, which is deliberate but is a choice worth revisiting.
5. Re-run Cycle 64's crash-safety verification on MCRMINI2 once it is built from this
   manifest. That was the blocker this cycle existed to clear.

---

*Cycle 65 — MCRMINI1's `.venv` was never modified; the verification venv was built in the
scratchpad and is disposable. No database was written.*
