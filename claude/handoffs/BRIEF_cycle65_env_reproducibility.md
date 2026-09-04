# BRIEF — Cycle 65: Environment Reproducibility

**Estimated Scope:** S–M (30–90 min)
**Estimated Cost:** none
**Mode:** Code (audit + manifest repair + clean-build verification)
**Retro:** `claude/retros/RETRO_cycle65_env_reproducibility.md`
**Blocks:** re-verification of Cycle 64's crash-safety claims on MCRMINI2.

---

## Why now

The MCRMINI1/MCRMINI2 comparison found four divergences, and every one traces to
something installed outside the manifest:

- **dotenv** — fails on MCRMINI2 only, so `python-dotenv` is an ad-hoc install on
  MCRMINI1, absent from every `pyproject` group
- **pytest** 9.0.3 vs 9.1.1
- **plugins** — `anyio` + `Faker` + `cov` vs `cov` alone
- **1425 tests collected vs 1418**

Consequence: test counts are not comparable across machines, and Cycle 64's crash-safety
verification — which asserted exit codes, retained row counts and gap rows — ran entirely
on MCRMINI1. The logic it tested is fine; **the claim is not portable** until a clean
install reproduces the environment.

The dotenv case is the serious one. `load_dotenv()` before accessing any key is a standing
Praxis rule, so `python-dotenv` is a hard **runtime** dependency, not a test one. Its
absence from the manifest means **a clean install of Praxis is broken today.** MCRMINI2 is
not a new problem; it is the first machine to report an old one.

---

## T1 — Find every unmanifested dependency, not just dotenv

Do not fix dotenv and stop. It was found by accident; assume there are others.

- Enumerate every third-party import across `engines/`, `scripts/`, `services/`, `src/`,
  `servers/` and `tests/`.
- Diff that set against what `pyproject.toml` declares, across **all** groups.
- Report anything imported but undeclared, and anything declared but unused.
- Flag which are **runtime** (a collector or engine imports it) versus **test-only**. A
  missing runtime dep breaks the machine; a missing test dep breaks the count. Different
  severities — state both.

## T2 — Pin the manifest so the environment is reproducible

- Add every missing dependency to the correct `pyproject` group.
- `pytest` and its plugins are the reason two machines collect different test counts — pin
  them explicitly rather than letting resolution drift.
- Do **not** pin so hard that routine upgrades break. State the pinning policy chosen and
  why.

## T3 — Account for the 7-test collection gap

1425 vs 1418 has two very different explanations and the count alone cannot distinguish
them:

- **(a) plugin-driven collection** — `anyio`/`Faker` generating or enabling tests
- **(b) tests that genuinely do not run on MCRMINI2**

Identify **which 7**. Name them. (b) is a real coverage hole; (a) is an artifact.

## T4 — The `mctheory` import failure (separate problem, do not conflate)

Fails on **both** machines, so it is not a handover issue. A test imports a package no
manifest provides.

- Which test, which import, and is `mctheory` a real external package, a different local
  project, or a stale reference?
- If it is another McTheory project, that is a cross-project import inside Praxis tests and
  should be **removed, not satisfied**.

Report the cause; do not install anything to make it pass without saying what it is.

## T5 — Verify by clean build, not by reading the manifest

The whole point is reproducibility, so prove it:

- Build a throwaway venv from `pyproject` alone, in a temp location.
- Run the full suite in it.
- Report collected / passed / failed and compare against MCRMINI1's current numbers.

A manifest that looks complete is the describing layer. A clean venv that runs the suite is
the acting layer.

---

## Constraints

- **Do not touch the existing `.venv` on MCRMINI1**; build the test venv elsewhere.
- No changes to collector behaviour, schemas or scheduled tasks.
- Read-only against `crypto_data.db` and `smart_money.db`.
- Keep any test run away from **03:00 and 03:30** (mirror and backup windows).

---

## Hand back

The undeclared-dependency list split runtime vs test-only, the 7 named tests and which
explanation they fall under, the `mctheory` cause, and the clean-venv run result compared
against MCRMINI1. **State plainly whether a clean install now reproduces the environment.**

Final step, standing:

```
.venv\Scripts\python split_zip.py zip --repo-delta
```

---

*Last updated: 2026-09-04 (Chat: praxis_main_current)*
*Changes: Initial brief. Cycle 65 addresses the four MCRMINI1/MCRMINI2 environment
divergences, all of which trace to dependencies installed outside the manifest.*
