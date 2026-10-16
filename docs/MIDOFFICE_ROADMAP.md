# Mid-Office Roadmap — Audited Risk Workflow

**Status:** planning. No code yet.
**Owner:** Jeff (Chat: praxis_main_current)
**Created:** 2026-10-16

---

## What this is, and what it is for

A market/credit/liquidity risk and regulatory reporting stack built inside Praxis, used as
the test case for **formalised agent orchestration with full audit transparency and
mid-workflow signoff**.

The ordering of those two is deliberate and is the whole point. Most agent work builds the
agent first and bolts audit on afterwards. In mid-office the audit frame comes first and the
agent is a **constrained participant in an audited process**, not the process itself. Build
a workflow that is tight, transparent and reconstructible, and automating it with an agent
is nearly free. Build the agent first and the audit layer never fits.

**Primary purpose is interview capability**, not a product. That changes what "done" means:
an interviewer has thirty minutes and is probing judgement, not coverage. What lands is not
"I built an audited risk workflow" — that is a claim — but a specific hard decision explained
end to end. *The analyst rejects at step 7 of 12. Here is what happens to the graph state,
here is why I forked rather than rolled back, here is the query proving the rejected branch
is still reconstructible.* Nobody bluffs that.

So: **the build is organised around manufacturing two or three of those moments.** Depth on
the awkward parts. Whole subsystems may stay stubs.

Demo polish comes later. Product comes later, if ever.

**Learning is led by the build, not the other way round.** Coursera material running
concurrently should land on problems already felt — checkpointing, reducers and interrupt
semantics mean something different after a reconstruction has failed for want of state that
was never persisted. Hit the wall, then learn the named solution, then refactor to it.

---

## The one architectural rule

**Mid-office code never imports from `engines/` or any trading-side module.** It reaches
data through an explicit access interface, however thin and ugly that interface starts out.

Everything stays in Praxis for now, because the workflow and the data feeding it come from
Praxis and one roof is simpler at POC stage. The eventual split — **trading / data management
/ mid-office** — is cheap if built to the seam from day one and expensive if not.

Second benefit, and the reason this rule earns its place twice: an explicit interface is a
**place to record what was read and when**. The same discipline that makes the split cheap
makes the lineage tractable.

---

## Two workstreams, one interface

The risk engine and the orchestration spine are independent. Settle the interface early,
stub the engine behind it, and neither blocks the other. Switch between them by available
energy rather than by dependency.

### Workstream A — port the risk engine

A Python port of the existing C# factor risk engine, which is generic by design and ran
against Barra models. Barra is not available here; the engine stays generic and an S&P 500
model becomes its first instance (see below).

**Porting rules, in priority order:**

1. **No implicit time dependence.** No `now()`, no "latest available" lookup, nothing that
   resolves differently on a different day. Everything as-of, passed in. Production engines
   are usually full of implicit latest-ness because they run today against today's data —
   expect to find it and strip it. **This is what makes reconstruction possible and it is
   agonising to retrofit.**
2. **Pure computation.** Same inputs, same outputs. No hidden state, no I/O inside the maths.
   Data access happens at the boundary, never in the middle of a calculation.
3. **The model is data, not code.** The engine must not know which factor model it is running.
   The C# already had this boundary; preserve it.
4. **Port, do not redesign.** Design decisions in the original were made once and validated.
   Improvements are a later pass with a parity test to protect them.

**Acceptance: golden-file parity.** The original still runs - it produced daily risk reports
on roughly a thousand funds for fifteen years - and a SQL Server backup holding Barra data may
be recoverable. Parity on a stated test portfolio to a stated tolerance is therefore the bar,
and it is the strongest one available.

**Recover that backup early.** Parity against fifteen years of production output is worth more
than everything else in phase 1 combined, and old backups decay as an option: a `.bak` from a
sufficiently old SQL Server will not restore onto a modern instance and needs a hop through an
intermediate version, which needs that intermediate version to still be obtainable. If it
proves unrecoverable, acceptance falls back to analytic checks (single-factor portfolio
variance, known weights against hand calculation, a zero-exposure portfolio) plus internal
consistency - materially weaker, and worth saying so.

**Provenance:** the C# is Jeff's own, written before the TIAA-CREF engagement and supplied to
them under a body-shop arrangement in which software came free and on-site improvements
remained his.

**Worked example of why rules 1 and 2 matter.** The existing S&P 500 historical-constituents
builder (Wikipedia changes table, walked backwards) is sound in its core algorithm - reversing
each change going back does correctly resurrect names added and later dropped, which is what
most free approaches get wrong. It also carries three defects that are exactly the shape this
port has to eliminate:

1. **A silent type bug.** `changeDf.loc[str(d)]` returns a DataFrame when a date has several
   changes and a **Series** when it has one. In the single-change case `['removed']` is a bare
   string, so the loop iterates **characters**: `WHR` appends `'W'`, `'H'`, `'R'`. The `added`
   side is guarded by an `in` test so it no-ops; the `removed` side is not. `sorted(set(...))`
   then tidies the result into something that still looks like a ticker list. Several single
   letters (`F`, `T`, `C`, `K`, `V`, `X`, `A`, `M`) are real large-cap tickers that join
   cleanly against a price database, so the contamination is partly invisible even downstream.
   Fix by forcing the DataFrame shape (`.loc[[d]]`, or filter and `.itertuples()`), and **test
   both a single-change and a multi-change date** - a test with only the multi case passes.
   *Free control that can fail:* interval list lengths should sit near 500 and must not grow
   walking backwards.
2. **Implicit time dependence.** It reads Wikipedia live and calls `Timestamp.now()`, so the
   answer changes when Wikipedia edits its table and last month's run cannot be reproduced.
   Becomes: fetch once, snapshot with timestamp and content hash, all downstream reads against
   the snapshot. That also contains the `data[1]` MultiIndex fragility, since the table is
   restructured periodically.
3. **Confident extrapolation past coverage.** The earliest known interval is stretched back to
   1900, so a query outside the changes table's real coverage returns a guess rather than
   refusing. Irrelevant at three years, silently wrong the moment the window extends. **This is
   a data-quality gate asking to be written.**

The manual ticker-override map is a judgement artifact: version it and record it in lineage,
because changing it changes historical answers.

**Scope for round one:** equities only. Fixed income and derivatives are a later round.

### Workstream B — the orchestration spine

Run object with an ID and as-of date; immutable, addressable market-data snapshot; positions;
risk call; exception check; report. Plain deterministic Python on Core/Datastore. No LLM, no
orchestration framework yet.

This workstream is also **the first real stress test of McTheory Core building from Datastore
classes**, which is a stated goal in its own right.

It proceeds against a stubbed engine returning plausible numbers. It does not wait for A.

### Where they meet

```
compute_risk(as_of, portfolio_id, model_id, snapshot_id) -> RiskResult
```

- `snapshot_id` makes the inputs immutable and addressable.
- `model_id` pins the factor model version.
- `RiskResult` carries its own lineage: which snapshot, which model, which code version.
- The call is pure. Same four arguments, same result, forever.

Define this before either workstream is far along. It is the seam, the stub boundary, and the
first lineage record all at once.

---

## The factor model is a third, separate thing

Not part of the engine and not on its critical path.

Built by a **factor model builder**, which is its own component. The builder's contract is
narrow and fixed:

```
build_model(as_of, universe, spec) -> (exposures, factor_covariance, specific_risk)
```

How it arrived at those three is entirely its own business. That contract is what lets both
model families coexist without the engine changing:

- **Cross-sectional / fundamental (Barra-style):** observable descriptors are the exposures;
  factor returns estimated by cross-sectional regression each period. This is what the C#
  engine was built against.
- **Time-series (Fama-French style):** observable factor return series; asset betas estimated
  by regression become the exposure matrix.

**First model: Fama-French.** Crude on purpose, to exercise the engine and unblock the spine -
and it carries the same bonus the engine port has. French publishes the factor series, so the
first model arrives with an **external reference to validate against**, not merely internal
consistency. Model quality then improves on its own axis without touching engine code. Do not
let model-building block engine-building.

**Data reality for an S&P 500 fundamental model:**

| Input | Difficulty |
|---|---|
| Prices, returns | Easy |
| Industry classification | Easy if a substitute is accepted; GICS itself is proprietary |
| Market cap / size | Easy |
| **Point-in-time index membership** | **Solved** by the existing Wikipedia changes-table builder, subject to the three defects noted under Workstream A |
| **As-reported fundamentals** | **Hard.** Current values are trivial; as-reported-at-the-time is not. Matters for any value factor |

Neither hard case blocks a POC. Both want a known, documented workaround rather than a
discovery at build time.

**The data problems are feature-generating.** A workflow whose audit layer flags stale,
estimated or substituted inputs is a *better* demonstration than one running on clean data —
surfacing exactly that is what this architecture is for. Lean into it; do not apologise for it.

---

## Phases

Sized relatively. There is no deadline; depth beats speed. Each phase has a "done" so it is
clear when to stop polishing and move.

**1 — First end-to-end number.** Workstreams A and B to the point where one risk figure is
produced from an immutable snapshot through a real model and lands in a report.
*Done when:* the same run ID reproduces the same number twice, and the engine meets its
acceptance bar.

**2 — Make it a graph.** Express the spine in LangGraph with persisted state. **Zero new
functionality** — this phase is pure learning.
*Done when:* the process can be killed mid-run and resumed.

**3 — The gate.** Human-in-the-loop interrupt at exception review. Then the hard version:
who approved, when, what exactly were they shown, what was the full state at that instant,
tamper-evidence, and the reject path. Partial rollback with audit continuity is the real
design problem here.
*Done when:* a rejection at a middle step leaves a trail defensible to a regulator.
**This phase gets the most time. It is the differentiator.**

**4 — Reconstruction.** Given a run ID from weeks ago, rebuild what was computed, from what
inputs, approved by whom, seeing what.
*Done when:* it works for a run whose details have been forgotten. **This is the proof and
the interview moment.**

**5 — Agents.** Exception triage, commentary drafting, "why did risk move" answered from
lineage. Last, deliberately, and defensibly so.

**6 — Breadth and garnish.** Credit, liquidity, FRTB treatment, fixed income, derivatives,
the Creator Agent GUI wizard. **The wizard waits until the graph schema has stopped moving**,
or it becomes an editor for a schema about to change.

---

## Open items

1. ~~Can the C# engine still be run?~~ **Resolved: yes.** Golden-file parity is the bar.
   Open sub-item: recover the SQL Server backup holding Barra data, early (see Workstream A).
2. ~~Provenance of the C# source.~~ **Resolved: Jeff's own** (see Workstream A).
3. **Which risk measure the pilot computes.** A plain VaR is inert. Something with a
   regulatory test attached — P&L attribution, a data-quality or modellability gate — is more
   interesting, because those are audit requirements expressed *as computations*, which is
   what this architecture exists to carry. Check current FRTB text before committing to
   specifics; the Basel crypto-asset treatment is its own regime if a crypto book is used as
   more than a stand-in.
4. **Fundamentals data source.** Needs pricing out when the model build approaches.
5. **Which portfolio.** S&P 500 equity book, the Praxis crypto book, or both. The crypto data
   is already owned, which removes all sourcing friction; the equity model is the one with
   the C# lineage behind it.

---

*Last updated: 2026-10-16 (Chat: praxis_main_current)*
*Changes: 2026-10-16 (2) - open items 1 and 2 resolved; factor model builder contract added
with Fama-French as the first model; the S&P 500 constituents builder added as the worked
example of the port rules, with its three defects. Initial roadmap. Phase 1 expanded from a thin spine to a full engine port at Jeff's
direction, on the grounds that the C# original supplies a cross-implementation oracle
stronger than self-consistency. Split into two parallel workstreams to keep the orchestration
work off the port's critical path.*
