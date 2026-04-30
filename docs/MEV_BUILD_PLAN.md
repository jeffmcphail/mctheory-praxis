# MEV Build Plan

> **About this file**: The ordered roadmap for Maximum Extractable Value
> (MEV) work in Praxis. Phases listed here are sequenced by Jeff's
> preferred build order. Adding new phases or re-ordering requires Jeff's
> explicit go-ahead.
>
> Standalone document (not a TODO bullet) because the phase ordering is
> stable project knowledge and likely to grow as each phase produces its
> own detailed sub-plan over time.

---

## Phase ordering

| Phase | Name | Status | Notes |
|-------|------|--------|-------|
| 1 | MEV Scanner | **BUILT** | Pre-recovery. Identifies opportunities; passive, no execution. |
| 1c-AI | Event-Driven Spike Predictor | Not started | Deep learning model. Classifies event types; predicts remaining move using price + news + social velocity. Transformer architecture. Slot AFTER Phase 2 executor (intentional ordering -- need executor in place to test predictions live). |
| 2 | MEV Executor | Not started | Live execution path for opportunities surfaced by Phase 1 scanner. |
| 3 | AI Combinatorial Engine | Not started | Multi-opportunity composition. |
| 4 | Cross-Chain Monitor | Not started | Extend scanner across chains. |
| 5 | Cross-Chain Executor | Not started | Execute the cross-chain opportunities Phase 4 surfaces. |
| -- | DEX arb + flash loans | After Phase 5 | Once cross-chain execution is in place, layer DEX arb (with flash loan capital) on top. |

---

## Notes on the ordering

Phase 1c-AI (Spike Predictor) lands AFTER Phase 2 (Executor) is built.
The reasoning: a predictor without an executor produces signals you
can't act on -- the prediction quality can't be validated in production.
Building the executor first means the predictor's output has an
immediate use case and a feedback loop.

DEX arb + flash loans is the final layer because flash loan composability
is most valuable when there's a multi-chain executor to combine it with.
A single-chain flash loan arb is a smaller market and a less interesting
build than the cross-chain version.

---

## Phase-specific detail

(Each phase will get a subsection here as it becomes the active build
target. For now, this file just documents the ordering.)

### Phase 1 -- MEV Scanner (built)

(Brief notes about the existing scanner here once we re-document it.)

### Phase 1c-AI -- Event-Driven Spike Predictor (not started)

Architecture sketch (subject to revision before Brief):
- Input features: price velocity, news velocity, social velocity (X /
  Discord / Telegram signal density)
- Event-type classifier: market-moving event categories (token unlock,
  exploit, exchange listing, regulatory action, etc.)
- Output: probability distribution over remaining-move buckets
- Model class: transformer-based
- Training data: scanner-recorded events + price action

### Phase 2 -- MEV Executor (not started)

(To be detailed when Phase 1c-AI design is closer.)

### Phase 3 -- AI Combinatorial Engine (not started)

### Phase 4 -- Cross-Chain Monitor (not started)

### Phase 5 -- Cross-Chain Executor (not started)

### DEX arb + flash loans (after Phase 5)

---

## Revision history

| Date | Cycle | Change |
|---|---|---|
| 2026-04-30 | 16 | Initial. Captured 5-phase ordering from memory entry 16. Phase 1 scanner status flagged as BUILT pre-recovery. |
