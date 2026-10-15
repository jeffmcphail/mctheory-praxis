# BRIEF — Cycle 66: A2 Validation Against Labelled Liquidations

**Estimated Scope:** M (1–2 hr)
**Estimated Cost:** none (read-only against `crypto_data.db`)
**Mode:** Code (measurement only)
**Retro:** `claude/retros/RETRO_cycle66_a2_labelled_validation.md`
**Predecessor:** Cycle 62A (`624a2a9`) — Bybit liquidation collector adopted as T1 venue

---

## Why this is now possible, and was not before

Cycle 61 T1 built a liquidation-cascade detector from `trades` flow bursts and could not
compute a false-positive rate **because no ground truth existed**. A forced liquidation and
a large discretionary market order are identical in the tape. The retro said so plainly and
left the question open as uncomputable.

Cycle 62A closed that gap: the Bybit `allLiquidation` collector has been running since
**2026-08-19**, roughly eight weeks. **There is now ground truth.**

This cycle spends it on the question Cycle 61 could not answer.

### What the Cycle 61 detector found, and why Chat expects a refutation

The detector found real, clustered, cross-asset-correlated sub-minute flow bursts —
dispersion index 3.79–8.51 against Poisson 1.0, cross-asset lift 8–12×. But three
measurements said they were **not cascades**:

| Evidence | Value |
|---|---|
| Order-book stress vs matched control | depth ratio 1.02, spread ratio 1.00 — **no stress** |
| Survival at longer windows | 229 events at W=60s → **1** at W=300s → **0** at W=900s |
| Concentration on largest-move days | lift **0.29** (BTC) / **0.47** (ETH) — **below chance** |
| Threshold sensitivity | 80× (BTC) / 228× (ETH) event-count spread, **no plateau** |

**Chat's pre-stated expectation: the labels will confirm the detector was finding volume
bursts, not liquidation cascades.** Recording that before the measurement so the result is
a test rather than a story. **A clean refutation is a result, not a failure** — it closes an
open "cannot be determined" in the atlas, which is worth more than leaving it unresolved.

---

## T1 — Inventory the labelled data

Before any modelling, establish what actually exists.

- Row count, date span, and per-symbol breakdown of `liquidations` (expect 6 symbols: BTC,
  ETH, SOL, XRP, ADA, AVAX — the funding/OI universe; Bybit has no all-market topic, so
  anything outside it is **unobserved, not quiet**).
- **Collection gaps.** Query `collector_gaps` for the liquidation collector. Any gap still
  open, and any window where the task did not run, must be **excluded from both the event
  set and the control set** — otherwise an outage reads as a quiet market.
- **Cluster count, which is the number that decides whether T3 is viable.** Individual
  events are plentiful (4,973 in ~12 minutes during one cascade on 2026-08-19). Cascades are
  the unit of analysis and are bursty. Define a cluster as a parameterised gap-based
  grouping — e.g. events within N seconds of each other, N a parameter — and report the
  cluster count at 2–3 settings per symbol, with the size distribution.
- **Report the usable overlap window** between `liquidations` and `trades` /
  `order_book_snapshots`. `trades` covers **BTC and ETH only**; the other four symbols have
  labels but no flow or depth data.

**If the BTC+ETH labelled cluster count is small (say under ~50), say so plainly.** T2 is
still worth running on whatever exists; T3 is not.

## T2 — The confusion matrix Cycle 61 could not compute

This is the high-value, low-cost step. Re-run the Cycle 61 detector over the labelled window
and score it against the labels.

- Re-run `engines/forced_trade/`'s detector unchanged, at the **same threshold settings**
  Cycle 61 reported. Do not retune it — the question is how the detector *as built*
  performs, not how well a tuned version could.
- Build the confusion matrix: detector-fires × label-present, with a stated matching
  tolerance (a parameter — detector events and liquidation timestamps will not align to the
  millisecond).
- Report **precision, recall and false-positive rate** per symbol, at each threshold setting.
- **Then the inverse, which is the more interesting direction:** take the labelled cascades
  and ask what the detector saw. Did it fire? If real cascades produce no flow-burst
  signature, that says something different from "the detector fires on noise" — and it is
  the result that would actually rehabilitate A2 as a scenario.
- Re-run the **order-book stress test against labelled events**. Cycle 61 found depth ratio
  1.02 on detector events. If labelled liquidations *do* show depth collapse where detector
  events did not, that is the cleanest possible confirmation that the detector was measuring
  the wrong thing.

## T3 — Characterise real cascades (only if T1 says the sample supports it)

With labels in hand, describe what a cascade actually looks like — the input any future
detector needs.

- Event-time profile around labelled clusters: price, signed aggressive flow, book depth,
  spread. Before, during, after.
- **Matched controls**, same discipline as always: same time-of-day, same volatility bucket,
  drawn from the same tape. Cycle 61 T1 produced a spurious above-1.0 ratio purely by
  comparing a short-window mean against a 24h median of a right-skewed series. **The matched
  control is the number to read.**
- Size, duration and reversion: does price revert after a labelled cascade, over what
  horizon, and by how much in bps?
- **Report gross bps first.** Net of a cost model second. The reversion magnitude is what
  decides whether A2 is worth any further cycles at all.

## T4 — Scope the path-signature question (do NOT implement)

Signatures encode the **order** of moves across channels, not just magnitude — which is
exactly the gap in a detector built from scalar summary statistics. Potentially the right
tool for A2. But it is a separate cycle with its own pre-registration, and this brief only
sizes it.

Report, as a paragraph each:

- Labelled cluster count available for BTC and ETH specifically — the supervised sample size.
- Which channels are available at what granularity for the full path (price, signed flow,
  book depth, spread), and over what window.
- **The dimensionality the problem would carry.** A level-4 signature on 5 channels is 780
  terms. State the term count at 2–3 (channel, level) combinations against the available
  sample size. **If features outnumber labelled clusters by an order of magnitude, say so** —
  that is the finding, and it would mean the signature study is premature.
- Current state of the Python tooling (`iisignature`, `signax`, `signatory` or successors):
  what installs cleanly on this estate's Python 3.12. **Check, do not assume** — Chat does
  not know the current maintenance status of these packages.

**No implementation, no feature extraction, no model. Sizing only.**

---

## Constraints

- **Read-only.** No writes to `crypto_data.db`. No `VACUUM`, no migration — the database is
  24 GiB and the estate handover is unfinished.
- Do not retune the Cycle 61 detector. T2 measures it as built.
- Everything a parameter: cluster gap, matching tolerance, thresholds. `--validate` /
  `--verbose` with levels, defaulting to maximum.
- Keep runs away from **03:00 and 03:30** — the mirror and backup windows on this estate.
- If a step is blocked by insufficient data, **stop and report the count** rather than
  proceeding on a thin sample.

## Explicitly NOT in scope

- No strategy, no backtest, no Sharpe
- No signature implementation (T4 is sizing only)
- No detector retuning or redesign
- No scenario × regime grid work

---

## Hand back

Retro containing, in this order:

1. **T1 inventory** — rows, span, per-symbol, gaps excluded, and the **cluster counts**
2. **T2 confusion matrix** — precision / recall / FPR per symbol per threshold, both
   directions, plus the order-book stress test against labelled events
3. T3 cascade characterisation with matched controls, gross bps first — or an explicit
   "sample too small" with the number
4. T4 sizing: cluster count, channels, term counts, tooling state
5. **A plain verdict on A2**: does the Cycle 61 detector work, do real cascades show
   exploitable reversion, and is the signature study worth a cycle?

Final step, standing:

```
.venv\Scripts\python split_zip.py zip --repo-delta
```

---

*Last updated: 2026-10-15 (Chat: praxis_main_current)*
*Changes: Initial brief. Cycle 66 spends the eight weeks of Bybit liquidation labels on the
false-positive rate Cycle 61 recorded as uncomputable, characterises real cascades against
matched controls, and sizes the path-signature question without implementing it.*
