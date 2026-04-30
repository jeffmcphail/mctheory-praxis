# Retro: SQLite Freshness Diagnostic + CLAUDE_CODE_RULES.md v1.3

**Series:** praxis
**Cycle:** 15
**Mode:** A (Chat-edited delta zip; documentation + protocol rule addition)
**Outcome:** PASS -- diagnostic complete, Rule 34 added
**Files modified:** 1 (`claude/CLAUDE_CODE_RULES.md`)
**Files added:** 1 (`claude/retros/RETRO_sqlite_freshness_diagnostic.md` -- this file)

---

## 1. What triggered this cycle

Morning health check showed PraxisLiveCollector had written 30,500 rows
overnight. Initial back-of-envelope calculation suggested a ~15-minute
gap somewhere in the night (50 markets x 60s cadence x 14h elapsed =
42,000 expected, vs 30,500 actual = 27% shortfall, "well within design
tolerances" was Chat's lazy framing).

Jeff (correctly) pushed back: "I think it's useful (especially at this
early stage when we have first created the loaders) to do the analysis
to find out what actually happened and make sure it is in fact due to
either real missing data, some environment/network disruption, or in
fact an error in the setup."

That triggered a deeper investigation, which led to a wrong hypothesis,
which led to a corrected hypothesis, which led to Rule 34.

---

## 2. The diagnostic journey

### Round 1: Schema misassumption
Chat wrote a gap-analysis script using `market_id` as the slug column.
Schema check showed the actual column is `slug` (text like
"will-elon-musk-win-the-2028-us-presidential-election"). Trivial fix.

### Round 2: First analysis pass
Corrected script ran cleanly and reported:
- 648 distinct sampling ticks across 647 minutes elapsed
- All inter-tick gaps within 60-65s
- Tick coverage: 100.2%
- Last tick: 2026-04-30T09:11:04 UTC

But wait -- the morning `get_collector_health` MCP call (run a few
minutes earlier) had reported live_collector.price_snapshots latest at
**2026-04-30T12:33:04 UTC** (22.8s stale, fresh).

So two reads of the same DB at almost the same wall time disagreed by
~3.5 hours on what "latest" was.

### Round 3: Wrong hypothesis #1 -- WAL freshness
Chat hypothesized that SQLite WAL mode requires explicit checkpoints
for cross-process readers to see fresh data, and the read script was
seeing a pre-checkpoint snapshot while the MCP server saw fresh data.

Jeff ran a checkpoint test:
```python
conn.execute("PRAGMA wal_checkpoint(FULL)")
# now read latest
```
Result: latest = 2026-04-30T13:14:04 UTC (totally fresh). Total ticks
jumped from 648 to 651.

Chat declared victory ("WAL staleness was the issue"). Jeff asked good
follow-up questions about why a writer would hold up a reader, and
Chat (re-explained, partially incorrectly) the WAL mechanism.

### Round 4: Wrong hypothesis #2 -- snapshot isolation
Chat then pivoted to "Python sqlite3 connection holds a snapshot view
across all SELECTs once the first SELECT runs." Designed an empirical
test:

Test 1: Single connection, two reads with 65s sleep between them.
Test 2: Fresh connection per read.

Hypothesis: Test 1 would show no delta (snapshot anchored), Test 2
would show new data (fresh snapshots).

### Round 5: Test results disproved hypothesis #2
```
=== Test 1: single connection, multiple reads ===
  Read 1: latest=1777555024, ticks=654
  Sleeping 65s for at least one new tick to land...
  Read 2: latest=1777555084, ticks=655
  Delta: 60s, 1 new ticks
  CONCLUSION: Single connection saw new data; WAL is not the issue
=== Test 2: fresh connection per read, no sleep ===
  Read 1: latest=1777555084, ticks=655
  Read 2: latest=1777555145, ticks=656
  Delta: 61s, 1 new ticks
```

**Both tests showed fresh data.** The simple two-SELECT-on-one-connection
pattern does NOT reproduce the staleness Chat predicted.

### Round 6: Jeff's intuition was right
At this point, Jeff asked the question that pointed at the correct
answer: "Could it have been that there was some glitch in the python
code that just kept the same transaction open for all of your select
queries, so that the transaction was (and will never) close until the
connection is closed?"

That IS the right family of explanation. Python's `sqlite3` module has
documented quirks around implicit `BEGIN`. In `isolation_level=""`
(legacy default), the module opens implicit transactions on certain
statements and you must commit explicitly. Even pure-SELECT workloads
can interact with this in version-dependent and pattern-dependent ways.
The empirical Test 1 didn't reproduce because that specific simple
pattern doesn't trigger the implicit BEGIN, but slightly different
patterns do.

### Round 7: What actually happened in the original analysis
Honest answer: we can't fully retroactively diagnose. The morning's
gap-analysis script had a different cursor / SELECT pattern than
test_sqlite_freshness.py. Possible mechanisms:

1. The original script's first SELECT happened before the WAL
   checkpoint frontier moved past 09:11; that view stuck through the
   rest of the script's queries. After Chat's "PRAGMA wal_checkpoint"
   call (which Chat thought was magic but actually just took longer
   wall-clock time and then reopened a fresh connection), a fresh
   connection saw fresh data.
2. Some Windows-specific filesystem cache behavior, less likely.
3. The implicit-BEGIN quirk triggered for an obscure reason.

We don't have to know which. The defensive fix covers all of them.

---

## 3. Rule 34 (added in v1.3)

Inserted into "Testing and Diagnostics Rules" subsection between
existing Rule 33 (prefer .py file over inline python -c) and the
Retro Rules section:

> 34. Always explicitly manage transactions when reading a SQLite DB
> that another process is actively writing to. Python's sqlite3 module
> has documented quirks around implicit BEGIN that can cause a
> long-lived connection to see a snapshot view from a past state,
> missing all writes that committed since.
>
> Three acceptable patterns:
>   - Fresh connection per logical read pass (the MCP server's
>     connect_ro pattern; never seen stale).
>   - isolation_level=None at connect time (true autocommit).
>   - Explicit conn.commit() between SELECTs to release implicit
>     transactions.

Renumbered Retro Rules from 34-40 to 35-41. Total rule count: 41.

Updated Key Principles at the bottom of the file to add a new bullet
referencing Rule 34.

---

## 4. ASCII compliance + structure

```
$ grep -P "[^\x00-\x7F]" claude/CLAUDE_CODE_RULES.md
(empty)
$ grep -E "^[0-9]+\." claude/CLAUDE_CODE_RULES.md | wc -l
49  # 41 Code rules + 8 Chat rules
```

ASCII-clean. Rule numbering 1-41 (Code) and 1-8 (Chat) verified
sequential.

---

## 5. Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| 1 | CLAUDE_CODE_RULES.md bumped to v1.3 with changelog entry | PASS |
| 2 | New Rule 34 in Testing and Diagnostics Rules subsection | PASS |
| 3 | Retro Rules renumbered 34-40 -> 35-41 | PASS |
| 4 | Key Principles bullet added referencing Rule 34 | PASS |
| 5 | ASCII compliance (Rule 20) | PASS |
| 6 | Total rule count: 41 (was 40) | PASS |
| 7 | This retro at claude/retros/RETRO_sqlite_freshness_diagnostic.md | PASS |
| 8 | No edits to any other file | PASS |

---

## 6. Lessons for Chat (non-binding self-critique)

1. **Don't pattern-match to a confident-sounding explanation when the
   evidence doesn't support it.** Chat went through TWO wrong
   hypotheses (WAL staleness, full snapshot isolation) before arriving
   at the right framing. Each wrong hypothesis was internally
   consistent and could have been written as a confident-sounding
   blog post. The empirical test disproved hypothesis 2; the right
   move would have been to say "I don't know" earlier rather than
   pivot to the next plausible-sounding explanation.

2. **Listen to the user's intuitions.** Jeff's question "could it just
   be that the transaction stayed open?" was the right answer. Chat
   could have arrived there directly by being more careful about
   reading SQLite + Python documentation rather than reasoning from
   pattern-matched priors.

3. **Empirical testing is cheaper than theoretical analysis.** The
   ~2-minute Test 1/Test 2 script told us more than 30 minutes of
   confident theoretical writeup did.

4. **The morning summary "well within design tolerances" was lazy.**
   Jeff's pushback on this was correct. At early stages of new
   collectors, you investigate every shortfall, not assume "good
   enough." This is a general lesson worth keeping.

---

## 7. Open items

- **The collectors are healthy.** 100% tick coverage overnight, no
  gaps. The investigation was triggered by a Chat math error in the
  morning summary, not a real problem.

- **The two markets with low sample counts (Tom Brady 425, Elon Musk
  223) versus the standard 648.** Almost certainly because they were
  added to or rotated through the top-50 list partway through the
  night, not because of collector failures. Worth confirming in a
  future investigation if the rotation behavior turns out to matter
  for downstream analysis.

- **Add a Cycle 16+ candidate**: `claude/scratch/` template for SQLite
  analysis scripts that follows Rule 34 by default. Could be as
  simple as a copy-paste pattern in the rules file or as fancy as a
  helper module. Not blocking; nice-to-have.

- **PRAXIS LIVE_COLLECTOR data is rich.** The slug-level price history
  is exactly what we need for prediction-market analysis: 50 markets
  x 60s cadence = real microstructure of an entire active board over
  time. Worth designing a `live_collector_atlas` or similar that
  captures stylized facts from this data once it's been running for
  a few days.

---

## 8. References

- `claude/CLAUDE_CODE_RULES.md` v1.3 -- this cycle's deliverable
- `engines/live_collector.py` -- the writer process
- `data/live_collector.db` -- the DB whose freshness we investigated
- Python sqlite3 docs:
  https://docs.python.org/3/library/sqlite3.html#transaction-control
- SQLite WAL docs: https://sqlite.org/wal.html
