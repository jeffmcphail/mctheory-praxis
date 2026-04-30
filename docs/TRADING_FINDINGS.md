# Trading Findings

> **About this file**: Empirical findings about market structure that
> should be respected by future Praxis work. Each entry is an outcome of
> investigation: "we tried X and discovered Y."
>
> Distinct from `docs/TRADING_CONVENTIONS.md` (which captures stable
> rules) and from `claude/TODO.md` (which captures pending work).
> Findings here are durable conclusions that should change future
> design decisions.

---

## Polymarket NegRisk arb is non-executable

**Status**: Definitively confirmed. Scanner built and run.

**The setup**: Polymarket's NegRisk markets (multi-outcome markets where
exactly one outcome resolves YES) appear to offer arbitrage opportunities
when the sum of YES prices across outcomes does not equal $1.00. A naive
implementation would short the high outcomes and long the low outcomes
to lock in the spread.

**The finding**: All apparent mispricings are phantom. They are caused
by illiquid placeholder outcomes with **zero ask-side order books**. You
cannot actually execute the trades that close the arbitrage; the prices
that imply the mispricing aren't tradable.

**The flash loan workaround does not work either**: The Polymarket CLOB
is off-chain. Flash loans work against on-chain DEX liquidity; they
cannot atomically execute a CLOB trade and a DEX trade in the same
transaction.

**Implication**: NegRisk arb scanning is permanently dead as a Praxis
strategy. Do not rebuild it. If a future market structure change
introduces NegRisk markets with deep ask-side liquidity on every
outcome, the conclusion may need revisiting -- but as of 2026-04-29 the
structure does not support this approach.

**Source**: Praxis NegRisk arb scanner (built pre-recovery), confirmed
across hundreds of apparent opportunities.

---

## (Future entries)

Add findings here when investigation produces a durable conclusion that
should bind future design. Each finding should include:

- **Status**: confirmed / preliminary / contested
- **Setup**: what was tried
- **Finding**: what was discovered
- **Implication**: what future work should do (or not do) as a result

If a finding is provisional, prefer "preliminary -- needs replication"
over deletion. Failed experiments are data; future cycles can revisit.

---

## Revision history

| Date | Cycle | Change |
|---|---|---|
| 2026-04-30 | 16 | Initial. Captured "NegRisk arb non-executable" finding from memory entry 19. |
