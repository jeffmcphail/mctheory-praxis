# Implementation Brief: Polymarket Short-Duration Crypto Fee Structure Investigation

**Status:** **PARKED** -- DO NOT auto-execute. This Brief sits in the queue until Chat explicitly elevates it. Jeff has confirmed it is deferred behind current build queue items (microstructure data accumulation, MCP server v0.2, funding carry wiring, market making bot, convergence speed detector). When Chat is ready to elevate, it will rename this file from `BRIEF_*_PARKED.md` to `BRIEF_*.md` and the standard auto-pickup workflow resumes.

**Series:** praxis
**Priority:** P3 (gating decision for several queued items but not blocking current cycle)
**Mode:** B (touches Polymarket API; **optional** Phase 4 involves a real on-chain trade requiring explicit confirmation)

**Estimated Scope:** S (20-40 min for Phases 0-3 read-only; +15 min for Phase 4 if Jeff approves the live test)
**Estimated Cost:** ~$0.50-$2 USDC.e if Phase 4 executes (one tiny round-trip trade); $0 otherwise
**Estimated Data Volume:** Polymarket Gamma API metadata + CLOB API responses for ~20 short-duration crypto markets; on-chain Polygon trades from 3-5 known arb wallets (~500 trades); no DB writes (data lives in retro)
**Kill switch:** if Phase 4 (live test) is approved, **maximum spend is $5 USDC.e**. Stop and report at any single trade rejection, unexpected fee >5%, or any signal of stuck on-chain transaction.

Reference: `WORKFLOW_MODES_PRAXIS.md` Mode B (live API calls, real-money path). Rules 9-15 (progress reporting), 21 (load_dotenv), 22-23 (resolution oracle, money-confirmation gate).

---

## Context

The session that produced this Brief reviewed Liu (2026), *AI-Augmented Arbitrage in Short-Duration Prediction Markets* (Polymarket 5-min BTC binary options live trading). Two corroborating findings emerged:

1. **Directional prediction at 5-min horizons does not work.** Liu's live sessions on Polygon with real USDC.e showed 25-27% win rate vs. ~53% breakeven, even with an LLM filter, multi-component momentum signals, and Kelly sizing. The author concludes 5-min BTC binary options approximate a random walk at short horizons. This independently corroborates our own v8.1 triple-barrier finding (41.4% accuracy vs. 42.0% baseline) and the broader principle that standard TA signals show no persistent edge across asset classes.

2. **The arb plays that previously worked on these markets may be largely closed.** The famous bot success stories ($313->$414K, the $40M cumulative arb profit estimate across April 2024-April 2025) were primarily exploiting (a) Polymarket's price lag vs. Binance/Coinbase spot ("latency arb"), and (b) sum-of-prices arbitrage (`YES + NO < $1.00`). Polymarket has since introduced **dynamic taker fees** specifically targeting these strategies on 15-minute crypto markets, with fees reaching ~3.15% on a 50-cent contract -- exceeding typical arb margin and (per their stated intent) making the strategies unprofitable at retail scale.

The unanswered question is **what the current fee structure actually looks like across short-duration crypto markets**, and whether any flavor of arb or market-making is still viable post-fee. This determines:

- Whether memory #15 (market making bot) needs to factor maker rebates into its EV calculation, and at what level.
- Whether memory #29 (convergence speed detector) can still cleanly enter/exit short-duration markets, or whether fees eat the edge.
- Whether the latency-arb play is fully dead or still viable on any specific short-duration market.
- Whether 5-min and 15-min crypto have the same fee structure, or whether one was updated and not the other.

This investigation is **read-only-first**. A live test trade is gated behind Phase 4 and requires explicit confirmation per memory #23 / rule 23.

---

## Objective

Produce a retro at `claude/retros/RETRO_polymarket_fee_structure.md` that answers:

1. **Per-market-class fee structure.** For each of: 5-min BTC, 5-min ETH, 15-min BTC, 15-min ETH, 1-hour BTC, weather markets, sports markets -- what is the current taker fee schedule? Is it static or dynamic? If dynamic, what's the formula or schedule?
2. **Maker rebates.** Is the Maker Rebates Program active on each market class? What are the current rebate rates?
3. **Fee tiering by volume.** Are there fee discounts based on cumulative wallet volume or other criteria?
4. **Strategy viability assessment.** Given the fee structure found, calculate the post-fee breakeven for: (a) latency arb against Binance/Coinbase spot, (b) sum-of-prices arb (`YES + NO < $1`), (c) market making (capture spread, earn rebate), (d) convergence-detector entries near probability extremes.
5. **Discrepancies between docs and reality.** If the live API metadata or actual on-chain trades show a different fee than documented, flag it.

---

## Detailed Spec

### Phase 0 -- Documentation pull (5 min)

```
Visit (or web_fetch from Code's environment):
  - https://docs.polymarket.com/  (search: fees, taker fee, maker rebate, dynamic fee)
  - https://learn.polymarket.com/  (search: fees)
  - Polymarket blog/announcements posts about the fee change (cite finance magnates article in retro)
```

Capture the documented fee structure as Polymarket states it. Note the date of the documentation if visible.

### Phase 1 -- Live API metadata sample (10 min)

Existing repo has a Polymarket Gamma client (`engines/` -- find via grep, likely in `gamma_client.py` or similar). Use it (or write a one-off read-only script) to fetch metadata for currently-active markets in each class:

```
- 5-min BTC Up/Down (live or upcoming)
- 5-min ETH Up/Down
- 15-min BTC Up/Down
- 15-min ETH Up/Down
- 1-hour BTC range
- one weather market
- one sports market
```

Inspect the returned JSON for fee-related fields. Polymarket markets historically expose fields like `fee_rate_bps`, `taker_fee`, `maker_rebate`, etc. Document what's present, what's missing, and any class-specific differences.

### Phase 2 -- On-chain spot-check of arb wallets (10 min)

The CoinDesk and BeInCrypto articles cite specific arb wallets (e.g., the $313->$414K wallet referenced via Dexter's Lab). Identify a 2-3 of them via:

```
- Polymarket leaderboard / Hashdive / Dexter's Lab if URLs are findable
- Or query smart_money.db for known short-horizon high-frequency wallets
- The wallet in the latency-arb finance magnates piece
```

For each, pick 5-10 of their recent trades, compute the implied fee from the on-chain transaction data:

```
implied_fee = (taker_paid - filler_received) / taker_paid
```

If implied fees are ~3% on near-50c markets, that confirms the dynamic taker fee regime is active. If they're zero or trivial, fees are not currently being charged on those markets. Cross-reference against the API metadata from Phase 1.

**Why this matters:** documented fees and live fees can drift. The on-chain data is ground truth.

### Phase 3 -- Strategy viability calculations (5-10 min)

With the fee structure in hand, compute post-fee breakeven for each of the four strategy classes listed in the Objective. Assume realistic spreads from the order book snapshots Phase 1 captured. Output a table like:

| Strategy | Edge required | Post-fee edge required | Currently feasible? | Notes |
|---|---|---|---|---|
| Latency arb (5-min BTC) | ~0.3% per trade | ? | ? | |
| Sum-of-prices arb (5-min) | ~1% spread | ? | ? | |
| Market making (15-min ETH) | spread x frequency | ? | ? | rebate offsets fee? |
| Convergence detector entry near 95c | ~2% on 95c | ? | ? | one-sided exposure |

**Conclusion:** for each strategy class, mark as `viable` / `marginal` / `closed` and document the reasoning.

### Phase 4 -- OPTIONAL live test trade (15 min, GATED)

**Do not execute Phase 4 without explicit confirmation from Jeff in the chat.**

If Phases 0-3 leave any specific strategy ambiguous (e.g., "5-min markets seem fee-free per docs but on-chain shows non-zero fees"), a single tiny test trade resolves it. Procedure:

1. Print the planned trade (market, side, size = $1-$2 USDC.e, expected fee, expected payout) and call `input("Type YES to confirm: ")` before any RPC call. (Per rule 23.)
2. Execute through the existing Polymarket trading harness (find via `grep -r "place_order\|create_order" engines/ scripts/`).
3. Capture the on-chain receipt, parse the actual fee paid, and document the variance vs. the documented rate.
4. If the test trade fills below or above the expected price, document the slippage observed.
5. **Maximum total spend across Phase 4: $5 USDC.e.** If for any reason a transaction fails or gets stuck, stop, do not retry, document and return.

### Phase 5 -- Retro

Standard format per `WORKFLOW_MODES_PRAXIS.md` and rules 31-34. Per-strategy viability table is the primary deliverable. Open-items section should flag any follow-up Briefs (e.g., "if market-making is viable, here's what an MVP brief would need").

---

## Acceptance Criteria

1. Retro exists at `claude/retros/RETRO_polymarket_fee_structure.md` with all five Objective questions answered.
2. The strategy-viability table is populated with concrete post-fee numbers, not free-text guesses.
3. Documented vs. live-observed fee discrepancies are explicitly called out.
4. **Zero code/script files modified for Phases 0-3.** Phase 4, if executed, may add a one-off test script in `scripts/` (clearly named as a test, e.g., `scripts/test_polymarket_fee_probe.py`).
5. Phase 4 (if executed) total spend is documented down to the cent and stays <= $5 USDC.e.

---

## Known Pitfalls

- **Phase 4 is real money.** Memory #20 + #23 + rule 23 apply. No execution without `input("Type YES to confirm: ")` gate AND explicit chat-side confirmation in the same session as the live test.
- **Fee structure may have changed since Brief was written.** When this Brief is unparked, Phase 0 documentation pull is mandatory regardless of how stale the rest of the Brief feels -- fee structures are exactly the kind of thing that can change quietly.
- **Different market classes may have different fee structures.** Don't assume 5-min and 15-min match. Don't assume crypto and weather match.
- **The resolver bug from the Liu article applies here too.** When verifying expected payout for any test trade, compute against Polymarket's resolution oracle, not against spot at execution time.
- **Liquidity on 5-min markets is thin** (~$5K-15K per side per the CoinDesk article). Even a $2 test trade won't move anything, but be aware that any larger backtest assuming clean fills is unrealistic.
- **The `MIN_HOURS_TO_RESOLUTION` parameter pattern from the Moonsat weather bot** suggests bots are filtering out markets too close to resolution -- possibly due to fee changes near settlement, possibly due to liquidity collapse. Worth noting if a similar pattern shows in our data.
- **Don't run this in parallel with the live trades collector.** Both query Polymarket APIs; rate limits could stack.

---

## References

- Liu (2026), *AI-Augmented Arbitrage in Short-Duration Prediction Markets*, Medium, March 2026 -- primary trigger for this investigation.
- Finance Magnates, *Polymarket Introduces Dynamic Fees to Curb Latency Arbitrage* -- documents the fee change.
- BeInCrypto / Yahoo Finance, *Arbitrage Bots Dominate Polymarket With Millions in Profits* -- historical bot success stories.
- CoinDesk Feb 2026, *How AI is helping retail traders exploit prediction market 'glitches'* -- corroborating ecosystem context.
- Polymarket docs.
- `claude/CLAUDE_CODE_RULES.md` rules 9-15, 16-20, 21-25.
- `claude/WORKFLOW_MODES_PRAXIS.md`.
- Memory #15 (market making), memory #29 (convergence detector), memory #20 (resolution oracle), memory #23 (money confirmation gate).
