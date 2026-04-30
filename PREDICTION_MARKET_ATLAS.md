# PREDICTION MARKET ATLAS
## McTheory Praxis — Prediction Market Trading Research
### Created: 2026-04-04 | Status: Initial Research Phase

> **Sync state:** This file is the source of truth. After editing, run
> `python -m engines.atlas_sync` to update the queryable DB at
> `data/praxis_meta.db`. See `docs/ATLAS_DB.md`.

---

## 1. PLATFORM MECHANICS & MICROSTRUCTURE

### 1.1 Polymarket

**Structure:** Decentralized prediction market on Polygon blockchain. Hybrid CLOB (Central Limit Order Book) — orders are matched off-chain by an operator, settlement is on-chain via smart contracts using Conditional Token Framework (CTF). Users trade YES/NO outcome shares priced $0.01-$0.99, paying out $1 on correct resolution.

**Order types:** GTC (Good Till Cancelled, maker), FOK (Fill or Kill, taker). All orders are EIP-712 signed messages. Limit orders that rest on the book are maker orders (zero fees); orders that cross the spread are taker orders.

**Fees (critical — changed March 2026):**
- Maker: 0% (zero fees for limit orders providing liquidity)
- Taker: fees vary by market category, expanded in March 2026 to cover nearly all categories
- Winner fee: 2% on winning positions at settlement
- Breakeven spread for arb after fees: needs to exceed ~2.5-3% to be profitable
- Polymarket introduced a referral program (30% commission on direct referral fees) alongside fee expansion

**API infrastructure:**
- Gamma API (gamma-api.polymarket.com): market metadata, discovery, descriptions, token IDs
- CLOB API (clob.polymarket.com): trading — prices, orderbook, order placement/cancellation
- Data API (data-api.polymarket.com): user positions, trade history, fills
- WebSocket: wss://ws-subscriptions-clob.polymarket.com/ws/market — real-time orderbook updates
- Python SDK: `py-clob-client` (pip install py-clob-client)
- TypeScript SDK: `@dicedhq/polymarket`
- Rate limits: 15,000 req/10s general, 9,000/10s CLOB, POST /order 3,500/10s burst and 36,000/10min sustained
- Throttle-based (Cloudflare queues requests, doesn't immediately reject)

**Settlement:** UMA optimistic oracle system. 99.2% dispute resolution accuracy. Resolution disputes affect 3.4% of markets. Ambiguous market wording is the primary cause, not oracle manipulation. Markets >$1M volume face elevated manipulation risks.

**On-chain data:** All trades settle on Polygon. Order matching is off-chain but fills are recorded on-chain as conditional token transfers. Dune Analytics dashboards available for market intelligence. Can read order flow, whale positions, and historical pricing on-chain.

**US access:** Officially blocked US users until 2025. Polymarket acquired QXC and QC Clearing to gain CFTC DCM authorization. Now accessible to US users, though multiple states are challenging legality.

---

### 1.2 Kalshi

**Structure:** CFTC-regulated Designated Contract Market (DCM). Centralized exchange. Quote-driven market where makers post offers and takers accept them. Binary YES/NO contracts paying $1 on correct resolution.

**Order types:** Limit orders (maker, lower fees), market orders (taker, higher fees). Subaccounts supported. Fractional share trading rolling out March 2026.

**Fees:**
- Taker fee formula: `0.07 × P × (1-P)` where P is contract price
- Max taker fee: 1.75¢ per contract (at 50¢ price)
- Fee is lowest at price extremes (near 0¢ or 99¢) — parabolic structure
- Maker fees: lower than taker, applied only on some markets when order fills
- No settlement fee (unlike Polymarket's 2% winner fee)
- Market maker rebates up to 1% through tiered reward system (capped $7k/week)
- 4.05% annual interest on idle cash balances

**API:** REST API with WebSocket channels. Supports order groups, subaccounts, and batch operations. API tier limits available via GET /account/limits. Changelog shows active development (fractional trading, fixed-point migration, fee rounding).

**Key differentiators from Polymarket:**
- Fully US-regulated, fiat deposits (ACH, wire, PayPal, Venmo, crypto)
- $22B valuation (March 2026), $1B raised
- 85,000+ active markets, weekly volumes >$2B
- Sports = 75% of volume
- Margin trading approved (Kinetic Markets registered as FCM, awaiting CFTC final approval)
- Robinhood integration (though share of volume fell from 59% to 22.6% as users moved direct)

---

## 2. STRATEGY TAXONOMY

### Category A: Pure Arbitrage (No Opinion Required)

#### A1. Same-Market Parity Arbitrage

**Mechanism:** Buy YES at P_yes and NO at P_no when P_yes + P_no < $1. Guaranteed profit of $1 - (P_yes + P_no) at resolution regardless of outcome.

**Historical performance:** Generated estimated $40M in profits between April 2024 and April 2025.

**Current status: DEAD for retail.** Bid-ask spreads compressed from 4.5% (2023) to 1.2% (2025). Professional market makers and algorithmic bots detect and execute within 200 milliseconds. The top three arbitrage wallets combined made $4.2M in a year with ~$500k+ capital deployed. After Polymarket's 2% winner fee, arbitrage spreads need to exceed 2.5-3% to be profitable, and those spreads no longer exist on liquid markets.

**Automation:** Fully automated by sophisticated bots. Open-source implementations exist (GitHub) but cannot compete with professional infrastructure.

#### A1b. Asymmetric Temporal Micro-Arb (Gabagool22 Variant)

**Mechanism:** A specific sub-variant of A1 that IS alive on short-duration crypto markets (15-minute BTC/ETH up/down buckets on Polymarket). Instead of buying YES and NO simultaneously, the bot buys the cheap side first (taker), then waits for the other side to become cheap during intra-window volatility, and hedges with a passive limit order. Total cost is kept below $1.00, guaranteeing profit at resolution regardless of outcome.

**Key insight:** During high-volatility 15-minute windows, the YES+NO spread temporarily opens to 3-4% as retail traders panic-buy one side. The bot doesn't predict direction — it exploits the transient spread.

**Documented performance:**
- gabagool22: $246k profit, 86% win rate, 10,000+ predictions. Documented on Polymarket leaderboards.
- distinct-baguette: appears to be same operator, similar patterns
- Arbigab bot (gabagool22.com): commercial implementation claiming $300k+ profit. Rust-based, runs on VPS 24/7.
- Single 15-min window example: bought 1,267 YES @ $0.517 avg + 1,295 NO @ $0.449 avg = total cost $0.966 per pair. Profit: $58.52 on one window (3.4% return).

**Why this works when A1 is dead:** Short-duration crypto markets (15 min) have much higher volatility and more retail emotional trading than political or sports markets. The spread opens and closes within minutes. Professional market makers haven't fully compressed these because the windows are so short and the volumes per window are small.

**Current status: ALIVE but competitive.** Multiple bots now competing in these markets. Returns are compressing as more operators enter. Requires: Polymarket account with USDC on Polygon, VPS for 24/7 operation, sub-second execution.

**Open-source tools:**
- GitHub: Gabagool2-2/polymarket-Copy-trading-bot (copy-trading approach)
- GitHub: dexorynlabs/polymarket-trading-bot-ts (TypeScript, 7 strategy implementations)
- Commercial: gabagool22.com (Rust bot, $300+ for license)

**Risk:** Partial fills (only one leg executes), fee changes on crypto markets, Polymarket operational risk. If you buy YES but can't get the NO hedge filled before the window closes, you have naked directional exposure.

---

#### A2. Cross-Market Combinatorial Arbitrage

**Mechanism:** Logically related markets with inconsistent pricing. Example: "Trump wins the presidency" at 55¢ and "Republican wins the presidency" at 50¢ — the first implies the second, so the second should be at least 55¢.

**Academic evidence:** Cornell/IMDEA paper ("Unravelling the Probabilistic Forest," 2025) found over 7,000 Polymarket markets with measurable combinatorial mispricings. Two forms identified: Market Rebalancing Arbitrage (within single market/condition) and Combinatorial Arbitrage (across dependent markets). The challenge is O(2^(n+m)) comparisons at scale — requires heuristic-driven reduction strategies.

**Current status: ALIVE but requires sophistication.** Harder to automate than parity arb because detecting logical dependencies between markets requires NLP/semantic understanding. The relationships aren't always obvious. Bots that can detect "if X then Y" relationships across thousands of markets have a structural advantage.

**Returns:** Variable, depends on detection quality. Spreads typically 2-10% when found.

**Capital requirements:** Medium ($5-50k). Must hold positions until resolution.

**Key risk:** Non-atomic execution on Polymarket — you can't guarantee both legs fill. Partial fills create directional exposure.

---

#### A3. Cross-Platform Arbitrage (Polymarket vs Kalshi)

**Mechanism:** Same event priced differently on two platforms. Buy cheap side on one, sell expensive side on the other.

**Academic evidence:** Study found discrepancies between Kalshi and Polymarket during 2024 US election. Polymarket generally leads Kalshi due to higher liquidity, making it more informative in final hours before resolution.

**Current status: DYING.** Opportunities exist only for seconds to minutes. Capital efficiency is poor — money is locked on both platforms simultaneously. Fee structures differ (Polymarket 2% winner fee vs Kalshi's parabolic taker fee), making breakeven calculation complex. Tools exist (Polymarket-Kalshi Arbitrage Bot on GitHub) but profitability is marginal after fees.

**Key constraint:** Kalshi uses USD, Polymarket uses USDC on Polygon. Moving capital between platforms takes time.

---

#### A4. Temporal / Latency Arbitrage

**Mechanism:** Market prices lag real-world events by seconds to minutes. If you detect the event faster than the market adjusts, you can buy/sell before the price moves.

**Notable incidents:**
- Iran strike (Feb 28, 2026): six accounts placed bets hours before US missiles struck Tehran, earning $1.2M collectively. One account turned $61k into $493k (821% return). Most accounts created within 24 hours of the strikes. Federal prosecutors investigating.
- Maduro capture (Jan 2026): freshly created accounts netted $400k+ betting on Maduro's removal hours before the operation went public.
- These are almost certainly insider trading, not legitimate latency arb.

**Legitimate latency arb:** During major news events (debates, economic data releases, injury announcements), markets take 30-60 seconds to reprice. Bots monitoring news feeds can exploit this window.

**Current status: ALIVE but compressed.** AI-powered news parsing (using models like Mistral-7B) has reduced the window. Milliseconds matter. Requires VPS infrastructure near exchange servers.

---

### Category B: Statistical / Structural Edge (Pattern-Based)

#### B1. Favorite-Longshot Bias Exploitation

**Mechanism:** Retail traders systematically overpay for low-probability outcomes. Contracts priced below 10¢ win far less often than their price implies. Conversely, contracts above 50¢ earn a small positive return.

**Academic evidence (strongest finding in the atlas):**
- Bürgi, Deng & Whelan (2026, CEPR): Analysis of 300,000+ Kalshi contracts. "Investors who buy contracts costing less than 10¢ lose over 60% of their money." Contracts above 50¢ earn statistically significant positive returns.
- The bias exists for both makers and takers, but is much stronger for takers. Takers lose 32% on average; makers lose 10%.
- The bias persists across all categories (politics, sports, weather, etc.) though there's evidence it's weakening over time as users become more sophisticated.
- Polymarket data: outcomes below 10% implied probability occur only 14% of the time, but contracts at those prices are consistently overpriced.

**Strategy:** Systematically sell YES on longshots (contracts at 5-15¢) where true probability is lower than market price. Or equivalently, buy NO at 85-95¢.

**Current status: ALIVE — this is the most academically validated edge.**

**Returns:** Selling longshots: 60%+ average return on capital (but per-contract, and with tail risk). Buying favorites (>50¢): small positive returns (~2-5% annualized depending on turnover).

**Key risk:** The longshot actually hits. A portfolio of 20 longshot sells at 10¢ expects to pay out on ~1-2 of them. Need diversification across many uncorrelated events. The variance is real.

**Automation feasibility:** High. Scan for contracts in the 5-15¢ range, filter by volume (>$10k to avoid dead markets), sell YES or buy NO systematically. Exit early if price moves against you.

---

#### B2. Tail-End Trading (Near-Certain Outcomes)

**Mechanism:** Buy contracts priced 90-99¢ that are near-certain to resolve YES. Earn 1-10% return with very high probability. Capital efficiency play.

**Example:** "Will the sun rise tomorrow?" at 98¢ = 2% return in 24 hours. Obviously extreme, but many real markets approach this near resolution.

**Current status: ALIVE but capital-intensive.** Returns are small per contract (1-10 cents). Need large position sizes to be meaningful. Automated bots focus on high-probability contracts at 90-99¢ and accumulate thousands of micro-trades. Works best in short-duration crypto markets (BTC/ETH price thresholds).

**Returns:** 1-10% per resolution, which can compound to 35-95% APY with sufficient turnover.

**Key risk:** The "near-certain" outcome doesn't happen. A 99¢ contract that resolves NO costs you 99¢ per contract. One miss wipes 99 wins.

---

#### B3. Market Lifecycle Mispricing

**Mechanism:** Markets are most mispriced in their first quartile of life (new, low attention, retail-dominated). As volume grows and information accumulates, prices become more accurate.

**Quantitative evidence:**
- Markets below $50k volume: severe mispricing, wide bid-ask spreads, zero market-maker participation
- Markets above $100k: reliable pricing begins
- Markets above $1M: efficient, only market-making or fundamental models work
- First quartile of market life: highest mispricing, best for directional plays
- Final quartile: accuracy >90%, pivot to liquidity provision and spread harvesting

**Strategy:** Monitor newly created markets. Enter directional positions early when retail sentiment is driving prices. Exit as volume grows and prices converge to fair value.

**Current status: ALIVE — structural feature of how prediction markets work.**

**Automation:** Moderate. Requires scanning for new market creation events via API, estimating fair probability (potentially via LLM), and entering before the market becomes efficient.

---

#### B4. Sentiment / Momentum Overreaction (JEFF'S TYPE 2 — PRIMARY FOCUS)

**Mechanism:** Breaking news causes price overshoot. The market initially overreacts, then mean-reverts toward fair value. Take the opposite side of the overreaction and close when the price normalizes.

**Evidence:**
- Political markets show 72-hour overcorrection patterns following major developments (Fensory Research, 2026)
- Recency bias: markets overweight recent information, creating temporary mispricings
- Sports markets after injury/lineup announcements: rapid price moves that partially revert
- High-volatility events create 30-60 second windows of extreme mispricing

**Strategy variants:**
1. **News overreaction fade:** Wait for a major news event to spike a market price. Take the opposite side once the initial panic subsides. Close within hours/days as the price mean-reverts.
2. **Debate bounce fade:** After political debates, markets spike on perceived winners. Historical data shows partial mean-reversion within 48-72 hours.
3. **Liquidity cascade fade:** When whale traders move a low-liquidity market by 10-20%, retail follows. Take the contrarian position and wait for rebalancing.

**Current status: ALIVE — this is the core Type 2 edge.**

**Returns:** Variable. Individual trades can yield 5-20% on position size. Requires patience (hours to days) and discipline to not panic when the position moves against you initially.

**Key risks:**
- The overreaction wasn't an overreaction — the news genuinely changed the probability
- Time decay if holding positions in markets approaching resolution
- Low liquidity makes entry/exit expensive on smaller markets

**Automation feasibility:** Medium-High. Monitor price velocity (rapid moves in short windows), detect divergences from moving averages, enter contrarian positions. The challenge is distinguishing genuine probability shifts from overreactions.

---

#### B5. Anchoring and Calendar Effects

**Mechanism:** Markets near expiry behave differently than early-stage markets. Prices become sticky near round numbers. Markets approaching resolution show convergence patterns that are exploitable.

**Current status: Under-researched. Potential emerging edge.**

---

#### B6. Cross-Market Lead/Lag

**Mechanism:** Information flows between prediction markets and traditional markets, with one leading the other.

**Evidence:**
- Polymarket generally leads Kalshi in price discovery due to higher liquidity
- Economic prediction markets (Kalshi Fed rate contracts) rival CME FedWatch tool accuracy (Fed FEDS paper, 2026)
- Political prediction markets may lead polling aggregators by hours/days

**Strategy:** Monitor the leading platform/source for price changes, trade on the lagging one before it catches up.

**Current status: ALIVE but window is narrowing as platforms add cross-platform data.**

---

### Category C: Information Edge (Expertise-Based)

#### C1. Domain Expert Models

**Mechanism:** Build quantitative models that estimate event probabilities from exogenous data. Trade when model probability diverges significantly from market price.

**Best domains for exploitable gaps (ranked by edge potential):**
1. **Weather:** NOAA/GFS ensemble forecasts vs Kalshi/Polymarket weather markets. Most documented edge. See detailed analysis below.
2. **Economics:** CME FedWatch vs Kalshi rate markets. Fed paper confirms Kalshi is competitive with professional forecasts, but temporary mispricings exist around data releases.
3. **Sports:** Advanced analytics (injury reports 4-6 hours before market adjustment, historical coach decision patterns, demographic polling underweighting).
4. **Politics:** Poll aggregation + sentiment analysis. Most competitive domain — hardest to find edge.

**C1 DEEP DIVE: Weather Trading (Highest-Priority Implementation)**

Weather markets are the most promising domain for a systematic trader because:
- Resolution sources are completely fixed and known (specific NOAA weather stations, e.g., EGLC for London, Central Park for NYC)
- Professional weather models (GFS, ECMWF, NAM) are free and highly accurate (85-90% for 1-2 day forecasts)
- Retail traders set prices based on gut feelings, not models
- No special taker fees on Polymarket weather markets
- Regulatory safe harbor — weather markets are the least likely category to be banned

**Documented trader performance:**
- gopfan2: reportedly $2M+ in net profit, primarily from weather markets. Strategy: buy YES below $0.15, buy NO above $0.45, $1 per position, thousands of bets
- meropi: ~$30k profit using fully automated $1-3 micro-bets, some positions at $0.01/share producing 500x payoffs
- 1pixel: $18.5k profit from $2.3k deposits, trading only NYC and London weather. Individual trades turning $6→$590 or $15→$547
- Anonymous bot: $1k→$24k since April 2025 trading London weather markets

**The strategy (validated by all sources):**
1. Fetch GFS ensemble forecast (31 members) from Open-Meteo (free API)
2. Count fraction of ensemble members above/below temperature threshold
3. That fraction = model probability (e.g., 28/31 members above 70°F = 90%)
4. Compare to market price (e.g., Polymarket pricing "40-45°F range" at 15¢ = 15% implied)
5. If edge > 8%, buy the underpriced bucket
6. Position size via Kelly criterion: kelly × 0.15 × bankroll, capped at $75-100 per trade

**Open-source implementation:** GitHub: suislanchez/polymarket-kalshi-weather-bot
- Multi-platform (Kalshi KXHIGH series + Polymarket)
- 31-member GFS ensemble from Open-Meteo
- Kelly criterion sizing, 8% edge threshold
- FastAPI backend + React dashboard
- $1.8k documented profits (small scale test)
- All data sources are free — no paid APIs

**Market details:**
- Kalshi: KXHIGH series (KXHIGHNY, KXHIGHCHI, KXHIGHMIA, KXHIGHLAX, KXHIGHDEN). Daily high temp in specific cities. Resolution: NWS official data, released following morning.
- Polymarket: temperature bucket markets (ranges like 40-45°F), precipitation, wind. London and NYC most liquid.
- Multiple model sources: GFS (American, 4 runs/day), ECMWF/Euro (most accurate), NAM (high-res, best for <3 day)
- Weatherbetter.app: commercial ML-powered signal service for weather markets

**Current status: ALIVE — the 2026 "information arbitrage" strategy.** This is where practitioner consensus has shifted. "The winning arbitrage strategy in 2026 is systematically processing publicly available information faster, better, or more accurately than the market consensus."

**Capital requirements:** Low ($1-5k to start). Edge comes from knowledge, not capital.

---

#### C2. AI/LLM-Powered Probability Models

**Mechanism:** Use large language models to estimate event probabilities by processing news, historical data, and contextual information. Compare LLM output with market prices.

**Current state:** Active research area. Multiple bot frameworks attempt this. Results are mixed — LLMs can synthesize information but struggle with calibration (outputting well-calibrated probabilities vs just directional opinions).

**Current status: EMERGING. Rapidly evolving.**

---

### Category D: Market Making & Liquidity Provision

#### D1. Spread Harvesting

**Mechanism:** Post limit orders on both sides of the book. Capture the bid-ask spread on each round-trip. Manage inventory risk.

**Evidence:** Liquidity providers earned >$20M in the past year on Polymarket. Market makers receive rebates and earn from the spread.

**Current status: ALIVE but competitive.** Requires significant capital ($50k+), sophisticated inventory management, and infrastructure. Dominated by professional firms. Kalshi offers tiered rebate programs.

**Returns:** Professional market makers earn consistent but thin returns. The 2028 election market on Polymarket has a 4% yield program for liquidity providers.

---

## 3. DEAD / DYING / ALIVE / EMERGING

| Strategy | Status | Notes |
|----------|--------|-------|
| A1. Same-market parity arb | ❌ DEAD | Bots close in 200ms. Spreads 1.2%. $40M extracted 2024-2025. |
| A1b. Asymmetric micro-arb (gabagool) | ✅ ALIVE | 15-min crypto markets only. $246k profit documented. Competitive. |
| A2. Combinatorial arb | ⚠️ ALIVE (niche) | 7,000+ mispriced markets found. Requires NLP/semantic detection. |
| A3. Cross-platform arb | ⚠️ DYING | Opportunities last seconds. Capital locked on both sides. |
| A4. Latency arb | ⚠️ DYING | AI news parsing compressed windows. VPS + speed required. |
| B1. Longshot bias | ✅ ALIVE | Strongest academic evidence. 60%+ loss on <10¢ contracts. |
| B2. Tail-end trading | ✅ ALIVE | Capital-intensive. 1-10% per contract. Compounding works. |
| B3. Lifecycle mispricing | ✅ ALIVE | Structural. New markets = most mispriced. |
| B4. Sentiment overreaction | ✅ ALIVE | Core Type 2 edge. 72h mean-reversion in political markets. |
| B5. Anchoring effects | ❓ UNDER-RESEARCHED | Potential emerging edge. |
| B6. Lead/lag | ⚠️ DYING | Polymarket leads Kalshi. Window narrowing. |
| C1. Domain expert (weather) | ✅ ALIVE | $2M+ documented. NOAA vs market. Open-source bots exist. |
| C1. Domain expert (economics) | ✅ ALIVE | Fed validated. CME FedWatch comparison. |
| C2. AI probability models | 🆕 EMERGING | Active research. Calibration is the challenge. |
| D1. Market making | ✅ ALIVE (pros only) | $20M+ earned by LPs. Requires $50k+ capital. |

---

## 4. REGULATORY LANDSCAPE & RISK (CRITICAL — APRIL 2026)

### 4.1 Current Legal Status

Both Polymarket and Kalshi now operate as CFTC-regulated Designated Contract Markets. However, this status is being challenged from multiple directions simultaneously.

### 4.2 Active Legislation (as of April 2026)

| Bill | Sponsors | What it would do | Status |
|------|----------|-----------------|--------|
| Prediction Markets Are Gambling Act | Schiff (D), Curtis (R) | Ban event contracts on elections, sports, government actions | Introduced March 2026 |
| STOP Corrupt Bets Act | Merkley (D), Raskin (D) | Ban wagers on politics, sports, news, military | Introduced March 2026 |
| Public Integrity in Financial Prediction Markets Act | Young (R), Slotkin (D), Curtis (R), Schiff (D) | Ban government officials from insider trading on prediction markets | Introduced March 2026 |
| PREDICT Act | Smith (R), Budzinski (D) | Civil penalties for insider trading, extends to spouses/children | Introduced March 2026 |

**Key dynamics:**
- Democrats want broad bans (sports, politics, military)
- Bipartisan support for insider trading restrictions specifically
- Arizona filed criminal charges against Kalshi (March 2026) — first ever
- Nevada secured court order blocking sports contracts
- Trump administration counter-sued three states trying to regulate prediction markets (April 2, 2026)
- Federal prosecutors meeting with Polymarket re: potential charges from Iran/Maduro incidents
- CFTC public comment period on regulation closes April 30, 2026

### 4.3 What's Safest from Regulation

**High risk of ban:** Sports (75% of Kalshi volume), military/war, political elections
**Medium risk:** Entertainment, culture, celebrity
**Lower risk:** Economics/financial (Fed rates, CPI, GDP), weather, crypto prices
**Why:** Financial event contracts most closely resemble existing regulated derivatives. JPMorgan's Dimon said they'd consider offering prediction market services but "won't be in sports" or "politics" — signaling where the institutional safe harbor is.

### 4.4 Timeline Assessment

- **Next 90 days (April-June 2026):** Intense legislative activity. CFTC comment period closes April 30. Criminal cases in Arizona proceeding. Federal prosecutor investigations ongoing.
- **2026 midterms:** Political prediction markets will be the battleground. If bans pass, likely sports first, politics second.
- **Realistic outcome:** Sports betting on prediction markets likely gets restricted or banned. Political markets may survive with insider trading rules. Financial/economic markets almost certainly survive.
- **Implication for us:** Focus on economics/weather/crypto categories. These have the most regulatory durability AND the most quantifiable edges (exogenous data models).

---

## 5. INFRASTRUCTURE & TOOLS

### 5.1 Open-Source Tools

- **py-clob-client:** Official Polymarket Python SDK. Orders, positions, market data.
- **NautilusTrader:** Institutional-grade integration with Polymarket CLOB API. Sub-millisecond latency. Rust+Python.
- **OctoBot Prediction Market:** Visual interface for Polymarket strategies including arb detection.
- **Polymarket-Kalshi Arbitrage Bot:** GitHub. Monitors price discrepancies, executes both legs.
- **Dexoryn's TypeScript bot:** 7 strategy implementations (GitHub: dexorynlabs/polymarket-trading-bot-ts).
- **PredictEngine:** API wrapper with gasless transactions, managed wallets, strategy templates.

### 5.2 Data Sources

- Polymarket Gamma API: all active/archived markets, outcomes, prices, volume
- Polymarket CLOB API: real-time orderbook depth
- Polymarket on-chain: Polygon blockchain, Dune Analytics dashboards
- Kalshi API: market data, orderbook, historical prices, fills
- Kalshi changelog: detailed API updates, new features
- Metaculus: forecasting database for calibration comparison
- FiveThirtyEight, NOAA, CME FedWatch: exogenous data for domain models

---

## 6. QUANTITATIVE BENCHMARKS

### 6.1 Platform Statistics (April 2026)

| Metric | Polymarket | Kalshi |
|--------|-----------|-------|
| Valuation | ~$20B | $22B |
| Weekly volume | ~$2B+ | $2.3B+ |
| Monthly volume (peak) | — | $5.8B (late 2025) |
| Active markets | Thousands | 85,000+ |
| Regulation | CFTC DCM (via QXC) | CFTC DCM |
| Primary currency | USDC (Polygon) | USD (fiat) |
| Top category | Politics/Crypto | Sports (75%) |
| Accuracy (academic) | 67.2% | 78.4% |
| Winner/settlement fee | 2% | 0% |
| Maker fee | 0% | Low/varies |

### 6.2 Profitability Distribution

- Most traders lose money. The profit distribution is heavily skewed.
- Top arbitrage wallets: $4.2M combined in one year (~$500k capital deployed)
- Liquidity providers: $20M+ collectively in past year
- Estimated $40M extracted via all forms of arbitrage (2024-2025)
- Takers lose 32% on average; makers lose 10% (Kalshi academic data)
- Contracts below 10¢: buyers lose 60%+ of money

### 6.3 Accuracy & Calibration

- Polymarket: 73% overall accuracy, 81% on binary political, 62% on multi-outcome entertainment
- Kalshi: "almost perfectly calibrated" per company data; 78.4% in academic study
- PredictIt: 93.1% accuracy (but much lower volume and limited markets)
- Favorite-longshot bias: persistent across all platforms, all categories, all years
- Bias weakening slightly over time as user sophistication increases

---

## 7. PRIORITIZED STRATEGIES FOR $5-10K SYSTEMATIC TRADER (April 2026)

### Rank 1: Weather Model Trading (C1 — Weather)
**Why:** Most documented edge with real P&L. $2M+ by gopfan2, $24k from $1k by another trader. Resolution sources are fixed (specific NOAA stations), models are free (GFS ensemble), and weather markets are in the regulatory safe harbor. Open-source bot already exists.
**Implementation:** Fork suislanchez/polymarket-kalshi-weather-bot. Adapt to our Praxis framework. Start with NYC/London temperature markets. 31-member GFS ensemble, 8% edge threshold, Kelly sizing capped at $100/trade. Run on both Kalshi (KXHIGH series) and Polymarket.
**Capital:** $500-1,000 starting. Scale to $5k as edge is validated.
**Expected timeline:** 1-2 weeks to operational.

### Rank 2: Longshot Bias Exploitation (B1)
**Why:** Strongest academic evidence. 60%+ expected return on shorts of <10¢ contracts. Can be automated. Works across all categories. Regulatory risk: low.
**Implementation:** Scan for YES contracts at 5-15¢ with volume >$10k. Sell YES (or buy NO at 85-95¢). Diversify across 20-50 uncorrelated events. Max 2-5% of portfolio per event.

### Rank 3: Sentiment Overreaction Fade (B4)
**Why:** Core Type 2 strategy. 72-hour mean-reversion documented in political markets. Does not require domain expertise — just patience and a model of "normal" pricing.
**Implementation:** Monitor price velocity on high-volume markets. When price moves >15% in <1 hour, take the opposite side. Exit over 24-72 hours as price normalizes.

### Rank 4: Market Lifecycle Entry (B3)
**Why:** Structural feature — new markets are always mispriced. Combined with B1 (new longshots are even more overpriced). Zero competition from bots on brand-new markets.
**Implementation:** Monitor market creation events via API. Estimate fair probability (LLM or domain model). Enter within first 24-48 hours when mispricing is highest.

### Rank 4: Economic Domain Models (C1)
**Why:** Quantifiable edge using public data (NOAA, CME FedWatch, economic calendars). Regulatory safe harbor (financial/economic markets least likely to be banned). Fed research validates prediction markets compete with professional forecasts — meaning mispricings relative to existing tools are detectable.
**Implementation:** Build models comparing Kalshi economic contract prices with CME FedWatch, NOAA forecasts, and economic consensus. Trade divergences. Focus on weather (most exploitable) and Fed rate decisions (most liquid).

### Rank 5: Combinatorial Arbitrage (A2)
**Why:** 7,000+ mispriced markets found in academic research. Requires NLP to detect logical dependencies. High barrier to entry = less competition. But requires capital locked until resolution.
**Implementation:** Build semantic similarity engine to detect related markets. Monitor for price inconsistencies. Execute when spread exceeds fee threshold (~3%). Highest alpha potential but also highest complexity.

---

## 8. NEXT STEPS

1. **Set up Polymarket and Kalshi accounts** — deposit $500 each for initial exploration
2. **Weather bot (Experiment 1)** — fork/adapt the open-source weather bot (suislanchez/polymarket-kalshi-weather-bot). GFS ensemble vs Kalshi KXHIGH + Polymarket weather. Start with $100-500 on NYC/London temp markets.
3. **Build market scanner** — pull all active markets via both APIs, compute volume, spread, price history, time to resolution
4. **Implement B1 (longshot bias) scanner** — identify <15¢ contracts with volume >$10k, track historical win rates by price bucket
5. **Implement B4 (overreaction) detector** — monitor price velocity, flag rapid moves, backtest mean-reversion on historical data
6. **Track regulatory developments** — CFTC comment period closes April 30. Focus on weather/economics (safe harbor) over sports/politics (ban risk).

---

## 9. MEDIUM ARTICLE VALIDATION (2026-04-04)

Jeff's collection of 12 Medium articles tested against the atlas:

**Confirmed and expanded:**
- gabagool22 analysis → added A1b (asymmetric micro-arb) as live sub-strategy
- Weather trading articles → massively expanded C1 with documented $2M+ performance, open-source bot, and specific implementation details
- "Sips & Scale" skeptical articles → correctly identified that simple arb is dead and execution > strategy

**Classified as fluff/survivorship bias:**
- "$1k to $200k on Kalshi" — cherry-picked outlier, not reproducible strategy
- "5 ways to make $100k" — engagement bait, strategies are real but returns are inflated
- "$500 to $13k and why I'm down to $3k" — actually the most honest article (shows the drawdown side)

**Key insight from validation:** The atlas correctly identified the live strategies before seeing the articles. The articles added granularity (specific trader names, P&L numbers, open-source repos) but didn't reveal any strategy category we'd missed. QPT prompt worked.

---

## Atlas Metadata

| Field | Value |
|-------|-------|
| Created | 2026-04-04 |
| Updated | 2026-04-04 (Medium article validation) |
| Status | Initial research, pre-experiment |
| Primary sources | Academic papers (Whelan 2026, Cornell/IMDEA 2025, Vanderbilt 2025, Fed FEDS 2026), platform docs, practitioner articles, open-source repos |
| Platforms | Polymarket, Kalshi |
| Total strategies catalogued | 16 across 4 categories (added A1b) |
| Experiments completed | 0 (atlas phase) |
