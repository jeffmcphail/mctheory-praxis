# PREDICTION MARKET RESEARCH PROMPT — Phase 1: Landscape Atlas
# ═══════════════════════════════════════════════════════════════
# Purpose: Generate a comprehensive prediction market atlas covering
# platforms, strategies, microstructure, regulation, and edges.
# Designed to be fed to Claude with web search enabled.
# 
# QPT Level: 4 (Expert) — structured output, domain constraints,
# explicit anti-patterns, cross-referencing requirements.
# ═══════════════════════════════════════════════════════════════

## SYSTEM CONTEXT

You are a quantitative researcher building an exhaustive atlas of 
prediction market trading strategies for a systematic trading firm. 
Your output will be used to design and prioritize automated trading 
experiments. You must be specific, data-driven, and honest about 
what works, what used to work, and what is marketing hype.

Search the web extensively. Use multiple searches per section.
Prioritize sources from: academic papers (arxiv, SSRN), practitioner
blogs, Polymarket/Kalshi documentation, regulatory filings, and 
on-chain data analysis (Dune Analytics, Arkham). De-prioritize 
generic "how to trade prediction markets" SEO articles.

## OUTPUT FORMAT

Structure your response as a reference atlas with these exact sections.
For each strategy or pattern, include:
- Mechanism (how it works mechanically)
- Edge source (why does the opportunity exist?)
- Current viability (working/dying/dead as of April 2026)
- Capital requirements and expected returns
- Automation feasibility (can this be botted?)
- Key risks and failure modes
- Specific examples with numbers

═══════════════════════════════════════════════════════════════

## SECTION 1: PLATFORM MECHANICS & MICROSTRUCTURE

For BOTH Polymarket and Kalshi, document:

### 1.1 Order Book Structure
- How does the CLOB (Central Limit Order Book) work?
- Order types available (limit, market, IOC, etc.)
- Tick size, minimum order size, position limits
- How are YES/NO shares created, split, and merged?
- What is the SPLIT and MERGE mechanism and how do 
  sophisticated traders use it?

### 1.2 Fee Structure (CRITICAL — affects every strategy)
- Polymarket: What exactly is the 2% winner fee? When was 
  it introduced? What changed in March 2026?
- Kalshi: Fee schedule, maker/taker, volume tiers
- How do fees differ by market category (sports vs politics 
  vs crypto vs weather)?
- What is the breakeven spread for arbitrage after fees?
- How have fees changed over time and what's the trend?

### 1.3 Settlement & Resolution
- How are markets resolved? Who decides?
- Oracle systems (Polymarket uses UMA)
- Resolution disputes — how common, how exploitable?
- Settlement timing and its implications for capital efficiency

### 1.4 API & Automation
- Polymarket CLOB API: endpoints, rate limits, authentication
- Kalshi API: same details
- WebSocket availability for real-time data
- Known bot frameworks and open-source tools
- Latency considerations — where should infrastructure be hosted?

### 1.5 On-Chain vs Off-Chain
- Polymarket on Polygon — what's on-chain vs off-chain?
- Can you read order flow on-chain? What data is public?
- Kalshi's on-chain components (if any)
- Dune Analytics dashboards for market intelligence

═══════════════════════════════════════════════════════════════

## SECTION 2: STRATEGY TAXONOMY — COMPLETE CATALOGUE

### Category A: Pure Arbitrage (Mechanical, No Opinion Required)

A1. Same-Market Parity Arbitrage
    - YES + NO < $1 → buy both, guaranteed profit
    - Current state: dead for retail? What spread is needed?
    - Who dominates this now? Named bots/wallets?

A2. Cross-Market Combinatorial Arbitrage  
    - Logically related markets with inconsistent pricing
    - The Cornell/arxiv paper on "Probabilistic Forest" —
      what did they find? How many opportunities? How large?
    - Can this be automated? What's the detection algorithm?

A3. Cross-Platform Arbitrage (Polymarket vs Kalshi)
    - Same event, different prices on different platforms
    - Capital efficiency problem (money locked on both sides)
    - Current tools and bots for this

A4. Temporal Arbitrage / Latency Arbitrage
    - Price updates lag real-world events by seconds
    - News-driven: how fast do markets react to breaking news?
    - The Iran strike incident — what happened exactly?
    - Is this still viable or are bots too fast?

### Category B: Statistical/Structural Edge (Pattern-Based)

B1. Longshot Bias Exploitation
    - Retail systematically overpays for low-probability outcomes
    - Academic evidence: how large is the bias?
    - Practical strategy: sell YES on longshots at 10-15¢ when 
      true probability is 3-5%?
    - Risk: the longshot actually hits. Portfolio management.

B2. Favorite-Longshot Bias (Reverse)
    - Near-certain outcomes (90-99¢) — are they underpriced?
    - "Tail-end trading": buying near-certainties at 95¢ for 
      5% guaranteed return. Capital efficiency?
    - The "penny stock" equivalent: 1-2¢ NO shares

B3. Market Lifecycle Mispricing
    - New markets are most mispriced (first quartile)
    - Volume < $50k: wild inefficiency, wide spreads
    - Volume > $1M: efficient pricing, only market-making works
    - How to systematically identify early-stage mispricings

B4. Sentiment/Momentum Overreaction
    - Breaking news causes price overshoot
    - Political markets after debate performances
    - Sports markets after injury announcements
    - Mean reversion patterns: how fast? How reliable?
    - THIS IS THE CORE OF TYPE 2 — GIVE MAXIMUM DETAIL

B5. Anchoring and Adjustment Failures
    - Markets that should move but don't (sticky prices)
    - Markets that moved too much (overreaction)
    - Calendar effects (markets near expiry behave differently)

B6. Correlated Market Leads/Lags
    - Does the political betting market lead the financial market?
    - Sports odds markets vs prediction markets — price discovery
    - Which platform leads: Polymarket or Kalshi?

### Category C: Information Edge (Expertise-Based)

C1. Domain Expert Models
    - Weather (NOAA forecasts vs market prices)
    - Economics (CME FedWatch vs Kalshi rate markets)
    - Politics (poll aggregation vs market prices)
    - Sports (advanced analytics vs market prices)
    - Which domains have the most exploitable gaps?

C2. AI/LLM-Powered Probability Models
    - Using LLMs to estimate event probabilities
    - Comparison with market prices — does it work?
    - Existing research and results
    - Specific model architectures that have been tried

C3. Ensemble / Meta-Models
    - Aggregating multiple forecasting sources
    - Superforecaster techniques applied to markets
    - Poll-of-polls type approaches

### Category D: Market Making & Liquidity Provision

D1. Spread Harvesting
    - Posting limit orders on both sides
    - Inventory risk management
    - Capital requirements and expected returns
    - Competition from professional market makers

D2. Polymarket Reward Programs
    - The 4% yield program for 2028 election markets
    - Liquidity mining incentives — what's available?
    - Referral programs and fee rebates

═══════════════════════════════════════════════════════════════

## SECTION 3: WHAT'S DEAD, WHAT'S DYING, WHAT'S ALIVE

For each strategy from Section 2, classify explicitly:

### 3.1 DEAD (was profitable, no longer works)
- What killed it? (bots? fee changes? regulation?)
- When did it die?
- Approximate total profit extracted before death

### 3.2 DYING (still works but window closing)
- How fast is the window closing?
- What's accelerating the decline?
- Estimated remaining lifespan

### 3.3 ALIVE (currently profitable)
- Approximate returns being generated
- Required infrastructure/capital
- Competition level
- Barrier to entry

### 3.4 EMERGING (new patterns forming)
- What new strategies are appearing?
- What structural changes are creating new opportunities?
- Margin trading on Kalshi — implications?
- Institutional entry (Paradigm, JPMorgan) — what changes?

═══════════════════════════════════════════════════════════════

## SECTION 4: REGULATORY LANDSCAPE & RISK

### 4.1 Current Legal Status
- CFTC classification and jurisdiction
- State-level challenges (Arizona, Nevada, etc.)
- US access restrictions (Polymarket officially blocks US?)

### 4.2 Pending Legislation
- "Prediction Markets Are Gambling Act"
- "STOP Corrupt Bets Act" 
- "Public Integrity in Financial Prediction Markets Act"
- "PREDICT Act"
- Likelihood and timeline of passage
- What exactly would each ban?

### 4.3 Insider Trading Framework
- Current rules on each platform
- CFTC enforcement actions to date
- The Iran/Maduro incidents — full details
- Federal prosecutor investigations

### 4.4 Implications for Traders
- What categories of markets are safest from regulation?
- What's the realistic timeline before major changes?
- Should we focus on Kalshi (CFTC-regulated) or 
  Polymarket (decentralized) for durability?

═══════════════════════════════════════════════════════════════

## SECTION 5: INFRASTRUCTURE & TOOLS

### 5.1 Existing Open-Source Tools
- Trading bots (GitHub repos, capabilities, quality)
- Data collection and monitoring tools
- Backtesting frameworks for prediction markets
- Analytics dashboards (Dune, etc.)

### 5.2 Data Sources
- Historical price data availability
- Order book data (real-time and historical)
- Resolution data (which markets resolved how)
- Volume and liquidity metrics
- On-chain data for Polymarket

### 5.3 Required Infrastructure for Each Strategy
- What do you need for arbitrage? (speed, capital, API)
- What do you need for mean-reversion? (models, data, patience)
- What do you need for market making? (capital, risk mgmt)

═══════════════════════════════════════════════════════════════

## SECTION 6: QUANTITATIVE BENCHMARKS

### 6.1 Platform Statistics
- Total volume by platform (monthly, trending)
- Number of active markets
- Average bid-ask spread by market category
- Top traders by profit (anonymized if needed)
- Profit distribution: what % of traders are profitable?

### 6.2 Strategy Performance Data
- Documented returns for each strategy type
- Sharpe ratios where available
- Drawdown characteristics
- Capital requirements and turnover

### 6.3 Market Accuracy
- Brier scores by platform and category
- Calibration curves
- Comparison with polls, expert forecasts, models
- The 67.2% Polymarket vs 93.1% PredictIt accuracy study

═══════════════════════════════════════════════════════════════

## ANTI-PATTERNS (what NOT to include)

- Do NOT include generic "what is a prediction market" explainers
- Do NOT include get-rich-quick marketing language
- Do NOT recommend strategies without quantifying expected returns
- Do NOT ignore fees in profitability calculations
- Do NOT treat Polymarket and Kalshi as identical — document 
  every difference
- Do NOT assume US legal access to Polymarket
- Do NOT cite sources older than 2024 unless for historical context

## META-INSTRUCTIONS

- When information conflicts between sources, note the conflict
- When data is unavailable, say so explicitly — do not fabricate
- Include specific numbers (dollar amounts, percentages, dates)
- Name specific wallets, bots, or traders when publicly documented
- Cross-reference between sections (e.g., "see Section 2.B4 for 
  the strategy that exploits this microstructure feature")
- End with a prioritized list of the 5 most promising strategies 
  for a small ($5-10k) systematic trader in April 2026
