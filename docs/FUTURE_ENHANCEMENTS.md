# Praxis Trading Platform — Future Enhancements & TODO

> Tracks pending work, research items, and future enhancements for the praxis trading platform.
> Items are organized by priority and category.
> 
> **Last Updated:** 2026-02-16
> **Source:** Aggregated from Claude memory, chat transcripts, and in-code TODOs

---

## 🔬 Research & Experimental Frameworks

### VR Profile Multi-Timescale Trading Signals ⭐ HIGH PRIORITY
**Status:** Not started  
**Origin:** praxis_main chat, 2026-02-16  

Design an experimental framework for using VR (Variance Ratio) profile information to derive trading signals and time horizoning beyond what the Burgess model uses directly.

**Core Insight:** Even though the Burgess basket construction operates at a single timescale, the VR profile reveals the *shape* of mean-reversion across many timescales. This multi-timescale structure can inform:

- **Entry/exit signals:** VR profile curvature may indicate optimal entry points (e.g., when short-lag VR diverges from long-lag VR, a reversion opportunity may be forming)
- **Trade time horizons:** The lag at which VR is minimized suggests the natural reversion timescale — use this to set holding period expectations
- **Regime detection:** Changes in VR profile shape over time may signal regime transitions (mean-reverting → trending) before scalar tests detect them
- **Position sizing modulation:** Strength of mean-reversion at the trading timescale (from profile) could modulate Kelly fraction

**Experimental Design (to be fleshed out):**
1. Backtest: Compare fixed-horizon exits vs VR-profile-informed exits
2. Study: VR profile stability over rolling windows as regime indicator
3. Study: Cross-sectional VR profile ranking as basket selection criterion
4. Study: VR profile second eigenvector projection as entry timing signal

**Dependencies:** CompositeSurface system (implemented), VRProfileCollector (implemented), historical data pipeline

---

### Flash Loan + Statistical Arbitrage Deep Dive
**Status:** Not started — follow up from praxis_main chat  
**Origin:** praxis_main chat series, 2026-02-12  

Explore flash loan strategies combined with statistical arbitrage. Initial feasibility analysis was completed in the flash loan pipeline work (Phase 5). Deeper exploration needed on:

- Optimal stat-arb patterns for flash loan execution windows
- Cross-DEX arbitrage identification using Burgess-style cointegration
- Gas cost modeling and profitability thresholds
- MEV protection strategies

---

## 🏗️ Infrastructure & Compute

### Fire Up Production Burgess Surface Compute
**Status:** Ready to execute  
**Origin:** Round 2.1 battle test, 2026-02-15  

The CompositeSurface system is implemented and tested. Need to run the full production compute:
- ~4,352 grid points × 6 statistics per point
- Estimated 2-3 hours on 8 cores
- Uses `burgess_full_requirement()` pre-built configuration
- Script: `scripts/build_surface.py` (exists, may need update for CompositeSurface)

### Integrate Surface Lookups into Burgess Model
**Status:** Blocked by surface compute  

Replace the current uncorrected ADF thresholds in the Burgess pipeline with microsecond surface lookups. All tests applied to stepwise residuals need corrected critical values:
- ADF t-statistic
- Hurst exponent  
- Half-life
- Johansen trace
- VR profile eigenvector projections
- VR profile Mahalanobis distance

### Add Groq and DeepSeek LLM Providers
**Status:** Planned  
**Origin:** AI Factory memory  

For cost-effective high-volume testing and agent operations. Not directly trading-related but supports the broader McTheory ecosystem.

---

## 📊 Statistical & Analytical

### QPT Feedback Loop Optimization
**Status:** Planned  
**Origin:** ai_factory_main chat series  

Complete the Quantitative Prompting Theory feedback loop — four sophistication levels with manual override controls. Trading platform integration for prompt optimization of agent-driven trading decisions.

### Coreify Logger, Scheduler, DAG
**Status:** Planned — after trading platform components are battle-tested  
**Origin:** core_main chat series  

Abstract PraxisLogger, PraxisScheduler, and DAG executor into McTheory Core for reuse across Agent Factory and other projects. Follows the build-then-abstract workflow.

---

## 🧪 Testing

### Round 4 Battle Test Design
**Status:** Pending design decision  
**Origin:** Round 3 completion, 2026-02-16  

Options under consideration:
- Workflow DAG / CPOExecutor end-to-end test
- Live/paper trading agent integration test
- Full pipeline stress test with real market data

### End-to-End Meta-Agent Integration Testing
**Status:** Planned  
**Origin:** AI Factory memory  

Meta-agents (Creator, Tester, Optimizer, Evolution) have only been unit tested with mocks. Need end-to-end testing with real LLM calls. Estimated cost: $28-78 across 7 phases.

---

## 📝 Notes

- Items marked ⭐ are high priority and should be addressed in the near term
- This document is the praxis-specific counterpart to the AI Factory's `FUTURE_ENHANCEMENTS.md` and `TODO.md`
- Cross-project items (Core abstractions, shared infrastructure) are tracked here with references to the originating chat series

---

*Document Version: 1.0.0*  
*Last Updated: 2026-02-16*  
*Maintained by: trading_platform chat series*
