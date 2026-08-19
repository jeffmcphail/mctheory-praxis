"""
engines/forced_trade/ -- Cycle 61 forced-trade DATA AUDIT.

MEASUREMENT ONLY. This package contains no strategy, no P&L, no Sharpe and no
backtest. Its single job is to answer, empirically, how many forced-trade events
exist in the data Praxis already holds and whether they are cleanly identifiable
-- so that a scenario x regime grid can be designed on measured event counts
rather than guesses.

Companions:
    docs/FORCED_TRADE_SCREEN.md    -- the three-question design-stage filter
    docs/FORCED_TRADE_TAXONOMY.md  -- 16 compulsion scenarios, triaged

Modules
-------
common      shared read-only DB access, time helpers, output plumbing
cascade     T1  A2 liquidation cascades   (parameterised flow-burst detector)
unlocks     T2  F1 token unlocks          (circulating_supply jump detector)
leveraged   T3  D1 leveraged tokens       (candidate universe audit)
oi_audit    T4  open-interest gap         (absence + degradation + collectability)
occupancy   T5  scenario x regime cell occupancy
run_audit   CLI

EVERY threshold is a parameter. Nothing is tuned to make event counts look
better; each detector reports a sensitivity sweep, and the SPREAD across
settings is the finding.

READ-ONLY: every connection to crypto_data.db is opened with mode=ro.
"""

__all__ = ["common", "cascade", "unlocks", "leveraged", "oi_audit", "occupancy"]
