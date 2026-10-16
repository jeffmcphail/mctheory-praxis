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

Cycle 66 -- A2 validated against Bybit liquidation LABELS (the confusion matrix
Cycle 61 could not compute). The Cycle 61 modules above are imported, never
edited; the detector is scored exactly as built.

labels            input snapshot, coverage mask from the collector LOG
label_inventory   T1  run accounting, mask validation, clusters, bankruptcy bias
label_scoring     T2  2x2 / precision-recall both ways, detector internals, book stress
cascade_profile   T3  event-time profiles + reversion vs matched controls
signature_sizing  T4  signature term counts vs labelled sample (no implementation)
run_labelled      CLI

Cycle 62B -- D1 leveraged-token rebalance mechanism study.

leveraged_class   T4  re-measure of the leveraged-name class the Cycle 60
                      substring rule deleted (fix lives in xsec_reversal/universe.py)
blvt_mechanism    T1  locate rebalances from 1m market prices (token vs perp slope
                      breaks) and test scheduled vs triggered; FTX BULL/BEAR as the
                      positive control. Stops at the gate: measures no pressure.

EVERY threshold is a parameter. Nothing is tuned to make event counts look
better; each detector reports a sensitivity sweep, and the SPREAD across
settings is the finding.

READ-ONLY: every connection to crypto_data.db is opened with mode=ro.
"""

__all__ = ["common", "cascade", "unlocks", "leveraged", "oi_audit", "occupancy",
           "labels", "label_inventory", "label_scoring", "cascade_profile",
           "signature_sizing", "leveraged_class", "blvt_mechanism"]
