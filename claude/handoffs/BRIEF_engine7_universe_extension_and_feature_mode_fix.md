# Cycle 42 -- Engine 7 universe extension (42a) + run_cpo.py feature-mode fix (42d)

**Predecessor:** Cycle 41 -- Engine 7 live monitor pilot (commits
`e9c46d5` + `cf2abee`). Cycle 42 picks up items 41a (universe
extension) and 41d (feature-mode default fix) from that cycle's
Open items list.

**Mode:** RECON-then-implementation in one cycle, same shape as
Cycle 41.

## 42a -- Universe extension to atlas Exp 13's 6 assets

Goal: extend `PraxisFundingCollector` + `PraxisFundingMonitor` from
BTC+ETH (pilot) to the full atlas Exp 13 deployment universe:
BTC, ETH, SOL, XRP, ADA, AVAX. BNB excluded per atlas.

Components to update:
- Collector asset list (source-of-truth in
  `services/funding_collector_service.bat`)
- Monitor asset list (source-of-truth in
  `services/funding_monitor_service.bat` -- previously overrode
  the script's already-6-asset DEFAULT_ASSETS via explicit
  `--assets BTC,ETH`)
- `SUPPORTED_ASSETS` constant in `engines/crypto_data_collector.py`
  (gate for per-asset symbol lookups; previously held only BTC/ETH/SOL)
- Health check (`servers/praxis_mcp/tools/meta.py` `primary_monitored`
  dict) -- add `funding_signals` with threshold matching `funding_rates`

Acceptance for 42a:
- PraxisFundingCollector writes funding_rates rows for all 6 assets
  at each event
- PraxisFundingMonitor writes funding_signals rows for all 6 assets
  at each window
- funding_signals appears in get_collector_health (no longer unmonitored)
- Existing BTC/ETH rows unbroken (no schema migration needed)

## 42d -- --feature-mode default fix in run_cpo.py

Per memory #23: `--feature-mode` argparse default `"funding"` leaks
into output filenames via `features_suffix` for ALL strategies, not
just `funding_rate`. Caused the misnamed
`outputs/exp10_revival/cpo/phase3_models_funding.joblib` trap
(actually contained 64 Exp 10 TA models; deleted in Cycle 40 D2 setup).

Fix shape: Option A (gate by strategy). One-line override at the top
of `main()`:

    if args.strategy != "funding_rate":
        args.feature_mode = ""

Cycle 40's funding_rate path is unaffected (suffix stays `_funding`).
Future non-funding strategy runs get clean unsuffixed outputs.

## Out of scope (Cycle 43+ candidates)

- Cross-venue funding spread (atlas Exp 13 revival hypothesis #1)
- LSTM v2 architecture test
- Executor integration / real money
- Bear-regime accumulation analysis (needs ~30 days of monitor data)
- PMA backfill
- atlas_search engine-filter parameter
