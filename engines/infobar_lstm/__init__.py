"""
engines/infobar_lstm/ -- information-bar directional model pipeline (Cycle 58).

A leakage-free, walk-forward research pipeline that consumes information bars
(engines/info_bars + the info_bars table) and produces triple-barrier labels,
causal features, purged/embargoed walk-forward splits, realistic cost/turnover
accounting, and benchmarks -- the apparatus to test (later, when data matures)
whether an info-bar directional model has a durable edge net of cost OOS.

DESIGN STANCE (read CYCLE58_VALIDATION_PROTOCOL.md before running anything):
  - Null hypothesis = "no durable edge net of cost OOS". The atlas supports it
    for every prior directional/microstructure attempt; the burden is on the
    alternative.
  - The current trade tape is ONE ~8-week regime (2026-04-29 -> 2026-06-22).
    This pipeline can PROVE the plumbing and CATCH leakage; it cannot establish
    a durable edge. Deployment validation is gated on the data-maturity
    precondition (protocol section 7).
  - HARD PAUSE: no model is fit in this cycle. model.py defines the LSTM; its
    fit function is NOT called until Jeff reviews the protocol (protocol
    section 1, Run B). run_audit.py is Run A -- everything EXCEPT the fit.

Modules:
  pipeline.py    bars -> causal features, triple-barrier labels, walk-forward
  costs.py       long/flat strategy returns + turnover + spot cost
  benchmarks.py  buy-hold, momentum baseline, null-at-rate, beta controls
  model.py       the small LSTM (defined; NOT fit this cycle)
  run_audit.py   Run A: plumbing + leakage audit on real bars (no fit)
  tests/         automated leakage / causality unit tests
"""
