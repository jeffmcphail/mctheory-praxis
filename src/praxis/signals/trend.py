"""
Trend-following signal generators.

Registry paths:
  signals.sma_crossover → praxis.signals.trend.SMACrossover
  signals.ema_crossover → praxis.signals.trend.EMACrossover
"""

from praxis.signals import SMACrossover, EMACrossover

__all__ = ["SMACrossover", "EMACrossover"]
