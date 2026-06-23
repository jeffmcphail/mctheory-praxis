"""
engines/infobar_lstm/deflated_sharpe.py -- Deflated Sharpe Ratio
(Bailey & Lopez de Prado, 2014) for the pre-registered multiple-testing haircut.

WHY THIS EXISTS (protocol section 0 + 6.3): the validation pre-commits a
672-config grid (bar-type/threshold x model HPs). Nested walk-forward controls
the tune-vs-evaluate leak, but NOT the SELECTION BIAS of reporting the
best-of-672 -- the top config can clear a Sharpe threshold by luck alone. The
DSR deflates the observed Sharpe by the EXPECTED MAXIMUM Sharpe under the null
(no skill) given N (effectively independent) trials, and also corrects for
non-normal returns (skew/kurtosis) and finite sample length. It is the native
tool for exactly this problem, by the same author the pipeline is built on.

This module is wired BEFORE any model fit (Cycle 58b). It is the Run-C gating
metric; nothing calls it this cycle (no Sharpe to deflate without a model).

UNITS (read this -- the #1 way to get a wrong DSR): all Sharpe inputs are
PER-OBSERVATION (e.g. per-day), NOT annualized. The pipeline reports a
daily-aggregated Sharpe annualized by sqrt(365); convert back by dividing by
sqrt(periods/year), pass n_obs = number of daily observations, and skew/kurt of
the DAILY net returns. Mixing an annualized SR with a per-day n_obs silently
corrupts the DSR.

No external dependencies for the core: standard-normal CDF via math.erf,
inverse-CDF via Acklam's rational approximation (|err| < 1.2e-9). Identical
results in any environment -- important for a pre-registration artifact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EULER_MASCHERONI = 0.5772156649015329


# --- standard normal, dependency-free -------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's algorithm). Domain (0, 1)."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"_norm_ppf domain is (0,1), got {p}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return ((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0))
    if p <= p_high:
        q = p - 0.5
        r = q*q
        return ((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
                (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0))
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return (-(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0))


# --- PSR / DSR core --------------------------------------------------------

def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n_obs: int,
                               skew: float = 0.0, kurt: float = 3.0) -> float:
    """PSR: probability the true (per-obs) Sharpe exceeds sr_benchmark, given
    the observed sr, sample length n_obs, and return skew/kurtosis (kurt is
    NON-excess; normal = 3). Bailey & Lopez de Prado (2012)."""
    if n_obs < 2:
        return float("nan")
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    return _norm_cdf((sr - sr_benchmark) * math.sqrt(n_obs - 1) / denom)


def expected_max_sharpe(var_sr_across_trials: float, n_trials: int) -> float:
    """Expected MAXIMUM (per-obs) Sharpe under the null of zero skill across
    n_trials independent trials whose SR estimates have variance
    var_sr_across_trials. Gumbel / extreme-value approximation (Bailey & LdP
    2014). This is the deflation benchmark SR_0 -- the Sharpe you'd expect the
    best of n_trials to show by luck alone."""
    if n_trials < 2 or var_sr_across_trials <= 0.0:
        return 0.0
    g = EULER_MASCHERONI
    z1 = _norm_ppf(1.0 - 1.0 / n_trials)
    z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sr_across_trials) * ((1.0 - g) * z1 + g * z2)


@dataclass
class DSRResult:
    dsr: float                 # P(true SR > expected-max-under-null), in [0,1]
    sr0_expected_max: float    # the deflation benchmark used
    observed_sr: float
    n_trials: int
    n_obs: int


def deflated_sharpe_ratio(observed_sr: float, var_sr_across_trials: float,
                          n_trials: int, n_obs: int,
                          skew: float = 0.0, kurt: float = 3.0) -> DSRResult:
    """DSR = PSR evaluated at the expected-maximum-Sharpe benchmark for
    n_trials. It is the probability the SELECTED config's true Sharpe is
    positive AFTER accounting for having picked the best of n_trials (and for
    non-normality + finite sample). All SR inputs PER-OBSERVATION."""
    sr0 = expected_max_sharpe(var_sr_across_trials, n_trials)
    dsr = probabilistic_sharpe_ratio(observed_sr, sr0, n_obs, skew, kurt)
    return DSRResult(dsr=dsr, sr0_expected_max=sr0, observed_sr=observed_sr,
                     n_trials=n_trials, n_obs=n_obs)


# --- inputs for the "adjusted for config correlation" trial count ----------

def trial_sharpe_stats(trial_sharpes) -> dict:
    """{n, mean, var} of the per-observation SRs across all grid configs.
    `var` is the var_sr_across_trials input to expected_max_sharpe. Correlated
    configs (same bars / overlapping HPs) shrink this variance, which already
    tempers the deflation."""
    import numpy as np
    s = np.asarray([x for x in trial_sharpes
                    if x is not None and np.isfinite(x)], dtype=float)
    if s.size == 0:
        return {"n": 0, "mean": float("nan"), "var": float("nan")}
    return {"n": int(s.size), "mean": float(s.mean()),
            "var": float(s.var(ddof=1)) if s.size > 1 else 0.0}


def effective_independent_trials(corr_matrix) -> float:
    """Galwey (2009) effective number of independent tests from the correlation
    matrix of the N trial return series: N_eff = (sum sqrt(lambda_i))^2 /
    sum(lambda_i), lambda_i = eigenvalues of corr_matrix. Identity -> N (all
    independent); rank-1 (perfectly correlated) -> 1. This is the
    correlation-adjusted n_trials to pass to the DSR; the raw grid size (672) is
    the conservative upper bound (more deflation). Computable only once trial
    return series exist (Run C), so this is wired for then, not used now."""
    import numpy as np
    m = np.asarray(corr_matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] == 0:
        raise ValueError("corr_matrix must be a non-empty square matrix")
    lam = np.linalg.eigvalsh(m)
    lam = np.clip(lam, 0.0, None)           # PSD; clip tiny negatives
    denom = float(lam.sum())
    if denom <= 0.0:
        return float(m.shape[0])
    n_eff = float(np.sqrt(lam).sum() ** 2 / denom)
    return min(max(n_eff, 1.0), float(m.shape[0]))
