"""The eigenfactor risk adjustment (Menchero, Wang and Orr 2011; Barra USE4).

A covariance estimated from a finite history gets its eigenvalues wrong in a predictable way: the
largest come out too large and the smallest too small, because the estimate's eigenvectors are
fitted to the sample's own noise -- a direction that happened to move little in the sample is
found, and called quiet. An optimiser goes to exactly those directions, so the portfolios it
builds realise more risk than forecast. The adjustment measures that bias by simulation and
undoes it.

Take the day's covariance F0 as the truth and diagonalise it, F0 = U0 D0 U0'. Simulate M
histories of factor returns from it -- independent normal draws over as many periods as the day's
estimate rests on -- and re-estimate each with the same estimator, so with the same weights, the
two half-lives and the lags included; diagonalise each, F_m = U_m D_m U_m'. The k-th simulated
eigenfactor u_mk has true variance u_mk' F0 u_mk and estimated variance d_mk, and over the draws

    lambda(k) = sqrt( (1/M) sum_m (u_mk' F0 u_mk) / d_mk )

Above one, the k-th eigenvalue is estimated too small; below one, too large. The adjusted
covariance scales each of the day's eigenvalues and keeps its eigenvectors:

    F = U0 diag(gamma^2 D0) U0',   gamma(k) = a (lambda(k) - 1) + 1

a = 1 is the simulation alone. USE4 found real data needs a above one: the simulation knows only
sampling error from a stationary normal history, and a real one is neither. One simulation a day
serves every a, so a can be chosen by testing.

**One-sided** (midoffice15's brief), gamma(k) = max(1, a (lambda(k) - 1) + 1): only the eigenvalues
the simulation finds too small are raised, and none is lowered -- the largest, which it finds too
large, are left as estimated. midoffice13's two-sided adjustment lowered them, and the
minimum-variance portfolios realised more risk under it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .bias import eigenportfolios
from .covariance import CovarianceEstimator

EIGEN_DRAWS = 1000
"""Simulated histories per covariance. On the S&P 500 model's one-day covariance, 252 to 937
periods of 25 factors, a thousand hold every rank's lambda to a Monte Carlo standard error of
0.002 at most, within 0.003 of four thousand's (midoffice13)."""

EIGEN_SEED = 13
"""The simulation is seeded, so the same covariance, estimator and history length give the same
lambda every time -- an as-of refit and a refit on the panel cut at its date agree to the bit."""


@dataclass(frozen=True, eq=False)
class EigenBias:
    """A covariance's eigenfactor bias, by simulation.

    `values` and `vectors` are the covariance's eigenvalues, largest first, and its eigenvectors
    as columns, each signed so its largest component is positive (`bias.eigenportfolios`);
    `bias` is lambda(k) rank by rank in that order, and `standard_error` its Monte Carlo
    standard error. `degenerate` counts the draw-ranks whose estimated eigenvalue was not
    positive, left out of their rank's mean.
    """

    covariance: np.ndarray = field(repr=False)
    values: np.ndarray = field(repr=False)
    vectors: np.ndarray = field(repr=False)
    bias: np.ndarray
    standard_error: np.ndarray = field(repr=False)
    draws: int
    n_periods: int
    degenerate: int = 0
    seconds: float = field(default=0.0, compare=False)

    def scales(self, a: float, one_sided: bool = False) -> np.ndarray:
        """gamma(k) = a (lambda(k) - 1) + 1, rank by rank: what each eigenfactor's volatility is
        multiplied by; one-sided, never below one."""
        gamma = a * (self.bias - 1.0) + 1.0
        return np.maximum(gamma, 1.0) if one_sided else gamma

    def adjusted(self, a: float, one_sided: bool = False) -> np.ndarray:
        """The covariance with each eigenvalue scaled by gamma(k)^2 and its eigenvectors kept.
        At a = 0 every gamma is one, and the covariance is returned as given, to the bit."""
        if a == 0.0:
            return self.covariance
        return (self.vectors * (self.scales(a, one_sided) ** 2 * self.values)) @ self.vectors.T

    def basis(self, a: float, one_sided: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """The adjusted covariance's eigenvalues and eigenvectors in the unadjusted order: rank k
        stays the same eigenfactor at every a, its variance scaled by gamma(k)^2."""
        return self.scales(a, one_sided) ** 2 * self.values, self.vectors


def eigen_bias(
    covariance: np.ndarray,
    estimator: CovarianceEstimator,
    n_periods: int,
    *,
    draws: int = EIGEN_DRAWS,
    seed: int = EIGEN_SEED,
) -> EigenBias:
    """Each eigenvalue rank's bias in `covariance` as `estimator` estimates it from `n_periods`
    periods: `draws` histories simulated from the covariance, independent normal draws, each
    re-estimated by the estimator itself and diagonalised (see the module docstring)."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"the covariance must be square, got shape {covariance.shape}")
    if draws < 2:
        raise ValueError(f"the simulation needs at least two draws, got {draws}")
    if n_periods < 2:
        raise ValueError(f"a history needs at least two periods, got {n_periods}")
    began = time.perf_counter()
    values, vectors = eigenportfolios(covariance)
    root = vectors * np.sqrt(np.clip(values, 0.0, None))
    k = covariance.shape[0]
    rng = np.random.default_rng(seed)
    ratios = np.full((draws, k), np.nan)
    degenerate = 0
    for m in range(draws):
        history = rng.standard_normal((n_periods, k)) @ root.T
        estimate = np.asarray(estimator.estimate(history).matrix, dtype=float)
        found, directions = np.linalg.eigh((estimate + estimate.T) / 2.0)
        found, directions = found[::-1], directions[:, ::-1]
        true = np.einsum("ik,ij,jk->k", directions, covariance, directions)
        positive = found > 0.0
        degenerate += int(k - positive.sum())
        ratios[m, positive] = true[positive] / found[positive]
    mean = np.nanmean(ratios, axis=0)
    bias = np.sqrt(mean)
    counts = np.sum(np.isfinite(ratios), axis=0)
    spread = np.nanstd(ratios, axis=0, ddof=1)
    standard_error = spread / np.sqrt(counts) / (2.0 * bias)
    return EigenBias(
        covariance=covariance,
        values=values,
        vectors=vectors,
        bias=bias,
        standard_error=standard_error,
        draws=draws,
        n_periods=n_periods,
        degenerate=degenerate,
        seconds=time.perf_counter() - began,
    )
