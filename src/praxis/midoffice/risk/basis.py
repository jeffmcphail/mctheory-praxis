"""The factor basis a risk calculation runs on, and the covariance that lives on it.

A portfolio touches a subset of a model's factors.  Rather than padding every exposure
vector out to the model's full factor set so the dimensions conform -- which is what RAT
does, carrying a sparse `expVect` beside a zero-padded `expVectGlobal` -- this builds the
basis once from the union of the factors actually in play, slices the covariance to it, and
leaves every vector dense over that single ordering.

The point is not speed, though it helps when the covariance spans several models.  The point
is that there is nothing left to conform, so an entire class of defect cannot occur.  RAT
guards against it with a documented convention and uses the two vectors asymmetrically on
purpose; MCRLib dropped the distinction and its marginal-contribution calculation silently
passed the unpadded vector on both sides.  Neither failure is available here.

Slicing is numerically safe: a principal submatrix of a positive semi-definite matrix is
itself positive semi-definite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from .config import DEFAULTS, RiskConfig


@dataclass(frozen=True)
class PSDRepair:
    """What `nearest_psd` had to change, carried as data rather than written to a log.

    A repaired covariance matrix is a different matrix from the one the vendor published.
    Every number computed from it inherits that, so the record travels with the basis and
    ends up on the result.
    """

    repaired: bool
    n_negative: int
    most_negative: float
    floor_applied: float

    def describe(self) -> str:
        if not self.repaired:
            return "covariance was positive semi-definite; unchanged"
        return (
            f"covariance repaired: {self.n_negative} negative eigenvalue(s), "
            f"most negative {self.most_negative:.3e}, floored at {self.floor_applied:.3e}"
        )


def nearest_psd(
    matrix: np.ndarray,
    *,
    relative_floor: float | None = None,
    symmetry_tol: float | None = None,
    config: RiskConfig | None = None,
) -> tuple[np.ndarray, PSDRepair]:
    """Return the nearest positive semi-definite matrix, by eigenvalue clipping.

    Vendor covariance matrices arrive slightly non-PSD often enough to matter -- estimation
    noise, truncated precision, blocks pasted together from separately estimated models.  A
    marginally negative eigenvalue makes a variance come out negative, and the square root of
    a negative number ends a risk run.  RAT detects this and gives up: `RiskMetrics.cs:3361`
    raises "Risk Square is negative. Check if your matrix is positive definite."

    Negative eigenvalues are raised to `floor`, not reflected.  Reflecting the sign preserves
    magnitude, which is wrong: these eigenvalues are noise around zero and belong near zero.
    MCRLib's attempt at this computed `-lambda / 1e-5`, which flips the sign correctly and
    then multiplies the magnitude by 100,000 -- turning an eigenvalue of -1e-6 into +0.1 and
    making it one of the largest in the matrix.  `test_repair_does_not_amplify` pins that.

    `relative_floor` is expressed as a fraction of the largest eigenvalue, so the floor scales
    with the matrix.  Zero yields PSD; a positive value yields positive definite.

    **The floor is not only about invertibility, and zero is not the safe choice.**  Clipping
    to exactly zero leaves the matrix singular, and a book lying in a null direction then has
    its common-factor risk computed as a total cancellation -- the answer comes back 0.0, or
    worse, as rounding noise.  Measured over 15 matrix shapes, a factor-neutral book retained
    between -0.7 and 1.7 correct digits at a floor of zero and 9.3 at the calibrated default.
    `config.psd_relative_floor` carries the calibration; see `RiskConfig` for the table.

    Omit `relative_floor` to take the configured value.  Passing it explicitly overrides the
    configuration, and passing `0.0` deliberately restores the old singular behaviour.
    """
    config = config or DEFAULTS
    relative_floor = (
        config.psd_relative_floor if relative_floor is None else relative_floor
    )
    symmetry_tol = config.symmetry_tolerance if symmetry_tol is None else symmetry_tol
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"covariance must be square, got shape {matrix.shape}")
    if matrix.size == 0:
        return matrix.copy(), PSDRepair(False, 0, 0.0, 0.0)

    asymmetry = float(np.max(np.abs(matrix - matrix.T))) if matrix.size else 0.0
    if asymmetry > symmetry_tol:
        raise ValueError(
            f"covariance is not symmetric (max |A - A.T| = {asymmetry:.3e}); "
            "symmetrise deliberately rather than letting this pass silently"
        )
    # eigh assumes symmetry and is exact for it; average away float asymmetry below tolerance.
    symmetric = (matrix + matrix.T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    largest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    floor = relative_floor * max(largest, 0.0)

    below = eigenvalues < floor
    n_negative = int(np.count_nonzero(eigenvalues < 0.0))
    most_negative = float(eigenvalues.min()) if eigenvalues.size else 0.0

    if not below.any():
        return symmetric, PSDRepair(False, n_negative, most_negative, floor)

    clipped = np.where(below, floor, eigenvalues)
    repaired = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    repaired = (repaired + repaired.T) / 2.0  # kill reconstruction asymmetry
    return repaired, PSDRepair(True, n_negative, most_negative, floor)


@dataclass(frozen=True)
class FactorBasis:
    """An ordered factor set with its covariance, ready for dense arithmetic.

    `factor_ids` fixes the ordering; `covariance[i, j]` is the covariance of
    `factor_ids[i]` with `factor_ids[j]`.  Every exposure vector rendered against this basis
    is a dense array of the same length in the same order.
    """

    factor_ids: tuple[int, ...]
    covariance: np.ndarray
    proxied: Mapping[int, int] = field(default_factory=dict)
    repair: PSDRepair = field(default_factory=lambda: PSDRepair(False, 0, 0.0, 0.0))
    _index: Mapping[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self._index:
            object.__setattr__(
                self, "_index", {fid: i for i, fid in enumerate(self.factor_ids)}
            )
        cov = np.asarray(self.covariance, dtype=float)
        if cov.shape != (len(self.factor_ids), len(self.factor_ids)):
            raise ValueError(
                f"covariance shape {cov.shape} does not match "
                f"{len(self.factor_ids)} factors"
            )
        cov = cov.copy()
        cov.flags.writeable = False  # a basis is immutable; so is its matrix
        object.__setattr__(self, "covariance", cov)
        object.__setattr__(self, "proxied", dict(self.proxied))

    def __len__(self) -> int:
        return len(self.factor_ids)

    def index(self, factor_id: int) -> int:
        try:
            return self._index[factor_id]
        except KeyError:
            raise KeyError(
                f"factor {factor_id} is not in this basis "
                f"({len(self.factor_ids)} factors)"
            ) from None

    def __contains__(self, factor_id: int) -> bool:
        return factor_id in self._index

    def dense(self, exposures: Mapping[int, float]) -> np.ndarray:
        """Render a sparse {factor_id: exposure} mapping as a dense array on this basis.

        A factor absent from the mapping is zero.  A factor absent from the *basis* is an
        error rather than a silent drop -- that is the failure mode this whole design exists
        to make impossible, so it is not going to be swallowed here.
        """
        out = np.zeros(len(self.factor_ids), dtype=float)
        for factor_id, value in exposures.items():
            out[self.index(factor_id)] = value
        return out


@dataclass(frozen=True)
class FactorCovariance:
    """A model's full published factor covariance matrix, with its factor ordering."""

    factor_ids: tuple[int, ...]
    matrix: np.ndarray

    def __post_init__(self) -> None:
        ids = tuple(self.factor_ids)
        if len(set(ids)) != len(ids):
            dupes = sorted({f for f in ids if ids.count(f) > 1})
            raise ValueError(f"duplicate factor ids in covariance: {dupes[:8]}")
        m = np.asarray(self.matrix, dtype=float)
        if m.shape != (len(ids), len(ids)):
            raise ValueError(
                f"covariance shape {m.shape} does not match {len(ids)} factor ids"
            )
        object.__setattr__(self, "factor_ids", ids)
        object.__setattr__(self, "matrix", m)

    def basis_for(
        self,
        factor_ids: Iterable[int],
        *,
        proxies: Mapping[int, int] | None = None,
        relative_floor: float | None = None,
        config: RiskConfig | None = None,
    ) -> FactorBasis:
        """Slice this covariance down to `factor_ids`, repairing PSD if needed.

        `proxies` maps a factor absent from this matrix onto one that is present, which is how
        an artificial factor gets a covariance at all (RAT:
        `CovarMatrix.getCovarBlockExtraFactors`).  Note the semantics deliberately: two
        factors proxied to the same target become perfectly correlated with each other and
        share its variance, because as far as the covariance is concerned they are the same
        factor.  That is the intended meaning, and it is worth stating because it is
        surprising when met in output rather than in a docstring.
        """
        proxies = dict(proxies or {})
        wanted = tuple(sorted(set(factor_ids)))

        own = {f: i for i, f in enumerate(self.factor_ids)}
        rows: list[int] = []
        used_proxy: dict[int, int] = {}
        missing: list[int] = []
        for fid in wanted:
            if fid in own:
                rows.append(own[fid])
            elif fid in proxies and proxies[fid] in own:
                rows.append(own[proxies[fid]])
                used_proxy[fid] = proxies[fid]
            else:
                missing.append(fid)

        if missing:
            raise KeyError(
                f"{len(missing)} factor(s) are absent from the covariance and have no "
                f"proxy: {missing[:8]}"
            )

        idx = np.asarray(rows, dtype=int)
        block = self.matrix[np.ix_(idx, idx)] if idx.size else np.zeros((0, 0))
        repaired, repair = nearest_psd(
            block, relative_floor=relative_floor, config=config
        )
        return FactorBasis(
            factor_ids=wanted,
            covariance=repaired,
            proxied=used_proxy,
            repair=repair,
        )


def active_basis(
    covariance: FactorCovariance,
    *exposure_sets: Mapping[int, float],
    proxies: Mapping[int, int] | None = None,
    relative_floor: float | None = None,
    config: RiskConfig | None = None,
) -> FactorBasis:
    """Build the basis spanning every factor in play across the given exposure sets.

    Called once per (portfolio, benchmark, model date) and reused for every metric computed
    against it.  Passing portfolio and benchmark together is the usual case: active exposure
    is their difference, so both must live on one basis for the subtraction to mean anything.
    """
    union: set[int] = set()
    for exposures in exposure_sets:
        union.update(exposures.keys())
    return covariance.basis_for(
        union, proxies=proxies, relative_floor=relative_floor, config=config
    )
