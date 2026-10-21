"""Risk decomposed across factors and factor groups.

The same identity that governs position contributions governs this one.  Partition the
exposure vector into parts whose sum is the whole, and each part's contribution is

    contribution_g = cov(x_g, x) / sigma

so the contributions sum to `x'Fx / sigma`.  The remainder, `specific^2 / sigma`, is the
specific-risk row, and together they come to `sigma` exactly.  Positions partition the book
by holding; this partitions it by factor.  A single factor is a group of one, so factor-level
and group-level decomposition are one implementation.

That specific-risk row is not an afterthought.  RAT carries it as a named sibling of the
factor groups in the report tree -- `FactorTree.SPECIFIC_RISK` -- with
`contribToRisk = sRisk^2 / fgdTotalRisk` (`RiskMetrics.cs:4390-4394`), and
`fdTotalRisk` is `sqrt(sRisk^2 + x'Fx)`, total risk rather than common-factor risk.  Decompose
against common-factor risk instead and every proportion in the report is subtly wrong.

**Two proportions, and they answer different questions.**  `risk_proportion` is built from
standalone risk and ignores every correlation, so it does not sum to one -- it says how big
this factor's own risk is relative to the total.  `litterman_proportion` is built from the
contribution and does sum to one.  RAT reports both (`RISK_PROP` and `LITT_PROP`) precisely
because the intuitive one does not add up, and a reader who assumes it should will conclude
the numbers are broken.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .basis import FactorBasis
from .exposures import ExposureVector
from .factors import SPECIFIC_RISK, FactorTree

_RISK_TOL = 1e-12


@dataclass(frozen=True)
class DecompositionRow:
    """One row of a decomposition: a factor, a factor group, or the specific-risk row."""

    name: str
    factor_ids: tuple[int, ...]
    exposure: float
    standalone_risk: float
    marginal: float
    contribution: float
    risk_proportion: float
    litterman_proportion: float
    is_specific: bool = False


@dataclass(frozen=True)
class Decomposition:
    """A full decomposition, including the specific row, with the identity it satisfies."""

    rows: tuple[DecompositionRow, ...]
    total_risk: float
    common_factor_risk: float
    specific_risk: float
    sums_to: float = 0.0
    parent: str = ""

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def row(self, name: str) -> DecompositionRow:
        for row in self.rows:
            if row.name == name:
                return row
        raise KeyError(f"no row named {name!r} in this decomposition")

    @property
    def factor_rows(self) -> tuple[DecompositionRow, ...]:
        return tuple(r for r in self.rows if not r.is_specific)

    def residual(self) -> float:
        """`sum(contribution) - sums_to`, which must be zero up to float error.

        At the top level `sums_to` is total risk. Drilling into a group it is that group's
        own contribution: the children of a node sum to the node, not to the whole book.
        """
        return float(sum(r.contribution for r in self.rows) - self.sums_to)

    def check(self, *, tolerance: float = 1e-10) -> float:
        residual = self.residual()
        if abs(residual) > tolerance:
            raise ValueError(
                f"decomposition contributions sum to "
                f"{sum(r.contribution for r in self.rows):.12g} against an expected "
                f"{self.sums_to:.12g} (residual {residual:.3e}); the partition does not "
                "cover what it claims to decompose"
            )
        return residual


def _decompose(
    vector: ExposureVector,
    basis: FactorBasis,
    parts: Sequence[tuple[str, tuple[int, ...]]],
    *,
    include_specific: bool = True,
    parent: str = "",
) -> Decomposition:
    """Decompose over a partition, given as (name, factor ids) parts.

    `include_specific` adds the specific-risk row, making the contributions sum to total
    risk. Omit it when decomposing the children of a group: specific risk sits under no
    factor group, and the children sum to their parent's contribution instead.
    """
    x = vector.dense(basis)
    covariance = basis.covariance
    fx = covariance @ x if len(basis) else np.zeros(0)

    cf_variance = float(x @ fx) if len(basis) else 0.0
    specific_variance = vector.specific.variance()
    total_variance = cf_variance + specific_variance
    total_risk = float(np.sqrt(max(total_variance, 0.0)))

    rows: list[DecompositionRow] = []
    if total_risk <= _RISK_TOL:
        for name, ids in parts:
            rows.append(DecompositionRow(name, tuple(ids), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        if include_specific:
            rows.append(
                DecompositionRow(SPECIFIC_RISK, (), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, is_specific=True)
            )
        return Decomposition(tuple(rows), 0.0, 0.0, 0.0, sums_to=0.0, parent=parent)

    for name, ids in parts:
        positions = [basis.index(f) for f in ids if f in basis]
        if not positions:
            rows.append(DecompositionRow(name, tuple(ids), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        idx = np.asarray(positions, dtype=int)

        # The part's own exposure vector: this group's entries, zero elsewhere.
        x_g = np.zeros_like(x)
        x_g[idx] = x[idx]

        # Standalone risk ignores everything outside the group. For a single factor this
        # reduces to |x_f| * sqrt(F_ff).
        standalone = float(np.sqrt(max(x_g @ covariance @ x_g, 0.0)))
        contribution = float(x_g @ fx) / total_risk
        exposure = float(x[idx].sum()) if len(idx) > 1 else float(x[idx[0]])
        # Marginal is per unit of the group's own exposure, where that is meaningful.
        marginal = (
            float(fx[idx[0]]) / total_risk
            if len(idx) == 1
            else (contribution / exposure if abs(exposure) > _RISK_TOL else 0.0)
        )
        rows.append(
            DecompositionRow(
                name=name,
                factor_ids=tuple(sorted(ids)),
                exposure=exposure,
                standalone_risk=standalone,
                marginal=marginal,
                contribution=contribution,
                risk_proportion=(standalone / total_risk) ** 2,
                litterman_proportion=contribution / total_risk,
            )
        )

    specific_contribution = specific_variance / total_risk
    specific_risk = float(np.sqrt(max(specific_variance, 0.0)))
    if include_specific:
        rows.append(
            DecompositionRow(
                name=SPECIFIC_RISK,
                factor_ids=(),
                exposure=0.0,
                standalone_risk=specific_risk,
                marginal=0.0,
                contribution=specific_contribution,
                risk_proportion=(specific_risk / total_risk) ** 2,
                litterman_proportion=specific_contribution / total_risk,
                is_specific=True,
            )
        )

    return Decomposition(
        rows=tuple(rows),
        total_risk=total_risk,
        common_factor_risk=float(np.sqrt(max(cf_variance, 0.0))),
        specific_risk=specific_risk,
        sums_to=float(sum(r.contribution for r in rows)),
        parent=parent,
    )


def factor_decomposition(
    vector: ExposureVector, basis: FactorBasis, *, names: Mapping[int, str] | None = None
) -> Decomposition:
    """Decompose risk factor by factor. Every factor in the basis gets a row."""
    names = dict(names or {})
    parts = [(names.get(f, str(f)), (f,)) for f in basis.factor_ids]
    return _decompose(vector, basis, parts)


def group_decomposition(
    vector: ExposureVector,
    basis: FactorBasis,
    tree: FactorTree,
    *,
    parent: str | None = None,
) -> Decomposition:
    """Decompose risk across the children of a group, plus the specific-risk row.

    Decomposing the children of a node -- rather than every node in the tree -- is what makes
    the parts a partition, which is what makes them sum. Call it again on a child to drill in.
    """
    parent = parent or tree.root
    is_root = parent == tree.root
    children = tree.children_of(parent)
    if not children:
        return factor_decomposition(vector, basis)
    parts = [(child, tuple(sorted(tree.factors_under(child)))) for child in children]

    # A missing child loses risk silently, so check coverage of whatever this call claims to
    # decompose: the whole basis at the root, the parent's own factors when drilling in.
    covered = {f for _, ids in parts for f in ids}
    expected = set(basis.factor_ids) if is_root else set(tree.factors_under(parent))
    stray = sorted(expected - covered)
    if stray:
        where = "the basis" if is_root else repr(parent)
        raise ValueError(
            f"{len(stray)} factor(s) in {where} sit outside every child of {parent!r} "
            f"and would be silently dropped from the decomposition: {stray[:8]}"
        )
    return _decompose(vector, basis, parts, include_specific=is_root, parent=parent)
