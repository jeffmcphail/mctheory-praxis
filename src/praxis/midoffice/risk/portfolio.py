"""Positions, portfolios, and the risk triple at both levels.

A position holds its security's exposure vector **unweighted** and carries its weights
separately.  Aggregation multiplies by weight and sums.  That split follows RAT, where
`RiskData.expVect` is the security's own vector and the portfolio vector is built by
weighting and adding, and it matters: a security's standalone risk is a property of the
security, while its effect on the portfolio is a property of the holding.

Three levels come out of one structure.  Portfolio exposure is the weighted sum; benchmark
exposure is the same sum over benchmark weights; active is their difference.  All three live
on one basis -- the union of the factors either side touches -- because subtracting vectors
rendered on different bases is meaningless, and is the exact defect MCRLib shipped.

**Weights arrive from outside.** They come from prices, holdings and a choice between market
value, clean market value and exposure value, none of which belongs in a pure calculation.
This layer takes whichever basis the caller computed and does not ask which it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .basis import FactorBasis, FactorCovariance, active_basis
from .exposures import ExposureVector, SpecificRisk
from .metrics import RiskDecomposition, beta, correlation, covariance, decompose


@dataclass(frozen=True)
class Position:
    """One security's holding, in a portfolio and in its benchmark.

    A name held only in the benchmark is a position with `weight` zero and a non-zero
    `benchmark_weight`; a name held only in the portfolio is the reverse.  Both are real and
    both carry active risk, so both are positions rather than absences.

    `covered` says whether the risk model covers this security, and the loader decides it.
    Deriving it from an empty exposure vector would be wrong: cash has no factor exposure and
    no specific risk, and is covered -- it is genuinely riskless, not unmodelled.
    """

    security_code: str
    exposure: ExposureVector = field(default_factory=ExposureVector.zero)
    weight: float = 0.0
    benchmark_weight: float = 0.0
    covered: bool = True

    @property
    def active_weight(self) -> float:
        return self.weight - self.benchmark_weight

    @property
    def is_held(self) -> bool:
        return self.weight != 0.0

    @property
    def is_in_benchmark(self) -> bool:
        return self.benchmark_weight != 0.0

    def weighted(self) -> ExposureVector:
        return self.weight * self.exposure

    def benchmark_weighted(self) -> ExposureVector:
        return self.benchmark_weight * self.exposure

    def active_weighted(self) -> ExposureVector:
        return self.active_weight * self.exposure


@dataclass(frozen=True)
class CoverageSummary:
    """What the numbers were computed over, and what they quietly left out.

    RAT tracks all of this -- `isCovered`, `numBenchNotHeldPositions`,
    `numOutofBenchPositions`, the `*IncNc` value family -- and it is easy to leave until last
    and tedious to retrofit, so it is here from the start.  Weight sums are reported rather
    than asserted: a long/short book or a partly invested portfolio legitimately will not sum
    to one, but a sum near two is the signature of double counting.
    """

    n_positions: int
    n_held: int
    n_in_benchmark: int
    n_benchmark_not_held: int
    n_out_of_benchmark: int
    n_uncovered: int
    weight_sum: float
    benchmark_weight_sum: float
    uncovered_weight: float

    def qualifications(self) -> tuple[str, ...]:
        notes: list[str] = []
        if self.n_uncovered:
            notes.append(
                f"{self.n_uncovered} position(s) carrying {self.uncovered_weight:.4f} "
                "of weight are not covered by the risk model and contribute no risk"
            )
        if self.n_benchmark_not_held:
            notes.append(
                f"{self.n_benchmark_not_held} benchmark name(s) are not held"
            )
        if self.n_out_of_benchmark:
            notes.append(
                f"{self.n_out_of_benchmark} held name(s) are not in the benchmark"
            )
        return tuple(notes)


@dataclass(frozen=True)
class Portfolio:
    """A set of positions with their benchmark weights, and the exposures they imply."""

    name: str
    positions: tuple[Position, ...]
    benchmark_name: str = ""

    def __post_init__(self) -> None:
        positions = tuple(self.positions)
        seen: set[str] = set()
        for position in positions:
            if position.security_code in seen:
                raise ValueError(
                    f"duplicate security {position.security_code!r} in portfolio "
                    f"{self.name!r}; aggregate lots before constructing the portfolio"
                )
            seen.add(position.security_code)
        object.__setattr__(self, "positions", positions)

    @classmethod
    def of(cls, name: str, positions: Iterable[Position], benchmark_name: str = "") -> "Portfolio":
        return cls(name=name, positions=tuple(positions), benchmark_name=benchmark_name)

    def __len__(self) -> int:
        return len(self.positions)

    def __iter__(self):
        return iter(self.positions)

    def position(self, security_code: str) -> Position:
        for position in self.positions:
            if position.security_code == security_code:
                return position
        raise KeyError(f"no position for {security_code!r} in {self.name!r}")

    # -- exposures ----------------------------------------------------------

    def _sum(self, weighted) -> ExposureVector:
        total = ExposureVector.zero()
        for position in self.positions:
            if not position.covered:
                continue
            total = total + weighted(position)
        return total

    @property
    def exposure(self) -> ExposureVector:
        return self._sum(Position.weighted)

    @property
    def benchmark_exposure(self) -> ExposureVector:
        return self._sum(Position.benchmark_weighted)

    @property
    def active_exposure(self) -> ExposureVector:
        """Built directly from active weights rather than by differencing two sums.

        Algebraically identical, and it keeps the subtraction at the position level where the
        two sides are guaranteed to be the same security.
        """
        return self._sum(Position.active_weighted)

    # -- basis --------------------------------------------------------------

    def basis(
        self,
        covariance_matrix: FactorCovariance,
        *,
        proxies: Mapping[int, int] | None = None,
        relative_floor: float = 0.0,
    ) -> FactorBasis:
        """The basis spanning every factor either side touches. Build once, reuse."""
        return active_basis(
            covariance_matrix,
            self.exposure.exposures,
            self.benchmark_exposure.exposures,
            proxies=proxies,
            relative_floor=relative_floor,
        )

    # -- coverage -----------------------------------------------------------

    def coverage(self) -> CoverageSummary:
        held = [p for p in self.positions if p.is_held]
        in_bench = [p for p in self.positions if p.is_in_benchmark]
        uncovered = [p for p in self.positions if not p.covered]
        return CoverageSummary(
            n_positions=len(self.positions),
            n_held=len(held),
            n_in_benchmark=len(in_bench),
            n_benchmark_not_held=sum(
                1 for p in self.positions if p.is_in_benchmark and not p.is_held
            ),
            n_out_of_benchmark=sum(
                1 for p in self.positions if p.is_held and not p.is_in_benchmark
            ),
            n_uncovered=len(uncovered),
            weight_sum=float(sum(p.weight for p in self.positions)),
            benchmark_weight_sum=float(
                sum(p.benchmark_weight for p in self.positions)
            ),
            uncovered_weight=float(sum(p.weight for p in uncovered)),
        )


@dataclass(frozen=True)
class PortfolioRisk:
    """The risk triple at all three levels, with everything needed to read it.

    `active` is tracking error. The provenance on each decomposition and the coverage summary
    together say what the numbers were computed from and what was substituted to get them.
    """

    portfolio_name: str
    benchmark_name: str
    portfolio: RiskDecomposition
    benchmark: RiskDecomposition
    active: RiskDecomposition
    beta: float
    correlation: float
    coverage: CoverageSummary

    @property
    def tracking_error(self) -> float:
        """Conventional name for active total risk."""
        return self.active.total_risk

    @property
    def is_exact(self) -> bool:
        return (
            self.portfolio.is_exact
            and self.benchmark.is_exact
            and not self.coverage.qualifications()
        )

    def qualifications(self) -> tuple[str, ...]:
        """Everything a reader should know before trusting these numbers."""
        notes: list[str] = []
        notes.extend(self.portfolio.qualifications())
        notes.extend(self.coverage.qualifications())
        return tuple(dict.fromkeys(notes))  # de-duplicate, preserve order


def position_risk(position: Position, basis: FactorBasis) -> RiskDecomposition:
    """A security's standalone risk: sqrt(specific^2 + x'Fx) on its own unweighted vector.

    This is RAT's position-level `totalRisk(rd, prd)` (`RiskMetrics.cs:1038`), which uses
    `rd.expVect` -- the security's exposure, not the holding's. A position's *effect* on the
    portfolio is a contribution, and contributions are a separate family.
    """
    return decompose(position.exposure, basis)


def portfolio_risk(
    portfolio: Portfolio, basis: FactorBasis
) -> PortfolioRisk:
    """The full triple for a portfolio against its benchmark on a shared basis."""
    portfolio_vector = portfolio.exposure
    benchmark_vector = portfolio.benchmark_exposure
    active_vector = portfolio.active_exposure
    return PortfolioRisk(
        portfolio_name=portfolio.name,
        benchmark_name=portfolio.benchmark_name,
        portfolio=decompose(portfolio_vector, basis),
        benchmark=decompose(benchmark_vector, basis),
        active=decompose(active_vector, basis),
        beta=beta(portfolio_vector, benchmark_vector, basis),
        correlation=correlation(portfolio_vector, benchmark_vector, basis),
        coverage=portfolio.coverage(),
    )
