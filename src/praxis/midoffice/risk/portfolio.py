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
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .basis import FactorBasis, FactorCovariance, active_basis
from .exposures import ExposureVector, SpecificRisk
from .metrics import RiskDecomposition, beta, correlation, covariance, decompose


class WeightBasis(Enum):
    """Whether securities the risk model does not cover sit in the weight denominator.

    `FULL` keeps them there, so covered weights sum to less than one and the portfolio's
    factor exposure is scaled down by the uncovered fraction. **This was the production
    convention in RAT and is the default here**, and the reason is not a technical one:
    portfolio managers reconcile weights and market values against their own systems, and a
    tool whose weights do not tie is a tool they will not believe. Uncovered positions were
    made visible instead -- they show a blank total risk in the report, and a separate
    listing named them.

    It looks like an error, because risk is understated by roughly the uncovered fraction.
    It is a deliberate trade of a small, flagged bias for numbers a reader will trust, and
    it should not be quietly "corrected".

    `COVERED` redistributes the uncovered weight across the covered names, so the covered
    exposures carry the book's full weight and the risk is no longer scaled down. Choose it
    when the risk number matters more than the reconciliation.

    Note what the factor is: `full / covered`, not `1 / covered`. The uncovered weight is
    redistributed, which preserves the book's invested total; it is not a renormalisation to
    one. For a book whose weights sum to one the two are identical -- which is why a fixture
    that sums to one, as hand-built fixtures naturally do, cannot tell them apart. For a
    60%-invested sleeve they differ by a factor of 1.67, and `1 / covered` would report the
    sleeve as though it were fully invested.
    """

    FULL = "full"
    COVERED = "covered"


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
    benchmark_exposure_override: ExposureVector | None = None

    @property
    def benchmark_exposure(self) -> ExposureVector:
        """The exposure this security carries on the benchmark side.

        Normally identical to `exposure` -- a security is the same security in both books.
        It can differ for a bucketed holding whose composition is not the same on the two
        sides, which is why RAT's `car` has a `BucketSecurity` branch building the active
        vector as `w*x_port - bw*x_bench` instead of `(w - bw)*x` (`RiskMetrics.cs:2060`).
        That case is fixed-income and deferred, but the general form costs nothing here and
        reduces exactly to the simple one when the two are equal.
        """
        return self.exposure if self.benchmark_exposure_override is None else self.benchmark_exposure_override

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
        return self.benchmark_weight * self.benchmark_exposure

    def active_weighted(self) -> ExposureVector:
        """This position's contribution to active exposure: w*x_port - bw*x_bench.

        Reduces to `(w - bw) * x` whenever the two sides carry the same exposure, which is
        every case except a bucketed holding.
        """
        return self.weighted() - self.benchmark_weighted()


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
    covered_weight_sum: float = 0.0
    covered_benchmark_weight_sum: float = 0.0
    weight_basis: "WeightBasis" = None  # type: ignore[assignment]

    def qualifications(self) -> tuple[str, ...]:
        notes: list[str] = []
        if self.n_uncovered:
            basis = self.weight_basis or WeightBasis.FULL
            note = (
                f"{self.n_uncovered} position(s) carrying {self.uncovered_weight:.4f} "
                "of weight are not covered by the risk model and contribute no risk"
            )
            if basis is WeightBasis.FULL:
                note += (
                    "; weights keep them in the denominator, so factor exposure and every "
                    "risk number below are scaled down by that fraction"
                )
            else:
                note += (
                    "; weights were renormalised over covered names, so they no longer "
                    "reconcile against the full book"
                )
            notes.append(note)
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
    weight_basis: WeightBasis = WeightBasis.FULL

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
    def of(
        cls,
        name: str,
        positions: Iterable[Position],
        benchmark_name: str = "",
        weight_basis: WeightBasis = WeightBasis.FULL,
    ) -> "Portfolio":
        return cls(
            name=name,
            positions=tuple(positions),
            benchmark_name=benchmark_name,
            weight_basis=weight_basis,
        )

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

    def _covered_weight(self, of) -> float:
        return float(sum(of(p) for p in self.positions if p.covered))

    def weight_scale(self, weight_of) -> float:
        """The factor this side's aggregate exposure is multiplied by under the current basis.

        Public because anything that decomposes the aggregate has to apply the same factor to
        its parts. Scaling the aggregate but not the per-position vectors leaves the
        contributions summing to `risk / scale` instead of `risk` -- a real defect that lived
        here until the parity harness found it, and which `check_sum` diagnoses correctly as
        parts and whole "computed against different aggregates".
        """
        if self.weight_basis is not WeightBasis.COVERED:
            return 1.0
        covered = self._covered_weight(weight_of)
        full = float(sum(weight_of(p) for p in self.positions))
        if abs(covered) <= 1e-15 or abs(full - covered) <= 1e-15:
            return 1.0
        return full / covered

    def _sum(self, weighted, weight_of) -> ExposureVector:
        total = ExposureVector.zero()
        for position in self.positions:
            if not position.covered:
                continue
            total = total + weighted(position)
        scale = self.weight_scale(weight_of)
        return total * scale if scale != 1.0 else total

    @property
    def exposure(self) -> ExposureVector:
        return self._sum(Position.weighted, lambda p: p.weight)

    @property
    def benchmark_exposure(self) -> ExposureVector:
        return self._sum(Position.benchmark_weighted, lambda p: p.benchmark_weight)

    @property
    def active_exposure(self) -> ExposureVector:
        """Portfolio exposure less benchmark exposure.

        Built by differencing the two sides rather than from per-position active weights,
        because under `WeightBasis.COVERED` each side renormalises by its own factor and the
        per-position form would silently use the wrong one. Under `FULL` the two are
        identical, which `test_active_exposure_from_active_weights_equals_the_difference_of_sums`
        pins.
        """
        return self.exposure - self.benchmark_exposure

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
            covered_weight_sum=self._covered_weight(lambda p: p.weight),
            covered_benchmark_weight_sum=self._covered_weight(
                lambda p: p.benchmark_weight
            ),
            weight_basis=self.weight_basis,
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
