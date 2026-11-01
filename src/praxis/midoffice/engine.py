"""The engine's public entry point: four dates in, one result with its provenance.

Everything below this module computes a part.  `asof` resolves the four dates, `value` turns
holdings and market data into weights, `risk` turns weights and exposures into numbers.  This
assembles them, and the assembly is where the seam between the layers is actually visible:
`value` never imports `risk`, `risk` never imports `value`, and they meet here at the weights.

The interface is four dates rather than one, which corrects the single-date signature in an
earlier draft of the roadmap.  A model estimated on one date, applied to exposures measured on
another, priced on a third, against a security master from a fourth is an ordinary request.
RAT says so in its cache keys -- `getFundGroupKey(securityDate, marketDate)` and
`getRiskModelKey(modelDate, exposureDate)` -- and docs/ASOF_DATING.md has the detail.

**The weights come from the exposure basis, not the market basis.**  That is RAT's `expWeight`
(`RiskData.cs:99`), an exposure value over the book's market value, which makes a weight a
leverage ratio rather than a share of the book.  A futures overlay has a market value of zero
by formula, so market weights would be undefined for it; exposure weights state how levered
the book is through each position, which is what a risk number needs.  docs/VALUE_LAYER_SURVEY.md
has why.

**One provenance stream, not three.**  Each layer records what it substituted -- dates that
resolved backwards, prices that were missing, a covariance that was repaired, a book that
cannot resolve its own conditioning -- and a reader needs all of it in one place or they will
read none of it.  `RiskResult.qualifications()` is that place.  It is the structural claim the
whole port rests on: RAT computes all of this and writes it to a log stream running parallel to
the results, so an old number tells you what it was and not what it was computed over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .asof import AsOf
from .risk.basis import FactorBasis, FactorCovariance, active_basis
from .risk.config import DEFAULTS, RiskConfig
from .risk.contributions import Contribution, Level, contributions
from .risk.decomposition import Decomposition, factor_decomposition, group_decomposition
from .risk.exposures import ExposureVector
from .risk.factors import FactorTree
from .risk.portfolio import (
    Portfolio,
    PortfolioRisk,
    Position,
    WeightBasis,
    portfolio_risk,
)
from .risk.tail import TailRisk, TailSpec, tail_risk
from .value.market import LoadNotes
from .value.valuation import PortfolioValue, ValuationNotes, ValueBasis


@dataclass(frozen=True)
class RiskResult:
    """Every number a run produced, and everything that was substituted to produce it."""

    as_of: AsOf
    portfolio_name: str
    benchmark_name: str
    risk: PortfolioRisk
    basis: FactorBasis
    weights: Mapping[str, float]
    benchmark_weights: Mapping[str, float]
    valuation: ValuationNotes
    position_contributions: Mapping[str, tuple[Contribution, ...]] = field(
        default_factory=dict
    )
    factor_decompositions: Mapping[str, Decomposition] = field(default_factory=dict)
    group_decompositions: Mapping[str, Decomposition] = field(default_factory=dict)
    tails: Mapping[str, tuple[TailRisk, ...]] = field(default_factory=dict)
    load: LoadNotes | None = None
    covered_without_exposure: tuple[str, ...] = ()
    config: RiskConfig = DEFAULTS

    @property
    def total_risk(self) -> float:
        return self.risk.portfolio.total_risk

    @property
    def tracking_error(self) -> float:
        return self.risk.active.total_risk

    @property
    def is_exact(self) -> bool:
        """True when nothing anywhere was substituted, repaired, missing or ill-conditioned."""
        return not self.qualifications()

    def qualifications(self) -> tuple[str, ...]:
        """Everything a reader should know before trusting these numbers, in one list.

        Ordered from the outside in: which dates the run actually used, what the market data
        could not answer, what the book left out, then what the arithmetic itself could not
        resolve. De-duplicated, because the same repair is reported by every decomposition
        built on it and a reader who sees it twenty times reads it zero times.
        """
        notes: list[str] = []
        notes.extend(self.as_of.qualifications())
        if self.load is not None:
            notes.extend(self.load.qualifications())
        notes.extend(self.valuation.qualifications())
        if self.covered_without_exposure:
            shown = ", ".join(self.covered_without_exposure[:5])
            more = (
                f" and {len(self.covered_without_exposure) - 5} more"
                if len(self.covered_without_exposure) > 5
                else ""
            )
            notes.append(
                f"{len(self.covered_without_exposure)} position(s) are marked covered but the "
                f"model supplied no exposures for them, so they are carrying weight and "
                f"contributing no risk: {shown}{more}. Either the exposure date is wrong for "
                f"these names or they should be marked uncovered"
            )
        # The portfolio side carries the basis-level provenance -- a repaired covariance, a
        # proxied factor -- which is shared by all three sides, so it is reported once.
        notes.extend(self.risk.qualifications())
        # Conditioning is NOT shared: it depends on the exposure vector, and the active book
        # is where it bites hardest. A factor-neutral long-short book is the active book of
        # any index-hugging fund, and `PortfolioRisk.qualifications()` reports only the
        # portfolio side, so the other two are added here.
        for label, decomposition in (
            ("benchmark", self.risk.benchmark),
            ("active", self.risk.active),
        ):
            if not decomposition.is_well_conditioned:
                digits = decomposition.digits_intact
                surviving = (
                    f"about {digits:.1f} decimal digits survive"
                    if digits > 0.0
                    else "no digit of it is correct"
                )
                notes.append(
                    f"the {label} book's factor bets offset each other (amplification "
                    f"{decomposition.amplification:.2e}), so {surviving}. Report fewer "
                    f"significant figures for it, or raise psd_relative_floor"
                )
        return tuple(dict.fromkeys(notes))

    def describe(self) -> str:
        head = (
            f"{self.portfolio_name} vs {self.benchmark_name or '(no benchmark)'} "
            f"on {self.as_of.describe()}: total risk {self.total_risk:.4%}, "
            f"tracking error {self.tracking_error:.4%}"
        )
        notes = self.qualifications()
        if not notes:
            return head + " (nothing substituted)"
        return head + f" ({len(notes)} qualification(s))"


def positions_from(
    book: PortfolioValue,
    exposures: Mapping[str, ExposureVector],
    *,
    basis: ValueBasis = ValueBasis.EXPOSURE,
) -> tuple[tuple[Position, ...], tuple[str, ...]]:
    """Turn a valued book plus a model's exposures into risk positions.

    Returns the positions and the covered securities the model had no exposures for.

    Coverage comes from the holdings feed, never from an empty exposure vector: cash is covered
    and genuinely riskless, so inferring coverage from the absence of factor loadings would
    classify every cash position as unmodelled. A security marked covered with no exposures is
    a different thing -- a data gap -- and is returned so the result can say so.
    """
    held = book.weights(basis)
    bench = book.benchmark_weights(basis)
    missing: list[str] = []
    positions: list[Position] = []
    for value in book.positions:
        code = value.security_code
        exposure = exposures.get(code)
        if exposure is None:
            if value.covered and (held.get(code) or bench.get(code)):
                missing.append(code)
            exposure = ExposureVector.zero()
        positions.append(
            Position(
                security_code=code,
                exposure=exposure,
                weight=held.get(code, 0.0),
                benchmark_weight=bench.get(code, 0.0),
                covered=value.covered,
            )
        )
    return tuple(positions), tuple(missing)


def compute_risk(
    as_of: AsOf,
    book: PortfolioValue,
    exposures: Mapping[str, ExposureVector],
    covariance: FactorCovariance,
    *,
    tree: FactorTree | None = None,
    factor_names: Mapping[int, str] | None = None,
    tails: Sequence[TailSpec] = (),
    weight_basis: WeightBasis | None = None,
    value_basis: ValueBasis = ValueBasis.EXPOSURE,
    load: LoadNotes | None = None,
    proxies: Mapping[int, int] | None = None,
    config: RiskConfig | None = None,
) -> RiskResult:
    """Assemble one risk run from the pieces each layer produces.

    Takes already-loaded inputs rather than data sources: the adapters that build an `AsOf`,
    a `PortfolioValue` and a model's exposures from a database are separate and come later.
    What this fixes is the order of assembly and the consolidation of provenance, which is
    the part that does not depend on where the data came from.
    """
    config = config or DEFAULTS
    positions, missing = positions_from(book, exposures, basis=value_basis)

    portfolio = Portfolio.of(
        name=book.name,
        positions=positions,
        benchmark_name=book.benchmark_name,
        weight_basis=weight_basis or WeightBasis(config.weight_basis),
    )
    basis = active_basis(
        covariance,
        portfolio.exposure.exposures,
        portfolio.benchmark_exposure.exposures,
        proxies=proxies,
        config=config,
    )
    risk = portfolio_risk(portfolio, basis)

    sides = (
        ("portfolio", Level.PORTFOLIO, portfolio.exposure),
        ("benchmark", Level.BENCHMARK, portfolio.benchmark_exposure),
        ("active", Level.ACTIVE, portfolio.active_exposure),
    )
    names = dict(factor_names or {})
    position_contributions = {
        label: contributions(portfolio, basis, level) for label, level, _ in sides
    }
    factor_decompositions = {
        label: factor_decomposition(vector, basis, names=names)
        for label, _, vector in sides
    }
    group_decompositions = (
        {
            label: group_decomposition(vector, basis, tree, names=names)
            for label, _, vector in sides
        }
        if tree is not None
        else {}
    )
    decompositions = {
        "portfolio": risk.portfolio,
        "benchmark": risk.benchmark,
        "active": risk.active,
    }
    tail_results = {
        label: tuple(tail_risk(decompositions[label].total_risk, spec) for spec in tails)
        for label in decompositions
    } if tails else {}

    return RiskResult(
        as_of=as_of,
        portfolio_name=book.name,
        benchmark_name=book.benchmark_name,
        risk=risk,
        basis=basis,
        weights=book.weights(value_basis),
        benchmark_weights=book.benchmark_weights(value_basis),
        valuation=book.notes(),
        position_contributions=position_contributions,
        factor_decompositions=factor_decompositions,
        group_decompositions=group_decompositions,
        tails=tail_results,
        load=load,
        covered_without_exposure=missing,
        config=config,
    )
