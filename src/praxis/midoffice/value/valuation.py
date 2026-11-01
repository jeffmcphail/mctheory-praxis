"""What a book is worth, on every axis a risk report needs it.

RAT writes this out as a cross product of named fields: eighteen money fields on `Position`
(`marketValue`, `cleanMarketValue`, `exposureValue`, `cleanExposureValue`, `weight`,
`cleanWeight`, `exposureWeight`, then the `benchmark*` and `active*` mirrors of each) and
fourteen more on `Portfolio` (the same again plus the `*IncNc` variants and the `Numeraire`
accessors).  Every one has to be kept in step with the others by hand, and adding an axis
multiplies the set.

They are not thirty-two quantities.  They are one quantity on four axes:

| axis | values | RAT's spelling |
|---|---|---|
| basis | market, exposure | `marketValue` vs `expValue` |
| accrual | dirty, clean | `marketValue` vs `cleanMarketValue` |
| coverage | covered, all | `marketValue` vs `MarketValueIncNc` |
| side | portfolio, benchmark, active | the `benchmark*` / `active*` mirrors |

So this module has one per-unit formula and parameterises it, which is the same move the risk
package made with `cov(p, q)`.  The currency axis collapses too: everything is held in the
numeraire, because a value in two currencies is two numbers that have to agree and RAT's
`getMarketValueNumeraire` family exists to convert on the way out.

**float64 throughout, deliberately.**  RAT used `double` for every money quantity -- there is
no `decimal` anywhere in its live source -- and reconciled against portfolio managers' own
systems for fifteen years on that basis.  Two reasons to match it rather than reach for exact
decimal arithmetic.  Parity: a port using `Decimal` where the original used `double` can never
agree exactly, so the harness would need a tolerance on values and stop being a tight check.
Precision: summing five thousand positions totalling a billion carries about
`sqrt(5000) * 2.2e-16 * 1e9`, roughly 1.6e-5 of a currency unit -- three orders below a cent.
Exact decimal arithmetic belongs where money is the record of account, which this is not; this
is a reconciliation view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping


class ValueBasis(Enum):
    """What the number measures.

    `MARKET` is what the position is worth; `EXPOSURE` is what it is exposed to.  They differ
    for anything with leverage: a future's market value is its variation margin while its
    exposure is the full notional, so the two can differ by orders of magnitude.  For a cash
    equity they are the same number, which is exactly why a bug in this distinction stays
    invisible until the first derivative arrives.
    """

    MARKET = "market"
    EXPOSURE = "exposure"


class Accrual(Enum):
    """Whether accrued interest is in the price. RAT's `includeAccruedInterest`, defaulting
    to `DIRTY` as it does (`RATEngine.cs:202`, parameter `INCLUDE_ACCRUED_INTEREST_RISK`).

    RAT resolves this through memoised properties -- `RiskMarketValue`, `RiskExpValue` -- that
    cache on first read behind a `double.NaN` sentinel. The cache is why the flag has to be
    set before anything reads a value: changing it afterwards serves the stale choice with no
    indication. Here it is a parameter instead, so there is nothing to stale.
    """

    DIRTY = "dirty"
    CLEAN = "clean"


class Coverage(Enum):
    """Whether names the risk model does not cover are in the total. RAT's `*IncNc` family.

    `ALL` is the figure a portfolio manager reconciles against: it is the whole book.
    `COVERED` is the subset the risk model can speak about. Reporting one and labelling it the
    other is how a risk tool loses its reader's trust, which is the reason both exist.
    """

    COVERED = "covered"
    ALL = "all"


@dataclass(frozen=True)
class PositionValue:
    """One security's economics, per unit, plus the quantity held on each side.

    Per-unit prices with quantities, rather than pre-computed values, because the eight value
    figures a position needs are all products of the same four primitives. Supplying the
    values directly would mean eight inputs that can disagree with each other.

    `exposure_price` defaults to the clean price, which is correct for anything unleveraged
    and makes the common case silent. A derivative supplies it: notional per unit, which for a
    future is contract size times index level and bears no relation to what the contract is
    worth.
    """

    security_code: str
    clean_price: float = 0.0
    accrued_interest: float = 0.0
    exposure_price: float | None = None
    fx_rate: float = 1.0
    quantity: float = 0.0
    benchmark_quantity: float = 0.0
    covered: bool = True

    def __post_init__(self) -> None:
        if self.fx_rate <= 0.0:
            raise ValueError(
                f"{self.security_code}: fx_rate must be positive, got {self.fx_rate!r}; a "
                f"zero or negative rate would silently zero or flip the position's value"
            )

    @property
    def resolved_exposure_price(self) -> float:
        """Notional per unit, defaulting to the market price for anything unleveraged."""
        return self.clean_price if self.exposure_price is None else self.exposure_price

    @property
    def is_leveraged(self) -> bool:
        return self.resolved_exposure_price != self.clean_price

    def unit_value(self, basis: ValueBasis, accrual: Accrual) -> float:
        """Value of one unit in the numeraire, on the given axes. The whole of the formula."""
        price = (
            self.clean_price
            if basis is ValueBasis.MARKET
            else self.resolved_exposure_price
        )
        if accrual is Accrual.DIRTY:
            price += self.accrued_interest
        return price * self.fx_rate

    def value(self, basis: ValueBasis, accrual: Accrual) -> float:
        return self.quantity * self.unit_value(basis, accrual)

    def benchmark_value(self, basis: ValueBasis, accrual: Accrual) -> float:
        return self.benchmark_quantity * self.unit_value(basis, accrual)

    def active_value(self, basis: ValueBasis, accrual: Accrual) -> float:
        return (self.quantity - self.benchmark_quantity) * self.unit_value(basis, accrual)

    @property
    def is_priced(self) -> bool:
        """False for a name whose price came through as zero.

        RAT's `FILL_MISSING_PRICES` zeroes an unavailable price and logs it at Error level
        (`FundGroup.cs:251`), which Jeff described as the least bad option of several: the
        alternative was to fail the whole run, and the cases were rare and almost always well
        under 1% of total weight. The behaviour is kept; what changes is that the flag travels
        on the result rather than only into a log stream nobody reads next to the numbers.
        """
        return self.clean_price != 0.0 or self.accrued_interest != 0.0


CASH_PLUG = "CASH_PLUG"
"""Code for the synthetic cash position that forces a book's value to tie.

RAT's name for it is "Unavailible Cash" (sic), inserted at `Portfolio.cs:826`.
"""


@dataclass(frozen=True)
class ValuationNotes:
    """What was substituted, plugged or left out to arrive at these numbers.

    The same discipline as `PSDRepair` and `CoverageSummary`: RAT records most of this too,
    but into a log stream running beside the results, so reading an old number told you what
    it was and not what it was computed over.  Every field here is something that changes a
    weight, and therefore every risk number downstream.
    """

    n_positions: int = 0
    n_uncovered: int = 0
    n_leveraged: int = 0
    uncovered_market_value: float = 0.0
    unpriced: tuple[str, ...] = ()
    cash_plug: float = 0.0
    market_total_fell_back_to_exposure: bool = False
    stated_nav: float | None = None
    nav_gap: float | None = None

    @property
    def is_clean(self) -> bool:
        return not self.qualifications()

    def qualifications(self) -> tuple[str, ...]:
        notes: list[str] = []
        if self.cash_plug:
            notes.append(
                f"a synthetic cash position of {self.cash_plug:,.2f} was added to make the "
                f"book's value tie to its stated NAV; it carries no factor exposure and no "
                f"specific risk, but it is in the weight denominator"
            )
        if self.unpriced:
            shown = ", ".join(self.unpriced[:5])
            more = f" and {len(self.unpriced) - 5} more" if len(self.unpriced) > 5 else ""
            notes.append(
                f"{len(self.unpriced)} position(s) came through with no price and are "
                f"valued at zero: {shown}{more}. They sit in the position count and "
                f"contribute nothing to any total"
            )
        if self.market_total_fell_back_to_exposure:
            notes.append(
                "the book's market value is zero, so exposure value was used as the weight "
                "denominator instead. This is normal for a derivatives overlay and makes "
                "every weight a fraction of notional rather than of net asset value"
            )
        if self.n_leveraged:
            notes.append(
                f"{self.n_leveraged} position(s) have an exposure value that differs from "
                f"their market value, so market-weighted and exposure-weighted figures will "
                f"not agree and the exposure weights will not sum to one"
            )
        if self.n_uncovered:
            notes.append(
                f"{self.n_uncovered} position(s) carrying {self.uncovered_market_value:,.2f} "
                f"of market value are not covered by the risk model"
            )
        if self.nav_gap:
            notes.append(
                f"the book's computed market value differs from its stated NAV by "
                f"{self.nav_gap:,.2f}"
            )
        return tuple(notes)


@dataclass(frozen=True)
class PortfolioValue:
    """A book's positions and the totals and weights they imply.

    `accrual` is a field rather than a mutable flag, and that is the whole of the fix for
    RAT's staleness hazard: `RiskMarketValue` and `RiskExpValue` memoise the dirty/clean
    choice on first read behind a `double.NaN` sentinel (`RiskData.cs:110`), so the flag has
    to be set before anything reads a value or the cache serves the choice that was in force
    earlier. A frozen field cannot go stale, and methods still take an override for a report
    that shows both.
    """

    name: str
    positions: tuple[PositionValue, ...] = ()
    benchmark_name: str = ""
    accrual: Accrual = Accrual.DIRTY
    stated_nav: float | None = None

    def __post_init__(self) -> None:
        positions = tuple(self.positions)
        seen: set[str] = set()
        for p in positions:
            if p.security_code in seen:
                raise ValueError(
                    f"duplicate security {p.security_code!r} in book {self.name!r}; "
                    f"aggregate lots before valuing the book"
                )
            seen.add(p.security_code)
        object.__setattr__(self, "positions", positions)

    # -- construction ------------------------------------------------------------------

    @classmethod
    def with_cash_plug(
        cls,
        name: str,
        positions: Iterable[PositionValue],
        *,
        stated_nav: float,
        benchmark_name: str = "",
        accrual: Accrual = Accrual.DIRTY,
        tolerance: float = 0.005,
    ) -> "PortfolioValue":
        """Add a synthetic cash position for the gap between stated NAV and the positions.

        This is RAT's `CASH_PLUG`, and it is the mechanism behind the reconciliation
        requirement: portfolio managers check a risk tool's market value against their own
        system, and a tool whose value does not tie is a tool they will not use. Rather than
        explain a discrepancy, RAT absorbs it into cash.

        The plug is a real position here rather than an adjustment to a total, which is how
        RAT does it too, and it matters: cash is covered and riskless, so it enters the weight
        denominator and contributes no risk. A plug hidden in the denominator would look like
        a rounding difference; a plug that is a position can be seen in a holdings listing.

        `tolerance` suppresses a plug below half a cent, which would be float noise rather
        than a reconciliation gap.
        """
        held = tuple(positions)
        provisional = cls(
            name=name, positions=held, benchmark_name=benchmark_name, accrual=accrual
        )
        gap = stated_nav - provisional.total(ValueBasis.MARKET)
        if abs(gap) <= tolerance:
            return cls(
                name=name,
                positions=held,
                benchmark_name=benchmark_name,
                accrual=accrual,
                stated_nav=stated_nav,
            )
        plug = PositionValue(
            security_code=CASH_PLUG, clean_price=1.0, quantity=gap, covered=True
        )
        return cls(
            name=name,
            positions=held + (plug,),
            benchmark_name=benchmark_name,
            accrual=accrual,
            stated_nav=stated_nav,
        )

    # -- selection ---------------------------------------------------------------------

    def _accrual(self, accrual: Accrual | None) -> Accrual:
        return self.accrual if accrual is None else accrual

    def _selected(self, coverage: Coverage) -> tuple[PositionValue, ...]:
        if coverage is Coverage.ALL:
            return self.positions
        return tuple(p for p in self.positions if p.covered)

    # -- totals ------------------------------------------------------------------------

    def total(
        self,
        basis: ValueBasis = ValueBasis.MARKET,
        *,
        coverage: Coverage = Coverage.ALL,
        accrual: Accrual | None = None,
    ) -> float:
        a = self._accrual(accrual)
        return sum(p.value(basis, a) for p in self._selected(coverage))

    def benchmark_total(
        self,
        basis: ValueBasis = ValueBasis.MARKET,
        *,
        coverage: Coverage = Coverage.ALL,
        accrual: Accrual | None = None,
    ) -> float:
        a = self._accrual(accrual)
        return sum(p.benchmark_value(basis, a) for p in self._selected(coverage))

    # -- the weight denominator --------------------------------------------------------

    def denominator(
        self, *, coverage: Coverage = Coverage.ALL, accrual: Accrual | None = None
    ) -> float:
        """The book's market value, falling back to exposure value when it is zero.

        RAT: `if (this.marketValue == 0.0) { this.marketValue = this.numExpValue; }`
        (`Portfolio.cs:835`), with no log line. The fallback is right -- a futures overlay has
        a market value near zero and a large notional, and dividing by zero would end the run
        -- but it silently changes what every weight is a fraction of. Here it is reported
        through `notes()`.
        """
        market = self.total(ValueBasis.MARKET, coverage=coverage, accrual=accrual)
        if market != 0.0:
            return market
        return self.total(ValueBasis.EXPOSURE, coverage=coverage, accrual=accrual)

    # -- weights -----------------------------------------------------------------------

    def weights(
        self,
        basis: ValueBasis = ValueBasis.EXPOSURE,
        *,
        coverage: Coverage = Coverage.ALL,
        accrual: Accrual | None = None,
    ) -> dict[str, float]:
        """Each position's value on `basis`, over the book's MARKET value.

        **The numerator and denominator are deliberately on different bases.** This is RAT's
        `expWeight` (`RiskData.cs:99`):

            expWeight = RiskExpValue / portRiskData.RiskMarketValue

        An exposure value over a market value, which makes the weight a leverage ratio. A
        futures position has a market value near zero and a notional in the millions, so its
        exposure weight is large and the weights do not sum to one; their sum is the book's
        gross exposure as a multiple of its net asset value, which is the number a risk report
        should show.

        The alternative construction is BarraOne's: model the future as a basket of the future
        plus cash in the future's currency at the notional quantity, then add an offsetting
        cash position, so that everything carries a market value and ordinary market weights
        work. That keeps one weight definition at the cost of mixing security construction
        into position handling, and the synthetic legs then appear in the output. Carrying two
        value axes on one position is the cleaner trade, and it is the reason `ValueBasis`
        exists at all.

        Implementing this from the name alone gives `exposure / total_exposure`, which is
        wrong and which no cash-equity test can detect, because there market and exposure
        value are the same number.
        """
        a = self._accrual(accrual)
        denominator = self.denominator(coverage=coverage, accrual=a)
        selected = self._selected(coverage)
        if denominator == 0.0:
            return {p.security_code: 0.0 for p in selected}
        return {
            p.security_code: p.value(basis, a) / denominator for p in selected
        }

    def benchmark_weights(
        self,
        basis: ValueBasis = ValueBasis.EXPOSURE,
        *,
        coverage: Coverage = Coverage.ALL,
        accrual: Accrual | None = None,
    ) -> dict[str, float]:
        """The benchmark side, over the benchmark's own market value.

        Each side is divided by its own denominator, which is what makes the difference of
        the two a meaningful active weight.
        """
        a = self._accrual(accrual)
        selected = self._selected(coverage)
        market = self.benchmark_total(ValueBasis.MARKET, coverage=coverage, accrual=a)
        denominator = (
            market
            if market != 0.0
            else self.benchmark_total(ValueBasis.EXPOSURE, coverage=coverage, accrual=a)
        )
        if denominator == 0.0:
            return {p.security_code: 0.0 for p in selected}
        return {
            p.security_code: p.benchmark_value(basis, a) / denominator for p in selected
        }

    def active_weights(
        self,
        basis: ValueBasis = ValueBasis.EXPOSURE,
        *,
        coverage: Coverage = Coverage.ALL,
        accrual: Accrual | None = None,
    ) -> dict[str, float]:
        """Portfolio weight less benchmark weight, position by position."""
        held = self.weights(basis, coverage=coverage, accrual=accrual)
        bench = self.benchmark_weights(basis, coverage=coverage, accrual=accrual)
        return {code: held[code] - bench.get(code, 0.0) for code in held}

    @property
    def gross_exposure(self) -> float:
        """Sum of the exposure weights: the book's notional as a multiple of its value.

        1.0 for a fully invested cash book. Above 1.0 for anything with leverage, which is
        the information the asymmetric weight exists to carry.
        """
        return sum(self.weights(ValueBasis.EXPOSURE).values())

    # -- provenance --------------------------------------------------------------------

    def notes(self, *, accrual: Accrual | None = None) -> ValuationNotes:
        a = self._accrual(accrual)
        uncovered = [p for p in self.positions if not p.covered]
        plug = next((p for p in self.positions if p.security_code == CASH_PLUG), None)
        market = self.total(ValueBasis.MARKET, accrual=a)
        gap = (
            None
            if self.stated_nav is None
            else (self.stated_nav - market if abs(self.stated_nav - market) > 0.005 else 0.0)
        )
        return ValuationNotes(
            n_positions=len(self.positions),
            n_uncovered=len(uncovered),
            n_leveraged=sum(1 for p in self.positions if p.is_leveraged),
            uncovered_market_value=sum(p.value(ValueBasis.MARKET, a) for p in uncovered),
            unpriced=tuple(p.security_code for p in self.positions if not p.is_priced),
            cash_plug=0.0 if plug is None else plug.value(ValueBasis.MARKET, a),
            market_total_fell_back_to_exposure=(market == 0.0 and len(self.positions) > 0),
            stated_nav=self.stated_nav,
            nav_gap=gap,
        )
