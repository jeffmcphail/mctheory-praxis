"""Market data, asset-class valuation rules, and the loader that turns holdings into values.

A holdings feed gives quantities and identifiers.  Market data gives prices, accrued interest
and exchange rates.  Turning the two into the per-unit numeraire values that
`PositionValue` carries is this module's whole job, and the interesting part is that the
formula depends on the asset class.

RAT expresses that through two virtual accessors on `Security` -- `getPriceNumeraire` for a
unit's market value and `getExposureNumeraire` for a unit's exposure value -- overridden per
security type, with a `priceMultiplier` that is `1` on the base class and `.01` on the
fixed-income family.  The authoritative set, read from the code rather than from the
documentation comment:

| class | price multiplier | par amount | unit market value | unit exposure value |
|---|---|---|---|---|
| equity, cash, index, composite, CDS, FX swap | 1 | no | price * fx | same |
| fixed income | .01 | yes | price * .01 * par * fx | same |
| interest rate swap, index CDS | .01 | no | price * .01 * fx | same |
| **future** | .01 | yes | **0** | price * .01 * par * fx |

Two things worth stating plainly.

**Market and exposure value differ only for futures.**  Among the active asset classes every
other one has `getExposureNumeraire` equal to `getPriceNumeraire`.  That is why
`PositionValue.exposure_price` defaults to the market price: the default is right for every
class but one, and a future is expressed by giving it a market price of zero and an exposure
price of its notional.  `FIFutureSecurity.getPriceNumeraire` is literally `return 0.0;`.

**The documentation comment in `RATEngine.cs` is stale.**  It lists a formula per asset class
and disagrees with the code in two rows: it omits the `.01` for interest rate swaps, which
`IRSWPSecurity` does override, and asserts a `.01` for credit default swaps, which
`CDSSecurity` does not override and which therefore inherit `1`.  The table reads as
authoritative -- real asset class codes, real database column headers, a formula for every row
-- which is exactly why it is worth saying that the overrides are the acting layer and the
comment only describes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

from ..asof import ResolvedDate
from .valuation import Accrual, PortfolioValue, PositionValue

DEFAULT_SOURCE = "DEFAULT"
"""Stands in for RAT's `defaultPriceSource` parameter when a feed names no source."""

PAR_BASIS_FLOOR = 2.0
"""Below this, a price quoted as a percent of par is more likely a fraction than a quote.

A bond quoted per 100 trades between roughly 20 and 150; one quoted as a fraction sits near
1.0. The two bands overlap only in deep distress, so the floor is a flag rather than an error:
a single name below it is a distressed bond, while most of a feed below it is a basis
mismatch. `LoadNotes` reports the count against the total so the two can be told apart.

This exists because the per-100 normalisation lived in RAT's extraction query rather than in
its code, so nothing in the engine stated the basis it was relying on.
"""


class MissingPrice(LookupError):
    """No price for a security from any permitted source.

    RAT raises the equivalent from `MarketData.getDirtyPrice`, and `FundGroup`'s
    `FILL_MISSING_PRICES` catches it, values the position at zero and logs at Error level.
    """


class MissingRate(LookupError):
    """No exchange rate, in either direction. RAT raises here too, and nothing catches it."""


class AssetClass(Enum):
    """RAT's active asset classes. The inactive ones carry no valuation formula at all."""

    EQUITY = "EQ"
    FIXED_INCOME = "FI"
    CASH = "CA"
    FUTURE = "FU"
    INDEX = "IND"
    COMPOSITE = "CP"
    CREDIT_DEFAULT_SWAP = "CDSWP"
    INDEX_CDS = "CDX"
    INTEREST_RATE_SWAP = "IRSWP"
    FX_SWAP = "FXSWP"


@dataclass(frozen=True)
class ValuationRule:
    """How one asset class turns a quoted price into a per-unit numeraire value."""

    asset_class: AssetClass
    price_multiplier: float = 1.0
    """The default when a quote does not state its own. What RAT hardcoded per class."""
    requires_par_amount: bool = False
    """Whether this class must carry a real par amount from the security master.

    A validation requirement, not a formula switch: the formula always applies `par_amount`,
    which defaults to 1.0 and is therefore inert for a class that has no par. The flag exists
    so that a bond arriving with no par is refused rather than valued at a millionth of itself.
    """
    market_value_is_zero: bool = False
    looks_through: bool = False
    expects_accrued: bool = False
    """Whether an absent accrued-interest record is worth reporting for this class.

    A data-quality hint, not a formula difference: RAT adds accrued whenever the feed carries
    it, for every class, so absence changes no arithmetic. It is set only for fixed income,
    where a bond with no accrued record means either a feed gap or a security that is not the
    bond it appears to be. A future, an equity and a cash position all legitimately have none,
    and flagging those would bury the one case that matters.
    """

    @property
    def quoted_as_percent_of_par(self) -> bool:
        return self.price_multiplier != 1.0


_RULES: dict[AssetClass, ValuationRule] = {
    AssetClass.EQUITY: ValuationRule(AssetClass.EQUITY),
    AssetClass.CASH: ValuationRule(AssetClass.CASH),
    AssetClass.INDEX: ValuationRule(AssetClass.INDEX, looks_through=True),
    AssetClass.COMPOSITE: ValuationRule(AssetClass.COMPOSITE, looks_through=True),
    AssetClass.CREDIT_DEFAULT_SWAP: ValuationRule(
        AssetClass.CREDIT_DEFAULT_SWAP, looks_through=True
    ),
    AssetClass.FX_SWAP: ValuationRule(AssetClass.FX_SWAP, looks_through=True),
    AssetClass.FIXED_INCOME: ValuationRule(
        AssetClass.FIXED_INCOME, price_multiplier=0.01, requires_par_amount=True,
        expects_accrued=True,
    ),
    AssetClass.INTEREST_RATE_SWAP: ValuationRule(
        AssetClass.INTEREST_RATE_SWAP, price_multiplier=0.01, looks_through=True
    ),
    AssetClass.INDEX_CDS: ValuationRule(
        AssetClass.INDEX_CDS, price_multiplier=0.01, looks_through=True
    ),
    AssetClass.FUTURE: ValuationRule(
        AssetClass.FUTURE,
        price_multiplier=0.01,
        requires_par_amount=True,
        market_value_is_zero=True,
    ),
}


def rule_for(asset_class: AssetClass) -> ValuationRule:
    return _RULES[asset_class]


@dataclass(frozen=True)
class Quote:
    """One security's price from one source, in the currency the source quoted it in.

    `currency` belongs to the quote rather than to the security, which is RAT's arrangement:
    `getDirtyPrice` converts using `priceRecord.currency`, not the security's trade currency.
    The two are normally the same and the distinction matters for a security quoted on a
    foreign exchange.

    `accrued_interest` is `None` when the feed carried no accrued record, which RAT
    distinguishes from zero: `accruedInterest.ContainsKey(priceKey)` decides whether anything
    is added. Dirty and clean are then the same number, which is correct for an equity and
    suspicious for a bond.
    """

    security_code: str
    price: float
    """The quoted figure, on the basis the feed delivers it.

    **This is half of a two-part convention and the half this code does not control.** The
    `.01` multiplier on the fixed-income classes assumes the price arrives quoted per 100 of
    par -- 98.5, not 0.985. In RAT that normalisation was done in the extraction query, drawing
    on a security view rather than in the C# at all, so the multiplier and the query were two
    halves of one agreement with nothing in either to state it.

    A replacement feed that delivers a fraction instead is a hundredfold error with no symptom
    except implausible risk. `value_book` therefore checks the basis at ingestion -- see
    `PAR_BASIS_FLOOR` -- and the adapter for a real source has to recover the query's
    normalisation along with the data.
    """
    accrued_interest: float | None = None
    currency: str = ""
    source: str = DEFAULT_SOURCE
    price_multiplier: float | None = None
    """The factor converting the quoted figure to a per-unit price, when the feed states it.

    `None` is not the same as `1.0`, and the distinction is the point. `None` means the feed
    said nothing, so the asset class's RAT default applies -- which preserves parity, since
    that is what RAT hardcoded. `1.0` means the feed has stated that this price is already per
    unit, which is what a pre-normalising query delivers.

    Carrying it here rather than deriving it from the asset class is deliberate: a quote
    convention belongs to the quote, not to the instrument type. Hardcoding `.01` for fixed
    income assumes every bond in every feed is quoted per 100, and double-counts against a feed
    that has already normalised -- precisely the failure that has no symptom but a hundredfold
    error in the risk.
    """

    @property
    def has_accrued(self) -> bool:
        return self.accrued_interest is not None

    def quoted(self, accrual: Accrual) -> float:
        """The quoted figure, with accrued added when asked for and available."""
        if accrual is Accrual.DIRTY and self.accrued_interest is not None:
            return self.price + self.accrued_interest
        return self.price


@dataclass(frozen=True)
class MarketData:
    """Prices and exchange rates as of one market date.

    Keyed by `(security_code, source)`, as RAT keys its price hashtable on
    `securityCode + "|" + source`. Exchange rates are held one way round and inverted on
    demand.
    """

    as_of: ResolvedDate | None = None
    quotes: Mapping[tuple[str, str], Quote] = field(default_factory=dict)
    rates: Mapping[tuple[str, str], float] = field(default_factory=dict)
    numeraire: str = "USD"
    default_source: str = DEFAULT_SOURCE

    @classmethod
    def of(
        cls,
        quotes: Iterable[Quote],
        *,
        rates: Mapping[tuple[str, str], float] | None = None,
        numeraire: str = "USD",
        as_of: ResolvedDate | None = None,
        default_source: str = DEFAULT_SOURCE,
    ) -> "MarketData":
        indexed: dict[tuple[str, str], Quote] = {}
        for q in quotes:
            key = (q.security_code, q.source)
            if key in indexed:
                raise ValueError(
                    f"two quotes for {q.security_code!r} from source {q.source!r}; a source "
                    f"gives one price per security per date"
                )
            indexed[key] = q
        return cls(
            as_of=as_of,
            quotes=indexed,
            rates=dict(rates or {}),
            numeraire=numeraire,
            default_source=default_source,
        )

    # -- exchange rates ----------------------------------------------------------------

    def rate(self, numerator: str, denominator: str) -> float:
        """Units of `numerator` per unit of `denominator`, inverting if only the reverse is held.

        RAT's `getExchangeRate`, including the guard that refuses to invert a zero rather than
        dividing by it. A missing rate raises in both.
        """
        if numerator == denominator:
            return 1.0
        direct = self.rates.get((numerator, denominator))
        if direct is not None:
            return direct
        inverse = self.rates.get((denominator, numerator))
        if inverse:  # non-zero: inverting zero would be a division by zero
            return 1.0 / inverse
        raise MissingRate(
            f"no {numerator}/{denominator} exchange rate, in either direction"
            + (f" (the {denominator}/{numerator} rate is zero)" if inverse == 0.0 else "")
        )

    def rate_to_numeraire(self, currency: str) -> float:
        if not currency:
            return 1.0
        return self.rate(self.numeraire, currency)

    # -- prices ------------------------------------------------------------------------

    def quote(self, security_code: str, sources: Sequence[str] | None = None) -> Quote:
        """The first quote found across `sources` in order, then the default source.

        The ordered fallback is RAT's `getDirtyPrice(securityCode, string[] priceSources)`,
        which is how a primary and a secondary price provider are expressed. Which source
        answered is on the returned quote, because two funds priced from different providers
        produce different risk and the number should say so.
        """
        for source in list(sources or ()) + [self.default_source]:
            found = self.quotes.get((security_code, source))
            if found is not None:
                return found
        tried = list(sources or ()) + [self.default_source]
        raise MissingPrice(
            f"no price for security {security_code!r} from source(s) {tried}"
        )


@dataclass(frozen=True)
class Holding:
    """What a holdings feed gives for one security, before any market data is applied."""

    security_code: str
    asset_class: AssetClass = AssetClass.EQUITY
    quantity: float = 0.0
    benchmark_quantity: float = 0.0
    par_amount: float = 1.0
    covered: bool = True
    price_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.par_amount <= 0.0:
            raise ValueError(
                f"{self.security_code}: par_amount must be positive, got "
                f"{self.par_amount!r}; it multiplies the price, so a zero would value the "
                f"position at nothing without any price being missing"
            )


@dataclass(frozen=True)
class LoadNotes:
    """What the market data could not answer, and where a substitution was made."""

    priced: int = 0
    unpriced: tuple[str, ...] = ()
    fallback_sources: Mapping[str, str] = field(default_factory=dict)
    missing_accrued: tuple[str, ...] = ()
    par_basis_suspects: tuple[str, ...] = ()
    par_quoted: int = 0
    inferred_multiplier: tuple[str, ...] = ()

    def qualifications(self) -> tuple[str, ...]:
        notes: list[str] = []
        if self.unpriced:
            shown = ", ".join(self.unpriced[:5])
            more = f" and {len(self.unpriced) - 5} more" if len(self.unpriced) > 5 else ""
            notes.append(
                f"{len(self.unpriced)} security(ies) had no price from any permitted source "
                f"and are valued at zero: {shown}{more}"
            )
        if self.fallback_sources:
            pairs = ", ".join(
                f"{code} from {source}" for code, source in list(self.fallback_sources.items())[:5]
            )
            notes.append(
                f"{len(self.fallback_sources)} security(ies) were priced from a fallback "
                f"provider rather than the primary one: {pairs}"
            )
        if self.par_basis_suspects:
            fraction = len(self.par_basis_suspects) / max(self.par_quoted, 1)
            shown = ", ".join(self.par_basis_suspects[:5])
            if fraction > 0.5:
                notes.append(
                    f"{len(self.par_basis_suspects)} of {self.par_quoted} security(ies) "
                    f"quoted as a percent of par priced below {PAR_BASIS_FLOOR:g}, which "
                    f"across most of a feed means the prices arrived as fractions rather "
                    f"than per 100 of par. Every one of those values is a hundredfold too "
                    f"small. Check the extraction query's normalisation: {shown}"
                )
            else:
                notes.append(
                    f"{len(self.par_basis_suspects)} of {self.par_quoted} security(ies) "
                    f"quoted as a percent of par priced below {PAR_BASIS_FLOOR:g}; at this "
                    f"proportion that reads as genuine distress rather than a basis "
                    f"mismatch: {shown}"
                )
        if self.inferred_multiplier:
            notes.append(
                f"{len(self.inferred_multiplier)} security(ies) carried no price multiplier, "
                f"so the asset class default was assumed. That is what RAT hardcoded, but the "
                f"feed has not confirmed its own basis -- a source that pre-normalises prices "
                f"makes every one of these values a hundredfold too large"
            )
        if self.missing_accrued:
            shown = ", ".join(self.missing_accrued[:5])
            notes.append(
                f"{len(self.missing_accrued)} fixed-income security(ies) carried no "
                f"accrued interest record, so their dirty and clean values are identical: "
                f"{shown}"
            )
        return tuple(notes)


def value_holding(
    holding: Holding, market: MarketData, *, fill_missing_prices: bool = True
) -> tuple[PositionValue, str | None, bool]:
    """Turn one holding into a `PositionValue`, in the numeraire.

    Returns the position, the source that priced it (`None` when nothing did), and whether an
    expected accrued-interest record was absent.

    Cash is the one class that needs no price: RAT values it as `fx_rate * position`, so the
    quantity *is* the amount. A cash position therefore never counts as unpriced.
    """
    rule = rule_for(holding.asset_class)

    if holding.asset_class is AssetClass.CASH:
        return (
            PositionValue(
                security_code=holding.security_code,
                clean_price=1.0,
                quantity=holding.quantity,
                benchmark_quantity=holding.benchmark_quantity,
                covered=holding.covered,
            ),
            None,
            False,
        )

    try:
        quote = market.quote(holding.security_code, holding.price_sources)
    except MissingPrice:
        if not fill_missing_prices:
            raise
        # RAT's FILL_MISSING_PRICES: value at zero, flag it, do not fail the run.
        return (
            PositionValue(
                security_code=holding.security_code,
                clean_price=0.0,
                quantity=holding.quantity,
                benchmark_quantity=holding.benchmark_quantity,
                covered=holding.covered,
            ),
            None,
            False,
        )

    fx = market.rate_to_numeraire(quote.currency)
    multiplier = (
        rule.price_multiplier if quote.price_multiplier is None else quote.price_multiplier
    )
    scale = multiplier * holding.par_amount

    exposure_unit = quote.price * scale
    market_unit = 0.0 if rule.market_value_is_zero else exposure_unit
    accrued_unit = (
        (quote.accrued_interest or 0.0) * scale if quote.has_accrued else 0.0
    )

    position = PositionValue(
        security_code=holding.security_code,
        clean_price=market_unit,
        accrued_interest=accrued_unit if not rule.market_value_is_zero else 0.0,
        exposure_price=exposure_unit if rule.market_value_is_zero else None,
        fx_rate=fx,
        quantity=holding.quantity,
        benchmark_quantity=holding.benchmark_quantity,
        covered=holding.covered,
    )
    expected_accrued = rule.expects_accrued and not quote.has_accrued
    return position, quote.source, expected_accrued


def value_book(
    name: str,
    holdings: Iterable[Holding],
    market: MarketData,
    *,
    benchmark_name: str = "",
    accrual: Accrual = Accrual.DIRTY,
    stated_nav: float | None = None,
    fill_missing_prices: bool = True,
) -> tuple[PortfolioValue, LoadNotes]:
    """Value a whole book, returning it with a record of what the market data could not answer."""
    positions: list[PositionValue] = []
    unpriced: list[str] = []
    fallbacks: dict[str, str] = {}
    missing_accrued: list[str] = []
    par_suspects: list[str] = []
    inferred_multiplier: list[str] = []
    par_quoted = 0
    priced = 0

    for holding in holdings:
        position, source, expected_accrued = value_holding(
            holding, market, fill_missing_prices=fill_missing_prices
        )
        positions.append(position)
        if holding.asset_class is AssetClass.CASH:
            priced += 1
        elif source is None:
            unpriced.append(holding.security_code)
        else:
            priced += 1
            primary = (holding.price_sources or (market.default_source,))[0]
            if source != primary:
                fallbacks[holding.security_code] = source
        if expected_accrued:
            missing_accrued.append(holding.security_code)
        # Basis check, at ingestion rather than downstream. Only meaningful for a class whose
        # price is a percent of par, and only when a price was actually found.
        rule = rule_for(holding.asset_class)
        if source is not None:
            quote = market.quote(holding.security_code, holding.price_sources)
            inferred = quote.price_multiplier is None
            effective = rule.price_multiplier if inferred else quote.price_multiplier
            if effective != 1.0:
                par_quoted += 1
                if 0.0 < quote.price < PAR_BASIS_FLOOR:
                    par_suspects.append(holding.security_code)
            if inferred and rule.quoted_as_percent_of_par:
                inferred_multiplier.append(holding.security_code)

    notes = LoadNotes(
        priced=priced,
        unpriced=tuple(unpriced),
        fallback_sources=fallbacks,
        missing_accrued=tuple(missing_accrued),
        par_basis_suspects=tuple(par_suspects),
        par_quoted=par_quoted,
        inferred_multiplier=tuple(inferred_multiplier),
    )
    if stated_nav is None:
        book = PortfolioValue(
            name=name, positions=tuple(positions), benchmark_name=benchmark_name,
            accrual=accrual,
        )
    else:
        book = PortfolioValue.with_cash_plug(
            name, positions, stated_nav=stated_nav, benchmark_name=benchmark_name,
            accrual=accrual,
        )
    return book, notes
