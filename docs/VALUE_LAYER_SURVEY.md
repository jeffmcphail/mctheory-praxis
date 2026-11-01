# The value layer: what RAT does, and what the port keeps

*Mid-office market risk engine. Written 2026-11-01, from RAT's live source.*

The value layer produces the weights every risk number consumes, so getting it wrong moves
every figure downstream without breaking anything visibly. This records what the original
does, which parts are design worth preserving, and which are accidents.

## One quantity, four axes

RAT writes the value set out as a cross product of named fields: eighteen money fields on
`Position` (`marketValue`, `cleanMarketValue`, `exposureValue`, `cleanExposureValue`, `weight`,
`cleanWeight`, `exposureWeight`, then `benchmark*` and `active*` mirrors of each) and fourteen
more on `Portfolio`, including the `*IncNc` variants and the `Numeraire` accessors. Each has to
be kept in step with the others by hand, and adding an axis multiplies the set.

They are not thirty-two quantities. They are one quantity on four axes:

| axis | values | RAT's spelling |
|---|---|---|
| basis | market, exposure | `marketValue` vs `expValue` |
| accrual | dirty, clean | `marketValue` vs `cleanMarketValue` |
| coverage | covered, all | `marketValue` vs `MarketValueIncNc` |
| side | portfolio, benchmark, active | the `benchmark*` / `active*` mirrors |

`praxis.midoffice.value.valuation` has one per-unit formula and parameterises it, which is the
same move the risk package made with `cov(p, q)`. The currency axis collapses too: everything
is held in the numeraire, and RAT's `getMarketValueNumeraire` family exists to convert on the
way out.

## Why the exposure weight divides by market value

This is the most important thing in the layer and it is a design decision, not an artefact.

```csharp
// RiskData.cs:99
public double expWeight {
    get {
        if (this.portRiskData.RiskMarketValue != 0.0)
            return this.RiskExpValue / this.portRiskData.RiskMarketValue;
        return 0.0;
    }
}
```

An exposure value in the numerator over a **market** value in the denominator. The asymmetry is
the point: it makes the weight a leverage ratio. A future has a market value near zero -- it is
a margined contract -- and a notional in the millions, so dividing its notional by the book's
net asset value states how levered the book is through it. The weights then do not sum to one;
their sum is gross exposure as a multiple of NAV, which is the number a risk report should
show. Leaving the denominator as market value is what produces that, and changing it to total
exposure would throw the information away.

Exposure weight is used in place of market weight wherever the two differ, which in practice
means derivatives, futures being the common case.

**The alternative, for contrast.** BarraOne -- Barra's flagship engine -- uses market value
only. To make a future carry one, it models the position as a basket: the future plus cash in
the future's currency in the quantity of the future's exposure, with an offsetting amount of
the same cash as a separate position. Everything then has a market value and ordinary market
weights work. The costs are that security construction (building the basket) gets mixed into
position handling (the offset), and the synthetic legs show up in the results, so a future
appears as several rows that have to be read together. Carrying two value axes on one position
is the cleaner trade, and it is why `ValueBasis` exists.

## float64, deliberately

RAT uses `double` for every money quantity. There is no `decimal` anywhere in its live source
-- the only occurrences are commented-out casts of an FX forward's *maturity date* in
`FundGroup.cs` -- and it reconciled against portfolio managers' own systems for fifteen years
on that basis.

The port matches it, which reverses an earlier recommendation in this project to use `Decimal`
or integer minor units here. Two reasons:

* **Parity.** A port using `Decimal` where the original used `double` can never agree exactly,
  so the harness would need a tolerance on values and would stop being a tight check.
* **Precision does not bite.** Summing five thousand positions totalling a billion in float64
  carries about `sqrt(5000) * 2.2e-16 * 1e9`, roughly 1.6e-5 of a currency unit -- three orders
  below a cent.

Exact decimal arithmetic belongs where money is the record of account. This is a reconciliation
view, not a ledger.

## The `*IncNc` family is vestigial

`MarketValueIncNc`, `cleanMarketValueIncNc`, `numExpValueIncNc`, `cleanExpValueIncNc` and their
benchmark mirrors are assigned **equal** to their plain counterparts at every load path
(`Portfolio.cs:843`, `:910`, `:984`), and `refreshRiskValues()` (`PortfolioRiskData.cs:467`)
copies them back into the plain fields before each risk run. So the `IncNc` fields are a
load-time snapshot of the full book, and the plain fields are the working copy that an
exclusion would have reduced.

Nothing ever reduces them. The only `marketValue -=` in the file subtracts a *composite*
security's value, which is leg-unwinding to avoid double counting. The only
`ExcludeNotCovered` sites in the live source are `//ToDo:` comments reading "at extendedArgs
override here", and `includeNonCoveredSecurities = true` sits beside `includeAccruedInterest`
as a flag nothing acts on.

So **every value RAT ever produced is the all-inclusive figure.** Uncovered names stayed in the
weight denominator because nothing took them out -- the exclusion was planned and never built.
That explains a production detail from memory: uncovered names were kept in the denominator
because portfolio managers reconcile against their own systems, and the `//ToDo` marks where
the alternative would have gone.

Consequence for the port: `Coverage.COVERED` and the risk package's `WeightBasis.COVERED` are
both genuinely new behaviour rather than ports of anything, and value parity against RAT will
only ever exercise `Coverage.ALL`.

## Substitutions that change a weight

Three, all of which the port puts on the result through `ValuationNotes` rather than into a log
stream running beside it.

**`CASH_PLUG`** (`Portfolio.cs:826`). A synthetic cash position named "Unavailible Cash" (sic)
is inserted for the gap between the account's stated value and the sum of its positions. This
is the mechanism behind the reconciliation requirement: rather than explain a discrepancy, RAT
absorbs it into cash so the book's market value ties. The port keeps it as a real position --
cash is covered and riskless, so it enters the weight denominator and contributes no risk --
because a plug hidden in a total looks like a rounding difference, while a plug that is a
position can be seen in a holdings listing.

**The zero-market-value fallback** (`Portfolio.cs:835`).

```csharp
if (this.marketValue == 0.0) {
    this.marketValue = this.numExpValue;
    this.cleanMarketValue = this.cleanNumExpValue;
}
```

Unlogged. The behaviour is right -- a futures overlay has a market value near zero and a large
notional, and dividing by zero would end the run -- but it silently changes what every weight
is a fraction of, from net asset value to notional. Reported in the port.

**`FILL_MISSING_PRICES`** (`FundGroup.cs:251`). An unavailable price is zeroed, and logged at
Error level. Described at the time as the least bad of several options: the alternative was to
fail the whole run, and the cases were rare and almost always well under 1% of total weight.
Behaviour kept; the flag now travels on the result.

## The accrual selector, and its staleness hazard

`includeAccruedInterest` defaults to `true` (`RATEngine.cs:202`, parameter
`INCLUDE_ACCRUED_INTEREST_RISK`) and is resolved through memoised properties --
`RiskMarketValue`, `RiskExpValue`, `RiskBenchMarketValue` -- each caching on first read behind
a `double.NaN` sentinel:

```csharp
private double riskMarketValue = double.NaN;
public double RiskMarketValue {
    get {
        if (double.IsNaN(riskMarketValue))
            riskMarketValue = this.portfolio.includeAccruedInterest
                ? this.marketValue : this.cleanMarketValue;
        return riskMarketValue;
    }
}
```

The cache means the flag has to be set before anything reads a value, or the property serves
the choice that was in force earlier with no indication. In practice it is set at load, so it
holds. In the port `accrual` is a frozen field on `PortfolioValue` with a per-call override, so
there is nothing to go stale.

`NAV` (`PortfolioRiskData.cs:88`) is a separate hazard of the same kind: it reads the account
row's `Nav` column inside a `try` and falls back to `marketValue` on any `Exception`, so a null
or a type mismatch silently becomes the market value.

## Things the original encodes that are easy to get backwards

This table is about what a *reimplementation* gets wrong, not what RAT got wrong. Deliberate
departures by the port belong in RAT_PORT_SURVEY.md section 8 instead.

| | What it is | What you would assume |
|---|---|---|
| `expWeight` | exposure value / book **market** value, so the weight is leverage and the weights sum to gross exposure | exposure / total exposure, which sums to one and cannot express leverage. **Identical on a cash-equity book**, so no equity test detects the difference |
| `*IncNc` | a load-time snapshot of the full book, restored by `refreshRiskValues()` | a coverage-excluded variant. The exclusion was never built, so it equals the plain field always |
| zero market value | falls back to exposure value as the denominator | an error, or a zero weight |
| `RiskMarketValue` | a memoised selector over the dirty/clean pair | a stored value, or a live one |
| `exposure_price` for a cash equity | equals the market price | a separate input that must be supplied |
| `NAV` | the account row, falling back to market value on any exception | the account row |

## Status

Built: `PositionValue` (the four-axis per-unit formula, FX, leverage and pricing flags),
`PortfolioValue` (totals on every axis, the asymmetric weight, the denominator fallback,
`CASH_PLUG` insertion, `gross_exposure`), `ValuationNotes`. 28 tests in
`tests/midoffice/test_valuation.py`.

Not built: price and accrued-interest loading from a market-data source, FX rate sourcing, the
`CASH_PLUG` gap computation from an accounting feed, look-through expansion of index
securities, and value coverage in the parity harness.
