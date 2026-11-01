# The engine: four dates in, one result with its provenance

*Mid-office market risk engine. Written 2026-11-01.*

## What this layer owns

Everything below `praxis.midoffice.engine` computes a part. `asof` resolves the four dates.
`value` turns holdings and market data into weights. `risk` turns weights and exposures into
numbers. The engine owns two things neither of them can: the **order of assembly**, and the
**consolidation of provenance**.

```
compute_risk(as_of, book, exposures, covariance, *, tree, factor_names, tails,
             weight_basis, value_basis, load, proxies, config) -> RiskResult
```

It takes already-loaded inputs rather than data sources. The adapters that build an `AsOf`, a
`PortfolioValue` and a model's exposures from a database are separate and come later; what is
fixed here is the part that does not depend on where the data came from.

## The seam

`value` never imports `risk`. `risk` never imports `value`. They meet here, at the weights.
`asof` sits above both, because the model and exposure dates are risk-model concerns while the
market and security dates are value concerns, and putting it in either package would make one
import the other.

That is why `asof.py` is at `src/praxis/midoffice/` rather than under either.

## Weights come from the exposure basis

`positions_from` takes `ValueBasis.EXPOSURE`, not `MARKET`. This is RAT's `expWeight`
(`RiskData.cs:99`): an exposure value over the book's **market** value, which makes a weight a
leverage ratio rather than a share of the book.

Two reasons it has to be that way round. A futures overlay has a market value of zero by
formula -- `FIFutureSecurity.getPriceNumeraire` is `return 0.0;` -- so market weights would be
undefined for it. And the sum of the exposure weights is the book's gross exposure as a
multiple of its net asset value, which is the number a risk report should carry.
docs/VALUE_LAYER_SURVEY.md has the detail, including why BarraOne's basket-plus-offsetting-cash
construction is the alternative and what it costs.

## Coverage comes from the feed, never from an empty exposure vector

Cash is covered and genuinely riskless. Inferring coverage from the absence of factor loadings
would classify every cash position as unmodelled.

So `covered` arrives from the holdings feed and the engine respects it. A security marked
covered for which the model supplied no exposures is a different thing -- a data gap -- and is
collected in `RiskResult.covered_without_exposure` with a note saying it is carrying weight and
contributing no risk, and that either the exposure date is wrong for those names or they should
be marked uncovered. An *uncovered* name with no exposures is not flagged, because it is
already reported as uncovered and saying it twice is noise.

## One provenance stream

Each layer records what it substituted. A reader needs all of it in one place or they will read
none of it, so `RiskResult.qualifications()` gathers, in order from the outside in:

1. dates that resolved backwards, and which fell more than a long weekend behind the request
2. prices the market data could not supply, fallback price providers, an inferred price
   multiplier, a suspect par basis
3. the cash plug, zeroed prices, uncovered weight, leverage, a NAV gap
4. a repaired covariance, proxied factors, coverage exclusions, ill-conditioned arithmetic

De-duplicated, because the same repair is reported by every structure built on it and a reader
who sees it twenty times reads it zero times.

`is_exact` is true only when that list is empty.

### Conditioning is reported for all three sides

`PortfolioRisk.qualifications()` reports the portfolio side only. Conditioning is not shared
between the sides -- it depends on the exposure vector -- and **the active book is where it
bites hardest**, because a factor-neutral long-short book is the active book of any
index-hugging fund. So the engine adds the benchmark and active sides itself.

A basis-level substitution like a repaired covariance *is* shared by all three, so that stays
reported once.

## Status

Built, with 16 tests in `tests/midoffice/test_engine.py`: `RiskResult`, the assembly,
`positions_from`, and the consolidated provenance.

Not built: the adapters from real data sources (holdings, security master, market data, model
exposures), which is the same dependency as everywhere else in this layer; look-through
expansion of index securities, which is what activates the deferred defect in
MIDOFFICE_TODO.md; and the result's own serialisation, which should wait until there is a real
export to compare against.
