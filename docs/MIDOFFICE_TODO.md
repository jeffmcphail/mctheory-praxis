# Mid-office risk engine -- task and defect register

*Things deliberately deferred, with the reason and the condition for picking them up. Distinct
from MIDOFFICE_ROADMAP.md, which is the forward plan; this is what we decided not to do yet.*

---

## Deferred: correct the look-through clean-value defect, AFTER regression completes

**Status:** blocked on regression parity. Do not fix before then.

**The defect, in RAT.** `Portfolio.cs:910-913`, in `addPortfolioPositions(IndexSecurity)` --
the look-through path that expands an index security into its constituent positions:

```csharp
this.MarketValueIncNc      = this.marketValue;
this.cleanMarketValueIncNc = this.cleanMarketValue;
this.numExpValueIncNc      = this.numExpValue;
this.cleanMarketValueIncNc = this.cleanNumExpValue;   // should be cleanExpValueIncNc
```

The fourth line assigns `cleanMarketValueIncNc` a second time instead of `cleanExpValueIncNc`.
The correct version of the same block is at `Portfolio.cs:843-846`, inside
`addPositions(IList, IList)`, which is the main position-loading path -- so the two paths
disagree and the main one is right.

**Consequences.** Two:

1. `cleanMarketValueIncNc` ends up holding the clean *exposure* value. `refreshRiskValues()`
   (`PortfolioRiskData.cs:467`) then copies it into `cleanMarketValue`, and `cleanMarketValue`
   is the weight denominator whenever accrued interest is excluded
   (`RiskData.cs:110`: `riskMarketValue = includeAccruedInterest ? marketValue :
   cleanMarketValue`).
2. `cleanExpValueIncNc` is never assigned on that path, so it keeps its default while
   `PortfolioRiskData.cs:282` and `:309` read it.

**Conditions for a wrong number.** All three must coincide:

* the book holds an index security expanded through look-through, AND
* `INCLUDE_ACCRUED_INTEREST_RISK = false`, AND
* the expanded book contains leverage, so clean exposure and clean market value differ.

On a cash-equity look-through the two are equal and nothing happens.

**Why it was not fixed at the time (Jeff, from memory -- roughly 2006).** The excluded-accrued
configuration was used only in certain ad hoc runs, where market value had to line up with the
fund's figure in the accounting system under accounting requirements that excluded accrued
interest. It ran rarely, for very few funds, and the resulting active-risk impact was de
minimis, so a code correction was judged not worth it.

**What the port must do, and when.**

* **Now, while regression parity matters:** reproduce the behaviour. A port that computes the
  *correct* denominator where production computed the wrong one fails parity on exactly those
  funds, and a regression suite that disagrees with production for a good reason is
  indistinguishable from one that disagrees for a bad one. Reproduce it behind an explicitly
  named flag so it reads as a compatibility choice rather than as the same mistake made twice.
* **Note:** the port has no look-through path yet, so there is nothing to reproduce today.
  This item becomes live when look-through (index-security expansion) is built -- that is the
  moment to add the flag, not before.
* **After regression completes:** correct it, and record the correction as a deliberate
  divergence in RAT_PORT_SURVEY.md section 8 with the date regression signed off.

**Test to write when the flag goes in:** a look-through book with a levered constituent, valued
with accrued excluded, asserting that the flag changes the denominator and that the corrected
path gives clean market value rather than clean exposure value.

---

## Deferred: vendor-published PSD floor

`FactorCovariance` carries no vendor metadata, so there is nowhere to record a floor that Barra
or another provider publishes alongside the matrix. A published floor should take precedence
over the measured default of `1e-8` in `config/midoffice_risk.toml`. Picked up when a real
vendor matrix arrives, since inventing the metadata shape before seeing one is guesswork.

---

## Deferred: amplification on contributions and decomposition rows

`RiskDecomposition` reports `amplification` and `digits_intact` for a risk number. Position
contributions and decomposition rows divide by that risk, so they inherit its conditioning,
but nothing asserts it and no row carries its own figure. See docs/CONDITIONING.md.
