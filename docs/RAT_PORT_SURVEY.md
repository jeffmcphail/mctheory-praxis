# RAT Port Survey

**Status:** survey complete. No port code written.
**Sources:** `RAT.zip` (Risk Analysis Tool, production at TIAA-CREF ~2003-2018) and `MCRLib.zip`
(a later unfinished risk/trading library). Both by Jeff.
**Companion:** `docs/MIDOFFICE_ROADMAP.md` - this document is the reference for Workstream A.
**Date:** 2026-10-17

---

## Verdict in one paragraph

RAT is the port target, essentially in full. Its compiled surface is **about 35,000 lines, not
85,000** - three fifths of the files on disk are excluded from the build. The architecture is
already the right shape: one library, many adapters. The numerical core is already pure - no
database, file or configuration access anywhere in it. The as-of discipline the mid-office
build needs is **already present and better specified than the roadmap's own interface**. The
single reproducibility blocker is one mechanism, `getLatestDate`, reached from 77 call sites.
MCRLib contributes one genuinely valuable component (a real matrix library) carrying three
confirmed arithmetic bugs, plus two design ideas; its market-risk assembly is 25% finished and
silently returns zero for every metric, and must not be mined for logic.

---

## 1. What compiles, and what does not

`RATlib` is the entire engine. `RATServer`, `RATService`, `RATConsole`, `RATWindowsClient`,
`RATWebService`, `RATWebClient`, `RATCOM`, `RATAddIn` (Excel), `RATWebManager` and the test
projects are all adapters over it. **Only `RATlib` is in scope for the port.**

Within RATlib, the build list is the 72 `<Compile Include>` entries in `RATlib.csproj`.
On disk but **excluded from the build**, verified against every `.csproj` and `.sln` in the
archive:

| File | Lines | What it is |
|---|---|---|
| `RATEngineAllCode.cs` | 17,352 | pre-refactor single-file consolidation |
| `RATEngineFRED2.cs` | 14,340 | ditto, with a `ModelNameOld` enum - a migration variant |
| `RATEngineFRED1.cs` | 14,325 | ditto |
| `RATEngine20030818.cs` | 3,986 | date-stamped backup |
| `DB_new.cs` | 600 | superseded |
| 4 x `*.cs.bak` | - | `FundGroup`, `Portfolio`, `RATEngine`, `RiskModel` |

**50,603 of 85,419 lines are dead.** Each dead engine file declares `RATEngine` *plus*
`Parameters`, `FundGroup` and others that now live in their own files, which is how they are
identifiable as pre-refactor: the live `RATEngine.cs` (4,716 lines) declares only the engine.

Two compiled files are empty stubs: `MarketData2.cs`, `empty.cs`.

**Porting from a dead file would be the single worst way to start.** Derive the file list from
the csproj, never from the directory.

---

## 2. The public API, and a correction to the roadmap

Every entry point shares one signature shape:

```
riskSummary        (modelDate, exposureDate, marketDate, securityDate, modelCast,
                    metrics[], portName, benchName, ..., secFilter,
                    compLookThru, derivLookThru, sliceBenchmark, extArgs)
riskDecomposition  (... , factorGroups[], rootNode, ...)
calcSecurityRiskMetrics / calcAggregateRiskMetrics / sortSecurityRiskMetrics /
sortAggregateRiskMetrics / getAggRiskMetrics        -- same shape
```

**Four independent dates on every call**: the factor model, the exposures, the market prices
and the security reference data are each dated separately. Most risk systems conflate these.

**This corrects `MIDOFFICE_ROADMAP.md`.** The roadmap specifies

```
compute_risk(as_of, portfolio_id, model_id, snapshot_id) -> RiskResult
```

which collapses four independent dates into one. **RAT's signature is right and the roadmap's
is wrong.** The port keeps four dates. Amend the roadmap when the port interface is settled.

Other parameters worth carrying: `modelCast` (a bitmask selecting among models), `compLookThru`
and `derivLookThru` (composite and derivative look-through), `sliceBenchmark`, `secFilter`, and
a requested-`metrics[]` list so the output is driven by the request rather than computed whole.

---

## 3. The engine / model seam

Clean enough to preserve, with one inversion to fix.

`RiskModel` (`RiskModel.cs`, 886 lines) is the vendor-model abstraction and holds exactly the
three things the roadmap's builder contract names, plus the factor hierarchy:

| Concept | Where |
|---|---|
| exposures | `getExposureVector(SecurityData, model)`; `ExposureVector.cs` |
| factor covariance | `loadCovarMatrix(covarDate)` -> `CovarMatrix.cs` |
| specific risk | `SpecificRiskVector.cs`, carried on each `ExposureVector` |
| factor identity and hierarchy | `FactorList.cs`, `FactorTree.cs`, `FactorTreeNode.cs` |

`ExposureVector` carries operator overloads for `+`, `-` and scalar `*`, so exposures compose
linearly, and `getSubVector` / `getRestrictedVectorInclude` / `getRestrictedVectorRemove` for
decomposition. `CovarMatrix.getCovarBlock(factorList)` extracts the sub-block for a factor
subset - the mechanism factor-group decomposition runs on - and
`getCovarBlockExtraFactors(factorList, newFactorMapping)` maps factors absent from the vendor
matrix onto a proxy row and column.

**The inversion to fix:** `RiskModel`'s constructor takes `RATEngine`, and `FundGroup`,
`Portfolio` and `Tree` all reach back through `this.fg.ratEngine.parameters` to read
configuration mid-calculation. The model should not know the engine. In the port, configuration
is resolved at the boundary and passed in.

### Dating: three disciplines already present, one gap

1. **As-of, not latest.** `actualModelDate` comes from a query that fails with *"could not get a
   covar matrix on or before"*. That is a genuine as-of lookup.
2. **Dates injected into SQL**, via `$RATPARAMS_MODEL_DATE$` and `$RATPARAMS_EXPOSURE_DATE$`
   substitution, so a query cannot structurally drift to the wrong date.
3. **Requested vs actual distinguished.** `modelDate` is what was asked for; `actualModelDate`
   is what was found; when they differ it says so.

**The gap is where it says so: a log line.** The risk number does not carry the model date that
produced it. In the port, `actualModelDate` - and the resolved value of all four dates - belongs
**on the result object**, not in the log. This is the single most valuable change the port makes
to the design, and it is what makes lineage and reconstruction possible downstream.

---

## 4. What it computes

The core is one bilinear form. For any two exposure vectors:

```
cov(p1, p2) = sigma_spec(p1) . sigma_spec(p2)  +  x_p1' F x_p2
```

implemented as `matrixMult(exposures1, covarMatrix, exposures2)` plus a
`specificRiskVector * specificRiskVector` term. Everything else derives from it.

**Specific risk correlation is keyed, not assumed.** `SpecificRiskVector.cs:198`:

```csharp
return lSrVect.srKey == rSrVect.srKey ? lSrVect.sRisk * rSrVect.sRisk : 0.0;
```

Same issuer key -> contributions correlate at 1; different -> 0. A null key mints a synthetic
unique `"DEFAULT<n>"` so an unknown issuer correlates with nothing, which is the correct
treatment. **Port note:** the counter behind it is a non-atomic `static int` increment, there is
no `lock` anywhere in the library, and `RATEngine.cs:1554` spawns a background reload thread. A
race would mint duplicate keys and silently correlate two unrelated securities at 1. The port
drops the counter and uses an explicit null-means-own-issuer sentinel - no synthetic keys, no
shared counter, same semantics.

### The metric families

Three levels (total / benchmark / active) x two scopes (position / aggregate):

- **Risk:** total, common-factor, specific, benchmark, active (tracking error)
- **Contributions:** marginal and component contribution to each of the above, plus their
  percentage forms - roughly 24 methods on the benchmark/active side alone
- **Relative:** beta, benchmark correlation, cross-portfolio active correlation, implied alpha,
  active implied alpha
- **Tail:** VaR, CVaR, conditional VaR, and numeraire / benchmark / active variants.
  Expected shortfall under normality is `sigma * phi(z) / (1 - c) * sqrt(h/252)`, parameterised
  on `varHorizon` and `varConfidence`
- **Factor decomposition:** `getFactorDecomp` plus 33 `fd*` methods and `getFactorGroupDecomp`
  plus roughly 20 `fgd*` methods - exposure, risk, marginal risk, contribution, risk proportion,
  one-day and one-month return, each for portfolio / benchmark / active
- **Standalone factor risk:** `|x| . sqrt(diag(F))` - factor risk ignoring correlations
- **Coverage accounting:** `RiskMarketValue`, `RiskExpValue`, every `*IncNc` variant,
  `isCovered`, `numBenchNotHeldPositions`, `numOutOfBenchPositions`, `aggActiveShares`,
  `aggSDActiveWeight`, `aggActiveWeightRMS` - the "securities not covered by the risk model"
  bookkeeping, which is easy to overlook and tedious to add back

Position-level values are carried on `RiskData` at portfolio / benchmark / active, each in
market value, clean market value, exposure value and the three weight forms.

### The sparse / global exposure distinction - do not lose this

`RiskData` holds both `expVect` (non-zero exposures only, for speed) and `expVectGlobal`
(zero-padded to the portfolio's full factor set, so it conforms with the covariance matrix).
`mctr` uses them asymmetrically and deliberately:

```csharp
matrixMult(rd.expVectGlobal.Exposures, prd.covarMatrix, prd.expVect.Exposures)
```

MCRLib dropped this and its `mctr` passes the unpadded vector on both sides, which is
dimensionally wrong. The port keeps the distinction, or uses a representation where it cannot
arise.

---

## 5. Reproducibility inventory

Full sweep across the compiled set. Summary here; the finding that matters is category 2.

**Category 1 - clock reads.** There is **no `DateTime.Today`, no `UtcNow`, no
`Environment.TickCount` anywhere in the compiled library.** The time dependence in RAT is not
"now" leaking into arithmetic. One `DateTime.Now` in `RATManager.cs:65` builds output directory
and log file names; the remaining ~28 are logging and elapsed-time instrumentation, and five are
database-side `sysdate`/`getdate()` in status rows.

**Category 2 - "latest available". This is the whole problem, and it is concentrated.**

`RATEngine.getLatestDate` (three overloads, `RATEngine.cs:558-581`) is reached from 77 live
`getFundGroupKey` / `getRiskModelKey` call sites, each of which calls it twice:

- `getLatestDate(tableName)` - a date lookup with **no date argument**
- `getLatestDate(tableName, t)` - positionally indexes `dbInfo.Tables[name].Select()`, where
  `t = 0` is the newest row
- a null date parameter **defaults to latest**
- `"T3"` is matched by regex and means **"4th-newest row present"**, not "3 days before an
  as-of date"
- a caller-supplied date is **silently snapped backwards** to the newest row on or before it

Three consequences, in descending order of what they would have cost:

1. **The `order by ... desc` that makes `rows[0]` mean "latest" lives in an external query
   file**, not in the C# (`rat_default_queries.txt:5-7`). A port written from the source alone
   loses the invariant silently and indexes into arbitrary ordering.
2. **`"T0"` is not a date - it is a query against mutable process state.** It reads `dbInfo`,
   refreshed only when `getAvailibleDates()` happens to run, including from the unlocked
   background thread at `RATEngine.cs:1554`. Two calls with identical arguments, in the same
   process, can resolve differently. The engine is a remoting singleton
   (`RATManager.cs:217`), so its caches are process-global.
3. **`groupBy` re-resolves all four dates mid-calculation** (`RATEngine.cs:2256-2257`, reached
   from `Portfolio.cs:2236` and `:2763`), so a cache key is re-derived from "latest" partway
   through a computation that was supposed to be pinned.

Documentation and implementation also disagree: the comment at `RATEngine.cs:1101` says `T'i'`
counts back *"from the current day"*; the code counts back from the newest row present.

**In the port:** date resolution happens once, at the boundary, before any computation; the
resolved values travel on the result; `"T0"`-style relative dates are resolved against a stated
calendar, not against whatever rows exist.

**Category 3 - static mutable state.** Nine sites. The engine's caches
(`riskModelHash`, `fundGroupHash`, `marketDataHash`) are process-global by way of the remoting
singleton; `Log`'s statics early-return if already initialised, so a second `Parameters` object
silently inherits the first run's log file and verbosity; `SpecificRiskVector.defaultSequence`
as discussed. **There is not a single `lock` statement in the compiled library.** Benign and
worth recording so nobody re-flags it: the `static double[]` Acklam inverse-normal coefficients
in `PortfolioRiskData.cs:520-531` are initialised once and never written.

**Category 4 - ambient configuration.** No `ConfigurationManager`, no registry. Everything
arrives through environment variables (`RAT2012_PARAM_FILE`, `RAT2012_HOME`) and external
parameter and query text files. **The SQL is not in the source** - every `TABLE_*_QUERY` is read
from the file named by `QUERY_FILE`, and the code-built dates query is itself overridable
(`RATEngine.cs:520-521`). Hardcoded hosts, SMTP servers, paths and ports are site-specific and
simply do not survive the port. Database vendor is inferred by regex over the connection string
and changes the emitted SQL (`DBRiskLoader.cs:92`).

**Category 5 - hidden I/O inside maths. The good news, and it is substantial.** Verified zero
`OleDb*`, `File.*`, `StreamReader`/`StreamWriter`, `ExecuteReader`/`ExecuteNonQuery`/`Fill` in
`RiskMetrics.cs`, `Portfolio.cs`, `PortfolioRiskData.cs`, `CovarMatrix.cs`, `ExposureVector.cs`,
`Security.cs`, `IRCurve.cs`, `EquitySecurity.cs`, `FISecurity.cs`, `FactorTree.cs` - and zero
`parameters.items` reads in `RiskMetrics.cs`. **The numerical core is already pure.** Every
violation sits at the load/calculate seam, which is where the port puts a boundary anyway.

**One behaviour to replicate exactly, and the pattern it reveals.** `FundGroup.cacheData()`
(`:251-275`), gated on `FILL_MISSING_PRICES`, substitutes **zero** for a missing price and
mutates shared market data in place. **This was on in production** - a deliberate
least-bad-option call: zero and flag the position rather than fail a thousand-fund run, with
such errors rare and almost always well under 1% of total weight. The reference outputs
therefore contain zero-priced positions, so **the port must reproduce this to reach golden-file
parity**, not quietly improve on it.

It is flagged, not silent - an earlier draft of this survey said otherwise and was wrong:

```csharp
Log.WriteLine("Error:Null default price for " + security.securityCode +
              ", changing to 0 price for priceKey =" + priceKey + ".");
```

**And that is the pattern worth naming, because it runs through the whole system.** A
substituted model date is logged (`RiskModel.cs:267`). A zeroed price is logged. Coverage is
tracked on `RiskData.hasVector`, marked `//ToDone:ExcludeNotCovered`, and surfaced through the
`*IncNc` family. **RAT already records everything an audit layer needs. It records it in a
stream running parallel to the results rather than attached to them.**

So the port's single structural change is not *adding* provenance - it is **re-pointing**
provenance that already exists. Resolved dates, price substitutions and coverage exclusions
stop being log lines and become fields on the result object. That is a far smaller and far more
defensible change than bolting on an audit layer, and it is the design claim this port is
built on.

---

## 6. MCRLib: what to take

**Direct answer to the question that prompted this: for market risk, MCRLib is not an upgrade
over RAT.** Its `MarketRisk` assembly is roughly 25% transcribed and non-functional. It has no
risk model class, no factor tree, no factor decomposition and no tracking error. Of ~95 declared
metrics, 44 are registered and 100 are commented out.

**And it fails silently, which is the dangerous part.** `ExposureVector.Exposures` builds a
correctly-sized vector and leaves the assignment commented out, so it returns **all zeros**;
`FactorIds` returns an empty list; `RiskData`'s constructor ignores all four of its parameters;
`ObjectFactory.GetSecurity` tests `securityType.GetType() == typeof(Security)`, which is always
false, and returns null. Every metric is wrapped in `catch { return 0; }`. The result is that
`x'Fx` on a zero vector returns `0.0`, **which reads as a legitimate risk number**. Do not mine
this assembly for logic.

### Take, in this order

**1. `Mathematics/Matrix.cs` and `Vector.cs` - the one real gain.** RAT's `Matrix.cs` is 68
lines: constructors, `operator *`, `Diagonal`, an indexer. No transpose, inverse, determinant,
rank, norms or eigen decomposition - a grep for `Eigen` across the whole built RATlib returns
nothing. MCRLib adds all of those plus sub-matrix get/set, concatenation, and vector statistics
and windowing.

Most importantly it adds **PSD repair**, which RAT lacks. Both codebases carry the same guard -
`if (riskSquare < 0) throw ... "Check if your matrix is positive definite"` (RAT
`RiskMetrics.cs:3361`) - so RAT detects a non-PSD vendor covariance matrix and gives up.
MCRLib has the apparatus to fix it.

**Three arithmetic bugs, each verified by direct read, must be fixed first:**

| Site | Code | Effect |
|---|---|---|
| `Matrix.cs:261` | `Math.Pow(D[i,i], numerator/denominator)` | **integer division.** `pow(1,2)` gives `Math.Pow(d,0) == 1` for every eigenvalue, so matrix square root and inverse square root both silently return `V I V'`. This also makes `makeBlockDiagPosSemiDef` a no-op that looks like it worked. |
| `Matrix.cs:339` | `ones.Matrix*(1/ones.Count)*...` | `ones.Count` is `int`, so `1/ones.Count == 0` for any dimension >= 2. The de-meaning term vanishes and the property returns **uncentred** `X'X`, not a covariance matrix. Needs `1.0/`. |
| `Matrix.cs:388` | `D[i,i] = -D[i,i]/posDefScaleFactor` with `posDefScaleFactor = 1e-5` | flipping the sign is right; **dividing** by `1e-5` multiplies magnitude by 100,000. An eigenvalue of `-1e-6` returns as `+0.1`. Instead of flooring small negative eigenvalues near zero it inflates them into the largest in the matrix. Almost certainly meant `*`. |

The third is the sharpest: PSD repair is the capability RAT does not have, and as written it
would make a marginally non-PSD matrix substantially worse while reporting success.

**2. The `IntermediateResults` dictionary pattern - a design improvement, not a bug fix.** RAT
caches intermediate risk values in `double` fields and tests `if (x != 0) return x`, 14 times in
`RiskMetrics.cs` plus ~20 cache fields across `RiskData` and `PortfolioRiskData`. MCRLib
replaces the sentinel with `Dictionary<Metrics,double>` keyed presence, and separates
position-level from aggregate-level caches.

*Carefully stated, because it is easy to oversell:* the sentinel is **not** a correctness bug,
and the performance claim is mostly wrong too. `RiskMetrics.cs:1056` short-circuits the
empty-exposure case - cash, uncovered securities - with `if (rd.expVect.expHash.Count == 0)
return 0;` **before** any computation. The only values affected are non-empty exposure vectors
whose quadratic form is exactly `0.0`, which is measure-zero in floating point. RAT's answers
are correct and the cost is negligible. Take the dictionary because explicit presence is better
design and it removes ~20 ad-hoc fields, not because it fixes anything.

**3. Typed enum dispatch** for the metric table, replacing RAT's string-keyed `Hashtable`.
Mechanical, and the enum doubles as the metric catalogue.

**4. Two design ideas, without the code.** `Financial/ISecurity` splits RAT's `MarketData`
god-object into `IPrices` and `IFxRates` and makes local-vs-numeraire a first-class distinction,
with an arbitrary-length `string[] priceProviderSources` fallback chain in place of RAT's two
positional sources. That is the right shape - but every method body in `SecurityBase.cs` is
commented out, so there is nothing to carry but the shape. And MCRLib's *commented-out*
`getExposureVectorFromPortfolio` held a visited-node-list circular-reference guard that is
stronger than RAT's single `isProxiedToIndex` re-entrancy flag (`Security.cs:326`).

**Not for market risk, but complete and possibly useful later:** `Statistics/` (Johansen,
ADF, Box-Ljung, Durbin-Watson, variance ratio) and `Utils/HashDataTable` (a keyed-index layer
over ADO.NET that is a cleaner substrate than RAT's `DataView` + string `RowFilter` building).

---

## 7. Suggested port order

Each step ends somewhere testable. Golden-file parity against the running C# is the acceptance
bar throughout (`MIDOFFICE_ROADMAP.md`, Workstream A).

1. **Maths.** `Matrix` / `Vector` on numpy, carrying MCRLib's capability set with the three
   bugs fixed. PSD repair gets a test with a known marginally-non-PSD matrix.
2. **Model objects.** `FactorList`, `FactorTree`, `CovarMatrix` (including `getCovarBlock` and
   `getCovarBlockExtraFactors`), `ExposureVector`, `SpecificRiskVector` - with the null-issuer
   sentinel replacing the static counter, and the sparse/global distinction preserved.
3. **The bilinear form and the risk triple.** total / common-factor / specific, at position and
   aggregate level. First parity checkpoint, and the earliest point the port is worth anything.
4. **Benchmark and active**, including tracking error, and the marginal and component
   contribution families.
5. **Factor and factor-group decomposition** - the `fd*` and `fgd*` families. The largest single
   block of work.
6. **Tail metrics** - VaR, CVaR, conditional VaR and variants.
7. **Coverage accounting** - the `*IncNc` and `isCovered` family. Easy to defer and tedious to
   retrofit, so do not defer it past this point.

Deliberately **out of scope for the port**: remoting, COM, the Excel add-in, the web service,
the Windows client, `Mathematica.cs` (dead-but-compiled; nothing references it), `RATManager`'s
process and WMI management, and the hardcoded site configuration.

**Deferred to a later round** per the roadmap: fixed income and derivatives. The exclusion is
larger than it sounds - `BucketSecurity`, `IRCurve`, `Interpolation`, the swap and CDS hierarchy
and `updateFIExposureVectors` all sit behind it.

**Separate the two things that "defer fixed income" could mean.** Engine *capability* and model
*availability* are independent. The security hierarchy, `FISecurity` and its subclasses included,
is ported structurally as part of a complete engine - the engine can know what a swap is without
an FI factor model existing to run it against. What is deferred is building the model.

**When it is built, the agreed minimal set is: currency, term structure, one credit spread
factor, no mortgages.** Mortgages are the hardest piece and the least instructive - prepayment
behaviour is its own research programme and demonstrates nothing about audited workflow.

Term structure, when it arrives, is worth treating as a demonstration in its own right. Building
shift / twist / butterfly as the first three principal components of a curve universe makes the
factor model a **derived** input with a lineage of its own: which curve history, which window,
which estimation date, which sign convention on the rotation. That is a second lineage layer
sitting below the factor model, and reconstructing through both is a stronger demonstration than
reconstructing through one.

---

## 8. Port status and what it found

**Steps 1-7 are complete for equities** (`src/praxis/midoffice/risk/`, 169 tests).

| Step | Module | State |
|---|---|---|
| 1 Maths and basis | `basis.py` | done - `FactorCovariance`, `FactorBasis`, `nearest_psd` |
| 2 Model objects | `exposures.py`, `factors.py` | done - exposures, specific risk, factor identity and hierarchy |
| 3 The risk triple | `portfolio.py` | done - positions, portfolios, portfolio/benchmark/active |
| 4 Contributions | `contributions.py` | done - marginal and component at all three levels |
| 5 Decomposition | `decomposition.py` | done - factor and factor-group, both proportions |
| 6 Tail metrics | `tail.py` | done - VaR and expected shortfall, Acklam inverse normal |
| 7 Coverage | `portfolio.py` | done for risk; the `*IncNc` value family belongs with pricing |

Step 7 split in two on contact. The coverage accounting that affects **risk** -- counts, weight
sums, exclusion from exposure, qualifications on the result -- is done. The `*IncNc` family is
not a risk concept at all: it lives in `Portfolio.cs`'s value accumulation and needs prices,
accrued interest and `getPriceNumeraire`, which is the deferred loading layer. Note also that
`RiskMarketValue` is **not** a coverage concept despite the name -- it selects dirty or clean
value from `includeAccruedInterest`. Section 4 of this survey had that wrong.

### Things the original encodes that are easy to get backwards

Each of these cost a read of the C# to establish and would have produced plausible, wrong
numbers if assumed.

| Finding | Where it bites |
|---|---|
| **`fdTotalRisk` is `sqrt(sRisk^2 + x'Fx)`** -- total risk, not common-factor risk | It is the denominator of every factor proportion. Decompose against common-factor risk and the whole report is subtly wrong. It is also why `FactorTree` carries a `SPECIFIC_RISK` constant: the specific row is a named sibling of the factor groups, and factor contributions plus that row sum to total risk. |
| **`RISK_PROP` does not sum to one; `LITT_PROP` does** | `RISK_PROP` is built from standalone risk and ignores every correlation. Both are reported because the intuitive one does not add up. A reader who assumes it should concludes the numbers are broken. |
| **RAT's `mctr` is 1/100 of the textbook definition** | `mctr` divides by `100 * risk` and `ctr` multiplies back by 100, so components and percentages agree with the definitions here and only the marginals differ. Exported as `RAT_MARGINAL_SCALE`. |
| **`cvar` means *contribution to* VaR, not conditional VaR** | The group-level registrations (`VAR`, `CONTRIB_VAR`, `CONDITIONAL_VAR`, `CONTRIB_CONDITIONAL_VAR`) disambiguate it. Reading it the obvious way gives a number wrong by a factor of the weight. Nothing in `tail.py` is called `cvar`. |
| **Weights keep uncovered names in the denominator** | Production convention, and deliberate: PMs reconcile weights against their own systems and will not believe a tool whose weights do not tie. Risk is understated by roughly the uncovered fraction, and uncovered names are flagged instead. It looks like a bug. `WeightBasis.FULL` is the default; `COVERED` renormalises. |
| **The factor-set count in `FactorTreeNode` is vestigial** | Stored as a hashtable of factor id to an incrementing count; written in three places, read in none -- every consumer uses `.Keys` or `.Contains`. It is a set. |
| **Acklam's inverse normal has no refinement step** | So the port reproduces it exactly rather than reaching for `scipy.stats.norm.ppf`, at about 1.15e-9 relative error. `refine=True` adds a Halley step if the digits are ever wanted. |

### Deliberate divergences, and why

| Divergence | Reason |
|---|---|
| No `Matrix`/`Vector` wrapper classes | They exist in the C# because .NET had no ndarray. numpy is that wrapper; a class over it is overhead that makes every operation harder to read. |
| One basis, built from the union of portfolio and benchmark factors | Removes RAT's sparse `expVect` / padded `expVectGlobal` pair. Nothing is left to conform, so the dimensional-mismatch class of defect cannot occur -- the defect MCRLib actually shipped. |
| Specific risk keyed by security code when no issuer key exists | Same semantics as RAT's `"DEFAULT<n>"` -- unknown issuer correlates with nothing -- without a non-atomic `static` counter in a library with no locking and a background thread. |
| Factor rollup at construction, tree immutable | RAT needs a separate mutating `setAllNodeFactors()` pass and raises if called in the wrong order; it then catches "leaf has no factors", logs it and continues, so a malformed tree yields a silently incomplete rollup. `build` refuses. |
| One decomposition implementation, not 54 registrations | A factor is a group of one, so `sqrt(x_g' F x_g)` reduces to `|x_f| sqrt(F_ff)`. Factor-level and group-level are the same function. |
| VaR decomposition is a rescaling, not a calculation | Both tail measures are `risk * constant` and contributions sum to risk, so scaled contributions sum to the scaled measure. Writing a separate one creates somewhere for it to disagree. |
| `AmbiguousFactor` raised on a name defined in two models | RAT's single-argument `getFactorId` returns whichever it finds first: harmless with one model loaded, silently wrong with two. |
| Provenance on the result, not in a log | The one structural change this port makes. See section 5. |

### Still outstanding

- **The value layer**: prices, accrued interest, exposure value, and the `*IncNc` family that
  depends on them. It produces the weights this package consumes.
- **The parity harness**: a synthetic model in the expected schema, so the engine can be
  checked against RAT. The XML path (`LOAD_FROM_XML`) means the fixtures outlive the server
  that produced them -- one export, not an ongoing dependency.
- **Fixed income**, per the scoping in section 7.


---

*Last updated: 2026-10-17 (Chat: praxis_main_current)*
*Changes: 2026-10-21 - section 8 added: port status through step 7, the findings that
would have produced plausible wrong numbers if assumed, and the deliberate divergences.
Section 4's claim that RiskMarketValue is a coverage concept corrected there.
Initial survey. Establishes the compiled surface (~35k lines of 85k), the four-date
API signature that corrects the roadmap's interface, the engine/model seam and its one
inversion, the computation catalogue, the reproducibility inventory concentrated in
getLatestDate, the FILL_MISSING_PRICES behaviour and the provenance-re-pointing
claim the port rests on, the fixed-income scoping decision, and the MCRLib verdict - take the matrix library with three fixes, take two
design ideas, take nothing else.*
