# The parity harness

*Mid-office market risk engine. Written 2026-10-21.*

## Why it exists

Everything in `praxis.midoffice.risk` before this was verified against itself. The unit
tests check identities the implementation satisfies by construction, and hand-computed cases
checked by the same person who wrote the formulas. That is real verification and it caught
real errors, but it has a specific blind spot: a test suite written alongside an
implementation inherits its author's misreadings, and an identity like *contributions sum to
risk* holds just as well for a consistently wrong formulation as for a right one.

The harness closes that gap by computing every number twice, from two implementations that
share no code, and comparing them. It found three things on its first runs, one of which was
a real defect in committed engine code.

## Shape

```
src/praxis/midoffice/parity/
  case.py        the fixture: inputs AND the conventions they are computed under
  record.py      the flat record -- one number, four coordinates
  engine.py      drives praxis.midoffice.risk, emits records
  reference.py   an independent scalar implementation, emits the same records
  compare.py     tolerance policy and the comparison report
  harness.py     run_parity(case), and the declared-gap list
```

Both sides emit `Record(metric, level, entity, detail, value)`. The comparator takes two
record sets and does not know which is which. That symmetry is deliberate: a comparator
written against one side's data structure can only report on numbers that side produced, so
a metric that silently stopped being emitted would show up as a clean pass.

A synthetic case produces around 770 records.

## What makes the reference independent

Independence is the whole value, so it is engineered rather than assumed:

| | Engine | Reference |
|---|---|---|
| Linear algebra | numpy / BLAS | sorted double loop over sparse dicts |
| Active exposure | aggregates per-position differences | differences the two aggregates |
| Inverse normal | Acklam, ported from RAT | `statistics.NormalDist` (Wichura AS241) |
| Written from | the algebra, by hand | the *documented definitions*, not the engine source |

The last row is the important one. The reference was written from the docstrings and the
algebra, with the engine's implementation deliberately not consulted. Where the two
disagree, that disagreement is the finding -- and finding #2 below is exactly a case where
the code and its own docstring disagreed.

Agreement to 1e-10 across seeded random cases is therefore not an artifact of shared code or
a shared summation order.

### What the reference refuses

Narrow on purpose. Factor proxies and a non-zero PSD floor raise `UnsupportedCase` rather
than being approximated:

* **Proxies** have semantics that are surprising by design -- two factors proxied to one
  target become perfectly correlated and share its variance -- and a reference that guessed
  at them would manufacture disagreements that look like engine bugs.
* **A covariance the engine would repair** means the engine works from a different matrix
  than the reference does. The condition is that no eigenvalue sits below the floor, whatever
  the floor is -- not that the floor is zero. An earlier version refused any non-zero
  `relative_floor`, which became quietly wrong once the configured default stopped being
  zero: it meant the harness only ever exercised a configuration the engine no longer runs.
  The fixtures now default to the configured floor, so the harness tests the engine as it
  actually runs.

Both behaviours stay covered by the engine's own unit tests.

## Tolerances are derived, not chosen

A single hand-picked tolerance cannot work here, because the two sides use different
inverse-normal algorithms *on purpose*. RAT's Acklam approximation has a relative error of
1.15e-9 and no Halley refinement; AS241 is accurate to float precision. So the tail numbers
must differ by roughly Acklam's error when a case says `refine=False`, and must agree to
float noise when it says `refine=True`.

`policy_for(case)` reads the conventions the fixture records and derives the bound. **This is
why the conventions live in the fixture at all** -- a golden file that does not record its
weight basis, tail refinement and PSD floor cannot be interpreted later.

The two tail measures do not inherit the `z` error equally, and assuming they do was
finding #3:

```
VaR = sigma * z * sqrt(h/252)              d ln VaR / d ln z = 1
ES  = sigma * phi(z)/(1-c) * sqrt(h/252)   d ln ES  / d z     = -z
```

Acklam's error is relative, so `dz = rtol_z * |z|`. VaR carries `rtol_z` unchanged; ES
carries `|z| * dz = z^2 * rtol_z`. At 95% that is a factor of 2.7, at 99% a factor of 5.4.
A shared tolerance is either too loose for VaR or too tight for ES.

Everything else gets `rtol=1e-10` with an absolute floor of `1e-13` for numbers that are
legitimately near zero. Float noise for an O(1) number accumulated over a few hundred
thousand terms is about 1e-13 relative, so this leaves three orders of headroom while
staying far below any error a wrong formula produces.

## Absence is a result

Missing and extra keys are first-class findings. A number the reference computes and the
engine does not expose must be **declared** in `harness.KNOWN_GAPS`, with a reason:

* An undeclared gap fails -- a metric that quietly disappears is caught.
* A declared gap that is no longer a gap **also** fails -- so the list cannot accumulate
  stale excuses.

This is the same discipline as calibrating a detector's controls: a check that cannot fail
is not a check.

## What it found

### 1. `WeightBasis.COVERED` broke the contribution identity *(engine defect, fixed)*

`Portfolio._sum` multiplied the aggregate exposure by `full / covered`, but `contributions`
built each position's `v_i` from unscaled per-position weights. So `sum(v_i)` was the
*unscaled* aggregate while the risk being decomposed was the scaled one, and the
contributions summed to `risk / scale` rather than `risk` -- off by 15% on a representative
case, at every level.

Every individual number looked plausible. Only the total betrayed it, and `check_sum`
diagnosed it correctly once pointed at the case: *"they were computed against different
aggregates."*

It survived the original suite because no test combined `COVERED` with `check_sum`. The
step-7 tests checked the *ratio* of `FULL` to `COVERED` risk, which is a different property
and was correct.

Fixed by extracting `Portfolio.weight_scale(weight_of)` and threading it through
`_level_spec`, so the factor reaches the weights and the `v_i` as well as the aggregate.
Residuals are now ~1e-17 under both bases at all three levels, with regression tests in
`test_contributions.py` built on a book that does **not** sum to one.

### 2. The scale factor is `full / covered`, not `1 / covered` *(reference wrong, docstring imprecise)*

The reference used `1 / covered`, which is what "renormalises each side over its covered
names" naturally suggests. The engine used `full / covered`. The engine is right: it
redistributes the uncovered weight across the covered names, preserving the book's invested
total. `1 / covered` would gross a 60%-invested sleeve up to 100% and inflate every risk
number by two thirds.

**The two are identical for any book whose weights sum to one** -- which is what hand-built
fixtures naturally are, and why this was invisible until the generator started producing
books that do not. The docstring now says what the code does, and
`test_the_two_candidate_factors_coincide_when_the_book_sums_to_one` pins the coincidence so
the next reader knows it is understood rather than assumed.

### 3. Expected shortfall inherits `z^2` of the z error *(harness tolerance wrong)*

Covered above. My first tolerance assumed both tail measures were linear in `z`; ES failed
at 2.99e-9 against a 2.3e-9 bound, and the discrepancy was derivable rather than noise.

### 4. Drilling into a leaf group returned the whole basis *(engine defect, fixed)*

`group_decomposition(vector, basis, tree, parent=node)` is documented to partition `node`.
For an interior node it did. For a **leaf** -- and in a two-level model every group is a leaf,
since factors attach to leaves rather than being nodes themselves -- `children_of(node)` is
empty, and the function fell through to `factor_decomposition(vector, basis)`. That decomposes
the *entire basis*, so "break this group down" silently became "break the book down by
factor": the rows covered every factor in the model, `parent` came back empty, and the
contributions summed to total risk instead of to the group's own contribution.

Everything about the output looked plausible. `sums_to` matching `total_risk` is exactly what
a correct root-level decomposition does, so the one number that could have betrayed it was
the number that made it look right.

The existing suite did not merely miss this. It **asserted** it:

```python
def test_a_leaf_parent_falls_through_to_factor_level():
    d = group_decomposition(vector, basis, t, parent="Beta")
    assert [r.name for r in d.factor_rows] == [str(f) for f in basis.factor_ids]
```

That test was written against what the code did rather than against what the operation means,
and its name records the bug as though it were the design. Five tests replace it, pinning the
contract instead: a leaf partitions its own factors, the rows sum to that node's contribution
rather than to total risk, no specific row appears below the root, and the interior path that
was already correct stays that way.

Contribution is additive over a partition, so `sum(contribution of f for f in node)` must
equal `contribution(node)` as measured one level up. That identity is now checked through the
harness on **both** implementations -- the engine agreeing with a reference that made the same
mistake would prove nothing.

### Open gap

`CoverageSummary` reports uncovered weight for the portfolio only. The benchmark's uncovered
weight matters for the same reason the portfolio's does -- it scales benchmark risk and
therefore beta and tracking error. Declared in `KNOWN_GAPS`; the fix is to add the field and
delete the line.

## Coverage

**Compared** (about 770 records per synthetic case): total, common-factor and specific risk
at portfolio, benchmark and active level; beta; correlation; tracking error; coverage weights
and counts; per-position weight, component, marginal, percent, common-factor and specific
contributions at all three levels; per-factor exposure, standalone risk, marginal,
contribution, risk proportion and Litterman proportion at all three levels; VaR and expected
shortfall at every confidence and horizon in the case; the root-level group decomposition
and a drill-down into each of the root's children, with the sum of each decomposition's
contributions compared as a number of its own.

**Not yet compared:**

* Proxies and PSD repair -- refused by the reference, by design, covered by unit tests.
* `position_risk` standalone numbers.
* The value layer, which does not exist yet.

The fixture generator produces the awkward shapes deliberately -- names held but not in the
benchmark, names in the benchmark but not held, uncovered names carrying real exposure,
negative active weights, issuers spanning several securities -- and
`test_generator_actually_produces_the_awkward_shapes` asserts they are present, because the
agreement tests are only as good as the shapes the fixtures contain.

### The tests that make the agreement mean something

`test_parity.py` has 67 tests in three groups. The middle group is the one that matters:
each of eleven mutations perturbs exactly one input and asserts both that the harness reports
a disagreement **and** that it lands on the metrics that input should influence. If a mutation
produced no mismatch, that input would not be reaching any number the harness checks -- the
agreement would be real but vacuous over that part of the surface.
`test_every_mutation_is_distinguishable` additionally requires that no two mutations produce
the same signature.

Calibration tests pin the tolerance boundary from both sides, so it cannot be quietly
widened until the suite goes green.

### Where the tolerance story stops holding

The 1e-10 tolerance assumes the arithmetic is well conditioned, and that assumption is
checkable: `RiskDecomposition.digits_intact` reports how many digits survive. Where it is
comfortable, two machines agree to float noise and the tolerance has six orders of headroom.
Where amplification is large, two different BLAS implementations disagree without bound -- the
same factor-neutral book returned exactly `0.0` under OpenBLAS and `1.25e-10` under MKL.

So a parity disagreement on a result whose `digits_intact` is flagged is evidence about the
BLAS, not about the port. docs/CONDITIONING.md has the detail. This matters most for golden
files captured on one machine and compared on another.

### Which side is noisier

The reference uses naive sorted Python loops; the engine uses BLAS, whose pairwise summation
gives error growth of `log(n)` rather than `n`. So on the rare occasions the two disagree at
the 1e-13 level, **the reference is the more likely culprit**, not the engine. The 1e-10
default tolerance accommodates that comfortably, but if it ever needs tightening, the
reference is what will hit the wall first.

## Running it

```
pytest tests/midoffice/test_parity.py
```

```python
from praxis.midoffice.parity.harness import run_parity
from praxis.midoffice.parity.synthetic import synthetic_case

report = run_parity(synthetic_case(seed=7))
print(report.describe())
```

## Next: parity against RAT itself

Everything above compares the port to an independent implementation of the same
specification. It does not yet compare the port to fifteen years of production output. The
route to that is RAT's own XML path, and reading the source settled how it works.

**The flag runs both ways, which is better than expected.**

| Parameter | Effect |
|---|---|
| `EXPORT_TO_XML=TRUE` | on the DB path, writes each DataSet to XML after loading it |
| `LOAD_FROM_XML=TRUE` | reads from those files instead of the database |
| `XML_DIR` | output directory, defaulting to `%RAT2012_HOME%` |

So a single capture run against the database produces files that let **RAT itself run with no
SQL Server at all**. That is a live oracle that can be re-queried, not merely a stored
snapshot -- a materially stronger harness than golden files, and it means the Barra SQL Server
backup is needed once rather than kept alive.

The files are `DataSet.WriteXml(..., XmlWriteMode.WriteSchema)`, which embeds the XSD inline.
The format is therefore self-describing, so a loader can validate its assumptions against the
schema rather than trusting them.

| DataSet | Filename | Tables |
|---|---|---|
| `modelData` | `modelData_{modelDate}_{exposureDate}.xml` | `Factor`, `FactorGroup`, `FactorLink`, `FactorReturns`, `CovarMatrix`, plus one exposure table per model |
| `fundGroupData` | `fundGroupData_{securityDate}_{marketDate}.xml` | `Security`, `Position`, `FxFwdSecurity`, `EIFSecurity`, `CompositeSecurity`, `AggTreeLinks`, `TreeNodes` |
| `marketDataSet` | `marketDataSet_{marketDate}.xml` | `Prices`, `Fxrates`, `IRRates` |

Two things to know before relying on a capture run.

**The exposure table names are configuration, not constants.** `RiskModel` discovers them by
regex over its parameter keys (`modelTableNames[model] = parameters.items[key]`), so a loader
must read them from the file rather than hardcode them. Relatedly, `isModelTableFlat` -- the
flat-versus-nested exposure layout switch, decided by column count -- is consulted only for
`FI_MODEL`, so that ambiguity is confined to fixed income and does not touch the equity path.

**A failed export does not fail the run.** Both `extractFromXML` and `writeToXML` catch
`Exception`, log at Error level, and continue -- leaving the DataSet empty or the file
unwritten. The run completes and produces output. This is the same describing-versus-acting
pattern as `FILL_MISSING_PRICES`: the information is recorded, in a log stream parallel to
the results, where nobody reading the results will see it. So a capture run can hand back a
partial fixture set with nothing but a log line to say so, and **the first thing a loader
must do is verify the export rather than assume it** -- all three files present, every
expected table there, row counts non-zero, dates matching the request.

What is deliberately *not* built yet: the RAT-XML-to-`ParityCase` mapper. The table names
above are verified from the source, but the column names inside each table are not, and
inventing them would be building on a description rather than on the thing described. One
`SELECT TOP 1 *` per table, or one real export, settles it.
