# As-of dating and the four-date interface

*Mid-office market risk engine. Written 2026-11-01.*

## The four dates

A risk computation does not run "as of a date". It runs on four, and they are genuinely
independent:

| date | what it fixes |
|---|---|
| model | when the factor covariance was estimated |
| exposure | when the securities' factor exposures were measured |
| market | when prices, accrued interest and FX were taken |
| security | which security master, so which identifiers and terms |

A model published monthly applied to exposures from a different month, priced today, against
last night's security master is an ordinary request, not a mistake. RAT states this plainly in
its cache keys: `getFundGroupKey(securityDate, marketDate)` and
`getRiskModelKey(modelDate, exposureDate)` each join two of them with a pipe, which is the
clearest evidence in the source that collapsing them to one date would be wrong. The engine's
real signature is therefore:

```
compute_risk(model_date, exposure_date, market_date, security_date,
             portfolio_id, benchmark_id, model_id, snapshot_id, metrics) -> RiskResult
```

which corrects the single-date interface in an earlier draft of MIDOFFICE_ROADMAP.md.

## Every date is a request, and the answer may differ

The data has the dates it has. A request for 15 June is answered with the most recent date on
or before it, which is correct and routine. `praxis.midoffice.asof` keeps the request, the
answer and the distance between them.

RAT resolves all four through `RATEngine.getLatestDate` and returns a bare `yyyyMMdd` string:

```csharp
DataRow[] rows = this.dbInfo.Tables[tableName].Select("date <= #" + latestDate + "#",
                                                      "date desc");
return ((DateTime) rows[0]["date"]).ToString("yyyyMMdd");
```

The caller receives a date and cannot tell whether it is the one it asked for. A model six
months stale and one from yesterday are the same kind of string to everything downstream, and
the risk numbers built on either look equally plausible. This is the reproducibility hole the
survey flagged, and re-pointing it onto the result is the same structural change the port makes
everywhere else.

`ResolvedDate` carries `requested`, `actual`, `gap_days`, `offset` and `substituted`.
`AsOf.qualifications()` lists the substitutions, and separates *stale* from merely
*substituted* -- a two-day gap is a weekend, a thirty-day gap is a different month's model, and
only the second is worth a reader's attention. The threshold is four days and is deliberately
crude.

## Three request forms, kept from RAT

| form | meaning |
|---|---|
| `None`, `""`, `"latest"` | the most recent available date |
| `"T0"`, `"T3"` | counting back through the calendar |
| `"20150630"`, `"2015-06-30"`, a `date` | that day, or the most recent before it |

Asking for `latest` or `Tn` and receiving it is not a substitution: the request named a
position in the calendar rather than a day, so there was nothing to miss. Only an explicit date
can be substituted, which is why `substituted` is defined as `gap_days > 0`.

## Two things RAT leaves to chance

**Ordering.** `getLatestDate(tableName, int t)` -- the form answering `null` and the `Tn`
notation -- calls `Select()` with no sort argument and indexes the result. `DataTable.Select()`
returns load order, so `T0` is the latest date only if the query that filled the table sorted
descending. Those queries live in parameters (`TABLE_MODEL_DATES_QUERY`,
`TABLE_EXPOSURE_DATES_QUERY`, `TABLE_FUNDGROUP_DATES_QUERY`), not in code. The date form of the
same function passes `"date desc"` explicitly, so the author knew ordering mattered in one
branch and not the other.

It presumably held for fifteen years because the queries do sort. But it is a correctness
guarantee kept in configuration with nothing to enforce or check it, and a wrong date produces
entirely plausible risk numbers for the wrong day. `DateCalendar` sorts at construction and
refuses duplicates, which is the whole of the fix.

**Bounds.** `getLatestDate` checks `rows.Length == 0` and then indexes `rows[t]`, so an offset
past the start of the calendar is an `IndexOutOfRangeException` rather than a message naming
the range available.

A third, smaller one: an unparseable date string is handed to a `Select` filter, where it
becomes a malformed-expression error a long way from the typo that caused it. `parse_request`
refuses it where it is written.

## Resolving backwards only

A request before the calendar's earliest date raises rather than resolving forwards. Resolving
forwards would answer a question about 2015 with data from 2016, which is the one substitution
that cannot be defended -- it is not staleness but hindsight, and in a research context it is
the mechanism behind every accidental look-ahead.

## Tests

`tests/midoffice/test_asof.py`, 40 tests. The two that matter most:
`test_the_calendar_sorts_whatever_order_it_is_given`, which closes the configuration
dependency, and `test_a_date_with_no_data_resolves_backwards_and_records_the_gap`, which pins
the record RAT does not keep. `test_calendars_may_have_different_dates_available` shows why the
four resolve separately: a monthly model and daily prices answer the same request with
different dates, and that is correct.

## Not built

Calendar loading from a data source, which needs the dates tables -- the same dependency as the
market-data adapter. `DateCalendar.of` takes dates from anywhere, so the adapter is a thin
translation when it arrives.
