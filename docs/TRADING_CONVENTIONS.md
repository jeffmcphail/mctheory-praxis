# Trading Conventions

> **About this file**: Pure Praxis domain rules. These are facts about the
> markets and trading mechanics that any strategy / engine / scanner work
> must respect, regardless of who or what is doing the work.
>
> Not a TODO list (that's `claude/TODO.md`) and not Claude-behavior rules
> (those are in `claude/CLAUDE_CODE_RULES.md`). These are operating rules
> about the markets themselves.

---

## Polymarket: resolution oracle verification

**Always check the resolution oracle/source for any Polymarket bet before
trading.** Specifically verify:

1. **Data type**: seasonally adjusted vs. non-seasonally adjusted (matters
   for inflation, employment, GDP markets, etc.)
2. **Source agency**: which government / private agency publishes the
   number that resolves the market
3. **Threshold conditions**: exact comparison operators and reference
   values used in the market description
4. **Reporting cadence**: when the resolving number is published, with
   what lag, and whether the market closes before or after publication

The market title is not authoritative; the market rules document linked
from the Polymarket UI is. Always read it through, not just the title.

A live example of the danger: many "inflation > X%" markets resolve on a
specific MoM print at a specific lag. A trader thinking they're betting
on annual inflation when the market is monthly (or vice versa) will mis-
price the probability by a wide margin.

---

## Weather bets: airport stations + multi-model ensembles

When designing or reviewing weather-bet strategies (`scan_all_weather.py`
or any successor):

### Verify the resolution station

Polymarket weather markets resolve against specific airport weather
stations, NOT city center or major hub airports. The mismatch is a
common edge-leak source.

- **NYC**: KLGA (LaGuardia), NOT KJFK (JFK)
- **Dallas**: KDAL (Love Field), NOT KDFW (DFW)
- (More entries to be added as discovered)

Always cross-check the market's rules document before assuming a station.

### Use a multi-model forecast ensemble

For any temperature / precipitation bucket-range bet, do not bet on a
single point estimate. Use multi-model ensembles to construct a probability
distribution over the resolution buckets:

- **GFS** (US Global Forecast System, NOAA)
- **ECMWF** (European Centre for Medium-Range Weather Forecasts)
- **UKMO** (UK Met Office)
- **NWS** (National Weather Service official forecasts)

Aggregate via Open-Meteo (free API for GFS/ECMWF/UKMO ensemble) and
api.weather.gov (free, official NWS).

The bucket-probability distribution from the ensemble plus the market's
implied distribution gives the actual edge. Single-model point forecasts
usually do not.

### Be skeptical of "easy" weather edges in 2026+

Weather betting has been heavily content-farmed since early 2026.
Published P&L screenshots from late 2024 / early 2025 likely overstate
edge available today; sophisticated counterparties have closed most
naive sources of advantage. Assume any documented strategy has degraded
unless re-validated on recent data.

---

## (Future entries)

Add new conventions in domain-grouped sections as they're identified. A
convention belongs here when:

- It's a stable fact about the market or trading mechanics (not a
  current TODO)
- It applies to multiple strategies / engines, not a single Brief
- Forgetting it would cause a meaningful edge-leak or trade error

Convention entries should be terse (a few paragraphs max). For deeper
findings (e.g., "we tried X and found Y doesn't work"), see
`docs/TRADING_FINDINGS.md`. For multi-phase build plans, see
`docs/MEV_BUILD_PLAN.md` or future per-plan files.

---

## Revision history

| Date | Cycle | Change |
|---|---|---|
| 2026-04-30 | 16 | Initial. Polymarket oracle check + weather bet airport stations (KLGA / KDAL gotchas) + multi-model forecast ensemble guidance, captured from memory entries 17 and 22. |
