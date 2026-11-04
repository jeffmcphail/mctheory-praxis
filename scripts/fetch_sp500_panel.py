"""Fetch an S&P 500 daily panel from Polygon and save it for the factor model.

    python scripts/fetch_sp500_panel.py --probe          # what it would cost, fetch nothing
    python scripts/fetch_sp500_panel.py --limit 25       # a small run first
    python scripts/fetch_sp500_panel.py                  # the whole index

Point-in-time membership comes from `midoffice.refdata.sp500`, so the universe is the index
as it stood on the start date rather than as it stands today.  Without that, every company
dropped from the index over the window is missing -- and companies are usually dropped
because they shrank, so the Size cross-section would be biased by exactly the mechanism it
is trying to measure.

Nothing here is destructive and the client caches, so a re-run after an interruption costs
only the requests it had not already made.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from praxis.midoffice.refdata.polygon import (  # noqa: E402
    PolygonClient,
    PolygonError,
    read_api_key,
)
from praxis.midoffice.refdata import sp500  # noqa: E402
from praxis.midoffice.refdata.prices import (  # noqa: E402
    MARKET_PROXY,
    PanelRequest,
    build_security_panel,
    estimate_requests,
)

DEFAULT_OUT = Path("data/external/polygon")
DEFAULT_SNAPSHOTS = Path("data/external/sp500")


def _universe(args, start: dt.date) -> tuple[str, ...]:
    """Point-in-time membership, fetching a snapshot only if none is on disk."""
    directory = Path(args.snapshots)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = sp500.load_latest(directory)
        print(f"  snapshot {snapshot.content_hash[:12]} from {snapshot.fetched_at}")
    except (FileNotFoundError, ValueError, LookupError):
        print("  no snapshot on disk; fetching one from Wikipedia")
        try:
            snapshot = sp500.fetch()
        except sp500.SourceUnavailable as error:
            raise SystemExit(
                f"\nCould not retrieve the constituent page:\n  {error}\n"
                f"No snapshot was written, so any previous one still serves."
            ) from None
        except sp500.ParserUnavailable as error:
            # A missing package deserves one line, not a traceback through pandas.
            raise SystemExit(f"\n{error}") from None
        except sp500.StructureError as error:
            raise SystemExit(
                f"\nThe source page did not parse as expected:\n  {error}\n"
                f"No snapshot was written, so any previous one still serves."
            ) from None
        path = sp500.save(snapshot, directory)
        print(f"  wrote {path}")
    try:
        members = sp500.membership(snapshot, start)
    except sp500.CoverageError as error:
        raise SystemExit(
            f"the snapshot cannot answer {start}: {error}\n"
            f"Pick a start date inside its coverage, or re-fetch for a wider one."
        ) from None
    print(f"  {len(members)} constituents as of {start}")
    return members


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-01-03")
    parser.add_argument("--end", default=dt.date.today().isoformat())
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N constituents, for a trial run")
    parser.add_argument("--shares-every", type=int, default=63,
                        help="trading days between shares-outstanding samples")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS))
    parser.add_argument("--cache", default="data/external/polygon/cache")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--index", default=None,
                        help="market series symbol; defaults to SPY (I:SPX needs an "
                             "indices subscription)")
    parser.add_argument("--probe", action="store_true",
                        help="report the request count and exit without fetching")
    args = parser.parse_args(argv)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    print("Universe")
    members = _universe(args, start)
    if args.limit:
        members = members[: args.limit]
        print(f"  limited to {len(members)} for this run")

    request = PanelRequest(
        start=start,
        end=end,
        tickers=members,
        shares_every=args.shares_every,
        **({"index": args.index} if args.index else {}),
    )
    estimated = estimate_requests(request)
    print(f"\nPlan\n  {start} .. {end}, {len(members)} securities")
    print(f"  market series {request.index}")
    print(f"  about {estimated:,} requests "
          f"({request.shares_every} trading days between shares samples)")
    if args.probe:
        print("\n--probe given; nothing fetched.")
        return 0

    try:
        key = read_api_key(args.env)
    except PolygonError as error:
        raise SystemExit(str(error)) from None
    client = PolygonClient(api_key=key, cache_dir=Path(args.cache), min_interval=0.02)

    print("\nFetching")
    width = len(str(len(members)))

    def progress(done: int, total: int, ticker: str) -> None:
        if done % 25 == 0 or done == total:
            print(f"  {done:>{width}}/{total}  {ticker}")

    try:
        panel, market, report = build_security_panel(
            client, request, on_progress=progress
        )
    except PolygonError as error:
        raise SystemExit(
            f"\nFetch failed:\n  {error}\n"
            f"A 403 on the market series usually means the symbol needs a subscription the "
            f"plan does not include -- I:SPX needs indices, SPY does not."
        ) from None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"sp500_daily_{start}_{end}.npz"
    np.savez_compressed(
        destination,
        dates=np.array([d.isoformat() for d in panel.dates]),
        securities=np.array(panel.securities),
        prices=panel.prices,
        unadjusted_prices=panel.unadjusted_prices,
        volumes=panel.volumes,
        shares_outstanding=panel.shares_outstanding,
        highs=panel.highs,
        lows=panel.lows,
        market_returns=market,
    )

    print(f"\nResult\n  {report.describe()}")
    print(f"  wrote {destination}")
    for note in report.qualifications():
        print(f"  - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
