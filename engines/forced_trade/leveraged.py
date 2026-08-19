"""
engines/forced_trade/leveraged.py

T3 -- Scenario D1 (leveraged token daily rebalancing): what is the candidate
universe, and is the SECOND LEG available?

THE INVERSION
-------------
`engines/xsec_reversal/universe.py` excludes Binance leveraged tokens because
their daily leverage reset manufactures artificial mean reversion that would
have produced a false positive in the Cycle 60 cross-sectional study. Here that
same mechanic IS the hypothesis, so Cycle 60 exclusion list becomes this cycle
candidate list.

WHY THIS MODULE RE-ENUMERATES INSTEAD OF READING symbols_all.txt
----------------------------------------------------------------
`data/external/xsec/symbols_all.txt` is the POST-FILTER Cycle 60 universe --
`cmd_symbols` applies `filter_symbol_names` before writing it, so every
leveraged token has already been removed. Reading that file would report a
universe of zero. The enumeration is re-run against the S3 bucket listing with
the exclusions inverted, which is the same survivorship-bias-free path.

THE SUBSTRING TRAP
------------------
`DEFAULT_LEVERAGED_PATTERNS` is matched with `in`, not `endswith`, so "UPUSDT"
also matches JUPUSDT (Jupiter) and SYRUPUSDT (Maple SYRUP) -- real spot assets
with no leverage mechanic at all. This is exactly the failure the tokenized-
equity comment in universe.py warns about for "BUSDT". This module applies a
STRUCTURAL rule instead: a Binance leveraged token is <BASE> + one of
UP/DOWN/BULL/BEAR + <QUOTE>, and <BASE> must itself be a symbol that trades in
the archive. JUPUSDT fails because "J" is not a traded base.

THE SECOND LEG
--------------
A leveraged-token rebalance study needs BOTH legs: the token and its underlying.
Without the underlying there is no reference return, so there is no study. This
module resolves each candidate underlying and confirms it exists in the archive
before counting the candidate as usable.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger("forced_trade.leveraged")

# Binance leveraged-token (BLVT) suffixes and the FTX-style BULL/BEAR tokens
# Binance listed and later delisted. Order matters: longest first, so BULL is
# not shadowed by a shorter match.
DEFAULT_SUFFIXES = ("DOWN", "BULL", "BEAR", "UP")
DEFAULT_QUOTES = ("USDT", "BUSD", "USDC")


@dataclass(frozen=True)
class LeveragedParams:
    suffixes: tuple = DEFAULT_SUFFIXES
    quotes: tuple = DEFAULT_QUOTES
    require_underlying: bool = True
    # A symbol whose archive data stops this many WHOLE MONTHS before the
    # archive frontier is treated as delisted. 2 is deliberate: the current
    # month is never complete in the monthly archive, so a 1-month lag is
    # normal for a live symbol and would misclassify it.
    delisted_gap_months: int = 2


def split_leveraged(symbol: str, p: LeveragedParams = LeveragedParams()):
    """(base, suffix, quote) if `symbol` is structurally a leveraged token.

    Returns None otherwise. Bare BULLUSDT/BEARUSDT have an empty base -- those
    are the Binance-listed FTX 3x BTC tokens and are reported separately rather
    than silently dropped or silently attributed to a base.
    """
    su = symbol.upper()
    for q in p.quotes:
        if not su.endswith(q):
            continue
        stem = su[: -len(q)]
        for suf in p.suffixes:
            if stem.endswith(suf):
                return stem[: -len(suf)], suf, q
    return None


def classify(all_symbols, p: LeveragedParams = LeveragedParams()) -> pd.DataFrame:
    """Structural classification of every archive symbol.

    `underlying_exists` is what decides usability: it is the second leg.
    """
    universe = {s.upper() for s in all_symbols}
    rows = []
    for s in sorted(universe):
        parsed = split_leveraged(s, p)
        if parsed is None:
            continue
        base, suf, q = parsed
        underlying = f"{base}{q}" if base else None
        # A bare BULL/BEAR token has no base in its own name; Binance listed
        # these as 3x BTC products, so BTC is the documented underlying, but
        # that is an ASSUMPTION and is flagged rather than assumed silently.
        assumed = False
        if not base:
            underlying = f"BTC{q}"
            assumed = True
        exists = underlying in universe
        rows.append({
            "symbol": s,
            "base": base or "(none)",
            "suffix": suf,
            "quote": q,
            "direction": "long" if suf in ("UP", "BULL") else "short",
            "underlying": underlying,
            "underlying_exists": exists,
            "underlying_assumed": assumed,
            # A name can PARSE as leveraged and still be an ordinary spot
            # asset: SYRUPUSDT splits to SYR+UP+USDT, but SYRUSDT does not
            # trade. The existence of the underlying is what separates a real
            # leveraged token from a coincidental spelling, so it -- not the
            # parse -- decides genuineness.
            "is_genuine": bool(exists),
        })
    df = pd.DataFrame(rows)
    logger.info("classify: %d name-parses, %d genuine (underlying present), "
                "out of %d archive symbols",
                len(df), int(df["is_genuine"].sum()) if len(df) else 0, len(universe))
    return df


def substring_false_positives(all_symbols, patterns,
                              p: LeveragedParams = LeveragedParams()) -> list:
    """Symbols the Cycle 60 substring rule kills that are NOT leveraged tokens.

    Two ways to be a false positive, and only checking the first misses the
    interesting ones:
      1. the name does not parse as <BASE><SUFFIX><QUOTE> at all;
      2. it parses, but the implied underlying does not trade -- SYRUPUSDT ->
         SYR+UP+USDT with no SYRUSDT, and JUPUSDT -> J+UP+USDT with no JUSDT.
    Both were silently deleted from the Cycle 60 universe, so both are reported.
    """
    universe = {x.upper() for x in all_symbols}
    out = []
    for s in sorted(universe):
        if not any(pat in s for pat in patterns):
            continue
        parsed = split_leveraged(s, p)
        if parsed is None:
            out.append({"symbol": s, "reason": "does not parse as a leveraged token"})
            continue
        base, suf, q = parsed
        und = f"{base}{q}" if base else f"BTC{q}"
        if und not in universe:
            out.append({"symbol": s,
                        "reason": f"parses as {base}+{suf}+{q} but {und} does "
                                  f"not trade -- ordinary spot asset"})
    return out


# ============================================== coverage against the archive ==

_PERIOD_RE = re.compile(r"^(\d{4})-(\d{2})$")


def periods_to_span(periods) -> dict:
    """Turn a list of YYYY-MM archive periods into a coverage summary."""
    ok = sorted(p for p in periods if _PERIOD_RE.match(p))
    if not ok:
        return {"n_months": 0, "first": None, "last": None, "contiguous": None}
    first, last = ok[0], ok[-1]
    y0, m0 = map(int, first.split("-"))
    y1, m1 = map(int, last.split("-"))
    expected = (y1 - y0) * 12 + (m1 - m0) + 1
    return {"n_months": len(ok), "first": first, "last": last,
            "expected_months": expected, "contiguous": len(ok) == expected}


def coverage(client, symbols, interval="1d", archive_last_period=None,
             p: LeveragedParams = LeveragedParams()) -> pd.DataFrame:
    """Per-symbol archive coverage and a delisted flag.

    Delisting is inferred from where the data STOPS relative to the archive
    frontier -- the same technique the taxonomy notes for scenario F3, and the
    only one available without a delisting-notice feed.
    """
    rows = []
    for s in symbols:
        try:
            periods = client.list_symbol_periods(s, interval)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] period listing failed: %s", s, e)
            periods = []
        span = periods_to_span(periods)
        delisted, months_behind = None, None
        if span["last"] and archive_last_period:
            y0, m0 = map(int, span["last"].split("-"))
            y1, m1 = map(int, archive_last_period.split("-"))
            months_behind = (y1 - y0) * 12 + (m1 - m0)
            delisted = months_behind >= p.delisted_gap_months
        rows.append({"symbol": s, **span,
                     "months_behind_frontier": months_behind,
                     "delisted": delisted})
        logger.debug("[%s] %s", s, span)
    return pd.DataFrame(rows)
