"""
engines/xsec_reversal/archive.py

Client for the Binance public data archive (data.binance.vision).

WHY THIS MODULE EXISTS (the survivorship-bias fix)
--------------------------------------------------
Binance's REST API (`/api/v3/exchangeInfo`) and the official helper script
`fetch-all-trading-pairs.sh` both return only CURRENTLY LISTED symbols. Any
universe built from those sources has already deleted every coin that was
delisted — and delisted coins are disproportionately the illiquid ones that
collapsed. Since this experiment's hypothesis is specifically about the
ILLIQUID TIER, that bias would manufacture the very edge we're testing for.

The archive bucket, by contrast, retains data for delisted pairs (Binance's own
README uses ADABKRW — delisted with the BKRW stablecoin — as its example
download). So we enumerate symbols from the S3 bucket listing, not the API.

ARCHIVE LAYOUT (verified against Binance docs)
    base:    https://data.binance.vision
    monthly: /data/spot/monthly/klines/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{YYYY}-{MM}.zip
    daily:   /data/spot/daily/klines/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{YYYY}-{MM}-{DD}.zip
    checksum: same path + ".CHECKSUM" (sha256)
    listing: /?prefix=data/spot/monthly/klines/&delimiter=/  -> XML CommonPrefixes

KLINE COLUMNS (12, headerless in older archives)
    open_time, open, high, low, close, volume, close_time, quote_asset_volume,
    num_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore

*** TIMESTAMP TRAP ***
Binance switched SPOT archive timestamps from MILLISECONDS to MICROSECONDS on
2025-01-01. A naive `pd.to_datetime(..., unit='ms')` silently produces dates in
the year ~55000 for 2025+ data. We detect the unit by MAGNITUDE per file rather
than trusting the date, which is robust to future changes:
    ms  ~ 1.7e12   us ~ 1.7e15   ->  threshold 1e14

NETWORK NOTE: this module performs HTTP requests. It is designed to run on
Jeff's machine, not in the Claude sandbox (which cannot reach data.binance.vision).
"""
from __future__ import annotations

import hashlib
import io
import logging
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional
from xml.etree import ElementTree

import numpy as np
import pandas as pd

logger = logging.getLogger("xsec.archive")

BASE_URL = "https://data.binance.vision"

# *** LISTING vs DOWNLOAD ARE DIFFERENT ENDPOINTS ***
# data.binance.vision is a CDN front. Requesting it with ?prefix=&delimiter=
# returns the JavaScript FILE-BROWSER HTML PAGE (HTTP 200, ~2.5 KB), not S3
# XML -- which fails with a cryptic "mismatched tag" ParseError. Bucket
# enumeration must go to the S3 ORIGIN. File downloads work fine on the CDN.
LISTING_URL_CANDIDATES = (
    "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision",
    "https://data.binance.vision",  # fallback only; currently serves HTML
)

_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# Binance tickers are ASCII uppercase alphanumeric (e.g. BTCUSDT, 1INCHUSDT).
# The bucket listing can contain stray prefixes that are NOT valid symbols --
# including names with non-ASCII bytes. Those would 404 on download and be
# silently miscounted as "symbol had no data in the window", so they are
# rejected at enumeration time and reported.
_VALID_SYMBOL_RE = re.compile(r"^[A-Z0-9]{3,30}$")


def is_valid_symbol(name: str) -> bool:
    """True if `name` looks like a real Binance ticker."""
    return bool(_VALID_SYMBOL_RE.match(name))

# Magnitude threshold separating epoch-milliseconds from epoch-microseconds.
_US_THRESHOLD = 1e14


@dataclass(frozen=True)
class KlineSchema:
    """On-disk layout of a Binance kline CSV. All 12 columns, in order."""
    columns: tuple = (
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "num_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore",
    )
    numeric: tuple = (
        "open", "high", "low", "close", "volume", "quote_asset_volume",
        "num_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    )


DEFAULT_KLINE_SCHEMA = KlineSchema()


def _detect_time_unit(sample_value: float) -> str:
    """Return 'us' or 'ms' from the magnitude of an epoch timestamp."""
    return "us" if float(sample_value) > _US_THRESHOLD else "ms"


def parse_kline_csv(
    raw: bytes | str,
    schema: KlineSchema = DEFAULT_KLINE_SCHEMA,
    *,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """Parse one Binance kline CSV (bytes or text) into a typed DataFrame.

    Handles: optional header row (newer archives add one), the ms/us timestamp
    switch, and a hard column-count guard so a layout change fails loudly
    instead of silently mis-mapping columns.
    """
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw
    if not text.strip():
        raise ValueError(f"[{symbol}] empty kline payload")

    first_line = text.lstrip().splitlines()[0]
    n_cols = len(first_line.split(","))
    expected = len(schema.columns)
    if n_cols != expected:
        raise ValueError(
            f"[{symbol}] kline column-count mismatch: file has {n_cols}, schema "
            f"expects {expected}. First line: {first_line[:200]}\n"
            f"  -> Binance changed the archive layout; update KlineSchema."
        )

    # Header sniff: if the first field is not parseable as a number, it's a header.
    has_header = False
    try:
        float(first_line.split(",")[0])
    except ValueError:
        has_header = True

    df = pd.read_csv(
        io.StringIO(text),
        header=0 if has_header else None,
        names=list(schema.columns),
        skiprows=0,
        dtype=str,
        index_col=False,
    )
    if has_header:
        # names= was applied over the header row's position; drop the echoed row
        # if pandas kept it (it does not when header=0 with names=, but guard).
        first_cell = str(df.iloc[0]["open_time"]) if len(df) else ""
        try:
            float(first_cell)
        except ValueError:
            df = df.iloc[1:].reset_index(drop=True)

    for col in schema.numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ot = pd.to_numeric(df["open_time"], errors="coerce")
    ot = ot.dropna()
    if ot.empty:
        raise ValueError(f"[{symbol}] no parseable open_time values")
    unit = _detect_time_unit(ot.iloc[0])
    df["open_time"] = pd.to_datetime(
        pd.to_numeric(df["open_time"], errors="coerce"), unit=unit, utc=True
    )
    df["close_time"] = pd.to_datetime(
        pd.to_numeric(df["close_time"], errors="coerce"), unit=unit, utc=True
    )

    df = df.dropna(subset=["open_time", "close"]).set_index("open_time").sort_index()
    df.index.name = "dt"
    keep = [c for c in schema.numeric] + ["close_time"]
    df = df[keep]
    if symbol:
        df["symbol"] = symbol
    logger.debug("[%s] parsed %d klines (%s) %s -> %s",
                 symbol, len(df), unit,
                 df.index.min() if len(df) else "n/a",
                 df.index.max() if len(df) else "n/a")
    return df


@dataclass
class ArchiveClient:
    """Downloads and caches Binance archive files.

    cache_dir layout mirrors the archive:
        {cache_dir}/spot/monthly/klines/{SYMBOL}/{interval}/{file}.zip
    """
    cache_dir: Path = Path("data/external/binance_archive")
    base_url: str = BASE_URL          # file downloads (CDN is fine and fast)
    listing_url: Optional[str] = None  # bucket enumeration; None = try candidates
    market: str = "spot"
    timeout: int = 60
    max_retries: int = 3
    retry_backoff: float = 2.0
    verify_checksum: bool = True
    polite_delay: float = 0.05  # seconds between requests

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.rejected_symbols: list = []
        self._session = None

    @property
    def session(self):
        """Shared requests.Session with a pooled adapter.

        Without this every call pays a fresh TLS handshake -- the smoke run
        logged "Starting new HTTPS connection" on EVERY request, which is the
        difference between a 3-hour and a 10-hour collect.
        """
        if self._session is None:
            import requests
            from requests.adapters import HTTPAdapter
            s = requests.Session()
            adapter = HTTPAdapter(pool_connections=16, pool_maxsize=32)
            s.mount("https://", adapter)
            s.headers.update({"User-Agent": "praxis-xsec/1.0"})
            self._session = s
        return self._session

    # ---------------- symbol enumeration (the survivorship fix) -------------
    @staticmethod
    def _looks_like_html(content: bytes) -> bool:
        head = content[:512].lstrip().lower()
        return head.startswith(b"<!doctype html") or head.startswith(b"<html")

    @staticmethod
    def parse_listing_page(content: bytes, prefix: str) -> tuple[list[str], Optional[str]]:
        """Parse one S3 ListBucketResult page -> (symbols, continuation_token).

        Split out from the network call so it is unit-testable without HTTP.
        """
        root = ElementTree.fromstring(content)
        syms = []
        for cp in root.findall(f"{_S3_NS}CommonPrefixes"):
            p = cp.find(f"{_S3_NS}Prefix")
            if p is None or not p.text:
                continue
            sym = p.text[len(prefix):].strip("/")
            if sym:
                syms.append(sym)
        truncated = root.find(f"{_S3_NS}IsTruncated")
        nxt = root.find(f"{_S3_NS}NextContinuationToken")
        token = (nxt.text if (truncated is not None and truncated.text == "true"
                              and nxt is not None) else None)
        return syms, token

    def list_all_symbols(
        self,
        interval_scope: str = "monthly",
        quote: Optional[str] = "USDT",
        max_pages: int = 500,
    ) -> list[str]:
        """Enumerate EVERY symbol in the archive, including delisted pairs.

        Walks the S3 bucket listing with delimiter='/' so each CommonPrefix is a
        symbol directory. This is the survivorship-bias-free source of truth; do
        NOT substitute exchangeInfo or fetch-all-trading-pairs.sh (verified: that
        script calls api.binance.com/api/v3/exchangeInfo, i.e. LIVE SYMBOLS ONLY).

        Tries LISTING_URL_CANDIDATES in order and rejects HTML responses, because
        the CDN answers listing queries with a JS file-browser page (HTTP 200)
        rather than XML.
        """
        import requests  # local import: module is importable without network deps

        prefix = f"data/{self.market}/{interval_scope}/klines/"
        candidates = ([self.listing_url] if self.listing_url
                      else list(LISTING_URL_CANDIDATES))
        errors = []

        for url in candidates:
            symbols: list[str] = []
            token: Optional[str] = None
            pages = 0
            try:
                while pages < max_pages:
                    params = {"prefix": prefix, "delimiter": "/", "list-type": "2"}
                    if token:
                        params["continuation-token"] = token
                    resp = self.session.get(url, params=params, timeout=self.timeout)
                    resp.raise_for_status()

                    if self._looks_like_html(resp.content):
                        raise ValueError(
                            f"listing endpoint returned HTML, not S3 XML "
                            f"({len(resp.content)} bytes) -- this host serves the "
                            f"JS file-browser UI for listing queries"
                        )

                    page_syms, token = self.parse_listing_page(resp.content, prefix)
                    symbols.extend(page_syms)
                    pages += 1
                    if not token:
                        break
                    time.sleep(self.polite_delay)

                if not symbols:
                    raise ValueError("listing returned zero symbols")

                symbols = sorted(set(symbols))
                n_raw = len(symbols)
                invalid = [s for s in symbols if not is_valid_symbol(s)]
                if invalid:
                    logger.warning(
                        "rejected %d non-ticker prefix(es) from the bucket "
                        "listing (not ASCII-uppercase-alphanumeric): %r",
                        len(invalid), invalid[:10])
                    self.rejected_symbols = invalid
                symbols = [s for s in symbols if is_valid_symbol(s)]

                if quote:
                    symbols = [s for s in symbols if s.endswith(quote)]
                logger.info("enumerated %d archive symbols from %s "
                            "(raw=%d, rejected=%d, quote=%s, pages=%d) "
                            "-- INCLUDES DELISTED",
                            len(symbols), url, n_raw, len(invalid), quote, pages)
                return symbols

            except Exception as e:  # noqa: BLE001
                logger.warning("listing endpoint %s failed: %s", url, e)
                errors.append(f"{url}: {e}")
                continue

        raise RuntimeError(
            "could not enumerate archive symbols from any listing endpoint.\n  "
            + "\n  ".join(errors)
            + "\n\nDO NOT fall back to api.binance.com/exchangeInfo -- that "
              "returns only CURRENTLY LISTED symbols and reintroduces the "
              "survivorship bias this experiment exists to avoid."
        )

    def list_symbol_periods(
        self,
        symbol: str,
        interval: str,
        scope: str = "monthly",
    ) -> list[str]:
        """List the periods ('YYYY-MM') that actually EXIST for a symbol.

        One listing request replaces N blind month probes. A coin listed in 2024
        would otherwise 404 for ~36 straight months, and with checksum requests
        that wasted traffic dominates the whole collect.

        Returns [] if the symbol/interval prefix holds nothing.
        """
        prefix = f"data/{self.market}/{scope}/klines/{symbol}/{interval}/"
        candidates = ([self.listing_url] if self.listing_url
                      else list(LISTING_URL_CANDIDATES))
        pat = re.compile(
            rf"{re.escape(symbol)}-{re.escape(interval)}-(\d{{4}}-\d{{2}})\.zip$")

        for url in candidates:
            periods, token, pages = [], None, 0
            try:
                while pages < 20:
                    params = {"prefix": prefix, "list-type": "2"}
                    if token:
                        params["continuation-token"] = token
                    resp = self.session.get(url, params=params, timeout=self.timeout)
                    resp.raise_for_status()
                    if self._looks_like_html(resp.content):
                        raise ValueError("listing endpoint returned HTML, not XML")

                    root = ElementTree.fromstring(resp.content)
                    for c in root.findall(f"{_S3_NS}Contents"):
                        k = c.find(f"{_S3_NS}Key")
                        if k is None or not k.text:
                            continue
                        m = pat.search(k.text)
                        if m:
                            periods.append(m.group(1))
                    trunc = root.find(f"{_S3_NS}IsTruncated")
                    nxt = root.find(f"{_S3_NS}NextContinuationToken")
                    pages += 1
                    if trunc is not None and trunc.text == "true" and nxt is not None:
                        token = nxt.text
                    else:
                        break
                return sorted(set(periods))
            except Exception as e:  # noqa: BLE001
                logger.debug("[%s] period listing via %s failed: %s", symbol, url, e)
                continue
        logger.warning("[%s] could not list periods; falling back to probing", symbol)
        return []

    # ---------------- download -------------------------------------------
    def _rel_path(self, symbol: str, interval: str, period: str,
                  scope: str = "monthly") -> str:
        fname = f"{symbol}-{interval}-{period}.zip"
        return f"data/{self.market}/{scope}/klines/{symbol}/{interval}/{fname}"

    def download_period(
        self,
        symbol: str,
        interval: str,
        period: str,
        scope: str = "monthly",
        force: bool = False,
    ) -> Optional[Path]:
        """Download one archive file. Returns local path, or None if 404
        (symbol did not trade that period -- an expected, informative outcome)."""
        import requests

        rel = self._rel_path(symbol, interval, period, scope)
        local = self.cache_dir / rel
        if local.exists() and not force:
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{rel}"

        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.get(url, timeout=self.timeout)
                if r.status_code == 404:
                    logger.debug("[%s] no archive for %s (404) -- not listed then",
                                 symbol, period)
                    return None
                r.raise_for_status()
                content = r.content

                if self.verify_checksum:
                    cs = self.session.get(url + ".CHECKSUM", timeout=self.timeout)
                    if cs.status_code == 200:
                        want = cs.text.split()[0].strip().lower()
                        got = hashlib.sha256(content).hexdigest()
                        if want != got:
                            raise ValueError(
                                f"[{symbol}] checksum mismatch for {period}: "
                                f"expected {want[:16]}..., got {got[:16]}..."
                            )
                local.write_bytes(content)
                time.sleep(self.polite_delay)
                return local
            except Exception as e:  # noqa: BLE001
                if attempt >= self.max_retries:
                    logger.error("[%s] download failed for %s after %d attempts: %s",
                                 symbol, period, attempt, e)
                    raise
                sleep_s = self.retry_backoff ** attempt
                logger.warning("[%s] %s attempt %d failed (%s); retrying in %.1fs",
                               symbol, period, attempt, e, sleep_s)
                time.sleep(sleep_s)
        return None

    def load_period(
        self,
        symbol: str,
        interval: str,
        period: str,
        scope: str = "monthly",
        schema: KlineSchema = DEFAULT_KLINE_SCHEMA,
    ) -> Optional[pd.DataFrame]:
        """Download (or read cache) and parse one period into a DataFrame."""
        path = self.download_period(symbol, interval, period, scope)
        if path is None:
            return None
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"[{symbol}] no CSV inside {path}")
            raw = z.read(names[0])
        return parse_kline_csv(raw, schema, symbol=symbol)

    def load_symbol_range(
        self,
        symbol: str,
        interval: str,
        periods: Iterable[str],
        scope: str = "monthly",
        use_listing: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Concatenate several periods for one symbol. Missing periods are
        skipped (and counted) rather than raising -- absence IS the listing
        history we need for point-in-time universe construction."""
        periods = list(periods)
        if use_listing:
            available = set(self.list_symbol_periods(symbol, interval, scope))
            if available:
                wanted = [p for p in periods if p in available]
                logger.debug("[%s] %d/%d requested periods exist in the archive",
                             symbol, len(wanted), len(periods))
                periods = wanted
            # empty `available` -> listing failed; fall through to probing

        if not periods:
            logger.info("[%s] no periods available in the requested window", symbol)
            return None

        frames, missing = [], 0
        for p in periods:
            try:
                df = self.load_period(symbol, interval, p, scope)
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s] period %s errored: %s", symbol, p, e)
                continue
            if df is None or df.empty:
                missing += 1
                continue
            frames.append(df)
        if not frames:
            logger.info("[%s] no data across %d periods", symbol, missing)
            return None
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="first")]
        logger.info("[%s] %d bars, %d periods missing", symbol, len(out), missing)
        return out


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings, e.g. month_range('2021-01','2021-03')."""
    s = pd.Timestamp(start + "-01" if len(start) == 7 else start)
    e = pd.Timestamp(end + "-01" if len(end) == 7 else end)
    out, cur = [], s.replace(day=1)
    while cur <= e:
        out.append(f"{cur.year:04d}-{cur.month:02d}")
        cur = (cur + pd.Timedelta(days=32)).replace(day=1)
    return out
