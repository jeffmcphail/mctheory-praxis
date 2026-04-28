"""
engines/prediction_scanner.py — Prediction Market Scanner

Discovers weather markets on Kalshi, fetches current prices, and
compares against our weather model probabilities to find edges.

Supports:
  - Kalshi (primary — CFTC regulated, fiat, weather markets via KXHIGH series)
  - Polymarket (future — USDC, broader weather coverage)

Architecture:
  Scanner polls markets → Forecaster computes model probability →
  Edge detector finds divergences → Signal generator produces trades →
  Kelly sizing determines position size

Usage:
    from engines.prediction_scanner import KalshiScanner
    scanner = KalshiScanner(api_key="...", private_key_path="...")
    markets = scanner.discover_weather_markets()
    signals = scanner.generate_signals(markets)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)

SIGNALS_FILE = "data/weather_signals.json"
TRADE_LOG_FILE = "data/weather_trades.json"


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class WeatherMarket:
    """A single weather market contract."""
    ticker: str                  # e.g., "KXHIGHNY-26APR05-T75"
    event_ticker: str            # e.g., "KXHIGHNY-26APR05"
    city: str                    # e.g., "new_york"
    target_date: str             # "2026-04-05"
    threshold_f: float           # 75.0 (degrees F)
    direction: str               # "above" or "below"
    # Market pricing
    yes_bid: float               # best bid for YES
    yes_ask: float               # best ask for YES
    no_bid: float                # best bid for NO
    no_ask: float                # best ask for NO
    last_price: float            # last trade price
    volume: int                  # total volume
    open_interest: int           # open interest
    # Status
    status: str                  # "open", "closed", "settled"
    close_time: str              # when market closes

    @property
    def mid_price(self) -> float:
        if self.yes_bid > 0 and self.yes_ask > 0:
            return (self.yes_bid + self.yes_ask) / 2
        return self.last_price

    @property
    def spread_pct(self) -> float:
        if self.yes_bid > 0 and self.yes_ask > 0:
            return (self.yes_ask - self.yes_bid) / self.yes_ask * 100
        return 0


@dataclass
class WeatherSignal:
    """A trading signal from comparing model vs market."""
    market: WeatherMarket
    model_probability: float     # our model's probability
    market_probability: float    # market implied probability (mid price)
    edge: float                  # model_prob - market_prob
    edge_pct: float              # edge as percentage
    direction: str               # "buy_yes" or "buy_no"
    confidence: str              # "high", "medium", "low"
    # Kelly sizing
    kelly_full: float            # full Kelly fraction
    kelly_fraction: float        # fractional Kelly (15%)
    position_size_usd: float     # dollar amount to trade
    expected_value: float        # expected profit per dollar
    # Metadata
    timestamp: str
    ensemble_mean: float
    ensemble_std: float
    n_members: int

    def summary(self) -> str:
        return (f"{'🎯' if self.is_tradeable else '─'} "
                f"{self.market.city} {self.market.target_date} "
                f"T>={self.market.threshold_f}°F: "
                f"model={self.model_probability:.0%} "
                f"market={self.market_probability:.0%} "
                f"edge={self.edge:+.0%} "
                f"[{self.confidence}] "
                f"→ {self.direction} ${self.position_size_usd:.2f}")

    @property
    def is_tradeable(self) -> bool:
        return abs(self.edge) >= 0.08 and self.confidence in ("high", "medium")


# ═════════════════════════════════════════════════════════════════════════════
# KALSHI SCANNER
# ═════════════════════════════════════════════════════════════════════════════

class KalshiScanner:
    """
    Discovers and monitors weather markets on Kalshi.

    Authentication: Kalshi uses RSA-PSS signed API keys.
    For initial testing, we can use the public API endpoints
    that don't require auth (market discovery, pricing).
    Trading requires authenticated API access.
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"

    # Map Kalshi ticker prefixes to our city names
    TICKER_TO_CITY = {
        "KXHIGHNY": "new_york",
        "KXHIGHCHI": "chicago",
        "KXHIGHMIA": "miami",
        "KXHIGHLAX": "los_angeles",
        "KXHIGHDEN": "denver",
    }

    def __init__(
        self,
        email: str = "",
        password: str = "",
        demo: bool = True,
        bankroll: float = 500.0,
        min_edge: float = 0.08,
        max_position: float = 100.0,
        kelly_fraction: float = 0.15,
    ):
        """
        Args:
            email: Kalshi account email (for authenticated endpoints)
            password: Kalshi account password
            demo: Use demo API (no real money)
            bankroll: Total capital for position sizing
            min_edge: Minimum edge to trigger signal (default 8%)
            max_position: Maximum position size per trade
            kelly_fraction: Fraction of full Kelly to use (default 15%)
        """
        self.email = email or os.getenv("KALSHI_EMAIL", "")
        self.password = password or os.getenv("KALSHI_PASSWORD", "")
        self.demo = demo
        self.bankroll = bankroll
        self.min_edge = min_edge
        self.max_position = max_position
        self.kelly_fraction = kelly_fraction

        self.base_url = self.DEMO_URL if demo else self.BASE_URL
        self._token = None
        self._token_expiry = None

    # ── Authentication ───────────────────────────────────────────────────

    def _login(self) -> bool:
        """Login to Kalshi API and get auth token."""
        if not self.email or not self.password:
            logger.warning("No Kalshi credentials configured")
            return False

        try:
            resp = requests.post(
                f"{self.base_url}/log-in",
                json={"email": self.email, "password": self.password},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data.get("token")
            # Token valid for ~24 hours
            self._token_expiry = datetime.now(timezone.utc) + timedelta(hours=20)
            logger.info("Kalshi login successful")
            return True
        except Exception as e:
            logger.error(f"Kalshi login failed: {e}")
            return False

    def _get_headers(self) -> dict:
        """Get auth headers, refreshing token if needed."""
        if self._token and self._token_expiry and \
                datetime.now(timezone.utc) < self._token_expiry:
            return {"Authorization": f"Bearer {self._token}"}

        if self._login():
            return {"Authorization": f"Bearer {self._token}"}
        return {}

    def _get(self, endpoint: str, params: dict = None,
             auth_required: bool = False) -> dict | None:
        """Make authenticated GET request to Kalshi API."""
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers() if auth_required else {}

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Kalshi API error: {endpoint} — {e}")
            return None

    # ── Market Discovery ─────────────────────────────────────────────────

    def discover_weather_markets(
        self,
        cities: list[str] | None = None,
    ) -> list[WeatherMarket]:
        """
        Discover active weather markets on Kalshi.

        Searches for KXHIGH series markets across configured cities.
        """
        if cities is None:
            cities = list(self.TICKER_TO_CITY.keys())

        all_markets = []

        for prefix in cities:
            city_name = self.TICKER_TO_CITY.get(prefix, "unknown")
            print(f"  Scanning {prefix} ({city_name})...", end=" ", flush=True)

            data = self._get("/markets", params={
                "series_ticker": prefix,
                "status": "open",
                "limit": 100,
            })

            if data is None:
                print("FAIL")
                continue

            markets = data.get("markets", [])
            count = 0

            for m in markets:
                try:
                    market = self._parse_market(m, city_name)
                    if market:
                        all_markets.append(market)
                        count += 1
                except Exception as e:
                    logger.debug(f"Failed to parse market: {e}")

            print(f"{count} markets")

        return all_markets

    def _parse_market(self, raw: dict, city_name: str) -> WeatherMarket | None:
        """Parse raw Kalshi API market data into WeatherMarket."""
        ticker = raw.get("ticker", "")
        event_ticker = raw.get("event_ticker", "")
        status = raw.get("status", "")

        if status != "open":
            return None

        # Extract threshold from ticker (e.g., "KXHIGHNY-26APR05-T75")
        # Format varies — try to extract temperature
        threshold = self._extract_threshold(ticker, raw)
        if threshold is None:
            return None

        # Extract date
        target_date = self._extract_date(ticker, raw)
        if target_date is None:
            return None

        return WeatherMarket(
            ticker=ticker,
            event_ticker=event_ticker,
            city=city_name,
            target_date=target_date,
            threshold_f=threshold,
            direction="above",  # KXHIGH = high temp above threshold
            yes_bid=float(raw.get("yes_bid", 0)) / 100,
            yes_ask=float(raw.get("yes_ask", 0)) / 100,
            no_bid=float(raw.get("no_bid", 0)) / 100,
            no_ask=float(raw.get("no_ask", 0)) / 100,
            last_price=float(raw.get("last_price", 0)) / 100,
            volume=int(raw.get("volume", 0)),
            open_interest=int(raw.get("open_interest", 0)),
            status=status,
            close_time=raw.get("close_time", ""),
        )

    def _extract_threshold(self, ticker: str, raw: dict) -> float | None:
        """Extract temperature threshold from ticker or market data."""
        # Try subtitle/title first
        subtitle = raw.get("subtitle", "") or raw.get("title", "")
        # Look for patterns like "75°" or "above 75" or "75 or above"
        import re
        match = re.search(r'(\d+)\s*°', subtitle)
        if match:
            return float(match.group(1))

        # Try ticker pattern: ...-T75 or ...-B75
        match = re.search(r'-[TB](\d+)', ticker)
        if match:
            return float(match.group(1))

        # Try floor/ceiling from API
        floor = raw.get("floor_strike")
        ceiling = raw.get("cap_strike")
        if floor is not None:
            return float(floor) / 100  # Kalshi uses cents
        if ceiling is not None:
            return float(ceiling) / 100

        return None

    def _extract_date(self, ticker: str, raw: dict) -> str | None:
        """Extract target date from ticker or market data."""
        # Try expiration_time from API
        exp = raw.get("expiration_time", "") or raw.get("close_time", "")
        if exp:
            try:
                dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
                return dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass

        # Try ticker pattern: ...-26APR05-...
        import re
        match = re.search(r'-(\d{2})([A-Z]{3})(\d{2})-', ticker)
        if match:
            year = 2000 + int(match.group(1))
            month_map = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
                         "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
                         "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
            month = month_map.get(match.group(2), 0)
            day = int(match.group(3))
            if month > 0:
                return f"{year}-{month:02d}-{day:02d}"

        return None

    # ── Signal Generation ────────────────────────────────────────────────

    def generate_signals(
        self,
        markets: list[WeatherMarket],
        forecaster=None,
    ) -> list[WeatherSignal]:
        """
        Compare market prices against model probabilities
        and generate trading signals.

        Args:
            markets: List of discovered weather markets
            forecaster: WeatherForecaster instance

        Returns:
            List of WeatherSignal sorted by edge magnitude
        """
        if forecaster is None:
            from engines.weather_forecaster import WeatherForecaster
            forecaster = WeatherForecaster()

        signals = []
        now = datetime.now(timezone.utc)

        for market in markets:
            # Get model probability
            result = forecaster.get_threshold_probability(
                market.city,
                market.threshold_f,
                market.target_date,
                direction="above",
            )

            if result is None:
                continue

            model_prob = result.model_probability
            market_prob = market.mid_price

            if market_prob <= 0.01 or market_prob >= 0.99:
                continue  # Skip extreme prices — no room for edge

            edge = model_prob - market_prob

            # Determine trade direction
            if edge > 0:
                direction = "buy_yes"
                # Buy YES — model says more likely than market thinks
                buy_price = market.yes_ask if market.yes_ask > 0 else market_prob
                edge_after_spread = model_prob - buy_price
            else:
                direction = "buy_no"
                # Buy NO — model says less likely than market thinks
                buy_price = market.no_ask if market.no_ask > 0 else (1 - market_prob)
                edge_after_spread = (1 - model_prob) - buy_price

            # Kelly criterion
            if direction == "buy_yes":
                win_prob = model_prob
                odds = (1 - buy_price) / buy_price if buy_price > 0 else 0
            else:
                win_prob = 1 - model_prob
                odds = (1 - buy_price) / buy_price if buy_price > 0 else 0

            if odds <= 0:
                continue

            kelly_full = (win_prob * odds - (1 - win_prob)) / odds
            kelly_frac = max(0, kelly_full * self.kelly_fraction)
            position_size = min(kelly_frac * self.bankroll, self.max_position)

            # Compute expected value
            ev_per_dollar = edge_after_spread

            signal = WeatherSignal(
                market=market,
                model_probability=model_prob,
                market_probability=market_prob,
                edge=edge,
                edge_pct=edge * 100,
                direction=direction,
                confidence=result.confidence,
                kelly_full=kelly_full,
                kelly_fraction=kelly_frac,
                position_size_usd=position_size,
                expected_value=ev_per_dollar * position_size,
                timestamp=now.isoformat(),
                ensemble_mean=result.ensemble_mean,
                ensemble_std=result.ensemble_std,
                n_members=result.n_members,
            )
            signals.append(signal)

        # Sort by absolute edge (biggest opportunities first)
        signals.sort(key=lambda s: abs(s.edge), reverse=True)
        return signals

    # ── Signal Persistence ───────────────────────────────────────────────

    def save_signals(self, signals: list[WeatherSignal],
                     path: str = SIGNALS_FILE):
        """Save signals to JSON for review."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for s in signals:
            d = {
                "ticker": s.market.ticker,
                "city": s.market.city,
                "date": s.market.target_date,
                "threshold": s.market.threshold_f,
                "model_prob": s.model_probability,
                "market_prob": s.market_probability,
                "edge": s.edge,
                "direction": s.direction,
                "confidence": s.confidence,
                "position_size": s.position_size_usd,
                "expected_value": s.expected_value,
                "timestamp": s.timestamp,
                "ensemble_mean": s.ensemble_mean,
                "tradeable": s.is_tradeable,
            }
            data.append(d)

        with open(p, "w") as f:
            json.dump(data, f, indent=2)
