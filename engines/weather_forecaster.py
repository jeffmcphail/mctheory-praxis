"""
engines/weather_forecaster.py — Weather Forecast Probability Engine

Fetches GFS ensemble forecasts from Open-Meteo (free, no auth) and
computes probability distributions for temperature thresholds. These
probabilities are compared against prediction market prices to find
exploitable mispricings.

Data source: Open-Meteo GFS ensemble API (31 members)
  - Free, no API key required
  - Updated 4x daily (00Z, 06Z, 12Z, 18Z)
  - Hourly resolution, up to 16 days ahead
  - 31 ensemble members for uncertainty quantification

Usage:
    from engines.weather_forecaster import WeatherForecaster
    forecaster = WeatherForecaster()
    prob = forecaster.get_threshold_probability("new_york", 75.0, "2026-04-05")
    # Returns: 0.84 (84% of ensemble members predict >= 75°F)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# CITY CONFIGURATIONS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CityConfig:
    """Configuration for a city's weather market."""
    name: str
    display_name: str
    latitude: float
    longitude: float
    # Kalshi ticker prefix for this city's high temp markets
    kalshi_ticker_prefix: str
    # Resolution weather station (for reference)
    resolution_station: str
    # Timezone offset from UTC for "today" determination
    tz_offset_hours: int

# Major cities with Kalshi weather markets
CITIES: dict[str, CityConfig] = {
    "new_york": CityConfig(
        name="new_york", display_name="New York City",
        latitude=40.7128, longitude=-74.0060,
        kalshi_ticker_prefix="KXHIGHNY",
        resolution_station="KNYC (Central Park)",
        tz_offset_hours=-4,  # EDT
    ),
    "chicago": CityConfig(
        name="chicago", display_name="Chicago",
        latitude=41.8781, longitude=-87.6298,
        kalshi_ticker_prefix="KXHIGHCHI",
        resolution_station="KORD (O'Hare)",
        tz_offset_hours=-5,  # CDT
    ),
    "miami": CityConfig(
        name="miami", display_name="Miami",
        latitude=25.7617, longitude=-80.1918,
        kalshi_ticker_prefix="KXHIGHMIA",
        resolution_station="KMIA",
        tz_offset_hours=-4,
    ),
    "los_angeles": CityConfig(
        name="los_angeles", display_name="Los Angeles",
        latitude=34.0522, longitude=-118.2437,
        kalshi_ticker_prefix="KXHIGHLAX",
        resolution_station="KLAX",
        tz_offset_hours=-7,  # PDT
    ),
    "denver": CityConfig(
        name="denver", display_name="Denver",
        latitude=39.7392, longitude=-104.9903,
        kalshi_ticker_prefix="KXHIGHDEN",
        resolution_station="KDEN",
        tz_offset_hours=-6,  # MDT
    ),
    "london": CityConfig(
        name="london", display_name="London",
        latitude=51.5074, longitude=-0.1278,
        kalshi_ticker_prefix="",  # Polymarket only
        resolution_station="EGLC (London City Airport)",
        tz_offset_hours=1,  # BST
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# FORECAST RESULT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ForecastResult:
    """Result of a weather probability computation."""
    city: str
    target_date: str
    metric: str                 # "high_temp", "low_temp", "precip"
    threshold: float            # Temperature threshold in °F
    direction: str              # "above" or "below"

    # Ensemble statistics
    ensemble_values: list[float]    # All 31 member forecasts (°F)
    model_probability: float        # Fraction of members above/below threshold
    ensemble_mean: float
    ensemble_median: float
    ensemble_std: float
    ensemble_min: float
    ensemble_max: float
    n_members: int

    # Confidence assessment
    confidence: str             # "high", "medium", "low"
    agreement_pct: float        # How one-sided the ensemble is

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence == "high"

    def summary(self) -> str:
        return (f"{self.city} {self.target_date}: "
                f"P(high {'≥' if self.direction == 'above' else '<'} "
                f"{self.threshold}°F) = {self.model_probability:.0%} "
                f"[{self.confidence}] "
                f"(mean={self.ensemble_mean:.1f}, "
                f"range={self.ensemble_min:.1f}-{self.ensemble_max:.1f})")


# ═════════════════════════════════════════════════════════════════════════════
# FORECASTER
# ═════════════════════════════════════════════════════════════════════════════

class WeatherForecaster:
    """
    GFS ensemble-based weather probability forecaster.

    Uses Open-Meteo's free API to fetch 31-member GFS ensemble forecasts,
    then computes the probability that the daily high temperature will
    exceed various thresholds. These probabilities are compared against
    prediction market prices to find edges.
    """

    BASE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}
        self._cache_time: dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=30)

    def _fetch_ensemble(self, city: CityConfig,
                        forecast_days: int = 7) -> pd.DataFrame | None:
        """
        Fetch GFS ensemble forecast from Open-Meteo.

        Returns DataFrame with columns for each ensemble member's
        temperature forecast, indexed by datetime.
        """
        cache_key = f"{city.name}_{forecast_days}"
        now = datetime.now(timezone.utc)

        # Check cache
        if (cache_key in self._cache
                and now - self._cache_time.get(cache_key, now) < self._cache_ttl):
            return self._cache[cache_key]

        params = {
            "latitude": city.latitude,
            "longitude": city.longitude,
            "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit",
            "models": "gfs_seamless",
            "forecast_days": forecast_days,
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch forecast for {city.name}: {e}")
            return None

        # Parse ensemble members
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        if not times:
            logger.warning(f"No forecast data for {city.name}")
            return None

        # Find all ensemble member columns
        members = {}
        for key, values in hourly.items():
            if key.startswith("temperature_2m"):
                member_name = key  # e.g., "temperature_2m", "temperature_2m_member01", etc.
                members[member_name] = values

        if not members:
            logger.warning(f"No ensemble members found for {city.name}")
            return None

        df = pd.DataFrame(members, index=pd.to_datetime(times))
        df.index.name = "time"

        # Cache
        self._cache[cache_key] = df
        self._cache_time[cache_key] = now

        logger.info(f"Fetched {len(df)} hours × {len(members)} members "
                     f"for {city.display_name}")
        return df

    def _extract_daily_highs(self, df: pd.DataFrame,
                              target_date: str,
                              city: CityConfig) -> np.ndarray | None:
        """
        Extract daily high temperature for each ensemble member.

        Filters to the target date in local time, then takes the max
        temperature across all hours for each member.
        """
        # Convert target date to datetime range in local time
        try:
            date = pd.Timestamp(target_date)
        except ValueError:
            logger.error(f"Invalid date: {target_date}")
            return None

        # Filter to daytime hours (roughly 6am-9pm local) for daily high
        # Adjust for timezone
        utc_offset = timedelta(hours=city.tz_offset_hours)
        local_start = date.replace(hour=6) - utc_offset
        local_end = date.replace(hour=21) - utc_offset

        mask = (df.index >= local_start) & (df.index <= local_end)
        day_data = df[mask]

        if day_data.empty:
            logger.warning(f"No data for {target_date} in {city.name}")
            return None

        # Get daily high for each member
        highs = day_data.max().values
        return highs

    def get_threshold_probability(
        self,
        city_name: str,
        threshold_f: float,
        target_date: str,
        direction: str = "above",
    ) -> ForecastResult | None:
        """
        Compute probability that daily high exceeds a threshold.

        Args:
            city_name: Key from CITIES dict (e.g., "new_york")
            threshold_f: Temperature threshold in Fahrenheit
            target_date: Date string "YYYY-MM-DD"
            direction: "above" (P(high >= threshold)) or "below" (P(high < threshold))

        Returns:
            ForecastResult with probability and ensemble statistics
        """
        city = CITIES.get(city_name)
        if city is None:
            logger.error(f"Unknown city: {city_name}")
            return None

        df = self._fetch_ensemble(city)
        if df is None:
            return None

        highs = self._extract_daily_highs(df, target_date, city)
        if highs is None or len(highs) == 0:
            return None

        # Compute probability
        n = len(highs)
        if direction == "above":
            n_above = np.sum(highs >= threshold_f)
            probability = n_above / n
        else:
            n_below = np.sum(highs < threshold_f)
            probability = n_below / n

        # Confidence based on ensemble agreement
        agreement = max(probability, 1 - probability)
        if agreement >= 0.85:
            confidence = "high"
        elif agreement >= 0.65:
            confidence = "medium"
        else:
            confidence = "low"

        return ForecastResult(
            city=city_name,
            target_date=target_date,
            metric="high_temp",
            threshold=threshold_f,
            direction=direction,
            ensemble_values=highs.tolist(),
            model_probability=probability,
            ensemble_mean=float(np.mean(highs)),
            ensemble_median=float(np.median(highs)),
            ensemble_std=float(np.std(highs)),
            ensemble_min=float(np.min(highs)),
            ensemble_max=float(np.max(highs)),
            n_members=n,
            confidence=confidence,
            agreement_pct=agreement,
        )

    def scan_thresholds(
        self,
        city_name: str,
        target_date: str,
        thresholds: list[float] | None = None,
    ) -> list[ForecastResult]:
        """
        Compute probabilities for multiple temperature thresholds.

        If thresholds not provided, auto-generates based on ensemble range.
        """
        city = CITIES.get(city_name)
        if city is None:
            return []

        df = self._fetch_ensemble(city)
        if df is None:
            return []

        highs = self._extract_daily_highs(df, target_date, city)
        if highs is None or len(highs) == 0:
            return []

        # Auto-generate thresholds based on ensemble range
        if thresholds is None:
            low = int(np.floor(np.min(highs) / 5) * 5) - 5
            high = int(np.ceil(np.max(highs) / 5) * 5) + 5
            thresholds = list(range(low, high + 1, 5))

        results = []
        for t in thresholds:
            result = self.get_threshold_probability(
                city_name, t, target_date, direction="above"
            )
            if result:
                results.append(result)

        return results

    def scan_all_cities(
        self,
        target_date: str,
    ) -> dict[str, list[ForecastResult]]:
        """Scan all configured cities for a target date."""
        results = {}
        for city_name in CITIES:
            city_results = self.scan_thresholds(city_name, target_date)
            if city_results:
                results[city_name] = city_results
        return results
