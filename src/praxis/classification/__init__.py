"""
Classification Mappings (Phase 4.13, §15.5).

Static lookup tables mapping external vendor classifications to
core platform classifications. Covers:
- CFI (Classification of Financial Instruments, ISO 10962)
- S&P credit ratings → core credit tier
- Moody's credit ratings → core credit tier
- MSCI market classification → core market tier
- MSCI ESG ratings → core ESG tier
- GICS sector → core asset_class + instrument_type (supplemental to existing)

Usage:
    from praxis.classification.mappings import (
        cfi_to_core, sp_rating_to_tier, moody_to_tier, msci_market_to_tier
    )
    core = cfi_to_core("ESVUFR")
    tier = sp_rating_to_tier("AA+")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ═══════════════════════════════════════════════════════════════════
#  Core Classification Types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CoreClassification:
    """Minimal core classification for platform routing."""
    asset_class: str          # EQUITY, FIXED_INCOME, FX, CRYPTO, COMMODITY, DERIVATIVE
    instrument_type: str      # COMMON_STOCK, ETF, CORPORATE_BOND, FUTURE, etc.
    settlement_type: str = "" # CASH, PHYSICAL, ON_CHAIN


@dataclass
class CreditTier:
    """Normalized credit rating tier."""
    tier: str                 # INVESTMENT_GRADE, HIGH_YIELD, DISTRESSED, UNRATED
    numeric_rank: int = 0     # 1 (AAA) to 22 (D)
    is_investment_grade: bool = True


@dataclass
class MarketTier:
    """MSCI market classification tier."""
    tier: str                 # DEVELOPED, EMERGING, FRONTIER, STANDALONE
    region: str = ""          # AMERICAS, EMEA, ASIA_PACIFIC


@dataclass
class ESGTier:
    """MSCI ESG classification."""
    rating: str               # AAA, AA, A, BBB, BB, B, CCC
    tier: str                 # LEADER, AVERAGE, LAGGARD
    numeric: float = 0.0      # 0-10 scale


# ═══════════════════════════════════════════════════════════════════
#  CFI Mappings (ISO 10962)
# ═══════════════════════════════════════════════════════════════════

# CFI code structure: 6 characters
# Char 1: Category (E=equity, D=debt, R=entitlement, O=option, F=future, etc.)
# Char 2: Group within category
# Chars 3-6: Attributes

_CFI_CATEGORY_MAP: dict[str, CoreClassification] = {
    # Equities
    "ES": CoreClassification("EQUITY", "COMMON_STOCK"),
    "EP": CoreClassification("EQUITY", "PREFERRED_STOCK"),
    "EU": CoreClassification("EQUITY", "UNIT_TRUST"),
    "EC": CoreClassification("EQUITY", "CLOSED_END_FUND"),
    "EF": CoreClassification("EQUITY", "ETF"),
    "EM": CoreClassification("EQUITY", "MISC_EQUITY"),

    # Debt
    "DB": CoreClassification("FIXED_INCOME", "CORPORATE_BOND"),
    "DC": CoreClassification("FIXED_INCOME", "CONVERTIBLE_BOND"),
    "DG": CoreClassification("FIXED_INCOME", "GOVERNMENT_BOND"),
    "DT": CoreClassification("FIXED_INCOME", "TREASURY"),
    "DN": CoreClassification("FIXED_INCOME", "MUNICIPAL_BOND"),
    "DM": CoreClassification("FIXED_INCOME", "MONEY_MARKET"),
    "DW": CoreClassification("FIXED_INCOME", "COVERED_BOND"),
    "DA": CoreClassification("FIXED_INCOME", "ABS"),

    # Entitlements (Rights, Warrants)
    "RA": CoreClassification("DERIVATIVE", "RIGHT"),
    "RW": CoreClassification("DERIVATIVE", "WARRANT"),

    # Options
    "OC": CoreClassification("DERIVATIVE", "CALL_OPTION"),
    "OP": CoreClassification("DERIVATIVE", "PUT_OPTION"),

    # Futures
    "FF": CoreClassification("DERIVATIVE", "FINANCIAL_FUTURE"),
    "FC": CoreClassification("COMMODITY", "COMMODITY_FUTURE", "PHYSICAL"),

    # Collective Investment
    "CI": CoreClassification("EQUITY", "MUTUAL_FUND"),
    "CE": CoreClassification("EQUITY", "ETF"),

    # Misc
    "MC": CoreClassification("FX", "SPOT"),
    "MR": CoreClassification("FX", "FX_FORWARD"),
}


def cfi_to_core(cfi_code: str) -> CoreClassification | None:
    """
    Map a CFI code to core classification.

    Uses first 2 characters for category+group lookup.
    Falls back to first character for broad category.
    """
    if not cfi_code or len(cfi_code) < 2:
        return None

    key = cfi_code[:2].upper()
    result = _CFI_CATEGORY_MAP.get(key)

    if result is None:
        # Fallback to first character
        char1 = cfi_code[0].upper()
        fallbacks = {
            "E": CoreClassification("EQUITY", "COMMON_STOCK"),
            "D": CoreClassification("FIXED_INCOME", "CORPORATE_BOND"),
            "R": CoreClassification("DERIVATIVE", "RIGHT"),
            "O": CoreClassification("DERIVATIVE", "OPTION"),
            "F": CoreClassification("DERIVATIVE", "FUTURE"),
            "C": CoreClassification("EQUITY", "COLLECTIVE_INVESTMENT"),
            "M": CoreClassification("FX", "MISC"),
        }
        result = fallbacks.get(char1)

    return result


# ═══════════════════════════════════════════════════════════════════
#  S&P Credit Rating Mappings
# ═══════════════════════════════════════════════════════════════════

_SP_RATINGS: dict[str, CreditTier] = {
    "AAA":  CreditTier("INVESTMENT_GRADE", 1, True),
    "AA+":  CreditTier("INVESTMENT_GRADE", 2, True),
    "AA":   CreditTier("INVESTMENT_GRADE", 3, True),
    "AA-":  CreditTier("INVESTMENT_GRADE", 4, True),
    "A+":   CreditTier("INVESTMENT_GRADE", 5, True),
    "A":    CreditTier("INVESTMENT_GRADE", 6, True),
    "A-":   CreditTier("INVESTMENT_GRADE", 7, True),
    "BBB+": CreditTier("INVESTMENT_GRADE", 8, True),
    "BBB":  CreditTier("INVESTMENT_GRADE", 9, True),
    "BBB-": CreditTier("INVESTMENT_GRADE", 10, True),
    "BB+":  CreditTier("HIGH_YIELD", 11, False),
    "BB":   CreditTier("HIGH_YIELD", 12, False),
    "BB-":  CreditTier("HIGH_YIELD", 13, False),
    "B+":   CreditTier("HIGH_YIELD", 14, False),
    "B":    CreditTier("HIGH_YIELD", 15, False),
    "B-":   CreditTier("HIGH_YIELD", 16, False),
    "CCC+": CreditTier("DISTRESSED", 17, False),
    "CCC":  CreditTier("DISTRESSED", 18, False),
    "CCC-": CreditTier("DISTRESSED", 19, False),
    "CC":   CreditTier("DISTRESSED", 20, False),
    "C":    CreditTier("DISTRESSED", 21, False),
    "D":    CreditTier("DISTRESSED", 22, False),
}


def sp_rating_to_tier(rating: str) -> CreditTier | None:
    """Map S&P rating string to CreditTier."""
    return _SP_RATINGS.get(rating.upper().strip())


# ═══════════════════════════════════════════════════════════════════
#  Moody's Credit Rating Mappings
# ═══════════════════════════════════════════════════════════════════

_MOODY_RATINGS: dict[str, CreditTier] = {
    "Aaa":  CreditTier("INVESTMENT_GRADE", 1, True),
    "Aa1":  CreditTier("INVESTMENT_GRADE", 2, True),
    "Aa2":  CreditTier("INVESTMENT_GRADE", 3, True),
    "Aa3":  CreditTier("INVESTMENT_GRADE", 4, True),
    "A1":   CreditTier("INVESTMENT_GRADE", 5, True),
    "A2":   CreditTier("INVESTMENT_GRADE", 6, True),
    "A3":   CreditTier("INVESTMENT_GRADE", 7, True),
    "Baa1": CreditTier("INVESTMENT_GRADE", 8, True),
    "Baa2": CreditTier("INVESTMENT_GRADE", 9, True),
    "Baa3": CreditTier("INVESTMENT_GRADE", 10, True),
    "Ba1":  CreditTier("HIGH_YIELD", 11, False),
    "Ba2":  CreditTier("HIGH_YIELD", 12, False),
    "Ba3":  CreditTier("HIGH_YIELD", 13, False),
    "B1":   CreditTier("HIGH_YIELD", 14, False),
    "B2":   CreditTier("HIGH_YIELD", 15, False),
    "B3":   CreditTier("HIGH_YIELD", 16, False),
    "Caa1": CreditTier("DISTRESSED", 17, False),
    "Caa2": CreditTier("DISTRESSED", 18, False),
    "Caa3": CreditTier("DISTRESSED", 19, False),
    "Ca":   CreditTier("DISTRESSED", 20, False),
    "C":    CreditTier("DISTRESSED", 21, False),
}


def moody_to_tier(rating: str) -> CreditTier | None:
    """Map Moody's rating string to CreditTier."""
    return _MOODY_RATINGS.get(rating.strip())


def sp_to_moody_equivalent(sp_rating: str) -> str | None:
    """Map S&P rating to equivalent Moody's rating."""
    sp_tier = sp_rating_to_tier(sp_rating)
    if sp_tier is None:
        return None
    # Find Moody's with same numeric rank
    for moody_str, moody_tier in _MOODY_RATINGS.items():
        if moody_tier.numeric_rank == sp_tier.numeric_rank:
            return moody_str
    return None


# ═══════════════════════════════════════════════════════════════════
#  MSCI Market Classification
# ═══════════════════════════════════════════════════════════════════

_MSCI_MARKETS: dict[str, MarketTier] = {
    # Developed Markets
    "US": MarketTier("DEVELOPED", "AMERICAS"),
    "CA": MarketTier("DEVELOPED", "AMERICAS"),
    "GB": MarketTier("DEVELOPED", "EMEA"),
    "FR": MarketTier("DEVELOPED", "EMEA"),
    "DE": MarketTier("DEVELOPED", "EMEA"),
    "CH": MarketTier("DEVELOPED", "EMEA"),
    "NL": MarketTier("DEVELOPED", "EMEA"),
    "SE": MarketTier("DEVELOPED", "EMEA"),
    "NO": MarketTier("DEVELOPED", "EMEA"),
    "DK": MarketTier("DEVELOPED", "EMEA"),
    "FI": MarketTier("DEVELOPED", "EMEA"),
    "IE": MarketTier("DEVELOPED", "EMEA"),
    "ES": MarketTier("DEVELOPED", "EMEA"),
    "IT": MarketTier("DEVELOPED", "EMEA"),
    "PT": MarketTier("DEVELOPED", "EMEA"),
    "AT": MarketTier("DEVELOPED", "EMEA"),
    "BE": MarketTier("DEVELOPED", "EMEA"),
    "IL": MarketTier("DEVELOPED", "EMEA"),
    "JP": MarketTier("DEVELOPED", "ASIA_PACIFIC"),
    "AU": MarketTier("DEVELOPED", "ASIA_PACIFIC"),
    "NZ": MarketTier("DEVELOPED", "ASIA_PACIFIC"),
    "HK": MarketTier("DEVELOPED", "ASIA_PACIFIC"),
    "SG": MarketTier("DEVELOPED", "ASIA_PACIFIC"),

    # Emerging Markets
    "CN": MarketTier("EMERGING", "ASIA_PACIFIC"),
    "IN": MarketTier("EMERGING", "ASIA_PACIFIC"),
    "KR": MarketTier("EMERGING", "ASIA_PACIFIC"),
    "TW": MarketTier("EMERGING", "ASIA_PACIFIC"),
    "TH": MarketTier("EMERGING", "ASIA_PACIFIC"),
    "MY": MarketTier("EMERGING", "ASIA_PACIFIC"),
    "ID": MarketTier("EMERGING", "ASIA_PACIFIC"),
    "PH": MarketTier("EMERGING", "ASIA_PACIFIC"),
    "BR": MarketTier("EMERGING", "AMERICAS"),
    "MX": MarketTier("EMERGING", "AMERICAS"),
    "CL": MarketTier("EMERGING", "AMERICAS"),
    "CO": MarketTier("EMERGING", "AMERICAS"),
    "PE": MarketTier("EMERGING", "AMERICAS"),
    "ZA": MarketTier("EMERGING", "EMEA"),
    "SA": MarketTier("EMERGING", "EMEA"),
    "AE": MarketTier("EMERGING", "EMEA"),
    "QA": MarketTier("EMERGING", "EMEA"),
    "KW": MarketTier("EMERGING", "EMEA"),
    "TR": MarketTier("EMERGING", "EMEA"),
    "PL": MarketTier("EMERGING", "EMEA"),
    "CZ": MarketTier("EMERGING", "EMEA"),
    "HU": MarketTier("EMERGING", "EMEA"),
    "EG": MarketTier("EMERGING", "EMEA"),
    "GR": MarketTier("EMERGING", "EMEA"),

    # Frontier Markets
    "VN": MarketTier("FRONTIER", "ASIA_PACIFIC"),
    "BD": MarketTier("FRONTIER", "ASIA_PACIFIC"),
    "LK": MarketTier("FRONTIER", "ASIA_PACIFIC"),
    "KE": MarketTier("FRONTIER", "EMEA"),
    "NG": MarketTier("FRONTIER", "EMEA"),
    "RO": MarketTier("FRONTIER", "EMEA"),
    "HR": MarketTier("FRONTIER", "EMEA"),
    "RS": MarketTier("FRONTIER", "EMEA"),
    "BH": MarketTier("FRONTIER", "EMEA"),
    "OM": MarketTier("FRONTIER", "EMEA"),
    "JO": MarketTier("FRONTIER", "EMEA"),
}


def msci_market_to_tier(country_code: str) -> MarketTier | None:
    """Map ISO country code to MSCI market classification tier."""
    return _MSCI_MARKETS.get(country_code.upper().strip())


def get_markets_by_tier(tier: str) -> list[str]:
    """Get all country codes for a given tier (DEVELOPED, EMERGING, FRONTIER)."""
    return [k for k, v in _MSCI_MARKETS.items() if v.tier == tier.upper()]


def get_markets_by_region(region: str) -> list[str]:
    """Get all country codes for a region (AMERICAS, EMEA, ASIA_PACIFIC)."""
    return [k for k, v in _MSCI_MARKETS.items() if v.region == region.upper()]


# ═══════════════════════════════════════════════════════════════════
#  MSCI ESG Rating
# ═══════════════════════════════════════════════════════════════════

_ESG_RATINGS: dict[str, ESGTier] = {
    "AAA": ESGTier("AAA", "LEADER", 10.0),
    "AA":  ESGTier("AA", "LEADER", 8.6),
    "A":   ESGTier("A", "AVERAGE", 7.1),
    "BBB": ESGTier("BBB", "AVERAGE", 5.7),
    "BB":  ESGTier("BB", "LAGGARD", 4.3),
    "B":   ESGTier("B", "LAGGARD", 2.9),
    "CCC": ESGTier("CCC", "LAGGARD", 1.4),
}


def msci_esg_to_tier(rating: str) -> ESGTier | None:
    """Map MSCI ESG rating to ESGTier."""
    return _ESG_RATINGS.get(rating.upper().strip())


# ═══════════════════════════════════════════════════════════════════
#  Cross-System Rating Comparison
# ═══════════════════════════════════════════════════════════════════

def compare_credit_ratings(
    sp_rating: str | None = None,
    moody_rating: str | None = None,
) -> dict[str, Any]:
    """
    Compare credit ratings across agencies.

    Returns alignment info: whether they agree on investment grade
    status and how many notches apart they are.
    """
    sp_tier = sp_rating_to_tier(sp_rating) if sp_rating else None
    moody_tier = moody_to_tier(moody_rating) if moody_rating else None

    result: dict[str, Any] = {
        "sp_rating": sp_rating,
        "moody_rating": moody_rating,
        "sp_tier": sp_tier.tier if sp_tier else None,
        "moody_tier": moody_tier.tier if moody_tier else None,
    }

    if sp_tier and moody_tier:
        result["notch_difference"] = abs(sp_tier.numeric_rank - moody_tier.numeric_rank)
        result["ig_agreement"] = sp_tier.is_investment_grade == moody_tier.is_investment_grade
        result["split_rated"] = not result["ig_agreement"]
    else:
        result["notch_difference"] = None
        result["ig_agreement"] = None
        result["split_rated"] = None

    return result


# ═══════════════════════════════════════════════════════════════════
#  Bootstrap Helper
# ═══════════════════════════════════════════════════════════════════

def get_all_mappings() -> dict[str, Any]:
    """
    Return all classification mappings as a dict for bulk loading
    into dim_classification_mapping.
    """
    return {
        "cfi": dict(_CFI_CATEGORY_MAP),
        "sp_ratings": dict(_SP_RATINGS),
        "moody_ratings": dict(_MOODY_RATINGS),
        "msci_markets": dict(_MSCI_MARKETS),
        "msci_esg": dict(_ESG_RATINGS),
    }
