"""Industry factors from SIC codes.

Two properties carry the module and both are easy to break silently: industry exposures are
indicators rather than standardised scores, and a complete set of them spans the market
column exactly. Everything else here is about the grouping, which is a statistical judgement
dressed as a taxonomy and deserves to be visible.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from praxis.midoffice.model.characteristics import CharacteristicPanel
from praxis.midoffice.model.panel import Frequency
from praxis.midoffice.refdata.industries import (
    MIN_MEMBERS,
    OTHER,
    UNCLASSIFIED,
    build_scheme,
    combine,
    division_of,
    industry_exposures,
    major_group_of,
    merge_industries,
)


def _dates(n: int = 5) -> tuple[date, ...]:
    return tuple(date(2024, 1, 1 + i) for i in range(n))


def _codes(per_group: int = 20) -> dict[str, str]:
    """A universe with three well-populated major groups and one thin one."""
    out: dict[str, str] = {}
    for i in range(per_group):
        out[f"MFG{i:02d}"] = "3571"       # electronic computers
        out[f"FIN{i:02d}"] = "6021"       # national commercial banks
        out[f"RET{i:02d}"] = "5812"       # eating places
    # Six members: thin enough to merge into its division, populous enough to survive
    # the hard floor once there. The distinction between the two thresholds matters and
    # a three-member fixture could not show it.
    for i in range(6):
        out[f"MIN{i:02d}"] = "1311"       # crude petroleum
    return out


# ------------------------------------------------------------------- the SIC mapping


def test_major_group_and_division_are_the_published_ranges():
    assert major_group_of("3571") == "SIC35"
    assert division_of("3571") == "Manufacturing"
    assert division_of("6021") == "Finance & Real Estate"
    assert division_of("1311") == "Mining"
    assert division_of("4911") == "Transport & Utilities"


def test_a_leading_zero_code_survives():
    """SIC codes are identifiers, not numbers. 0100 is agriculture, not 100."""
    assert major_group_of("0100") == "SIC01"
    assert division_of("0100") == "Agriculture"


def test_an_unusable_code_is_unclassified_rather_than_guessed():
    for bad in ("", None, "xx", "9999"):
        assert division_of(bad) in (UNCLASSIFIED, "Public Administration")
    assert major_group_of("") == UNCLASSIFIED


# ------------------------------------------------------------------- the grouping


def test_a_thin_major_group_is_merged_into_its_division():
    """A factor return estimated from three securities is three securities' specific
    returns wearing an industry label."""
    scheme = build_scheme(_codes(), min_members=MIN_MEMBERS)
    assert "SIC13" not in scheme.industries        # six members, below min_members
    assert "Mining" in scheme.industries           # but its division clears the hard floor
    assert scheme.merged.get("SIC13") == "Mining"
    assert all(scheme.assignment[f"MIN{i:02d}"] == "Mining" for i in range(6))
    assert "Mining" not in scheme.pooled


def test_a_well_populated_group_is_left_alone():
    scheme = build_scheme(_codes(), min_members=MIN_MEMBERS)
    for group in ("SIC35", "SIC60", "SIC58"):
        assert group in scheme.industries
    assert scheme.counts()["SIC35"] == 20


def test_the_threshold_changes_the_grouping_and_is_reported():
    coarse = build_scheme(_codes(), min_members=50)
    assert coarse.industries == ("Finance & Real Estate", "Manufacturing", "Mining", "Retail")
    assert len(coarse.merged) == 4
    assert "4 thin group(s) merged" in coarse.describe()
    # Raising the hard floor past the Mining division pools it instead.
    pooled = build_scheme(_codes(), min_members=50, hard_floor=10)
    assert "Mining" not in pooled.industries
    assert "Mining" in pooled.pooled


def test_a_security_with_no_code_is_grouped_and_named():
    codes = dict(_codes())
    for i in range(5):              # enough to clear the hard floor, which binds Unclassified too
        codes[f"MYSTERY{i}"] = None
    scheme = build_scheme(codes)
    assert scheme.assignment["MYSTERY0"] == UNCLASSIFIED
    assert UNCLASSIFIED in scheme.industries
    assert "MYSTERY0" in scheme.unclassified
    assert any("no usable SIC code" in q for q in scheme.qualifications())


def test_a_dominant_industry_is_called_out():
    """If one bucket holds a third of the universe its factor return is nearly the
    market's, and the two will be hard to tell apart in the output."""
    codes = {f"MFG{i:03d}": "3571" for i in range(80)}
    codes.update({f"FIN{i:02d}": "6021" for i in range(20)})
    scheme = build_scheme(codes)
    assert any("close to the market's" in q for q in scheme.qualifications())


def test_industries_still_thin_after_merging_are_reported():
    codes = {f"A{i}": "9100" for i in range(6)}    # over the hard floor, under min_members
    codes.update({f"B{i}": "3571" for i in range(30)})
    scheme = build_scheme(codes)
    assert any("largely their members' specific returns" in q for q in scheme.qualifications())


# --------------------------------------------------------------- the exposure block


def test_exposures_are_indicators_that_sum_to_one():
    scheme = build_scheme(_codes())
    panel = industry_exposures(scheme, _dates(), tuple(scheme.assignment))
    block = panel.values[0]
    assert set(np.unique(block)) <= {0.0, 1.0}
    assert np.allclose(block.sum(axis=1), 1.0)


def test_a_complete_industry_set_spans_the_market_column():
    """Which is why a model using these must be fitted with fit_market=False: adding a
    unit column makes the design exactly singular, not nearly so."""
    scheme = build_scheme(_codes())
    panel = industry_exposures(scheme, _dates(), tuple(scheme.assignment))
    block = panel.values[-1]
    with_market = np.column_stack([np.ones(block.shape[0]), block])
    assert np.linalg.matrix_rank(block) == block.shape[1]
    assert np.linalg.matrix_rank(with_market) == block.shape[1]   # the column added nothing


def test_an_unassigned_security_is_missing_rather_than_belonging_to_nothing():
    scheme = build_scheme(_codes())
    securities = tuple(scheme.assignment) + ("GHOST",)
    panel = industry_exposures(scheme, _dates(), securities)
    assert np.isnan(panel.values[:, -1, :]).all()


def test_the_block_is_marked_standardised_so_nothing_rescales_it():
    """The flag means "on the scale the model expects", and for an indicator that is the
    indicator. Centring and scaling it would destroy the only property it has."""
    scheme = build_scheme(_codes())
    panel = industry_exposures(scheme, _dates(), tuple(scheme.assignment))
    assert panel.standardised
    restandardised = panel.standardise()
    assert not np.allclose(restandardised.values[0], panel.values[0])


def test_an_empty_scheme_is_refused():
    with pytest.raises(ValueError, match="assigns no industries"):
        industry_exposures(build_scheme({}), _dates(), ())


# ------------------------------------------------------------------- combining


def _styles(securities, dates):
    rng = np.random.default_rng(0)
    return CharacteristicPanel(
        dates, securities, ("Size", "Momentum"),
        rng.normal(size=(len(dates), len(securities), 2)), Frequency.DAILY,
    ).standardise()


def test_combining_puts_industries_first_and_keeps_both_blocks_intact():
    scheme = build_scheme(_codes())
    securities = tuple(scheme.assignment)
    dates = _dates()
    industries = industry_exposures(scheme, dates, securities)
    styles = _styles(securities, dates)
    both = combine(styles, industries)
    assert both.names[: len(industries.names)] == industries.names
    assert both.names[len(industries.names):] == styles.names
    assert np.allclose(both.values[:, :, : len(industries.names)], industries.values)
    assert both.standardised


def test_combining_refuses_mismatched_or_colliding_panels():
    scheme = build_scheme(_codes())
    securities = tuple(scheme.assignment)
    dates = _dates()
    industries = industry_exposures(scheme, dates, securities)
    with pytest.raises(ValueError, match="share dates and securities"):
        combine(_styles(securities, _dates(4)), industries)
    clashing = CharacteristicPanel(
        dates, securities, industries.names[:1],
        np.zeros((len(dates), len(securities), 1)), Frequency.DAILY,
    )
    with pytest.raises(ValueError, match="duplicate factor name"):
        combine(clashing, industries)


# ------------------------------------- a factor must never be a single security


def test_a_lone_security_never_becomes_its_own_factor():
    """The failure the noise-share diagnostic cannot see.

    With one member the cross-sectional regression fits it exactly, the residual is zero,
    and noise_share -- sampling variance over observed variance -- reads near zero. A real
    S&P 500 run produced Agriculture with one member, 107 percent annual volatility and a
    6 percent noise share: the best-looking factor in the model, and entirely one stock.
    """
    from praxis.midoffice.refdata.industries import HARD_FLOOR, OTHER

    codes = {f"M{i}": "3571" for i in range(40)}
    codes["LONE"] = "0100"          # Agriculture: one member, and its division has one too
    scheme = build_scheme(codes)
    assert "Agriculture" not in scheme.industries
    assert scheme.assignment["LONE"] == OTHER
    assert "Agriculture" in scheme.pooled
    # Pooling it alone would only rename it, so the pool is held to the floor as well. No
    # factor is exempt -- not OTHER, not Unclassified.
    assert OTHER not in scheme.industries
    assert all(count >= HARD_FLOOR for count in scheme.counts().values())


def test_the_pooling_is_reported_as_heterogeneous():
    codes = {f"M{i}": "3571" for i in range(40)}
    codes["LONE"] = "0100"
    codes["C1"], codes["C2"] = "1521", "1531"
    scheme = build_scheme(codes)
    note = next(q for q in scheme.qualifications() if "pooled into" in q)
    assert "heterogeneous by construction" in note
    assert "Agriculture" in note and "Construction" in note


def test_the_hard_floor_is_separate_from_the_advisory_threshold():
    """min_members merges into a division and is a judgement; hard_floor pools and is not."""
    codes = {f"M{i}": "3571" for i in range(40)}
    codes.update({f"B{i}": "6021" for i in range(8)})     # 8 members
    lenient = build_scheme(codes, min_members=15, hard_floor=5)
    assert "Finance & Real Estate" in lenient.industries   # merged, not pooled
    strict = build_scheme(codes, min_members=15, hard_floor=12)
    assert "Finance & Real Estate" in strict.pooled        # pooled, not merged
    assert "Finance & Real Estate" not in strict.industries
    # A pool of eight is under a floor of twelve as well, so OTHER is not a factor either.
    assert OTHER not in strict.industries


# ------------------------------------- counts over the securities actually regressed


def test_an_industry_thin_only_over_the_universe_merges_into_its_division():
    """The counts that decide are over the universe; the classification is not."""
    codes = {f"M{i:02d}": "3571" for i in range(20)}
    codes.update({f"B{i:02d}": "6021" for i in range(20)})
    gone = {f"B{i:02d}" for i in range(12)}            # in no estimated period's regression
    assert "SIC60" in build_scheme(codes).industries   # twenty banks clear min_members
    scheme = build_scheme(codes, universe=set(codes) - gone)
    assert "SIC60" not in scheme.industries            # the eight that count do not
    assert scheme.merged["SIC60"] == "Finance & Real Estate"
    assert scheme.counts() == {"Finance & Real Estate": 8, "SIC35": 20}
    # Still forty classifications: a bank outside the universe is a bank, just not counted.
    assert len(scheme.assignment) == 40
    assert scheme.assignment["B00"] == "Finance & Real Estate"
    assert "28 securities in the universe (40 classified)" in scheme.describe()
    assert any(
        "12 classified security(ies) are outside the universe" in q
        for q in scheme.qualifications()
    )


def test_an_industry_with_no_member_in_the_universe_is_dropped_outright():
    """Not merged, not pooled: absent. As a factor it would be an all-zero column in every
    period's regression."""
    codes = _codes()
    miners = {f"MIN{i:02d}" for i in range(6)}
    scheme = build_scheme(codes, universe=set(codes) - miners)
    assert scheme.dropped == ("SIC13",)
    assert "SIC13" not in scheme.industries
    assert "Mining" not in scheme.industries
    # Dropped before any merge or pool, so it is neither a thin group nor part of OTHER.
    assert "SIC13" not in scheme.merged
    assert scheme.pooled == ()
    assert OTHER not in scheme.assignment.values()
    assert scheme.assignment["MIN00"] == "SIC13"        # still classified
    securities = tuple(codes)
    block = industry_exposures(scheme, _dates(), securities)
    assert (block.values[:, securities.index("MIN00"), :] == 0.0).all()
    assert any(
        "no member in the universe" in q and "SIC13" in q for q in scheme.qualifications()
    )


def test_unclassified_is_held_to_the_hard_floor_like_any_other_industry():
    """The pooling pass used to skip it, and a real run's Unclassified kept four members --
    DISCA, DISCK, FRC, SBNY -- past a floor of five."""
    codes = {f"M{i:02d}": "3571" for i in range(20)}
    codes.update({f"U{i}": None for i in range(4)})
    codes["LONE"] = "0100"                              # Agriculture: the real run's CTVA
    scheme = build_scheme(codes, hard_floor=5)
    assert UNCLASSIFIED in scheme.pooled
    assert UNCLASSIFIED not in scheme.industries
    assert all(scheme.assignment[f"U{i}"] == OTHER for i in range(4))
    assert scheme.unclassified == ("U0", "U1", "U2", "U3")    # still named as unclassified
    # Four unclassified and one farmer: the pool reaches the floor and is a factor.
    assert scheme.counts()[OTHER] == 5
    assert any(
        "no usable SIC code and are grouped as Other Industries" in q
        for q in scheme.qualifications()
    )


def test_a_pool_below_the_floor_is_not_a_factor_and_its_members_rows_are_zero():
    """The floor binds OTHER too: pooling three securities makes a three-stock factor.

    This is the real run once the counts are over the regression. DISCA and DISCK are in no
    estimated period, which leaves CTVA, FRC and SBNY: three, so no factor. Their rows go to
    zero rather than missing -- they stay in the regression on their styles alone -- and the
    industry effect falls into specific risk.
    """
    codes = {f"M{i:02d}": "3571" for i in range(20)}
    codes.update({f"U{i}": None for i in range(4)})
    codes["LONE"] = "0100"
    scheme = build_scheme(codes, hard_floor=5, universe=set(codes) - {"U0", "U1"})
    assert scheme.industries == ("SIC35",)
    assert OTHER in scheme.excluded
    assert scheme.factorless == ("LONE", "U2", "U3")
    assert any(
        q.endswith(
            "3 security(ies) have no industry factor: the industry effect is unmodelled "
            "and falls into specific risk: LONE, U2, U3"
        )
        for q in scheme.qualifications()
    )
    securities = tuple(codes)
    block = industry_exposures(scheme, _dates(), securities)
    for ticker in ("LONE", "U2", "U3"):
        assert (block.values[:, securities.index(ticker), :] == 0.0).all()
    assert np.allclose(block.values[:, securities.index("M00"), :].sum(axis=1), 1.0)


def test_cached_reference_dates_finds_what_is_on_disk(tmp_path):
    """A delisted name has no record at the end of the window and several before it."""
    from praxis.midoffice.refdata.polygon import PolygonClient

    for day in ("2022-01-03", "2022-06-30", "2023-01-03"):
        (tmp_path / f"ref-ABMD-{day}.json").write_text("{}")
    (tmp_path / "ref-AAPL-2024-01-02.json").write_text("{}")
    (tmp_path / "bars-ABMD-2022-01-03-2026-11-04-adj.json").write_text("{}")
    client = PolygonClient(api_key="k" * 32, cache_dir=tmp_path)
    found = client.cached_reference_dates("ABMD")
    assert found == (date(2023, 1, 3), date(2022, 6, 30), date(2022, 1, 3))
    assert client.cached_reference_dates("NOTHING") == ()


# ------------------------------------------------------------------- folding industries


def test_an_industry_folded_into_another_takes_its_members_with_it():
    scheme = merge_industries(build_scheme(_codes(), min_members=MIN_MEMBERS), {"Mining": "SIC35"})
    assert "Mining" not in scheme.industries
    assert scheme.counts()["SIC35"] == 26
    assert all(scheme.assignment[f"MIN{i:02d}"] == "SIC35" for i in range(6))
    assert scheme.folded == {"Mining": "SIC35"}
    assert scheme.merged.get("SIC13") == "Mining"      # the grouping's own record stands
    assert scheme.describe().endswith(
        "1 industry/industries folded into another: Mining into SIC35"
    )


def test_a_chain_of_folds_resolves_to_its_end_and_a_cycle_is_refused():
    base = build_scheme(_codes(), min_members=MIN_MEMBERS)
    scheme = merge_industries(base, {"Mining": "SIC58", "SIC58": "SIC60"})
    assert scheme.folded == {"Mining": "SIC60", "SIC58": "SIC60"}
    assert scheme.counts()["SIC60"] == 46
    stepwise = merge_industries(merge_industries(base, {"Mining": "SIC58"}), {"SIC58": "SIC60"})
    assert stepwise.folded == scheme.folded and stepwise.assignment == scheme.assignment
    with pytest.raises(ValueError, match="cycle"):
        merge_industries(base, {"SIC58": "SIC60", "SIC60": "SIC58"})


def test_a_fold_names_only_industries_that_are_factors():
    base = build_scheme(_codes(), min_members=MIN_MEMBERS)
    for merges in ({"SIC13": "Mining"}, {"Mining": "Construction"}, {"Mining": "Mining"}):
        with pytest.raises(ValueError):
            merge_industries(base, merges)
