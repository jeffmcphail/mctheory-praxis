"""Industry factors from SIC codes, and the grouping problem in the middle.

Industries are the largest single thing missing from this model.  USE3 runs 55 of them and
calls them "typically the most important systematic component of a stock's total return".
Without them every industry effect lands in the specific term, where it is assumed
uncorrelated across securities and therefore assumed to diversify -- which is exactly what
it does not do when a whole sector moves together.

**Industry exposures are not styles and must not be treated as such.**  They are indicator
weights, not standardised scores: non-negative, summing to one per security.  Nothing is
centred, nothing is scaled, and running them through `standardise()` would destroy the one
property that makes them interpretable.

**They also span the market.**  A complete set of industry exposures sums to one for every
security, which is the unit column exactly.  Fit a market factor alongside them and the
design is singular -- not nearly singular, exactly so.  That is why Barra's single-country
models have no market factor: the handbook describes "a regression of asset return against
all risk indices and all industries", and the industries do the market's job.  So a model
built with these must use `fit_market=False`.

**The grouping is the real decision.**  SIC is four digits and roughly a thousand values --
far too granular, and most would hold one security.  The two obvious collapses both fail:
the ten SIC divisions put about a third of the S&P 500 into Manufacturing alone, while the
eighty-odd two-digit major groups leave roughly eight securities each, which is too few to
estimate a factor return from.

So this starts at major groups and merges anything below `min_members` into its division.
The threshold is a statistical judgement, not a taxonomy: a factor return estimated from six
securities is mostly those six securities' specific returns wearing an industry label.
`IndustryScheme.describe()` reports what the merging actually did, because a scheme that
collapsed half the market into one bucket is worth noticing before the model is believed.

**The counts are taken over the regression, not the panel.**  A member that is never in a
period's regression contributes nothing to its factor, so counting it is how an industry
passes a floor it does not meet.  `build_scheme` takes a `universe` -- for a model, the
securities in at least one estimated period's regression -- and every count that decides
something, or is reported, is taken over it.  Everything is still classified; only the counts
change.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..model.characteristics import CharacteristicPanel
from ..model.panel import Frequency

# SIC divisions: the coarse fallback a thin major group is merged into.
_DIVISIONS: tuple[tuple[int, int, str], ...] = (
    (1, 9, "Agriculture"),
    (10, 14, "Mining"),
    (15, 17, "Construction"),
    (20, 39, "Manufacturing"),
    (40, 49, "Transport & Utilities"),
    (50, 51, "Wholesale"),
    (52, 59, "Retail"),
    (60, 67, "Finance & Real Estate"),
    (70, 89, "Services"),
    (91, 99, "Public Administration"),
)

MIN_MEMBERS = 15
"""Below this a major group is merged into its division.

Not a taxonomic claim. A factor return estimated from six securities is mostly those six
securities' specific returns with an industry label on it, and it will correlate with
whatever they happened to do.
"""

UNCLASSIFIED = "Unclassified"
OTHER = "Other Industries"

HARD_FLOOR = 5
"""No factor may be estimated from fewer members than this, after every merge.

A one-member industry's factor return *is* that security's return, and the usual diagnostic
cannot see it: with a single member the cross-sectional regression fits it exactly, the
residual is zero, and `noise_share` -- which is sampling variance over observed variance --
therefore reads near zero. The factor looks better estimated than any real one while being
pure specific risk wearing an industry label. A real run produced Agriculture with one
member, 107 percent annual volatility and a 6 percent noise share.

So this is enforced rather than reported, and it has no exemptions. Anything still below the
floor after merging into its division is pooled into OTHER -- Unclassified included, which has
no division to merge into and is no better estimated for lacking one. OTHER is then held to the
floor itself: a pool of three is three stocks, so if it falls short it is not a factor and its
members get no industry exposure at all. Their industry effect falls into specific risk, which
is at least where the model then says it is.

A later run showed why both halves are needed. Agriculture was the only industry under the
floor, so pooling renamed it rather than merging it -- Other Industries, one member, the same
107 percent -- and Unclassified, which the pass skipped, kept four members of whom only one was
in most periods' regression: Signature Bank's post-failure OTC series, at 159 percent.
"""


def division_of(sic: str) -> str:
    """The SIC division a code belongs to."""
    try:
        major = int(str(sic)[:2])
    except (TypeError, ValueError):
        return UNCLASSIFIED
    for low, high, name in _DIVISIONS:
        if low <= major <= high:
            return name
    return UNCLASSIFIED


def major_group_of(sic: str) -> str:
    """The two-digit SIC major group, as a label."""
    text = str(sic).strip()
    if len(text) < 2 or not text[:2].isdigit():
        return UNCLASSIFIED
    return f"SIC{text[:2]}"


@dataclass(frozen=True)
class IndustryScheme:
    """A mapping from security to industry, and a record of how it was arrived at.

    `assignment` classifies every security the scheme was given, including those whose
    industry ends up carrying no factor; `industries` is the labels that are factors. With a
    `universe`, every count -- the ones that decided the grouping and the ones reported -- is
    over the members inside it.
    """

    assignment: Mapping[str, str]
    merged: Mapping[str, str] = field(default_factory=dict)
    min_members: int = MIN_MEMBERS
    hard_floor: int = HARD_FLOOR
    pooled: tuple[str, ...] = ()
    unclassified: tuple[str, ...] = ()
    labels: Mapping[str, str] = field(default_factory=dict)
    # The securities the counts were taken over; None means every one in `assignment`.
    universe: frozenset[str] | None = None
    # Industries with no member in the universe, dropped before anything was merged or pooled.
    dropped: tuple[str, ...] = ()
    # Labels in `assignment` that carry no factor: the dropped industries, and OTHER when the
    # pool itself falls short of the hard floor.
    excluded: tuple[str, ...] = ()
    # Industries folded into another after the grouping, by `merge_industries`: each source
    # industry and the industry its members went to. Apart from `merged`, which is the
    # grouping's own record of major groups into divisions and thin ones into OTHER.
    folded: Mapping[str, str] = field(default_factory=dict)

    @property
    def industries(self) -> tuple[str, ...]:
        """The labels that are factors."""
        return tuple(sorted(set(self.assignment.values()) - set(self.excluded)))

    @property
    def counted(self) -> tuple[str, ...]:
        """The classified securities every count is taken over."""
        if self.universe is None:
            return tuple(self.assignment)
        return tuple(t for t in self.assignment if t in self.universe)

    @property
    def factorless(self) -> tuple[str, ...]:
        """Counted securities whose industry carries no factor.

        They stay in the regression on their styles alone, with an all-zero industry row, and
        their industry effect falls into specific risk.
        """
        factors = set(self.industries)
        return tuple(sorted(t for t in self.counted if self.assignment[t] not in factors))

    def counts(self) -> dict[str, int]:
        """Members per factor, over the universe when the scheme has one."""
        factors = set(self.industries)
        return dict(
            Counter(self.assignment[t] for t in self.counted if self.assignment[t] in factors)
        )

    def describe(self) -> str:
        counts = self.counts()
        if not counts:
            return "no industries assigned"
        total = len(self.counted)
        scope = (
            f"{total} securities"
            if self.universe is None
            else f"{total} securities in the universe ({len(self.assignment)} classified)"
        )
        biggest = max(counts, key=counts.get)
        return (
            f"{len(counts)} industries over {scope}; "
            f"largest {biggest} with {counts[biggest]} "
            f"({counts[biggest] / total:.0%}), "
            f"{len(self.merged)} thin group(s) merged into their division"
            + (f", {len(self.folded)} industry/industries folded into another: "
               + ", ".join(f"{a} into {b}" for a, b in sorted(self.folded.items()))
               if self.folded else "")
        )

    def qualifications(self) -> tuple[str, ...]:
        out: list[str] = []
        counts = self.counts()
        if not self.assignment:
            return ("no security could be classified, so there are no industry factors",)
        if not counts:
            return (
                f"no industry reaches the hard floor of {self.hard_floor} member(s), so there "
                f"are no industry factors",
            )
        total = len(self.counted)
        biggest = max(counts, key=counts.get)
        if counts[biggest] > 0.35 * total:
            out.append(
                f"{biggest} holds {counts[biggest] / total:.0%} of the universe, so its "
                f"factor return is close to the market's and the two are hard to tell "
                f"apart"
            )
        if self.dropped:
            out.append(
                f"{len(self.dropped)} industry/industries have no member in the universe "
                f"and are not factors: {', '.join(self.dropped[:5])}"
            )
        if self.pooled:
            out.append(
                f"{len(self.pooled)} industry/industries had fewer than "
                f"{self.hard_floor} members even after merging into their division and "
                f"were pooled into {OTHER}, which is heterogeneous by construction: "
                + ", ".join(sorted(self.pooled)[:5])
            )
        factorless = self.factorless
        if factorless:
            reason = (
                f"{OTHER} pooled only {len(factorless)}, below the hard floor of "
                f"{self.hard_floor}, so it is not a factor. "
                if OTHER in self.excluded
                else ""
            )
            out.append(
                f"{reason}{len(factorless)} security(ies) have no industry factor: the "
                f"industry effect is unmodelled and falls into specific risk: "
                f"{', '.join(factorless)}"
            )
        thin = {k: v for k, v in counts.items() if v < self.min_members}
        if thin:
            out.append(
                f"{len(thin)} industry/industries still have fewer than "
                f"{self.min_members} members after merging and their factor returns are "
                f"largely their members' specific returns: "
                + ", ".join(f"{k} ({v})" for k, v in sorted(thin.items())[:5])
            )
        unclassified = [
            t for t in self.unclassified if self.universe is None or t in self.universe
        ]
        if unclassified:
            # All of them share one label, which says where the group ended up.
            label = self.assignment[unclassified[0]]
            fate = (
                f"grouped as {label}" if label in counts else "left with no industry factor"
            )
            out.append(
                f"{len(unclassified)} security(ies) have no usable SIC code and are "
                f"{fate}: {', '.join(unclassified[:5])}"
            )
        if self.universe is not None:
            outside = sorted(t for t in self.assignment if t not in self.universe)
            if outside:
                out.append(
                    f"{len(outside)} classified security(ies) are outside the universe and "
                    f"count toward no industry: {', '.join(outside[:5])}"
                )
        return tuple(out)


def build_scheme(
    sic_codes: Mapping[str, str | None],
    *,
    min_members: int = MIN_MEMBERS,
    hard_floor: int = HARD_FLOOR,
    descriptions: Mapping[str, str] | None = None,
    universe: Iterable[str] | None = None,
) -> IndustryScheme:
    """Group securities into industries: major group, then division, then a pooled remainder.

    `min_members` is advisory -- below it a major group merges into its division, which is
    coarser but better populated. `hard_floor` is not advisory: anything still below it is
    pooled into OTHER, Unclassified included, because a factor with one or two members is
    those members and the noise-share diagnostic reads it as well estimated. OTHER is held to
    the floor as well, and a pool that falls short is not a factor at all.

    `universe` is what the counts are taken over: for a model, the securities in at least one
    estimated period's regression (`crosssectional.regression_universe`). An industry with no
    member in it is dropped before anything is merged or pooled. Every security in
    `sic_codes` is still classified -- only the counts change -- and `None` counts them all.
    """
    inside = None if universe is None else frozenset(universe)

    def counted(ticker: str) -> bool:
        return inside is None or ticker in inside

    groups: dict[str, list[str]] = defaultdict(list)
    unclassified: list[str] = []
    for ticker, code in sorted(sic_codes.items()):
        if not code:
            unclassified.append(ticker)
            groups[UNCLASSIFIED].append(ticker)
            continue
        group = major_group_of(code)
        if group == UNCLASSIFIED:
            unclassified.append(ticker)
        groups[group].append(ticker)

    # An industry nobody in the universe belongs to is not thin, it is absent: as a factor it
    # would be an all-zero column in every regression. Dropped before anything is merged, so
    # it neither passes for a thin group nor turns up in OTHER's list. Its members keep the
    # label, and with it an all-zero industry row.
    empty = {
        group for group, members in groups.items() if not any(map(counted, members))
    }

    merged: dict[str, str] = {}
    assignment: dict[str, str] = {}
    for group, members in groups.items():
        if group in empty:
            target = group
        elif group != UNCLASSIFIED and sum(map(counted, members)) < min_members:
            # Merge into the division, which is coarser but usually populated.
            sample = next(
                (sic_codes[t] for t in members if sic_codes.get(t)), None
            )
            target = division_of(sample) if sample else UNCLASSIFIED
            merged[group] = target
        else:
            target = group
        for ticker in members:
            assignment[ticker] = target

    # Second pass: a division can be as thin as the group that merged into it. Agriculture
    # held one security in a real S&P 500 run. Pool whatever is still below the hard floor,
    # because a factor estimated from one member is that member, and noise_share will not
    # tell you so. Unclassified included: having no division to fall back on makes it no
    # better estimated. Only OTHER is left out, since it cannot pool into itself.
    sizes = Counter(label for ticker, label in assignment.items() if counted(ticker))
    # An empty group's label can still be a merge target -- Unclassified is both -- so only
    # the ones that no merge put a member into are dropped.
    dropped = tuple(sorted(group for group in empty if sizes[group] == 0))
    pooled = {
        name: size for name, size in sizes.items() if name != OTHER and size < hard_floor
    }
    if pooled:
        for ticker, name in list(assignment.items()):
            if name in pooled:
                assignment[ticker] = OTHER
        for name in pooled:
            merged[name] = OTHER

    # Then OTHER is held to the floor too. Pooling a single industry renames it rather than
    # merging it -- the run that pooled Agriculture alone got "Other Industries" with the same
    # one member. Whatever still falls short, OTHER or a dropped industry, carries no factor.
    final = Counter(label for ticker, label in assignment.items() if counted(ticker))
    excluded = tuple(
        sorted(label for label in set(assignment.values()) if final[label] < max(hard_floor, 1))
    )

    labels = dict(descriptions or {})
    return IndustryScheme(
        assignment=assignment,
        merged=merged,
        min_members=min_members,
        hard_floor=hard_floor,
        pooled=tuple(sorted(pooled)),
        unclassified=tuple(unclassified),
        labels=labels,
        universe=inside,
        dropped=dropped,
        excluded=excluded,
    )


def merge_industries(scheme: IndustryScheme, merges: Mapping[str, str]) -> IndustryScheme:
    """The scheme with each industry in `merges` folded into the industry it names: every
    member of A labelled B. A chain resolves to its end -- A into B and B into C puts A's
    members in C -- and a cycle is refused, as is a name that is not one of the scheme's
    industry factors. The merges are recorded in `folded`; the grouping's own, in `merged`,
    are left as they were."""
    factors = set(scheme.industries)
    for source, target in merges.items():
        for name in (source, target):
            if name not in factors:
                raise ValueError(
                    f"{name!r} is not an industry factor of the scheme; they are "
                    f"{', '.join(sorted(factors))}"
                )
        if source == target:
            raise ValueError(f"{source!r} cannot be folded into itself")

    def end(name: str) -> str:
        path = [name]
        while name in merges:
            name = merges[name]
            if name in path:
                raise ValueError("the merges form a cycle: " + " into ".join(path + [name]))
            path.append(name)
        return name

    resolved = {source: end(source) for source in merges}
    assignment = {
        ticker: resolved.get(label, label) for ticker, label in scheme.assignment.items()
    }
    folded = {source: resolved.get(target, target) for source, target in scheme.folded.items()}
    folded.update(resolved)
    return replace(scheme, assignment=assignment, folded=folded)


def industry_exposures(
    scheme: IndustryScheme,
    dates: Sequence,
    securities: Sequence[str],
    frequency: Frequency = Frequency.DAILY,
) -> CharacteristicPanel:
    """Indicator exposures: one per industry factor, non-negative, summing to one per security.

    A security whose industry carries no factor -- OTHER below the floor, an industry with no
    member in the universe -- gets an all-zero row instead: classified, so not missing, but
    in the regression on its styles alone.

    Marked `standardised=True` so the estimator leaves them alone. That flag means "these
    exposures are on the scale the model expects", not "these went through standardise()" --
    and for an indicator the correct scale is the indicator itself.
    """
    industries = scheme.industries
    if not industries:
        raise ValueError("the scheme assigns no industries")
    index = {name: i for i, name in enumerate(industries)}
    values = np.zeros((len(dates), len(securities), len(industries)))
    for j, ticker in enumerate(securities):
        industry = scheme.assignment.get(ticker)
        if industry is None:
            values[:, j, :] = np.nan      # unknown, not "belongs to nothing"
            continue
        if industry in index:
            values[:, j, index[industry]] = 1.0
    return CharacteristicPanel(
        dates=tuple(dates),
        securities=tuple(securities),
        names=industries,
        values=values,
        frequency=frequency,
        standardised=True,
    )


def combine(styles: CharacteristicPanel, industries: CharacteristicPanel) -> CharacteristicPanel:
    """Industries then styles, in one panel, for a model fitted with `fit_market=False`.

    The order matters only for reading the output; the regression does not care. What does
    matter is that nothing re-standardises the industry block afterwards.
    """
    if styles.dates != industries.dates or styles.securities != industries.securities:
        raise ValueError(
            "the style and industry panels must share dates and securities; align them "
            "before combining"
        )
    overlap = set(styles.names) & set(industries.names)
    if overlap:
        raise ValueError(f"duplicate factor name(s) across the two panels: {sorted(overlap)}")
    return CharacteristicPanel(
        dates=styles.dates,
        securities=styles.securities,
        names=industries.names + styles.names,
        values=np.concatenate([industries.values, styles.values], axis=2),
        frequency=styles.frequency,
        standardised=True,
    )
