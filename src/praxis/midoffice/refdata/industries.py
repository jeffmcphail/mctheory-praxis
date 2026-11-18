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

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
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

**No member goes without the market** (ruling 3 of midoffice19's brief). In a model with no
market factor a member with no industry factor has no market exposure at all, and a
minimum-variance optimiser treats it as a diversifier: in the S&P 500 run's bias test both
optimiser portfolios held CTVA, alone in Agriculture below the floor, at about a tenth of the
portfolio. So with `joins` a group still below the floor after its division merge joins the
industry factor the mapping names -- its nearest, by the evidence recorded with the mapping --
instead of being pooled into OTHER, and a group below the floor with no entry stops the build.
Unclassified has no nearest industry: a member with no SIC code needs one, by an override with
its evidence. Without `joins`, the pooling above, as it was.
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
    # Groups below the hard floor that joined their nearest industry (`build_scheme`'s `joins`,
    # midoffice19), each with the industry it joined, and its members in the universe.
    joined: Mapping[str, str] = field(default_factory=dict)
    joined_members: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

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
            + (f", {len(self.joined)} group(s) below the hard floor of {self.hard_floor} joined "
               f"their nearest industry: "
               + ", ".join(f"{a} ({len(self.joined_members.get(a, ()))}) into {b}"
                           for a, b in sorted(self.joined.items()))
               if self.joined else "")
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
        if self.joined:
            out.append(
                f"{len(self.joined)} group(s) had fewer than {self.hard_floor} members even "
                f"after merging into their division and joined their nearest industry instead "
                f"of going without one, their industry effect the nearest industry's: "
                + "; ".join(f"{a} into {b} ({', '.join(self.joined_members.get(a, ()))})"
                            for a, b in sorted(self.joined.items()))
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
    joins: Mapping[str, str] | None = None,
) -> IndustryScheme:
    """Group securities into industries: major group, then division, then a pooled remainder --
    or, with `joins`, the industry each group below the floor joins.

    `min_members` is advisory -- below it a major group merges into its division, which is
    coarser but better populated. `hard_floor` is not advisory: anything still below it is
    pooled into OTHER, Unclassified included, because a factor with one or two members is
    those members and the noise-share diagnostic reads it as well estimated. OTHER is held to
    the floor as well, and a pool that falls short is not a factor at all.

    `joins` (midoffice19) maps a group to the industry it joins when it is below the floor after
    its division merge: its nearest, by the evidence that goes with the mapping. Given, nothing
    is pooled and no member in the universe goes without an industry factor: a group below the
    floor joins its entry's industry, which must itself be a factor at or above the floor, and
    one with no entry is refused, as is Unclassified below the floor -- a member with no SIC code
    needs a code, not a neighbour. Entries for groups at or above the floor are not used.

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
    below = {
        name: size for name, size in sizes.items() if name != OTHER and size < hard_floor
    }
    joined: dict[str, str] = {}
    joined_members: dict[str, tuple[str, ...]] = {}
    if joins is None:
        pooled = below
    else:
        # Each group below the floor joins its nearest industry (midoffice19), so no member in
        # the universe is left with no industry factor -- and no market.
        pooled = {}
        for name, size in sorted(below.items()):
            members = ", ".join(sorted(t for t, label in assignment.items()
                                       if label == name and counted(t)))
            if name == UNCLASSIFIED:
                raise ValueError(
                    f"{size} member(s) in the universe have no usable SIC code ({members}): "
                    f"give each its code by an override with its evidence -- {UNCLASSIFIED} "
                    f"has no nearest industry to join"
                )
            target = joins.get(name)
            if target is None:
                raise ValueError(
                    f"{name} has {size} member(s) in the universe ({members}), below the hard "
                    f"floor of {hard_floor}, and no industry to join: name its nearest, with "
                    f"the evidence"
                )
            if (target in (name, OTHER, UNCLASSIFIED) or target in below
                    or sizes.get(target, 0) < hard_floor):
                raise ValueError(
                    f"{name} cannot join {target!r}: not an industry factor at or above the "
                    f"hard floor of {hard_floor}"
                )
            joined[name] = target
            joined_members[name] = tuple(sorted(t for t, label in assignment.items()
                                                if label == name and counted(t)))
        for ticker, name in list(assignment.items()):
            if name in joined:
                assignment[ticker] = joined[name]
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
        joined=joined,
        joined_members=joined_members,
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


@dataclass(frozen=True)
class IndustryOverrides:
    """An industries' rulings file, read (midoffice19): a SIC code by label for each security whose
    records carry none, each entry with the security's key and the evidence; and by group, the
    industry a group below the hard floor joins, each with the evidence."""

    path: str
    codes: Mapping[str, Mapping[str, str]]
    joins: Mapping[str, Mapping[str, str]]

    @property
    def join_map(self) -> dict[str, str]:
        """Each group and the industry it joins, as `build_scheme` takes them."""
        return {group: str(entry["industry"]) for group, entry in self.joins.items()}


def read_industry_overrides(path: str | Path) -> IndustryOverrides:
    """The rulings file: `codes` entries, each a `key`, `label`, `sic` (four digits) and
    `evidence`; `joins` entries, each a `group`, the `industry` it joins and `evidence`. An entry
    lacking one, a label or group given twice, or a code not four digits is refused."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    codes: dict[str, dict] = {}
    for entry in data.get("codes", []):
        missing = [k for k in ("key", "label", "sic", "evidence") if not entry.get(k)]
        if missing:
            raise ValueError(f"a code entry lacks {', '.join(missing)}: {entry}")
        label, sic = str(entry["label"]), str(entry["sic"])
        if label in codes:
            raise ValueError(f"two code entries for {label}")
        if len(sic) != 4 or not sic.isdigit():
            raise ValueError(f"{label}'s SIC code is four digits, not {sic!r}")
        codes[label] = dict(entry)
    joins: dict[str, dict] = {}
    for entry in data.get("joins", []):
        missing = [k for k in ("group", "industry", "evidence") if not entry.get(k)]
        if missing:
            raise ValueError(f"a join entry lacks {', '.join(missing)}: {entry}")
        if entry["group"] in joins:
            raise ValueError(f"two join entries for {entry['group']}")
        joins[str(entry["group"])] = dict(entry)
    return IndustryOverrides(path=str(path), codes=codes, joins=joins)


def apply_code_overrides(
    codes: Mapping[str, str | None],
    overrides: IndustryOverrides,
    keys: Mapping[str, str] | None = None,
) -> tuple[dict[str, str | None], list[str], list[str]]:
    """The codes with each override applied where the records carry none (ruling 3 of
    midoffice19's brief: by its SIC code where the data carry one, by an override with its
    evidence where they do not). `keys`, a security's key by label, checks each entry's key: a
    mismatch is refused, the entry naming another security. Returns the codes, the labels given a
    code, and a note on each entry not applied -- its label not among the codes, or its records
    carrying a code, which stands."""
    out = dict(codes)
    used: list[str] = []
    notes: list[str] = []
    for label, entry in sorted(overrides.codes.items()):
        if label not in out:
            notes.append(f"{label} is not in the panel")
            continue
        if keys is not None and keys.get(label) != entry["key"]:
            raise ValueError(f"the override for {label} names {entry['key']}, and the panel's "
                             f"{label} is {keys.get(label)}")
        if out[label]:
            notes.append(f"{label}'s records carry {out[label]}, which stands; its override "
                         f"({entry['sic']}) is not applied")
            continue
        out[label] = str(entry["sic"])
        used.append(label)
    return out, used, notes


def members_without_industry(
    indices: CharacteristicPanel,
    n_industries: int,
    membership: np.ndarray | None = None,
    first_row: int = 0,
) -> tuple[int, dict[str, int]]:
    """Member-days from `first_row` on with every style exposure and an industry row of zeros:
    each a member a model with no market factor forecasts with no market exposure (midoffice19).
    Their count, and by security. The industries are the panel's first `n_industries` factors,
    as `combine` puts them; `membership` None counts every security."""
    values = np.asarray(indices.values)[first_row:]
    complete = np.isfinite(values[:, :, n_industries:]).all(axis=2)
    none = np.nan_to_num(values[:, :, :n_industries], nan=0.0).sum(axis=2) == 0.0
    held = (np.ones(complete.shape, dtype=bool) if membership is None
            else np.asarray(membership, dtype=bool)[first_row:])
    hits = held & complete & none
    by_security = {indices.securities[j]: int(hits[:, j].sum())
                   for j in np.flatnonzero(hits.any(axis=0))}
    return int(hits.sum()), by_security


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
