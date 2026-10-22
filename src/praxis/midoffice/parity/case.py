"""The parity case: a complete, self-contained input set for one risk computation.

A case carries everything needed to reproduce a number, and that deliberately includes the
*conventions* and not just the data.  A golden file generated from a case that does not
record its weight basis, its tail refinement setting or its PSD floor cannot be interpreted
later: the numbers are reproducible only if the choices that produced them travel with them.

This is the same lesson as the engine's provenance work, applied to the test fixtures.  RAT
recorded substituted dates, zeroed prices and coverage exclusions in a log stream running
parallel to its results; reading an old result told you the number but not what it was
computed over.  A fixture without its conventions has the same defect.

`source` is the fixture's own provenance and is not optional in spirit even though the field
has a default: a case is either `synthetic:<generator>:<seed>` or
`rat-export:<dates>:<dir>`, and a case whose origin nobody can state is a case nobody should
trust when it disagrees with the engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1

# Symmetry is asserted rather than silently repaired.  A covariance that arrives asymmetric
# means the producer is wrong, and quietly averaging it would hide that while changing every
# number downstream.  PSD repair is different -- it is a documented, reported operation the
# engine performs and the case records its floor.
_SYMMETRY_TOL = 1e-10


@dataclass(frozen=True)
class FactorSpec:
    """One factor: its id, its display name, and where it sits in the group tree."""

    factor_id: int
    name: str
    model: str = ""
    group: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "name": self.name,
            "model": self.model,
            "group": self.group,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "FactorSpec":
        return cls(
            factor_id=int(raw["factor_id"]),
            name=str(raw["name"]),
            model=str(raw.get("model", "")),
            group=str(raw.get("group", "")),
        )


@dataclass(frozen=True)
class PositionSpec:
    """One security's holding on both sides, with the exposures the model gives it.

    `issuer_key` empty means "key specific risk by security code", which is the engine's
    default and the right behaviour for equities.  It is spelled out here rather than left
    implicit because RAT keyed uncovered names by a mutating static counter, and a fixture
    that cannot express the issuer grouping cannot test the thing that replaced it.
    """

    security_code: str
    weight: float = 0.0
    benchmark_weight: float = 0.0
    exposures: Mapping[int, float] = field(default_factory=dict)
    specific_volatility: float = 0.0
    issuer_key: str = ""
    covered: bool = True
    benchmark_exposures: Mapping[int, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "exposures", {int(k): float(v) for k, v in self.exposures.items()})
        if self.benchmark_exposures is not None:
            object.__setattr__(
                self,
                "benchmark_exposures",
                {int(k): float(v) for k, v in self.benchmark_exposures.items()},
            )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "security_code": self.security_code,
            "weight": self.weight,
            "benchmark_weight": self.benchmark_weight,
            "exposures": {str(k): v for k, v in sorted(self.exposures.items())},
            "specific_volatility": self.specific_volatility,
            "issuer_key": self.issuer_key,
            "covered": self.covered,
        }
        if self.benchmark_exposures is not None:
            out["benchmark_exposures"] = {
                str(k): v for k, v in sorted(self.benchmark_exposures.items())
            }
        return out

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "PositionSpec":
        bench = raw.get("benchmark_exposures")
        return cls(
            security_code=str(raw["security_code"]),
            weight=float(raw.get("weight", 0.0)),
            benchmark_weight=float(raw.get("benchmark_weight", 0.0)),
            exposures={int(k): float(v) for k, v in (raw.get("exposures") or {}).items()},
            specific_volatility=float(raw.get("specific_volatility", 0.0)),
            issuer_key=str(raw.get("issuer_key", "")),
            covered=bool(raw.get("covered", True)),
            benchmark_exposures=(
                None if bench is None else {int(k): float(v) for k, v in bench.items()}
            ),
        )


@dataclass(frozen=True)
class TailCase:
    """One tail measure to compute, including whether z is refined.

    `refine` is in the fixture because it is the difference between agreeing with RAT and
    being right.  RAT uses Acklam's approximation with no Halley step
    (`PortfolioRiskData.cs:576`), whose relative error is 1.15e-9 -- small, but five orders
    of magnitude above float noise, so a comparison that does not pin this setting either
    fails mysteriously or has its tolerance loosened until it stops testing anything.
    """

    confidence: float
    horizon_days: int = 1
    refine: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "horizon_days": self.horizon_days,
            "refine": self.refine,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "TailCase":
        return cls(
            confidence=float(raw["confidence"]),
            horizon_days=int(raw.get("horizon_days", 1)),
            refine=bool(raw.get("refine", False)),
        )


class InvalidCase(ValueError):
    """A case that cannot produce a meaningful comparison."""


@dataclass(frozen=True)
class ParityCase:
    """A complete input set plus the conventions under which it is to be computed."""

    name: str
    factors: tuple[FactorSpec, ...]
    covariance: tuple[tuple[float, ...], ...]
    positions: tuple[PositionSpec, ...]
    portfolio_name: str = "PORT"
    benchmark_name: str = "BENCH"
    weight_basis: str = "full"
    relative_floor: float = 0.0
    proxies: Mapping[int, int] = field(default_factory=dict)
    tails: tuple[TailCase, ...] = ()
    groups: Mapping[str, str] = field(default_factory=dict)
    source: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "factors", tuple(self.factors))
        object.__setattr__(self, "covariance", tuple(tuple(float(x) for x in row) for row in self.covariance))
        object.__setattr__(self, "positions", tuple(self.positions))
        object.__setattr__(self, "tails", tuple(self.tails))
        object.__setattr__(self, "proxies", {int(k): int(v) for k, v in self.proxies.items()})
        object.__setattr__(self, "groups", {str(k): str(v) for k, v in self.groups.items()})
        self.validate()

    # -- derived views ---------------------------------------------------------------

    @property
    def factor_ids(self) -> tuple[int, ...]:
        return tuple(f.factor_id for f in self.factors)

    @property
    def factor_names(self) -> dict[int, str]:
        return {f.factor_id: f.name for f in self.factors}

    # -- group tree ------------------------------------------------------------------
    #
    # Both sides of the comparison read these, which is deliberate: *whether* a case has a
    # comparable group tree is a property of the fixture, not a computation, and the two
    # sides disagreeing about the scope of the comparison would show up as a flood of
    # missing records rather than as the numeric disagreement it is not.

    @property
    def has_group_tree(self) -> bool:
        """True when the groups form a tree that spans every factor.

        A factor with no group would sit under no child of the root, and
        `group_decomposition` refuses that rather than silently dropping its risk.
        """
        return bool(self.groups) and all(f.group for f in self.factors)

    @property
    def group_root(self) -> str:
        """The unique node that is never a child, or empty if there isn't exactly one."""
        children = set(self.groups)
        parents = set(self.groups.values())
        roots = sorted(parents - children)
        return roots[0] if len(roots) == 1 else ""

    def group_edges(self) -> list[tuple[str, str]]:
        """(parent, child) pairs, which is the order `FactorTree.build` expects."""
        return [(parent, child) for child, parent in sorted(self.groups.items())]

    def leaf_factors(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for f in self.factors:
            out.setdefault(f.group, []).append(f.factor_id)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def root_children(self) -> tuple[str, ...]:
        root = self.group_root
        return tuple(sorted(c for c, p in self.groups.items() if p == root))

    def validate(self) -> None:
        """Refuse a case that cannot produce a meaningful comparison.

        Every check here is one that would otherwise surface as a confusing numerical
        disagreement rather than as a bad fixture.
        """
        if self.schema_version != SCHEMA_VERSION:
            raise InvalidCase(
                f"case {self.name!r} declares schema_version {self.schema_version}, "
                f"this build reads version {SCHEMA_VERSION}; regenerate or migrate it "
                f"rather than comparing across versions"
            )

        ids = self.factor_ids
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise InvalidCase(f"duplicate factor ids in case {self.name!r}: {dupes[:8]}")

        n = len(ids)
        if len(self.covariance) != n or any(len(row) != n for row in self.covariance):
            shape = (len(self.covariance), max((len(r) for r in self.covariance), default=0))
            raise InvalidCase(
                f"covariance shape {shape} does not match {n} factors in case {self.name!r}"
            )
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.covariance[i][j], self.covariance[j][i]
                if abs(a - b) > _SYMMETRY_TOL * max(1.0, abs(a), abs(b)):
                    raise InvalidCase(
                        f"covariance in case {self.name!r} is not symmetric at "
                        f"({ids[i]}, {ids[j]}): {a!r} vs {b!r}"
                    )
        for i in range(n):
            if self.covariance[i][i] < 0.0:
                raise InvalidCase(
                    f"negative variance for factor {ids[i]} in case {self.name!r}: "
                    f"{self.covariance[i][i]!r}"
                )

        codes = [p.security_code for p in self.positions]
        if len(set(codes)) != len(codes):
            dupes = sorted({c for c in codes if codes.count(c) > 1})
            raise InvalidCase(
                f"duplicate security codes in case {self.name!r}: {dupes[:8]}; "
                f"aggregate lots before building the case"
            )

        known = set(ids) | set(self.proxies)
        for p in self.positions:
            for source_name, exposures in (
                ("exposures", p.exposures),
                ("benchmark_exposures", p.benchmark_exposures or {}),
            ):
                stray = sorted(set(exposures) - known)
                if stray:
                    raise InvalidCase(
                        f"{p.security_code} carries {source_name} on factor(s) {stray[:8]} "
                        f"which are neither in the model nor proxied, in case {self.name!r}"
                    )
            if p.specific_volatility < 0.0:
                raise InvalidCase(
                    f"negative specific volatility for {p.security_code} in case "
                    f"{self.name!r}: {p.specific_volatility!r}"
                )
            for label, value in (("weight", p.weight), ("benchmark_weight", p.benchmark_weight)):
                if value != value or value in (float("inf"), float("-inf")):
                    raise InvalidCase(
                        f"non-finite {label} for {p.security_code} in case {self.name!r}"
                    )

        if self.weight_basis not in ("full", "covered"):
            raise InvalidCase(
                f"weight_basis {self.weight_basis!r} in case {self.name!r} is not "
                f"'full' or 'covered'"
            )

        for t in self.tails:
            if not 0.0 < t.confidence < 1.0:
                raise InvalidCase(
                    f"tail confidence {t.confidence!r} in case {self.name!r} must be "
                    f"strictly between 0 and 1"
                )
            if t.horizon_days < 1:
                raise InvalidCase(
                    f"tail horizon {t.horizon_days!r} in case {self.name!r} must be at "
                    f"least one day"
                )

        for child, parent in self.groups.items():
            if child == parent:
                raise InvalidCase(
                    f"group {child!r} is its own parent in case {self.name!r}"
                )
        # A cycle in the group map would hang the tree builder, so find it here where the
        # message can name the loop.
        for child in self.groups:
            seen = [child]
            node = self.groups.get(child, "")
            while node:
                if node in seen:
                    raise InvalidCase(
                        f"cycle in group tree of case {self.name!r}: "
                        f"{' -> '.join(seen + [node])}"
                    )
                seen.append(node)
                node = self.groups.get(node, "")

    # -- serialisation ---------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "source": self.source,
            "portfolio_name": self.portfolio_name,
            "benchmark_name": self.benchmark_name,
            "weight_basis": self.weight_basis,
            "relative_floor": self.relative_floor,
            "factors": [f.to_json() for f in self.factors],
            "covariance": [list(row) for row in self.covariance],
            "proxies": {str(k): v for k, v in sorted(self.proxies.items())},
            "groups": dict(sorted(self.groups.items())),
            "positions": [p.to_json() for p in self.positions],
            "tails": [t.to_json() for t in self.tails],
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "ParityCase":
        return cls(
            name=str(raw["name"]),
            factors=tuple(FactorSpec.from_json(f) for f in raw["factors"]),
            covariance=tuple(tuple(float(x) for x in row) for row in raw["covariance"]),
            positions=tuple(PositionSpec.from_json(p) for p in raw["positions"]),
            portfolio_name=str(raw.get("portfolio_name", "PORT")),
            benchmark_name=str(raw.get("benchmark_name", "BENCH")),
            weight_basis=str(raw.get("weight_basis", "full")),
            relative_floor=float(raw.get("relative_floor", 0.0)),
            proxies={int(k): int(v) for k, v in (raw.get("proxies") or {}).items()},
            tails=tuple(TailCase.from_json(t) for t in (raw.get("tails") or [])),
            groups={str(k): str(v) for k, v in (raw.get("groups") or {}).items()},
            source=str(raw.get("source", "")),
            schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
        )

    def dumps(self) -> str:
        """Serialise stably: sorted keys, two-space indent, trailing newline.

        Stability is the point -- a fixture lives in git, so a regenerated case that differs
        only in key order produces a diff that hides the change that matters.
        """
        return json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n"

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.dumps(), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ParityCase":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_source(self, source: str) -> "ParityCase":
        return replace(self, source=source)
