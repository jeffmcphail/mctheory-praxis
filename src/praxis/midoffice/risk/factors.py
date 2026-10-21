"""Factor identity and the factor-group hierarchy.

Two things live here.  `FactorList` answers what a factor id means.  `FactorTree` answers
which factors sit beneath a reporting group, which is what factor-group decomposition needs
in order to slice the covariance: hand `factors_under(node)` to `FactorCovariance.basis_for`
and you have the sub-block for that group.

**The rollup happens once, at construction, and the result is immutable.** RAT builds the
tree in stages and then requires a separate mutating pass -- `addLeafFactors` followed by
`setAllNodeFactors` -- with the second raising "You must call addLeafFactors before calling
setAllNodeFactors" if you get the order wrong (`FactorTree.cs:130`).  A half-built tree is
reachable state there and is not reachable here.

RAT also catches that exception, logs it and continues (`FactorTree.cs:136-139`), so a leaf
with no factors yields a silently incomplete rollup and every group risk computed above it is
quietly wrong.  `build` raises instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping, Sequence

# RAT carries these as reporting rows alongside the factor groups rather than as factors.
# Named here so callers do not re-invent the strings; nothing in this module treats them
# specially.
TOTAL_RISK = "Total Risk"
SPECIFIC_RISK = "Specific Risk"


@dataclass(frozen=True)
class Factor:
    """One factor's identity.

    `model` namespaces the factor.  It matters whenever several models are pasted into one
    grand covariance -- an equity model beside a fixed income model -- where the same factor
    name can legitimately appear in both and mean different things.
    """

    id: int
    name: str
    model: str = ""
    description: str = ""


class AmbiguousFactor(LookupError):
    """A factor name resolves to more than one id and no model was given to disambiguate."""


@dataclass(frozen=True)
class FactorList:
    """The factor registry: id to identity, and name to id."""

    factors: tuple[Factor, ...]
    _by_id: Mapping[int, Factor] = field(default_factory=dict, repr=False)
    _by_name: Mapping[str, tuple[Factor, ...]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        factors = tuple(self.factors)
        by_id: dict[int, Factor] = {}
        for factor in factors:
            if factor.id in by_id:
                raise ValueError(
                    f"duplicate factor id {factor.id}: "
                    f"{by_id[factor.id].name!r} and {factor.name!r}"
                )
            by_id[factor.id] = factor

        by_name: dict[str, list[Factor]] = {}
        for factor in factors:
            by_name.setdefault(factor.name, []).append(factor)

        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "_by_id", by_id)
        object.__setattr__(self, "_by_name", {k: tuple(v) for k, v in by_name.items()})

    @classmethod
    def of(cls, rows: Iterable[tuple]) -> "FactorList":
        """Build from (id, name[, model[, description]]) tuples."""
        return cls(tuple(Factor(*row) for row in rows))

    def __len__(self) -> int:
        return len(self.factors)

    def __iter__(self) -> Iterator[Factor]:
        return iter(self.factors)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_id))

    def by_id(self, factor_id: int) -> Factor:
        try:
            return self._by_id[factor_id]
        except KeyError:
            raise KeyError(f"no factor with id {factor_id}") from None

    def id_for(self, name: str, model: str | None = None) -> int:
        """Resolve a factor name to its id, optionally within a model.

        **Raises on ambiguity rather than picking one.** RAT's single-argument
        `getFactorId(factor)` returns whichever match it finds first, which is harmless while
        one model is loaded and silently wrong once two are. If a name is ambiguous the caller
        knows which model it meant; this makes them say so.
        """
        matches = self._by_name.get(name, ())
        if not matches:
            raise KeyError(f"no factor named {name!r}")
        if model is not None:
            scoped = [f for f in matches if f.model == model]
            if not scoped:
                raise KeyError(
                    f"no factor named {name!r} in model {model!r} "
                    f"(present in {sorted({f.model for f in matches})})"
                )
            if len(scoped) > 1:
                raise AmbiguousFactor(
                    f"{name!r} appears {len(scoped)} times in model {model!r}"
                )
            return scoped[0].id
        if len(matches) > 1:
            raise AmbiguousFactor(
                f"{name!r} is defined in {len(matches)} models "
                f"({sorted(f.model for f in matches)}); pass model= to disambiguate"
            )
        return matches[0].id

    def ids_for_model(self, model: str) -> frozenset[int]:
        return frozenset(f.id for f in self.factors if f.model == model)

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(sorted({f.model for f in self.factors}))


@dataclass(frozen=True)
class FactorNode:
    """One node of the factor-group hierarchy, with its rolled-up factor set."""

    name: str
    parent: str | None
    children: tuple[str, ...]
    own_factors: frozenset[int]
    factors: frozenset[int]
    depth: int

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def is_root(self) -> bool:
        return self.parent is None


@dataclass(frozen=True)
class FactorTree:
    """A hierarchy of factor groups, each knowing every factor beneath it.

    `factors` on a node is the union of its own factors and all its descendants'.  RAT stores
    this as a hashtable of factor id to an incrementing count; nothing anywhere reads the
    count, only `.Keys` and `.Contains`, so it is a set and is represented as one.
    """

    nodes: Mapping[str, FactorNode]
    root: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", dict(self.nodes))

    @classmethod
    def build(
        cls,
        edges: Iterable[tuple[str, str]],
        leaf_factors: Mapping[str, Iterable[int]],
        *,
        root: str | None = None,
    ) -> "FactorTree":
        """Build and validate the tree, rolling factors up in one pass.

        `edges` are (parent, child) pairs; `leaf_factors` attaches factor ids to leaf names.
        The root is inferred as the unique node that is never a child unless given.
        """
        edges = [(str(p), str(c)) for p, c in edges]
        leaf_factors = {str(k): frozenset(int(f) for f in v) for k, v in leaf_factors.items()}

        names: set[str] = set(leaf_factors)
        for parent, child in edges:
            names.add(parent)
            names.add(child)
        if not names:
            raise ValueError("cannot build a factor tree with no nodes")

        children: dict[str, list[str]] = {n: [] for n in names}
        parents: dict[str, str] = {}
        for parent, child in edges:
            if child in parents:
                raise ValueError(
                    f"node {child!r} has two parents: {parents[child]!r} and {parent!r}"
                )
            parents[child] = parent
            children[parent].append(child)

        roots = sorted(n for n in names if n not in parents)
        if root is None:
            if len(roots) != 1:
                raise ValueError(
                    f"expected exactly one root, found {len(roots)}: {roots[:5]}"
                )
            root = roots[0]
        elif root not in names:
            raise ValueError(f"named root {root!r} is not a node in the tree")
        elif roots != [root]:
            orphans = [r for r in roots if r != root]
            raise ValueError(
                f"nodes unreachable from root {root!r}: {orphans[:5]}"
            )

        # Walk from the root: fixes depth, detects cycles, and proves reachability.
        depth: dict[str, int] = {}
        order: list[str] = []
        stack: list[tuple[str, int]] = [(root, 0)]
        on_path: set[str] = set()
        while stack:
            name, d = stack.pop()
            if name in depth:
                raise ValueError(f"cycle or diamond in the factor tree at {name!r}")
            depth[name] = d
            order.append(name)
            on_path.add(name)
            for child in sorted(children[name]):
                stack.append((child, d + 1))
        unreachable = sorted(names - set(depth))
        if unreachable:
            raise ValueError(f"nodes unreachable from root {root!r}: {unreachable[:5]}")

        leaves = [n for n in names if not children[n]]
        attached_to_branch = sorted(set(leaf_factors) - set(leaves))
        if attached_to_branch:
            raise ValueError(
                "factors may only be attached to leaf nodes; these have children: "
                f"{attached_to_branch[:5]}"
            )
        bare_leaves = sorted(n for n in leaves if not leaf_factors.get(n))
        if bare_leaves:
            raise ValueError(
                f"{len(bare_leaves)} leaf node(s) have no factors, so every group above "
                f"them would roll up incomplete: {bare_leaves[:5]}"
            )

        # Roll up in reverse walk order, which is a valid post-order for a tree.
        rolled: dict[str, frozenset[int]] = {}
        for name in reversed(order):
            own = leaf_factors.get(name, frozenset())
            below: frozenset[int] = own
            for child in children[name]:
                below = below | rolled[child]
            rolled[name] = below

        nodes = {
            name: FactorNode(
                name=name,
                parent=parents.get(name),
                children=tuple(sorted(children[name])),
                own_factors=leaf_factors.get(name, frozenset()),
                factors=rolled[name],
                depth=depth[name],
            )
            for name in names
        }
        return cls(nodes=nodes, root=root)

    # -- access -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, name: str) -> bool:
        return name in self.nodes

    def node(self, name: str) -> FactorNode:
        try:
            return self.nodes[name]
        except KeyError:
            raise KeyError(f"no node named {name!r} in this factor tree") from None

    def factors_under(self, name: str) -> frozenset[int]:
        """Every factor beneath this group. Hand straight to `FactorCovariance.basis_for`."""
        return self.node(name).factors

    @property
    def leaves(self) -> tuple[str, ...]:
        return tuple(sorted(n for n, node in self.nodes.items() if node.is_leaf))

    def children_of(self, name: str) -> tuple[str, ...]:
        return self.node(name).children

    def descendants(self, name: str) -> tuple[str, ...]:
        out: list[str] = []
        stack = list(self.node(name).children)
        while stack:
            current = stack.pop()
            out.append(current)
            stack.extend(self.nodes[current].children)
        return tuple(sorted(out))

    def ancestors(self, name: str) -> tuple[str, ...]:
        out: list[str] = []
        parent = self.node(name).parent
        while parent is not None:
            out.append(parent)
            parent = self.nodes[parent].parent
        return tuple(out)

    def walk(self, start: str | None = None) -> Iterator[FactorNode]:
        """Depth-first from `start` (default root), children in sorted order."""
        stack = [start or self.root]
        while stack:
            node = self.node(stack.pop())
            yield node
            stack.extend(reversed(node.children))

    def render(self, start: str | None = None, factor_list: FactorList | None = None) -> str:
        """An indented text rendering, for debugging a tree and for report headers."""
        lines: list[str] = []
        base = self.node(start or self.root).depth
        for node in self.walk(start):
            indent = "  " * (node.depth - base)
            detail = f"{len(node.factors)} factor(s)"
            if node.is_leaf and factor_list is not None:
                named = sorted(factor_list.by_id(f).name for f in node.factors)
                detail = ", ".join(named[:6]) + ("..." if len(named) > 6 else "")
            lines.append(f"{indent}{node.name}  [{detail}]")
        return "\n".join(lines)
