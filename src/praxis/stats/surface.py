"""
Generic Critical Value Surface (Phase 4.6).

Pre-computes and persists correction surfaces for any statistical test
applied to residuals from any generating process. Eliminates the need
to run Monte Carlo at model runtime — just a microsecond interpolation lookup.

Architecture:
    1. ResidualGenerator — any function: synthetic_universe → residual_series
    2. StatExtractor    — any function: residual_series → scalar test statistic
    3. UniverseFactory  — creates null-hypothesis synthetic data
    4. SurfaceSpec      — defines the parameter grid + links generator/stat/factory
    5. SurfaceStore     — DuckDB persistence with checkpointing
    6. SurfaceComputer  — parallel computation (embarrassingly parallel per grid point)
    7. SurfaceLookup    — N-dimensional interpolation for arbitrary query points
    8. CriticalValueSurface — main orchestrator

Model Integration:
    A model declares a SurfaceRequirement. On construction:
        surface.ensure(requirement)  →  check / build / extend as needed
    On execution:
        surface.query(requirement, n_assets=97, n_obs=347, n_vars=3)  →  CriticalValues

Built-in implementations:
    - StepwiseResidualGenerator  (Burgess successive regression)
    - ADFStatExtractor           (ADF t-statistic on residuals)
    - JohansenStatExtractor      (Johansen trace statistic)
    - RandomWalkFactory          (null hypothesis: independent random walks)

Usage:
    # Define what a model needs
    req = SurfaceRequirement(
        generator="stepwise_regression",
        stat_test="adf_t",
        universe_factory="random_walk",
        axes=[
            SurfaceAxis("n_assets", [3,4,...,1000]),
            SurfaceAxis("n_obs", [200,250,...,1000]),
            SurfaceAxis("n_vars", [2,3,4,5]),
        ],
        n_samples=1000,
    )

    # Build/verify surface
    surface = CriticalValueSurface("data/surfaces.duckdb")
    status = surface.ensure(req)        # READY | PARTIAL | MISSING
    surface.compute(req, n_workers=8)   # parallel with checkpointing

    # Query at runtime
    cv = surface.query(req, n_assets=97, n_obs=347, n_vars=3)
    print(cv.at(5))   # 5% critical value
    print(cv.is_significant(-4.5, 5))  # True if t < cv
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from praxis.logger.core import PraxisLogger


# ═══════════════════════════════════════════════════════════════════
#  Protocols — The Flaggable Interfaces
# ═══════════════════════════════════════════════════════════════════


@runtime_checkable
class ResidualGenerator(Protocol):
    """
    Produces a residual series from a synthetic universe.

    A model flags one of these on its generating function.
    The function takes a synthetic universe matrix and a target index,
    runs whatever regression/decomposition it uses, and returns
    the residual series that will be tested for stationarity.

    The `params` dict carries any additional configuration that
    varies across the surface grid (e.g., n_vars for stepwise).
    """

    @property
    def name(self) -> str:
        """Unique name for registry and surface keying."""
        ...

    def generate(
        self,
        universe: np.ndarray,
        target_index: int,
        params: dict[str, Any],
    ) -> np.ndarray | None:
        """
        Run generating process on synthetic data.

        Args:
            universe: (n_obs, n_assets) synthetic price/return matrix.
            target_index: Which asset column is the target.
            params: Grid parameters (e.g., {"n_vars": 3}).

        Returns:
            1D residual series, or None if generation failed.
        """
        ...


@runtime_checkable
class StatExtractor(Protocol):
    """
    Extracts a scalar test statistic from a residual series.

    A model flags one of these on each statistical test it wants
    correction surfaces for.
    """

    @property
    def name(self) -> str:
        """Unique name for registry and surface keying."""
        ...

    def extract(self, residuals: np.ndarray) -> float:
        """
        Compute test statistic from residual series.

        Args:
            residuals: 1D array of residuals.

        Returns:
            Scalar test statistic (e.g., ADF t-value).
        """
        ...


@runtime_checkable
class UniverseFactory(Protocol):
    """
    Creates synthetic null-hypothesis data.

    For most use cases this is random walks (no cointegration),
    but could be anything: random returns, bootstrap, etc.
    """

    @property
    def name(self) -> str:
        """Unique name for registry and surface keying."""
        ...

    def create(
        self,
        n_obs: int,
        n_assets: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Generate a synthetic universe.

        Args:
            n_obs: Number of observations (rows).
            n_assets: Number of assets (columns).
            seed: Random seed for reproducibility.

        Returns:
            (n_obs, n_assets) matrix.
        """
        ...


# ═══════════════════════════════════════════════════════════════════
#  Surface Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SurfaceAxis:
    """One axis of the parameter grid."""

    name: str
    values: list[int | float]

    @property
    def size(self) -> int:
        return len(self.values)

    def nearest(self, value: float) -> int | float:
        """Find nearest grid value."""
        arr = np.array(self.values)
        idx = np.argmin(np.abs(arr - value))
        return self.values[idx]

    def bracket(self, value: float) -> tuple[int, int]:
        """Find indices of surrounding grid values for interpolation."""
        arr = np.array(self.values)
        if value <= arr[0]:
            return 0, 0
        if value >= arr[-1]:
            return len(arr) - 1, len(arr) - 1
        idx = np.searchsorted(arr, value)
        return idx - 1, idx


class SurfaceStatus(Enum):
    """Status of a surface relative to a requirement."""

    READY = "ready"  # All grid points computed
    PARTIAL = "partial"  # Some grid points computed, some missing
    MISSING = "missing"  # No surface exists for this requirement


@dataclass
class SurfaceRequirement:
    """
    What a model needs from the critical value surface system.

    A model constructs this to declare its correction needs.
    The surface system uses it to check, build, or extend.
    """

    generator: str  # ResidualGenerator name
    stat_test: str  # StatExtractor name
    universe_factory: str  # UniverseFactory name
    axes: list[SurfaceAxis]
    n_samples: int = 1000
    pct_conf: list[int] = field(default_factory=lambda: [10, 5, 1])
    seed: int | None = 42

    @property
    def surface_id(self) -> str:
        """Deterministic hash of the requirement identity (not grid values)."""
        key = f"{self.generator}|{self.stat_test}|{self.universe_factory}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @property
    def table_name(self) -> str:
        """DuckDB table name for this surface."""
        return f"cv_surface_{self.surface_id}"

    @property
    def total_points(self) -> int:
        """Total grid points in the surface."""
        n = 1
        for axis in self.axes:
            n *= axis.size
        return n

    @property
    def axis_names(self) -> list[str]:
        return [a.name for a in self.axes]

    def grid_points(self) -> list[dict[str, int | float]]:
        """Generate all parameter combinations."""
        import itertools

        axis_values = [a.values for a in self.axes]
        axis_names = [a.name for a in self.axes]
        points = []
        for combo in itertools.product(*axis_values):
            points.append(dict(zip(axis_names, combo)))
        return points


# ═══════════════════════════════════════════════════════════════════
#  Registry — Maps names to implementations
# ═══════════════════════════════════════════════════════════════════


class SurfaceRegistry:
    """
    Global registry of generators, stat extractors, and universe factories.

    Models register their functions here. The surface system looks them up by name.
    """

    _generators: dict[str, ResidualGenerator] = {}
    _extractors: dict[str, StatExtractor] = {}
    _factories: dict[str, UniverseFactory] = {}

    @classmethod
    def register_generator(cls, gen: ResidualGenerator) -> None:
        cls._generators[gen.name] = gen

    @classmethod
    def register_extractor(cls, ext: StatExtractor) -> None:
        cls._extractors[ext.name] = ext

    @classmethod
    def register_factory(cls, fac: UniverseFactory) -> None:
        cls._factories[fac.name] = fac

    @classmethod
    def get_generator(cls, name: str) -> ResidualGenerator:
        if name not in cls._generators:
            raise KeyError(
                f"ResidualGenerator '{name}' not registered. "
                f"Available: {list(cls._generators.keys())}"
            )
        return cls._generators[name]

    @classmethod
    def get_extractor(cls, name: str) -> StatExtractor:
        if name not in cls._extractors:
            raise KeyError(
                f"StatExtractor '{name}' not registered. "
                f"Available: {list(cls._extractors.keys())}"
            )
        return cls._extractors[name]

    @classmethod
    def get_factory(cls, name: str) -> UniverseFactory:
        if name not in cls._factories:
            raise KeyError(
                f"UniverseFactory '{name}' not registered. "
                f"Available: {list(cls._factories.keys())}"
            )
        return cls._factories[name]

    @classmethod
    def clear(cls) -> None:
        cls._generators.clear()
        cls._extractors.clear()
        cls._factories.clear()


# ═══════════════════════════════════════════════════════════════════
#  DuckDB Storage
# ═══════════════════════════════════════════════════════════════════


class SurfaceStore:
    """
    DuckDB-backed persistence for critical value surfaces.

    Schema per surface (table per surface_id):
        axis_1 INT, axis_2 INT, ..., axis_N INT,
        pct_10 DOUBLE, pct_5 DOUBLE, pct_1 DOUBLE,
        n_samples INT, mean_stat DOUBLE, std_stat DOUBLE,
        computed_at TIMESTAMP
    """

    def __init__(self, db_path: str | Path | None = None):
        import duckdb

        self._path = Path(db_path) if db_path else None
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(self._path))
        else:
            self._conn = duckdb.connect(":memory:")

        # Metadata table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cv_surface_meta (
                surface_id VARCHAR PRIMARY KEY,
                generator VARCHAR NOT NULL,
                stat_test VARCHAR NOT NULL,
                universe_factory VARCHAR NOT NULL,
                axis_names VARCHAR NOT NULL,
                n_samples INT NOT NULL,
                pct_conf VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
        """)

    @property
    def connection(self):
        return self._conn

    def _ensure_table(self, req: SurfaceRequirement) -> None:
        """Create the surface table if it doesn't exist."""
        table = req.table_name
        axis_cols = ", ".join(f"{a.name} DOUBLE NOT NULL" for a in req.axes)
        pct_cols = ", ".join(f"pct_{p} DOUBLE" for p in req.pct_conf)

        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                {axis_cols},
                {pct_cols},
                n_samples INT,
                mean_stat DOUBLE,
                std_stat DOUBLE,
                computed_at TIMESTAMP DEFAULT current_timestamp,
                PRIMARY KEY ({', '.join(a.name for a in req.axes)})
            )
        """)

        # Register metadata
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cv_surface_meta
            (surface_id, generator, stat_test, universe_factory,
             axis_names, n_samples, pct_conf)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                req.surface_id,
                req.generator,
                req.stat_test,
                req.universe_factory,
                json.dumps(req.axis_names),
                req.n_samples,
                json.dumps(req.pct_conf),
            ],
        )

    def status(self, req: SurfaceRequirement) -> tuple[SurfaceStatus, int, int]:
        """
        Check how much of a surface is computed.

        Returns:
            (status, computed_count, total_count)
        """
        table = req.table_name
        total = req.total_points

        try:
            result = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            computed = result[0]
        except Exception:
            return SurfaceStatus.MISSING, 0, total

        if computed == 0:
            return SurfaceStatus.MISSING, 0, total
        elif computed >= total:
            return SurfaceStatus.READY, computed, total
        else:
            return SurfaceStatus.PARTIAL, computed, total

    def get_computed_keys(self, req: SurfaceRequirement) -> set[tuple]:
        """Get all already-computed parameter tuples."""
        table = req.table_name
        axis_cols = ", ".join(a.name for a in req.axes)
        try:
            rows = self._conn.execute(
                f"SELECT {axis_cols} FROM {table}"
            ).fetchall()
            return {tuple(r) for r in rows}
        except Exception:
            return set()

    def insert_point(
        self,
        req: SurfaceRequirement,
        params: dict[str, int | float],
        critical_values: dict[int, float],
        n_samples: int,
        mean_stat: float,
        std_stat: float,
    ) -> None:
        """Insert or replace a single computed grid point."""
        self._ensure_table(req)
        table = req.table_name
        axis_names = [a.name for a in req.axes]
        pct_names = [f"pct_{p}" for p in req.pct_conf]

        cols = axis_names + pct_names + ["n_samples", "mean_stat", "std_stat"]
        placeholders = ", ".join(["?"] * len(cols))
        values = (
            [params[a] for a in axis_names]
            + [critical_values.get(p, float("nan")) for p in req.pct_conf]
            + [n_samples, mean_stat, std_stat]
        )

        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
            values,
        )

    def insert_batch(
        self,
        req: SurfaceRequirement,
        rows: list[dict[str, Any]],
    ) -> None:
        """Insert multiple grid points efficiently."""
        self._ensure_table(req)
        for row in rows:
            self.insert_point(
                req,
                params=row["params"],
                critical_values=row["critical_values"],
                n_samples=row["n_samples"],
                mean_stat=row["mean_stat"],
                std_stat=row["std_stat"],
            )

    def load_surface(self, req: SurfaceRequirement) -> dict[tuple, dict[int, float]]:
        """
        Load entire surface into memory for fast interpolation.

        Returns:
            Dict mapping param_tuple → {pct: critical_value}
        """
        table = req.table_name
        axis_cols = [a.name for a in req.axes]
        pct_cols = [f"pct_{p}" for p in req.pct_conf]

        try:
            rows = self._conn.execute(
                f"SELECT {', '.join(axis_cols + pct_cols)} FROM {table}"
            ).fetchall()
        except Exception:
            return {}

        surface = {}
        for row in rows:
            n_axes = len(axis_cols)
            key = tuple(row[:n_axes])
            cvs = {
                req.pct_conf[i]: row[n_axes + i] for i in range(len(req.pct_conf))
            }
            surface[key] = cvs
        return surface

    def close(self) -> None:
        self._conn.close()


# ═══════════════════════════════════════════════════════════════════
#  Interpolation Lookup
# ═══════════════════════════════════════════════════════════════════


class SurfaceLookup:
    """
    N-dimensional interpolation over a pre-computed critical value surface.

    Supports:
    - Exact lookup (grid hit)
    - Linear interpolation between neighbors
    - Nearest-neighbor extrapolation for out-of-range queries
    """

    def __init__(
        self,
        req: SurfaceRequirement,
        surface_data: dict[tuple, dict[int, float]],
    ):
        self._req = req
        self._surface = surface_data
        self._axes = req.axes
        self._pct_conf = req.pct_conf

        # Build sorted axis arrays for interpolation
        self._axis_arrays = [np.array(sorted(a.values)) for a in self._axes]

    def query(self, **kwargs: float) -> dict[int, float]:
        """
        Interpolated critical values at arbitrary parameter point.

        Args:
            **kwargs: Parameter values (e.g., n_assets=97, n_obs=347, n_vars=3)

        Returns:
            {pct: critical_value} dict (e.g., {10: -3.2, 5: -3.8, 1: -4.5})
        """
        point = [float(kwargs[a.name]) for a in self._axes]
        return self._interpolate(point)

    def _interpolate(self, point: list[float]) -> dict[int, float]:
        """N-dimensional linear interpolation."""
        n_dims = len(self._axes)

        # Find bracketing indices and interpolation weights for each dimension
        brackets = []
        weights = []
        for i, (val, axis_arr) in enumerate(zip(point, self._axis_arrays)):
            if val <= axis_arr[0]:
                brackets.append((0, 0))
                weights.append(0.0)
            elif val >= axis_arr[-1]:
                brackets.append((len(axis_arr) - 1, len(axis_arr) - 1))
                weights.append(0.0)
            else:
                idx = np.searchsorted(axis_arr, val)
                lo, hi = idx - 1, idx
                span = axis_arr[hi] - axis_arr[lo]
                w = (val - axis_arr[lo]) / span if span > 0 else 0.0
                brackets.append((lo, hi))
                weights.append(w)

        # Generate all 2^N corners of the interpolation hypercube
        import itertools

        result = {p: 0.0 for p in self._pct_conf}
        total_weight = 0.0

        for corner_bits in itertools.product([0, 1], repeat=n_dims):
            # Build the parameter key for this corner
            key = tuple(
                float(self._axis_arrays[d][brackets[d][bit]])
                for d, bit in enumerate(corner_bits)
            )

            if key not in self._surface:
                continue

            # Compute multilinear weight for this corner
            w = 1.0
            for d, bit in enumerate(corner_bits):
                if brackets[d][0] == brackets[d][1]:
                    w *= 1.0  # exact match on this axis
                elif bit == 1:
                    w *= weights[d]
                else:
                    w *= 1.0 - weights[d]

            corner_cv = self._surface[key]
            for p in self._pct_conf:
                result[p] += w * corner_cv.get(p, 0.0)
            total_weight += w

        # Normalize (handles missing corners gracefully)
        if total_weight > 0:
            for p in self._pct_conf:
                result[p] /= total_weight

        return result

    def query_as_critical_values(self, **kwargs: float) -> "CriticalValues":
        """Query and return as a CriticalValues object (from monte_carlo module)."""
        from praxis.stats.monte_carlo import CriticalValues

        cv = self.query(**kwargs)
        return CriticalValues(
            values=cv,
            n_samples=self._req.n_samples,
            n_assets=int(kwargs.get("n_assets", 0)),
            n_obs=int(kwargs.get("n_obs", 0)),
            n_vars=int(kwargs.get("n_vars", 0)),
        )


# ═══════════════════════════════════════════════════════════════════
#  Single Grid Point Computation (picklable for multiprocessing)
# ═══════════════════════════════════════════════════════════════════


def _compute_single_point(
    generator_name: str,
    extractor_name: str,
    factory_name: str,
    params: dict[str, int | float],
    n_samples: int,
    pct_conf: list[int],
    seed: int | None,
) -> dict[str, Any]:
    """
    Compute critical values for one grid point.

    Designed to be called via ProcessPoolExecutor — all arguments
    are picklable primitives, implementations are looked up from
    the registry inside the worker process.
    """
    # Ensure built-ins are registered in this process
    register_builtins()

    generator = SurfaceRegistry.get_generator(generator_name)
    extractor = SurfaceRegistry.get_extractor(extractor_name)
    factory = SurfaceRegistry.get_factory(factory_name)

    n_assets = int(params.get("n_assets", 10))
    n_obs = int(params.get("n_obs", 250))

    t_values = np.zeros(n_samples)
    current_seed = seed
    j = 0

    while j < n_samples:
        universe = factory.create(n_obs, n_assets, seed=current_seed)
        current_seed = None  # Only seed the first batch

        for i in range(universe.shape[1]):
            if j >= n_samples:
                break

            residuals = generator.generate(universe, i, params)
            if residuals is not None and len(residuals) > 10:
                t_values[j] = extractor.extract(residuals)
            else:
                t_values[j] = 0.0
            j += 1

    t_values.sort()

    # Extract critical values via percentile interpolation
    n = len(t_values)
    indices = np.arange(n)
    critical_values = {}
    for cv in pct_conf:
        idx = n * cv / 100.0 - 1
        critical_values[cv] = float(np.interp(idx, indices, t_values))

    return {
        "params": params,
        "critical_values": critical_values,
        "n_samples": n_samples,
        "mean_stat": float(np.mean(t_values)),
        "std_stat": float(np.std(t_values)),
    }


# ═══════════════════════════════════════════════════════════════════
#  Main Orchestrator
# ═══════════════════════════════════════════════════════════════════


class CriticalValueSurface:
    """
    Main interface for the critical value surface system.

    Lifecycle:
        1. surface = CriticalValueSurface("data/surfaces.duckdb")
        2. status = surface.ensure(requirement)   # check what's needed
        3. surface.compute(requirement, n_workers=8)  # parallel build
        4. cv = surface.query(requirement, **params)  # fast lookup
    """

    def __init__(self, db_path: str | Path | None = None):
        self._store = SurfaceStore(db_path)
        self._lookup_cache: dict[str, SurfaceLookup] = {}
        self._log = PraxisLogger.instance()

    @property
    def store(self) -> SurfaceStore:
        return self._store

    def ensure(self, req: SurfaceRequirement) -> SurfaceStatus:
        """
        Check if the surface is ready for a requirement.

        Returns:
            SurfaceStatus.READY if all points computed,
            SurfaceStatus.PARTIAL if some missing,
            SurfaceStatus.MISSING if no surface exists.
        """
        status, computed, total = self._store.status(req)
        self._log.info(
            f"Surface '{req.surface_id}' ({req.generator}+{req.stat_test}): "
            f"{status.value} ({computed}/{total})",
            tags={"surface"},
        )
        return status

    def compute(
        self,
        req: SurfaceRequirement,
        n_workers: int = 1,
        checkpoint_interval: int = 10,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """
        Compute missing grid points for a surface requirement.

        Args:
            req: Surface requirement defining the grid.
            n_workers: Number of parallel workers (1 = sequential).
            checkpoint_interval: Write to DB every N completions.
            progress_callback: Called with (completed, total) counts.

        Returns:
            Number of grid points computed.
        """
        self._store._ensure_table(req)

        # Find what's already done
        computed_keys = self._store.get_computed_keys(req)
        all_points = req.grid_points()
        missing = [
            p
            for p in all_points
            if tuple(p[a.name] for a in req.axes) not in computed_keys
        ]

        if not missing:
            self._log.info(
                f"Surface '{req.surface_id}': all {len(all_points)} points already computed",
                tags={"surface"},
            )
            return 0

        total = len(missing)
        self._log.info(
            f"Surface '{req.surface_id}': computing {total} missing points "
            f"({len(computed_keys)} already done, {n_workers} workers)",
            tags={"surface"},
        )

        t0 = time.monotonic()
        completed = 0
        pending_writes: list[dict[str, Any]] = []

        if n_workers <= 1:
            # Sequential
            for params in missing:
                result = _compute_single_point(
                    req.generator,
                    req.stat_test,
                    req.universe_factory,
                    params,
                    req.n_samples,
                    req.pct_conf,
                    req.seed,
                )
                pending_writes.append(result)
                completed += 1

                if len(pending_writes) >= checkpoint_interval:
                    self._store.insert_batch(req, pending_writes)
                    pending_writes.clear()

                if progress_callback:
                    progress_callback(completed, total)

                if completed % 50 == 0 or completed == total:
                    elapsed = time.monotonic() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    self._log.info(
                        f"Surface '{req.surface_id}': {completed}/{total} "
                        f"({rate:.1f}/s, ETA {eta:.0f}s)",
                        tags={"surface"},
                    )
        else:
            # Parallel
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {}
                for params in missing:
                    fut = pool.submit(
                        _compute_single_point,
                        req.generator,
                        req.stat_test,
                        req.universe_factory,
                        params,
                        req.n_samples,
                        req.pct_conf,
                        req.seed,
                    )
                    futures[fut] = params

                for fut in as_completed(futures):
                    try:
                        result = fut.result()
                        pending_writes.append(result)
                        completed += 1

                        if len(pending_writes) >= checkpoint_interval:
                            self._store.insert_batch(req, pending_writes)
                            pending_writes.clear()

                        if progress_callback:
                            progress_callback(completed, total)

                        if completed % 50 == 0 or completed == total:
                            elapsed = time.monotonic() - t0
                            rate = completed / elapsed if elapsed > 0 else 0
                            eta = (total - completed) / rate if rate > 0 else 0
                            self._log.info(
                                f"Surface '{req.surface_id}': {completed}/{total} "
                                f"({rate:.1f}/s, ETA {eta:.0f}s)",
                                tags={"surface"},
                            )

                    except Exception as e:
                        self._log.error(
                            f"Surface point failed: {futures[fut]}: {e}",
                            tags={"surface"},
                        )

        # Flush remaining
        if pending_writes:
            self._store.insert_batch(req, pending_writes)

        # Invalidate lookup cache
        self._lookup_cache.pop(req.surface_id, None)

        elapsed = time.monotonic() - t0
        self._log.info(
            f"Surface '{req.surface_id}': computed {completed} points in {elapsed:.1f}s",
            tags={"surface"},
        )

        return completed

    def query(
        self, req: SurfaceRequirement, **kwargs: float
    ) -> dict[int, float]:
        """
        Fast interpolated lookup of critical values.

        Args:
            req: Surface requirement (identifies which surface).
            **kwargs: Parameter values (e.g., n_assets=97, n_obs=347, n_vars=3).

        Returns:
            {pct: critical_value} dict.

        Raises:
            ValueError: If surface not computed.
        """
        lookup = self._get_lookup(req)
        return lookup.query(**kwargs)

    def query_cv(
        self, req: SurfaceRequirement, **kwargs: float
    ) -> "CriticalValues":
        """Query and return as CriticalValues object."""
        lookup = self._get_lookup(req)
        return lookup.query_as_critical_values(**kwargs)

    def _get_lookup(self, req: SurfaceRequirement) -> SurfaceLookup:
        """Get or create a cached SurfaceLookup for a requirement."""
        if req.surface_id not in self._lookup_cache:
            data = self._store.load_surface(req)
            if not data:
                raise ValueError(
                    f"Surface '{req.surface_id}' ({req.generator}+{req.stat_test}) "
                    f"has no computed points. Run compute() first."
                )
            self._lookup_cache[req.surface_id] = SurfaceLookup(req, data)
        return self._lookup_cache[req.surface_id]

    def extend(
        self,
        req: SurfaceRequirement,
        extra_points: list[dict[str, int | float]],
        n_workers: int = 1,
    ) -> int:
        """
        Compute specific additional grid points not in the original spec.

        Use when a model query falls outside the pre-computed surface.

        Args:
            req: Surface requirement.
            extra_points: List of parameter dicts to compute.
            n_workers: Parallel workers.

        Returns:
            Number of points computed.
        """
        self._store._ensure_table(req)
        computed_keys = self._store.get_computed_keys(req)

        missing = [
            p
            for p in extra_points
            if tuple(p[a.name] for a in req.axes) not in computed_keys
        ]

        if not missing:
            return 0

        self._log.info(
            f"Surface '{req.surface_id}': extending with {len(missing)} additional points",
            tags={"surface"},
        )

        for params in missing:
            result = _compute_single_point(
                req.generator,
                req.stat_test,
                req.universe_factory,
                params,
                req.n_samples,
                req.pct_conf,
                req.seed,
            )
            self._store.insert_point(
                req,
                params=result["params"],
                critical_values=result["critical_values"],
                n_samples=result["n_samples"],
                mean_stat=result["mean_stat"],
                std_stat=result["std_stat"],
            )

        # Also add these axis values to the requirement for future interpolation
        self._lookup_cache.pop(req.surface_id, None)
        return len(missing)

    def query_or_compute(
        self,
        req: SurfaceRequirement,
        n_workers: int = 1,
        **kwargs: float,
    ) -> dict[int, float]:
        """
        Query the surface. If the point is outside the grid, compute on-demand
        and extend the surface, then return.

        This is the "just works" method for model runtime.
        """
        try:
            return self.query(req, **kwargs)
        except ValueError:
            # Surface entirely missing — need full compute
            self.compute(req, n_workers=n_workers)
            return self.query(req, **kwargs)

    def close(self) -> None:
        self._store.close()


# ═══════════════════════════════════════════════════════════════════
#  Built-in Implementations
# ═══════════════════════════════════════════════════════════════════


class RandomWalkFactory:
    """Null hypothesis: independent random walks."""

    @property
    def name(self) -> str:
        return "random_walk"

    def create(
        self, n_obs: int, n_assets: int, seed: int | None = None
    ) -> np.ndarray:
        from praxis.stats.regression import generate_random_walk_universe

        return generate_random_walk_universe(
            n_steps=n_obs, n_paths=n_assets, seed=seed
        )


class StepwiseResidualGenerator:
    """
    Burgess successive regression: target → find best correlated partners
    → Ridge regress → extract residuals.
    """

    @property
    def name(self) -> str:
        return "stepwise_regression"

    def generate(
        self,
        universe: np.ndarray,
        target_index: int,
        params: dict[str, Any],
    ) -> np.ndarray | None:
        from praxis.stats.regression import successive_regression

        n_vars = int(params.get("n_vars", 3))
        result = successive_regression(
            target_index=target_index,
            asset_matrix=universe,
            n_vars=n_vars,
            compute_stats=False,  # We'll run stats separately
        )
        if result.regression is not None and len(result.regression.residuals) > 10:
            return result.regression.residuals
        return None


class ADFStatExtractor:
    """Extracts ADF t-statistic from a residual series."""

    @property
    def name(self) -> str:
        return "adf_t"

    def extract(self, residuals: np.ndarray) -> float:
        from praxis.stats import adf_test

        try:
            result = adf_test(residuals)
            return result.t_statistic
        except Exception:
            return 0.0


class JohansenTraceExtractor:
    """Extracts maximum Johansen trace statistic."""

    @property
    def name(self) -> str:
        return "johansen_trace"

    def extract(self, residuals: np.ndarray) -> float:
        # Johansen needs multivariate input — for single residual series,
        # construct a lagged embedding.
        try:
            n = len(residuals)
            if n < 50:
                return 0.0
            # 2-column embedding: series and its lag
            matrix = np.column_stack([residuals[1:], residuals[:-1]])
            from praxis.stats import johansen_test

            result = johansen_test(matrix)
            return float(np.max(result.trace_stats))
        except Exception:
            return 0.0


class HurstExtractor:
    """Extracts Hurst exponent from residual series."""

    @property
    def name(self) -> str:
        return "hurst"

    def extract(self, residuals: np.ndarray) -> float:
        from praxis.stats import hurst_exponent

        try:
            result = hurst_exponent(residuals)
            return result.hurst_exponent
        except Exception:
            return 0.5


def register_builtins() -> None:
    """Register all built-in generators, extractors, and factories."""
    SurfaceRegistry.register_factory(RandomWalkFactory())
    SurfaceRegistry.register_generator(StepwiseResidualGenerator())
    SurfaceRegistry.register_extractor(ADFStatExtractor())
    SurfaceRegistry.register_extractor(JohansenTraceExtractor())
    SurfaceRegistry.register_extractor(HurstExtractor())


# Auto-register on import
register_builtins()


# ═══════════════════════════════════════════════════════════════════
#  Pre-built Surface Specs
# ═══════════════════════════════════════════════════════════════════


def burgess_adf_requirement(
    n_samples: int = 1000,
    seed: int | None = 42,
) -> SurfaceRequirement:
    """
    Standard Burgess ADF critical value surface requirement.

    Covers the production grid:
        n_assets: 3-35 (every), 40-95 (5-step), 95-105, 150-1000 (50-step ±5)
        n_obs:    200-1000 (50-step ±2)
        n_vars:   2, 3, 4, 5

    For production use, compute with the COARSE grid (center points only)
    and rely on interpolation for the dense windows.
    """
    # Coarse grid — center points only, interpolation handles the rest
    n_assets = sorted(set(
        list(range(3, 36))            # 3-35 inclusive
        + list(range(40, 100, 5))     # 40,45,...,95
        + [100]                       # key point
        + list(range(150, 1001, 50))  # 150,200,...,1000
    ))

    n_obs = list(range(200, 1001, 50))  # 200,250,...,1000

    n_vars = [2, 3, 4, 5]

    return SurfaceRequirement(
        generator="stepwise_regression",
        stat_test="adf_t",
        universe_factory="random_walk",
        axes=[
            SurfaceAxis("n_assets", n_assets),
            SurfaceAxis("n_obs", n_obs),
            SurfaceAxis("n_vars", n_vars),
        ],
        n_samples=n_samples,
        seed=seed,
    )


def burgess_hurst_requirement(
    n_samples: int = 1000,
    seed: int | None = 42,
) -> SurfaceRequirement:
    """Hurst exponent surface for Burgess stepwise regression residuals."""
    req = burgess_adf_requirement(n_samples=n_samples, seed=seed)
    return SurfaceRequirement(
        generator="stepwise_regression",
        stat_test="hurst",
        universe_factory="random_walk",
        axes=req.axes,
        n_samples=n_samples,
        seed=seed,
    )


# ═══════════════════════════════════════════════════════════════════
#  Multi-Statistic Surface System
# ═══════════════════════════════════════════════════════════════════
#
#  Extends the scalar surface system with:
#    1. Multi-extractor compute pass (generate residuals ONCE)
#    2. ProfileCollector protocol (VR profiles, etc.)
#    3. Artifact storage (covariance matrices, eigenvectors)
#    4. Composite queries (scalar CVs + matrix artifacts)
#
#  The VR profile pipeline at each grid point:
#    1. Generate N residuals from stepwise regression on random walks
#    2. For each residual: extract ALL scalar stats AND VR profile vector
#    3. For scalar stats: percentile critical values → per-stat tables
#    4. For VR profiles: collect N profiles → covariance + eigenvectors
#    5. Derive scalars: N projection values, N Mahalanobis distances → CVs
#    6. Store: scalar CVs per stat + raw matrices as artifacts
# ═══════════════════════════════════════════════════════════════════


@runtime_checkable
class ProfileCollector(Protocol):
    """
    Collects a vector (not scalar) per residual, then aggregates.

    Unlike StatExtractor which produces a scalar per residual,
    ProfileCollector produces a vector (e.g., variance ratio profile)
    and computes aggregate null-hypothesis artifacts (covariance matrix,
    eigenvectors) from the collection of all profiles at a grid point.
    """

    @property
    def name(self) -> str:
        """Unique name for registry and surface keying."""
        ...

    def collect(
        self, residuals: np.ndarray, params: dict[str, Any]
    ) -> np.ndarray | None:
        """
        Extract a profile vector from one residual series.

        Args:
            residuals: 1D residual array.
            params: Grid parameters (may include profile-specific
                    config like max_lag).

        Returns:
            1D profile vector, or None if extraction failed.
        """
        ...

    @property
    def profile_length(self) -> int | None:
        """Expected length of profile vector, or None if variable."""
        ...

    @property
    def derived_stat_names(self) -> list[str]:
        """
        Names of scalar statistics derived from the profile aggregate.

        These get their own scalar critical value tables.
        E.g., ["vr_eigen2_proj", "vr_mahalanobis"]
        """
        ...


@dataclass
class ProfileArtifact:
    """
    Null-hypothesis artifacts computed from a collection of profiles
    at a single grid point.

    These are stored as binary blobs and loaded at query time for
    computing test statistics on candidate baskets.
    """
    collector_name: str
    n_samples: int
    profile_dim: int

    # The null distribution parameters
    mean_profile: np.ndarray           # (profile_dim,)
    covariance: np.ndarray             # (profile_dim, profile_dim)
    covariance_inv: np.ndarray         # (profile_dim, profile_dim)
    eigenvalues: np.ndarray            # (n_eigenvectors,)
    eigenvectors: np.ndarray           # (profile_dim, n_eigenvectors)
    n_eigenvectors: int = 3

    # Derived scalar distributions (for critical values)
    # Each entry: stat_name → sorted array of N scalar values
    derived_distributions: dict[str, np.ndarray] = field(default_factory=dict)

    def serialize(self) -> bytes:
        """Serialize to bytes for blob storage."""
        import pickle
        return pickle.dumps({
            "collector_name": self.collector_name,
            "n_samples": self.n_samples,
            "profile_dim": self.profile_dim,
            "mean_profile": self.mean_profile,
            "covariance": self.covariance,
            "covariance_inv": self.covariance_inv,
            "eigenvalues": self.eigenvalues,
            "eigenvectors": self.eigenvectors,
            "n_eigenvectors": self.n_eigenvectors,
            "derived_distributions": self.derived_distributions,
        })

    @classmethod
    def deserialize(cls, data: bytes) -> "ProfileArtifact":
        """Reconstruct from serialized bytes."""
        import pickle
        d = pickle.loads(data)
        return cls(**d)

    def project(self, profile: np.ndarray, eigenvector_index: int = 1) -> float:
        """Project a candidate profile onto a null eigenvector."""
        centered = profile - self.mean_profile
        return float(centered @ self.eigenvectors[:, eigenvector_index])

    def mahalanobis(self, profile: np.ndarray) -> float:
        """Mahalanobis distance of candidate profile from null mean."""
        centered = profile - self.mean_profile
        return float(np.sqrt(centered @ self.covariance_inv @ centered))

    def all_projections(self, profile: np.ndarray) -> dict[str, float]:
        """Compute all derived scalar statistics for a candidate profile."""
        centered = profile - self.mean_profile
        stats = {}
        for k in range(self.n_eigenvectors):
            stats[f"eigen{k+1}_proj"] = float(centered @ self.eigenvectors[:, k])
        stats["mahalanobis"] = float(np.sqrt(centered @ self.covariance_inv @ centered))
        return stats


@dataclass
class MultiSurfaceRequirement:
    """
    Defines a multi-statistic surface: one generator, multiple extractors
    and profile collectors sharing the same residual generation pass.

    This is the production requirement for Burgess — one compute pass
    generates all the correction surfaces needed.
    """
    generator: str
    universe_factory: str
    axes: list[SurfaceAxis]
    n_samples: int = 1000
    seed: int | None = 42
    pct_conf: list[int] = field(default_factory=lambda: [1, 5, 10, 90, 95, 99])

    # Scalar extractors (ADF, Hurst, Johansen, half-life, etc.)
    scalar_extractors: list[str] = field(default_factory=list)

    # Profile collectors (VR profile, etc.)
    profile_collectors: list[str] = field(default_factory=list)

    # Profile-specific parameters
    profile_params: dict[str, Any] = field(default_factory=dict)

    @property
    def surface_group_id(self) -> str:
        """Unique ID for the entire group of surfaces."""
        key = (
            f"{self.generator}|{self.universe_factory}|"
            f"{','.join(sorted(self.scalar_extractors))}|"
            f"{','.join(sorted(self.profile_collectors))}|"
            f"{json.dumps(self.profile_params, sort_keys=True)}"
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]

    @property
    def axis_names(self) -> list[str]:
        return [a.name for a in self.axes]

    @property
    def total_points(self) -> int:
        total = 1
        for a in self.axes:
            total *= len(a.values)
        return total

    def grid_points(self) -> list[dict[str, int | float]]:
        """Generate all parameter combinations."""
        from itertools import product as cartesian
        axes_values = [a.values for a in self.axes]
        points = []
        for combo in cartesian(*axes_values):
            points.append({
                self.axes[i].name: combo[i] for i in range(len(self.axes))
            })
        return points

    def scalar_requirement(self, extractor_name: str) -> SurfaceRequirement:
        """Generate a standard SurfaceRequirement for one scalar extractor."""
        return SurfaceRequirement(
            generator=self.generator,
            stat_test=extractor_name,
            universe_factory=self.universe_factory,
            axes=self.axes,
            n_samples=self.n_samples,
            seed=self.seed,
            pct_conf=self.pct_conf,
        )


# ═══════════════════════════════════════════════════════════════════
#  Built-in Profile Collectors
# ═══════════════════════════════════════════════════════════════════


class VRProfileCollector:
    """
    Variance Ratio Profile collector.

    For each residual, computes the VR profile: VR(2), VR(3), ..., VR(max_lag+1).
    Then aggregates across all MC samples to produce:
      - Mean VR profile (null baseline)
      - Covariance matrix of centered VR profiles
      - Eigenvectors (principal components of VR profile shape variation)
      - Derived scalars: 2nd eigenvector projection, Mahalanobis distance

    The 2nd eigenvector captures the mean-reversion curvature signature:
    genuine mean-reversion shows VR sagging below 1.0 at longer lags,
    creating a specific shape pattern that this eigenvector detects.
    """

    def __init__(self, max_lag: int = 100, n_eigenvectors: int = 3):
        self._max_lag = max_lag
        self._n_eigenvectors = n_eigenvectors

    @property
    def name(self) -> str:
        return "vr_profile"

    def collect(
        self, residuals: np.ndarray, params: dict[str, Any]
    ) -> np.ndarray | None:
        """Compute VR profile vector from one residual series."""
        max_lag = int(params.get("vr_max_lag", self._max_lag))

        x = np.asarray(residuals).ravel()
        n = len(x)
        if n < max_lag + 10:
            return None

        # Variance of first differences (denominator for all VR)
        diffs = np.diff(x)
        var_1 = np.var(diffs, ddof=1)
        if var_1 < 1e-15:
            return None

        profile = np.zeros(max_lag)
        for i in range(max_lag):
            lag = i + 2  # VR(2), VR(3), ..., VR(max_lag+1)
            if lag >= n:
                profile[i] = 1.0
                continue
            diffs_lag = x[lag:] - x[:-lag]
            var_lag = np.var(diffs_lag, ddof=1) if len(diffs_lag) > 1 else 0.0
            profile[i] = var_lag / (lag * var_1) if var_1 > 0 else 1.0

        return profile

    @property
    def profile_length(self) -> int | None:
        return self._max_lag

    @property
    def derived_stat_names(self) -> list[str]:
        names = [f"vr_eigen{k+1}_proj" for k in range(self._n_eigenvectors)]
        names.append("vr_mahalanobis")
        return names

    def aggregate(
        self, profiles: np.ndarray, n_eigenvectors: int | None = None
    ) -> ProfileArtifact:
        """
        From (n_samples, profile_dim) matrix, compute null statistics.

        This is where the Burgess VR profile magic happens:
          1. Mean profile → null baseline
          2. Centered profiles → shape variation
          3. Covariance → how lag VR values co-vary under null
          4. Eigenvectors → principal modes of VR shape variation
          5. Projections → scalar test of curvature
          6. Mahalanobis → joint multivariate deviation
        """
        n_eig = n_eigenvectors or self._n_eigenvectors
        n_samples, profile_dim = profiles.shape

        # 1. Null baseline
        mean_profile = profiles.mean(axis=0)

        # 2. Center
        centered = profiles - mean_profile

        # 3. Covariance of VR profile vectors
        cov = np.cov(centered.T)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)

        # Regularize if near-singular (possible for very large max_lag)
        cond = np.linalg.cond(cov)
        if cond > 1e10:
            cov += np.eye(profile_dim) * (np.trace(cov) / profile_dim * 1e-6)

        # 4. Inverse and eigen
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)  # eigh for symmetric
        # Sort descending by eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Trim to top n_eigenvectors
        eigenvalues = eigenvalues[:n_eig]
        eigenvectors = eigenvectors[:, :n_eig]

        # 5. Derived scalar distributions
        derived = {}

        # Eigenvector projections
        for k in range(n_eig):
            projections = centered @ eigenvectors[:, k]
            projections.sort()
            derived[f"vr_eigen{k+1}_proj"] = projections

        # Mahalanobis distances
        # d_i = sqrt(c_i^T @ Σ^{-1} @ c_i)
        maha = np.sqrt(np.sum(centered @ cov_inv * centered, axis=1))
        maha.sort()
        derived["vr_mahalanobis"] = maha

        return ProfileArtifact(
            collector_name=self.name,
            n_samples=n_samples,
            profile_dim=profile_dim,
            mean_profile=mean_profile,
            covariance=cov,
            covariance_inv=cov_inv,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            n_eigenvectors=n_eig,
            derived_distributions=derived,
        )


# ═══════════════════════════════════════════════════════════════════
#  Additional Scalar Extractors
# ═══════════════════════════════════════════════════════════════════


class HalfLifeExtractor:
    """Extracts half-life of mean reversion from residual series."""

    @property
    def name(self) -> str:
        return "half_life"

    def extract(self, residuals: np.ndarray) -> float:
        from praxis.stats import half_life as hl_func

        try:
            result = hl_func(residuals)
            # Cap at 10000 to avoid inf messing up percentiles
            return min(result.half_life, 10000.0)
        except Exception:
            return 10000.0


class VarianceRatioExtractor:
    """Extracts single-lag variance ratio (default lag=10)."""

    def __init__(self, lag: int = 10):
        self._lag = lag

    @property
    def name(self) -> str:
        return f"vr_{self._lag}"

    def extract(self, residuals: np.ndarray) -> float:
        from praxis.stats import variance_ratio as vr_func

        try:
            result = vr_func(residuals, lag=self._lag)
            return result.ratio
        except Exception:
            return 1.0


# ═══════════════════════════════════════════════════════════════════
#  Multi-Point Computation (single pass, all stats)
# ═══════════════════════════════════════════════════════════════════


def _compute_multi_point(
    generator_name: str,
    factory_name: str,
    scalar_extractor_names: list[str],
    profile_collector_names: list[str],
    params: dict[str, int | float],
    n_samples: int,
    pct_conf: list[int],
    seed: int | None,
    profile_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute ALL statistics for one grid point in a single residual pass.

    Returns a dict with:
        "params": the grid parameters
        "scalars": {extractor_name: {"critical_values": {...}, "mean": ..., "std": ...}}
        "profiles": {collector_name: ProfileArtifact.serialize()}
        "derived_scalars": {stat_name: {"critical_values": {...}, "mean": ..., "std": ...}}
    """
    register_builtins()
    _register_multi_builtins()

    generator = SurfaceRegistry.get_generator(generator_name)
    factory = SurfaceRegistry.get_factory(factory_name)

    scalar_extractors = {
        name: SurfaceRegistry.get_extractor(name) for name in scalar_extractor_names
    }

    profile_collectors = {}
    for name in profile_collector_names:
        profile_collectors[name] = _PROFILE_REGISTRY.get(name)
        if profile_collectors[name] is None:
            raise ValueError(f"Unknown profile collector: {name}")

    n_assets = int(params.get("n_assets", 10))
    n_obs = int(params.get("n_obs", 250))

    merged_params = {**params}
    if profile_params:
        merged_params.update(profile_params)

    # Pre-allocate collection arrays
    scalar_values = {name: np.zeros(n_samples) for name in scalar_extractor_names}
    profile_values = {
        name: [] for name in profile_collector_names
    }

    current_seed = seed
    j = 0

    while j < n_samples:
        universe = factory.create(n_obs, n_assets, seed=current_seed)
        current_seed = None

        for i in range(universe.shape[1]):
            if j >= n_samples:
                break

            residuals = generator.generate(universe, i, params)
            if residuals is None or len(residuals) <= 10:
                # Fill defaults
                for name in scalar_extractor_names:
                    scalar_values[name][j] = 0.0
                for name in profile_collector_names:
                    profile_values[name].append(None)
                j += 1
                continue

            # Extract ALL scalar stats from this one residual
            for name, ext in scalar_extractors.items():
                scalar_values[name][j] = ext.extract(residuals)

            # Collect ALL profiles from this one residual
            for name, coll in profile_collectors.items():
                prof = coll.collect(residuals, merged_params)
                profile_values[name].append(prof)

            j += 1

    # ── Assemble scalar results ──────────────────────────────
    result_scalars = {}
    for name in scalar_extractor_names:
        vals = scalar_values[name].copy()
        vals.sort()
        n = len(vals)
        indices = np.arange(n)
        cvs = {}
        for pct in pct_conf:
            idx = n * pct / 100.0 - 1
            cvs[pct] = float(np.interp(idx, indices, vals))
        result_scalars[name] = {
            "critical_values": cvs,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    # ── Assemble profile results ─────────────────────────────
    result_profiles = {}
    result_derived = {}

    for name, coll in profile_collectors.items():
        # Filter out None (failed extractions)
        valid_profiles = [p for p in profile_values[name] if p is not None]
        if len(valid_profiles) < 50:
            continue

        profile_matrix = np.stack(valid_profiles)  # (n_valid, profile_dim)

        # Aggregate → covariance, eigenvectors, etc.
        artifact = coll.aggregate(profile_matrix)

        result_profiles[name] = artifact.serialize()

        # Extract critical values for derived scalars
        for stat_name, dist in artifact.derived_distributions.items():
            n = len(dist)
            indices = np.arange(n)
            cvs = {}
            for pct in pct_conf:
                idx = n * pct / 100.0 - 1
                cvs[pct] = float(np.interp(idx, indices, dist))
            result_derived[stat_name] = {
                "critical_values": cvs,
                "mean": float(np.mean(dist)),
                "std": float(np.std(dist)),
            }

    return {
        "params": params,
        "scalars": result_scalars,
        "profiles": result_profiles,
        "derived_scalars": result_derived,
    }


# ═══════════════════════════════════════════════════════════════════
#  Profile Registry
# ═══════════════════════════════════════════════════════════════════

_PROFILE_REGISTRY: dict[str, Any] = {}


def _register_multi_builtins() -> None:
    """Register multi-stat built-in extractors and collectors."""
    if "vr_profile" not in _PROFILE_REGISTRY:
        _PROFILE_REGISTRY["vr_profile"] = VRProfileCollector()

    # Additional scalar extractors
    if not SurfaceRegistry._extractors.get("half_life"):
        SurfaceRegistry.register_extractor(HalfLifeExtractor())
    if not SurfaceRegistry._extractors.get("vr_10"):
        SurfaceRegistry.register_extractor(VarianceRatioExtractor(10))
    if not SurfaceRegistry._extractors.get("vr_50"):
        SurfaceRegistry.register_extractor(VarianceRatioExtractor(50))


# ═══════════════════════════════════════════════════════════════════
#  Storage Extensions for Artifacts
# ═══════════════════════════════════════════════════════════════════


def _ensure_artifact_table(store: SurfaceStore, req: MultiSurfaceRequirement) -> None:
    """Create an artifact storage table if needed."""
    table = f"cv_artifact_{req.surface_group_id}"
    axis_cols = ", ".join(f"{a.name} DOUBLE NOT NULL" for a in req.axes)

    store.connection.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            {axis_cols},
            collector_name VARCHAR NOT NULL,
            artifact BLOB NOT NULL,
            computed_at TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY ({', '.join(a.name for a in req.axes)}, collector_name)
        )
    """)


def _insert_artifact(
    store: SurfaceStore,
    req: MultiSurfaceRequirement,
    params: dict[str, int | float],
    collector_name: str,
    artifact_bytes: bytes,
) -> None:
    """Store a profile artifact blob for a grid point."""
    _ensure_artifact_table(store, req)
    table = f"cv_artifact_{req.surface_group_id}"
    axis_names = [a.name for a in req.axes]

    cols = axis_names + ["collector_name", "artifact"]
    placeholders = ", ".join(["?"] * len(cols))
    values = [params[a] for a in axis_names] + [collector_name, artifact_bytes]

    store.connection.execute(
        f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
        values,
    )


def _load_artifact(
    store: SurfaceStore,
    req: MultiSurfaceRequirement,
    params: dict[str, int | float],
    collector_name: str,
) -> ProfileArtifact | None:
    """Load a profile artifact for a specific grid point."""
    table = f"cv_artifact_{req.surface_group_id}"
    axis_names = [a.name for a in req.axes]

    where = " AND ".join(f"{a} = ?" for a in axis_names)
    where += " AND collector_name = ?"
    values = [params[a] for a in axis_names] + [collector_name]

    try:
        row = store.connection.execute(
            f"SELECT artifact FROM {table} WHERE {where}", values
        ).fetchone()
        if row:
            return ProfileArtifact.deserialize(row[0])
    except Exception:
        pass
    return None


def _load_nearest_artifact(
    store: SurfaceStore,
    req: MultiSurfaceRequirement,
    collector_name: str,
    **kwargs: float,
) -> ProfileArtifact | None:
    """
    Load the artifact from the nearest grid point.

    For profile artifacts we snap to nearest rather than interpolating,
    since matrix interpolation is ill-defined. The covariance structure
    changes smoothly enough that nearest-neighbor is adequate.
    """
    table = f"cv_artifact_{req.surface_group_id}"
    axis_names = [a.name for a in req.axes]

    # Find nearest grid point by Euclidean distance in normalized space
    try:
        rows = store.connection.execute(
            f"SELECT {', '.join(axis_names)}, artifact FROM {table} "
            f"WHERE collector_name = ?",
            [collector_name],
        ).fetchall()
    except Exception:
        return None

    if not rows:
        return None

    # Normalize each axis to [0, 1] for distance calculation
    target = np.array([kwargs.get(a, 0) for a in axis_names])
    axis_ranges = []
    for a in req.axes:
        vals = a.values
        axis_ranges.append((min(vals), max(vals) - min(vals)) if max(vals) > min(vals) else (0, 1))

    best_dist = float("inf")
    best_blob = None

    for row in rows:
        point = np.array(row[:len(axis_names)], dtype=float)
        # Normalized distance
        normed = np.array([
            (point[i] - axis_ranges[i][0]) / axis_ranges[i][1]
            if axis_ranges[i][1] > 0 else 0
            for i in range(len(axis_names))
        ])
        normed_target = np.array([
            (target[i] - axis_ranges[i][0]) / axis_ranges[i][1]
            if axis_ranges[i][1] > 0 else 0
            for i in range(len(axis_names))
        ])
        dist = np.sum((normed - normed_target) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_blob = row[-1]

    if best_blob is not None:
        return ProfileArtifact.deserialize(best_blob)
    return None


# ═══════════════════════════════════════════════════════════════════
#  Composite Surface Orchestrator
# ═══════════════════════════════════════════════════════════════════


class CompositeSurface:
    """
    Multi-statistic surface: one residual generation pass, all stats.

    This is the production interface for Burgess. A single compute()
    call generates:
        - ADF critical value surface
        - Hurst critical value surface
        - Half-life critical value surface
        - Johansen trace critical value surface
        - VR profile covariance + eigenvector artifacts
        - VR eigenvector projection critical value surface
        - VR Mahalanobis distance critical value surface

    All from ONE pass of stepwise regression on random walks.
    """

    def __init__(self, db_path: str | Path | None = None):
        self._surface = CriticalValueSurface(db_path)
        self._store = self._surface.store
        self._log = PraxisLogger.instance()

    @property
    def scalar_surface(self) -> CriticalValueSurface:
        """Access to the underlying scalar surface system."""
        return self._surface

    def compute(
        self,
        req: MultiSurfaceRequirement,
        n_workers: int = 1,
        checkpoint_interval: int = 10,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """
        Compute all surfaces in a single pass.

        For each grid point:
            1. Generate N residuals (once)
            2. Extract all scalar stats
            3. Collect all VR profiles
            4. Aggregate profiles → artifacts
            5. Store everything

        Returns number of grid points computed.
        """
        _register_multi_builtins()

        # Ensure scalar tables exist for all extractors
        all_scalar_names = list(req.scalar_extractors)
        for coll_name in req.profile_collectors:
            coll = _PROFILE_REGISTRY.get(coll_name)
            if coll:
                all_scalar_names.extend(coll.derived_stat_names)

        for ext_name in all_scalar_names:
            scalar_req = req.scalar_requirement(ext_name)
            self._store._ensure_table(scalar_req)

        # Ensure artifact table
        if req.profile_collectors:
            _ensure_artifact_table(self._store, req)

        # Find what's already done (use first scalar extractor as proxy)
        if req.scalar_extractors:
            proxy_req = req.scalar_requirement(req.scalar_extractors[0])
            computed_keys = self._store.get_computed_keys(proxy_req)
        else:
            computed_keys = set()

        all_points = req.grid_points()
        missing = [
            p for p in all_points
            if tuple(p[a.name] for a in req.axes) not in computed_keys
        ]

        if not missing:
            self._log.info(
                f"CompositeSurface: all {len(all_points)} points already computed",
                tags={"surface"},
            )
            return 0

        total = len(missing)
        self._log.info(
            f"CompositeSurface: computing {total} missing points "
            f"({len(computed_keys)} done) — "
            f"{len(req.scalar_extractors)} scalars, "
            f"{len(req.profile_collectors)} profiles, "
            f"{n_workers} workers",
            tags={"surface"},
        )

        t0 = time.monotonic()
        completed = 0

        if n_workers <= 1:
            for params in missing:
                result = _compute_multi_point(
                    req.generator,
                    req.universe_factory,
                    req.scalar_extractors,
                    req.profile_collectors,
                    params,
                    req.n_samples,
                    req.pct_conf,
                    req.seed,
                    req.profile_params,
                )
                self._store_multi_result(req, result)
                completed += 1

                if progress_callback:
                    progress_callback(completed, total)

                if completed % 10 == 0 or completed == total:
                    elapsed = time.monotonic() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    self._log.info(
                        f"CompositeSurface: {completed}/{total} "
                        f"({rate:.1f}/s, ETA {eta:.0f}s)",
                        tags={"surface"},
                    )
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {}
                for params in missing:
                    fut = pool.submit(
                        _compute_multi_point,
                        req.generator,
                        req.universe_factory,
                        req.scalar_extractors,
                        req.profile_collectors,
                        params,
                        req.n_samples,
                        req.pct_conf,
                        req.seed,
                        req.profile_params,
                    )
                    futures[fut] = params

                for fut in as_completed(futures):
                    try:
                        result = fut.result()
                        self._store_multi_result(req, result)
                        completed += 1

                        if progress_callback:
                            progress_callback(completed, total)

                        if completed % 10 == 0 or completed == total:
                            elapsed = time.monotonic() - t0
                            rate = completed / elapsed if elapsed > 0 else 0
                            eta = (total - completed) / rate if rate > 0 else 0
                            self._log.info(
                                f"CompositeSurface: {completed}/{total} "
                                f"({rate:.1f}/s, ETA {eta:.0f}s)",
                                tags={"surface"},
                            )
                    except Exception as e:
                        self._log.error(
                            f"CompositeSurface point failed: {futures[fut]}: {e}",
                            tags={"surface"},
                        )

        elapsed = time.monotonic() - t0
        self._log.info(
            f"CompositeSurface: computed {completed} points in {elapsed:.1f}s",
            tags={"surface"},
        )
        return completed

    def _store_multi_result(
        self, req: MultiSurfaceRequirement, result: dict[str, Any]
    ) -> None:
        """Store all results from one grid point."""
        params = result["params"]

        # Store scalar critical values
        for ext_name, data in result["scalars"].items():
            scalar_req = req.scalar_requirement(ext_name)
            self._store.insert_point(
                scalar_req,
                params=params,
                critical_values=data["critical_values"],
                n_samples=req.n_samples,
                mean_stat=data["mean"],
                std_stat=data["std"],
            )

        # Store derived scalar critical values (from profiles)
        for stat_name, data in result["derived_scalars"].items():
            scalar_req = req.scalar_requirement(stat_name)
            self._store.insert_point(
                scalar_req,
                params=params,
                critical_values=data["critical_values"],
                n_samples=req.n_samples,
                mean_stat=data["mean"],
                std_stat=data["std"],
            )

        # Store profile artifacts
        for coll_name, artifact_bytes in result["profiles"].items():
            _insert_artifact(self._store, req, params, coll_name, artifact_bytes)

    def query_scalar(
        self, req: MultiSurfaceRequirement, stat_name: str, **kwargs: float
    ) -> dict[int, float]:
        """Query scalar critical values for any statistic."""
        scalar_req = req.scalar_requirement(stat_name)
        return self._surface.query(scalar_req, **kwargs)

    def query_scalar_cv(
        self, req: MultiSurfaceRequirement, stat_name: str, **kwargs: float
    ):
        """Query scalar critical values as CriticalValues object."""
        scalar_req = req.scalar_requirement(stat_name)
        return self._surface.query_cv(scalar_req, **kwargs)

    def query_artifact(
        self, req: MultiSurfaceRequirement, collector_name: str, **kwargs: float
    ) -> ProfileArtifact | None:
        """
        Query a profile artifact (covariance, eigenvectors, etc.).

        Snaps to nearest grid point since matrix interpolation is
        ill-defined. Covariance structure changes smoothly enough
        that nearest-neighbor is adequate.
        """
        return _load_nearest_artifact(
            self._store, req, collector_name, **kwargs
        )

    def test_candidate(
        self,
        req: MultiSurfaceRequirement,
        residuals: np.ndarray,
        n_assets: int,
        n_obs: int,
        n_vars: int,
    ) -> dict[str, dict[str, Any]]:
        """
        Full statistical assessment of a candidate basket's residuals.

        Returns a dict of stat_name → {value, critical_values, significant_at}.

        This is the production query: "Is this candidate real?"
        One call, all tests, all corrected for data-mining.
        """
        _register_multi_builtins()
        results = {}
        query_params = dict(n_assets=n_assets, n_obs=n_obs, n_vars=n_vars)

        # ── Scalar tests ─────────────────────────────────────
        for ext_name in req.scalar_extractors:
            extractor = SurfaceRegistry.get_extractor(ext_name)
            value = extractor.extract(residuals)
            try:
                cvs = self.query_scalar(req, ext_name, **query_params)
            except Exception:
                cvs = {}

            # Determine significance
            sig_at = {}
            for pct in sorted(req.pct_conf):
                if pct in cvs:
                    if ext_name in ("johansen_trace",):
                        # Higher = more significant → right tail
                        mirror = 100 - pct
                        if mirror in cvs:
                            sig_at[pct] = value >= cvs[mirror]
                    elif ext_name in ("adf_t", "hurst", "vr_10", "vr_50", "half_life"):
                        # Lower = more significant → left tail
                        sig_at[pct] = value <= cvs[pct]
                    else:
                        sig_at[pct] = value <= cvs[pct]

            results[ext_name] = {
                "value": value,
                "critical_values": cvs,
                "significant_at": sig_at,
            }

        # ── Profile tests ────────────────────────────────────
        for coll_name in req.profile_collectors:
            coll = _PROFILE_REGISTRY.get(coll_name)
            if coll is None:
                continue

            # Get candidate's profile
            merged_params = {**query_params, **req.profile_params}
            profile = coll.collect(residuals, merged_params)
            if profile is None:
                continue

            # Load null artifact (nearest grid point)
            artifact = self.query_artifact(req, coll_name, **query_params)
            if artifact is None:
                continue

            # Compute all derived statistics
            candidate_stats = artifact.all_projections(profile)

            # Get critical values for each derived statistic
            for stat_name, value in candidate_stats.items():
                full_stat_name = f"{coll.name.split('_')[0]}_{stat_name}"
                # Map to the stored table name
                table_stat_name = f"vr_{stat_name}"
                try:
                    cvs = self.query_scalar(req, table_stat_name, **query_params)
                except Exception:
                    cvs = {}

                sig_at = {}
                for pct in sorted(req.pct_conf):
                    if pct in cvs:
                        if "mahalanobis" in stat_name:
                            # Higher = more anomalous → right tail
                            # Significant at X% means value exceeds the (100-X)th percentile
                            mirror = 100 - pct
                            if mirror in cvs:
                                sig_at[pct] = value >= cvs[mirror]
                        else:
                            # Eigenvector projection: left-tail (more negative = more MR)
                            sig_at[pct] = value <= cvs[pct]

                results[table_stat_name] = {
                    "value": value,
                    "critical_values": cvs,
                    "significant_at": sig_at,
                    "artifact_available": True,
                }

        return results


# ═══════════════════════════════════════════════════════════════════
#  Pre-built Requirements
# ═══════════════════════════════════════════════════════════════════


def burgess_full_requirement(
    n_samples: int = 1000,
    seed: int | None = 42,
    vr_max_lag: int = 100,
) -> MultiSurfaceRequirement:
    """
    Complete Burgess statistical correction surface requirement.

    One compute pass generates ALL correction surfaces:
        - ADF t-statistic
        - Hurst exponent
        - Half-life of mean reversion
        - Johansen trace statistic
        - VR profile eigenvector projections (1st, 2nd, 3rd)
        - VR profile Mahalanobis distance

    Grid covers production range:
        n_assets: 3-35 (every), 40-95 (5-step), 100, 150-1000 (50-step)
        n_obs:    200-1000 (50-step)
        n_vars:   2, 3, 4, 5
    """
    n_assets = sorted(set(
        list(range(3, 36))
        + list(range(40, 100, 5))
        + [100]
        + list(range(150, 1001, 50))
    ))

    n_obs = list(range(200, 1001, 50))
    n_vars = [2, 3, 4, 5]

    return MultiSurfaceRequirement(
        generator="stepwise_regression",
        universe_factory="random_walk",
        axes=[
            SurfaceAxis("n_assets", n_assets),
            SurfaceAxis("n_obs", n_obs),
            SurfaceAxis("n_vars", n_vars),
        ],
        n_samples=n_samples,
        seed=seed,
        pct_conf=[10, 5, 1],
        scalar_extractors=["adf_t", "hurst", "half_life", "johansen_trace"],
        profile_collectors=["vr_profile"],
        profile_params={"vr_max_lag": vr_max_lag},
    )


# Auto-register multi-stat builtins
_register_multi_builtins()
