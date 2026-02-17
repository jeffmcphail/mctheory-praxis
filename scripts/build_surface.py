#!/usr/bin/env python3
"""
Build the Burgess Multi-Statistic Critical Value Surfaces.

One compute pass generates ALL correction surfaces simultaneously:
    - ADF t-statistic
    - Hurst exponent
    - Half-life of mean reversion
    - VR profile eigenvector projections (1st, 2nd, 3rd)
    - VR profile Mahalanobis distance

Phased compute strategy:
    Phase 1: Core range (n_assets<=50, n_obs<=500) — ~1h on 8 cores @ 500 samples
    Phase 2: Extended observations (n_assets<=50, n_obs>500)
    Phase 3: Large universes (n_assets>50, n_obs<=500)
    Phase 4: Full coverage (n_assets>50, n_obs>500)
    Full:    All phases (~7h on 8 cores @ 500 samples)

Usage:
    # Phase 1 only (recommended first run) — ~1h on 8 cores
    python scripts/build_surface.py --phase 1

    # Full grid (all phases) — ~7h on 8 cores
    python scripts/build_surface.py --phase all

    # Quick smoke test (tiny grid, 50 samples)
    python scripts/build_surface.py --smoke-test

    # Check status only
    python scripts/build_surface.py --status-only

    # Resume interrupted build (checkpoints automatically)
    python scripts/build_surface.py --phase 1  # just re-run

    # Override defaults
    python scripts/build_surface.py --phase 1 --n-samples 1000 --n-workers 4 --vr-max-lag 100
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from praxis.logger.core import PraxisLogger
from praxis.stats.surface import (
    CompositeSurface,
    MultiSurfaceRequirement,
    SurfaceAxis,
    _register_multi_builtins,
)


# =====================================================================
#  Grid Definitions
# =====================================================================

def _n_assets_small():
    """3-50: every integer 3-35, then 40,45,50."""
    return sorted(set(list(range(3, 36)) + list(range(40, 51, 5))))

def _n_assets_large():
    """55-1000: 55-95 step 5, 100, 150-1000 step 50."""
    return sorted(set(
        list(range(55, 100, 5)) + [100] + list(range(150, 1001, 50))
    ))

def _n_obs_short():
    """200-500 step 50."""
    return list(range(200, 501, 50))

def _n_obs_long():
    """550-1000 step 50."""
    return list(range(550, 1001, 50))

def _n_vars():
    return [2, 3, 4, 5]


def phase_requirement(
    phase: int,
    n_samples: int = 500,
    seed: int = 42,
    vr_max_lag: int = 50,
) -> MultiSurfaceRequirement:
    """Build requirement for a specific phase."""
    if phase == 1:
        na, no = _n_assets_small(), _n_obs_short()
    elif phase == 2:
        na, no = _n_assets_small(), _n_obs_long()
    elif phase == 3:
        na, no = _n_assets_large(), _n_obs_short()
    elif phase == 4:
        na, no = _n_assets_large(), _n_obs_long()
    else:
        raise ValueError(f"Unknown phase: {phase}")

    return MultiSurfaceRequirement(
        generator="stepwise_regression",
        universe_factory="random_walk",
        axes=[
            SurfaceAxis("n_assets", na),
            SurfaceAxis("n_obs", no),
            SurfaceAxis("n_vars", _n_vars()),
        ],
        n_samples=n_samples,
        seed=seed,
        scalar_extractors=["adf_t", "hurst", "half_life"],
        profile_collectors=["vr_profile"],
        profile_params={"vr_max_lag": vr_max_lag},
        pct_conf=[1, 5, 10, 90, 95, 99],
    )


def smoke_requirement(vr_max_lag: int = 50) -> MultiSurfaceRequirement:
    """Tiny grid for validation."""
    return MultiSurfaceRequirement(
        generator="stepwise_regression",
        universe_factory="random_walk",
        axes=[
            SurfaceAxis("n_assets", [10, 20, 30]),
            SurfaceAxis("n_obs", [200, 300]),
            SurfaceAxis("n_vars", [3]),
        ],
        n_samples=50,
        seed=42,
        scalar_extractors=["adf_t", "hurst", "half_life"],
        profile_collectors=["vr_profile"],
        profile_params={"vr_max_lag": vr_max_lag},
        pct_conf=[1, 5, 10, 90, 95, 99],
    )


# =====================================================================
#  Grid Sizing
# =====================================================================

PHASE_DESCRIPTIONS = {
    1: "Core range (n_assets <= 50, n_obs <= 500)",
    2: "Extended observations (n_assets <= 50, n_obs 550-1000)",
    3: "Large universes (n_assets 55-1000, n_obs <= 500)",
    4: "Full coverage (n_assets 55-1000, n_obs 550-1000)",
}


def grid_size(req: MultiSurfaceRequirement) -> int:
    total = 1
    for axis in req.axes:
        total *= len(axis.values)
    return total


def estimate_hours(req: MultiSurfaceRequirement, n_workers: int = 1) -> float:
    """Rough estimate based on benchmarks: ms/sample ~ 15 + 0.12*n_obs."""
    total_sec = 0
    for na in req.axes[0].values:
        for no in req.axes[1].values:
            for nv in req.axes[2].values:
                ms_per_sample = 15 + 0.12 * no + 0.05 * max(na - 50, 0)
                total_sec += ms_per_sample * req.n_samples / 1000
    return total_sec / 3600 / max(n_workers, 1)


# =====================================================================
#  CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build multi-statistic critical value surfaces for Burgess stat arb.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_surface.py --phase 1                   # Core range, ~1h on 8 cores
  python scripts/build_surface.py --phase all                  # Full grid, ~7h on 8 cores
  python scripts/build_surface.py --smoke-test                 # Quick validation, ~2min
  python scripts/build_surface.py --status-only                # Check progress
  python scripts/build_surface.py --phase 1 --n-samples 1000   # Higher precision
        """,
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="1",
        help="Phase to compute: 1, 2, 3, 4, 'all', or 'smoke' (default: 1)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/surfaces.duckdb",
        help="DuckDB file path (default: data/surfaces.duckdb)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="MC samples per grid point (default: 500)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--vr-max-lag",
        type=int,
        default=50,
        help="Max lag for VR profile (default: 50, affects storage significantly)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Write to DB every N completions (default: 10)",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Just check status, don't compute",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run tiny grid for validation (~2 min)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


# =====================================================================
#  Main
# =====================================================================

def main():
    args = parse_args()

    if args.n_workers is None:
        args.n_workers = max(1, os.cpu_count() - 1)

    if args.smoke_test:
        args.phase = "smoke"

    PraxisLogger.reset()
    log = PraxisLogger.instance()
    log.configure_defaults()
    _register_multi_builtins()

    # -- Banner --------------------------------------------------------
    print("=" * 65)
    print("  McTheory Praxis -- Multi-Statistic Surface Builder")
    print(f"  DB: {args.db_path}")
    print(f"  Samples: {args.n_samples}/point | Workers: {args.n_workers}")
    print(f"  VR max lag: {args.vr_max_lag} | Seed: {args.seed}")
    print("=" * 65)

    # -- Build requirements list ----------------------------------------
    requirements: list[tuple[str, MultiSurfaceRequirement]] = []

    if args.phase == "smoke":
        requirements.append(("Smoke Test", smoke_requirement(args.vr_max_lag)))
    elif args.phase == "all":
        for p in [1, 2, 3, 4]:
            requirements.append((
                f"Phase {p}: {PHASE_DESCRIPTIONS[p]}",
                phase_requirement(p, args.n_samples, args.seed, args.vr_max_lag),
            ))
    else:
        try:
            p = int(args.phase)
            requirements.append((
                f"Phase {p}: {PHASE_DESCRIPTIONS[p]}",
                phase_requirement(p, args.n_samples, args.seed, args.vr_max_lag),
            ))
        except (ValueError, KeyError):
            print(f"ERROR: Unknown phase '{args.phase}'. Use 1-4, 'all', or 'smoke'.")
            sys.exit(1)

    # -- Status ---------------------------------------------------------
    surface = CompositeSurface(db_path=args.db_path)

    print()
    total_grid_pts = 0
    total_est_hours = 0

    for name, req in requirements:
        pts = grid_size(req)
        est_h = estimate_hours(req, args.n_workers)
        total_grid_pts += pts
        total_est_hours += est_h

        stats_list = ", ".join(req.scalar_extractors)
        profiles_list = ", ".join(req.profile_collectors)

        print(f"  {name}")
        print(f"    Grid: {' x '.join(f'{a.name}[{len(a.values)}]' for a in req.axes)} = {pts:,} points")
        print(f"    Scalars: {stats_list}")
        print(f"    Profiles: {profiles_list} (max_lag={req.profile_params.get('vr_max_lag', '?')})")
        print(f"    Estimated: {est_h:.1f}h on {args.n_workers} workers")
        print()

    print(f"  TOTAL: {total_grid_pts:,} points, ~{total_est_hours:.1f}h estimated")
    print()

    if args.status_only:
        if hasattr(surface, "close"): surface.close()
        return

    # -- Compute --------------------------------------------------------
    total_computed = 0
    t0 = time.monotonic()

    for name, req in requirements:
        print(f"  Computing: {name}")
        print(f"  Started:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        phase_t0 = time.monotonic()
        surface.compute(req, n_workers=args.n_workers)
        phase_elapsed = time.monotonic() - phase_t0

        pts = grid_size(req)
        total_computed += pts
        rate = pts / phase_elapsed if phase_elapsed > 0 else 0

        print(f"\n  + {name}")
        print(f"    {pts:,} points in {phase_elapsed:.0f}s ({rate:.1f} pts/sec)")
        print()

    elapsed = time.monotonic() - t0

    # -- Final report ---------------------------------------------------
    db_path = Path(args.db_path)
    size_str = f"{db_path.stat().st_size / (1024*1024):.1f} MB" if db_path.exists() else "N/A"

    print("=" * 65)
    print(f"  COMPLETE")
    print(f"  Points computed: {total_computed:,}")
    print(f"  Wall time:       {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"  Surface file:    {args.db_path} ({size_str})")
    print()
    print("  Statistics available:")
    print("    Scalar:  adf_t, hurst, half_life")
    print("    Profile: vr_profile -> eigen1_proj, eigen2_proj, eigen3_proj, mahalanobis")
    print()
    print("  Next steps:")
    if args.phase == "smoke":
        print("    + Smoke test passed. Run --phase 1 for production Phase 1.")
    elif args.phase == "1":
        print("    + Phase 1 complete. Core range ready for use.")
        print("    -> Run --phase 2 to extend observation range")
        print("    -> Or start using surfaces in Burgess pipeline now")
    elif args.phase == "all":
        print("    + Full grid complete. All ranges covered.")
    print("=" * 65)

    if hasattr(surface, "close"): surface.close()


if __name__ == "__main__":
    main()
