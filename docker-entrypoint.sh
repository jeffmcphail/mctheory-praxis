#!/bin/bash
# =====================================================================
#  McTheory Praxis — Container Entrypoint
# =====================================================================
#  Modes:
#    surface-build [args]  — Run build_surface.py with given args
#    test [args]           — Run pytest with given args
#    shell                 — Interactive Python shell
#    script <name> [args]  — Run scripts/<name>.py with args
#    merge <file>          — Merge remote surface DB into local
#    *                     — Pass through to exec
# =====================================================================

set -e

MODE="${1:-shell}"
shift 2>/dev/null || true

case "$MODE" in
    surface-build)
        echo "=== Praxis Surface Builder ==="
        echo "  Data dir: /app/data"
        echo "  Args: $@"
        echo ""
        exec python scripts/build_surface.py --db-path /app/data/surfaces.duckdb "$@"
        ;;

    test)
        echo "=== Praxis Test Suite ==="
        exec python -m pytest tests/ "$@"
        ;;

    shell)
        echo "=== Praxis Interactive Shell ==="
        echo "  PYTHONPATH=$PYTHONPATH"
        echo "  Data dir: /app/data"
        echo ""
        exec python -i -c "
import numpy as np
from praxis.stats.surface import CompositeSurface, _register_multi_builtins
_register_multi_builtins()
print('Praxis shell ready. CompositeSurface, numpy available.')
print('  surface = CompositeSurface(\"/app/data/surfaces.duckdb\")')
"
        ;;

    script)
        SCRIPT_NAME="${1:?Script name required}"
        shift
        echo "=== Running: scripts/${SCRIPT_NAME}.py ==="
        exec python "scripts/${SCRIPT_NAME}.py" "$@"
        ;;

    merge)
        REMOTE_DB="${1:?Remote DB path required}"
        echo "=== Merging surface data ==="
        echo "  Remote: $REMOTE_DB"
        echo "  Local:  /app/data/surfaces.duckdb"
        exec python scripts/merge_surfaces.py \
            --source "$REMOTE_DB" \
            --target /app/data/surfaces.duckdb
        ;;

    *)
        # Pass through — allows: docker run praxis python my_script.py
        exec "$MODE" "$@"
        ;;
esac
