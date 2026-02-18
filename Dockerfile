# =====================================================================
#  McTheory Praxis — Docker Image
# =====================================================================
#  Multi-mode container:
#    surface-build  — Compute critical value surfaces (CPU-intensive)
#    test           — Run pytest suite
#    shell          — Interactive Python shell
#    script         — Run arbitrary scripts
#
#  Build:
#    docker build -t praxis:latest .
#
#  Run examples:
#    docker run praxis surface-build --phase 2
#    docker run praxis test
#    docker run -it praxis shell
# =====================================================================

FROM python:3.12-slim AS base

# System deps for numpy/scipy compiled extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || \
    pip install --no-cache-dir \
    "duckdb>=1.0.0" \
    "numpy>=1.26.0" \
    "pandas>=2.0.0" \
    "polars>=1.0.0" \
    "pyarrow>=14.0.0" \
    "xxhash>=3.0.0" \
    "pydantic>=2.0.0" \
    "pyyaml>=6.0" \
    "scikit-learn>=1.4.0" \
    "statsmodels>=0.14.0" \
    "scipy>=1.12.0" \
    "pytest>=8.0.0" \
    "python-dotenv>=1.0.0"

# ── Production stage ─────────────────────────────────────────────
FROM base AS production

WORKDIR /app

# Copy source
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY pyproject.toml ./

# Set Python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Data directory (mount point for surfaces.duckdb)
RUN mkdir -p /app/data
VOLUME /app/data

# Entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["shell"]
