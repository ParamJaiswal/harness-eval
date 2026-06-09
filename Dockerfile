# ============================================================
# AI Eval Harness — Multi-Stage Production Dockerfile
# ============================================================
# Build:   docker build -t evalharness .
# Run:     docker run -p 8000:8000 evalharness
# ============================================================

# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copy application source
COPY . .

# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

LABEL maintainer="AI Eval Harness Team <team@evalharness.dev>"
LABEL description="Production runtime for the AI Eval Harness evaluation platform"
LABEL version="1.0.0"

# Install runtime-only OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 evaluser && \
    useradd --uid 1000 --gid evaluser --shell /bin/bash --create-home evaluser

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Set working directory
WORKDIR /app

# Copy application code
COPY --from=builder /build/evalharness ./evalharness
COPY --from=builder /build/benchmarks ./benchmarks
COPY --from=builder /build/dashboard ./dashboard

# Create data directory for SQLite persistence
RUN mkdir -p /app/data && chown -R evaluser:evaluser /app

# Switch to non-root user
USER evaluser

# Expose the application port
EXPOSE 8000

# Health check — hit the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EVAL_HARNESS_ENV=production \
    EVAL_HARNESS_DB_PATH=/app/data/evalharness.db \
    EVAL_HARNESS_LOG_LEVEL=INFO

# Run with uvicorn — workers controlled by MAX_WORKERS env var
CMD ["uvicorn", "evalharness.main:app", "--host", "0.0.0.0", "--port", "8000"]
