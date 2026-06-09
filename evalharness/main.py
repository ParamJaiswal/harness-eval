"""Main FastAPI application entrypoint.

Sets up the ASGI application, includes CORS middleware (config-driven origins),
optional API-key authentication middleware, registers the API routes, initialises
the SQLite database tables on startup, eager-loads benchmarks, and serves the
static HTML/CSS/JS frontend dashboard.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from evalharness.api.routes import router as api_router, loader
from evalharness.config import get_settings
from evalharness.models.schemas import init_db

settings = get_settings()

# Setup basic logging configuration
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler for startup/shutdown database and loader initialization."""
    logger.info("Initializing database schema...")
    await init_db()
    logger.info("Eagerly loading all benchmarks from YAML files...")
    loader.load_all()
    logger.info("AI Eval Harness is ready.")
    yield
    logger.info("Shutting down AI Eval Harness server...")


app = FastAPI(
    title="AI Eval Harness",
    description=(
        "Production-grade evaluation harness for statistical verification of "
        "LLMs, Agent tool trajectories, and RAG pipelines."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# CORS — driven by EVAL_HARNESS_CORS_ORIGINS env var
# ---------------------------------------------------------------------------
cors_origins = settings.get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Optional API-key authentication middleware
# ---------------------------------------------------------------------------
_API_KEY = settings.API_KEY.strip()


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Enforce Bearer-token authentication on /api/* when API_KEY is configured."""
    if _API_KEY and request.url.path.startswith("/api/"):
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip()
        if token != _API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized. Provide a valid Bearer token."},
                headers={"WWW-Authenticate": "Bearer"},
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Health check — mounted before API router so it is never blocked by auth
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"], include_in_schema=True)
async def health_check() -> dict:
    """Health check endpoint used by Docker / load-balancer probes."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "env": settings.ENV,
    }


# ---------------------------------------------------------------------------
# API router
# ---------------------------------------------------------------------------
# Include the API router before mounting static files so API endpoints take precedence
app.include_router(api_router)

# ---------------------------------------------------------------------------
# Static dashboard
# ---------------------------------------------------------------------------
# Mount the frontend dashboard at root. StaticFiles(html=True) handles serving
# index.html at root. Must be the *last* mount.
app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")


if __name__ == "__main__":
    uvicorn.run("evalharness.main:app", host="0.0.0.0", port=8000, reload=False)
