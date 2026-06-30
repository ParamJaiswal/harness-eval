"""SQLAlchemy 2.0 async ORM models and Pydantic v2 request/response schemas.

This module defines the persistence layer (``EvalRun``, ``TaskResult``) and the
API contract (``EvalRunCreate``, ``EvalRunResponse``, ``MetricsResponse``, etc.).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import Float, ForeignKey, Integer, Text, String
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from evalharness.config import get_settings

# ---------------------------------------------------------------------------
# SQLAlchemy base
# ---------------------------------------------------------------------------


class Base(AsyncAttrs, DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------


class EvalRun(Base):
    """A single evaluation run — one model against one benchmark."""

    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    benchmark_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    eval_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default="llm"
    )  # llm | agent | rag
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending"
    )  # pending | running | completed | failed
    total_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_tasks: Mapped[int] = mapped_column(Integer, default=0)
    completed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    config_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    task_results: Mapped[list["TaskResult"]] = relationship(
        back_populates="eval_run",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<EvalRun id={self.id!r} model={self.model_name!r} "
            f"status={self.status!r}>"
        )


class TaskResult(Base):
    """Result of a single task within an evaluation run."""

    __tablename__ = "task_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("eval_runs.id"), nullable=False, index=True
    )
    task_id: Mapped[str] = mapped_column(String(128), nullable=False)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    raw_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    scoring_method: Mapped[str] = mapped_column(String(32), default="exact_match")
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    eval_run: Mapped["EvalRun"] = relationship(back_populates="task_results")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<TaskResult id={self.id!r} task={self.task_id!r} score={self.score}>"


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine():
    """Return (and lazily create) the global async engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.get_database_url(), echo=False)
    return _engine


def async_session_maker() -> AsyncSession:
    """Return a new ``AsyncSession`` from the global factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _session_factory()


async def init_db(engine=None) -> None:
    """Create all tables (idempotent)."""
    eng = engine or get_engine()
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class EvalRunCreate(BaseModel):
    """Request body to start a new evaluation run.

    For **mock/built-in adapters** (demo):
        Provide ``model_name`` + ``benchmark_name`` + ``eval_type``.

    For **customer endpoints** (production):
        Also provide ``endpoint_url``, ``api_key``, ``provider_type``,
        and optionally ``model_id``. The harness builds an HTTP adapter
        on-the-fly and evaluates the live endpoint.
    """

    # --- Core fields ---
    model_name: str = Field(..., description="Name/label for this model (used in results)")
    benchmark_name: str = Field(..., description="Name or filename-stem of the benchmark to run")
    eval_type: str = Field(
        default="llm",
        description="Evaluation type: 'llm', 'agent', or 'rag'",
    )

    # --- Customer endpoint fields (optional — omit to use mock adapters) ---
    endpoint_url: str = Field(
        default="",
        description=(
            "Full URL of the customer's model/RAG/agent endpoint. "
            "When provided, the harness calls this URL instead of a mock adapter."
        ),
    )
    api_key: str = Field(
        default="",
        description="Bearer token / API key for the customer's endpoint.",
    )
    provider_type: str = Field(
        default="openai_compatible",
        description=(
            "Provider format: 'openai_compatible' | 'anthropic' | "
            "'custom_llm' | 'rag_webhook' | 'agent_webhook'"
        ),
    )
    model_id: str = Field(
        default="",
        description="Model identifier string to pass to the endpoint (e.g. 'llama-3-70b').",
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers to send with each request (e.g. custom auth headers).",
    )
    display_name: str = Field(
        default="",
        description="Human-readable name shown in results (defaults to model_name).",
    )
    top_k: int = Field(
        default=5,
        description="Number of contexts to retrieve (RAG only).",
    )
    
    # --- Custom Benchmark fields ---
    custom_tasks: list[dict[str, Any]] | None = Field(
        default=None,
        description="List of custom task definitions. If provided and benchmark_name is 'custom', these tasks will be used instead of loading a benchmark from disk.",
    )
    
    # --- Dynamic API Key Overrides ---
    openai_api_key: str | None = Field(
        default=None,
        description="Dynamic OpenAI API key to use for real models, bypassing .env.",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Dynamic Anthropic API key to use for real models, bypassing .env.",
    )
    custom_base_url: str | None = Field(
        default=None,
        description="Custom base URL for OpenAI-compatible endpoints (e.g. OpenRouter, Groq).",
    )


class TaskResultResponse(BaseModel):
    """Serialised task result for the API."""

    id: str
    run_id: str
    task_id: str
    score: float
    latency_ms: float
    tokens_used: int
    cost_usd: float
    raw_output: str | None = None
    expected_output: str | None = None
    scoring_method: str
    metadata_json: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class EvalRunResponse(BaseModel):
    """Full evaluation run, including nested task results."""

    id: str
    model_name: str
    benchmark_name: str
    eval_type: str
    status: str
    total_score: float | None = None
    total_tasks: int
    completed_tasks: int
    created_at: datetime
    completed_at: datetime | None = None
    config_json: str | None = None
    task_results: list[TaskResultResponse] = []

    model_config = {"from_attributes": True}


class MetricsResponse(BaseModel):
    """Aggregated metrics for one evaluation run."""

    run_id: str
    accuracy: float
    avg_latency_ms: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    total_tokens: int
    total_cost_usd: float
    score_distribution: dict[str, int] = {}
    per_task_scores: list[dict[str, Any]] = []


class CompareResponse(BaseModel):
    """Side-by-side comparison of multiple runs."""

    run_ids: list[str]
    metrics: list[MetricsResponse]
    dimension_comparison: dict[str, dict[str, float]] = {}


class ModelInfo(BaseModel):
    """Metadata about an available model / adapter."""

    name: str
    adapter_type: str  # llm | agent | rag
    description: str = ""
    config: dict[str, Any] = {}
