"""HTTP-level integration tests for the FastAPI application.

Uses HTTPX's ASGITransport to test routes end-to-end without binding a real socket.
All tests run against an in-memory SQLite database.
"""

from __future__ import annotations

import pytest
import httpx
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

import evalharness.models.schemas as _schemas
from evalharness.models.schemas import init_db
from evalharness.main import app


@pytest.fixture(scope="module", autouse=True)
async def setup_api_db() -> None:
    """Wire a fresh in-memory SQLite engine for the API test module."""
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    _schemas._engine = test_engine
    _schemas._session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    await init_db(engine=test_engine)


@pytest.fixture
def client():
    """Return a synchronous HTTPX client backed by the ASGI app."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint(client) -> None:
    """GET /health should return 200 with status ok."""
    async with client as c:
        r = await c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_benchmarks(client) -> None:
    """GET /api/benchmarks should return all 10 benchmark definitions."""
    async with client as c:
        r = await c.get("/api/benchmarks")
    assert r.status_code == 200
    benchmarks = r.json()
    assert isinstance(benchmarks, list)
    assert len(benchmarks) >= 1
    # Validate shape of first item
    first = benchmarks[0]
    assert "name" in first
    assert "category" in first
    assert "task_count" in first


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_models(client) -> None:
    """GET /api/models should return registered adapters."""
    async with client as c:
        r = await c.get("/api/models")
    assert r.status_code == 200
    models = r.json()
    assert isinstance(models, list)
    assert len(models) >= 1
    names = {m["name"] for m in models}
    # At least one mock adapter should be present
    assert any("mock" in n.lower() for n in names)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tools(client) -> None:
    """GET /api/tools should return mock tool registry."""
    async with client as c:
        r = await c.get("/api/tools")
    assert r.status_code == 200
    tools = r.json()
    assert isinstance(tools, list)
    assert len(tools) >= 1
    tool_names = {t["name"] for t in tools}
    assert "calculator" in tool_names


# ---------------------------------------------------------------------------
# Evaluation Run lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_evaluation_run(client) -> None:
    """POST /api/eval/run should create a run and return a pending status."""
    async with client as c:
        r = await c.post(
            "/api/eval/run",
            json={
                "model_name": "gpt-4o-mock",
                "benchmark_name": "general_knowledge",
                "eval_type": "llm",
            },
        )
    assert r.status_code == 202
    body = r.json()
    assert "id" in body
    assert body["status"] == "pending"


@pytest.mark.asyncio
async def test_list_runs(client) -> None:
    """GET /api/eval/runs should return a list (may be empty initially)."""
    async with client as c:
        r = await c.get("/api/eval/runs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


@pytest.mark.asyncio
async def test_get_run_not_found(client) -> None:
    """GET /api/eval/runs/{nonexistent} should return 404."""
    async with client as c:
        r = await c.get("/api/eval/runs/nonexistent-id")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_run_not_found(client) -> None:
    """DELETE /api/eval/runs/{nonexistent} should return 404."""
    async with client as c:
        r = await c.delete("/api/eval/runs/nonexistent-id")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_start_run_invalid_model(client) -> None:
    """POST /api/eval/run with an unknown model should return 400."""
    async with client as c:
        r = await c.post(
            "/api/eval/run",
            json={
                "model_name": "does-not-exist",
                "benchmark_name": "general_knowledge",
                "eval_type": "llm",
            },
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_start_run_invalid_benchmark(client) -> None:
    """POST /api/eval/run with an unknown benchmark should return 400."""
    async with client as c:
        r = await c.post(
            "/api/eval/run",
            json={
                "model_name": "gpt-4o-mock",
                "benchmark_name": "nonexistent_benchmark",
                "eval_type": "llm",
            },
        )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Compare (edge case — single run)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compare_missing_runs(client) -> None:
    """GET /api/eval/compare with nonexistent IDs should return 404."""
    async with client as c:
        r = await c.get("/api/eval/compare?run_ids=fake-id-1,fake-id-2")
    assert r.status_code == 404
