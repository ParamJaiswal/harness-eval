"""Integration tests for dynamic API keys and custom benchmarks."""

from __future__ import annotations

import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

import evalharness.models.schemas as _schemas
from evalharness.models.schemas import init_db, EvalRun
from evalharness.main import app


@pytest.fixture(scope="module", autouse=True)
async def setup_api_db() -> None:
    """Wire a fresh in-memory SQLite engine for this test module."""
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    _schemas._engine = test_engine
    _schemas._session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    await init_db(engine=test_engine)


@pytest.fixture
def client():
    """Return an async HTTPX client backed by the ASGI app."""
    import httpx
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )


@pytest.mark.asyncio
async def test_run_custom_benchmark_success(client) -> None:
    """POST /api/eval/run with a custom benchmark should succeed and execute."""
    async with client as c:
        # Start evaluation
        payload = {
            "model_name": "gpt-4o-mock",
            "benchmark_name": "custom",
            "eval_type": "llm",
            "openai_api_key": "mock-openai-api-key",
            "custom_tasks": [
                {
                    "id": "custom_1",
                    "type": "llm_task",
                    "prompt": "What is 2+2?",
                    "expected_output": "4",
                    "scoring": "exact_match"
                }
            ]
        }
        r = await c.post("/api/eval/run", json=payload)
        assert r.status_code == 202
        body = r.json()
        assert "id" in body
        run_id = body["id"]
        
        # Poll run status until completed (with timeout)
        for _ in range(20):
            status_resp = await c.get(f"/api/eval/runs/{run_id}")
            assert status_resp.status_code == 200
            run_data = status_resp.json()
            if run_data["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.1)
            
        assert run_data["status"] == "completed"
        assert run_data["total_tasks"] == 1
        assert run_data["completed_tasks"] == 1
        assert "task_results" in run_data
        assert len(run_data["task_results"]) == 1
        result = run_data["task_results"][0]
        assert result["task_id"] == "custom_1"
        assert result["scoring_method"] == "exact_match"
