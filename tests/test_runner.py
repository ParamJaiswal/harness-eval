"""Async integration tests for the EvalRunner engine using mock adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

import evalharness.models.schemas as _schemas
from evalharness.config import get_settings
from evalharness.engine.runner import EvalRunner
from evalharness.models.schemas import init_db, async_session_maker, EvalRun, TaskResult


# ---------------------------------------------------------------------------
# Mock Benchmark Classes
# ---------------------------------------------------------------------------


@dataclass
class MockBenchmarkTask:
    """Mock structure for benchmark tasks used in testing."""

    id: str
    type: str
    prompt: str
    expected_output: str
    scoring: str
    difficulty: str = "easy"
    tags: list[str] = field(default_factory=list)
    available_tools: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    max_steps: int = 5
    ground_truth_contexts: list[str] = field(default_factory=list)
    ground_truth_answer: str = ""


@dataclass
class MockBenchmark:
    """Mock structure for benchmark suites used in testing."""

    name: str
    description: str
    category: str
    version: str
    tasks: list[MockBenchmarkTask]


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
async def setup_db() -> None:
    """Wire an in-memory SQLite engine for all runner tests."""
    # Directly replace the module-level singletons so all code paths use the
    # in-memory DB — this avoids touching the file-system database.
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    _schemas._engine = test_engine
    _schemas._session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    # Speed up tests
    settings = get_settings()
    settings.EVAL_RUNS_PER_TASK = 1
    settings.DEFAULT_CONCURRENCY = 2
    await init_db(engine=test_engine)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_llm_evaluation() -> None:
    """Test LLM run workflow with mock LLM adapter."""
    # 1. Create EvalRun in DB
    async with async_session_maker() as session:
        run = EvalRun(
            model_name="gpt-4o-mock",
            benchmark_name="test_llm_bench",
            eval_type="llm",
            status="pending",
        )
        session.add(run)
        await session.commit()
        run_id = run.id

    # 2. Build mock benchmark
    task = MockBenchmarkTask(
        id="t1",
        type="llm_task",
        prompt="capital of France",
        expected_output="Paris",
        scoring="exact_match",
    )
    benchmark = MockBenchmark(
        name="test_llm_bench",
        description="test llm",
        category="llm",
        version="1.0",
        tasks=[task],
    )

    # 3. Execute runner
    runner = EvalRunner()
    completed_run = await runner.run_evaluation(
        run_id=run_id,
        model_name="gpt-4o-mock",
        benchmark=benchmark,
        eval_type="llm",
    )

    # 4. Verify results
    assert completed_run.status == "completed"
    assert completed_run.total_tasks == 1
    assert completed_run.completed_tasks == 1
    assert completed_run.total_score is not None
    assert completed_run.total_score >= 0.0

    async with async_session_maker() as session:
        stmt = select(TaskResult).where(TaskResult.run_id == run_id)
        res = await session.execute(stmt)
        results = res.scalars().all()
        assert len(results) == 1
        assert results[0].task_id == "t1"
        assert results[0].score is not None


@pytest.mark.asyncio
async def test_run_agent_evaluation() -> None:
    """Test Agent run workflow with mock Agent adapter."""
    async with async_session_maker() as session:
        run = EvalRun(
            model_name="agent-expert-mock",
            benchmark_name="test_agent_bench",
            eval_type="agent",
            status="pending",
        )
        session.add(run)
        await session.commit()
        run_id = run.id

    task = MockBenchmarkTask(
        id="t_agent",
        type="agent_task",
        prompt="Calculate sum of 5 and 10",
        expected_output="15",
        scoring="exact_match",
        available_tools=["calculator"],
        expected_tools=["calculator"],
        max_steps=5,
    )
    benchmark = MockBenchmark(
        name="test_agent_bench",
        description="test agent",
        category="agent",
        version="1.0",
        tasks=[task],
    )

    runner = EvalRunner()
    completed_run = await runner.run_evaluation(
        run_id=run_id,
        model_name="agent-expert-mock",
        benchmark=benchmark,
        eval_type="agent",
    )

    assert completed_run.status == "completed"
    assert completed_run.total_tasks == 1
    assert completed_run.completed_tasks == 1

    async with async_session_maker() as session:
        stmt = select(TaskResult).where(TaskResult.run_id == run_id)
        res = await session.execute(stmt)
        results = res.scalars().all()
        assert len(results) == 1
        assert results[0].score is not None
        assert "scorecard" in results[0].metadata_json


@pytest.mark.asyncio
async def test_run_rag_evaluation() -> None:
    """Test RAG run workflow with mock RAG adapter."""
    async with async_session_maker() as session:
        run = EvalRun(
            model_name="rag-precise-mock",
            benchmark_name="test_rag_bench",
            eval_type="rag",
            status="pending",
        )
        session.add(run)
        await session.commit()
        run_id = run.id

    task = MockBenchmarkTask(
        id="t_rag",
        type="rag_task",
        prompt="What is photosynthesis?",
        expected_output="process by which plants convert sunlight",
        scoring="contains",
        ground_truth_contexts=["Photosynthesis is the process by which green plants convert sunlight..."],
        ground_truth_answer="Plants convert sunlight to glucose and oxygen.",
    )
    benchmark = MockBenchmark(
        name="test_rag_bench",
        description="test rag",
        category="rag",
        version="1.0",
        tasks=[task],
    )

    runner = EvalRunner()
    completed_run = await runner.run_evaluation(
        run_id=run_id,
        model_name="rag-precise-mock",
        benchmark=benchmark,
        eval_type="rag",
    )

    assert completed_run.status == "completed"
    assert completed_run.total_tasks == 1

    async with async_session_maker() as session:
        stmt = select(TaskResult).where(TaskResult.run_id == run_id)
        res = await session.execute(stmt)
        results = res.scalars().all()
        assert len(results) == 1
        assert "scorecard" in results[0].metadata_json
