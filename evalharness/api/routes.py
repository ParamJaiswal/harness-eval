"""FastAPI API routes for the AI Eval Harness dashboard and CLI.

Defines all endpoints for starting runs, polling statuses, retrieving metrics,
comparing multiple model runs, listing available benchmarks/models/tools,
and exporting results.

Customer endpoints
------------------
When ``endpoint_url`` is provided in ``POST /api/eval/run``, the harness
builds a live HTTP adapter on-the-fly (bypassing the mock registry) and stores
it in ``_LIVE_ADAPTERS`` keyed by ``run_id``. The background task picks it up
from there.
"""

from __future__ import annotations

import csv
import json
import logging
from io import StringIO
from typing import Any

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from evalharness.adapters import ADAPTER_REGISTRY
from evalharness.adapters.base import LLMAdapter, AgentAdapter, RAGAdapter
from evalharness.adapters.http_llm_adapter import build_llm_adapter
from evalharness.adapters.http_rag_adapter import RAGWebhookAdapter
from evalharness.adapters.http_agent_adapter import AgentWebhookAdapter
from evalharness.benchmarks.loader import BenchmarkLoader
from evalharness.config import get_settings
from evalharness.engine.runner import EvalRunner
from evalharness.engine.scorer import compute_aggregate_metrics
from evalharness.models.schemas import (
    EvalRun,
    EvalRunCreate,
    EvalRunResponse,
    MetricsResponse,
    CompareResponse,
    ModelInfo,
    TaskResultResponse,
    TaskResult,
    async_session_maker,
)
from evalharness.tools.mock_tools import create_default_tool_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Global instances
loader = BenchmarkLoader()
runner = EvalRunner()
tool_registry = create_default_tool_registry()

# Temporary store for live customer adapters (keyed by run_id)
# They live only as long as the background task needs them.
_LIVE_ADAPTERS: dict[str, LLMAdapter | AgentAdapter | RAGAdapter] = {}


# -- Helper functions --------------------------------------------------------


async def _run_evaluation_background(
    run_id: str,
    model_name: str,
    benchmark_name: str,
    eval_type: str,
    live_adapter: LLMAdapter | AgentAdapter | RAGAdapter | None = None,
    custom_tasks: list[dict] | None = None,
) -> None:
    """Background task execution — supports both registry and live HTTP adapters."""
    try:
        # If a live adapter was provided (customer endpoint), inject it temporarily
        if live_adapter is not None:
            _orig = ADAPTER_REGISTRY.get(model_name)
            ADAPTER_REGISTRY[model_name] = live_adapter

        if benchmark_name == "custom" and custom_tasks:
            from evalharness.benchmarks.loader import Benchmark, BenchmarkTask
            tasks = [
                BenchmarkTask(
                    id=str(t.get("id", f"task_{i}")),
                    type=str(t.get("type", "llm_task")),
                    prompt=str(t.get("prompt", "")),
                    expected_output=str(t.get("expected_output", "")),
                    scoring=str(t.get("scoring", "exact_match")),
                )
                for i, t in enumerate(custom_tasks)
            ]
            benchmark = Benchmark(
                name="custom",
                description="Custom benchmark from UI",
                category=eval_type,
                version="1.0",
                tasks=tasks,
            )
        else:
            benchmark = loader.get_benchmark(benchmark_name)
            
        await runner.run_evaluation(run_id, model_name, benchmark, eval_type)
    except Exception as e:
        logger.exception(f"Background run {run_id} failed: {e}")
        async with async_session_maker() as session:
            stmt = select(EvalRun).where(EvalRun.id == run_id)
            res = await session.execute(stmt)
            eval_run = res.scalar_one_or_none()
            if eval_run:
                eval_run.status = "failed"
                eval_run.config_json = json.dumps({"error": str(e)})
                await session.commit()
    finally:
        # Clean up the live adapter — remove it from the registry
        if live_adapter is not None and ADAPTER_REGISTRY.get(model_name) is live_adapter:
            del ADAPTER_REGISTRY[model_name]
        _LIVE_ADAPTERS.pop(run_id, None)


# -- Endpoints ---------------------------------------------------------------


@router.post("/eval/run", status_code=202)
async def start_evaluation(payload: EvalRunCreate, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Start an evaluation run as a background task.

    Supports **two modes**:

    1. **Mock/registry mode** (demo): ``endpoint_url`` is empty →
       looks up ``model_name`` in the built-in adapter registry.

    2. **Customer endpoint mode**: ``endpoint_url`` is provided →
       builds a live HTTP adapter from ``provider_type``, ``api_key``,
       and ``model_id`` and runs the evaluation against the real endpoint.

    Returns the created run ID and ``"pending"`` status.
    """
    live_adapter: LLMAdapter | AgentAdapter | RAGAdapter | None = None

    if payload.endpoint_url:
        # ── Customer endpoint mode ──────────────────────────────────────
        try:
            pt = (payload.provider_type or "openai_compatible").lower()
            name = payload.display_name or payload.model_name

            if pt in ("rag_webhook", "rag"):
                live_adapter = RAGWebhookAdapter(
                    endpoint_url=payload.endpoint_url,
                    api_key=payload.api_key,
                    top_k=payload.top_k,
                    extra_headers=payload.extra_headers,
                    display_name=name,
                )
            elif pt in ("agent_webhook", "agent"):
                live_adapter = AgentWebhookAdapter(
                    endpoint_url=payload.endpoint_url,
                    api_key=payload.api_key,
                    extra_headers=payload.extra_headers,
                    display_name=name,
                )
            else:
                live_adapter = build_llm_adapter(
                    provider_type=pt,
                    endpoint_url=payload.endpoint_url,
                    api_key=payload.api_key,
                    model_id=payload.model_id or payload.model_name,
                    extra_headers=payload.extra_headers,
                    display_name=name,
                )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    else:
        # ── Registry mode (mock adapters / pre-registered) ──────────────
        if payload.openai_api_key and "mock" not in payload.model_name.lower() and ("gpt" in payload.model_name.lower() or payload.custom_base_url):
            if payload.custom_base_url:
                from evalharness.adapters.http_llm_adapter import OpenAICompatibleAdapter
                live_adapter = OpenAICompatibleAdapter(
                    endpoint_url=payload.custom_base_url,
                    api_key=payload.openai_api_key,
                    model_id=payload.model_name,
                )
            else:
                from evalharness.adapters.openai_adapter import OpenAILLMAdapter
                live_adapter = OpenAILLMAdapter(model=payload.model_name, api_key=payload.openai_api_key)
        elif payload.anthropic_api_key and "mock" not in payload.model_name.lower() and ("claude" in payload.model_name.lower() or "anthropic" in payload.model_name.lower()):
            from evalharness.adapters.http_llm_adapter import AnthropicAdapter
            live_adapter = AnthropicAdapter(
                api_key=payload.anthropic_api_key,
                model_id=payload.model_name,
            )
        elif payload.model_name not in ADAPTER_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model adapter '{payload.model_name}' not found in registry. "
                    f"To evaluate a custom endpoint, provide 'endpoint_url' in the request."
                ),
            )

    # Verify benchmark exists
    try:
        if payload.benchmark_name != "custom":
            loader.get_benchmark(payload.benchmark_name)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Benchmark '{payload.benchmark_name}' not found.",
        )

    # Create EvalRun in DB
    config = {}
    if live_adapter is not None:
        config = {
            "provider_type": payload.provider_type,
            "endpoint_url": payload.endpoint_url,
            "model_id": payload.model_id,
        }

    async with async_session_maker() as session:
        eval_run = EvalRun(
            model_name=payload.display_name or payload.model_name,
            benchmark_name=payload.benchmark_name,
            eval_type=payload.eval_type,
            status="pending",
            config_json=json.dumps(config) if config else None,
        )
        session.add(eval_run)
        await session.commit()
        run_id = eval_run.id

    if live_adapter is not None:
        _LIVE_ADAPTERS[run_id] = live_adapter

    background_tasks.add_task(
        _run_evaluation_background,
        run_id=eval_run.id,
        model_name=payload.display_name or payload.model_name,
        benchmark_name=payload.benchmark_name,
        eval_type=payload.eval_type,
        live_adapter=live_adapter,
        custom_tasks=payload.custom_tasks,
    )

    return {"id": run_id, "status": "pending"}


@router.get("/eval/runs", response_model=list[EvalRunResponse])
async def list_runs(
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
    eval_type: str | None = None,
) -> list[EvalRunResponse]:
    """List historical evaluation runs with filtering and pagination."""
    async with async_session_maker() as session:
        stmt = select(EvalRun).order_by(EvalRun.created_at.desc()).offset(offset).limit(limit)
        if status:
            stmt = stmt.where(EvalRun.status == status)
        if eval_type:
            stmt = stmt.where(EvalRun.eval_type == eval_type)
        res = await session.execute(stmt)
        runs = res.scalars().all()
        return [EvalRunResponse.model_validate(r) for r in runs]


@router.get("/eval/runs/{run_id}", response_model=EvalRunResponse)
async def get_run_details(run_id: str) -> EvalRunResponse:
    """Get complete run details including nested task results."""
    async with async_session_maker() as session:
        stmt = select(EvalRun).where(EvalRun.id == run_id)
        res = await session.execute(stmt)
        run = res.scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=404, detail="Evaluation run not found.")
        return EvalRunResponse.model_validate(run)


@router.delete("/eval/runs/{run_id}", status_code=204)
async def delete_run(run_id: str) -> None:
    """Delete an evaluation run and all its associated task results."""
    async with async_session_maker() as session:
        stmt = select(EvalRun).where(EvalRun.id == run_id)
        res = await session.execute(stmt)
        run = res.scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=404, detail="Evaluation run not found.")
        await session.delete(run)
        await session.commit()
        logger.info(f"Deleted evaluation run {run_id}")


@router.get("/eval/runs/{run_id}/trajectory/{task_id}")
async def get_agent_trajectory(run_id: str, task_id: str) -> dict[str, Any]:
    """Retrieve the step-by-step agent trajectory for a specific task."""
    async with async_session_maker() as session:
        stmt = (
            select(TaskResult)
            .where(TaskResult.run_id == run_id)
            .where(TaskResult.task_id == task_id)
        )
        res = await session.execute(stmt)
        result = res.scalar_one_or_none()
        if not result:
            raise HTTPException(status_code=404, detail="Task result not found.")

        if not result.metadata_json:
            return {"steps": [], "error": "No trajectory metadata captured."}

        meta = json.loads(result.metadata_json)
        trace = meta.get("trace", {})
        scorecard = meta.get("scorecard", {})
        return {"trace": trace, "scorecard": scorecard}


@router.get("/eval/compare", response_model=CompareResponse)
async def compare_runs(run_ids: str = Query(..., description="Comma-separated run IDs")) -> CompareResponse:
    """Compare multiple runs side by side along dimensions like accuracy, latency, cost."""
    ids = [rid.strip() for rid in run_ids.split(",") if rid.strip()]
    if len(ids) < 1:
        raise HTTPException(status_code=400, detail="At least one run ID is required.")

    async with async_session_maker() as session:
        stmt = select(EvalRun).where(EvalRun.id.in_(ids))
        res = await session.execute(stmt)
        runs = res.scalars().all()

        if len(runs) != len(ids):
            found_ids = {r.id for r in runs}
            missing = set(ids) - found_ids
            raise HTTPException(status_code=404, detail=f"Runs not found: {list(missing)}")

        metrics_list = []
        dimension_comparison: dict[str, dict[str, float]] = {
            "accuracy": {},
            "latency": {},
            "cost": {},
            "consistency": {},
            "tool_accuracy": {},
            "step_efficiency": {},
            "faithfulness": {},
            "context_relevance": {},
            "retrieval_precision": {},
        }

        # Find max average latency and cost across compared runs for normalization
        avg_latencies = []
        total_costs = []

        # First pass to compute basic metrics
        for run in runs:
            results = run.task_results
            raw_metrics = compute_aggregate_metrics(results, run_id=run.id)
            metrics_response = MetricsResponse(
                run_id=run.id,
                accuracy=raw_metrics["accuracy"],
                avg_latency_ms=raw_metrics["avg_latency_ms"],
                p50_latency=raw_metrics["p50_latency"],
                p95_latency=raw_metrics["p95_latency"],
                p99_latency=raw_metrics["p99_latency"],
                total_tokens=raw_metrics["total_tokens"],
                total_cost_usd=raw_metrics["total_cost_usd"],
                score_distribution=raw_metrics["score_distribution"],
                per_task_scores=raw_metrics["per_task_scores"],
            )
            metrics_list.append(metrics_response)
            avg_latencies.append(raw_metrics["avg_latency_ms"])
            total_costs.append(raw_metrics["total_cost_usd"])

        max_latency = max(avg_latencies) if avg_latencies else 1.0
        max_cost = max(total_costs) if total_costs else 1.0

        for idx, run in enumerate(runs):
            metrics = metrics_list[idx]
            run_id = run.id
            results = run.task_results
            scores = [r.score for r in results]

            # Accuracy (0.0 - 1.0)
            dimension_comparison["accuracy"][run_id] = metrics.accuracy

            # Latency Score (Normalized: 1.0 is low latency, 0.0 is high)
            norm_latency = 1.0 - (metrics.avg_latency_ms / max(1000.0, max_latency))
            dimension_comparison["latency"][run_id] = max(0.0, norm_latency)

            # Cost Score (Normalized: 1.0 is cheap/zero cost, 0.0 is expensive)
            norm_cost = 1.0 - (metrics.total_cost_usd / max(0.01, max_cost))
            dimension_comparison["cost"][run_id] = max(0.0, norm_cost)

            # Consistency Score (1.0 - standard deviation of task scores)
            std = np.std(scores) if len(scores) > 1 else 0.0
            dimension_comparison["consistency"][run_id] = max(0.0, 1.0 - std)

            # Extract adapter-specific scores from result metadata
            tool_accs = []
            step_effs = []
            faithfulness_vals = []
            context_rels = []
            retrieval_precs = []

            for r in results:
                if r.metadata_json:
                    try:
                        meta = json.loads(r.metadata_json)
                        card = meta.get("scorecard", {})
                        if "tool_selection_accuracy" in card:
                            tool_accs.append(card["tool_selection_accuracy"])
                        if "step_efficiency" in card:
                            step_effs.append(card["step_efficiency"])
                        if "faithfulness" in card:
                            faithfulness_vals.append(card["faithfulness"])
                        if "context_relevance" in card:
                            context_rels.append(card["context_relevance"])
                        if "retrieval_precision" in card:
                            retrieval_precs.append(card["retrieval_precision"])
                    except Exception:
                        pass

            dimension_comparison["tool_accuracy"][run_id] = (
                float(np.mean(tool_accs)) if tool_accs else 0.0
            )
            dimension_comparison["step_efficiency"][run_id] = (
                float(np.mean(step_effs)) if step_effs else 0.0
            )
            dimension_comparison["faithfulness"][run_id] = (
                float(np.mean(faithfulness_vals)) if faithfulness_vals else 0.0
            )
            dimension_comparison["context_relevance"][run_id] = (
                float(np.mean(context_rels)) if context_rels else 0.0
            )
            dimension_comparison["retrieval_precision"][run_id] = (
                float(np.mean(retrieval_precs)) if retrieval_precs else 0.0
            )

        # Cleanup empty comparison arrays if they are not relevant to the evaluation runs compared
        # (e.g. if comparing only LLMs, agent/RAG dimensions will be all 0.0s). The frontend chart
        # will filter them or handle them.
        return CompareResponse(
            run_ids=ids,
            metrics=metrics_list,
            dimension_comparison=dimension_comparison,
        )


@router.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """List all available model/agent/pipeline profiles in the adapter registry."""
    models = []
    for name, adapter in ADAPTER_REGISTRY.items():
        if isinstance(adapter, LLMAdapter):
            info = getattr(adapter, "get_model_info", lambda: {})()
            models.append(
                ModelInfo(
                    name=name,
                    adapter_type="llm",
                    description=info.get("description", "Mock LLM adapter profile"),
                    config=info.get("config", {}),
                )
            )
        elif isinstance(adapter, AgentAdapter):
            info = getattr(adapter, "get_model_info", lambda: {})()
            models.append(
                ModelInfo(
                    name=name,
                    adapter_type="agent",
                    description=info.get("description", "Mock Agent adapter profile"),
                    config=info.get("config", {}),
                )
            )
        elif isinstance(adapter, RAGAdapter):
            info = getattr(adapter, "get_pipeline_info", lambda: {})()
            models.append(
                ModelInfo(
                    name=name,
                    adapter_type="rag",
                    description=info.get("description", "Mock RAG adapter profile"),
                    config=info.get("config", {}),
                )
            )
    return models


@router.get("/benchmarks")
async def list_benchmarks() -> list[dict[str, Any]]:
    """List all available benchmark YAML files loaded from the directory."""
    return loader.list_benchmarks()


@router.get("/tools")
async def list_tools() -> list[dict[str, Any]]:
    """List all available tools registered for agent evaluation."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
        for t in tool_registry.list_tools()
    ]


@router.get("/metrics/{run_id}", response_model=MetricsResponse)
async def get_run_metrics(run_id: str) -> MetricsResponse:
    """Retrieve computed aggregate metrics for a specific run."""
    async with async_session_maker() as session:
        stmt = select(EvalRun).where(EvalRun.id == run_id)
        res = await session.execute(stmt)
        run = res.scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=404, detail="Evaluation run not found.")

        results = run.task_results
        raw_metrics = compute_aggregate_metrics(results, run_id=run_id)

        return MetricsResponse(
            run_id=run_id,
            accuracy=raw_metrics["accuracy"],
            avg_latency_ms=raw_metrics["avg_latency_ms"],
            p50_latency=raw_metrics["p50_latency"],
            p95_latency=raw_metrics["p95_latency"],
            p99_latency=raw_metrics["p99_latency"],
            total_tokens=raw_metrics["total_tokens"],
            total_cost_usd=raw_metrics["total_cost_usd"],
            score_distribution=raw_metrics["score_distribution"],
            per_task_scores=raw_metrics["per_task_scores"],
        )


@router.get("/export/{run_id}")
async def export_run(run_id: str, format: str = "json") -> StreamingResponse:
    """Export evaluation results for a run in JSON or CSV format."""
    async with async_session_maker() as session:
        stmt = select(EvalRun).where(EvalRun.id == run_id)
        res = await session.execute(stmt)
        run = res.scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=404, detail="Evaluation run not found.")

        if format.lower() == "csv":
            output = StringIO()
            writer = csv.writer(output)
            # Write header
            writer.writerow([
                "run_id",
                "model_name",
                "benchmark_name",
                "eval_type",
                "task_id",
                "score",
                "latency_ms",
                "tokens_used",
                "cost_usd",
                "scoring_method",
                "raw_output",
                "expected_output",
            ])
            for result in run.task_results:
                writer.writerow([
                    run.id,
                    run.model_name,
                    run.benchmark_name,
                    run.eval_type,
                    result.task_id,
                    result.score,
                    result.latency_ms,
                    result.tokens_used,
                    result.cost_usd,
                    result.scoring_method,
                    result.raw_output,
                    result.expected_output,
                ])
            output.seek(0)
            return StreamingResponse(
                output,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=run_{run_id}.csv"},
            )
        else:
            # Default JSON
            data = EvalRunResponse.model_validate(run).model_dump(mode="json")
            json_str = json.dumps(data, indent=2)
            return StreamingResponse(
                StringIO(json_str),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=run_{run_id}.json"},
            )
