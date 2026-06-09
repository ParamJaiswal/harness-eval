"""Evaluation runner engine — manages execution of evaluation tasks.

Supports LLM, Agent, and RAG adapters. Runs evaluation runs in parallel with
concurrency limits, logs progress, averages runs for statistical confidence,
scores results using specific metrics, and commits results to the database.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from evalharness.adapters import ADAPTER_REGISTRY
from evalharness.adapters.base import LLMAdapter, AgentAdapter, RAGAdapter
from evalharness.config import get_settings
from evalharness.engine.scorer import compute_score, compute_aggregate_metrics
from evalharness.engine.judge import LLMJudge
from evalharness.engine.trajectory import score_trajectory
from evalharness.engine.rag_scorer import score_rag_response
from evalharness.models.schemas import EvalRun, TaskResult, async_session_maker

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EvalRunner:
    """Orchestrates evaluation runs across LLMs, Agents, and RAG pipelines."""

    def __init__(self, db_session: Any = None, settings: Any = None) -> None:
        self.db_session = db_session
        self.settings = settings or get_settings()
        self.judge = LLMJudge()
        # Lazily register default mock tools
        from evalharness.tools.mock_tools import create_default_tool_registry
        self.tool_registry = create_default_tool_registry()

    async def run_evaluation(
        self, run_id: str, model_name: str, benchmark: Any, eval_type: str
    ) -> EvalRun:
        """Run a complete evaluation: load benchmark, execute all tasks, score results."""
        logger.info(
            f"Starting evaluation run {run_id} for model {model_name} "
            f"on benchmark {benchmark.name} ({eval_type})"
        )

        # 1. Update EvalRun status to running
        async with async_session_maker() as session:
            stmt = select(EvalRun).where(EvalRun.id == run_id)
            res = await session.execute(stmt)
            eval_run = res.scalar_one_or_none()
            if not eval_run:
                raise ValueError(f"EvalRun {run_id} not found in database.")

            eval_run.status = "running"
            eval_run.total_tasks = len(benchmark.tasks)
            eval_run.completed_tasks = 0
            await session.commit()

        # 2. Get the adapter
        adapter = ADAPTER_REGISTRY.get(model_name)
        if not adapter:
            async with async_session_maker() as session:
                stmt = select(EvalRun).where(EvalRun.id == run_id)
                res = await session.execute(stmt)
                eval_run = res.scalar_one_or_none()
                if eval_run:
                    eval_run.status = "failed"
                    eval_run.completed_at = _utcnow()
                    await session.commit()
            raise ValueError(f"Model adapter {model_name!r} not found in registry.")

        # 3. Semaphore for concurrency control
        sem = asyncio.Semaphore(self.settings.DEFAULT_CONCURRENCY)

        async def run_task_wrapper(task: Any) -> None:
            async with sem:
                scores_list = []
                latencies_list = []
                tokens_list = []
                costs_list = []
                outputs_list = []
                metadatas_list = []

                # Run each task EVAL_RUNS_PER_TASK times for statistical rigor
                runs_count = max(1, self.settings.EVAL_RUNS_PER_TASK)

                for _ in range(runs_count):
                    task_start = time.time()
                    try:
                        if eval_type == "llm":
                            output, task_meta = await self._execute_llm_task(adapter, task)
                            # Score LLM output
                            if task.scoring == "llm_judge":
                                raw_score = await self.judge.judge_response(task.prompt, output)
                                task_score = raw_score / 100.0
                            else:
                                task_score = compute_score(output, task.expected_output, task.scoring)

                        elif eval_type == "agent":
                            # Resolve available tools
                            tools = []
                            for tool_name in getattr(task, "available_tools", []):
                                try:
                                    tools.append(self.tool_registry.get_tool(tool_name))
                                except KeyError:
                                    logger.warning(f"Tool {tool_name!r} not found in registry.")

                            output, task_meta = await self._execute_agent_task(adapter, task, tools)
                            
                            # Score Agent trajectory
                            trace = task_meta.get("trace")
                            if trace:
                                scorecard = score_trajectory(
                                    trace,
                                    getattr(task, "expected_tools", []),
                                    getattr(task, "max_steps", 5),
                                )
                                task_score = scorecard.overall_score
                                task_meta["scorecard"] = {
                                    "tool_selection_accuracy": scorecard.tool_selection_accuracy,
                                    "parameter_accuracy": scorecard.parameter_accuracy,
                                    "step_efficiency": scorecard.step_efficiency,
                                    "error_recovery": scorecard.error_recovery,
                                    "goal_completion": scorecard.goal_completion,
                                    "overall_score": scorecard.overall_score,
                                }
                            else:
                                task_score = 0.0

                        elif eval_type == "rag":
                            output, task_meta = await self._execute_rag_task(adapter, task)
                            
                            # Score RAG pipeline output
                            contexts = task_meta.get("retrieved_contexts", [])
                            scorecard = score_rag_response(
                                question=task.prompt,
                                answer=output,
                                retrieved_contexts=contexts,
                                ground_truth_contexts=getattr(task, "ground_truth_contexts", []),
                                ground_truth_answer=getattr(task, "ground_truth_answer", "") or task.expected_output or "",
                            )
                            task_score = scorecard.overall_score
                            task_meta["scorecard"] = {
                                "faithfulness": scorecard.faithfulness,
                                "context_relevance": scorecard.context_relevance,
                                "answer_relevance": scorecard.answer_relevance,
                                "retrieval_precision": scorecard.retrieval_precision,
                                "retrieval_recall": scorecard.retrieval_recall,
                                "overall_score": scorecard.overall_score,
                            }
                        else:
                            raise ValueError(f"Unknown eval_type {eval_type!r}")

                        task_latency = (time.time() - task_start) * 1000.0
                        scores_list.append(task_score)
                        latencies_list.append(task_latency)
                        tokens_list.append(task_meta.get("tokens_used", 0))
                        costs_list.append(task_meta.get("cost_usd", 0.0))
                        outputs_list.append(output)

                        # Clean metadata serialization
                        if "trace" in task_meta:
                            t_obj = task_meta["trace"]
                            task_meta["trace"] = {
                                "success": t_obj.success,
                                "final_answer": t_obj.final_answer,
                                "total_tokens": t_obj.total_tokens,
                                "total_latency_ms": t_obj.total_latency_ms,
                                "total_cost_usd": t_obj.total_cost_usd,
                                "error": t_obj.error,
                                "steps": [
                                    {
                                        "step_number": s.step_number,
                                        "thought": s.thought,
                                        "tool_name": s.tool_name,
                                        "tool_input": s.tool_input,
                                        "tool_output": s.tool_output,
                                        "step_latency_ms": s.step_latency_ms,
                                        "tokens_used": s.tokens_used,
                                    }
                                    for s in t_obj.steps
                                ],
                            }
                        if "retrieved_contexts" in task_meta:
                            c_objs = task_meta["retrieved_contexts"]
                            task_meta["retrieved_contexts"] = [
                                {
                                    "text": c.text,
                                    "source": c.source,
                                    "relevance_score": c.relevance_score,
                                    "chunk_id": c.chunk_id,
                                }
                                for c in c_objs
                            ]

                        metadatas_list.append(task_meta)

                    except Exception as e:
                        logger.exception(f"Error running task {task.id}: {e}")
                        scores_list.append(0.0)
                        latencies_list.append((time.time() - task_start) * 1000.0)
                        tokens_list.append(0)
                        costs_list.append(0.0)
                        outputs_list.append(f"ERROR: {str(e)}")
                        metadatas_list.append({"error": str(e)})

                # Average the metrics across runs
                score = sum(scores_list) / len(scores_list)
                latency_ms = sum(latencies_list) / len(latencies_list)
                tokens_used = int(sum(tokens_list) / len(tokens_list))
                cost_usd = sum(costs_list) / len(costs_list)
                raw_output = outputs_list[0] if outputs_list else ""
                metadata = metadatas_list[0] if metadatas_list else {}
                metadata["runs"] = {
                    "scores": scores_list,
                    "latencies": latencies_list,
                    "tokens": tokens_list,
                    "costs": costs_list,
                }

                # Save TaskResult in DB
                async with async_session_maker() as session:
                    task_result = TaskResult(
                        run_id=run_id,
                        task_id=task.id,
                        score=score,
                        latency_ms=latency_ms,
                        tokens_used=tokens_used,
                        cost_usd=cost_usd,
                        raw_output=raw_output,
                        expected_output=task.expected_output or getattr(task, "ground_truth_answer", ""),
                        scoring_method=task.scoring,
                        metadata_json=json.dumps(metadata),
                        created_at=_utcnow(),
                    )
                    session.add(task_result)

                    # Update progress in EvalRun
                    stmt = select(EvalRun).where(EvalRun.id == run_id)
                    res = await session.execute(stmt)
                    r = res.scalar_one()
                    r.completed_tasks += 1
                    await session.commit()

        # Run all tasks concurrently with Semaphore control
        await asyncio.gather(*(run_task_wrapper(task) for task in benchmark.tasks))

        # Compute aggregate metrics and update run
        async with async_session_maker() as session:
            stmt = select(EvalRun).where(EvalRun.id == run_id)
            res = await session.execute(stmt)
            eval_run = res.scalar_one()

            stmt_results = select(TaskResult).where(TaskResult.run_id == run_id)
            res_results = await session.execute(stmt_results)
            task_results = res_results.scalars().all()

            metrics = compute_aggregate_metrics(task_results, run_id=run_id)

            eval_run.status = "completed"
            eval_run.total_score = metrics["accuracy"]
            eval_run.completed_at = _utcnow()

            await session.commit()

            # Reload to return the fresh state
            stmt = select(EvalRun).where(EvalRun.id == run_id)
            res = await session.execute(stmt)
            eval_run = res.scalar_one()
            return eval_run

    async def _execute_llm_task(self, adapter: LLMAdapter, task: Any) -> tuple[str, dict]:
        """Execute a single LLM task and return (output, metadata)."""
        response = await adapter.generate(task.prompt)
        metadata = {
            "tokens_used": response.tokens_used,
            "cost_usd": response.cost_usd,
            "model": response.model,
            "metadata": response.metadata,
        }
        return response.text, metadata

    async def _execute_agent_task(
        self, adapter: AgentAdapter, task: Any, tools: list
    ) -> tuple[str, dict]:
        """Execute a single agent task and return (output, metadata with trajectory)."""
        trace = await adapter.execute_task(
            task.prompt, tools, max_steps=getattr(task, "max_steps", 10)
        )
        metadata = {
            "tokens_used": trace.total_tokens,
            "cost_usd": trace.total_cost_usd,
            "trace": trace,
        }
        return trace.final_answer, metadata

    async def _execute_rag_task(self, adapter: RAGAdapter, task: Any) -> tuple[str, dict]:
        """Execute a single RAG task and return (output, metadata with contexts)."""
        response = await adapter.query(task.prompt)
        metadata = {
            "tokens_used": response.tokens_used,
            "cost_usd": response.cost_usd,
            "retrieved_contexts": response.retrieved_contexts,
            "retrieval_latency_ms": response.retrieval_latency_ms,
            "generation_latency_ms": response.generation_latency_ms,
        }
        return response.answer, metadata
