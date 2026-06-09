"""Scoring engine for LLM evaluation tasks.

Provides exact-match, contains, fuzzy-match, and regex scoring plus aggregate
metric computation (accuracy, latency percentiles, cost summaries).
"""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

import numpy as np
from thefuzz import fuzz

if TYPE_CHECKING:
    from evalharness.models.schemas import MetricsResponse, TaskResult


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------


def score_exact_match(output: str, expected: str) -> float:
    """Return 1.0 if *output* exactly matches *expected* (case-insensitive, stripped)."""
    return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0


def score_contains(output: str, expected: str) -> float:
    """Return 1.0 if *expected* is a substring of *output* (case-insensitive)."""
    return 1.0 if expected.strip().lower() in output.strip().lower() else 0.0


def score_fuzzy_match(output: str, expected: str, threshold: float = 0.8) -> float:
    """Return a 0.0–1.0 score based on fuzzy string similarity.

    Uses ``thefuzz.fuzz.token_sort_ratio`` for order-invariant comparison.
    If the ratio meets or exceeds *threshold* (expressed as 0–1), the raw
    ratio is returned; otherwise 0.0.
    """
    ratio = fuzz.token_sort_ratio(output.strip().lower(), expected.strip().lower()) / 100.0
    return ratio if ratio >= threshold else 0.0


def score_regex(output: str, pattern: str) -> float:
    """Return 1.0 if *pattern* matches anywhere in *output*."""
    try:
        return 1.0 if re.search(pattern, output, re.IGNORECASE) else 0.0
    except re.error:
        return 0.0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_SCORING_METHODS: dict[str, Any] = {
    "exact_match": score_exact_match,
    "contains": score_contains,
    "fuzzy_match": score_fuzzy_match,
    "regex": score_regex,
}


def compute_score(output: str, expected: str, method: str = "exact_match") -> float:
    """Dispatch to the right scoring function.

    Parameters
    ----------
    output:
        The model's raw output text.
    expected:
        The expected (gold) output text or regex pattern.
    method:
        One of ``exact_match``, ``contains``, ``fuzzy_match``, ``regex``.

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    scorer = _SCORING_METHODS.get(method)
    if scorer is None:
        raise ValueError(
            f"Unknown scoring method {method!r}. "
            f"Choose from: {list(_SCORING_METHODS)}"
        )
    return scorer(output, expected)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def compute_aggregate_metrics(results: list[Any], run_id: str = "") -> dict[str, Any]:
    """Compute aggregate metrics from a list of ``TaskResult`` rows.

    Parameters
    ----------
    results:
        Sequence of objects with ``.score``, ``.latency_ms``, ``.tokens_used``,
        ``.cost_usd``, and ``.task_id`` attributes (typically ORM ``TaskResult``
        instances).
    run_id:
        Identifier for the evaluation run.

    Returns
    -------
    dict
        A dictionary matching the ``MetricsResponse`` schema.
    """
    if not results:
        return {
            "run_id": run_id,
            "accuracy": 0.0,
            "avg_latency_ms": 0.0,
            "p50_latency": 0.0,
            "p95_latency": 0.0,
            "p99_latency": 0.0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "score_distribution": {},
            "per_task_scores": [],
        }

    scores = np.array([r.score for r in results], dtype=float)
    latencies = np.array([r.latency_ms for r in results], dtype=float)
    tokens = sum(r.tokens_used for r in results)
    costs = sum(r.cost_usd for r in results)

    # Score distribution buckets
    dist: dict[str, int] = {
        "perfect_1.0": int(np.sum(scores == 1.0)),
        "high_0.8-1.0": int(np.sum((scores >= 0.8) & (scores < 1.0))),
        "medium_0.5-0.8": int(np.sum((scores >= 0.5) & (scores < 0.8))),
        "low_0.2-0.5": int(np.sum((scores >= 0.2) & (scores < 0.5))),
        "fail_0.0-0.2": int(np.sum(scores < 0.2)),
    }

    per_task = [
        {
            "task_id": r.task_id,
            "score": r.score,
            "latency_ms": r.latency_ms,
            "tokens_used": r.tokens_used,
            "cost_usd": r.cost_usd,
        }
        for r in results
    ]

    return {
        "run_id": run_id,
        "accuracy": float(np.mean(scores)),
        "avg_latency_ms": float(np.mean(latencies)),
        "p50_latency": float(np.percentile(latencies, 50)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "p99_latency": float(np.percentile(latencies, 99)),
        "total_tokens": int(tokens),
        "total_cost_usd": float(round(costs, 8)),
        "score_distribution": dist,
        "per_task_scores": per_task,
    }
