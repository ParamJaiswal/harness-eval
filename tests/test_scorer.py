"""Tests for the scoring module of the AI Eval Harness."""

from __future__ import annotations

from dataclasses import dataclass

from evalharness.engine.scorer import (
    score_exact_match,
    score_contains,
    score_fuzzy_match,
    score_regex,
    compute_score,
    compute_aggregate_metrics,
)


@dataclass
class MockResult:
    """Mock structure mimicking TaskResult DB rows for scoring calculations."""

    task_id: str
    score: float
    latency_ms: float
    tokens_used: int
    cost_usd: float


def test_exact_match() -> None:
    """Test exact match scoring (case-insensitive, stripped)."""
    assert score_exact_match("Paris", "paris") == 1.0
    assert score_exact_match(" Paris  ", "paris") == 1.0
    assert score_exact_match("London", "paris") == 0.0


def test_contains() -> None:
    """Test substring contains scoring."""
    assert score_contains("The capital is Paris.", "Paris") == 1.0
    assert score_contains("paris is nice", "PARIS") == 1.0
    assert score_contains("Berlin is cold", "paris") == 0.0


def test_fuzzy_match() -> None:
    """Test fuzzy matching with threshold."""
    assert score_fuzzy_match("apple pie", "aple py", threshold=0.7) > 0.7
    assert score_fuzzy_match("apple pie", "banana", threshold=0.7) == 0.0


def test_regex() -> None:
    """Test regex matching."""
    assert score_regex("User ID: 12345", r"user id: \d+") == 1.0
    assert score_regex("No ID here", r"user id: \d+") == 0.0


def test_compute_score_dispatcher() -> None:
    """Test the score dispatcher function."""
    assert compute_score("Paris", "paris", "exact_match") == 1.0
    assert compute_score("The capital is Paris", "paris", "contains") == 1.0


def test_compute_aggregate_metrics() -> None:
    """Test compute_aggregate_metrics aggregates lists of results correctly."""
    results = [
        MockResult("t1", 1.0, 200.0, 100, 0.002),
        MockResult("t2", 0.5, 400.0, 200, 0.004),
        MockResult("t3", 0.0, 600.0, 300, 0.006),
    ]
    metrics = compute_aggregate_metrics(results, run_id="test-run")

    assert metrics["run_id"] == "test-run"
    assert metrics["accuracy"] == 0.5
    assert metrics["avg_latency_ms"] == 400.0
    assert metrics["p50_latency"] == 400.0
    assert metrics["p95_latency"] > 400.0
    assert metrics["total_tokens"] == 600
    assert metrics["total_cost_usd"] == 0.012
    assert metrics["score_distribution"]["perfect_1.0"] == 1
    assert metrics["score_distribution"]["medium_0.5-0.8"] == 1
    assert metrics["score_distribution"]["fail_0.0-0.2"] == 1
