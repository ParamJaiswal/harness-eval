"""Agent trajectory scorer — evaluates tool selection, efficiency, and recovery.

Produces an :class:`AgentScorecard` with five dimensions plus a weighted
overall score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalharness.adapters.base import AgentTrace


@dataclass
class AgentScorecard:
    """Multi-dimensional score for an agent execution trace."""

    tool_selection_accuracy: float
    parameter_accuracy: float
    step_efficiency: float
    error_recovery: float
    goal_completion: float
    overall_score: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_trajectory(
    trace: "AgentTrace",
    expected_tools: list[str],
    max_optimal_steps: int,
) -> AgentScorecard:
    """Score an agent's full execution trajectory.

    Parameters
    ----------
    trace:
        The agent's recorded execution trace.
    expected_tools:
        The list of tool names that should have been used.
    max_optimal_steps:
        The ideal (minimum) number of steps to complete the task.

    Returns
    -------
    AgentScorecard
    """
    used_tools = [
        s.tool_name for s in trace.steps if s.tool_name is not None
    ]

    tool_acc = compute_tool_selection_accuracy(used_tools, expected_tools)
    param_acc = _compute_parameter_accuracy(trace)
    efficiency = compute_step_efficiency(len(trace.steps), max_optimal_steps)
    recovery = compute_error_recovery(trace)
    goal = 1.0 if trace.success else 0.3  # partial credit if unsuccessful

    # Weighted average (weights sum to 1.0)
    overall = (
        0.25 * tool_acc
        + 0.15 * param_acc
        + 0.20 * efficiency
        + 0.15 * recovery
        + 0.25 * goal
    )

    return AgentScorecard(
        tool_selection_accuracy=round(tool_acc, 4),
        parameter_accuracy=round(param_acc, 4),
        step_efficiency=round(efficiency, 4),
        error_recovery=round(recovery, 4),
        goal_completion=round(goal, 4),
        overall_score=round(overall, 4),
    )


def compute_tool_selection_accuracy(
    used_tools: list[str], expected_tools: list[str]
) -> float:
    """Compute precision/recall F1 of tool selection.

    Returns
    -------
    float
        The F1 score of the tool usage relative to the expected set.
    """
    if not expected_tools:
        # No expected tools — any choice is acceptable
        return 1.0

    used_set = set(used_tools)
    expected_set = set(expected_tools)

    if not used_set:
        return 0.0

    true_positives = len(used_set & expected_set)
    precision = true_positives / len(used_set) if used_set else 0.0
    recall = true_positives / len(expected_set) if expected_set else 0.0

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_step_efficiency(actual_steps: int, optimal_steps: int) -> float:
    """Ratio of optimal to actual steps, capped at 1.0.

    A perfect agent completes the task in exactly *optimal_steps*; extra
    steps reduce the score.
    """
    if actual_steps <= 0 or optimal_steps <= 0:
        return 0.0
    return min(1.0, optimal_steps / actual_steps)


def compute_error_recovery(trace: "AgentTrace") -> float:
    """Score how well the agent recovered from tool failures.

    Heuristic:
    - If no errors were encountered, return 1.0 (perfect — nothing to recover
      from).
    - For each error, check whether the next step retries or uses a fallback.
    - The score is the fraction of errors that were followed by a recovery
      attempt.
    """
    error_indices: list[int] = []
    for i, step in enumerate(trace.steps):
        if step.tool_output and "error" in step.tool_output.lower():
            error_indices.append(i)

    if not error_indices:
        return 1.0  # no errors — perfect by default

    recovered = 0
    for idx in error_indices:
        if idx + 1 < len(trace.steps):
            next_step = trace.steps[idx + 1]
            # Recovery signals: same tool retried, or explicit recovery thought
            if next_step.tool_name is not None:
                recovery_phrases = [
                    "retry",
                    "try again",
                    "adjust",
                    "recover",
                    "fallback",
                    "alternative",
                    "different",
                    "failed",
                ]
                thought_lower = next_step.thought.lower()
                if any(p in thought_lower for p in recovery_phrases):
                    recovered += 1
                elif next_step.tool_name == trace.steps[idx].tool_name:
                    recovered += 1  # same tool = retry

    return recovered / len(error_indices)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_parameter_accuracy(trace: "AgentTrace") -> float:
    """Heuristic for parameter accuracy.

    Since we don't have ground-truth parameters, we use a proxy: did the
    tool return a successful (non-error) output?  Each successful tool call
    is counted as a correct-parameter call.
    """
    tool_steps = [s for s in trace.steps if s.tool_name is not None]
    if not tool_steps:
        return 1.0

    success_count = sum(
        1 for s in tool_steps
        if s.tool_output and "error" not in s.tool_output.lower()
    )
    return success_count / len(tool_steps)
