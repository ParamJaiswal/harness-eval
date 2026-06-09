"""Mock Agent adapters with expert and novice profiles.

Each profile simulates a tool-using agent that reasons, selects tools, and
builds a multi-step trajectory to solve a task.  The *expert* agent picks
optimal tools and recovers from errors; the *novice* makes wrong choices,
takes extra steps, and handles errors poorly.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from evalharness.adapters.base import AgentAdapter, AgentStep, AgentTrace

# ---------------------------------------------------------------------------
# Built-in task→tool mappings
# ---------------------------------------------------------------------------

_TASK_TOOL_HINTS: dict[str, list[str]] = {
    "search": ["web_search"],
    "find": ["web_search"],
    "look up": ["web_search"],
    "calculate": ["calculator"],
    "compute": ["calculator"],
    "math": ["calculator"],
    "sum": ["calculator"],
    "code": ["code_executor"],
    "run": ["code_executor"],
    "execute": ["code_executor"],
    "program": ["code_executor"],
    "query": ["database_query"],
    "database": ["database_query"],
    "sql": ["database_query"],
    "read": ["file_reader"],
    "file": ["file_reader"],
    "open": ["file_reader"],
}

_EXPERT_THOUGHTS: list[str] = [
    "I need to break this task down into clear steps.",
    "Let me identify the right tool for this sub-problem.",
    "Based on the previous result, I should now proceed to the next step.",
    "I have enough information to formulate the final answer.",
    "Let me verify my intermediate result before continuing.",
    "The tool output confirms my hypothesis. Moving on.",
]

_NOVICE_THOUGHTS: list[str] = [
    "Hmm, I'm not sure which tool to use here. Let me try this one.",
    "That didn't work as expected. Let me try a different approach.",
    "I think I need more information. Let me search again.",
    "Wait, I should have used a different tool. Let me redo this.",
    "I'm going to try the same query with slightly different parameters.",
    "I'm confused by the output. Let me re-read the task.",
    "Maybe I should start over with a different strategy.",
]

_TOOL_OUTPUTS: dict[str, list[str]] = {
    "web_search": [
        '{"results": [{"title": "Relevant article", "snippet": "The answer to the query is 42.", "url": "https://example.com/article"}]}',
        '{"results": [{"title": "Wikipedia entry", "snippet": "According to established research, the key finding is that the process involves three main steps.", "url": "https://en.wikipedia.org/wiki/Example"}]}',
        '{"results": [{"title": "Research paper", "snippet": "A 2023 meta-analysis found statistically significant results (p<0.01) supporting the hypothesis.", "url": "https://arxiv.org/abs/2023.12345"}]}',
    ],
    "calculator": [
        '{"result": 42}',
        '{"result": 3.14159}',
        '{"result": 256}',
        '{"result": 1024}',
    ],
    "code_executor": [
        '{"stdout": "Output: success\\nResult: [1, 2, 3, 4, 5]", "stderr": "", "exit_code": 0}',
        '{"stdout": "Processing complete. Found 3 matches.", "stderr": "", "exit_code": 0}',
    ],
    "database_query": [
        '{"rows": [{"id": 1, "name": "Alice", "value": 100}, {"id": 2, "name": "Bob", "value": 200}], "row_count": 2}',
        '{"rows": [{"count": 42}], "row_count": 1}',
    ],
    "file_reader": [
        '{"content": "Line 1: Important data\\nLine 2: Key=Value\\nLine 3: Status=Active", "lines": 3}',
        '{"content": "Configuration file contents:\\nhost=localhost\\nport=8080\\ndebug=false", "lines": 3}',
    ],
}

# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

_AGENT_PROFILES: dict[str, dict[str, Any]] = {
    "agent-expert-mock": {
        "quality": "expert",
        "correct_tool_probability": 0.95,
        "extra_step_probability": 0.10,
        "error_recovery_probability": 0.90,
        "base_step_latency_ms": 400,
        "tokens_per_step": 200,
        "cost_per_1k_tokens": 0.006,
        "description": "Expert agent — efficient tool selection, good error recovery",
    },
    "agent-novice-mock": {
        "quality": "novice",
        "correct_tool_probability": 0.55,
        "extra_step_probability": 0.45,
        "error_recovery_probability": 0.35,
        "base_step_latency_ms": 500,
        "tokens_per_step": 280,
        "cost_per_1k_tokens": 0.006,
        "description": "Novice agent — frequent wrong tools, extra steps, poor recovery",
    },
}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class MockAgentAdapter(AgentAdapter):
    """Simulates an agentic system that reasons over tools to complete tasks.

    Parameters
    ----------
    profile_name:
        One of ``"agent-expert-mock"`` or ``"agent-novice-mock"``.
    """

    def __init__(self, profile_name: str) -> None:
        if profile_name not in _AGENT_PROFILES:
            raise ValueError(
                f"Unknown agent profile {profile_name!r}. "
                f"Available: {list(_AGENT_PROFILES)}"
            )
        self.profile_name = profile_name
        self._profile = _AGENT_PROFILES[profile_name]

    # -- AgentAdapter interface ----------------------------------------------

    async def execute_task(
        self, task: str, tools: list, max_steps: int = 10
    ) -> AgentTrace:
        """Execute *task* using *tools* and return an :class:`AgentTrace`."""
        profile = self._profile
        is_expert = profile["quality"] == "expert"
        steps: list[AgentStep] = []
        total_tokens = 0
        total_latency = 0.0

        # Determine which tools are needed
        needed_tools = self._infer_needed_tools(task, tools)
        tool_names: list[str] = [t if isinstance(t, str) else getattr(t, "name", str(t)) for t in tools]

        # Build the step sequence
        step_num = 0
        error_encountered = False

        for needed_tool in needed_tools:
            step_num += 1
            if step_num > max_steps:
                break

            # Decide whether to pick the correct tool
            if random.random() < profile["correct_tool_probability"]:
                chosen_tool = needed_tool if needed_tool in tool_names else random.choice(tool_names) if tool_names else None
            else:
                # Pick a wrong tool
                wrong_tools = [t for t in tool_names if t != needed_tool]
                chosen_tool = random.choice(wrong_tools) if wrong_tools else needed_tool

            # Thought
            thoughts = _EXPERT_THOUGHTS if is_expert else _NOVICE_THOUGHTS
            thought = random.choice(thoughts)

            # Tool input
            tool_input = self._generate_tool_input(chosen_tool, task)

            # Simulate tool failure (10% chance, higher for novice)
            failure_chance = 0.05 if is_expert else 0.15
            tool_failed = random.random() < failure_chance

            if tool_failed:
                tool_output = '{"error": "ToolExecutionError: Connection timed out after 30s"}'
                error_encountered = True
            else:
                tool_output = self._generate_tool_output(chosen_tool)

            # Latency
            step_latency = profile["base_step_latency_ms"] * random.uniform(0.7, 1.3)
            step_tokens = int(profile["tokens_per_step"] * random.uniform(0.8, 1.2))

            steps.append(
                AgentStep(
                    step_number=step_num,
                    thought=thought,
                    tool_name=chosen_tool,
                    tool_input=tool_input,
                    tool_output=tool_output,
                    step_latency_ms=round(step_latency, 2),
                    tokens_used=step_tokens,
                )
            )

            total_tokens += step_tokens
            total_latency += step_latency

            # If expert and tool failed, add a recovery step
            if tool_failed and is_expert and random.random() < profile["error_recovery_probability"]:
                step_num += 1
                recovery_latency = profile["base_step_latency_ms"] * random.uniform(0.5, 0.9)
                recovery_tokens = int(profile["tokens_per_step"] * 0.6)
                steps.append(
                    AgentStep(
                        step_number=step_num,
                        thought="The previous tool call failed. Let me retry with adjusted parameters.",
                        tool_name=chosen_tool,
                        tool_input=tool_input,
                        tool_output=self._generate_tool_output(chosen_tool),
                        step_latency_ms=round(recovery_latency, 2),
                        tokens_used=recovery_tokens,
                    )
                )
                total_tokens += recovery_tokens
                total_latency += recovery_latency
                error_encountered = False  # recovered

            # Novice: sometimes add redundant steps
            if not is_expert and random.random() < profile["extra_step_probability"]:
                step_num += 1
                if step_num > max_steps:
                    break
                extra_latency = profile["base_step_latency_ms"] * random.uniform(0.5, 1.0)
                extra_tokens = int(profile["tokens_per_step"] * random.uniform(0.5, 0.8))
                extra_tool = random.choice(tool_names) if tool_names else None
                steps.append(
                    AgentStep(
                        step_number=step_num,
                        thought=random.choice(_NOVICE_THOUGHTS),
                        tool_name=extra_tool,
                        tool_input=self._generate_tool_input(extra_tool, task),
                        tool_output=self._generate_tool_output(extra_tool),
                        step_latency_ms=round(extra_latency, 2),
                        tokens_used=extra_tokens,
                    )
                )
                total_tokens += extra_tokens
                total_latency += extra_latency

        # Final reasoning step
        step_num += 1
        final_latency = profile["base_step_latency_ms"] * random.uniform(0.3, 0.6)
        final_tokens = int(profile["tokens_per_step"] * 0.5)
        steps.append(
            AgentStep(
                step_number=step_num,
                thought="I now have enough information to provide the final answer.",
                tool_name=None,
                tool_input=None,
                tool_output=None,
                step_latency_ms=round(final_latency, 2),
                tokens_used=final_tokens,
            )
        )
        total_tokens += final_tokens
        total_latency += final_latency

        # Simulate async delay
        await asyncio.sleep(total_latency / 1000.0 * 0.1)  # compressed sleep

        success = not error_encountered or is_expert
        final_answer = self._generate_final_answer(task, success)

        return AgentTrace(
            steps=steps,
            final_answer=final_answer,
            total_tokens=total_tokens,
            total_latency_ms=round(total_latency, 2),
            total_cost_usd=round(total_tokens / 1000 * profile["cost_per_1k_tokens"], 8),
            success=success,
            error="Task partially failed due to tool errors" if not success else None,
        )

    def get_model_info(self) -> dict:
        """Return adapter metadata."""
        return {
            "name": self.profile_name,
            "adapter_type": "agent",
            "description": self._profile["description"],
            "config": {
                "correct_tool_probability": self._profile["correct_tool_probability"],
                "error_recovery_probability": self._profile["error_recovery_probability"],
            },
        }

    # -- Private helpers -----------------------------------------------------

    @staticmethod
    def _infer_needed_tools(task: str, tools: list) -> list[str]:
        """Heuristically decide which tools are needed for *task*."""
        task_lower = task.lower()
        tool_names = [t if isinstance(t, str) else getattr(t, "name", str(t)) for t in tools]
        needed: list[str] = []

        for keyword, tool_hints in _TASK_TOOL_HINTS.items():
            if keyword in task_lower:
                for hint in tool_hints:
                    if hint in tool_names and hint not in needed:
                        needed.append(hint)

        # If nothing matched, pick 1-2 random tools
        if not needed and tool_names:
            needed = random.sample(tool_names, k=min(2, len(tool_names)))

        return needed or (tool_names[:1] if tool_names else [])

    @staticmethod
    def _generate_tool_input(tool_name: str | None, task: str) -> dict:
        """Create realistic tool input for *tool_name*."""
        if tool_name is None:
            return {}
        inputs: dict[str, dict] = {
            "web_search": {"query": task[:60]},
            "calculator": {"expression": "2 + 2"},
            "code_executor": {"code": f"# Task: {task[:40]}\nprint('result')"},
            "database_query": {"sql": "SELECT COUNT(*) FROM records WHERE status = 'active'"},
            "file_reader": {"path": "/data/input.txt"},
        }
        return inputs.get(tool_name, {"input": task[:50]})

    @staticmethod
    def _generate_tool_output(tool_name: str | None) -> str:
        """Return a realistic tool output string."""
        if tool_name is None:
            return ""
        outputs = _TOOL_OUTPUTS.get(tool_name, ['{"result": "ok"}'])
        return random.choice(outputs)

    @staticmethod
    def _generate_final_answer(task: str, success: bool) -> str:
        """Generate a plausible final answer."""
        if success:
            return (
                f"Based on my research and analysis, I've completed the task. "
                f"After examining the relevant data and using the appropriate tools, "
                f"I found that the answer to '{task[:60]}' involves the information "
                f"gathered in the previous steps. The key findings support a clear conclusion."
            )
        return (
            f"I attempted to complete the task '{task[:60]}' but encountered some "
            f"difficulties with tool execution. Based on the partial information "
            f"gathered, here is my best answer: the available evidence suggests "
            f"a preliminary conclusion, but further investigation may be needed."
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_all_mock_agent_adapters() -> dict[str, MockAgentAdapter]:
    """Return a dict of ``name → MockAgentAdapter`` for every profile."""
    return {name: MockAgentAdapter(name) for name in _AGENT_PROFILES}
