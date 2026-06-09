"""HTTP adapter for evaluating customer-hosted agent systems.

Customers expose an agent webhook. The harness sends the task + available tools,
receives a final answer + execution trace, and scores tool selection + step efficiency.

Expected webhook contract
-------------------------

Request (POST <endpoint_url>)::

    {
        "task": "Find the weather in London and calculate the temperature in Celsius",
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {"city": {"type": "string"}}
            }
        ],
        "max_steps": 10
    }

Response (one of the following formats is accepted)::

    # Format A — with full trace
    {
        "answer": "The temperature in London is 18°C",
        "steps": [
            {
                "step": 1,
                "thought": "I need to get the weather first",
                "tool": "get_weather",
                "input": {"city": "London"},
                "output": "15°C / 59°F, Partly Cloudy",
                "latency_ms": 120
            }
        ],
        "tokens_used": 450
    }

    # Format B — minimal (answer only)
    {
        "answer": "The temperature in London is 18°C"
    }
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from evalharness.adapters.base import AgentAdapter, AgentStep, AgentTrace


class AgentWebhookAdapter(AgentAdapter):
    """Calls a customer-hosted agent system via HTTP webhook."""

    def __init__(
        self,
        endpoint_url: str,
        api_key: str = "",
        timeout: int = 120,
        max_retries: int = 2,
        extra_headers: dict | None = None,
        display_name: str | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_headers = extra_headers or {}
        self.display_name = display_name or "agent-webhook"

    async def execute_task(self, task: str, tools: list, max_steps: int = 10) -> AgentTrace:
        # Serialise tool definitions for the customer endpoint
        tool_defs = []
        for t in tools:
            tool_defs.append({
                "name": t.name if hasattr(t, "name") else str(t),
                "description": t.description if hasattr(t, "description") else "",
                "parameters": t.parameters if hasattr(t, "parameters") else {},
            })

        payload = {
            "task": task,
            "tools": tool_defs,
            "max_steps": max_steps,
        }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(self.endpoint_url, json=payload, headers=headers)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()

                    # Parse answer
                    answer = (
                        data.get("answer")
                        or data.get("response")
                        or data.get("output")
                        or data.get("result")
                        or ""
                    )

                    # Parse steps / trace
                    raw_steps = (
                        data.get("steps")
                        or data.get("trace")
                        or data.get("trajectory")
                        or []
                    )
                    parsed_steps: list[AgentStep] = []
                    for i, s in enumerate(raw_steps):
                        if isinstance(s, dict):
                            parsed_steps.append(AgentStep(
                                step_number=int(s.get("step") or s.get("step_number") or i + 1),
                                thought=str(s.get("thought") or s.get("reasoning") or ""),
                                tool_name=s.get("tool") or s.get("tool_name") or s.get("action"),
                                tool_input=s.get("input") or s.get("tool_input") or s.get("parameters"),
                                tool_output=str(s.get("output") or s.get("tool_output") or s.get("observation") or ""),
                                step_latency_ms=float(s.get("latency_ms") or 0.0),
                                tokens_used=int(s.get("tokens_used") or s.get("tokens") or 0),
                            ))

                    tokens = int(data.get("tokens_used") or data.get("tokens") or 0)
                    success = data.get("success", True)
                    error = data.get("error")

                    return AgentTrace(
                        steps=parsed_steps,
                        final_answer=str(answer),
                        total_tokens=tokens,
                        total_latency_ms=round(elapsed_ms, 2),
                        total_cost_usd=0.0,
                        success=bool(success),
                        error=error,
                    )

                if resp.status_code >= 500:
                    await asyncio.sleep(2 ** attempt)
                    last_exc = Exception(f"Agent endpoint server error: {resp.status_code}")
                    continue

                raise RuntimeError(f"Agent endpoint returned {resp.status_code}: {resp.text[:400]}")

            except httpx.TimeoutException as exc:
                last_exc = exc
                await asyncio.sleep(2 ** attempt)
            except RuntimeError:
                raise
            except Exception as exc:
                last_exc = exc
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"Agent webhook failed after {self.max_retries} retries: {last_exc}")

    def get_model_info(self) -> dict:
        return {
            "name": self.display_name,
            "adapter_type": "agent",
            "description": f"Agent webhook endpoint: {self.endpoint_url}",
            "config": {"endpoint": self.endpoint_url},
        }
