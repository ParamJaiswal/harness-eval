"""Agent tool registry — define, register, and execute tools.

The registry is the bridge between benchmark task descriptions (which reference
tools by name) and the actual callable handlers that produce tool outputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class ToolDefinition:
    """Metadata and handler for a single agent tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON-Schema style
    handler: Callable[..., Any]


@dataclass
class ToolResult:
    """Outcome of a single tool execution."""

    success: bool
    output: str
    error: str | None = None
    execution_time_ms: float = 0.0


class ToolRegistry:
    """Central registry for agent tools.

    Tools are registered once at startup and can then be listed, looked up,
    or executed by name.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    # -- public API ----------------------------------------------------------

    def register_tool(
        self,
        name: str,
        description: str,
        parameters_schema: dict[str, Any],
        handler_fn: Callable[..., Any],
    ) -> None:
        """Register a new tool."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters_schema,
            handler=handler_fn,
        )

    def get_tool(self, name: str) -> ToolDefinition:
        """Return the :class:`ToolDefinition` for *name*.

        Raises
        ------
        KeyError
            If the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool {name!r} is not registered.")
        return self._tools[name]

    def list_tools(self) -> list[ToolDefinition]:
        """Return all registered tools."""
        return list(self._tools.values())

    def execute_tool(self, name: str, **params: Any) -> ToolResult:
        """Look up *name* and invoke its handler with *params*.

        Tool failures are caught and returned as a :class:`ToolResult` with
        ``success=False`` rather than propagating exceptions.
        """
        try:
            tool = self.get_tool(name)
        except KeyError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        should_fail = params.pop("should_fail", False)
        if should_fail:
            return ToolResult(
                success=False,
                output="",
                error=f"Simulated failure for tool '{name}'",
                execution_time_ms=0.0,
            )

        t0 = time.perf_counter()
        try:
            output = tool.handler(**params)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ToolResult(
                success=True,
                output=str(output),
                execution_time_ms=round(elapsed, 2),
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ToolResult(
                success=False,
                output="",
                error=f"{type(exc).__name__}: {exc}",
                execution_time_ms=round(elapsed, 2),
            )
