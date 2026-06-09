"""Pre-built mock tools for agent benchmarks.

Each tool has realistic, deterministic responses mapped to common inputs, plus
a ``should_fail`` escape-hatch for simulating tool failures in benchmarks.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from typing import Any

from evalharness.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

# ---- 1. web_search --------------------------------------------------------

_SEARCH_RESULTS: dict[str, list[dict[str, str]]] = {
    "capital of france": [
        {"title": "Paris — Wikipedia", "snippet": "Paris is the capital and most populous city of France.", "url": "https://en.wikipedia.org/wiki/Paris"},
        {"title": "France — Capital City", "snippet": "The capital of France is Paris, known as the City of Light.", "url": "https://www.britannica.com/place/France"},
    ],
    "machine learning": [
        {"title": "Machine Learning — Stanford", "snippet": "Machine learning is a subset of AI focused on algorithms that learn from data.", "url": "https://cs229.stanford.edu"},
        {"title": "ML Tutorial", "snippet": "Supervised, unsupervised, and reinforcement learning are the three main paradigms.", "url": "https://scikit-learn.org"},
    ],
    "climate change effects": [
        {"title": "NASA Climate", "snippet": "Global temperatures have risen 1.1°C since pre-industrial times. Sea levels are rising.", "url": "https://climate.nasa.gov"},
        {"title": "IPCC AR6", "snippet": "Human-caused climate change is already affecting weather extremes globally.", "url": "https://www.ipcc.ch"},
    ],
    "python asyncio": [
        {"title": "asyncio — Python docs", "snippet": "asyncio is a library to write concurrent code using the async/await syntax.", "url": "https://docs.python.org/3/library/asyncio.html"},
    ],
    "fibonacci sequence": [
        {"title": "Fibonacci Numbers", "snippet": "The Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34…", "url": "https://mathworld.wolfram.com/FibonacciNumber.html"},
    ],
}


def web_search(query: str, **kwargs: Any) -> str:
    """Simulate a web search returning JSON results."""
    query_lower = query.lower().strip()
    for key, results in _SEARCH_RESULTS.items():
        if key in query_lower:
            return json.dumps({"query": query, "results": results, "total": len(results)})
    # Generic fallback
    return json.dumps({
        "query": query,
        "results": [
            {"title": f"Result for: {query}", "snippet": f"Information about {query} from a reliable source.", "url": "https://example.com/result"},
        ],
        "total": 1,
    })


# ---- 2. calculator --------------------------------------------------------

# Safe operators for expression evaluation
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_expr(node: ast.AST) -> float:
    """Recursively evaluate an AST expression using only arithmetic ops."""
    if isinstance(node, ast.Expression):
        return _safe_eval_expr(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval_expr(node.left)
        right = _safe_eval_expr(node.right)
        return op_fn(left, right)
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval_expr(node.operand))
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def calculator(expression: str, **kwargs: Any) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval_expr(tree)
        return json.dumps({"expression": expression, "result": result})
    except Exception as exc:
        return json.dumps({"expression": expression, "error": str(exc)})


# ---- 3. code_executor -----------------------------------------------------

_CODE_OUTPUTS: dict[str, dict[str, str]] = {
    "hello": {"stdout": "Hello, World!", "stderr": "", "exit_code": "0"},
    "fibonacci": {"stdout": "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]", "stderr": "", "exit_code": "0"},
    "sort": {"stdout": "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]", "stderr": "", "exit_code": "0"},
    "error": {"stdout": "", "stderr": "Traceback (most recent call last):\n  File \"<stdin>\", line 1\nSyntaxError: invalid syntax", "exit_code": "1"},
    "import": {"stdout": "Module imported successfully.", "stderr": "", "exit_code": "0"},
}


def code_executor(code: str, **kwargs: Any) -> str:
    """Simulate code execution with pre-defined outputs."""
    code_lower = code.lower()
    for keyword, output in _CODE_OUTPUTS.items():
        if keyword in code_lower:
            return json.dumps({"code": code[:80], **output})
    return json.dumps({
        "code": code[:80],
        "stdout": f"Executed successfully. Output: result_value",
        "stderr": "",
        "exit_code": "0",
    })


# ---- 4. database_query ----------------------------------------------------

_SQL_RESULTS: dict[str, dict[str, Any]] = {
    "select": {
        "rows": [
            {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "status": "active"},
            {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "status": "active"},
            {"id": 3, "name": "Carol Williams", "email": "carol@example.com", "status": "inactive"},
        ],
        "row_count": 3,
        "execution_time_ms": 12.5,
    },
    "count": {
        "rows": [{"count": 42}],
        "row_count": 1,
        "execution_time_ms": 3.2,
    },
    "insert": {
        "rows": [],
        "row_count": 0,
        "affected_rows": 1,
        "execution_time_ms": 8.1,
    },
    "update": {
        "rows": [],
        "row_count": 0,
        "affected_rows": 3,
        "execution_time_ms": 15.4,
    },
    "delete": {
        "rows": [],
        "row_count": 0,
        "affected_rows": 1,
        "execution_time_ms": 5.7,
    },
}


def database_query(sql: str, **kwargs: Any) -> str:
    """Simulate a database query with pre-defined results."""
    sql_lower = sql.lower().strip()
    for keyword, result in _SQL_RESULTS.items():
        if sql_lower.startswith(keyword):
            return json.dumps({"sql": sql[:100], **result})
    return json.dumps({
        "sql": sql[:100],
        "rows": [{"result": "query executed"}],
        "row_count": 1,
        "execution_time_ms": 10.0,
    })


# ---- 5. file_reader -------------------------------------------------------

_FILE_CONTENTS: dict[str, str] = {
    "config": "# Application Configuration\nDATABASE_HOST=localhost\nDATABASE_PORT=5432\nDATABASE_NAME=evaldb\nDEBUG=false\nLOG_LEVEL=INFO",
    "data": "id,name,value,timestamp\n1,sensor_a,23.5,2024-01-15T10:00:00\n2,sensor_b,18.2,2024-01-15T10:01:00\n3,sensor_c,31.7,2024-01-15T10:02:00",
    "readme": "# Project README\n\nThis project implements an AI evaluation harness.\n\n## Setup\n1. Install dependencies: pip install -r requirements.txt\n2. Run the server: python -m evalharness.main\n",
    "log": "[2024-01-15 10:00:01] INFO  Server started on port 8000\n[2024-01-15 10:00:05] INFO  Database connected\n[2024-01-15 10:01:12] WARN  High latency detected: 2500ms\n[2024-01-15 10:02:30] ERROR Connection pool exhausted",
    "json": '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "total": 2}',
}


def file_reader(path: str, **kwargs: Any) -> str:
    """Simulate reading a file with pre-defined contents."""
    path_lower = path.lower()
    for keyword, content in _FILE_CONTENTS.items():
        if keyword in path_lower:
            return json.dumps({"path": path, "content": content, "size_bytes": len(content)})
    return json.dumps({
        "path": path,
        "content": f"Contents of {path}:\nLine 1: Sample data\nLine 2: More data\nLine 3: End of file",
        "size_bytes": 64,
    })


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------


def create_default_tool_registry() -> ToolRegistry:
    """Return a :class:`ToolRegistry` pre-loaded with all mock tools."""
    registry = ToolRegistry()

    registry.register_tool(
        name="web_search",
        description="Search the web for information. Returns JSON with title, snippet, and URL for each result.",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        },
        handler_fn=web_search,
    )

    registry.register_tool(
        name="calculator",
        description="Safely evaluate a mathematical expression. Supports +, -, *, /, **, %, //.",
        parameters_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression to evaluate"},
            },
            "required": ["expression"],
        },
        handler_fn=calculator,
    )

    registry.register_tool(
        name="code_executor",
        description="Execute a code snippet and return stdout, stderr, and exit code.",
        parameters_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The code to execute"},
            },
            "required": ["code"],
        },
        handler_fn=code_executor,
    )

    registry.register_tool(
        name="database_query",
        description="Execute a SQL query against a database and return the results.",
        parameters_schema={
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "The SQL query to execute"},
            },
            "required": ["sql"],
        },
        handler_fn=database_query,
    )

    registry.register_tool(
        name="file_reader",
        description="Read the contents of a file given its path.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
            },
            "required": ["path"],
        },
        handler_fn=file_reader,
    )

    return registry
