"""Benchmark loader — reads and validates YAML benchmark definitions.

Provides BenchmarkLoader, Benchmark, and BenchmarkTask classes. Parses metadata,
validates task schemas, and returns objects consumable by the runner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkTask:
    """Represents a single test task inside a benchmark."""

    id: str
    type: str  # llm_task | agent_task | rag_task
    prompt: str
    expected_output: str
    scoring: str  # exact_match | contains | fuzzy_match | llm_judge | regex
    difficulty: str = "medium"
    tags: list[str] = field(default_factory=list)
    # Agent tasks
    available_tools: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    max_steps: int = 10
    # RAG tasks
    ground_truth_contexts: list[str] = field(default_factory=list)
    ground_truth_answer: str = ""


@dataclass
class Benchmark:
    """Represents a full benchmark suite containing metadata and tasks."""

    name: str
    description: str
    category: str  # llm | agent | rag
    version: str
    tasks: list[BenchmarkTask]


class BenchmarkLoader:
    """Finds, loads, and manages evaluation benchmark files."""

    def __init__(self, benchmarks_dir: str = "benchmarks") -> None:
        self.benchmarks_dir = Path(benchmarks_dir)
        self._benchmarks: dict[str, Benchmark] = {}

    def load_all(self) -> dict[str, Benchmark]:
        """Load all YAML benchmark files from the benchmarks directory."""
        if not self.benchmarks_dir.exists():
            logger.warning(f"Benchmarks directory {self.benchmarks_dir} does not exist.")
            return {}

        for filepath in sorted(self.benchmarks_dir.glob("*.yaml")):
            try:
                benchmark = self.load_benchmark(filepath)
                # Index by display name (e.g. "General Knowledge")
                self._benchmarks[benchmark.name] = benchmark
                # Also index by filename stem (e.g. "general_knowledge") for programmatic access
                self._benchmarks[filepath.stem] = benchmark
            except Exception as e:
                logger.error(f"Failed to load benchmark from {filepath}: {e}")

        for filepath in sorted(self.benchmarks_dir.glob("*.yml")):
            try:
                benchmark = self.load_benchmark(filepath)
                self._benchmarks[benchmark.name] = benchmark
                self._benchmarks[filepath.stem] = benchmark
            except Exception as e:
                logger.error(f"Failed to load benchmark from {filepath}: {e}")

        return self._benchmarks

    def load_benchmark(self, filepath: Path) -> Benchmark:
        """Load and parse a single benchmark YAML file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty benchmark file: {filepath}")

        # Basic validations
        name = data.get("name")
        description = data.get("description", "")
        category = data.get("category", "llm")
        version = str(data.get("version", "1.0"))
        tasks_data = data.get("tasks", [])

        if not name:
            raise ValueError(f"Missing 'name' in benchmark {filepath}")

        tasks: list[BenchmarkTask] = []
        for task_dict in tasks_data:
            task = BenchmarkTask(
                id=str(task_dict.get("id")),
                type=str(task_dict.get("type", "llm_task")),
                prompt=str(task_dict.get("prompt", "")),
                expected_output=str(task_dict.get("expected_output", "")),
                scoring=str(task_dict.get("scoring", "exact_match")),
                difficulty=str(task_dict.get("difficulty", "medium")),
                tags=list(task_dict.get("tags", [])),
                available_tools=list(task_dict.get("available_tools", [])),
                expected_tools=list(task_dict.get("expected_tools", [])),
                max_steps=int(task_dict.get("max_steps", 10)),
                ground_truth_contexts=list(task_dict.get("ground_truth_contexts", [])),
                ground_truth_answer=str(task_dict.get("ground_truth_answer", "")),
            )
            tasks.append(task)

        benchmark = Benchmark(
            name=name,
            description=description,
            category=category,
            version=version,
            tasks=tasks,
        )
        return benchmark

    def get_benchmark(self, name: str) -> Benchmark:
        """Get a loaded benchmark by name."""
        if name not in self._benchmarks:
            # Try to load again just in case
            self.load_all()
        if name not in self._benchmarks:
            raise KeyError(f"Benchmark {name!r} not found.")
        return self._benchmarks[name]

    def list_benchmarks(self) -> list[dict[str, Any]]:
        """List all loaded benchmarks with metadata (deduplicated)."""
        if not self._benchmarks:
            self.load_all()
        # Deduplicate by object identity — same Benchmark may be stored under multiple keys
        seen: set[int] = set()
        result = []
        for b in self._benchmarks.values():
            if id(b) not in seen:
                seen.add(id(b))
                result.append(
                    {
                        "name": b.name,
                        "description": b.description,
                        "category": b.category,
                        "version": b.version,
                        "task_count": len(b.tasks),
                    }
                )
        return result
