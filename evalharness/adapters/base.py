"""Abstract base classes for model adapters.

Every concrete adapter (mock, OpenAI, etc.) inherits from one of these ABCs.
The dataclasses here form the common *lingua franca* that the evaluation engine
consumes regardless of which backend is driving the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Response data-classes
# ---------------------------------------------------------------------------


@dataclass
class ModelResponse:
    """Return value from an LLM ``generate()`` call."""

    text: str
    tokens_used: int
    latency_ms: float
    cost_usd: float
    model: str
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentStep:
    """One step in an agentic execution trace."""

    step_number: int
    thought: str
    tool_name: str | None
    tool_input: dict | None
    tool_output: str | None
    step_latency_ms: float
    tokens_used: int


@dataclass
class AgentTrace:
    """Full agent execution trace (multiple steps → final answer)."""

    steps: list[AgentStep]
    final_answer: str
    total_tokens: int
    total_latency_ms: float
    total_cost_usd: float
    success: bool
    error: str | None = None


@dataclass
class RetrievedContext:
    """A single chunk returned by the retrieval stage of a RAG pipeline."""

    text: str
    source: str
    relevance_score: float
    chunk_id: str


@dataclass
class RAGResponse:
    """Return value from a RAG ``query()`` call."""

    answer: str
    retrieved_contexts: list[RetrievedContext]
    tokens_used: int
    latency_ms: float
    cost_usd: float
    retrieval_latency_ms: float
    generation_latency_ms: float


# ---------------------------------------------------------------------------
# Abstract adapter contracts
# ---------------------------------------------------------------------------


class LLMAdapter(ABC):
    """Contract for language-model backends."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a completion for *prompt*."""
        ...

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return metadata about this adapter / model."""
        ...


class AgentAdapter(ABC):
    """Contract for agentic systems that use tools."""

    @abstractmethod
    async def execute_task(
        self, task: str, tools: list, max_steps: int = 10
    ) -> AgentTrace:
        """Execute *task* using the given *tools*, returning a full trace."""
        ...

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return metadata about this agent adapter."""
        ...


class RAGAdapter(ABC):
    """Contract for retrieval-augmented generation pipelines."""

    @abstractmethod
    async def query(self, question: str, **kwargs) -> RAGResponse:
        """Run the full retrieve→generate pipeline for *question*."""
        ...

    @abstractmethod
    def get_pipeline_info(self) -> dict:
        """Return metadata about this RAG pipeline."""
        ...
