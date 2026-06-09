"""Adapter registry — auto-discovers and registers all mock adapters.

The ``ADAPTER_REGISTRY`` dict maps adapter names to adapter instances and is
the single source of truth used by the evaluation engine and API layer.
"""

from __future__ import annotations

import logging
from typing import Any

from evalharness.adapters.base import (
    AgentAdapter,
    AgentStep,
    AgentTrace,
    LLMAdapter,
    ModelResponse,
    RAGAdapter,
    RAGResponse,
    RetrievedContext,
)
from evalharness.adapters.mock_adapter import MockLLMAdapter, get_all_mock_llm_adapters
from evalharness.adapters.mock_agent_adapter import (
    MockAgentAdapter,
    get_all_mock_agent_adapters,
)
from evalharness.adapters.mock_rag_adapter import (
    MockRAGAdapter,
    get_all_mock_rag_adapters,
)

logger = logging.getLogger(__name__)

# Build the global registry eagerly so it is available at import time.
ADAPTER_REGISTRY: dict[str, LLMAdapter | AgentAdapter | RAGAdapter] = {}


def _build_registry() -> None:
    """Populate ``ADAPTER_REGISTRY`` with all available adapters."""
    global ADAPTER_REGISTRY

    # Mock LLM adapters
    for name, adapter in get_all_mock_llm_adapters().items():
        ADAPTER_REGISTRY[name] = adapter

    # Mock Agent adapters
    for name, adapter in get_all_mock_agent_adapters().items():
        ADAPTER_REGISTRY[name] = adapter

    # Mock RAG adapters
    for name, adapter in get_all_mock_rag_adapters().items():
        ADAPTER_REGISTRY[name] = adapter

    # Try to register the real OpenAI adapter (only if API key is available)
    try:
        from evalharness.adapters.openai_adapter import OpenAILLMAdapter

        for model in ("gpt-4o", "gpt-4o-mini"):
            try:
                adapter = OpenAILLMAdapter(model=model)
                ADAPTER_REGISTRY[model] = adapter
            except ValueError:
                # No API key configured — skip silently
                pass
    except Exception:
        logger.debug("OpenAI adapter not available — skipping.")


_build_registry()

__all__ = [
    "ADAPTER_REGISTRY",
    "LLMAdapter",
    "AgentAdapter",
    "RAGAdapter",
    "ModelResponse",
    "AgentStep",
    "AgentTrace",
    "RAGResponse",
    "RetrievedContext",
    "MockLLMAdapter",
    "MockAgentAdapter",
    "MockRAGAdapter",
    "get_all_mock_llm_adapters",
    "get_all_mock_agent_adapters",
    "get_all_mock_rag_adapters",
]
