"""OpenAI LLM adapter using *httpx* for async HTTP calls.

Supports ``gpt-4o`` and ``gpt-4o-mini`` with retry, exponential backoff,
and rate-limit awareness.  If no API key is configured the adapter raises
a descriptive error at instantiation time so the rest of the system can
fall back to mock adapters gracefully.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from evalharness.adapters.base import LLMAdapter, ModelResponse
from evalharness.config import get_settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

# Pricing per 1 000 tokens (USD) — May 2024 rates
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

_DEFAULT_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class OpenAILLMAdapter(LLMAdapter):
    """Async OpenAI chat-completions adapter with retry and rate-limit logic.

    Parameters
    ----------
    model:
        OpenAI model identifier (default ``gpt-4o``).
    api_key:
        Explicit API key.  Falls back to ``get_settings().OPENAI_API_KEY``.
    max_retries:
        Number of retry attempts on transient errors.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int | None = None,
        timeout: int | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required.  Set the EVAL_OPENAI_API_KEY "
                "environment variable or pass api_key= explicitly."
            )
        self.max_retries = max_retries if max_retries is not None else settings.DEFAULT_RETRIES
        self.timeout = timeout if timeout is not None else settings.DEFAULT_TIMEOUT
        self._semaphore = asyncio.Semaphore(10)  # rate-limit concurrency

    # -- LLMAdapter interface ------------------------------------------------

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Send a chat completion request with retry / backoff."""
        messages = kwargs.pop("messages", None) or [{"role": "user", "content": prompt}]
        temperature = kwargs.pop("temperature", 0.7)
        max_tokens = kwargs.pop("max_tokens", 1024)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            async with self._semaphore:
                t0 = time.perf_counter()
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.post(
                            _OPENAI_CHAT_URL,
                            json=payload,
                            headers=headers,
                        )

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    if resp.status_code == 200:
                        data = resp.json()
                        choice = data["choices"][0]
                        usage = data.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

                        pricing = _PRICING.get(self.model, _PRICING[_DEFAULT_MODEL])
                        cost = (
                            prompt_tokens / 1000 * pricing["input"]
                            + completion_tokens / 1000 * pricing["output"]
                        )

                        return ModelResponse(
                            text=choice["message"]["content"],
                            tokens_used=total_tokens,
                            latency_ms=round(elapsed_ms, 2),
                            cost_usd=round(cost, 8),
                            model=self.model,
                            metadata={
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "finish_reason": choice.get("finish_reason"),
                            },
                        )

                    # Rate-limited — honour Retry-After header
                    if resp.status_code == 429:
                        retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                        await asyncio.sleep(retry_after)
                        last_exc = httpx.HTTPStatusError(
                            f"Rate limited (429)", request=resp.request, response=resp
                        )
                        continue

                    # Server error — retry with backoff
                    if resp.status_code >= 500:
                        await asyncio.sleep(2 ** attempt)
                        last_exc = httpx.HTTPStatusError(
                            f"Server error ({resp.status_code})",
                            request=resp.request,
                            response=resp,
                        )
                        continue

                    # Client error — don't retry
                    resp.raise_for_status()

                except httpx.TimeoutException as exc:
                    last_exc = exc
                    await asyncio.sleep(2 ** attempt)
                except httpx.HTTPStatusError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    await asyncio.sleep(2 ** attempt)

        raise RuntimeError(
            f"OpenAI request failed after {self.max_retries} retries: {last_exc}"
        )

    def get_model_info(self) -> dict:
        """Return adapter metadata."""
        return {
            "name": self.model,
            "adapter_type": "llm",
            "description": f"OpenAI {self.model} via chat completions API",
            "config": {
                "max_retries": self.max_retries,
                "timeout": self.timeout,
            },
        }
