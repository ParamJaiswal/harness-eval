"""Generic HTTP adapters for evaluating customer-hosted LLM endpoints.

Supports:
- ``openai_compatible`` — any endpoint speaking OpenAI chat-completions
  (OpenAI, Azure, Groq, Together AI, Anyscale, vLLM, LM Studio, Ollama, etc.)
- ``anthropic``        — Anthropic Claude messages API
- ``custom_llm``       — simple {"prompt": "..."} → {"response": "..."} contract

Usage
-----
Pass provider details directly in the evaluation run request::

    POST /api/eval/run
    {
        "model_name": "my-customer-model",
        "benchmark_name": "general_knowledge",
        "eval_type": "llm",
        "endpoint_url": "https://api.example.com/v1/chat/completions",
        "api_key": "sk-...",
        "provider_type": "openai_compatible",
        "model_id": "llama-3-70b"
    }
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from evalharness.adapters.base import LLMAdapter, ModelResponse


# ---------------------------------------------------------------------------
# OpenAI-compatible adapter (covers OpenAI, Groq, Ollama, vLLM, etc.)
# ---------------------------------------------------------------------------


class OpenAICompatibleAdapter(LLMAdapter):
    """Calls any OpenAI chat-completions compatible endpoint.

    Compatible with: OpenAI, Azure OpenAI, Groq, Together AI, Anyscale,
    Perplexity, vLLM, LM Studio, Ollama (use http://localhost:11434/v1/...).
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: str = "",
        model_id: str = "gpt-4o",
        timeout: int = 60,
        max_retries: int = 3,
        extra_headers: dict | None = None,
        display_name: str | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        # Normalise: if user gives base URL (e.g. https://api.openai.com/v1) add path
        if not self.endpoint_url.endswith("/chat/completions"):
            self.endpoint_url = self.endpoint_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model_id = model_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_headers = extra_headers or {}
        self.display_name = display_name or model_id
        self._semaphore = asyncio.Semaphore(10)

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        messages = kwargs.pop("messages", None) or [{"role": "user", "content": prompt}]
        temperature = kwargs.pop("temperature", 0.0)
        max_tokens = kwargs.pop("max_tokens", 1024)

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(kwargs)

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            async with self._semaphore:
                t0 = time.perf_counter()
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.post(self.endpoint_url, json=payload, headers=headers)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    if resp.status_code == 200:
                        data = resp.json()
                        choice = data["choices"][0]
                        usage = data.get("usage", {})
                        total_tokens = usage.get("total_tokens", 0)
                        return ModelResponse(
                            text=choice["message"]["content"],
                            tokens_used=total_tokens,
                            latency_ms=round(elapsed_ms, 2),
                            cost_usd=0.0,  # Unknown cost for customer endpoints
                            model=self.model_id,
                            metadata={
                                "provider": "openai_compatible",
                                "endpoint": self.endpoint_url,
                                "finish_reason": choice.get("finish_reason"),
                            },
                        )

                    if resp.status_code == 429:
                        retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                        await asyncio.sleep(retry_after)
                        last_exc = Exception(f"Rate limited by endpoint (429)")
                        continue

                    if resp.status_code >= 500:
                        await asyncio.sleep(2 ** attempt)
                        last_exc = Exception(f"Server error from endpoint: {resp.status_code} — {resp.text[:200]}")
                        continue

                    # Non-retryable client error
                    raise RuntimeError(
                        f"Endpoint returned {resp.status_code}: {resp.text[:400]}"
                    )

                except httpx.TimeoutException as exc:
                    last_exc = exc
                    await asyncio.sleep(2 ** attempt)
                except RuntimeError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    await asyncio.sleep(2 ** attempt)

        raise RuntimeError(
            f"OpenAI-compatible endpoint failed after {self.max_retries} retries: {last_exc}"
        )

    def get_model_info(self) -> dict:
        return {
            "name": self.display_name,
            "adapter_type": "llm",
            "description": f"OpenAI-compatible endpoint: {self.endpoint_url}",
            "config": {"model_id": self.model_id, "endpoint": self.endpoint_url},
        }


# ---------------------------------------------------------------------------
# Anthropic adapter (Claude 3, Claude 3.5, etc.)
# ---------------------------------------------------------------------------


class AnthropicAdapter(LLMAdapter):
    """Calls the Anthropic messages API for Claude models."""

    _ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
    _ANTHROPIC_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        model_id: str = "claude-3-5-sonnet-20241022",
        timeout: int = 60,
        max_retries: int = 3,
        display_name: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Anthropic API key is required.")
        self.api_key = api_key
        self.model_id = model_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.display_name = display_name or model_id
        self._semaphore = asyncio.Semaphore(5)

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        max_tokens = kwargs.pop("max_tokens", 1024)
        temperature = kwargs.pop("temperature", 0.0)

        payload: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self._ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            async with self._semaphore:
                t0 = time.perf_counter()
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.post(self._ANTHROPIC_URL, json=payload, headers=headers)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    if resp.status_code == 200:
                        data = resp.json()
                        text = data["content"][0]["text"]
                        usage = data.get("usage", {})
                        total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                        return ModelResponse(
                            text=text,
                            tokens_used=total_tokens,
                            latency_ms=round(elapsed_ms, 2),
                            cost_usd=0.0,
                            model=self.model_id,
                            metadata={
                                "provider": "anthropic",
                                "stop_reason": data.get("stop_reason"),
                            },
                        )

                    if resp.status_code in (429, 529):
                        await asyncio.sleep(2 ** attempt)
                        last_exc = Exception(f"Anthropic rate limited ({resp.status_code})")
                        continue

                    if resp.status_code >= 500:
                        await asyncio.sleep(2 ** attempt)
                        last_exc = Exception(f"Anthropic server error: {resp.status_code}")
                        continue

                    raise RuntimeError(f"Anthropic API returned {resp.status_code}: {resp.text[:400]}")

                except httpx.TimeoutException as exc:
                    last_exc = exc
                    await asyncio.sleep(2 ** attempt)
                except RuntimeError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"Anthropic request failed after {self.max_retries} retries: {last_exc}")

    def get_model_info(self) -> dict:
        return {
            "name": self.display_name,
            "adapter_type": "llm",
            "description": f"Anthropic {self.model_id}",
            "config": {"model_id": self.model_id},
        }


# ---------------------------------------------------------------------------
# Custom HTTP LLM adapter (simple REST contract)
# ---------------------------------------------------------------------------


class CustomHTTPLLMAdapter(LLMAdapter):
    """Calls a customer's custom LLM endpoint.

    Expected request::

        POST <endpoint_url>
        Authorization: Bearer <api_key>
        Content-Type: application/json
        {"prompt": "...", "model": "..."}

    Expected response::

        {"response": "...", "tokens_used": 123}   # tokens_used optional
        OR
        {"text": "...", "tokens": 123}             # alternative key names
        OR
        {"output": "...", "usage": {"total_tokens": 123}}
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: str = "",
        model_id: str = "custom",
        timeout: int = 60,
        max_retries: int = 2,
        extra_headers: dict | None = None,
        display_name: str | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model_id = model_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_headers = extra_headers or {}
        self.display_name = display_name or "custom"

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        payload = {"prompt": prompt, "model": self.model_id}
        payload.update(kwargs)

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
                    # Flexible response key detection
                    text = (
                        data.get("response")
                        or data.get("text")
                        or data.get("output")
                        or data.get("content")
                        or data.get("answer")
                        or str(data)
                    )
                    usage = data.get("usage", {})
                    tokens = (
                        data.get("tokens_used")
                        or data.get("tokens")
                        or usage.get("total_tokens")
                        or 0
                    )
                    return ModelResponse(
                        text=str(text),
                        tokens_used=int(tokens),
                        latency_ms=round(elapsed_ms, 2),
                        cost_usd=0.0,
                        model=self.model_id,
                        metadata={"provider": "custom_http", "endpoint": self.endpoint_url},
                    )

                if resp.status_code >= 500:
                    await asyncio.sleep(2 ** attempt)
                    last_exc = Exception(f"Server error: {resp.status_code} — {resp.text[:200]}")
                    continue

                raise RuntimeError(f"Endpoint returned {resp.status_code}: {resp.text[:400]}")

            except httpx.TimeoutException as exc:
                last_exc = exc
                await asyncio.sleep(2 ** attempt)
            except RuntimeError:
                raise
            except Exception as exc:
                last_exc = exc
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"Custom HTTP LLM failed after {self.max_retries} retries: {last_exc}")

    def get_model_info(self) -> dict:
        return {
            "name": self.display_name,
            "adapter_type": "llm",
            "description": f"Custom HTTP endpoint: {self.endpoint_url}",
            "config": {"endpoint": self.endpoint_url},
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_llm_adapter(
    provider_type: str,
    endpoint_url: str = "",
    api_key: str = "",
    model_id: str = "",
    extra_headers: dict | None = None,
    display_name: str | None = None,
    timeout: int = 60,
) -> LLMAdapter:
    """Build the correct LLM adapter from provider configuration.

    Parameters
    ----------
    provider_type:
        One of ``openai_compatible``, ``anthropic``, ``custom_llm``.
    endpoint_url:
        Full base URL or chat completions URL of the provider.
    api_key:
        Bearer token / API key for the endpoint.
    model_id:
        Model identifier string (e.g. ``llama-3-70b``, ``claude-3-5-sonnet``).
    """
    pt = provider_type.lower().strip()

    if pt in ("openai_compatible", "openai", "groq", "together", "vllm", "ollama", "lmstudio", "azure"):
        if not endpoint_url:
            endpoint_url = "https://api.openai.com/v1/chat/completions"
        return OpenAICompatibleAdapter(
            endpoint_url=endpoint_url,
            api_key=api_key,
            model_id=model_id or "gpt-4o",
            timeout=timeout,
            extra_headers=extra_headers,
            display_name=display_name,
        )

    if pt in ("anthropic", "claude"):
        return AnthropicAdapter(
            api_key=api_key,
            model_id=model_id or "claude-3-5-sonnet-20241022",
            timeout=timeout,
            display_name=display_name,
        )

    if pt in ("custom_llm", "custom", "custom_http"):
        if not endpoint_url:
            raise ValueError("endpoint_url is required for custom_llm provider type.")
        return CustomHTTPLLMAdapter(
            endpoint_url=endpoint_url,
            api_key=api_key,
            model_id=model_id or "custom",
            timeout=timeout,
            extra_headers=extra_headers,
            display_name=display_name,
        )

    raise ValueError(
        f"Unknown LLM provider_type {provider_type!r}. "
        f"Choose from: openai_compatible, anthropic, custom_llm"
    )
