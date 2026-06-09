"""HTTP adapter for evaluating customer RAG pipeline endpoints.

Customers expose a single webhook endpoint. The harness sends the query,
receives the answer + retrieved contexts, and scores faithfulness/retrieval.

Expected webhook contract
-------------------------

Request (POST <endpoint_url>)::

    {
        "query": "What is photosynthesis?",
        "top_k": 5            # optional, harness hint
    }

Response (one of the following formats is accepted)::

    # Format A — preferred
    {
        "answer": "Photosynthesis is...",
        "contexts": [
            {"text": "...", "source": "doc1.pdf", "score": 0.92},
            {"text": "...", "source": "doc2.pdf", "score": 0.87}
        ]
    }

    # Format B — minimal (no contexts)
    {
        "answer": "Photosynthesis is..."
    }

    # Format C — alternative keys
    {
        "response": "...",
        "retrieved_documents": [{"content": "...", "id": "doc-1"}]
    }
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from evalharness.adapters.base import RAGAdapter, RAGResponse, RetrievedContext


class RAGWebhookAdapter(RAGAdapter):
    """Calls a customer-hosted RAG pipeline via HTTP webhook."""

    def __init__(
        self,
        endpoint_url: str,
        api_key: str = "",
        timeout: int = 60,
        max_retries: int = 2,
        top_k: int = 5,
        extra_headers: dict | None = None,
        display_name: str | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.top_k = top_k
        self.extra_headers = extra_headers or {}
        self.display_name = display_name or "rag-webhook"

    async def query(self, question: str, **kwargs: Any) -> RAGResponse:
        payload = {"query": question, "top_k": self.top_k}
        payload.update(kwargs)

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            t0_total = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(self.endpoint_url, json=payload, headers=headers)
                total_elapsed_ms = (time.perf_counter() - t0_total) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()

                    # --- Parse answer ---
                    answer = (
                        data.get("answer")
                        or data.get("response")
                        or data.get("text")
                        or data.get("output")
                        or ""
                    )

                    # --- Parse contexts ---
                    raw_contexts = (
                        data.get("contexts")
                        or data.get("retrieved_documents")
                        or data.get("documents")
                        or data.get("chunks")
                        or []
                    )
                    contexts = []
                    for i, ctx in enumerate(raw_contexts):
                        if isinstance(ctx, str):
                            contexts.append(RetrievedContext(
                                text=ctx,
                                source=f"chunk_{i}",
                                relevance_score=1.0,
                                chunk_id=str(i),
                            ))
                        elif isinstance(ctx, dict):
                            contexts.append(RetrievedContext(
                                text=ctx.get("text") or ctx.get("content") or ctx.get("page_content") or "",
                                source=ctx.get("source") or ctx.get("id") or ctx.get("doc_id") or f"chunk_{i}",
                                relevance_score=float(ctx.get("score") or ctx.get("relevance_score") or 1.0),
                                chunk_id=str(ctx.get("chunk_id") or ctx.get("id") or i),
                            ))

                    # --- Parse timing hints if provided ---
                    retrieval_ms = float(data.get("retrieval_latency_ms") or total_elapsed_ms * 0.3)
                    generation_ms = float(data.get("generation_latency_ms") or total_elapsed_ms * 0.7)
                    tokens = int(data.get("tokens_used") or data.get("tokens") or 0)

                    return RAGResponse(
                        answer=str(answer),
                        retrieved_contexts=contexts,
                        tokens_used=tokens,
                        latency_ms=round(total_elapsed_ms, 2),
                        cost_usd=0.0,
                        retrieval_latency_ms=round(retrieval_ms, 2),
                        generation_latency_ms=round(generation_ms, 2),
                    )

                if resp.status_code >= 500:
                    await asyncio.sleep(2 ** attempt)
                    last_exc = Exception(f"RAG endpoint server error: {resp.status_code} — {resp.text[:200]}")
                    continue

                raise RuntimeError(f"RAG endpoint returned {resp.status_code}: {resp.text[:400]}")

            except httpx.TimeoutException as exc:
                last_exc = exc
                await asyncio.sleep(2 ** attempt)
            except RuntimeError:
                raise
            except Exception as exc:
                last_exc = exc
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"RAG webhook failed after {self.max_retries} retries: {last_exc}")

    def get_pipeline_info(self) -> dict:
        return {
            "name": self.display_name,
            "adapter_type": "rag",
            "description": f"RAG webhook endpoint: {self.endpoint_url}",
            "config": {"endpoint": self.endpoint_url, "top_k": self.top_k},
        }
