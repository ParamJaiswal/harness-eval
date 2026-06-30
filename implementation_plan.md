# Add Dynamic API Keys & Custom Benchmarks

This plan outlines the changes required to allow you to input API keys and build custom test questions directly in the dashboard UI, without needing to save `.env` or YAML files on the server.

## Goal Description
Provide a seamless, browser-based experience for configuring real LLMs and creating custom test sets on-the-fly. We will use browser `localStorage` for API keys and dynamic, in-memory execution for custom benchmarks.

> [!IMPORTANT]
> **User Review Required**
> Please review the proposed changes below. By keeping API keys in `localStorage` and passing custom tasks directly in the API payload, we completely avoid modifying `.env` or saving YAML files, which is secure and clean. We will also add support for OpenRouter/Groq custom base URLs and clear RAG examples.

## Proposed Changes

### Frontend (Dashboard UI)

#### [MODIFY] `dashboard/index.html`
- Add a **Settings Modal** (gear icon in the header) with input fields for:
  - OpenAI API Key
  - Anthropic API Key
  - **Custom OpenAI-Compatible API Key** (e.g., Groq, OpenRouter)
  - **Custom Base URL** (e.g., `https://openrouter.ai/api/v1` or `https://api.groq.com/openai/v1`)
- Add a **Custom Benchmark Builder** section inside the "Run Evaluation" tab to add/remove custom tasks.
- Change the benchmark dropdown to include a "Create Custom Benchmark..." option.
- In the "Customer Endpoint" tab, add a clear **RAG Pipeline Example** section showing how to test a custom RAG API with the existing `rag_faithfulness` and `rag_retrieval` benchmarks.

#### [MODIFY] `dashboard/app.js`
- Add functions to save and load API keys from `localStorage`.
- Update the `runEval()` function:
  - If a custom benchmark is being built, serialize the form into a `custom_tasks` JSON array.
  - Read API keys from `localStorage` and attach them to the `POST /api/eval/run` payload.

#### [MODIFY] `dashboard/styles.css`
- Add styles for the new Settings modal and the Custom Benchmark builder form (task cards, add/remove buttons).

---

### Backend (API & Engine)

#### [MODIFY] `evalharness/models/schemas.py`
- Update the `EvalRunCreate` Pydantic schema to accept new optional fields:
  - `custom_tasks: list[dict] | None = None`
  - `openai_api_key: str | None = None`
  - `anthropic_api_key: str | None = None`
  - `custom_base_url: str | None = None` (for Groq/OpenRouter overrides)

#### [MODIFY] `evalharness/api/routes.py`
- Update `start_evaluation` (`POST /api/eval/run`):
  - If real models (e.g., `gpt-4o`) are requested, use the provided `openai_api_key` to build a live `OpenAILLMAdapter` on-the-fly (bypassing the need for a server `.env` variable).
  - Pass the `custom_tasks` array to the background runner if provided.

#### [MODIFY] `evalharness/engine/runner.py`
- Update `run_evaluation` to check if `benchmark_name == "custom"`.
- If so, bypass the YAML `BenchmarkLoader` and dynamically construct a `Benchmark` object in memory using the `custom_tasks` provided in the request payload.

---

## Verification Plan

### Automated Tests
- The existing pytest suite will be run to ensure no regressions in the core `EvalRunner` and `Scorer` logic.

### Manual Verification
1. Open the dashboard and input an OpenAI API key into the new Settings modal.
2. Select "Create Custom Benchmark..." and add two custom questions (e.g., "What is 2+2?", expected: "4").
3. Select a real model (e.g., `gpt-4o`) and run the evaluation.
4. Verify the run completes successfully, grades the custom questions correctly, and doesn't save any files to the backend server.
