"""LLM-as-Judge scoring — with a heuristic fallback when no LLM is available.

The ``LLMJudge`` can use a real LLM adapter to grade free-form responses on
a 0–100 scale, or fall back to a surprisingly effective keyword/structural
heuristic that works well enough for demos and testing.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalharness.adapters.base import LLMAdapter


class LLMJudge:
    """Scores responses using either an LLM or heuristic rules.

    Parameters
    ----------
    judge_adapter:
        An optional LLM adapter to use for AI-based judging.  If ``None``,
        the heuristic fallback is used automatically.
    """

    def __init__(self, judge_adapter: "LLMAdapter | None" = None) -> None:
        self._adapter = judge_adapter

    async def judge_response(
        self, prompt: str, response: str, criteria: str = ""
    ) -> float:
        """Score *response* to *prompt* on a 0–100 scale.

        If an adapter is configured, sends a structured judging prompt; otherwise
        falls back to :meth:`_heuristic_judge`.
        """
        if self._adapter is not None:
            return await self._llm_judge(prompt, response, criteria)
        return self._heuristic_judge(prompt, response)

    async def judge_pairwise(
        self, prompt: str, response_a: str, response_b: str
    ) -> str:
        """Compare two responses. Returns ``'A'``, ``'B'``, or ``'tie'``."""
        if self._adapter is not None:
            return await self._llm_pairwise(prompt, response_a, response_b)
        score_a = self._heuristic_judge(prompt, response_a)
        score_b = self._heuristic_judge(prompt, response_b)
        if abs(score_a - score_b) < 5:
            return "tie"
        return "A" if score_a > score_b else "B"

    # -- LLM-backed judging --------------------------------------------------

    async def _llm_judge(
        self, prompt: str, response: str, criteria: str
    ) -> float:
        """Use an LLM to score the response."""
        judge_prompt = (
            "You are an expert evaluator. Score the following response on a "
            "scale of 0 to 100.\n\n"
            f"## Original Prompt\n{prompt}\n\n"
            f"## Response\n{response}\n\n"
        )
        if criteria:
            judge_prompt += f"## Evaluation Criteria\n{criteria}\n\n"
        judge_prompt += (
            "Provide ONLY a numeric score between 0 and 100. "
            "No explanation needed."
        )

        result = await self._adapter.generate(judge_prompt)  # type: ignore[union-attr]
        try:
            numbers = re.findall(r"\d+\.?\d*", result.text)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(100.0, score))
        except (ValueError, IndexError):
            pass
        return 50.0  # fallback if parsing fails

    async def _llm_pairwise(
        self, prompt: str, response_a: str, response_b: str
    ) -> str:
        """Use an LLM for pairwise comparison."""
        judge_prompt = (
            "You are an expert evaluator. Compare the two responses below and "
            "decide which is better.\n\n"
            f"## Prompt\n{prompt}\n\n"
            f"## Response A\n{response_a}\n\n"
            f"## Response B\n{response_b}\n\n"
            "Reply with ONLY 'A', 'B', or 'tie'."
        )
        result = await self._adapter.generate(judge_prompt)  # type: ignore[union-attr]
        text = result.text.strip().upper()
        if "TIE" in text:
            return "tie"
        if "A" in text and "B" not in text:
            return "A"
        if "B" in text and "A" not in text:
            return "B"
        return "tie"

    # -- Heuristic fallback ---------------------------------------------------

    @staticmethod
    def _heuristic_judge(prompt: str, response: str) -> float:
        """Score based on response-quality signals (no LLM needed).

        Signals:
        - **Length appropriateness** — penalise very short or extremely long
        - **Keyword relevance** — overlap between prompt and response tokens
        - **Structure** — presence of sentences, punctuation, paragraphs
        - **Coherence proxies** — capitalisation, no excessive repetition
        """
        if not response or not response.strip():
            return 0.0

        score = 50.0  # start in the middle

        # --- 1. Length appropriateness ---
        word_count = len(response.split())
        if word_count < 5:
            score -= 20
        elif word_count < 15:
            score -= 10
        elif 15 <= word_count <= 500:
            score += 10
        elif word_count > 500:
            score += 5   # slightly verbose

        # --- 2. Keyword relevance ---
        prompt_words = set(re.findall(r"\w+", prompt.lower()))
        response_words = set(re.findall(r"\w+", response.lower()))
        # Remove stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "what",
            "which", "who", "whom", "this", "that", "these", "those", "it",
            "and", "or", "but", "not", "no", "if", "than", "how", "when",
            "where", "why",
        }
        prompt_keywords = prompt_words - stop_words
        response_keywords = response_words - stop_words

        if prompt_keywords:
            overlap = len(prompt_keywords & response_keywords) / len(prompt_keywords)
            score += overlap * 15  # up to +15

        # --- 3. Structure ---
        sentence_count = len(re.findall(r"[.!?]+", response))
        if sentence_count >= 2:
            score += 5
        if sentence_count >= 5:
            score += 5

        # Paragraph breaks
        if "\n\n" in response or "\n" in response:
            score += 3

        # Bullet points or numbered lists
        if re.search(r"^[\-\*\d]+[\.\)]\s", response, re.MULTILINE):
            score += 3

        # --- 4. Coherence ---
        # Starts with a capital letter
        if response[0].isupper():
            score += 2

        # Ends with punctuation
        if response.rstrip()[-1] in ".!?)\"'":
            score += 2

        # Check for excessive repetition
        words_list = response.lower().split()
        if len(words_list) > 10:
            unique_ratio = len(set(words_list)) / len(words_list)
            if unique_ratio < 0.3:
                score -= 20  # very repetitive
            elif unique_ratio < 0.5:
                score -= 10

        # --- 5. Contains errors / uncertainty phrases ---
        uncertainty_phrases = [
            "i'm not sure",
            "i don't know",
            "i cannot",
            "i can't",
            "unable to",
            "no information",
        ]
        response_lower = response.lower()
        for phrase in uncertainty_phrases:
            if phrase in response_lower:
                score -= 10
                break

        return max(0.0, min(100.0, score))
