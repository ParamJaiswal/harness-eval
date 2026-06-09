"""Mock RAG adapters with precise and noisy retrieval profiles.

Two pipeline profiles simulate retrieval-augmented generation with different
quality characteristics.  A built-in document store provides ~20 passages
covering common knowledge topics so that retrieval scores are meaningful.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from evalharness.adapters.base import RAGAdapter, RAGResponse, RetrievedContext

# ---------------------------------------------------------------------------
# Built-in document store (~20 passages)
# ---------------------------------------------------------------------------

_DOCUMENT_STORE: list[dict[str, str]] = [
    {
        "chunk_id": "doc-001",
        "source": "science/physics.md",
        "text": "Gravity is a fundamental force of nature described by Newton's law of universal gravitation F=Gm1m2/r². Einstein's general relativity reinterprets gravity as spacetime curvature. On Earth, gravitational acceleration is approximately 9.81 m/s².",
        "topics": "gravity physics newton einstein",
    },
    {
        "chunk_id": "doc-002",
        "source": "science/biology.md",
        "text": "Photosynthesis is the process by which green plants convert sunlight, carbon dioxide, and water into glucose and oxygen. The overall equation is 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂. It occurs in chloroplasts through light-dependent reactions and the Calvin cycle.",
        "topics": "photosynthesis plants biology chloroplast",
    },
    {
        "chunk_id": "doc-003",
        "source": "science/biology.md",
        "text": "DNA (deoxyribonucleic acid) is the molecule carrying genetic instructions. Its double-helix structure was described by Watson and Crick in 1953. The four nucleotide bases are adenine (A), thymine (T), guanine (G), and cytosine (C), pairing A-T and G-C.",
        "topics": "dna genetics biology watson crick",
    },
    {
        "chunk_id": "doc-004",
        "source": "science/biology.md",
        "text": "Mitochondria are membrane-bound organelles in eukaryotic cells that produce ATP through oxidative phosphorylation. Often called the 'powerhouse of the cell,' they contain their own circular DNA and likely originated through endosymbiosis.",
        "topics": "mitochondria cell biology atp energy",
    },
    {
        "chunk_id": "doc-005",
        "source": "geography/capitals.md",
        "text": "Paris is the capital and largest city of France, with a population of approximately 2.1 million in the city proper. It has been the French capital since the late 10th century and is renowned for landmarks like the Eiffel Tower and the Louvre.",
        "topics": "paris france capital city europe",
    },
    {
        "chunk_id": "doc-006",
        "source": "geography/capitals.md",
        "text": "Tokyo is the capital of Japan, originally known as Edo. It became the imperial capital in 1868 and the Greater Tokyo Area is the most populous metropolitan region in the world with over 37 million inhabitants.",
        "topics": "tokyo japan capital city asia",
    },
    {
        "chunk_id": "doc-007",
        "source": "cs/algorithms.md",
        "text": "Common sorting algorithms include QuickSort (average O(n log n), in-place), MergeSort (O(n log n), stable), HeapSort, TimSort (used in Python), BubbleSort (O(n²)), and InsertionSort. The theoretical lower bound for comparison-based sorting is Ω(n log n).",
        "topics": "sorting algorithm quicksort mergesort computer science",
    },
    {
        "chunk_id": "doc-008",
        "source": "cs/ml.md",
        "text": "Machine learning is a subset of AI where algorithms learn patterns from data. The three main paradigms are supervised learning (labeled data), unsupervised learning (unlabeled data), and reinforcement learning (reward signals). Deep learning uses multi-layer neural networks.",
        "topics": "machine learning ai deep learning neural network",
    },
    {
        "chunk_id": "doc-009",
        "source": "cs/ml.md",
        "text": "An artificial neural network consists of layers of interconnected nodes: input, hidden, and output layers. Connections have weights adjusted via backpropagation. Architectures include CNNs for vision, RNNs for sequences, and transformers for language tasks.",
        "topics": "neural network cnn rnn transformer deep learning",
    },
    {
        "chunk_id": "doc-010",
        "source": "math/theorems.md",
        "text": "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a² + b² = c². It has applications in geometry, trigonometry, physics, and engineering.",
        "topics": "pythagorean theorem mathematics geometry triangle",
    },
    {
        "chunk_id": "doc-011",
        "source": "math/sequences.md",
        "text": "The Fibonacci sequence is 0, 1, 1, 2, 3, 5, 8, 13, 21, 34… where each number is the sum of the two preceding ones. The ratio of consecutive Fibonacci numbers converges to the golden ratio φ ≈ 1.618. The sequence appears in branching patterns and spiral arrangements in nature.",
        "topics": "fibonacci sequence mathematics golden ratio nature",
    },
    {
        "chunk_id": "doc-012",
        "source": "science/physics.md",
        "text": "Einstein's theory of relativity includes special relativity (1905, E=mc²) and general relativity (1915, gravity as spacetime curvature). Predictions include gravitational time dilation, gravitational lensing, and gravitational waves, all confirmed experimentally.",
        "topics": "relativity einstein physics spacetime",
    },
    {
        "chunk_id": "doc-013",
        "source": "science/chemistry.md",
        "text": "Water (H₂O) boils at 100°C (212°F) at standard atmospheric pressure (1 atm). The boiling point decreases at higher altitudes. Adding solutes raises the boiling point through boiling-point elevation, a colligative property.",
        "topics": "water boiling point chemistry temperature",
    },
    {
        "chunk_id": "doc-014",
        "source": "cs/python.md",
        "text": "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. It emphasizes readability through significant indentation. Python supports procedural, OOP, and functional paradigms. Its ecosystem includes NumPy, pandas, Django, and TensorFlow.",
        "topics": "python programming language guido van rossum",
    },
    {
        "chunk_id": "doc-015",
        "source": "science/environment.md",
        "text": "Climate change refers to long-term shifts in global temperatures driven primarily by burning fossil fuels since the Industrial Revolution. Atmospheric CO₂ has risen from ~280 ppm to over 420 ppm. Global temperatures have increased approximately 1.1°C above pre-industrial levels.",
        "topics": "climate change global warming environment co2",
    },
    {
        "chunk_id": "doc-016",
        "source": "history/science.md",
        "text": "The scientific method involves observation, hypothesis formation, experimentation, data analysis, and conclusion. Peer review ensures quality control. Reproducibility is a cornerstone — results must be independently verifiable for broad acceptance.",
        "topics": "scientific method research hypothesis experiment",
    },
    {
        "chunk_id": "doc-017",
        "source": "cs/databases.md",
        "text": "Relational databases store data in tables with rows and columns, using SQL for queries. Key concepts include primary keys, foreign keys, normalization, ACID properties (Atomicity, Consistency, Isolation, Durability), and indexing for performance optimization.",
        "topics": "database sql relational acid normalization",
    },
    {
        "chunk_id": "doc-018",
        "source": "cs/networking.md",
        "text": "The TCP/IP model has four layers: Application (HTTP, FTP, DNS), Transport (TCP, UDP), Internet (IP, ICMP), and Network Access (Ethernet, Wi-Fi). TCP provides reliable, ordered delivery; UDP provides fast, connectionless communication.",
        "topics": "networking tcp ip http protocol internet",
    },
    {
        "chunk_id": "doc-019",
        "source": "math/calculus.md",
        "text": "Calculus, developed independently by Newton and Leibniz in the 17th century, has two main branches: differential calculus (rates of change, derivatives) and integral calculus (accumulation, areas under curves). The fundamental theorem of calculus connects the two.",
        "topics": "calculus derivative integral mathematics newton leibniz",
    },
    {
        "chunk_id": "doc-020",
        "source": "science/astronomy.md",
        "text": "The Solar System consists of the Sun, eight planets (Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune), dwarf planets like Pluto, and numerous moons, asteroids, and comets. It formed approximately 4.6 billion years ago from a solar nebula.",
        "topics": "solar system planets astronomy sun earth",
    },
]

# Irrelevant passages for the noisy pipeline
_IRRELEVANT_PASSAGES: list[dict[str, str]] = [
    {
        "chunk_id": "noise-001",
        "source": "recipes/cooking.md",
        "text": "To make the perfect sourdough bread, combine 500g flour with 350g water and 100g active starter. Let it autolyse for 30 minutes before adding 10g salt. Bulk ferment for 4-6 hours at room temperature.",
    },
    {
        "chunk_id": "noise-002",
        "source": "sports/football.md",
        "text": "The FIFA World Cup is held every four years. Brazil has won the most titles (5), followed by Germany and Italy (4 each). The 2022 World Cup in Qatar was won by Argentina, led by Lionel Messi.",
    },
    {
        "chunk_id": "noise-003",
        "source": "entertainment/movies.md",
        "text": "The highest-grossing film of all time is Avatar (2009), directed by James Cameron, earning over $2.9 billion worldwide. The Marvel Cinematic Universe is the highest-grossing film franchise.",
    },
    {
        "chunk_id": "noise-004",
        "source": "lifestyle/gardening.md",
        "text": "Tomatoes thrive in full sunlight and well-drained soil with a pH between 6.0 and 6.8. Plant seedlings 18-24 inches apart after the last frost date. Water consistently and fertilize every two weeks.",
    },
]

# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

_RAG_PROFILES: dict[str, dict[str, Any]] = {
    "rag-precise-mock": {
        "quality": "precise",
        "top_k": 3,
        "relevance_noise": 0.05,       # low noise in relevance scores
        "include_irrelevant_prob": 0.05,
        "hallucination_prob": 0.05,
        "base_retrieval_latency_ms": 80,
        "base_generation_latency_ms": 400,
        "avg_tokens": 220,
        "cost_per_1k_tokens": 0.008,
        "description": "Precise RAG pipeline — highly relevant retrieval, faithful generation",
    },
    "rag-noisy-mock": {
        "quality": "noisy",
        "top_k": 5,
        "relevance_noise": 0.30,        # high noise in relevance scores
        "include_irrelevant_prob": 0.50,
        "hallucination_prob": 0.30,
        "base_retrieval_latency_ms": 120,
        "base_generation_latency_ms": 500,
        "avg_tokens": 260,
        "cost_per_1k_tokens": 0.008,
        "description": "Noisy RAG pipeline — includes irrelevant contexts, occasional hallucination",
    },
}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class MockRAGAdapter(RAGAdapter):
    """Simulates a retrieve-then-generate pipeline with configurable quality.

    Parameters
    ----------
    profile_name:
        One of ``"rag-precise-mock"`` or ``"rag-noisy-mock"``.
    """

    def __init__(self, profile_name: str) -> None:
        if profile_name not in _RAG_PROFILES:
            raise ValueError(
                f"Unknown RAG profile {profile_name!r}. "
                f"Available: {list(_RAG_PROFILES)}"
            )
        self.profile_name = profile_name
        self._profile = _RAG_PROFILES[profile_name]

    # -- RAGAdapter interface ------------------------------------------------

    async def query(self, question: str, **kwargs: Any) -> RAGResponse:
        """Run mock retrieve → generate pipeline for *question*."""
        profile = self._profile

        # --- Retrieval phase ---
        retrieval_latency = profile["base_retrieval_latency_ms"] * random.uniform(0.7, 1.3)
        await asyncio.sleep(retrieval_latency / 1000.0 * 0.1)

        contexts = self._retrieve(question)

        # --- Generation phase ---
        gen_latency = profile["base_generation_latency_ms"] * random.uniform(0.7, 1.3)
        await asyncio.sleep(gen_latency / 1000.0 * 0.1)

        answer = self._generate_answer(question, contexts)

        tokens = int(profile["avg_tokens"] * random.uniform(0.8, 1.2))
        cost = tokens / 1000 * profile["cost_per_1k_tokens"]
        total_latency = retrieval_latency + gen_latency

        return RAGResponse(
            answer=answer,
            retrieved_contexts=contexts,
            tokens_used=tokens,
            latency_ms=round(total_latency, 2),
            cost_usd=round(cost, 8),
            retrieval_latency_ms=round(retrieval_latency, 2),
            generation_latency_ms=round(gen_latency, 2),
        )

    def get_pipeline_info(self) -> dict:
        """Return metadata about this RAG pipeline."""
        return {
            "name": self.profile_name,
            "adapter_type": "rag",
            "description": self._profile["description"],
            "config": {
                "top_k": self._profile["top_k"],
                "hallucination_prob": self._profile["hallucination_prob"],
            },
        }

    # -- Private helpers -----------------------------------------------------

    def _retrieve(self, question: str) -> list[RetrievedContext]:
        """Simulate vector-similarity retrieval from the document store."""
        profile = self._profile
        question_lower = question.lower()
        question_words = set(question_lower.split())

        # Score each document by keyword overlap
        scored: list[tuple[float, dict]] = []
        for doc in _DOCUMENT_STORE:
            topic_words = set(doc["topics"].split())
            overlap = len(question_words & topic_words)
            # Normalized score with some randomness
            raw_score = overlap / max(len(question_words), 1)
            noise = random.uniform(-profile["relevance_noise"], profile["relevance_noise"])
            score = max(0.0, min(1.0, raw_score + noise))
            scored.append((score, doc))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = profile["top_k"]
        selected = scored[:top_k]

        contexts: list[RetrievedContext] = []
        for score, doc in selected:
            contexts.append(
                RetrievedContext(
                    text=doc["text"],
                    source=doc["source"],
                    relevance_score=round(score, 4),
                    chunk_id=doc["chunk_id"],
                )
            )

        # Noisy pipeline: inject irrelevant documents
        if random.random() < profile["include_irrelevant_prob"]:
            noise_doc = random.choice(_IRRELEVANT_PASSAGES)
            contexts.append(
                RetrievedContext(
                    text=noise_doc["text"],
                    source=noise_doc["source"],
                    relevance_score=round(random.uniform(0.1, 0.4), 4),
                    chunk_id=noise_doc["chunk_id"],
                )
            )

        return contexts

    def _generate_answer(
        self, question: str, contexts: list[RetrievedContext]
    ) -> str:
        """Generate an answer grounded (or not) in the retrieved contexts."""
        profile = self._profile

        # Gather context text
        context_texts = [c.text for c in contexts if c.relevance_score > 0.1]

        if not context_texts:
            return (
                "Based on the available information, I could not find a definitive "
                "answer to your question. The retrieved documents did not contain "
                "sufficiently relevant information."
            )

        # Build answer from best context
        best_context = context_texts[0]

        # Faithful answer: summarise the best context
        answer = (
            f"Based on the retrieved information: {best_context[:200]} "
            f"In summary, the evidence from the sources directly addresses "
            f"the question about {question[:50]}."
        )

        # Hallucination injection for noisy pipeline
        if random.random() < profile["hallucination_prob"]:
            hallucinations = [
                " Additionally, a groundbreaking 2025 study at MIT found completely contradictory results that have not yet been published.",
                " Furthermore, according to unpublished research, these findings may be entirely incorrect due to methodological flaws.",
                " It's also worth noting that a secret government report from 2024 revealed additional factors not captured in public literature.",
                " Recent leaked documents suggest that the mainstream understanding is only partially correct.",
            ]
            answer += random.choice(hallucinations)

        return answer


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_all_mock_rag_adapters() -> dict[str, MockRAGAdapter]:
    """Return a dict of ``name → MockRAGAdapter`` for every profile."""
    return {name: MockRAGAdapter(name) for name in _RAG_PROFILES}
