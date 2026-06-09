"""Mock LLM adapters with four model profiles of varying quality.

Each profile simulates realistic latency, token usage, and cost while returning
plausible responses drawn from a built-in knowledge base.  The quality of the
response is scaled by the profile so that comparative benchmarks are meaningful.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from evalharness.adapters.base import LLMAdapter, ModelResponse

# ---------------------------------------------------------------------------
# Built-in response knowledge base
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE: dict[str, dict[str, str]] = {
    # keyword -> {quality_tier: answer}
    "capital of france": {
        "high": "The capital of France is Paris. It has been the capital since the late 10th century and is the country's largest city, home to approximately 2.1 million people in the city proper.",
        "medium": "The capital of France is Paris.",
        "low": "France's capital is Paris, a city in Europe.",
    },
    "capital of japan": {
        "high": "The capital of Japan is Tokyo (東京). Originally known as Edo, it was renamed Tokyo in 1868 when it became the imperial capital. The Greater Tokyo Area is the most populous metropolitan region in the world with over 37 million inhabitants.",
        "medium": "Tokyo is the capital of Japan.",
        "low": "Japan's capital is Tokyo.",
    },
    "photosynthesis": {
        "high": "Photosynthesis is the biological process by which green plants, algae, and certain bacteria convert light energy (usually from the sun) into chemical energy stored in glucose. The overall equation is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂. It occurs primarily in the chloroplasts and involves two stages: the light-dependent reactions (in the thylakoid membranes) and the Calvin cycle (in the stroma).",
        "medium": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. The equation is 6CO2 + 6H2O → C6H12O6 + 6O2.",
        "low": "Photosynthesis is when plants use sunlight to make food. They take in CO2 and water and produce sugar and oxygen.",
    },
    "pythagorean theorem": {
        "high": "The Pythagorean theorem states that in a right-angled triangle, the square of the length of the hypotenuse (c) equals the sum of the squares of the other two sides (a and b): a² + b² = c². This theorem is attributed to the ancient Greek mathematician Pythagoras, though it was known to Babylonian mathematicians much earlier. It has applications in geometry, trigonometry, physics, and engineering.",
        "medium": "The Pythagorean theorem says that for a right triangle, a² + b² = c², where c is the hypotenuse.",
        "low": "Pythagorean theorem: a squared plus b squared equals c squared for right triangles.",
    },
    "machine learning": {
        "high": "Machine learning is a subset of artificial intelligence in which algorithms learn patterns from data to make predictions or decisions without being explicitly programmed for each scenario. The three main paradigms are supervised learning (labeled data), unsupervised learning (unlabeled data), and reinforcement learning (reward signals). Common algorithms include linear regression, decision trees, random forests, support vector machines, and neural networks. Deep learning, a sub-field using multi-layer neural networks, has driven recent breakthroughs in computer vision, NLP, and generative AI.",
        "medium": "Machine learning is a branch of AI where computers learn patterns from data. There are three types: supervised learning, unsupervised learning, and reinforcement learning. Popular algorithms include decision trees, SVMs, and neural networks.",
        "low": "Machine learning is when computers learn from data to make predictions. It's part of AI.",
    },
    "theory of relativity": {
        "high": "Einstein's theory of relativity consists of two interrelated theories: special relativity (1905) and general relativity (1915). Special relativity introduced the famous equation E = mc² and showed that the laws of physics are the same for all non-accelerating observers, while the speed of light is constant regardless of the observer's motion. General relativity extends this to include gravity, describing it not as a force but as a curvature of spacetime caused by mass and energy. Predictions include gravitational time dilation, gravitational lensing, and gravitational waves — all confirmed experimentally.",
        "medium": "Einstein's theory of relativity includes special relativity (E=mc², constant speed of light) and general relativity (gravity as curvature of spacetime). Both have been experimentally confirmed.",
        "low": "Relativity is Einstein's famous theory. E=mc² is the key equation. It's about how space and time work.",
    },
    "water boiling point": {
        "high": "Water boils at 100 °C (212 °F) at standard atmospheric pressure (1 atm, 101.325 kPa). The boiling point decreases at higher altitudes due to lower atmospheric pressure — for example, water boils at about 95 °C at 1,500 m elevation. Adding solutes such as salt raises the boiling point through the colligative property known as boiling-point elevation.",
        "medium": "Water boils at 100°C (212°F) at standard atmospheric pressure.",
        "low": "Water boils at 100 degrees Celsius.",
    },
    "python programming": {
        "high": "Python is a high-level, interpreted, general-purpose programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability through significant indentation and a clean syntax. Python supports multiple paradigms including procedural, object-oriented, and functional programming. Its extensive standard library and rich ecosystem of third-party packages (NumPy, pandas, Django, Flask, TensorFlow) make it popular for web development, data science, machine learning, automation, and scientific computing. CPython is the reference implementation, though alternatives like PyPy and Jython exist.",
        "medium": "Python is a high-level programming language created by Guido van Rossum in 1991. It's known for readable syntax and is widely used in web development, data science, and AI.",
        "low": "Python is a popular programming language used for many things like web development and AI.",
    },
    "mitochondria": {
        "high": "Mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells. Often called the 'powerhouses of the cell,' they generate most of the cell's supply of adenosine triphosphate (ATP) through oxidative phosphorylation. Mitochondria have their own circular DNA (mtDNA) and are believed to have originated from an endosymbiotic event in which an ancestral eukaryotic cell engulfed an aerobic bacterium. Each mitochondrion has an outer membrane, an intermembrane space, an inner membrane folded into cristae, and a matrix where the citric acid cycle occurs.",
        "medium": "Mitochondria are organelles in eukaryotic cells that produce ATP through cellular respiration. They have their own DNA and are often called the 'powerhouse of the cell.'",
        "low": "Mitochondria are the powerhouse of the cell. They make energy (ATP).",
    },
    "fibonacci": {
        "high": "The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, … Formally, F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2). Named after Leonardo of Pisa (Fibonacci), the sequence appears in biological settings such as branching in trees, the arrangement of leaves, and the spiral patterns of sunflowers. The ratio of consecutive Fibonacci numbers converges to the golden ratio φ ≈ 1.6180339887.",
        "medium": "The Fibonacci sequence starts 0, 1, 1, 2, 3, 5, 8, 13, … where each number is the sum of the previous two. It appears in nature and relates to the golden ratio.",
        "low": "Fibonacci numbers are 0, 1, 1, 2, 3, 5, 8, 13... each number is the sum of the two before it.",
    },
    "gravity": {
        "high": "Gravity is one of the four fundamental forces of nature. In Newtonian mechanics it is described by F = G·m₁·m₂/r², where G is the gravitational constant (6.674 × 10⁻¹¹ N·m²/kg²). Einstein's general relativity reinterprets gravity as the curvature of spacetime caused by mass-energy. On Earth, objects experience gravitational acceleration of approximately 9.81 m/s². Gravity governs planetary orbits, tidal forces, and the large-scale structure of the universe.",
        "medium": "Gravity is a fundamental force of attraction between objects with mass. On Earth, it causes objects to accelerate at about 9.81 m/s². Newton described it with F = Gm1m2/r².",
        "low": "Gravity is the force that pulls things toward Earth. It's about 9.8 m/s².",
    },
    "dna": {
        "high": "DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions for the development, functioning, growth, and reproduction of all known organisms and many viruses. Its structure — a double helix composed of two polynucleotide strands — was elucidated by Watson and Crick in 1953, building on X-ray crystallography by Rosalind Franklin. The four nucleotide bases are adenine (A), thymine (T), guanine (G), and cytosine (C), paired A-T and G-C. The human genome contains approximately 3.2 billion base pairs organized into 23 pairs of chromosomes.",
        "medium": "DNA is a double-helix molecule that carries genetic information. It's made of four bases: A, T, G, C. Watson and Crick described its structure in 1953.",
        "low": "DNA is the genetic material in cells. It has a double helix shape with four bases: A, T, G, C.",
    },
    "sorting algorithm": {
        "high": "Common sorting algorithms include: QuickSort (average O(n log n), in-place, unstable), MergeSort (O(n log n) guaranteed, stable, requires O(n) extra space), HeapSort (O(n log n), in-place, unstable), TimSort (hybrid of MergeSort and InsertionSort, O(n log n), used in Python's built-in sort and Java), BubbleSort (O(n²), simple but inefficient), and InsertionSort (O(n²) worst case but efficient for small or nearly-sorted data). The theoretical lower bound for comparison-based sorting is Ω(n log n). Non-comparison sorts like RadixSort and CountingSort can achieve O(n·k) in specific cases.",
        "medium": "Major sorting algorithms include QuickSort and MergeSort (both O(n log n)), and BubbleSort and InsertionSort (O(n²)). Python uses TimSort, a hybrid algorithm.",
        "low": "Sorting algorithms put things in order. QuickSort and MergeSort are fast ones. BubbleSort is slow.",
    },
    "neural network": {
        "high": "An artificial neural network (ANN) is a computational model inspired by biological neural networks. It consists of layers of interconnected nodes (neurons): an input layer, one or more hidden layers, and an output layer. Each connection has a weight adjusted during training via backpropagation and gradient descent. Activation functions (ReLU, sigmoid, tanh, softmax) introduce non-linearity. Architectures include feedforward networks, convolutional neural networks (CNNs) for vision, recurrent networks (RNNs/LSTMs) for sequences, and transformers for language. Deep neural networks with many hidden layers have driven breakthroughs in image recognition, natural language processing, and game playing.",
        "medium": "Neural networks are computing models with layers of connected nodes. They learn by adjusting weights through backpropagation. Types include CNNs for images and RNNs for sequences.",
        "low": "Neural networks are AI models with layers that learn patterns from data. They're used in image recognition and language processing.",
    },
    "climate change": {
        "high": "Climate change refers to long-term shifts in global temperatures and weather patterns. While natural factors like volcanic eruptions and solar cycles play a role, human activities — primarily burning fossil fuels (coal, oil, natural gas) — have been the dominant driver since the Industrial Revolution. Atmospheric CO₂ has risen from ~280 ppm pre-industrial to over 420 ppm today. The IPCC's Sixth Assessment Report (2021–2023) concluded that global surface temperature has increased by approximately 1.1 °C above pre-industrial levels. Consequences include rising sea levels, more frequent extreme weather events, ocean acidification, and biodiversity loss.",
        "medium": "Climate change is the long-term shift in temperatures and weather, primarily driven by human activities like burning fossil fuels. Global temperatures have risen about 1.1°C since pre-industrial times.",
        "low": "Climate change is when Earth's temperatures rise due to pollution and burning fossil fuels.",
    },
}

# Generic fallback responses for prompts not matching the knowledge base.
_FALLBACK_RESPONSES: dict[str, list[str]] = {
    "high": [
        "Based on the available information, I can provide a comprehensive analysis. {topic} is a multifaceted subject with significant implications across several domains. Key considerations include the historical context, current research findings, and practical applications that have emerged in recent years.",
        "This is an excellent question. {topic} has been extensively studied and documented. The consensus among experts is that it involves complex interactions between multiple factors. Let me outline the main points and their interrelationships.",
        "To address your question about {topic}: the current understanding, supported by peer-reviewed research, indicates several important aspects. First, the foundational principles establish the framework. Second, empirical evidence supports the theoretical predictions. Third, practical applications continue to evolve.",
    ],
    "medium": [
        "{topic} is a well-known subject. The key points are that it involves several important concepts and has practical applications in everyday life and various fields.",
        "Regarding {topic}: it's an important area with several key facts and principles that are widely accepted in the relevant field.",
    ],
    "low": [
        "{topic} is something that exists and is studied by experts. It has some key features and is relevant to certain fields.",
        "I know a bit about {topic}. It's a topic that people study and has some important aspects.",
    ],
}

# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------

_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "gpt-4o-mock": {
        "quality": "high",
        "base_latency_ms": 800,
        "correct_probability": 0.92,
        "cost_per_1k_input": 0.005,
        "cost_per_1k_output": 0.015,
        "avg_output_tokens": 280,
        "description": "High-quality GPT-4o mock — accurate, detailed, slower",
    },
    "gpt-4o-mini-mock": {
        "quality": "medium",
        "base_latency_ms": 300,
        "correct_probability": 0.78,
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
        "avg_output_tokens": 150,
        "description": "Mid-tier GPT-4o-mini mock — decent quality, fast, cheap",
    },
    "claude-3.5-mock": {
        "quality": "high",
        "base_latency_ms": 600,
        "correct_probability": 0.90,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "avg_output_tokens": 260,
        "description": "High-quality Claude 3.5 mock — thorough and accurate",
    },
    "llama-3-mock": {
        "quality": "low",
        "base_latency_ms": 200,
        "correct_probability": 0.65,
        "cost_per_1k_input": 0.0001,
        "cost_per_1k_output": 0.0001,
        "avg_output_tokens": 100,
        "description": "Open-source Llama-3 mock — fastest, lower quality",
    },
}


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------


class MockLLMAdapter(LLMAdapter):
    """A deterministic-ish LLM adapter that returns realistic mock responses.

    Each instance is bound to a model *profile* that controls quality, latency,
    and cost characteristics.
    """

    def __init__(self, model_name: str) -> None:
        if model_name not in _MODEL_PROFILES:
            raise ValueError(
                f"Unknown mock model profile {model_name!r}. "
                f"Available: {list(_MODEL_PROFILES)}"
            )
        self.model_name = model_name
        self._profile = _MODEL_PROFILES[model_name]

    # -- LLMAdapter interface ------------------------------------------------

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a mock completion with realistic latency / token counts."""
        profile = self._profile
        quality = profile["quality"]

        # 1. Simulate latency (± 30 % variance)
        base_ms: float = profile["base_latency_ms"]
        latency_ms = base_ms * random.uniform(0.7, 1.3)
        await asyncio.sleep(latency_ms / 1000.0)

        # 2. Choose response
        response_text = self._pick_response(prompt, quality)

        # Occasionally return a wrong answer for lower-quality profiles
        if random.random() > profile["correct_probability"]:
            response_text = self._introduce_error(response_text, quality)

        # 3. Token counts
        input_tokens = max(1, len(prompt.split()) * 4 // 3)
        output_tokens = int(
            profile["avg_output_tokens"] * random.uniform(0.7, 1.3)
        )

        # 4. Cost
        cost = (
            input_tokens / 1000 * profile["cost_per_1k_input"]
            + output_tokens / 1000 * profile["cost_per_1k_output"]
        )

        return ModelResponse(
            text=response_text,
            tokens_used=input_tokens + output_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=round(cost, 8),
            model=self.model_name,
            metadata={"quality_tier": quality},
        )

    def get_model_info(self) -> dict:
        """Return adapter metadata."""
        return {
            "name": self.model_name,
            "adapter_type": "llm",
            "description": self._profile["description"],
            "config": {
                "base_latency_ms": self._profile["base_latency_ms"],
                "correct_probability": self._profile["correct_probability"],
            },
        }

    # -- Private helpers -----------------------------------------------------

    @staticmethod
    def _pick_response(prompt: str, quality: str) -> str:
        """Return a plausible response by matching prompt keywords."""
        prompt_lower = prompt.lower()

        for keyword, answers in _KNOWLEDGE_BASE.items():
            if keyword in prompt_lower:
                return answers.get(quality, answers.get("medium", ""))

        # Fallback: generate a generic response
        topic = prompt.strip()[:80]
        templates = _FALLBACK_RESPONSES.get(quality, _FALLBACK_RESPONSES["medium"])
        return random.choice(templates).format(topic=topic)

    @staticmethod
    def _introduce_error(correct_response: str, quality: str) -> str:
        """Mutate a correct response to simulate an incorrect answer."""
        error_strategies = [
            # Add a wrong qualifier
            lambda r: r + " However, this is widely debated and some sources suggest otherwise.",
            # Truncate — simulate cut-off
            lambda r: r[: len(r) // 2] + "...",
            # Swap a key word
            lambda r: r.replace("is", "might be", 1),
            # Add a hallucinated fact
            lambda r: r + " Interestingly, recent studies in 2024 have completely overturned this understanding.",
        ]
        if quality == "low":
            # Low-quality models make bigger mistakes
            error_strategies.append(
                lambda _: "I'm not entirely sure about this topic, but I think the answer involves several factors that are complex and interconnected."
            )
        return random.choice(error_strategies)(correct_response)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_all_mock_llm_adapters() -> dict[str, MockLLMAdapter]:
    """Return a dict of ``name → MockLLMAdapter`` for every profile."""
    return {name: MockLLMAdapter(name) for name in _MODEL_PROFILES}
