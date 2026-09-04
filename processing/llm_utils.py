"""LLM utilities using Mistral API for text generation."""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Tried in order when the configured model is throttled. Both are on the free
# tier's open-weights allowance and share none of mistral-small's quota.
FALLBACK_MODELS = ["open-mistral-nemo", "ministral-8b-latest"]

# Process-wide memory of which models are currently throttled, so a five-stage
# pipeline does not pay the 429-and-wait tax on every stage. Entries expire.
_THROTTLED_UNTIL: dict[str, float] = {}
THROTTLE_COOLDOWN_SECONDS = 180.0


def mark_throttled(model: str) -> None:
    import time

    _THROTTLED_UNTIL[model] = time.time() + THROTTLE_COOLDOWN_SECONDS


def is_throttled(model: str) -> bool:
    import time

    return _THROTTLED_UNTIL.get(model, 0.0) > time.time()


def model_chain(primary: str) -> list[str]:
    """Primary then fallbacks, with recently throttled models moved to the end."""
    ordered = [primary] + [m for m in FALLBACK_MODELS if m != primary]
    ready = [m for m in ordered if not is_throttled(m)]
    cooling = [m for m in ordered if is_throttled(m)]
    return ready + cooling


class MistralLLM:
    """Wrapper for Mistral API text generation with rate limiting."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mistral-small-latest",
        max_retries: int = 6,
        base_delay: float = 0.5,
    ):
        """Initialize Mistral LLM client.

        Args:
            api_key: Mistral API key. Defaults to MISTRAL_API_KEY env var.
            model: Model to use for generation.
            max_retries: Max retry attempts for rate limits.
            base_delay: Base delay between requests.
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment")

        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._client = None
        self._last_request_time = 0

    @property
    def client(self):
        """Lazy-load Mistral client."""
        if self._client is None:
            from mistralai import Mistral

            self._client = Mistral(api_key=self.api_key)
        return self._client

    def _wait_for_rate_limit(self):
        """Wait to respect rate limits."""
        import time
        elapsed = time.time() - self._last_request_time
        if elapsed < self.base_delay:
            time.sleep(self.base_delay - elapsed)
        self._last_request_time = time.time()

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        """Generate text from prompt with retry logic.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        import time

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Rate limits on the free tier are per model, so a capped primary model
        # does not mean the key is exhausted: after two throttled attempts, fall
        # through to the next model in the chain rather than burning all retries
        # on one that will keep saying 429.
        models = model_chain(self.model)
        attempt = 0
        for model_index, model in enumerate(models):
            throttled_here = 0
            while attempt < self.max_retries:
                try:
                    self._wait_for_rate_limit()
                    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
                    response = self.client.chat.complete(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **extra,
                    )
                    if model != self.model:
                        logger.info(f"LLM served by fallback model {model}")
                    return response.choices[0].message.content

                except Exception as e:
                    error_str = str(e).lower()
                    if "429" not in error_str and "rate" not in error_str:
                        raise
                    attempt += 1
                    throttled_here += 1
                    mark_throttled(model)
                    if model_index < len(models) - 1:
                        logger.warning(f"LLM model {model} rate limited, trying {models[model_index + 1]}")
                        break
                    wait_time = min(30.0, 2.0 * (2 ** throttled_here) + 1)
                    logger.warning(f"LLM rate limited on {model}, waiting {wait_time:.1f}s (attempt {attempt}/{self.max_retries})")
                    time.sleep(wait_time)

        raise Exception(f"LLM generation failed after {self.max_retries} retries")

    def generate_trend_name(self, sample_texts: list[str]) -> str:
        """Generate a concise trend name from sample content.

        Args:
            sample_texts: List of representative texts from the trend cluster

        Returns:
            Short trend name (3-6 words)
        """
        combined = "\n".join(sample_texts[:5])  # Use top 5 samples

        system_prompt = """You are a hospitality trend analyst. Generate concise, catchy trend names.
Output ONLY the trend name, nothing else. Keep it 3-6 words."""

        prompt = f"""Based on these social media posts about hotels and travel, generate a single trend name:

{combined}

Trend name:"""

        return self.generate(prompt, system_prompt, max_tokens=20, temperature=0.5).strip()

    def generate_trend_description(self, sample_texts: list[str], trend_name: str) -> str:
        """Generate an insightful description of the trend.

        Args:
            sample_texts: Representative texts from the cluster
            trend_name: The trend name

        Returns:
            2-3 sentence description with specific insights
        """
        combined = "\n---\n".join(sample_texts[:7])

        system_prompt = """You are a hospitality trend analyst writing for hotel executives.
Your descriptions must be SPECIFIC and INSIGHTFUL - not generic.

BAD example: "This trend shows travelers are interested in wellness."
GOOD example: "Travelers are specifically seeking hotels with in-room yoga mats, meditation apps, and 24-hour wellness centers - not just spas. The demand is driven by remote workers wanting to maintain routines while traveling."

Focus on:
- WHAT specifically travelers want (concrete details from the posts)
- WHY this is happening (underlying drivers)
- WHO is driving this trend (demographics, traveler types)

Keep to 2-3 sentences. Be specific, not generic."""

        prompt = f"""Analyze these social media posts and explain the "{trend_name}" trend:

{combined}

Write a specific, insight-driven description of what this trend reveals about traveler preferences and behavior:"""

        return self.generate(prompt, system_prompt, max_tokens=200, temperature=0.6).strip()

    def generate_why_it_matters(
        self,
        trend_name: str,
        description: str,
        metrics: dict,
        sample_texts: list[str],
    ) -> str:
        """Generate strategic "Why it matters" analysis.

        Args:
            trend_name: Name of the trend
            description: Trend description
            metrics: Dict with volume, sentiment, engagement data
            sample_texts: Representative content

        Returns:
            Strategic analysis paragraph
        """
        combined = "\n".join(sample_texts[:3])

        system_prompt = """You are a hospitality strategy consultant advising hotel owners and brands.
Write actionable insights about why trends matter for business decisions.
Be specific about opportunities, risks, or strategic implications.
Keep responses to 2-3 sentences."""

        prompt = f"""Trend: {trend_name}
Description: {description}
Volume: {metrics.get('volume', 'N/A')} mentions
Sentiment: {metrics.get('sentiment', 'N/A')}

Sample posts:
{combined}

Why this trend matters for hospitality businesses:"""

        return self.generate(prompt, system_prompt, max_tokens=200, temperature=0.7).strip()

    def extract_topics(self, texts: list[str], max_topics: int = 5) -> list[str]:
        """Extract key topics/themes from texts.

        Args:
            texts: List of texts to analyze
            max_topics: Maximum number of topics

        Returns:
            List of topic strings
        """
        combined = "\n".join(texts[:10])

        system_prompt = """Extract key topics from hospitality content.
Output ONLY a comma-separated list of topics, nothing else.
Focus on specific themes like: wellness, sustainability, luxury, budget, remote work, etc."""

        prompt = f"""Extract {max_topics} key topics from these posts:

{combined}

Topics:"""

        result = self.generate(prompt, system_prompt, max_tokens=50, temperature=0.3)
        topics = [t.strip() for t in result.split(",")]
        return topics[:max_topics]


# Singleton instance
_llm: MistralLLM | None = None


def get_llm() -> MistralLLM:
    """Get the singleton MistralLLM instance."""
    global _llm
    if _llm is None:
        _llm = MistralLLM()
    return _llm


def generate_trend_insights(
    sample_texts: list[str],
    metrics: dict,
) -> dict:
    """Generate all trend insights in one call.

    Args:
        sample_texts: Representative texts from trend cluster
        metrics: Dict with volume, sentiment, engagement

    Returns:
        Dict with name, description, why_it_matters, topics
    """
    llm = get_llm()

    name = llm.generate_trend_name(sample_texts)
    description = llm.generate_trend_description(sample_texts, name)
    why_it_matters = llm.generate_why_it_matters(name, description, metrics, sample_texts)
    topics = llm.extract_topics(sample_texts)

    return {
        "name": name,
        "description": description,
        "why_it_matters": why_it_matters,
        "topics": topics,
    }
