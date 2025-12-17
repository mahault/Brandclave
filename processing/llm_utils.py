"""LLM utilities using Mistral API for text generation."""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class MistralLLM:
    """Wrapper for Mistral API text generation with rate limiting."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mistral-small-latest",
        max_retries: int = 3,
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

        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content

            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate" in error_str:
                    wait_time = self.base_delay * (2 ** attempt) + 1
                    logger.warning(f"LLM rate limited, waiting {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise

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
        """Generate a brief description of the trend.

        Args:
            sample_texts: Representative texts from the cluster
            trend_name: The trend name

        Returns:
            1-2 sentence description
        """
        combined = "\n".join(sample_texts[:5])

        system_prompt = """You are a hospitality trend analyst. Write brief, insightful trend descriptions.
Keep responses to 1-2 sentences."""

        prompt = f"""Describe the "{trend_name}" trend based on these posts:

{combined}

Description:"""

        return self.generate(prompt, system_prompt, max_tokens=100, temperature=0.6).strip()

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
