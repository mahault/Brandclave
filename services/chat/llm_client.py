"""Mistral LLM client for chat generation."""

import logging
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    raw: Any = None


class MistralLLMClient:
    """Mistral chat client for response generation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mistral-small-latest",
    ):
        """Initialize Mistral client.

        Args:
            api_key: Mistral API key. Defaults to MISTRAL_API_KEY env var.
            model: Model to use for chat.
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment")

        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy-load Mistral client."""
        if self._client is None:
            from mistralai import Mistral
            self._client = Mistral(api_key=self.api_key)
        return self._client

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        """Generate a chat response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        # Free-tier rate limits are per model; when the configured model is
        # throttled, the open-weights models still answer. Same chain as
        # processing.llm_utils so every LLM surface degrades the same way.
        from processing.llm_utils import FALLBACK_MODELS

        models = [self.model] + [m for m in FALLBACK_MODELS if m != self.model]
        last_error: Exception | None = None
        for model in models:
            try:
                response = self.client.chat.complete(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if model != self.model:
                    logger.info(f"Chat served by fallback model {model}")
                content = response.choices[0].message.content
                return LLMResponse(content=content, raw=response)

            except Exception as e:
                last_error = e
                if "429" in str(e) or "rate" in str(e).lower():
                    logger.warning(f"Mistral chat rate limited on {model}, trying next model")
                    continue
                logger.error(f"Mistral chat failed: {e}")
                raise

        logger.error(f"Mistral chat failed on every model: {last_error}")
        raise last_error


def get_llm_client() -> MistralLLMClient | None:
    """Get LLM client if API key is available."""
    try:
        return MistralLLMClient()
    except ValueError:
        logger.warning("Mistral API key not available, LLM disabled")
        return None
