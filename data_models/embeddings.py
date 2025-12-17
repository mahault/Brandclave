"""Embedding utilities with Mistral API and local fallback."""

import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class MistralEmbedding(EmbeddingProvider):
    """Mistral Embed API provider with rate limiting."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mistral-embed",
        max_retries: int = 5,
        base_delay: float = 1.0,
    ):
        """Initialize Mistral embedding provider.

        Args:
            api_key: Mistral API key. Defaults to MISTRAL_API_KEY env var.
            model: Model name. Defaults to mistral-embed.
            max_retries: Maximum retry attempts on rate limit errors.
            base_delay: Base delay between requests in seconds.
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

    def _call_with_retry(self, func, *args, **kwargs):
        """Call API function with exponential backoff retry."""
        import time
        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                # Check for rate limit error (429)
                if "429" in error_str or "rate" in error_str or "limit" in error_str:
                    wait_time = self.base_delay * (2 ** attempt) + (attempt * 0.5)
                    print(f"Rate limited, waiting {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                # Check for server errors (5xx)
                elif "500" in error_str or "502" in error_str or "503" in error_str:
                    wait_time = self.base_delay * (2 ** attempt)
                    print(f"Server error, waiting {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        raise Exception(f"Failed after {self.max_retries} retries")

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text with retry logic."""
        response = self._call_with_retry(
            self.client.embeddings.create,
            model=self.model,
            inputs=[text]
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """Generate embeddings for multiple texts in batches with rate limiting."""
        import time
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._call_with_retry(
                self.client.embeddings.create,
                model=self.model,
                inputs=batch
            )
            embeddings.extend([item.embedding for item in response.data])
            # Small delay between batches
            time.sleep(0.2)
        return embeddings

    @property
    def dimension(self) -> int:
        """Mistral embed dimension."""
        return 1024


class LocalEmbedding(EmbeddingProvider):
    """Local sentence-transformers embedding provider."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """Initialize local embedding provider.

        Args:
            model_name: Sentence-transformers model name.
            device: Device to run on ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy-load sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Model-dependent dimension."""
        # all-MiniLM-L6-v2 has 384 dimensions
        return self.model.get_sentence_embedding_dimension()


def get_embedding_provider(provider: str | None = None) -> EmbeddingProvider:
    """Get embedding provider based on configuration.

    Args:
        provider: Provider name ('mistral' or 'local'). Defaults to EMBEDDING_PROVIDER env var.

    Returns:
        Configured embedding provider.
    """
    provider = provider or os.getenv("EMBEDDING_PROVIDER", "mistral")

    if provider == "mistral":
        try:
            return MistralEmbedding()
        except ValueError:
            print("Warning: Mistral API key not found, falling back to local embeddings")
            return LocalEmbedding()
    elif provider == "local":
        return LocalEmbedding()
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# Convenience functions
_provider: EmbeddingProvider | None = None


def get_default_provider() -> EmbeddingProvider:
    """Get the default embedding provider (singleton)."""
    global _provider
    if _provider is None:
        _provider = get_embedding_provider()
    return _provider


def embed_text(text: str) -> list[float]:
    """Embed a single text using the default provider."""
    return get_default_provider().embed(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts using the default provider."""
    return get_default_provider().embed_batch(texts)
