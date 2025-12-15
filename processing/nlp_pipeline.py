"""NLP Pipeline for processing scraped content."""

import logging
import os
from dataclasses import dataclass
from datetime import datetime

import yaml
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
from textblob import TextBlob

from data_models.embeddings import get_embedding_provider, EmbeddingProvider
from db.database import SessionLocal
from db.models import RawContentModel
from db.vector_store import get_vector_store, VectorStore
from processing.cleaning import clean_text

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single content item."""

    content_id: str
    success: bool
    language: str | None = None
    sentiment_score: float | None = None
    embedding_id: str | None = None
    error: str | None = None


class NLPPipeline:
    """Main NLP processing pipeline."""

    def __init__(
        self,
        config_path: str = "configs/nlp.yaml",
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ):
        """Initialize the NLP pipeline.

        Args:
            config_path: Path to NLP configuration YAML
            embedding_provider: Custom embedding provider (uses default if None)
            vector_store: Custom vector store (uses default if None)
        """
        self.config = self._load_config(config_path)
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.vector_store = vector_store or get_vector_store()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {
            "text_cleaning": {
                "remove_html": True,
                "remove_urls": True,
                "remove_emails": True,
                "lowercase": False,
                "remove_extra_whitespace": True,
                "min_text_length": 50,
            },
            "sentiment": {"thresholds": {"positive": 0.1, "negative": -0.1}},
            "language_detection": {"supported_languages": ["en"], "default_language": "en"},
        }

    def detect_language(self, text: str) -> str | None:
        """Detect language of text.

        Args:
            text: Input text

        Returns:
            ISO language code or None if detection fails
        """
        try:
            lang = detect(text)
            supported = self.config.get("language_detection", {}).get(
                "supported_languages", ["en"]
            )
            if lang in supported:
                return lang
            return self.config.get("language_detection", {}).get("default_language", "en")
        except LangDetectException:
            logger.warning("Language detection failed")
            return None

    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text.

        Args:
            text: Input text

        Returns:
            Sentiment score from -1 (negative) to 1 (positive)
        """
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def process_content(self, content: RawContentModel) -> ProcessingResult:
        """Process a single content item through the pipeline.

        Args:
            content: RawContentModel to process

        Returns:
            ProcessingResult with outcomes
        """
        try:
            # Clean text
            cleaning_config = self.config.get("text_cleaning", {})
            cleaned_text = clean_text(
                content.content,
                remove_html_tags=cleaning_config.get("remove_html", True),
                remove_url=cleaning_config.get("remove_urls", True),
                remove_email=cleaning_config.get("remove_emails", True),
                lowercase=cleaning_config.get("lowercase", False),
                normalize_whitespace=cleaning_config.get("remove_extra_whitespace", True),
                min_length=cleaning_config.get("min_text_length", 50),
            )

            if not cleaned_text:
                return ProcessingResult(
                    content_id=content.id,
                    success=False,
                    error="Text too short after cleaning",
                )

            # Detect language
            language = self.detect_language(cleaned_text)

            # Analyze sentiment
            sentiment_score = self.analyze_sentiment(cleaned_text)

            # Generate embedding
            embedding = self.embedding_provider.embed(cleaned_text)

            # Store in vector store
            metadata = {
                "source": content.source,
                "source_type": content.source_type,
                "language": language,
                "sentiment": sentiment_score,
            }
            embedding_id = self.vector_store.add_content_embedding(
                id=content.id,
                embedding=embedding,
                text=cleaned_text[:1000],  # Store truncated text
                metadata=metadata,
            )

            return ProcessingResult(
                content_id=content.id,
                success=True,
                language=language,
                sentiment_score=sentiment_score,
                embedding_id=embedding_id,
            )

        except Exception as e:
            logger.error(f"Error processing content {content.id}: {e}")
            return ProcessingResult(
                content_id=content.id,
                success=False,
                error=str(e),
            )

    def process_batch(
        self,
        contents: list[RawContentModel],
        update_db: bool = True,
    ) -> list[ProcessingResult]:
        """Process multiple content items.

        Args:
            contents: List of RawContentModel items
            update_db: Whether to update the database with results

        Returns:
            List of ProcessingResult
        """
        results = []

        for content in contents:
            result = self.process_content(content)
            results.append(result)

            if update_db and result.success:
                self._update_content_in_db(content.id, result)

        return results

    def _update_content_in_db(self, content_id: str, result: ProcessingResult) -> None:
        """Update content record with processing results."""
        db = SessionLocal()
        try:
            content = db.query(RawContentModel).filter(RawContentModel.id == content_id).first()
            if content:
                content.is_processed = True
                content.language = result.language
                content.sentiment_score = result.sentiment_score
                content.embedding_id = result.embedding_id
                db.commit()
        finally:
            db.close()

    def process_unprocessed(self, limit: int = 100) -> list[ProcessingResult]:
        """Process unprocessed content from database.

        Args:
            limit: Maximum number of items to process

        Returns:
            List of ProcessingResult
        """
        db = SessionLocal()
        try:
            contents = (
                db.query(RawContentModel)
                .filter(RawContentModel.is_processed == False)
                .limit(limit)
                .all()
            )

            if not contents:
                logger.info("No unprocessed content found")
                return []

            logger.info(f"Processing {len(contents)} items")
            return self.process_batch(contents, update_db=True)

        finally:
            db.close()


def run_pipeline(limit: int = 100) -> dict:
    """Run the NLP pipeline on unprocessed content.

    Args:
        limit: Maximum items to process

    Returns:
        Summary statistics
    """
    pipeline = NLPPipeline()
    results = pipeline.process_unprocessed(limit=limit)

    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    summary = {
        "total_processed": len(results),
        "successful": successful,
        "failed": failed,
        "timestamp": datetime.utcnow().isoformat(),
    }

    logger.info(f"Pipeline complete: {summary}")
    return summary
