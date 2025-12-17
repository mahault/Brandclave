"""Hotelier Bets service - Strategic move extraction and management.

Integrates with Move Extraction POMDP for adaptive extraction strategy.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from db.database import SessionLocal
from db.models import HotelierMoveModel, RawContentModel
from processing.move_extraction import (
    extract_move_from_article,
    generate_why_it_matters,
    extract_competitive_impact,
    fallback_move_detection,
)

logger = logging.getLogger(__name__)

# Import Move Extraction POMDP
try:
    from services.active_inference.move_extraction_pomdp import get_extraction_pomdp, MoveExtractionPOMDP
    EXTRACTION_POMDP_AVAILABLE = True
except ImportError:
    EXTRACTION_POMDP_AVAILABLE = False
    logger.info("Move Extraction POMDP not available")


class HotelierBetsService:
    """Service for extracting and managing Hotelier Bets moves.

    Supports adaptive extraction via Move Extraction POMDP.
    """

    def __init__(
        self,
        use_llm: bool = True,
        confidence_threshold: float = 0.5,
        use_adaptive: bool = True,
    ):
        """Initialize Hotelier Bets service.

        Args:
            use_llm: Whether to use LLM for extraction
            confidence_threshold: Minimum confidence to include a move
            use_adaptive: Whether to use POMDP for adaptive extraction strategy
        """
        self.use_llm = use_llm
        self.confidence_threshold = confidence_threshold

        # Initialize POMDP for adaptive extraction
        self.use_adaptive = use_adaptive and EXTRACTION_POMDP_AVAILABLE
        self.extraction_pomdp: Optional[MoveExtractionPOMDP] = None
        if self.use_adaptive:
            try:
                self.extraction_pomdp = get_extraction_pomdp()
                logger.info("Move Extraction POMDP enabled for adaptive strategy selection")
            except Exception as e:
                logger.warning(f"Failed to initialize Move Extraction POMDP: {e}")
                self.use_adaptive = False

    def extract_moves(
        self,
        days_back: int = 30,
        limit: int = 100,
        source_types: list[str] | None = None,
    ) -> list[dict]:
        """Extract strategic moves from news content.

        Args:
            days_back: Days of content to analyze
            limit: Maximum articles to process
            source_types: Filter by source types (default: ['news'])

        Returns:
            List of extracted move dicts
        """
        source_types = source_types or ["news"]
        logger.info(f"Extracting moves (days_back={days_back}, limit={limit})")

        db = SessionLocal()
        moves = []

        try:
            # Get news content that hasn't been processed for moves
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)

            query = db.query(RawContentModel).filter(
                RawContentModel.source_type.in_(source_types),
                RawContentModel.scraped_at >= cutoff_date,
            )

            # Get articles with sufficient content
            articles = query.order_by(
                RawContentModel.scraped_at.desc()
            ).limit(limit).all()

            if not articles:
                logger.warning("No news articles found")
                return []

            logger.info(f"Processing {len(articles)} articles for move extraction")

            for article in articles:
                try:
                    move = self._extract_from_article(article)
                    if move and move.get("confidence_score", 0) >= self.confidence_threshold:
                        move["source_content_id"] = article.id
                        move["published_at"] = article.published_at
                        moves.append(move)
                except Exception as e:
                    logger.warning(f"Error processing article {article.id}: {e}")
                    continue

            logger.info(f"Extracted {len(moves)} moves from {len(articles)} articles")
            return moves

        finally:
            db.close()

    def _extract_from_article(self, article: RawContentModel) -> dict | None:
        """Extract move from a single article.

        Uses POMDP for adaptive method selection when available.

        Args:
            article: RawContentModel instance

        Returns:
            Move dict or None
        """
        title = article.title or ""
        content = article.content or ""

        # Skip articles with minimal content
        if len(content) < 100:
            return None

        # Prepare article dict for POMDP
        article_dict = {
            "id": article.id,
            "title": title,
            "content": content,
            "source": article.source,
        }

        # Use POMDP for method selection if available
        if self.extraction_pomdp is not None:
            decision = self.extraction_pomdp.select_extraction_method(article_dict)
            method = decision["method"]
            logger.debug(f"POMDP selected method: {method} (EFE={decision.get('efe', 0):.3f})")

            move = None
            if method == "skip":
                # POMDP decided to skip this article
                self.extraction_pomdp.observe_extraction_result(article_dict, method, None)
                return None
            elif method == "regex":
                # Use regex-based detection (no LLM cost)
                move = fallback_move_detection(title, content)
                if move:
                    move["source_url"] = article.url
                    move["source_name"] = article.source
                    move["extraction_method"] = "regex"
            elif method == "llm_single":
                # Standard LLM extraction
                move = extract_move_from_article(
                    title=title,
                    content=content,
                    source_url=article.url,
                    source_name=article.source,
                )
                if move:
                    move["extraction_method"] = "llm_single"
            else:  # llm_multi
                # Multi-sample LLM extraction (higher quality, higher cost)
                move = self._multi_sample_extraction(title, content, article.url, article.source)
                if move:
                    move["extraction_method"] = "llm_multi"

            # Update POMDP with result
            self.extraction_pomdp.observe_extraction_result(article_dict, method, move)

            # Generate additional insights if move found
            if move:
                move = self._enhance_move(move, title)

            return move

        # Non-adaptive path (original logic)
        if self.use_llm:
            move = extract_move_from_article(
                title=title,
                content=content,
                source_url=article.url,
                source_name=article.source,
            )

            if move:
                move = self._enhance_move(move, title)
                return move
        else:
            move = fallback_move_detection(title, content)
            if move:
                move["source_url"] = article.url
                move["source_name"] = article.source
                return move

        return None

    def _enhance_move(self, move: dict, title: str) -> dict:
        """Add additional insights to a move."""
        # Generate why it matters if not present
        if not move.get("why_it_matters"):
            move["why_it_matters"] = generate_why_it_matters(
                title=move.get("title", title),
                summary=move.get("summary", ""),
                company=move.get("company", "Unknown"),
                move_type=move.get("move_type", "other"),
                market=move.get("market"),
            )

        # Generate competitive impact
        move["competitive_impact"] = extract_competitive_impact(
            company=move.get("company", "Unknown"),
            move_type=move.get("move_type", "other"),
            market=move.get("market"),
            summary=move.get("summary", ""),
        )

        return move

    def _multi_sample_extraction(
        self,
        title: str,
        content: str,
        source_url: str,
        source_name: str,
    ) -> dict | None:
        """Multi-sample LLM extraction for higher quality.

        Runs multiple LLM calls and aggregates results.
        """
        # Run 3 extractions and pick best
        results = []
        for _ in range(3):
            move = extract_move_from_article(
                title=title,
                content=content,
                source_url=source_url,
                source_name=source_name,
            )
            if move:
                results.append(move)

        if not results:
            return None

        # Pick result with highest confidence
        best = max(results, key=lambda m: m.get("confidence_score", 0))

        # Boost confidence slightly for multi-sample
        best["confidence_score"] = min(1.0, best.get("confidence_score", 0.5) * 1.1)

        return best

    def get_pomdp_status(self) -> dict:
        """Get Move Extraction POMDP status.

        Returns:
            Dict with POMDP state information
        """
        if self.extraction_pomdp is not None:
            return self.extraction_pomdp.get_status()
        return {"enabled": False, "reason": "POMDP not available"}

    def save_moves(self, moves: list[dict]) -> int:
        """Save extracted moves to database.

        Args:
            moves: List of move dicts

        Returns:
            Number of moves saved
        """
        db = SessionLocal()
        saved = 0

        try:
            for move in moves:
                # Check for duplicate by source_content_id
                existing = db.query(HotelierMoveModel).filter(
                    HotelierMoveModel.source_content_id == move.get("source_content_id")
                ).first()

                if existing:
                    logger.debug(f"Move already exists for content: {move.get('source_content_id')}")
                    continue

                db_move = HotelierMoveModel(
                    title=move.get("title", "Untitled Move"),
                    summary=move.get("summary", ""),
                    why_it_matters=move.get("why_it_matters", ""),
                    company=move.get("company", "Unknown"),
                    company_type=move.get("company_type"),
                    move_type=move.get("move_type", "other"),
                    market=move.get("market"),
                    investment_amount=move.get("investment_amount"),
                    strategic_implications=move.get("strategic_implications", []),
                    competitive_impact=move.get("competitive_impact"),
                    source_url=move.get("source_url", ""),
                    source_name=move.get("source_name", ""),
                    published_at=move.get("published_at"),
                    source_content_id=move.get("source_content_id"),
                    confidence_score=move.get("confidence_score", 0.5),
                    metadata_json=move.get("metadata", {}),
                )
                db.add(db_move)
                saved += 1

            db.commit()
            logger.info(f"Saved {saved} moves to database")

        except Exception as e:
            db.rollback()
            logger.error(f"Error saving moves: {e}")
            raise

        finally:
            db.close()

        return saved

    def get_moves(
        self,
        limit: int = 20,
        company: str | None = None,
        move_type: str | None = None,
        market: str | None = None,
        min_confidence: float = 0,
    ) -> list[dict]:
        """Get moves from database with filters.

        Args:
            limit: Maximum moves to return
            company: Filter by company name (partial match)
            move_type: Filter by move type
            market: Filter by market (partial match)
            min_confidence: Minimum confidence score

        Returns:
            List of move dicts
        """
        db = SessionLocal()
        try:
            query = db.query(HotelierMoveModel).filter(
                HotelierMoveModel.confidence_score >= min_confidence
            )

            if company:
                query = query.filter(
                    HotelierMoveModel.company.ilike(f"%{company}%")
                )
            if move_type:
                query = query.filter(HotelierMoveModel.move_type == move_type)
            if market:
                query = query.filter(
                    HotelierMoveModel.market.ilike(f"%{market}%")
                )

            moves = query.order_by(
                HotelierMoveModel.extracted_at.desc()
            ).limit(limit).all()

            return [self._model_to_dict(m) for m in moves]

        finally:
            db.close()

    def get_move_by_id(self, move_id: str) -> dict | None:
        """Get a single move by ID.

        Args:
            move_id: Move ID

        Returns:
            Move dict or None
        """
        db = SessionLocal()
        try:
            move = db.query(HotelierMoveModel).filter(
                HotelierMoveModel.id == move_id
            ).first()

            return self._model_to_dict(move) if move else None

        finally:
            db.close()

    def get_companies(self) -> list[str]:
        """Get list of unique companies.

        Returns:
            List of company names
        """
        db = SessionLocal()
        try:
            results = db.query(HotelierMoveModel.company).distinct().all()
            return sorted([r[0] for r in results if r[0]])
        finally:
            db.close()

    def get_move_types(self) -> list[str]:
        """Get list of move types in use.

        Returns:
            List of move type strings
        """
        db = SessionLocal()
        try:
            results = db.query(HotelierMoveModel.move_type).distinct().all()
            return sorted([r[0] for r in results if r[0]])
        finally:
            db.close()

    def get_markets(self) -> list[str]:
        """Get list of unique markets.

        Returns:
            List of market names
        """
        db = SessionLocal()
        try:
            results = db.query(HotelierMoveModel.market).distinct().all()
            return sorted([r[0] for r in results if r[0]])
        finally:
            db.close()

    def _model_to_dict(self, model: HotelierMoveModel) -> dict:
        """Convert HotelierMoveModel to dict."""
        return {
            "id": model.id,
            "title": model.title,
            "summary": model.summary,
            "why_it_matters": model.why_it_matters,
            "company": model.company,
            "company_type": model.company_type,
            "move_type": model.move_type,
            "market": model.market,
            "investment_amount": model.investment_amount,
            "strategic_implications": model.strategic_implications or [],
            "competitive_impact": model.competitive_impact,
            "source_url": model.source_url,
            "source_name": model.source_name,
            "published_at": model.published_at.isoformat() if model.published_at else None,
            "extracted_at": model.extracted_at.isoformat() if model.extracted_at else None,
            "confidence_score": model.confidence_score,
        }


def generate_hotelier_bets(
    days_back: int = 30,
    limit: int = 100,
    save: bool = True,
) -> list[dict]:
    """Convenience function to generate Hotelier Bets moves.

    Args:
        days_back: Days of content to analyze
        limit: Maximum articles to process
        save: Whether to save to database

    Returns:
        List of extracted moves
    """
    service = HotelierBetsService()
    moves = service.extract_moves(days_back=days_back, limit=limit)

    if save and moves:
        service.save_moves(moves)

    return moves
