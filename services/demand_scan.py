"""Demand Scan service - Property analysis and trend matching."""

import logging
from datetime import datetime
from typing import Any

from db.database import SessionLocal
from db.models import PropertyFeaturesModel, TrendSignalModel
from data_models.property_features import PropertyFeatures, PropertyType, PriceSegment
from ingestion.properties.property_scraper import PropertyScraper
from processing.property_analysis import (
    extract_property_features,
    detect_region,
    extract_price_indicators,
)

logger = logging.getLogger(__name__)


class DemandScanService:
    """Service for property analysis and demand fitting."""

    def __init__(self, use_llm: bool = True):
        """Initialize Demand Scan service.

        Args:
            use_llm: Whether to use LLM for feature extraction
        """
        self.use_llm = use_llm

    def scan_property(self, url: str) -> dict | None:
        """Scan a property URL and analyze against demand trends.

        Args:
            url: Property website URL

        Returns:
            Dict with property features and demand analysis, or None
        """
        logger.info(f"Scanning property: {url}")

        # Step 1: Scrape the property website
        with PropertyScraper() as scraper:
            raw_content = scraper.scrape_url(url)

        if not raw_content:
            logger.error(f"Failed to scrape property: {url}")
            return None

        # Step 2: Extract property features
        features = extract_property_features(raw_content.content, url)

        # Step 3: Detect region if not extracted
        if not features.get("location"):
            features["region"] = detect_region(raw_content.content)
        else:
            features["region"] = detect_region(raw_content.content, features.get("location"))

        # Step 4: Extract price indicators
        features["price_indicators"] = extract_price_indicators(raw_content.content)

        # Step 5: Load regional trends and compute demand fit
        trends = self._get_regional_trends(features.get("region"))
        demand_fit = self._compute_demand_fit(features, trends)

        # Step 6: Identify gaps and opportunities
        gaps = self._identify_experience_gaps(features, trends)
        opportunities = self._identify_opportunities(features, trends)
        advantages = self._identify_competitive_advantages(features, trends)
        recommendations = self._generate_recommendations(features, gaps, opportunities)

        # Ensure we have a property name
        property_name = features.get("name")
        if not property_name:
            property_name = self._extract_name_from_url(url)

        # Build final result
        result = {
            "url": url,
            "name": property_name,
            "property_type": features.get("property_type", "hotel"),
            "brand_positioning": features.get("brand_positioning"),
            "tagline": features.get("tagline"),
            "tone": features.get("tone"),
            "themes": features.get("themes", []),
            "amenities": features.get("amenities", []),
            "room_types": features.get("room_types", []),
            "dining_options": features.get("dining_options", []),
            "experiences": features.get("experiences", []),
            "location": features.get("location"),
            "region": features.get("region"),
            "price_segment": features.get("price_segment", "unknown"),
            "price_indicators": features.get("price_indicators", []),
            "demand_fit_score": demand_fit["score"],
            "experience_gaps": gaps,
            "opportunity_lanes": opportunities,
            "competitive_advantages": advantages,
            "recommendations": recommendations,
            "matching_trend_ids": demand_fit["matching_trend_ids"],
            "scraped_at": datetime.utcnow().isoformat(),
            "source_content_id": None,  # Could link to RawContent if saved
        }

        return result

    def _extract_name_from_url(self, url: str) -> str:
        """Extract a property name from the URL.

        Args:
            url: Property website URL

        Returns:
            Property name extracted from URL
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove common prefixes
        domain = domain.replace("www.", "")

        # Extract the main domain name
        parts = domain.split(".")
        if parts:
            # Get the main name (usually first or second part)
            name = parts[0]

            # Handle subdomains
            if name in ["hotel", "hotels", "resort", "booking"]:
                if len(parts) > 1:
                    name = parts[1]

            # Clean up and format
            name = name.replace("-", " ").replace("_", " ")

            # Capitalize words
            name = " ".join(word.capitalize() for word in name.split())

            # Add context if it looks like a hotel
            if not any(word.lower() in ["hotel", "resort", "inn", "lodge", "suites"] for word in name.split()):
                return f"{name} Property"

            return name

        return "Analyzed Property"

    def _get_regional_trends(self, region: str | None) -> list[dict]:
        """Load trends, optionally filtered by region.

        Args:
            region: Region to filter by, or None for all

        Returns:
            List of trend dicts
        """
        db = SessionLocal()
        try:
            query = db.query(TrendSignalModel)

            if region:
                # Include trends matching region OR global trends (no region)
                query = query.filter(
                    (TrendSignalModel.region == region) |
                    (TrendSignalModel.region.is_(None))
                )

            trends = query.order_by(
                TrendSignalModel.strength_score.desc()
            ).limit(50).all()

            return [
                {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "topics": t.topics or [],
                    "strength_score": t.strength_score,
                    "white_space_score": t.white_space_score,
                    "audience_segment": t.audience_segment,
                    "region": t.region,
                }
                for t in trends
            ]
        finally:
            db.close()

    def _compute_demand_fit(
        self,
        features: dict,
        trends: list[dict],
    ) -> dict:
        """Compute demand fit score based on trend matching.

        Args:
            features: Property features dict
            trends: List of trend dicts

        Returns:
            Dict with score and matching_trend_ids
        """
        if not trends:
            return {"score": 0.5, "matching_trend_ids": []}

        property_keywords = set()

        # Collect property keywords
        for theme in features.get("themes", []):
            property_keywords.add(theme.lower())
        for amenity in features.get("amenities", []):
            property_keywords.update(amenity.lower().split())
        for exp in features.get("experiences", []):
            property_keywords.update(exp.lower().split())

        # Add positioning keywords
        if features.get("brand_positioning"):
            property_keywords.update(features["brand_positioning"].lower().split())

        matching_trends = []
        total_weight = 0
        match_weight = 0

        for trend in trends:
            trend_keywords = set()
            trend_keywords.add(trend["name"].lower())
            for topic in trend.get("topics", []):
                trend_keywords.add(topic.lower())

            # Check overlap
            overlap = property_keywords & trend_keywords

            if overlap:
                matching_trends.append(trend["id"])
                # Weight by trend strength
                match_weight += trend["strength_score"]

            total_weight += trend["strength_score"]

        # Calculate score
        if total_weight > 0:
            score = min(1.0, (match_weight / total_weight) * 2)  # Scale up for better differentiation
        else:
            score = 0.5

        return {
            "score": round(score, 2),
            "matching_trend_ids": matching_trends,
        }

    def _identify_experience_gaps(
        self,
        features: dict,
        trends: list[dict],
    ) -> list[str]:
        """Identify trending experiences not offered by property.

        Args:
            features: Property features dict
            trends: List of trend dicts

        Returns:
            List of gap descriptions
        """
        gaps = []

        property_offerings = set()
        for amenity in features.get("amenities", []):
            property_offerings.add(amenity.lower())
        for exp in features.get("experiences", []):
            property_offerings.add(exp.lower())
        for theme in features.get("themes", []):
            property_offerings.add(theme.lower())

        # Check high-strength trends
        strong_trends = [t for t in trends if t["strength_score"] > 0.3]

        for trend in strong_trends[:10]:
            trend_topics = [t.lower() for t in trend.get("topics", [])]
            trend_name = trend["name"].lower()

            # Check if property covers this trend
            covered = any(
                topic in " ".join(property_offerings)
                for topic in trend_topics
            ) or any(
                offering in trend_name
                for offering in property_offerings
            )

            if not covered:
                strength_pct = int(trend["strength_score"] * 100)
                gaps.append(
                    f"{trend['name']} (trending at {strength_pct}% strength)"
                )

        return gaps[:5]

    def _identify_opportunities(
        self,
        features: dict,
        trends: list[dict],
    ) -> list[str]:
        """Identify positioning opportunities based on trends.

        Args:
            features: Property features dict
            trends: List of trend dicts

        Returns:
            List of opportunity descriptions
        """
        opportunities = []

        # Look for high white-space trends
        whitespace_trends = sorted(
            trends,
            key=lambda t: t.get("white_space_score", 0),
            reverse=True,
        )[:5]

        for trend in whitespace_trends:
            if trend.get("white_space_score", 0) > 0.3:
                opportunities.append(
                    f"Position as leader in '{trend['name']}' - high demand, low competition"
                )

        # Suggest based on property type
        property_type = features.get("property_type", "hotel")
        themes = features.get("themes", [])

        if property_type == "boutique" and "design" not in themes:
            opportunities.append("Emphasize design-forward positioning for boutique appeal")

        if "wellness" in themes:
            opportunities.append("Expand wellness programming to capture growing mindfulness trend")

        if features.get("price_segment") == "luxury":
            opportunities.append("Develop exclusive experiences for ultra-high-net-worth travelers")

        return opportunities[:5]

    def _identify_competitive_advantages(
        self,
        features: dict,
        trends: list[dict],
    ) -> list[str]:
        """Identify property's competitive advantages.

        Args:
            features: Property features dict
            trends: List of trend dicts

        Returns:
            List of advantage descriptions
        """
        advantages = []

        # Premium amenities
        premium_amenities = ["spa", "pool", "fitness", "concierge", "butler", "private beach"]
        property_amenities = [a.lower() for a in features.get("amenities", [])]

        for premium in premium_amenities:
            if any(premium in a for a in property_amenities):
                advantages.append(f"Premium {premium} facilities")

        # Strong positioning
        if features.get("brand_positioning"):
            advantages.append(f"Clear brand positioning: {features['brand_positioning'][:100]}")

        # Unique themes
        unique_themes = ["eco", "wellness", "adventure", "cultural"]
        property_themes = [t.lower() for t in features.get("themes", [])]

        for unique in unique_themes:
            if unique in property_themes:
                advantages.append(f"Strong {unique} positioning differentiates from competitors")

        # Location advantages
        if features.get("location"):
            location = features["location"].lower()
            if any(kw in location for kw in ["beach", "ocean", "sea", "coast"]):
                advantages.append("Prime beachfront/coastal location")
            elif any(kw in location for kw in ["downtown", "central", "city center"]):
                advantages.append("Central urban location with accessibility")

        return advantages[:5]

    def _generate_recommendations(
        self,
        features: dict,
        gaps: list[str],
        opportunities: list[str],
    ) -> list[str]:
        """Generate actionable recommendations.

        Args:
            features: Property features dict
            gaps: Identified experience gaps
            opportunities: Identified opportunities

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Address top gaps
        for gap in gaps[:2]:
            # Extract trend name from gap string
            trend_name = gap.split(" (")[0] if " (" in gap else gap
            recommendations.append(
                f"Consider adding {trend_name.lower()} offerings to capture emerging demand"
            )

        # Leverage opportunities
        for opp in opportunities[:2]:
            if "position" in opp.lower():
                recommendations.append(
                    f"Marketing focus: {opp}"
                )

        # Property-specific suggestions
        themes = features.get("themes", [])
        amenities = features.get("amenities", [])

        if "wellness" in themes and "yoga" not in " ".join(amenities).lower():
            recommendations.append("Add yoga/meditation programs to complement wellness positioning")

        if "business" in themes:
            recommendations.append("Enhance digital nomad facilities (fast wifi, co-working spaces)")

        if features.get("price_segment") in ["luxury", "ultra_luxury"]:
            recommendations.append("Develop signature experiences unique to your property")

        if not features.get("tagline"):
            recommendations.append("Develop a memorable tagline to strengthen brand recall")

        return recommendations[:5]

    def save_property(self, property_data: dict) -> str:
        """Save property features to database.

        Args:
            property_data: Property dict from scan_property

        Returns:
            Property ID
        """
        db = SessionLocal()
        try:
            # Check for existing
            existing = db.query(PropertyFeaturesModel).filter(
                PropertyFeaturesModel.url == property_data["url"]
            ).first()

            if existing:
                # Update existing
                for key, value in property_data.items():
                    if key not in ["id", "scraped_at"]:
                        setattr(existing, key, value)
                existing.scraped_at = datetime.utcnow()
                db.commit()
                logger.info(f"Updated property: {existing.id}")
                return existing.id

            # Create new
            db_property = PropertyFeaturesModel(
                url=property_data["url"],
                name=property_data.get("name"),
                property_type=property_data.get("property_type", "hotel"),
                brand_positioning=property_data.get("brand_positioning"),
                tagline=property_data.get("tagline"),
                tone=property_data.get("tone"),
                themes=property_data.get("themes", []),
                amenities=property_data.get("amenities", []),
                room_types=property_data.get("room_types", []),
                dining_options=property_data.get("dining_options", []),
                experiences=property_data.get("experiences", []),
                location=property_data.get("location"),
                region=property_data.get("region"),
                price_segment=property_data.get("price_segment", "unknown"),
                price_indicators=property_data.get("price_indicators", []),
                demand_fit_score=property_data.get("demand_fit_score"),
                experience_gaps=property_data.get("experience_gaps", []),
                opportunity_lanes=property_data.get("opportunity_lanes", []),
                competitive_advantages=property_data.get("competitive_advantages", []),
                recommendations=property_data.get("recommendations", []),
                matching_trend_ids=property_data.get("matching_trend_ids", []),
                source_content_id=property_data.get("source_content_id"),
            )
            db.add(db_property)
            db.commit()
            db.refresh(db_property)

            logger.info(f"Saved property: {db_property.id}")
            return db_property.id

        except Exception as e:
            db.rollback()
            logger.error(f"Error saving property: {e}")
            raise

        finally:
            db.close()

    def get_property(self, property_id: str) -> dict | None:
        """Get property by ID.

        Args:
            property_id: Property ID

        Returns:
            Property dict or None
        """
        db = SessionLocal()
        try:
            prop = db.query(PropertyFeaturesModel).filter(
                PropertyFeaturesModel.id == property_id
            ).first()

            return self._model_to_dict(prop) if prop else None

        finally:
            db.close()

    def get_property_by_url(self, url: str) -> dict | None:
        """Get property by URL.

        Args:
            url: Property URL

        Returns:
            Property dict or None
        """
        db = SessionLocal()
        try:
            prop = db.query(PropertyFeaturesModel).filter(
                PropertyFeaturesModel.url == url
            ).first()

            return self._model_to_dict(prop) if prop else None

        finally:
            db.close()

    def get_properties(
        self,
        limit: int = 20,
        region: str | None = None,
        property_type: str | None = None,
        min_demand_fit: float = 0,
    ) -> list[dict]:
        """Get properties with filters.

        Args:
            limit: Maximum to return
            region: Filter by region
            property_type: Filter by property type
            min_demand_fit: Minimum demand fit score

        Returns:
            List of property dicts
        """
        db = SessionLocal()
        try:
            query = db.query(PropertyFeaturesModel)

            if region:
                query = query.filter(PropertyFeaturesModel.region == region)
            if property_type:
                query = query.filter(PropertyFeaturesModel.property_type == property_type)
            if min_demand_fit > 0:
                query = query.filter(
                    PropertyFeaturesModel.demand_fit_score >= min_demand_fit
                )

            properties = query.order_by(
                PropertyFeaturesModel.scraped_at.desc()
            ).limit(limit).all()

            return [self._model_to_dict(p) for p in properties]

        finally:
            db.close()

    def _model_to_dict(self, model: PropertyFeaturesModel) -> dict:
        """Convert PropertyFeaturesModel to dict."""
        return {
            "id": model.id,
            "url": model.url,
            "name": model.name,
            "property_type": model.property_type,
            "brand_positioning": model.brand_positioning,
            "tagline": model.tagline,
            "tone": model.tone,
            "themes": model.themes or [],
            "amenities": model.amenities or [],
            "room_types": model.room_types or [],
            "dining_options": model.dining_options or [],
            "experiences": model.experiences or [],
            "location": model.location,
            "region": model.region,
            "price_segment": model.price_segment,
            "price_indicators": model.price_indicators or [],
            "demand_fit_score": model.demand_fit_score,
            "experience_gaps": model.experience_gaps or [],
            "opportunity_lanes": model.opportunity_lanes or [],
            "competitive_advantages": model.competitive_advantages or [],
            "recommendations": model.recommendations or [],
            "matching_trend_ids": model.matching_trend_ids or [],
            "scraped_at": model.scraped_at.isoformat() if model.scraped_at else None,
        }


def scan_property_url(url: str, save: bool = True) -> dict | None:
    """Convenience function to scan a property URL.

    Args:
        url: Property website URL
        save: Whether to save to database

    Returns:
        Property analysis dict or None
    """
    service = DemandScanService()
    result = service.scan_property(url)

    if result and save:
        property_id = service.save_property(result)
        result["id"] = property_id

    return result
