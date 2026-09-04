"""Demand Scan service - Property analysis and trend matching."""

import json
import logging
import math
from datetime import datetime
from typing import Any
from urllib.parse import urlparse, urlunparse

from db.database import SessionLocal
from db.models import PropertyFeaturesModel, TrendSignalModel
from data_models.property_features import PropertyFeatures, PropertyType, PriceSegment
from ingestion.properties.property_scraper import PropertyScraper
from data_models.embeddings import get_embedding_provider
from processing.trend_names import normalize_trend_title
from processing.property_analysis import (
    extract_property_features,
    detect_region,
    extract_price_indicators,
)

logger = logging.getLogger(__name__)


def normalize_url(url: str) -> str:
    """Normalize a URL by removing query parameters and fragments.

    This prevents duplicate entries when the same property is scanned
    with different tracking params (utm_source, gclid, etc.).

    Args:
        url: The URL to normalize

    Returns:
        Normalized URL without query params or fragments
    """
    parsed = urlparse(url)
    # Reconstruct URL without query string or fragment
    normalized = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path.rstrip('/'),  # Also normalize trailing slashes
        '',  # params
        '',  # query
        '',  # fragment
    ))
    return normalized


# Trend vectors are stable for the life of a trend row, so embed each once per
# process. Keyed by trend id; cleared implicitly on restart.
_TREND_VECTORS: dict[str, list[float]] = {}

# Cosine similarity between a property profile and a trend, on this corpus with
# mistral-embed, runs roughly 0.66 (unrelated) to 0.84 (same theme). The fit score
# maps the top-5 mean onto 0-1 across that band; ALIGNED/UNALIGNED cut the list
# into "already speaks to this" and "does not speak to this".
_SIM_FLOOR = 0.70
_SIM_SPAN = 0.12
_ALIGNED = 0.77
_UNALIGNED = 0.745


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _property_profile_text(features: dict) -> str:
    """One paragraph that describes the property the way a trend is described."""
    parts = [
        features.get("name") or "",
        features.get("brand_positioning") or "",
        features.get("tagline") or "",
        f"Tone: {features['tone']}." if features.get("tone") else "",
        f"A {features.get('property_type', 'hotel')} in the {features.get('price_segment', 'unknown')} segment.",
        ("Themes: " + ", ".join(features.get("themes", [])) + ".") if features.get("themes") else "",
        ("Amenities: " + ", ".join(features.get("amenities", [])) + ".") if features.get("amenities") else "",
        ("Experiences: " + ", ".join(features.get("experiences", [])) + ".") if features.get("experiences") else "",
        ("Dining: " + ", ".join(features.get("dining_options", [])) + ".") if features.get("dining_options") else "",
        f"Location: {features['location']}." if features.get("location") else "",
    ]
    return " ".join(part for part in parts if part).strip()


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
        # Normalize URL to prevent duplicates from tracking params
        normalized_url = normalize_url(url)
        logger.info(f"Scanning property: {normalized_url}")

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
        alignment = demand_fit.get("alignment") or []

        # Step 6: Identify gaps, opportunities, and misalignments
        gaps = self._identify_experience_gaps(features, trends, alignment)
        opportunities = self._identify_opportunities(features, trends, alignment)
        advantages = self._identify_competitive_advantages(features, trends, alignment)
        misalignment_flags = self._identify_positioning_misalignment(features)
        recommendations = self._generate_recommendations(features, gaps, opportunities)

        # Step 7: One LLM pass that reads the evidence into an executive brief
        demand_brief = self._write_demand_brief(features, alignment, gaps, opportunities, demand_fit["score"])

        # Ensure we have a property name
        property_name = features.get("name")
        if not property_name:
            property_name = self._extract_name_from_url(url)

        # Build final result (use normalized URL for storage)
        result = {
            "url": normalized_url,
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
            "positioning_misalignment_flags": misalignment_flags,
            "recommendations": recommendations,
            "matching_trend_ids": demand_fit["matching_trend_ids"],
            "trend_alignment": alignment[:12],
            "demand_brief": demand_brief,
            "fit_method": demand_fit.get("method", "keywords"),
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
            ).limit(80).all()

            # Clustering re-discovers themes under quote/plural variants; keep the
            # strongest instance of each so a property is not told the same gap thrice.
            seen: set[str] = set()
            unique = []
            for t in trends:
                key = normalize_trend_title(t.name)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(t)
            trends = unique[:50]

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
        """Semantic demand fit: embed the property profile and every trend, score by
        cosine similarity. Falls back to keyword overlap if embeddings are unavailable.

        Returns:
            Dict with score, matching_trend_ids, alignment (per-trend similarity,
            strongest first) and method.
        """
        if not trends:
            return {"score": 0.5, "matching_trend_ids": [], "alignment": [], "method": "none"}

        profile = _property_profile_text(features)
        try:
            provider = get_embedding_provider()
            missing = [t for t in trends if t["id"] not in _TREND_VECTORS]
            if missing:
                vectors = provider.embed_batch(
                    [f"{t['name']}. {(t.get('description') or '')[:600]}" for t in missing]
                )
                for t, v in zip(missing, vectors):
                    _TREND_VECTORS[t["id"]] = v
            pv = provider.embed(profile)
        except Exception as exc:
            logger.warning(f"Embedding unavailable for demand fit, using keyword overlap: {exc}")
            result = self._compute_demand_fit_keywords(features, trends)
            result["alignment"] = []
            result["method"] = "keywords"
            return result

        alignment = []
        for t in trends:
            sim = _cosine(pv, _TREND_VECTORS[t["id"]])
            alignment.append(
                {
                    "trend_id": t["id"],
                    "name": t["name"],
                    "similarity": round(sim, 3),
                    "strength_score": round(t.get("strength_score") or 0, 3),
                    "white_space_score": round(t.get("white_space_score") or 0, 3),
                    "region": t.get("region"),
                }
            )
        alignment.sort(key=lambda a: a["similarity"], reverse=True)

        top = [a["similarity"] for a in alignment[:5]]
        top_mean = sum(top) / len(top)
        score = max(0.0, min(1.0, (top_mean - _SIM_FLOOR) / _SIM_SPAN))
        matching = [a["trend_id"] for a in alignment if a["similarity"] >= _ALIGNED]

        return {
            "score": round(score, 2),
            "matching_trend_ids": matching,
            "alignment": alignment,
            "method": "embedding",
        }

    def _compute_demand_fit_keywords(
        self,
        features: dict,
        trends: list[dict],
    ) -> dict:
        """Keyword-overlap fallback (the original heuristic).

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
        alignment: list[dict] | None = None,
    ) -> list[str]:
        """Identify trending experiences not offered by property.

        With semantic alignment available, a gap is a strong trend the property's
        own profile does not resonate with. Without it, fall back to keyword cover.

        Returns:
            List of gap descriptions (deduplicated)
        """
        if alignment:
            strong = [a for a in alignment if a["strength_score"] > 0.3 and a["similarity"] < _UNALIGNED]
            strong.sort(key=lambda a: (a["strength_score"] * (1 - a["similarity"])), reverse=True)
            return [
                f"{a['name']} (trending at {int(a['strength_score'] * 100)}% strength; "
                f"property alignment {int(a['similarity'] * 100)}%)"
                for a in strong[:5]
            ]

        gaps = []
        seen_trend_names = set()  # Track unique trend names to avoid duplicates

        property_offerings = set()
        for amenity in features.get("amenities", []):
            property_offerings.add(amenity.lower())
        for exp in features.get("experiences", []):
            property_offerings.add(exp.lower())
        for theme in features.get("themes", []):
            property_offerings.add(theme.lower())

        # Check high-strength trends, deduplicate by name
        strong_trends = [t for t in trends if t["strength_score"] > 0.3]

        for trend in strong_trends[:10]:
            trend_name = trend["name"]
            trend_name_lower = trend_name.lower()

            # Skip if we've already seen this trend name
            if trend_name_lower in seen_trend_names:
                continue

            trend_topics = [t.lower() for t in trend.get("topics", [])]

            # Check if property covers this trend
            covered = any(
                topic in " ".join(property_offerings)
                for topic in trend_topics
            ) or any(
                offering in trend_name_lower
                for offering in property_offerings
            )

            if not covered:
                strength_pct = int(trend["strength_score"] * 100)
                gaps.append(
                    f"{trend_name} (trending at {strength_pct}% strength)"
                )
                seen_trend_names.add(trend_name_lower)

        return gaps[:5]

    def _identify_opportunities(
        self,
        features: dict,
        trends: list[dict],
        alignment: list[dict] | None = None,
    ) -> list[str]:
        """Identify positioning opportunities based on trends.

        The best lanes are white-space trends one step from the property's current
        positioning: close enough to be credible, far enough to be new.

        Returns:
            List of opportunity descriptions (deduplicated)
        """
        opportunities = []
        seen_trend_names = set()  # Track unique trend names to avoid duplicates

        if alignment:
            adjacent = [
                a for a in alignment
                if a["white_space_score"] > 0.3 and _UNALIGNED <= a["similarity"] < 0.82
            ]
            adjacent.sort(key=lambda a: a["white_space_score"] * a["similarity"], reverse=True)
            for a in adjacent[:3]:
                opportunities.append(
                    f"Adjacent white space: '{a['name']}' - {int(a['white_space_score'] * 100)}% unmet demand, "
                    f"{int(a['similarity'] * 100)}% aligned with current positioning"
                )
                seen_trend_names.add(a["name"].lower())

        # Look for high white-space trends
        whitespace_trends = sorted(
            trends,
            key=lambda t: t.get("white_space_score", 0),
            reverse=True,
        )[:10]  # Check more trends but deduplicate

        for trend in whitespace_trends:
            trend_name = trend["name"]
            trend_name_lower = trend_name.lower()

            # Skip if we've already seen this trend name
            if trend_name_lower in seen_trend_names:
                continue

            if trend.get("white_space_score", 0) > 0.3:
                opportunities.append(
                    f"Position as leader in '{trend_name}' - high demand, low competition"
                )
                seen_trend_names.add(trend_name_lower)

        # Suggest based on property type (these are unique by nature)
        property_type = features.get("property_type", "hotel")
        themes = [t.lower() for t in features.get("themes", [])]

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
        alignment: list[dict] | None = None,
    ) -> list[str]:
        """Identify property's competitive advantages.

        Returns:
            List of advantage descriptions
        """
        advantages = []

        # Demand the property already speaks to, strongest resonance first
        for a in (alignment or [])[:3]:
            if a["similarity"] >= _ALIGNED:
                advantages.append(
                    f"Already speaks to '{a['name']}' ({int(a['similarity'] * 100)}% alignment, "
                    f"{int(a['strength_score'] * 100)}% demand strength)"
                )

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

    def _identify_positioning_misalignment(
        self,
        features: dict,
    ) -> list[str]:
        """Identify positioning misalignments and inconsistencies.

        Detects when property claims don't match offerings, such as
        luxury positioning with budget amenities.

        Args:
            features: Property features dict

        Returns:
            List of misalignment flag descriptions
        """
        flags = []

        price = features.get("price_segment", "unknown")
        amenities = [a.lower() for a in features.get("amenities", [])]
        themes = [t.lower() for t in features.get("themes", [])]
        positioning = (features.get("brand_positioning") or "").lower()
        experiences = [e.lower() for e in features.get("experiences", [])]
        dining = [d.lower() for d in features.get("dining_options", [])]
        room_types = [r.lower() for r in features.get("room_types", [])]

        # Build comprehensive text from all property data for searching
        all_offerings_text = " ".join(amenities + experiences + dining + room_types + [positioning])

        # 1. Luxury pricing without luxury amenities
        luxury_indicators = {"spa", "fine dining", "butler", "concierge", "valet", "pool", "suite", "premium"}
        has_luxury_amenities = any(li in all_offerings_text for li in luxury_indicators)

        if price in ["luxury", "ultra_luxury"] and not has_luxury_amenities:
            flags.append("Price-tier mismatch: Luxury pricing without premium amenities (spa, concierge, fine dining)")

        # 2. Wellness positioning without wellness offerings
        # Check ALL sources: amenities, experiences, dining, positioning, room types
        wellness_indicators = {"spa", "yoga", "meditation", "massage", "wellness", "sauna", "steam", "fitness", "gym", "health"}
        has_wellness = any(wi in all_offerings_text for wi in wellness_indicators)
        wellness_positioned = "wellness" in themes or "wellness" in positioning or "mindfulness" in positioning

        if wellness_positioned and not has_wellness:
            flags.append("Theme mismatch: Wellness positioning without spa or wellness facilities")

        # 3. Boutique positioning without character elements
        boutique_indicators = {"design", "art", "unique", "curated", "bespoke", "artisan", "character", "historic", "heritage"}
        has_boutique_elements = any(bi in all_offerings_text for bi in boutique_indicators)

        if "boutique" in themes and not has_boutique_elements:
            flags.append("Theme mismatch: Boutique positioning without distinctive design or character elements")

        # 4. Conflicting themes
        conflicting_pairs = [
            ("budget", "luxury"),
            ("budget", "ultra_luxury"),
            ("business", "romantic"),
            ("family", "adults-only"),
        ]
        for t1, t2 in conflicting_pairs:
            if t1 in themes and t2 in themes:
                flags.append(f"Conflicting positioning: {t1.title()} and {t2.title()} themes mixed")

        # 5. Eco/sustainable positioning without evidence
        eco_indicators = {"solar", "recycl", "sustain", "organic", "eco", "green", "carbon", "environment"}
        has_eco_evidence = any(ei in all_offerings_text for ei in eco_indicators)
        eco_positioned = "eco" in themes or "sustainable" in themes or "green" in themes

        if eco_positioned and not has_eco_evidence:
            flags.append("Theme mismatch: Eco/sustainable positioning without visible sustainability practices")

        return flags[:5]

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

    def _write_demand_brief(
        self,
        features: dict,
        alignment: list[dict],
        gaps: list[str],
        opportunities: list[str],
        score: float,
    ) -> dict | None:
        """One LLM pass turning the scored evidence into an executive brief.

        The model only sees what the scan measured; it is asked to interpret, not
        to invent. Returns None when the LLM is disabled or fails.
        """
        if not self.use_llm:
            return None
        try:
            from processing.llm_utils import MistralLLM

            llm = MistralLLM()
        except Exception as exc:
            logger.warning(f"Demand brief skipped, LLM unavailable: {exc}")
            return None

        aligned = [a for a in alignment if a["similarity"] >= _ALIGNED][:4]
        nl = "\n"
        aligned_lines = nl.join(
            f"- {a['name']} (alignment {int(a['similarity'] * 100)}%, demand strength {int(a['strength_score'] * 100)}%)"
            for a in aligned
        ) or "- none above threshold"
        gap_lines = nl.join(f"- {g}" for g in gaps[:4]) or "- none"
        opp_lines = nl.join(f"- {o}" for o in opportunities[:3]) or "- none"

        system_prompt = (
            "You are a hospitality strategist writing for a hotel owner. Use only the evidence given. "
            "Be concrete and specific to this property; never generic. Respond with valid JSON only."
        )
        prompt = (
            "Property profile:" + nl + _property_profile_text(features) + nl + nl
            + f"Demand fit score: {int(score * 100)}/100 (semantic match between the property and current demand signals)." + nl + nl
            + "Demand the property already resonates with:" + nl + aligned_lines + nl + nl
            + "Strong demand the property does not speak to:" + nl + gap_lines + nl + nl
            + "Adjacent white space:" + nl + opp_lines + nl + nl
            + "Write JSON with exactly these keys:" + nl
            + '{"headline": "one sentence, max 18 words, the single most important thing the owner should know", '
            + '"read": "2-3 sentences interpreting the evidence for this property specifically", '
            + '"moves": ["three concrete moves, each one sentence, each tied to a named trend above"]}' + nl
            + "JSON:"
        )
        try:
            raw = llm.generate(prompt, system_prompt, max_tokens=500, temperature=0.4, json_mode=True).strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
            moves = data.get("moves") or []
            return {
                "headline": str(data.get("headline") or "").strip(),
                "read": str(data.get("read") or "").strip(),
                "moves": [str(m).strip() for m in moves][:3],
                "model": llm.model,
                "generated_at": datetime.utcnow().isoformat(),
            }
        except Exception as exc:
            logger.warning(f"Demand brief generation failed: {exc}")
            return None

    def save_property(self, property_data: dict) -> str:
        """Save property features to database.

        Args:
            property_data: Property dict from scan_property

        Returns:
            Property ID
        """
        # Normalize URL before saving/checking
        normalized_url = normalize_url(property_data["url"])
        property_data["url"] = normalized_url

        db = SessionLocal()
        try:
            # Check for existing by normalized URL
            existing = db.query(PropertyFeaturesModel).filter(
                PropertyFeaturesModel.url == normalized_url
            ).first()

            if existing:
                # Update existing
                columns = set(PropertyFeaturesModel.__table__.columns.keys())
                for key, value in property_data.items():
                    if key in columns and key not in ["id", "scraped_at"]:
                        setattr(existing, key, value)
                existing.metadata_json = self._analysis_metadata(property_data)
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
                positioning_misalignment_flags=property_data.get("positioning_misalignment_flags", []),
                recommendations=property_data.get("recommendations", []),
                matching_trend_ids=property_data.get("matching_trend_ids", []),
                source_content_id=property_data.get("source_content_id"),
                metadata_json=self._analysis_metadata(property_data),
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
        normalized_url = normalize_url(url)
        db = SessionLocal()
        try:
            prop = db.query(PropertyFeaturesModel).filter(
                PropertyFeaturesModel.url == normalized_url
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

    @staticmethod
    def _analysis_metadata(property_data: dict) -> dict:
        """Derived analysis that has no column of its own lives in metadata_json."""
        return {
            "trend_alignment": property_data.get("trend_alignment") or [],
            "demand_brief": property_data.get("demand_brief"),
            "fit_method": property_data.get("fit_method", "keywords"),
        }

    def _model_to_dict(self, model: PropertyFeaturesModel) -> dict:
        """Convert PropertyFeaturesModel to dict."""
        meta = model.metadata_json or {}
        return {
            "trend_alignment": meta.get("trend_alignment") or [],
            "demand_brief": meta.get("demand_brief"),
            "fit_method": meta.get("fit_method", "keywords"),
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
            "positioning_misalignment_flags": model.positioning_misalignment_flags or [],
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
