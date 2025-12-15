"""Property analysis module for Demand Scan feature extraction."""

import json
import logging
import re

from data_models.property_features import PropertyType, PriceSegment
from processing.llm_utils import get_llm

logger = logging.getLogger(__name__)

# Keyword lists for fallback detection
AMENITY_KEYWORDS = [
    "spa", "pool", "gym", "fitness", "restaurant", "bar", "wifi", "parking",
    "room service", "concierge", "laundry", "business center", "meeting room",
    "beach", "garden", "terrace", "balcony", "kitchen", "minibar", "safe",
    "air conditioning", "heating", "tv", "coffee", "breakfast", "lounge",
    "rooftop", "sauna", "jacuzzi", "massage", "yoga", "co-working", "coworking",
    "pet friendly", "kids club", "babysitting", "shuttle", "airport transfer",
    "valet", "doorman", "butler", "private beach", "infinity pool",
]

THEME_KEYWORDS = {
    "wellness": ["spa", "wellness", "health", "yoga", "meditation", "retreat", "detox", "mindful"],
    "adventure": ["adventure", "hiking", "outdoor", "safari", "diving", "surfing", "extreme"],
    "luxury": ["luxury", "premium", "exclusive", "vip", "five star", "5 star", "opulent"],
    "boutique": ["boutique", "design", "artisan", "unique", "curated", "intimate"],
    "eco": ["eco", "sustainable", "green", "organic", "nature", "conservation", "solar"],
    "family": ["family", "kids", "children", "playground", "babysitting", "family-friendly"],
    "romantic": ["romantic", "honeymoon", "couples", "intimate", "secluded", "private"],
    "business": ["business", "corporate", "meeting", "conference", "work", "executive"],
    "cultural": ["culture", "heritage", "historic", "local", "authentic", "traditional"],
    "beach": ["beach", "seaside", "oceanfront", "coastal", "waterfront", "island"],
    "urban": ["city", "urban", "downtown", "metropolitan", "central"],
    "modern": ["modern", "contemporary", "minimalist", "sleek", "innovative"],
}

PRICE_INDICATORS = {
    "budget": ["budget", "affordable", "cheap", "value", "hostel", "backpacker", "$"],
    "midscale": ["mid-range", "comfortable", "standard", "$$"],
    "upscale": ["upscale", "premium", "superior", "$$$"],
    "luxury": ["luxury", "five star", "5 star", "exclusive", "$$$$", "high-end"],
    "ultra_luxury": ["ultra luxury", "palatial", "world-class", "legendary", "$$$$$"],
}

PROPERTY_TYPE_KEYWORDS = {
    "resort": ["resort", "all-inclusive", "beachfront resort"],
    "boutique": ["boutique hotel", "boutique", "design hotel"],
    "hostel": ["hostel", "backpacker", "dormitory"],
    "vacation_rental": ["vacation rental", "villa", "apartment", "airbnb"],
    "bed_and_breakfast": ["bed and breakfast", "b&b", "guesthouse", "inn"],
    "motel": ["motel", "motor lodge"],
}


def extract_property_features(content: str, url: str) -> dict:
    """Extract all property features using LLM.

    Args:
        content: Scraped website content
        url: Property URL

    Returns:
        Dict with extracted features
    """
    llm = get_llm()

    # Truncate content if too long
    if len(content) > 4000:
        content = content[:4000] + "..."

    system_prompt = """You are a hospitality analyst extracting structured information from hotel websites.
Extract the requested information in JSON format. Be specific and accurate.
If information is not clearly present, use null.
You MUST respond with valid JSON only, no other text."""

    property_types = ", ".join([pt.value for pt in PropertyType])
    price_segments = ", ".join([ps.value for ps in PriceSegment])

    prompt = f"""Analyze this hotel website content and extract:

{content}

Extract the following in JSON format:
{{
    "name": "Hotel name or null",
    "property_type": one of [{property_types}],
    "brand_positioning": "1-2 sentence summary of how the hotel positions itself, or null",
    "tagline": "Hotel tagline/slogan if found, or null",
    "tone": "Brand voice description (e.g., 'warm and welcoming', 'sophisticated and exclusive'), or null",
    "themes": ["theme1", "theme2", ...] (e.g., wellness, adventure, luxury, eco, family, romantic, business),
    "amenities": ["amenity1", "amenity2", ...] (specific amenities mentioned),
    "room_types": ["room type 1", "room type 2", ...] (room categories if mentioned),
    "dining_options": ["restaurant/bar 1", ...] (F&B options if mentioned),
    "experiences": ["experience1", ...] (activities/experiences offered),
    "location": "Location description or address if found",
    "price_segment": one of [{price_segments}] (based on language and positioning)
}}

JSON response:"""

    try:
        response = llm.generate(prompt, system_prompt, max_tokens=800, temperature=0.3)

        # Clean response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        data = json.loads(response)

        # Validate enums
        if data.get("property_type") not in [pt.value for pt in PropertyType]:
            data["property_type"] = "hotel"
        if data.get("price_segment") not in [ps.value for ps in PriceSegment]:
            data["price_segment"] = "unknown"

        # Ensure lists
        for field in ["themes", "amenities", "room_types", "dining_options", "experiences"]:
            if not isinstance(data.get(field), list):
                data[field] = []

        return data

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        return fallback_feature_extraction(content, url)
    except Exception as e:
        logger.error(f"Error in LLM extraction: {e}")
        return fallback_feature_extraction(content, url)


def fallback_feature_extraction(content: str, url: str) -> dict:
    """Fallback keyword-based feature extraction.

    Args:
        content: Website content
        url: Property URL

    Returns:
        Dict with extracted features
    """
    content_lower = content.lower()

    # Extract amenities
    amenities = [a for a in AMENITY_KEYWORDS if a in content_lower]

    # Extract themes
    themes = []
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in content_lower for kw in keywords):
            themes.append(theme)

    # Determine price segment
    price_segment = "unknown"
    for segment, indicators in PRICE_INDICATORS.items():
        if any(ind in content_lower for ind in indicators):
            price_segment = segment
            break

    # Determine property type
    property_type = "hotel"
    for ptype, keywords in PROPERTY_TYPE_KEYWORDS.items():
        if any(kw in content_lower for kw in keywords):
            property_type = ptype
            break

    # Try to extract name from first line
    lines = content.split("\n")
    name = lines[0][:100] if lines else None

    return {
        "name": name,
        "property_type": property_type,
        "brand_positioning": None,
        "tagline": None,
        "tone": None,
        "themes": themes[:5],
        "amenities": amenities[:15],
        "room_types": [],
        "dining_options": [],
        "experiences": [],
        "location": None,
        "price_segment": price_segment,
    }


def detect_region(content: str, location: str | None = None) -> str | None:
    """Detect geographic region from content.

    Args:
        content: Website content
        location: Extracted location string

    Returns:
        Region string or None
    """
    text = f"{content} {location or ''}".lower()

    region_keywords = {
        "europe": ["europe", "paris", "london", "barcelona", "lisbon", "amsterdam", "rome", "berlin", "madrid", "vienna", "prague", "portugal", "spain", "france", "italy", "germany", "uk", "england"],
        "asia": ["asia", "tokyo", "bangkok", "bali", "singapore", "hong kong", "seoul", "vietnam", "thailand", "japan", "china", "india", "indonesia", "philippines", "malaysia"],
        "north_america": ["usa", "united states", "new york", "los angeles", "miami", "san francisco", "canada", "mexico", "california", "florida", "texas", "chicago"],
        "south_america": ["brazil", "argentina", "colombia", "peru", "chile", "rio", "buenos aires", "south america"],
        "middle_east": ["dubai", "abu dhabi", "qatar", "saudi", "uae", "israel", "jordan", "oman", "middle east"],
        "oceania": ["australia", "sydney", "melbourne", "new zealand", "fiji", "oceania"],
        "caribbean": ["caribbean", "bahamas", "jamaica", "aruba", "turks", "caicos", "st lucia", "barbados"],
        "africa": ["africa", "morocco", "south africa", "kenya", "tanzania", "egypt", "cape town"],
    }

    region_scores = {}
    for region, keywords in region_keywords.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            region_scores[region] = score

    if region_scores:
        return max(region_scores, key=region_scores.get)
    return None


def extract_price_indicators(content: str) -> list[str]:
    """Extract price-related text from content.

    Args:
        content: Website content

    Returns:
        List of price indicator strings
    """
    indicators = []
    content_lower = content.lower()

    # Look for price mentions
    price_patterns = [
        r'\$[\d,]+',
        r'€[\d,]+',
        r'£[\d,]+',
        r'from \$[\d,]+',
        r'starting at \$[\d,]+',
        r'rates from',
        r'per night',
        r'nightly rate',
    ]

    for pattern in price_patterns:
        matches = re.findall(pattern, content_lower)
        indicators.extend(matches[:3])

    # Look for price segment language
    for segment, keywords in PRICE_INDICATORS.items():
        for kw in keywords:
            if kw in content_lower:
                indicators.append(f"{segment}: {kw}")
                break

    return list(set(indicators))[:5]
