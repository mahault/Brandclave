"""City Desires data model for BrandClave.

Captures what travelers want in a city but can't easily find -
the "unmet desire" layer that identifies white space opportunities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class DesireCategory(str, Enum):
    """Categories of traveler desires."""
    ACCOMMODATION = "accommodation"
    EXPERIENCE = "experience"
    AMENITY = "amenity"
    LOCATION = "location"
    SERVICE = "service"
    VIBE = "vibe"
    VALUE = "value"
    SAFETY = "safety"


class TravelerSegment(str, Enum):
    """Traveler segments expressing desires."""
    SOLO = "solo_traveler"
    COUPLES = "couples"
    FAMILIES = "families"
    BUSINESS = "business"
    DIGITAL_NOMAD = "digital_nomad"
    LUXURY = "luxury_seeker"
    BUDGET = "budget_conscious"
    ADVENTURE = "adventure_seeker"
    WELLNESS = "wellness_focused"
    PARTY = "party_traveler"
    CULTURE = "culture_explorer"
    FOODIE = "foodie"


class SentimentType(str, Enum):
    """Sentiment of the desire expression."""
    FRUSTRATION = "frustration"  # "I wish there was...", "Can't find..."
    DESIRE = "desire"  # "Would love to find...", "Looking for..."
    COMPLAINT = "complaint"  # "The problem is...", "Hotels here don't..."
    QUESTION = "question"  # "Does anyone know...", "Is there a..."
    DELIGHT = "delight"  # Positive - what IS working (for comparison)


@dataclass
class DesireSignal:
    """A single signal of traveler desire from content."""

    text: str  # The actual quote/snippet
    source: str  # reddit, youtube, tripadvisor, etc.
    source_url: Optional[str] = None
    city: str = ""
    country: str = ""
    category: DesireCategory = DesireCategory.ACCOMMODATION
    sentiment: SentimentType = SentimentType.DESIRE
    segments: list[TravelerSegment] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    extracted_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "source_url": self.source_url,
            "city": self.city,
            "country": self.country,
            "category": self.category.value if self.category else None,
            "sentiment": self.sentiment.value if self.sentiment else None,
            "segments": [s.value for s in self.segments],
            "keywords": self.keywords,
            "extracted_at": self.extracted_at.isoformat() if self.extracted_at else None,
        }


@dataclass
class DesireTheme:
    """A clustered theme of desires for a city."""

    theme_name: str  # e.g., "Design-forward affordable stays"
    description: str  # What this theme represents
    city: str
    country: str
    intensity_score: float  # 0-1, how strong is this desire
    frustration_score: float  # 0-1, how frustrated are people
    frequency: int  # How many times mentioned
    category: DesireCategory = DesireCategory.ACCOMMODATION
    segments: list[TravelerSegment] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    example_snippets: list[str] = field(default_factory=list)
    supply_gap: float = 0.0  # 0-1, how underserved is this (desire vs supply)
    opportunity_score: float = 0.0  # Combined score for white space

    def to_dict(self) -> dict:
        return {
            "theme_name": self.theme_name,
            "description": self.description,
            "city": self.city,
            "country": self.country,
            "intensity_score": self.intensity_score,
            "frustration_score": self.frustration_score,
            "frequency": self.frequency,
            "category": self.category.value if self.category else None,
            "segments": [s.value for s in self.segments],
            "keywords": self.keywords,
            "example_snippets": self.example_snippets[:5],  # Limit to 5
            "supply_gap": self.supply_gap,
            "opportunity_score": self.opportunity_score,
        }


@dataclass
class CityDesireProfile:
    """Complete desire profile for a city."""

    city: str
    country: str
    region: Optional[str] = None

    # Aggregated metrics
    total_signals: int = 0
    total_sources: int = 0
    avg_frustration: float = 0.0

    # Top desires and themes
    top_desires: list[DesireTheme] = field(default_factory=list)
    underserved_segments: list[str] = field(default_factory=list)
    white_space_opportunities: list[str] = field(default_factory=list)

    # Concept recommendations
    concept_lanes: list[dict] = field(default_factory=list)

    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    data_freshness_days: int = 30

    def to_dict(self) -> dict:
        return {
            "city": self.city,
            "country": self.country,
            "region": self.region,
            "total_signals": self.total_signals,
            "total_sources": self.total_sources,
            "avg_frustration": self.avg_frustration,
            "top_desires": [d.to_dict() for d in self.top_desires],
            "underserved_segments": self.underserved_segments,
            "white_space_opportunities": self.white_space_opportunities,
            "concept_lanes": self.concept_lanes,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "data_freshness_days": self.data_freshness_days,
        }


# Desire detection patterns
DESIRE_PATTERNS = {
    "frustration": [
        r"i wish there was",
        r"i wish there were",
        r"can'?t find",
        r"couldn'?t find",
        r"hard to find",
        r"difficult to find",
        r"there aren'?t any",
        r"there'?s no",
        r"nowhere to",
        r"no good",
        r"lack of",
        r"missing",
        r"the problem with",
        r"frustrating",
        r"disappointed",
        r"why don'?t",
        r"why isn'?t there",
    ],
    "desire": [
        r"looking for",
        r"searching for",
        r"trying to find",
        r"would love",
        r"would be great",
        r"need a",
        r"want a",
        r"hoping to find",
        r"ideally",
        r"dream hotel",
        r"perfect hotel would",
        r"best place to stay",
        r"recommend.*hotel",
        r"any suggestions",
        r"does anyone know",
    ],
    "complaint": [
        r"overpriced",
        r"too expensive",
        r"not worth",
        r"poor quality",
        r"terrible",
        r"awful",
        r"avoid",
        r"don'?t stay",
        r"worst",
        r"scam",
        r"rip.?off",
        r"disappointing",
    ],
}

# Segment detection keywords
SEGMENT_KEYWORDS = {
    TravelerSegment.SOLO: ["solo", "alone", "by myself", "single traveler"],
    TravelerSegment.COUPLES: ["couple", "romantic", "honeymoon", "anniversary", "partner", "girlfriend", "boyfriend", "wife", "husband"],
    TravelerSegment.FAMILIES: ["family", "kids", "children", "toddler", "baby", "family-friendly"],
    TravelerSegment.BUSINESS: ["business", "work trip", "conference", "meeting", "corporate"],
    TravelerSegment.DIGITAL_NOMAD: ["digital nomad", "remote work", "work from", "coworking", "long stay", "monthly", "workation"],
    TravelerSegment.LUXURY: ["luxury", "5 star", "five star", "high end", "upscale", "premium", "exclusive"],
    TravelerSegment.BUDGET: ["budget", "cheap", "affordable", "backpacker", "hostel", "low cost", "value"],
    TravelerSegment.ADVENTURE: ["adventure", "hiking", "outdoor", "active", "sports", "surfing", "diving"],
    TravelerSegment.WELLNESS: ["wellness", "spa", "yoga", "meditation", "retreat", "detox", "healthy"],
    TravelerSegment.PARTY: ["party", "nightlife", "club", "bar", "drinking", "pub crawl"],
    TravelerSegment.CULTURE: ["culture", "museum", "art", "history", "local", "authentic", "traditional"],
    TravelerSegment.FOODIE: ["food", "restaurant", "cuisine", "culinary", "michelin", "gastronomy", "eating"],
}

# Category detection keywords
CATEGORY_KEYWORDS = {
    DesireCategory.ACCOMMODATION: ["hotel", "hostel", "airbnb", "apartment", "stay", "room", "bed", "accommodation"],
    DesireCategory.EXPERIENCE: ["experience", "activity", "tour", "excursion", "thing to do", "attraction"],
    DesireCategory.AMENITY: ["pool", "gym", "wifi", "breakfast", "parking", "ac", "air conditioning", "balcony", "view"],
    DesireCategory.LOCATION: ["location", "neighborhood", "area", "district", "near", "close to", "walkable", "central"],
    DesireCategory.SERVICE: ["service", "staff", "concierge", "reception", "check-in", "helpful"],
    DesireCategory.VIBE: ["vibe", "atmosphere", "aesthetic", "design", "style", "boutique", "cozy", "modern", "hip"],
    DesireCategory.VALUE: ["value", "price", "cost", "worth", "expensive", "cheap", "affordable", "budget"],
    DesireCategory.SAFETY: ["safe", "security", "dangerous", "sketchy", "secure", "female solo"],
}
