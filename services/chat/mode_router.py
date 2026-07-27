"""Mode Router - Bayesian intent inference for chat mode selection."""

import json
import logging
import re
from typing import Any

from services.chat.schemas import ChatMode, RouterOutput, SlotValues

logger = logging.getLogger(__name__)


# Priors for mode selection based on signals
MODE_PRIORS = {
    ChatMode.INSIGHT: 0.5,      # Default mode
    ChatMode.BRAND_BUILD: 0.3,
    ChatMode.DEMAND_SCAN: 0.2,
}

# Keywords that shift probabilities
INSIGHT_KEYWORDS = [
    "trend", "trends", "what's hot", "market", "opportunity", "opportunities",
    "demand", "travelers want", "guests want", "emerging", "growing",
    "popular", "forecast", "prediction", "analysis", "insight",
    "white space", "gap", "unmet", "desire", "need", "sentiment",
]

BRAND_BUILD_KEYWORDS = [
    "build", "create", "design", "concept", "brand", "hotel concept",
    "new hotel", "develop", "blueprint", "positioning", "name",
    "experience", "pillars", "thesis", "boutique", "lifestyle",
    "help me build", "want to build", "building a", "create a hotel",
    "new brand", "brand concept", "hotel brand", "develop a",
]

DEMAND_SCAN_KEYWORDS = [
    "scan", "analyze", "property", "hotel", "website", "url",
    "review", "audit", "evaluate", "assess", "check",
    ".com", "https://", "http://", "www.",
]

# Slot extraction patterns
URL_PATTERN = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+')
ADR_PATTERN = re.compile(r'\$?\d{2,4}(?:\s*(?:per night|/night|adr|average))?', re.IGNORECASE)
LOCATION_KEYWORDS = [
    # North America
    "new york", "nyc", "miami", "los angeles", "la", "san francisco", "sf",
    "chicago", "boston", "seattle", "austin", "denver", "nashville", "atlanta",
    "washington dc", "dc", "washington", "las vegas", "portland", "san diego",
    "phoenix", "dallas", "houston", "philadelphia", "toronto", "vancouver",
    "montreal", "mexico city", "cancun", "tulum",
    # Europe
    "london", "paris", "barcelona", "madrid", "lisbon", "rome", "milan",
    "berlin", "amsterdam", "copenhagen", "stockholm", "vienna", "prague",
    "budapest", "dublin", "edinburgh", "athens", "santorini", "mykonos",
    "nice", "monaco", "zurich", "geneva", "brussels", "porto",
    # Asia Pacific
    "tokyo", "kyoto", "osaka", "seoul", "singapore", "hong kong", "bangkok",
    "bali", "phuket", "vietnam", "hanoi", "ho chi minh", "manila", "taipei",
    "shanghai", "beijing", "shenzhen", "kuala lumpur", "jakarta",
    # Middle East & Africa
    "dubai", "abu dhabi", "doha", "riyadh", "tel aviv", "marrakech", "cairo",
    "cape town", "johannesburg", "nairobi",
    # Oceania
    "sydney", "melbourne", "brisbane", "auckland", "queenstown",
    # Caribbean & Latin America
    "san juan", "havana", "cartagena", "medellin", "bogota", "lima",
    "buenos aires", "rio de janeiro", "sao paulo",
]
SEGMENT_KEYWORDS = {
    "luxury": ["luxury", "5-star", "five star", "high-end", "premium"],
    "boutique": ["boutique", "design", "independent", "unique"],
    "lifestyle": ["lifestyle", "millennial", "gen z", "trendy"],
    "budget": ["budget", "affordable", "hostel", "economy"],
    "wellness": ["wellness", "spa", "retreat", "health"],
    "business": ["business", "corporate", "conference"],
}


class ModeRouter:
    """Routes chat messages to appropriate mode using Bayesian inference.

    Computes P(mode | message) using:
    - Prior probabilities
    - Keyword likelihood
    - Structural signals (URLs, numbers)
    """

    def __init__(self, llm_client: Any = None):
        """Initialize router.

        Args:
            llm_client: Optional LLM client for advanced routing
        """
        self.llm_client = llm_client

    def route(self, message: str, context: list[dict] | None = None) -> RouterOutput:
        """Route a message to determine mode and extract slots.

        Args:
            message: User message
            context: Previous conversation context

        Returns:
            RouterOutput with mode probabilities and slot values
        """
        message_lower = message.lower()

        # Extract slots first
        slots = self._extract_slots(message)

        # Compute keyword-based likelihoods
        insight_score = self._compute_keyword_score(message_lower, INSIGHT_KEYWORDS)
        brand_score = self._compute_keyword_score(message_lower, BRAND_BUILD_KEYWORDS)
        scan_score = self._compute_keyword_score(message_lower, DEMAND_SCAN_KEYWORDS)

        # Apply structural priors
        if slots.url:
            scan_score += 0.5  # Strong signal for demand scan
        if slots.location and slots.segment:
            brand_score += 0.3  # Has inputs for brand building

        # Normalize to probabilities (softmax-like)
        total = insight_score + brand_score + scan_score + 0.001  # avoid div by zero

        # Combine with priors (Bayesian update)
        p_insight = (MODE_PRIORS[ChatMode.INSIGHT] * (1 + insight_score)) / (1 + total)
        p_brand = (MODE_PRIORS[ChatMode.BRAND_BUILD] * (1 + brand_score)) / (1 + total)
        p_scan = (MODE_PRIORS[ChatMode.DEMAND_SCAN] * (1 + scan_score)) / (1 + total)

        # Normalize
        p_total = p_insight + p_brand + p_scan
        p_insight /= p_total
        p_brand /= p_total
        p_scan /= p_total

        # Compute confidence (max prob)
        confidence = max(p_insight, p_brand, p_scan)

        # Determine needed slots based on predicted mode
        slots_needed = self._get_needed_slots(
            ChatMode.INSIGHT if p_insight >= max(p_brand, p_scan) else
            ChatMode.BRAND_BUILD if p_brand > p_scan else
            ChatMode.DEMAND_SCAN,
            slots,
        )

        return RouterOutput(
            p_insight=round(p_insight, 3),
            p_brand_build=round(p_brand, 3),
            p_demand_scan=round(p_scan, 3),
            confidence=round(confidence, 3),
            slots_detected=slots,
            slots_needed=slots_needed,
        )

    def _compute_keyword_score(self, text: str, keywords: list[str]) -> float:
        """Compute score based on keyword matches.

        Args:
            text: Lowercase message text
            keywords: Keywords to match

        Returns:
            Score (0-1 range, can exceed with many matches)
        """
        score = 0.0
        for kw in keywords:
            if kw in text:
                # Multi-word phrases get higher weight
                word_count = len(kw.split())
                if word_count >= 3:
                    score += 0.4  # Strong signal for 3+ word phrases
                elif word_count == 2:
                    score += 0.25  # Medium signal for 2-word phrases
                else:
                    score += 0.15  # Single word
        return min(score, 1.0)

    def _extract_slots(self, message: str) -> SlotValues:
        """Extract slot values from message.

        Args:
            message: User message

        Returns:
            SlotValues with extracted values
        """
        message_lower = message.lower()

        # Extract URL
        url_match = URL_PATTERN.search(message)
        url = url_match.group(0) if url_match else None

        # Extract ADR
        adr = None
        adr_match = ADR_PATTERN.search(message)
        if adr_match:
            adr_str = adr_match.group(0)
            # Extract number
            nums = re.findall(r'\d+', adr_str)
            if nums:
                adr = float(nums[0])

        # Extract location
        location = None
        for loc in LOCATION_KEYWORDS:
            if loc in message_lower:
                location = loc.title()
                break

        # Extract segment
        segment = None
        for seg_name, seg_keywords in SEGMENT_KEYWORDS.items():
            if any(kw in message_lower for kw in seg_keywords):
                segment = seg_name
                break

        # Extract developer goal (if present)
        developer_goal = None
        goal_patterns = [
            r'(?:i want to|goal is to|trying to|need to)\s+(.+?)(?:\.|$)',
            r'(?:my goal|objective|aim)(?:\s+is)?\s*:?\s*(.+?)(?:\.|$)',
        ]
        for pattern in goal_patterns:
            match = re.search(pattern, message_lower)
            if match:
                developer_goal = match.group(1).strip()[:200]
                break

        return SlotValues(
            location=location,
            segment=segment,
            adr=adr,
            url=url,
            developer_goal=developer_goal,
        )

    def _get_needed_slots(self, mode: ChatMode, current_slots: SlotValues) -> list[str]:
        """Determine which slots are still needed for the mode.

        Args:
            mode: Predicted mode
            current_slots: Currently extracted slots

        Returns:
            List of missing slot names
        """
        needed = []

        if mode == ChatMode.BRAND_BUILD:
            if not current_slots.location:
                needed.append("location")
            if not current_slots.segment:
                needed.append("segment")
            # ADR is optional but helpful

        elif mode == ChatMode.DEMAND_SCAN:
            if not current_slots.url:
                needed.append("url")

        # Insight mode doesn't require specific slots

        return needed

    async def route_with_llm(
        self,
        message: str,
        context: list[dict] | None = None,
    ) -> RouterOutput:
        """Route using LLM for more accurate classification.

        Falls back to keyword-based routing if LLM fails.

        Args:
            message: User message
            context: Conversation context

        Returns:
            RouterOutput
        """
        if not self.llm_client:
            return self.route(message, context)

        try:
            # Build prompt for LLM
            prompt = self._build_router_prompt(message, context)

            # Call LLM
            response = await self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            # Parse response
            result = json.loads(response.content)

            # Extract slots from both LLM and keywords
            keyword_slots = self._extract_slots(message)
            llm_slots = SlotValues(
                location=result.get("location") or keyword_slots.location,
                segment=result.get("segment") or keyword_slots.segment,
                adr=result.get("adr") or keyword_slots.adr,
                url=result.get("url") or keyword_slots.url,
                developer_goal=result.get("developer_goal") or keyword_slots.developer_goal,
            )

            return RouterOutput(
                p_insight=result.get("p_insight", 0.33),
                p_brand_build=result.get("p_brand_build", 0.33),
                p_demand_scan=result.get("p_demand_scan", 0.33),
                confidence=result.get("confidence", 0.5),
                slots_detected=llm_slots,
                slots_needed=result.get("slots_needed", []),
            )

        except Exception as e:
            logger.warning(f"LLM routing failed, using keyword fallback: {e}")
            return self.route(message, context)

    def _build_router_prompt(self, message: str, context: list[dict] | None) -> str:
        """Build prompt for LLM router."""
        context_str = ""
        if context:
            context_str = "\n".join(
                f"{m['role']}: {m['content'][:200]}"
                for m in context[-3:]  # Last 3 messages
            )
            context_str = f"\nPrevious conversation:\n{context_str}\n"

        return f"""You are a router for a hospitality intelligence chat system.
Classify the user's intent and extract relevant slots.

{context_str}
User message: {message}

Classify into one of three modes:
1. insight - User wants trends, market analysis, opportunities, forecasts
2. brand_build - User wants to create/design a hotel brand concept
3. demand_scan - User wants to analyze a specific property/website

Return JSON:
{{
  "p_insight": 0.0-1.0,
  "p_brand_build": 0.0-1.0,
  "p_demand_scan": 0.0-1.0,
  "confidence": 0.0-1.0,
  "location": "city name or null",
  "segment": "luxury|boutique|lifestyle|budget|wellness|business or null",
  "adr": number or null,
  "url": "url string or null",
  "developer_goal": "goal text or null",
  "slots_needed": ["list", "of", "missing", "slots"]
}}

Probabilities must sum to 1.0. Only include slots that are clearly stated."""
