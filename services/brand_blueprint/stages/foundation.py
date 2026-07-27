"""Stage 1: Foundation - Brand names, one-liner, thesis."""

import logging
from typing import Any

from .base import BaseStage, PipelineContext
from services.brand_blueprint.prompts import (
    FOUNDATION_SYSTEM_PROMPT,
    FOUNDATION_USER_TEMPLATE,
    RAG_QUERIES,
)

logger = logging.getLogger(__name__)


class FoundationStage(BaseStage):
    """Stage 1: Generate foundational brand elements.

    Outputs:
    - brand_names (primary + 2 alternates)
    - one_liner
    - thesis
    """

    name = "foundation"
    required = True
    max_retries = 2

    def get_system_prompt(self) -> str:
        return FOUNDATION_SYSTEM_PROMPT

    def get_rag_queries(self, context: PipelineContext) -> list[str]:
        queries = []
        for template in RAG_QUERIES["foundation"]:
            query = template.format(
                location=context.inputs.location,
                segment=context.inputs.segment,
            )
            queries.append(query)
        return queries

    def build_user_prompt(self, context: PipelineContext, rag_context: str) -> str:
        # Build trend context if available
        trend_context = ""
        if context.inputs.source_trend_id and context.inputs.profile_data:
            profile = context.inputs.profile_data
            if "source_trend_name" in profile:
                trend_context = f"""
ATTACHED TREND SIGNAL:
- Trend: {profile.get('source_trend_name', '')}
- Description: {profile.get('description', '')}
- Why it matters: {profile.get('why_it_matters', '')}
- White Space Score: {profile.get('white_space_score', 'N/A')}
Build the brand to capitalize on this trend.
"""

        return FOUNDATION_USER_TEMPLATE.format(
            location=context.inputs.location,
            segment=context.inputs.segment,
            adr=context.inputs.adr,
            rooms=context.inputs.rooms,
            developer_goal=context.inputs.developer_goal,
            trend_context=trend_context,
            rag_context=rag_context,
        )

    def parse_response(self, response: str) -> dict[str, Any]:
        data = self._extract_json(response)

        # Validate required fields
        if "brand_names" not in data:
            raise ValueError("Missing brand_names in response")

        brand_names = data["brand_names"]
        if not isinstance(brand_names, dict):
            raise ValueError("brand_names must be an object")

        required_keys = ["primary", "alternate_1", "alternate_2"]
        for key in required_keys:
            if key not in brand_names or not brand_names[key]:
                raise ValueError(f"Missing {key} in brand_names")

        if "one_liner" not in data or not data["one_liner"]:
            raise ValueError("Missing one_liner in response")

        if "thesis" not in data or not data["thesis"]:
            raise ValueError("Missing thesis in response")

        return {
            "brand_names": {
                "primary": brand_names["primary"].strip(),
                "alternate_1": brand_names["alternate_1"].strip(),
                "alternate_2": brand_names["alternate_2"].strip(),
            },
            "one_liner": data["one_liner"].strip(),
            "thesis": data["thesis"].strip(),
        }

    def get_fallback(self, context: PipelineContext) -> dict[str, Any]:
        """Generate fallback brand name from location + segment."""
        location_word = context.inputs.location.split(",")[0].split()[0]
        segment = context.inputs.segment.title()

        return {
            "brand_names": {
                "primary": f"The {location_word} House",
                "alternate_1": f"{location_word} {segment}",
                "alternate_2": f"Casa {location_word}",
            },
            "one_liner": f"A {context.inputs.segment} retreat in the heart of {context.inputs.location}.",
            "thesis": f"A new {context.inputs.segment} hotel concept for {context.inputs.location}, designed to capture the spirit of the destination while delivering exceptional guest experiences at ${context.inputs.adr} ADR.",
        }
