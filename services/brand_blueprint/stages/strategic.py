"""Stage 2: Strategic - Pillars, positioning, unmet desires."""

import logging
from typing import Any

from .base import BaseStage, PipelineContext
from services.brand_blueprint.prompts import (
    STRATEGIC_SYSTEM_PROMPT,
    STRATEGIC_USER_TEMPLATE,
    RAG_QUERIES,
    FALLBACK_PILLARS,
)

logger = logging.getLogger(__name__)


class StrategicStage(BaseStage):
    """Stage 2: Generate strategic positioning elements.

    Outputs:
    - pillars (3-5)
    - positioning_statement
    - unmet_desires_solved
    """

    name = "strategic"
    required = True
    max_retries = 2

    def get_system_prompt(self) -> str:
        return STRATEGIC_SYSTEM_PROMPT

    def get_rag_queries(self, context: PipelineContext) -> list[str]:
        queries = []
        for template in RAG_QUERIES["strategic"]:
            query = template.format(
                location=context.inputs.location,
                segment=context.inputs.segment,
            )
            queries.append(query)
        return queries

    def build_user_prompt(self, context: PipelineContext, rag_context: str) -> str:
        # Get foundation output
        foundation = context.get_stage_output("foundation") or {}
        brand_names = foundation.get("brand_names", {})
        brand_name = brand_names.get("primary", "The Hotel")

        # Build trend context
        trend_context = ""
        if context.inputs.profile_data:
            profile = context.inputs.profile_data
            if "topics" in profile:
                trend_context = f"""
RESEARCH CONTEXT:
- Topics of interest: {', '.join(profile.get('topics', [])[:5])}
- Regions: {', '.join(profile.get('regions', [])[:3])}
"""

        return STRATEGIC_USER_TEMPLATE.format(
            brand_name=brand_name,
            one_liner=foundation.get("one_liner", ""),
            thesis=foundation.get("thesis", ""),
            location=context.inputs.location,
            segment=context.inputs.segment,
            adr=context.inputs.adr,
            developer_goal=context.inputs.developer_goal,
            trend_context=trend_context,
            rag_context=rag_context,
        )

    def parse_response(self, response: str) -> dict[str, Any]:
        data = self._extract_json(response)

        # Validate pillars
        if "pillars" not in data or not isinstance(data["pillars"], list):
            raise ValueError("Missing or invalid pillars")

        pillars = [p.strip() for p in data["pillars"] if p and isinstance(p, str)]
        if len(pillars) < 3:
            raise ValueError("Need at least 3 pillars")

        pillars = pillars[:5]  # Max 5

        # Validate positioning
        if "positioning_statement" not in data or not data["positioning_statement"]:
            raise ValueError("Missing positioning_statement")

        # Parse unmet desires
        unmet_desires = []
        raw_desires = data.get("unmet_desires_solved", [])
        if isinstance(raw_desires, list):
            for desire in raw_desires:
                if isinstance(desire, dict) and "desire" in desire:
                    unmet_desires.append({
                        "desire": desire.get("desire", "").strip(),
                        "how_solved": desire.get("how_solved", "").strip(),
                        "linked_trend_id": desire.get("linked_trend_id"),
                        "demand_strength": float(desire.get("demand_strength", 0.5)),
                    })

        return {
            "pillars": pillars,
            "positioning_statement": data["positioning_statement"].strip(),
            "unmet_desires_solved": unmet_desires,
        }

    def get_fallback(self, context: PipelineContext) -> dict[str, Any]:
        """Use segment-based default pillars."""
        segment = context.inputs.segment.lower()
        pillars = FALLBACK_PILLARS.get(segment, FALLBACK_PILLARS["lifestyle"])

        return {
            "pillars": pillars,
            "positioning_statement": f"A distinctive {context.inputs.segment} hotel in {context.inputs.location} that delivers exceptional value at ${context.inputs.adr} ADR.",
            "unmet_desires_solved": [
                {
                    "desire": "Authentic local experiences",
                    "how_solved": "Curated local partnerships and experiences",
                    "linked_trend_id": None,
                    "demand_strength": 0.6,
                }
            ],
        }
