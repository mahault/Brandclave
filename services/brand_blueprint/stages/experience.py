"""Stage 3: Experience - Personas, experiences, guest journey."""

import logging
from typing import Any

from .base import BaseStage, PipelineContext
from services.brand_blueprint.prompts import (
    EXPERIENCE_SYSTEM_PROMPT,
    EXPERIENCE_USER_TEMPLATE,
    RAG_QUERIES,
    FALLBACK_EXPERIENCES,
)

logger = logging.getLogger(__name__)


class ExperienceStage(BaseStage):
    """Stage 3: Generate guest experience elements.

    Outputs:
    - guest_personas (2-3)
    - signature_experiences (3-5)
    - guest_journey
    """

    name = "experience"
    required = False  # Can fallback if needed
    max_retries = 1

    def get_system_prompt(self) -> str:
        return EXPERIENCE_SYSTEM_PROMPT

    def get_rag_queries(self, context: PipelineContext) -> list[str]:
        queries = []
        for template in RAG_QUERIES["experience"]:
            query = template.format(
                location=context.inputs.location,
                segment=context.inputs.segment,
            )
            queries.append(query)
        return queries

    def build_user_prompt(self, context: PipelineContext, rag_context: str) -> str:
        # Get previous outputs
        foundation = context.get_stage_output("foundation") or {}
        strategic = context.get_stage_output("strategic") or {}

        brand_names = foundation.get("brand_names", {})
        brand_name = brand_names.get("primary", "The Hotel")
        pillars = strategic.get("pillars", [])

        # Build trend context
        trend_context = ""
        if context.inputs.profile_data:
            profile = context.inputs.profile_data
            if "segments" in profile:
                trend_context = f"""
TARGET SEGMENTS FROM RESEARCH:
- Segments: {', '.join(profile.get('segments', [])[:3])}
"""

        return EXPERIENCE_USER_TEMPLATE.format(
            brand_name=brand_name,
            one_liner=foundation.get("one_liner", ""),
            thesis=foundation.get("thesis", ""),
            pillars=", ".join(pillars),
            positioning_statement=strategic.get("positioning_statement", ""),
            location=context.inputs.location,
            segment=context.inputs.segment,
            adr=context.inputs.adr,
            rooms=context.inputs.rooms,
            trend_context=trend_context,
            rag_context=rag_context,
        )

    def parse_response(self, response: str) -> dict[str, Any]:
        data = self._extract_json(response)

        # Parse guest personas
        personas = []
        raw_personas = data.get("guest_personas", [])
        if isinstance(raw_personas, list):
            for p in raw_personas[:3]:  # Max 3
                if isinstance(p, dict) and "name" in p:
                    personas.append({
                        "name": p.get("name", "").strip(),
                        "description": p.get("description", "").strip(),
                        "spend_behavior": p.get("spend_behavior", "").strip(),
                    })

        if len(personas) < 2:
            raise ValueError("Need at least 2 guest personas")

        # Parse signature experiences
        experiences = []
        raw_exp = data.get("signature_experiences", [])
        if isinstance(raw_exp, list):
            for e in raw_exp[:5]:  # Max 5
                if isinstance(e, dict) and "name" in e:
                    experiences.append({
                        "name": e.get("name", "").strip(),
                        "description": e.get("description", "").strip(),
                        "why_it_matters": e.get("why_it_matters", "").strip(),
                    })

        if len(experiences) < 3:
            raise ValueError("Need at least 3 signature experiences")

        # Parse guest journey
        journey = data.get("guest_journey", {})
        if not isinstance(journey, dict):
            raise ValueError("Invalid guest_journey format")

        guest_journey = {
            "arrival": journey.get("arrival", "").strip(),
            "stay": journey.get("stay", "").strip(),
            "departure": journey.get("departure", "").strip(),
        }

        if not all(guest_journey.values()):
            raise ValueError("Guest journey missing required phases")

        return {
            "guest_personas": personas,
            "signature_experiences": experiences,
            "guest_journey": guest_journey,
        }

    def get_fallback(self, context: PipelineContext) -> dict[str, Any]:
        """Use segment-based default experiences."""
        segment = context.inputs.segment.lower()
        experiences = FALLBACK_EXPERIENCES.get(segment, FALLBACK_EXPERIENCES.get("lifestyle", []))

        return {
            "guest_personas": [
                {
                    "name": "The Design-Conscious Traveler",
                    "description": "Values aesthetics and experiences over amenities. Seeks authentic, photo-worthy moments.",
                    "spend_behavior": "Splurges on unique experiences and F&B, moderate on room upgrades.",
                },
                {
                    "name": "The Connected Professional",
                    "description": "Works remotely while traveling. Values reliable wifi, good coffee, and social spaces.",
                    "spend_behavior": "Consistent spender on extended stays, values workspace amenities.",
                },
            ],
            "signature_experiences": experiences or [
                {
                    "name": "Welcome Ritual",
                    "description": "A personalized arrival experience",
                    "why_it_matters": "Sets the tone for the stay",
                },
                {
                    "name": "Local Discovery Program",
                    "description": "Curated neighborhood experiences",
                    "why_it_matters": "Enables authentic exploration",
                },
                {
                    "name": "Evening Social Hour",
                    "description": "Complimentary drinks and conversation",
                    "why_it_matters": "Creates community among guests",
                },
            ],
            "guest_journey": {
                "arrival": "Seamless check-in with welcome drink and neighborhood orientation.",
                "stay": "Daily opportunities for discovery, relaxation, and connection.",
                "departure": "Thoughtful farewell with local gift and easy checkout.",
            },
        }
