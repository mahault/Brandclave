"""Stage 4: Atmosphere - Design direction, F&B, revenue logic."""

import logging
from typing import Any

from .base import BaseStage, PipelineContext
from services.brand_blueprint.prompts import (
    ATMOSPHERE_SYSTEM_PROMPT,
    ATMOSPHERE_USER_TEMPLATE,
    RAG_QUERIES,
)

logger = logging.getLogger(__name__)


class AtmosphereStage(BaseStage):
    """Stage 4: Generate atmosphere and revenue elements.

    Outputs:
    - design_direction
    - fnb_concepts
    - revenue_logic
    """

    name = "atmosphere"
    required = False  # Can fallback if needed
    max_retries = 1

    def get_system_prompt(self) -> str:
        return ATMOSPHERE_SYSTEM_PROMPT

    def get_rag_queries(self, context: PipelineContext) -> list[str]:
        queries = []
        for template in RAG_QUERIES["atmosphere"]:
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
        experience = context.get_stage_output("experience") or {}

        brand_names = foundation.get("brand_names", {})
        brand_name = brand_names.get("primary", "The Hotel")
        pillars = strategic.get("pillars", [])

        # Summarize personas
        personas = experience.get("guest_personas", [])
        personas_summary = "\n".join([
            f"- {p.get('name', 'Guest')}: {p.get('description', '')}"
            for p in personas
        ]) if personas else "Design-conscious travelers seeking authentic experiences."

        # Build trend context (minimal for this stage)
        trend_context = ""

        return ATMOSPHERE_USER_TEMPLATE.format(
            brand_name=brand_name,
            one_liner=foundation.get("one_liner", ""),
            pillars=", ".join(pillars),
            personas_summary=personas_summary,
            location=context.inputs.location,
            segment=context.inputs.segment,
            adr=context.inputs.adr,
            rooms=context.inputs.rooms,
            trend_context=trend_context,
            rag_context=rag_context,
        )

    def parse_response(self, response: str) -> dict[str, Any]:
        data = self._extract_json(response)

        # Validate design direction
        if "design_direction" not in data or not data["design_direction"]:
            raise ValueError("Missing design_direction")

        # Parse F&B concepts
        fnb_concepts = []
        raw_fnb = data.get("fnb_concepts", [])
        if isinstance(raw_fnb, list):
            for f in raw_fnb[:4]:  # Max 4
                if isinstance(f, dict) and "name" in f:
                    fnb_concepts.append({
                        "name": f.get("name", "").strip(),
                        "concept": f.get("concept", "").strip(),
                        "vibe": f.get("vibe", "").strip(),
                    })

        # Validate revenue logic
        if "revenue_logic" not in data or not data["revenue_logic"]:
            raise ValueError("Missing revenue_logic")

        return {
            "design_direction": data["design_direction"].strip(),
            "fnb_concepts": fnb_concepts,
            "revenue_logic": data["revenue_logic"].strip(),
        }

    def get_fallback(self, context: PipelineContext) -> dict[str, Any]:
        """Generate segment-appropriate fallback."""
        segment = context.inputs.segment.lower()

        design_templates = {
            "luxury": "Rich natural materials - warm woods, marble, brass accents. Soft, layered lighting. Custom furnishings with artisanal details. Scent signature: warm amber and sandalwood.",
            "lifestyle": "Industrial-modern aesthetic with exposed elements and curated art. Bold color accents. Flexible social spaces. Music-forward atmosphere with curated playlists.",
            "boutique": "Eclectic, collected aesthetic with vintage pieces and local art. Intimate scale with personal touches. Warm, residential feel with character details.",
            "wellness": "Organic materials - natural wood, stone, plants. Clean lines and calm palettes. Natural light maximized. Subtle aromatherapy throughout.",
        }

        fnb_templates = {
            "luxury": [
                {"name": "The Restaurant", "concept": "Modern fine dining with local ingredients", "vibe": "Elegant, occasion-worthy"},
                {"name": "The Bar", "concept": "Classic cocktails and rare spirits", "vibe": "Sophisticated, intimate"},
            ],
            "lifestyle": [
                {"name": "The Lobby Bar", "concept": "All-day cafe and cocktail bar", "vibe": "Social, energetic"},
                {"name": "The Rooftop", "concept": "Casual dining with views", "vibe": "Scene-y, sunset-focused"},
            ],
        }

        return {
            "design_direction": design_templates.get(segment, design_templates["lifestyle"]),
            "fnb_concepts": fnb_templates.get(segment, fnb_templates["lifestyle"]),
            "revenue_logic": f"The ${context.inputs.adr} ADR is justified through distinctive design, curated experiences, and strategic positioning in {context.inputs.location}'s {context.inputs.segment} market. Premium F&B and experience programming drive ancillary revenue.",
        }
