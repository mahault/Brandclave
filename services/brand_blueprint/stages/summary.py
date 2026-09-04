"""Stage 5: Summary - Investor summary."""

import logging
from typing import Any

from .base import BaseStage, PipelineContext, coerce_text
from services.brand_blueprint.prompts import (
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


class SummaryStage(BaseStage):
    """Stage 5: Generate investor summary.

    Outputs:
    - investor_summary

    This stage synthesizes all previous outputs into a compelling pitch.
    No RAG context needed - uses previous stage outputs.
    """

    name = "summary"
    required = True
    max_retries = 2

    def get_system_prompt(self) -> str:
        return SUMMARY_SYSTEM_PROMPT

    def get_rag_queries(self, context: PipelineContext) -> list[str]:
        # No RAG needed for summary - uses previous outputs
        return []

    async def retrieve_context(self, context: PipelineContext) -> str:
        # Override to skip RAG retrieval
        return ""

    def build_user_prompt(self, context: PipelineContext, rag_context: str) -> str:
        # Get all previous outputs
        foundation = context.get_stage_output("foundation") or {}
        strategic = context.get_stage_output("strategic") or {}
        experience = context.get_stage_output("experience") or {}
        atmosphere = context.get_stage_output("atmosphere") or {}

        brand_names = foundation.get("brand_names", {})
        brand_name = brand_names.get("primary", "The Hotel")
        pillars = strategic.get("pillars", [])
        unmet_desires = strategic.get("unmet_desires_solved", [])

        # Summarize personas
        personas = experience.get("guest_personas", [])
        personas_summary = "\n".join([
            f"- {p.get('name', 'Guest')}: {p.get('description', '')}"
            for p in personas
        ]) if personas else "Design-conscious travelers seeking authentic experiences."

        # Summarize experiences
        experiences = experience.get("signature_experiences", [])
        experiences_summary = "\n".join([
            f"- {e.get('name', 'Experience')}: {e.get('description', '')}"
            for e in experiences
        ]) if experiences else "Curated local experiences and social programming."

        # Summarize F&B
        fnb = atmosphere.get("fnb_concepts", [])
        fnb_summary = ", ".join([f.get("name", "") for f in fnb]) if fnb else "Restaurant and bar concepts"

        # Format unmet desires
        desires_text = "\n".join([
            f"- {d.get('desire', '')}: {d.get('how_solved', '')}"
            for d in unmet_desires
        ]) if unmet_desires else "Authentic local experiences and modern comfort."

        return SUMMARY_USER_TEMPLATE.format(
            brand_name=brand_name,
            one_liner=foundation.get("one_liner", ""),
            thesis=foundation.get("thesis", ""),
            pillars=", ".join(pillars),
            positioning_statement=strategic.get("positioning_statement", ""),
            unmet_desires=desires_text,
            personas_summary=personas_summary,
            experiences_summary=experiences_summary,
            design_direction=atmosphere.get("design_direction", ""),
            fnb_summary=fnb_summary,
            revenue_logic=atmosphere.get("revenue_logic", ""),
            location=context.inputs.location,
            segment=context.inputs.segment,
            adr=context.inputs.adr,
            rooms=context.inputs.rooms,
            developer_goal=context.inputs.developer_goal,
        )

    def parse_response(self, response: str) -> dict[str, Any]:
        data = self._extract_json(response)

        if "investor_summary" not in data or not data["investor_summary"]:
            raise ValueError("Missing investor_summary")

        return {
            "investor_summary": coerce_text(data["investor_summary"]),
        }

    def get_fallback(self, context: PipelineContext) -> dict[str, Any]:
        """Generate template-based summary from previous outputs."""
        foundation = context.get_stage_output("foundation") or {}
        strategic = context.get_stage_output("strategic") or {}

        brand_names = foundation.get("brand_names", {})
        brand_name = brand_names.get("primary", "The Hotel")
        thesis = foundation.get("thesis", "")
        pillars = strategic.get("pillars", [])

        summary = f"""{brand_name} represents a compelling opportunity in {context.inputs.location}'s {context.inputs.segment} hospitality market.

With {context.inputs.rooms} keys targeting ${context.inputs.adr} ADR, the concept addresses clear market demand for {', '.join(pillars[:2]) if pillars else 'distinctive experiences'}.

{thesis[:300] if thesis else 'The brand is positioned to capture share from travelers seeking authentic, design-forward accommodations.'}

The investment thesis is supported by strong market fundamentals in {context.inputs.location} and a differentiated product offering that commands premium positioning."""

        return {
            "investor_summary": summary,
        }
