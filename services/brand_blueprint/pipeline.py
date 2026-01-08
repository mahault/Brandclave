"""Brand Blueprint Generation Pipeline.

Orchestrates the 5-stage pipeline for generating complete brand blueprints.
"""

import logging
from typing import Any, Callable

from .schemas import (
    BlueprintInputs,
    BrandBlueprintFull,
    AlternateBrandNames,
    UnmetDesireSolved,
    StageProgress,
    StageStatus,
    TokenUsage,
)
from .stages import (
    FoundationStage,
    StrategicStage,
    ExperienceStage,
    AtmosphereStage,
    SummaryStage,
)
from .stages.base import PipelineContext
from services.chat.schemas import (
    SignatureExperience,
    GuestJourney,
    GuestPersona,
    FnBConcept,
)

logger = logging.getLogger(__name__)


class BlueprintPipeline:
    """Multi-stage brand blueprint generation pipeline.

    Stages:
    1. Foundation: brand names, one-liner, thesis
    2. Strategic: pillars, positioning, unmet desires
    3. Experience: personas, experiences, guest journey
    4. Atmosphere: design direction, F&B, revenue logic
    5. Summary: investor summary
    """

    # Mistral pricing (approximate)
    COST_PER_1K_INPUT = 0.0001
    COST_PER_1K_OUTPUT = 0.0003

    def __init__(self, llm_client: Any, rag: Any | None = None):
        """Initialize the pipeline.

        Args:
            llm_client: LLM client for generation
            rag: Optional BayesianRAG instance for context retrieval
        """
        self.llm_client = llm_client
        self.rag = rag

        # Initialize stages
        self.stages = [
            FoundationStage(llm_client, rag),
            StrategicStage(llm_client, rag),
            ExperienceStage(llm_client, rag),
            AtmosphereStage(llm_client, rag),
            SummaryStage(llm_client),  # No RAG needed
        ]

    async def generate(
        self,
        inputs: BlueprintInputs,
        progress_callback: Callable[[StageProgress], None] | None = None,
    ) -> BrandBlueprintFull:
        """Generate a complete brand blueprint.

        Args:
            inputs: The input form data
            progress_callback: Optional callback for progress updates

        Returns:
            Complete brand blueprint
        """
        context = PipelineContext(inputs)
        stage_results: list[StageProgress] = []

        logger.info(f"Starting blueprint generation for {inputs.location} {inputs.segment}")

        for i, stage in enumerate(self.stages):
            progress_pct = int((i / len(self.stages)) * 100)

            # Report starting
            if progress_callback:
                progress_callback(StageProgress(
                    stage=stage.name,
                    status=StageStatus.RUNNING,
                    progress_pct=progress_pct,
                ))

            try:
                # Execute stage
                output = await stage.execute(context)
                context.add_stage_output(stage.name, output)

                # Report completion
                stage_result = StageProgress(
                    stage=stage.name,
                    status=StageStatus.COMPLETED,
                    progress_pct=progress_pct + int(100 / len(self.stages)),
                    output=output,
                )
                stage_results.append(stage_result)

                if progress_callback:
                    progress_callback(stage_result)

                logger.info(f"Stage {stage.name} completed")

            except Exception as e:
                logger.error(f"Stage {stage.name} failed: {e}")

                stage_result = StageProgress(
                    stage=stage.name,
                    status=StageStatus.FAILED,
                    progress_pct=progress_pct,
                    error=str(e),
                )
                stage_results.append(stage_result)

                if progress_callback:
                    progress_callback(stage_result)

                if stage.required:
                    # Return partial blueprint with error status
                    return self._build_partial_blueprint(
                        context,
                        stage_results,
                        status="failed",
                        error=str(e),
                    )
                else:
                    # Continue with fallback
                    fallback = stage.get_fallback(context)
                    context.add_stage_output(stage.name, fallback)

        # Build final blueprint
        return self._build_blueprint(context, stage_results)

    def _build_blueprint(
        self,
        context: PipelineContext,
        stage_results: list[StageProgress],
    ) -> BrandBlueprintFull:
        """Build the final blueprint from stage outputs."""
        foundation = context.get_stage_output("foundation") or {}
        strategic = context.get_stage_output("strategic") or {}
        experience = context.get_stage_output("experience") or {}
        atmosphere = context.get_stage_output("atmosphere") or {}
        summary = context.get_stage_output("summary") or {}

        # Build brand names
        brand_names_data = foundation.get("brand_names", {})
        brand_names = AlternateBrandNames(
            primary=brand_names_data.get("primary", ""),
            alternate_1=brand_names_data.get("alternate_1", ""),
            alternate_2=brand_names_data.get("alternate_2", ""),
        )

        # Build unmet desires
        unmet_desires = [
            UnmetDesireSolved(**d)
            for d in strategic.get("unmet_desires_solved", [])
        ]

        # Build personas
        guest_personas = [
            GuestPersona(**p)
            for p in experience.get("guest_personas", [])
        ]

        # Build experiences
        signature_experiences = [
            SignatureExperience(**e)
            for e in experience.get("signature_experiences", [])
        ]

        # Build journey
        journey_data = experience.get("guest_journey")
        guest_journey = GuestJourney(**journey_data) if journey_data else None

        # Build F&B concepts
        fnb_concepts = [
            FnBConcept(**f)
            for f in atmosphere.get("fnb_concepts", [])
        ]

        # Calculate token usage
        token_usage = TokenUsage(
            input_tokens=context.total_input_tokens,
            output_tokens=context.total_output_tokens,
            total_tokens=context.total_input_tokens + context.total_output_tokens,
            estimated_cost_usd=(
                (context.total_input_tokens / 1000 * self.COST_PER_1K_INPUT) +
                (context.total_output_tokens / 1000 * self.COST_PER_1K_OUTPUT)
            ),
        )

        # Determine overall status and confidence
        failed_stages = [s for s in stage_results if s.status == StageStatus.FAILED]
        if failed_stages:
            status = "partial"
            confidence = 0.6
        else:
            status = "completed"
            confidence = 0.85

        return BrandBlueprintFull(
            inputs=context.inputs,
            # Stage 1
            brand_names=brand_names,
            one_liner=foundation.get("one_liner", ""),
            thesis=foundation.get("thesis", ""),
            # Stage 2
            pillars=strategic.get("pillars", []),
            positioning_statement=strategic.get("positioning_statement", ""),
            unmet_desires_solved=unmet_desires,
            # Stage 3
            guest_personas=guest_personas,
            signature_experiences=signature_experiences,
            guest_journey=guest_journey,
            # Stage 4
            design_direction=atmosphere.get("design_direction", ""),
            fnb_concepts=fnb_concepts,
            revenue_logic=atmosphere.get("revenue_logic", ""),
            # Stage 5
            investor_summary=summary.get("investor_summary", ""),
            # Meta
            status=status,
            confidence=confidence,
            warnings=context.warnings,
            token_usage=token_usage,
        )

    def _build_partial_blueprint(
        self,
        context: PipelineContext,
        stage_results: list[StageProgress],
        status: str,
        error: str,
    ) -> BrandBlueprintFull:
        """Build a partial blueprint when pipeline fails."""
        # Use the same builder but with error status
        blueprint = self._build_blueprint(context, stage_results)
        blueprint.status = status
        blueprint.confidence = 0.3
        blueprint.warnings.append(f"Pipeline failed: {error}")
        return blueprint
