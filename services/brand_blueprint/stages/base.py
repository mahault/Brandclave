"""Base class for pipeline stages."""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from services.brand_blueprint.schemas import BlueprintInputs

logger = logging.getLogger(__name__)


class PipelineContext:
    """Context passed between pipeline stages."""

    def __init__(self, inputs: "BlueprintInputs"):
        self.inputs = inputs
        self.stage_outputs: dict[str, dict[str, Any]] = {}
        self.warnings: list[str] = []
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def add_stage_output(self, stage_name: str, output: dict[str, Any]) -> None:
        """Add output from a completed stage."""
        self.stage_outputs[stage_name] = output

    def get_stage_output(self, stage_name: str) -> dict[str, Any] | None:
        """Get output from a previous stage."""
        return self.stage_outputs.get(stage_name)

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Track token usage."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens


class BaseStage(ABC):
    """Base class for pipeline stages.

    Each stage:
    1. Retrieves RAG context (optional)
    2. Builds a prompt with inputs + previous outputs
    3. Calls the LLM
    4. Parses and validates the response
    """

    name: str = "base"
    required: bool = True
    max_retries: int = 2

    def __init__(self, llm_client: Any, rag: Any | None = None):
        """Initialize the stage.

        Args:
            llm_client: The LLM client for generation
            rag: Optional RAG instance for context retrieval
        """
        self.llm_client = llm_client
        self.rag = rag

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this stage."""
        pass

    @abstractmethod
    def build_user_prompt(self, context: PipelineContext, rag_context: str) -> str:
        """Build the user prompt from context.

        Args:
            context: The pipeline context with inputs and previous outputs
            rag_context: Retrieved RAG context string

        Returns:
            The formatted user prompt
        """
        pass

    @abstractmethod
    def parse_response(self, response: str) -> dict[str, Any]:
        """Parse the LLM response into structured output.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed structured output
        """
        pass

    @abstractmethod
    def get_fallback(self, context: PipelineContext) -> dict[str, Any]:
        """Get fallback output if generation fails.

        Args:
            context: The pipeline context

        Returns:
            Fallback output dict
        """
        pass

    def get_rag_queries(self, context: PipelineContext) -> list[str]:
        """Get RAG queries for this stage.

        Override in subclasses for stage-specific queries.

        Args:
            context: The pipeline context

        Returns:
            List of RAG query strings
        """
        return []

    async def retrieve_context(self, context: PipelineContext) -> str:
        """Retrieve RAG context for this stage.

        Args:
            context: The pipeline context

        Returns:
            Formatted RAG context string
        """
        if self.rag is None:
            return "No market context available."

        queries = self.get_rag_queries(context)
        if not queries:
            return "No market context available."

        all_chunks = []
        for query in queries[:3]:  # Limit to 3 queries
            try:
                result = self.rag.retrieve(query, top_k=3)
                chunks = result.get("chunks", [])
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"RAG query failed: {query}, error: {e}")

        if not all_chunks:
            return "No relevant market context found."

        # Dedupe and format
        seen = set()
        context_parts = []
        for chunk in all_chunks[:8]:  # Limit to 8 chunks
            text = chunk.get("text", "")[:500]
            if text and text not in seen:
                seen.add(text)
                source = chunk.get("source", "unknown")
                context_parts.append(f"- [{source}] {text}")

        return "\n".join(context_parts) if context_parts else "No relevant market context found."

    async def execute(self, context: PipelineContext) -> dict[str, Any]:
        """Execute this stage.

        Args:
            context: The pipeline context

        Returns:
            Stage output dict

        Raises:
            Exception: If stage fails after retries
        """
        # Retrieve RAG context
        rag_context = await self.retrieve_context(context)

        # Build prompts
        system_prompt = self.get_system_prompt()
        user_prompt = self.build_user_prompt(context, rag_context)

        # Try with retries
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # Call LLM
                response, input_tokens, output_tokens = await self._call_llm(
                    system_prompt, user_prompt
                )

                # Track tokens
                context.add_tokens(input_tokens, output_tokens)

                # Parse response
                output = self.parse_response(response)

                logger.info(f"Stage {self.name} completed successfully")
                return output

            except Exception as e:
                last_error = e
                logger.warning(f"Stage {self.name} attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    continue

        # All retries failed
        if self.required:
            raise Exception(f"Stage {self.name} failed after {self.max_retries + 1} attempts: {last_error}")
        else:
            logger.warning(f"Stage {self.name} using fallback due to: {last_error}")
            context.add_warning(f"Stage {self.name} used fallback: {str(last_error)}")
            return self.get_fallback(context)

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        """Call the LLM and return response with token counts.

        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        try:
            # Use the LLM client's chat method
            result = await self.llm_client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )

            # Handle both dict and LLMResponse objects
            if hasattr(result, "content"):
                response_text = result.content
                # Try to get token usage from raw response
                raw = getattr(result, "raw", None)
                if raw and hasattr(raw, "usage"):
                    input_tokens = getattr(raw.usage, "prompt_tokens", 0)
                    output_tokens = getattr(raw.usage, "completion_tokens", 0)
                else:
                    # Estimate tokens (~4 chars per token)
                    input_tokens = (len(system_prompt) + len(user_prompt)) // 4
                    output_tokens = len(response_text) // 4
            else:
                response_text = result.get("content", "")
                input_tokens = result.get("input_tokens", 0)
                output_tokens = result.get("output_tokens", 0)

            return response_text, input_tokens, output_tokens

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from LLM response text.

        Handles responses that may have markdown code blocks or extra text.

        Args:
            text: Raw LLM response

        Returns:
            Parsed JSON dict
        """
        # Try to find JSON in code blocks first
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object directly
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Last resort: try the whole text
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON from response: {e}")
