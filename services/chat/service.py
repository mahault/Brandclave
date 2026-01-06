"""Chat Service - Main orchestrator for BrandClave Chat."""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

from services.chat.belief_manager import BeliefManager, DialogueAction
from services.chat.mode_router import ModeRouter
from services.chat.rag import BayesianRAG, RAGResult
from services.chat.llm_client import get_llm_client, MistralLLMClient
from services.chat.schemas import (
    ChatArtifact,
    ChatMessage,
    ChatMode,
    InsightBrief,
    KeySignal,
    Evidence,
    WhiteSpaceOpportunity,
    RecommendedNextStep,
    NextStepAction,
)

logger = logging.getLogger(__name__)


def _get_embedding_fn():
    """Get the embedding function from data_models."""
    try:
        from data_models.embeddings import embed_text
        return embed_text
    except ImportError:
        logger.warning("Could not import embed_text from data_models.embeddings")
        return None
    except Exception as e:
        logger.warning(f"Error getting embedding function: {e}")
        return None


class ChatService:
    """Main chat service orchestrating mode routing, RAG, and dialogue control.

    Handles the full chat pipeline:
    1. Route message to determine mode
    2. Update belief state
    3. Retrieve relevant context
    4. Generate response with LLM
    5. Create structured artifact
    """

    def __init__(self, llm_client: Any = None, embedding_fn: callable = None):
        """Initialize chat service.

        Args:
            llm_client: LLM client for generation (auto-initialized if None)
            embedding_fn: Function to generate embeddings (auto-initialized if None)
        """
        # Auto-initialize embedding function if not provided
        if embedding_fn is None:
            embedding_fn = _get_embedding_fn()

        # Auto-initialize LLM client if not provided
        if llm_client is None:
            llm_client = get_llm_client()

        self.router = ModeRouter(llm_client=llm_client)
        self.rag = BayesianRAG(embedding_fn=embedding_fn)
        self.belief_manager = BeliefManager()
        self.llm_client = llm_client
        self.embedding_fn = embedding_fn

        logger.info(f"ChatService initialized: LLM={llm_client is not None}, Embeddings={embedding_fn is not None}")

        # Conversation state
        self._conversation_id: str | None = None
        self._messages: list[ChatMessage] = []
        self._artifacts: list[ChatArtifact] = []

    def new_conversation(self, project_id: str | None = None) -> str:
        """Start a new conversation.

        Args:
            project_id: Optional project to associate with

        Returns:
            Conversation ID
        """
        self._conversation_id = str(uuid.uuid4())
        self._messages = []
        self._artifacts = []
        self.belief_manager.reset()

        logger.info(f"Started new conversation: {self._conversation_id}")
        return self._conversation_id

    async def chat(
        self,
        message: str,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a chat message and generate response.

        Args:
            message: User message
            project_id: Optional project context

        Returns:
            Dict with response, artifact, and state
        """
        if not self._conversation_id:
            self.new_conversation(project_id)

        # 1. Route message to determine mode
        router_output = self.router.route(message)
        logger.info(f"Router output: mode={router_output.get_mode()}, conf={router_output.confidence}")

        # 2. Update belief state
        self.belief_manager.update_from_router(router_output)
        self.belief_manager.update_stage(message)

        # 3. Select action based on belief
        action, action_meta = self.belief_manager.select_action()
        logger.info(f"Selected action: {action}, meta={action_meta}")

        # 4. Handle action
        if action == DialogueAction.ASK_CLARIFYING_Q:
            return self._handle_clarifying_question(message, action_meta)

        # 5. Retrieve context
        rag_result = await self._retrieve_context(
            message,
            router_output.slots_detected.location,
            router_output.slots_detected.segment,
        )

        # Update belief with retrieval results
        self.belief_manager.update_from_retrieval(
            rag_result.top_posterior,
            rag_result.entropy,
            rag_result.sources_used,
        )

        # Re-check action after retrieval
        action, action_meta = self.belief_manager.select_action()

        if action == DialogueAction.RETRIEVE_MORE:
            # Try broader retrieval
            rag_result = await self._retrieve_context(message, None, None)

        # 6. Generate response
        mode = self.belief_manager.belief.get_dominant_mode()
        response = await self._generate_response(message, rag_result, mode)

        # 7. Create artifact
        artifact = self._create_artifact(response, rag_result, mode)

        # 8. Store message and artifact
        user_msg = ChatMessage(
            id=str(uuid.uuid4()),
            project_id=project_id,
            role="user",
            content=message,
            mode=mode,
        )
        assistant_msg = ChatMessage(
            id=str(uuid.uuid4()),
            project_id=project_id,
            role="assistant",
            content=response["text"],
            mode=mode,
        )
        self._messages.extend([user_msg, assistant_msg])

        if artifact:
            self._artifacts.append(artifact)

        # 9. Build response
        state = self.belief_manager.get_state_summary()

        result = {
            "conversation_id": self._conversation_id,
            "response": response["text"],
            "mode": mode.value,
            "confidence": self.belief_manager.format_confidence_badge(),
            "sources_used": rag_result.sources_used,
            "state": state,
        }

        if artifact:
            result["artifact"] = artifact.model_dump()

        # Add suggestion if appropriate
        if action == DialogueAction.SUGGEST_BUILD_A_BRAND:
            result["suggested_action"] = {
                "type": "send_to_build_a_brand",
                "prefill": action_meta.get("prefill", {}),
            }

        return result

    def _handle_clarifying_question(
        self,
        user_message: str,
        action_meta: dict,
    ) -> dict[str, Any]:
        """Handle clarifying question action.

        Args:
            user_message: Original user message
            action_meta: Action metadata

        Returns:
            Response dict
        """
        question = action_meta.get("question", "Could you tell me more?")

        # Store messages
        user_msg = ChatMessage(
            id=str(uuid.uuid4()),
            role="user",
            content=user_message,
        )
        assistant_msg = ChatMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=question,
        )
        self._messages.extend([user_msg, assistant_msg])

        return {
            "conversation_id": self._conversation_id,
            "response": question,
            "mode": self.belief_manager.belief.get_dominant_mode().value,
            "confidence": "Low",
            "sources_used": 0,
            "state": self.belief_manager.get_state_summary(),
            "action": "clarifying_question",
            "reason": action_meta.get("reason"),
        }

    async def _retrieve_context(
        self,
        query: str,
        location: str | None,
        segment: str | None,
    ) -> RAGResult:
        """Retrieve relevant context using Bayesian RAG.

        Args:
            query: Search query
            location: Location filter
            segment: Segment filter

        Returns:
            RAGResult with scored chunks
        """
        try:
            return await self.rag.retrieve(
                query=query,
                location=location,
                segment=segment,
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return RAGResult(chunks=[], top_posterior=0.0, entropy=1.0, sources_used=0)

    async def _generate_response(
        self,
        user_message: str,
        rag_result: RAGResult,
        mode: ChatMode,
    ) -> dict[str, Any]:
        """Generate response using LLM.

        Args:
            user_message: User's message
            rag_result: Retrieved context
            mode: Current mode

        Returns:
            Dict with text response and structured data
        """
        # Format context
        context = self.rag.format_context(rag_result.chunks)

        # Get slots
        slots = self.belief_manager.belief.slots

        # Build prompt based on mode
        if mode == ChatMode.INSIGHT:
            prompt = self._build_insight_prompt(user_message, context, slots)
        elif mode == ChatMode.DEMAND_SCAN:
            prompt = self._build_demand_scan_prompt(user_message, context, slots)
        else:
            prompt = self._build_brand_build_prompt(user_message, context, slots)

        # If no LLM client, use fallback
        if not self.llm_client:
            return self._fallback_response(user_message, rag_result, mode)

        try:
            # Call LLM
            response = await self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
            )

            return {
                "text": response.content,
                "raw": response,
            }

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._fallback_response(user_message, rag_result, mode)

    def _build_insight_prompt(
        self,
        message: str,
        context: str,
        slots: Any,
    ) -> str:
        """Build prompt for insight mode."""
        location_str = f" in {slots.location}" if slots.location else ""

        return f"""You are a hospitality intelligence analyst. Answer the user's question about market trends, opportunities, or industry insights{location_str}.

Use the following context from our knowledge base:

{context}

User question: {message}

Provide a clear, actionable answer with:
1. Key signals and trends
2. Why they matter for hospitality
3. White space opportunities if relevant
4. Specific, data-backed insights

Be concise but comprehensive. Cite sources when possible."""

    def _build_demand_scan_prompt(
        self,
        message: str,
        context: str,
        slots: Any,
    ) -> str:
        """Build prompt for demand scan mode."""
        url_str = f"\nProperty URL: {slots.url}" if slots.url else ""

        return f"""You are a hospitality positioning analyst. Analyze the property and identify gaps and opportunities.{url_str}

Use the following market context:

{context}

User request: {message}

Provide:
1. Demand fit assessment
2. Positioning misalignment flags
3. Experience gaps (what guests want vs what's offered)
4. Opportunity lanes (strategic trajectories)

Be specific and actionable."""

    def _build_brand_build_prompt(
        self,
        message: str,
        context: str,
        slots: Any,
    ) -> str:
        """Build prompt for brand build mode."""
        inputs = []
        if slots.location:
            inputs.append(f"Location: {slots.location}")
        if slots.segment:
            inputs.append(f"Segment: {slots.segment}")
        if slots.adr:
            inputs.append(f"Target ADR: ${slots.adr}")
        if slots.developer_goal:
            inputs.append(f"Goal: {slots.developer_goal}")

        inputs_str = "\n".join(inputs) if inputs else "No specific inputs provided yet."

        return f"""You are a hospitality brand strategist. Help create a unique hotel brand concept.

Inputs:
{inputs_str}

Market context:
{context}

User request: {message}

Provide creative, specific brand direction including:
- Brand positioning and thesis
- Signature experiences
- Design direction
- Target guest personas

Be bold and distinctive. Make it memorable."""

    def _fallback_response(
        self,
        message: str,
        rag_result: RAGResult,
        mode: ChatMode,
    ) -> dict[str, Any]:
        """Generate fallback response without LLM.

        Args:
            message: User message
            rag_result: Retrieved context
            mode: Current mode

        Returns:
            Response dict
        """
        if not rag_result.chunks:
            text = "I don't have enough context to provide a detailed answer. Could you provide more details about what you're looking for?"
        else:
            # Summarize retrieved chunks
            chunks_text = "\n\n".join(
                f"• {chunk.text[:200]}..."
                for chunk in rag_result.chunks[:3]
            )
            text = f"Based on available data:\n\n{chunks_text}\n\nFor more detailed insights, please check the Social Pulse or Hotelier Bets sections."

        return {"text": text, "raw": None}

    def _create_artifact(
        self,
        response: dict,
        rag_result: RAGResult,
        mode: ChatMode,
    ) -> ChatArtifact | None:
        """Create structured artifact from response.

        Args:
            response: Generated response
            rag_result: RAG results
            mode: Current mode

        Returns:
            ChatArtifact or None
        """
        if not self.belief_manager.should_save_artifact():
            return None

        slots = self.belief_manager.belief.slots

        if mode == ChatMode.INSIGHT:
            # Create InsightBrief artifact
            artifact_data = InsightBrief(
                topic=self._extract_topic(response["text"]),
                location=slots.location,
                key_signals=[
                    KeySignal(
                        signal=chunk.text[:100],
                        why_it_matters="Identified from market data",
                        confidence=chunk.posterior,
                        evidence=[Evidence(
                            chunk_id=chunk.id,
                            text_snippet=chunk.text[:200],
                            source_type=chunk.source_type,
                            confidence=chunk.posterior,
                        )]
                    )
                    for chunk in rag_result.chunks[:3]
                ],
                confidence=rag_result.top_posterior,
                sources_used=rag_result.sources_used,
            )

            return ChatArtifact(
                id=str(uuid.uuid4()),
                artifact_type="insight_brief_v1",
                data=artifact_data.model_dump(),
                sources=[c.id for c in rag_result.chunks],
                confidence=rag_result.top_posterior,
            )

        # For other modes, return None for now
        return None

    def _extract_topic(self, text: str) -> str:
        """Extract topic from response text.

        Args:
            text: Response text

        Returns:
            Topic string
        """
        # Simple extraction - first sentence or first 50 chars
        first_line = text.split("\n")[0]
        if len(first_line) > 100:
            return first_line[:100] + "..."
        return first_line

    def get_conversation_history(self) -> list[dict]:
        """Get conversation history.

        Returns:
            List of message dicts
        """
        return [msg.model_dump() for msg in self._messages]

    def get_artifacts(self) -> list[dict]:
        """Get generated artifacts.

        Returns:
            List of artifact dicts
        """
        return [art.model_dump() for art in self._artifacts]


# Singleton instance
_chat_service: ChatService | None = None


def get_chat_service() -> ChatService:
    """Get singleton chat service instance."""
    global _chat_service
    if _chat_service is None:
        # ChatService now auto-initializes LLM and embeddings
        _chat_service = ChatService()
    return _chat_service
