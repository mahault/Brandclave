"""Chat API routes - BrandClave Chat interface."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response models
class ChatRequest(BaseModel):
    """Chat request body."""
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = None
    project_id: str | None = None


class ChatResponse(BaseModel):
    """Chat response body."""
    conversation_id: str
    response: str
    mode: str
    confidence: str  # "High", "Medium", "Low"
    sources_used: int
    artifact: dict | None = None
    suggested_action: dict | None = None
    state: dict | None = None


class ConversationHistoryResponse(BaseModel):
    """Conversation history response."""
    conversation_id: str
    messages: list[dict]
    artifacts: list[dict]


# In-memory conversation store (replace with DB in production)
_conversations: dict[str, Any] = {}


def get_or_create_service(conversation_id: str | None = None):
    """Get or create a chat service for a conversation."""
    from services.chat.service import ChatService

    if conversation_id and conversation_id in _conversations:
        return _conversations[conversation_id]

    # Create new service
    embedding_fn = None
    try:
        from processing.embeddings import get_embedding
        embedding_fn = get_embedding
    except ImportError:
        logger.warning("Embedding function not available")

    service = ChatService(embedding_fn=embedding_fn)
    conv_id = service.new_conversation()

    _conversations[conv_id] = service
    return service


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message.

    Send a message and receive an AI-powered response with
    market insights, trend analysis, or brand recommendations.

    The system automatically detects your intent:
    - **Insight mode**: Market trends, opportunities, forecasts
    - **Brand Build mode**: Create hotel brand concepts
    - **Demand Scan mode**: Analyze property websites
    """
    try:
        # Get or create service
        service = get_or_create_service(request.conversation_id)

        # Process message
        result = await service.chat(
            message=request.message,
            project_id=request.project_id,
        )

        return ChatResponse(
            conversation_id=result["conversation_id"],
            response=result["response"],
            mode=result["mode"],
            confidence=result["confidence"],
            sources_used=result["sources_used"],
            artifact=result.get("artifact"),
            suggested_action=result.get("suggested_action"),
            state=result.get("state"),
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/{conversation_id}/history", response_model=ConversationHistoryResponse)
async def get_history(conversation_id: str) -> ConversationHistoryResponse:
    """Get conversation history.

    Returns all messages and artifacts from a conversation.
    """
    if conversation_id not in _conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    service = _conversations[conversation_id]

    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        messages=service.get_conversation_history(),
        artifacts=service.get_artifacts(),
    )


@router.delete("/chat/{conversation_id}")
async def delete_conversation(conversation_id: str) -> dict:
    """Delete a conversation.

    Removes the conversation from memory.
    """
    if conversation_id in _conversations:
        del _conversations[conversation_id]
        return {"status": "deleted", "conversation_id": conversation_id}

    raise HTTPException(status_code=404, detail="Conversation not found")


@router.get("/chat/conversations")
async def list_conversations() -> dict:
    """List active conversations.

    Returns IDs of all active conversations.
    """
    return {
        "conversations": list(_conversations.keys()),
        "count": len(_conversations),
    }


# Route info for router output testing
@router.post("/chat/route")
async def route_message(request: ChatRequest) -> dict:
    """Test the mode router without generating a response.

    Useful for debugging intent classification.
    """
    from services.chat.mode_router import ModeRouter

    router_instance = ModeRouter()
    result = router_instance.route(request.message)

    return {
        "message": request.message,
        "p_insight": result.p_insight,
        "p_brand_build": result.p_brand_build,
        "p_demand_scan": result.p_demand_scan,
        "confidence": result.confidence,
        "predicted_mode": result.get_mode().value,
        "slots_detected": result.slots_detected.model_dump(),
        "slots_needed": result.slots_needed,
    }
