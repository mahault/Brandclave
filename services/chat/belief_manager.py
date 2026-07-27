"""Belief Manager - POMDP-lite dialogue control for chat."""

import logging
from enum import Enum
from typing import Any

from services.chat.schemas import (
    BeliefState,
    ChatMode,
    RetrievalState,
    RouterOutput,
    SlotValues,
)

logger = logging.getLogger(__name__)


class DialogueAction(str, Enum):
    """Actions the dialogue manager can take."""
    ASK_CLARIFYING_Q = "ask_clarifying_q"
    RETRIEVE_MORE = "retrieve_more"
    ANSWER_NOW = "answer_now"
    SUGGEST_BUILD_A_BRAND = "suggest_build_a_brand"
    SAVE_ARTIFACT = "save_artifact"


class BeliefManager:
    """POMDP-lite belief state manager for dialogue control.

    Tracks:
    - Mode probabilities (insight/brand_build/demand_scan)
    - Slot fill status (location, segment, ADR, URL)
    - Retrieval confidence and entropy
    - User stage (exploring vs committing)

    Uses rule-based policy (upgradable to learned policy later).
    """

    # Thresholds for policy decisions
    CONFIDENCE_THRESHOLD = 0.35      # Min confidence to proceed without asking (lowered)
    ENTROPY_THRESHOLD = 0.8          # High entropy = need more info (relaxed)
    COMMIT_THRESHOLD = 0.6           # When to suggest "Send to Build a Brand"
    MAX_CLARIFY_TURNS = 1            # Max turns to ask clarifying questions

    # Required slots per mode
    REQUIRED_SLOTS = {
        ChatMode.INSIGHT: [],  # No required slots
        ChatMode.BRAND_BUILD: ["location", "segment"],
        ChatMode.DEMAND_SCAN: ["url"],
    }

    # Clarifying questions per slot - detailed and helpful
    CLARIFYING_QUESTIONS = {
        "location": (
            "Which city or region are you focusing on? For example: Miami, Lisbon, Tokyo, DC, Bali... "
            "This helps me find relevant market trends and competitor data for that area."
        ),
        "segment": (
            "What hotel segment fits your vision? Options include:\n"
            "• **Luxury** - 5-star, high-end experiences ($400+ ADR)\n"
            "• **Lifestyle** - Design-forward, millennial/Gen-Z focused ($150-300 ADR)\n"
            "• **Boutique** - Independent, unique character properties\n"
            "• **Wellness** - Spa, retreat, health-focused\n"
            "• **Business** - Corporate, conference-oriented"
        ),
        "url": (
            "Please share the property's website URL (e.g., https://hotelname.com). "
            "I'll analyze their positioning, messaging, and identify gaps versus market demand."
        ),
        "adr": (
            "What's your target ADR (Average Daily Rate)? This helps me benchmark against "
            "competitors and identify the right positioning. For reference:\n"
            "• Budget: $50-150\n• Midscale: $150-250\n• Upscale: $250-400\n• Luxury: $400+"
        ),
        "developer_goal": (
            "What's your primary goal? For example:\n"
            "• Maximize RevPAR in a competitive market\n"
            "• Create a differentiated brand experience\n"
            "• Target an underserved guest segment\n"
            "• Reposition an existing property"
        ),
    }

    def __init__(self):
        """Initialize belief manager."""
        self.belief = BeliefState()
        self._turn_count = 0

    def reset(self) -> None:
        """Reset belief state for new conversation."""
        self.belief = BeliefState()
        self._turn_count = 0

    def update_from_router(self, router_output: RouterOutput) -> None:
        """Update belief state from router output.

        Args:
            router_output: Output from ModeRouter
        """
        # Update mode probabilities
        self.belief.mode_probs = {
            "insight": router_output.p_insight,
            "brand_build": router_output.p_brand_build,
            "demand_scan": router_output.p_demand_scan,
        }

        # Merge slot values (keep existing if new is None)
        current = self.belief.slots
        detected = router_output.slots_detected

        self.belief.slots = SlotValues(
            location=detected.location or current.location,
            segment=detected.segment or current.segment,
            adr=detected.adr or current.adr,
            url=detected.url or current.url,
            developer_goal=detected.developer_goal or current.developer_goal,
        )

        self._turn_count += 1
        self.belief.conversation_turns = self._turn_count

    def update_from_retrieval(
        self,
        top_posterior: float,
        entropy: float,
        chunks_retrieved: int,
    ) -> None:
        """Update belief state from retrieval results.

        Args:
            top_posterior: Highest posterior relevance
            entropy: Retrieval entropy (uncertainty)
            chunks_retrieved: Number of relevant chunks found
        """
        self.belief.retrieval = RetrievalState(
            top_posterior=top_posterior,
            entropy=entropy,
            chunks_retrieved=chunks_retrieved,
        )

        # Update stage based on retrieval quality
        if top_posterior > 0.7 and entropy < 0.5:
            # Good retrieval = user might be ready to commit
            self.belief.stage["commit"] = min(0.8, self.belief.stage["commit"] + 0.1)
            self.belief.stage["explore"] = 1.0 - self.belief.stage["commit"]

    def update_stage(self, user_message: str) -> None:
        """Update user stage based on message content.

        Args:
            user_message: User's message
        """
        message_lower = user_message.lower()

        # Commit signals
        commit_signals = [
            "build", "create", "design", "let's do", "go ahead",
            "sounds good", "yes", "make", "generate", "ready",
        ]

        # Explore signals
        explore_signals = [
            "what", "how", "tell me", "show me", "explain",
            "options", "alternatives", "other", "different",
        ]

        commit_score = sum(1 for s in commit_signals if s in message_lower)
        explore_score = sum(1 for s in explore_signals if s in message_lower)

        # Bayesian-ish update
        if commit_score > explore_score:
            self.belief.stage["commit"] = min(0.9, self.belief.stage["commit"] + 0.15)
        elif explore_score > commit_score:
            self.belief.stage["explore"] = min(0.9, self.belief.stage["explore"] + 0.1)

        # Normalize
        total = self.belief.stage["commit"] + self.belief.stage["explore"]
        self.belief.stage["commit"] /= total
        self.belief.stage["explore"] /= total

    def select_action(self) -> tuple[DialogueAction, dict[str, Any]]:
        """Select next action based on current belief state.

        Returns:
            Tuple of (action, metadata)
        """
        mode = self.belief.get_dominant_mode()
        confidence = self.belief.get_confidence()

        # Check for missing required slots
        missing_slots = self._get_missing_slots(mode)

        # Rule-based policy - PRIORITIZE ANSWERING over asking questions

        # 1. If this is the first turn and we have ANY signal, just answer
        if self._turn_count <= 1:
            # First turn: always try to answer, don't ask clarifying questions
            return DialogueAction.ANSWER_NOW, {
                "reason": "first_turn_answer",
                "mode": mode.value,
                "confidence": confidence,
            }

        # 2. Only ask for clarification if VERY low confidence AND haven't asked before
        if confidence < self.CONFIDENCE_THRESHOLD and self._turn_count < self.MAX_CLARIFY_TURNS + 1:
            return DialogueAction.ASK_CLARIFYING_Q, {
                "reason": "low_mode_confidence",
                "question": self._get_mode_clarification_question(),
            }

        # 3. For brand_build/demand_scan modes, gently prompt for slots but don't block
        # Only ask if we're clearly in that mode AND haven't already asked
        if missing_slots and confidence > 0.5 and self._turn_count < 3:
            slot = missing_slots[0]
            # Frame as helpful, not blocking
            return DialogueAction.ANSWER_NOW, {
                "reason": "will_answer_and_suggest_slot",
                "mode": mode.value,
                "suggest_slot": slot,
            }

        # 4. If retrieval entropy is high, try to retrieve more (but only once)
        if self.belief.is_high_entropy(self.ENTROPY_THRESHOLD):
            if self.belief.retrieval.chunks_retrieved < 3 and self._turn_count <= 2:
                return DialogueAction.RETRIEVE_MORE, {
                    "reason": "high_entropy",
                    "entropy": self.belief.retrieval.entropy,
                }

        # 5. If in commit stage and mode is insight, suggest Build a Brand
        if (
            mode == ChatMode.INSIGHT and
            self.belief.stage["commit"] > self.COMMIT_THRESHOLD and
            self._turn_count >= 2
        ):
            return DialogueAction.SUGGEST_BUILD_A_BRAND, {
                "reason": "ready_to_commit",
                "prefill": self._get_brand_prefill(),
            }

        # 6. Default: answer now (this should be the most common path)
        return DialogueAction.ANSWER_NOW, {
            "reason": "ready_to_answer",
            "mode": mode.value,
            "confidence": confidence,
        }

    def _get_missing_slots(self, mode: ChatMode) -> list[str]:
        """Get list of missing required slots for mode.

        Args:
            mode: Current mode

        Returns:
            List of missing slot names
        """
        required = self.REQUIRED_SLOTS.get(mode, [])
        slots = self.belief.slots

        missing = []
        for slot in required:
            value = getattr(slots, slot, None)
            if value is None:
                missing.append(slot)

        return missing

    def _get_mode_clarification_question(self) -> str:
        """Get a question to clarify user intent."""
        mode = self.belief.get_dominant_mode()

        if mode == ChatMode.INSIGHT:
            return (
                "I can help in several ways:\n\n"
                "**1. Market Insights** - Tell me a location (e.g., 'trends in Miami') and I'll share:\n"
                "   • Emerging traveler preferences\n"
                "   • White space opportunities\n"
                "   • Competitive landscape signals\n\n"
                "**2. Brand Building** - Say 'help me build a brand' with a location + segment\n\n"
                "**3. Property Analysis** - Share a hotel URL to analyze positioning gaps\n\n"
                "What would you like to explore?"
            )
        elif mode == ChatMode.BRAND_BUILD:
            return (
                "Great! To build a compelling brand concept, I'll need:\n\n"
                "• **Location** - Which city/market? (e.g., DC, Lisbon, Bali)\n"
                "• **Segment** - Luxury, lifestyle, boutique, wellness, or budget?\n"
                "• **Target ADR** (optional) - What price point are you targeting?\n\n"
                "For example: 'Build a lifestyle hotel brand in Austin targeting $200 ADR'"
            )
        else:
            return (
                "I can analyze any hotel property. Please share:\n\n"
                "• **Website URL** - The property's main website\n\n"
                "I'll evaluate their positioning, identify gaps versus market demand, "
                "and suggest opportunity areas."
            )

    def _get_brand_prefill(self) -> dict[str, Any]:
        """Get prefill data for Build a Brand action."""
        slots = self.belief.slots
        return {
            "location": slots.location,
            "segment": slots.segment,
            "adr": slots.adr,
            "developer_goal": slots.developer_goal,
        }

    def get_state_summary(self) -> dict[str, Any]:
        """Get a summary of current belief state.

        Returns:
            Dict with state information
        """
        mode = self.belief.get_dominant_mode()
        return {
            "dominant_mode": mode.value,
            "mode_confidence": self.belief.get_confidence(),
            "mode_probs": self.belief.mode_probs,
            "slots": {
                "location": self.belief.slots.location,
                "segment": self.belief.slots.segment,
                "adr": self.belief.slots.adr,
                "url": self.belief.slots.url,
            },
            "retrieval": {
                "top_posterior": self.belief.retrieval.top_posterior,
                "entropy": self.belief.retrieval.entropy,
                "chunks": self.belief.retrieval.chunks_retrieved,
            },
            "stage": self.belief.stage,
            "turns": self._turn_count,
            "missing_slots": self._get_missing_slots(mode),
        }

    def should_save_artifact(self) -> bool:
        """Check if we should save the generated artifact.

        Returns:
            True if artifact should be saved
        """
        # Save if good retrieval and user seems committed
        return (
            self.belief.retrieval.top_posterior > 0.5 and
            self.belief.stage["commit"] > 0.4
        )

    def format_confidence_badge(self) -> str:
        """Format confidence as user-friendly badge.

        Returns:
            "High", "Medium", or "Low"
        """
        posterior = self.belief.retrieval.top_posterior
        if posterior > 0.7:
            return "High"
        elif posterior > 0.4:
            return "Medium"
        else:
            return "Low"
