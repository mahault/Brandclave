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
    CONFIDENCE_THRESHOLD = 0.6       # Min confidence to proceed without asking
    ENTROPY_THRESHOLD = 0.7          # High entropy = need more info
    COMMIT_THRESHOLD = 0.6           # When to suggest "Send to Build a Brand"

    # Required slots per mode
    REQUIRED_SLOTS = {
        ChatMode.INSIGHT: [],  # No required slots
        ChatMode.BRAND_BUILD: ["location", "segment"],
        ChatMode.DEMAND_SCAN: ["url"],
    }

    # Clarifying questions per slot
    CLARIFYING_QUESTIONS = {
        "location": "What city or location are you interested in?",
        "segment": "What segment are you targeting? (luxury, boutique, lifestyle, budget, wellness, business)",
        "url": "Can you share the property website URL you'd like me to analyze?",
        "adr": "What's your target ADR (average daily rate)?",
        "developer_goal": "What's the main goal for this development?",
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

        # Rule-based policy

        # 1. If low confidence on mode, ask clarifying question
        if confidence < self.CONFIDENCE_THRESHOLD and self._turn_count < 3:
            return DialogueAction.ASK_CLARIFYING_Q, {
                "reason": "low_mode_confidence",
                "question": self._get_mode_clarification_question(),
            }

        # 2. If missing critical slots, ask for them
        if missing_slots:
            slot = missing_slots[0]  # Ask one at a time
            return DialogueAction.ASK_CLARIFYING_Q, {
                "reason": "missing_slot",
                "slot": slot,
                "question": self.CLARIFYING_QUESTIONS.get(slot, f"What is the {slot}?"),
            }

        # 3. If retrieval entropy is high, try to retrieve more
        if self.belief.is_high_entropy(self.ENTROPY_THRESHOLD):
            if self.belief.retrieval.chunks_retrieved < 5:
                return DialogueAction.RETRIEVE_MORE, {
                    "reason": "high_entropy",
                    "entropy": self.belief.retrieval.entropy,
                }

        # 4. If in commit stage and mode is insight, suggest Build a Brand
        if (
            mode == ChatMode.INSIGHT and
            self.belief.stage["commit"] > self.COMMIT_THRESHOLD and
            self._turn_count >= 2
        ):
            return DialogueAction.SUGGEST_BUILD_A_BRAND, {
                "reason": "ready_to_commit",
                "prefill": self._get_brand_prefill(),
            }

        # 5. Default: answer now
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
            return "Are you looking for market trends and insights, or would you like to analyze a specific property or build a brand concept?"
        elif mode == ChatMode.BRAND_BUILD:
            return "Would you like me to help create a new hotel brand concept? I'll need to know the location and target segment."
        else:
            return "Would you like me to analyze a specific property website? Please share the URL."

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
