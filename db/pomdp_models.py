"""Database models for POMDP state persistence."""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON, Index
from sqlalchemy.orm import declarative_base

from db.database import Base


class POMDPStateLog(Base):
    """Track POMDP state evolution for debugging/analysis."""

    __tablename__ = "pomdp_state_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    pomdp_type = Column(String, index=True)  # scraping, clustering, extraction, coordinator, user
    state_json = Column(JSON)  # Serialized belief state
    action_taken = Column(String)
    action_index = Column(Integer, nullable=True)
    reward = Column(Float, nullable=True)
    free_energy = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_pomdp_logs_type_time", "pomdp_type", "timestamp"),
    )


class ScrapingPOMDPState(Base):
    """Persistent state for scraping POMDP."""

    __tablename__ = "scraping_pomdp_state"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_beliefs_json = Column(JSON)  # Beliefs about each source's productivity
    A_matrix_json = Column(JSON, nullable=True)  # Learned likelihood matrix
    last_actions_json = Column(JSON)  # Recent action history
    cumulative_reward = Column(Float, default=0.0)
    total_observations = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow)


class ClusteringPOMDPState(Base):
    """Persistent state for clustering POMDP."""

    __tablename__ = "clustering_pomdp_state"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    param_beliefs_json = Column(JSON)  # Beliefs about parameter effectiveness
    A_matrix_json = Column(JSON, nullable=True)
    quality_history_json = Column(JSON)  # History of clustering results
    total_clusterings = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow)


class ExtractionPOMDPState(Base):
    """Persistent state for move extraction POMDP."""

    __tablename__ = "extraction_pomdp_state"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    method_beliefs_json = Column(JSON)  # Beliefs about method effectiveness
    A_matrix_json = Column(JSON, nullable=True)
    extraction_history_json = Column(JSON)
    total_extractions = Column(Integer, default=0)
    llm_calls_saved = Column(Integer, default=0)  # Track cost savings
    updated_at = Column(DateTime, default=datetime.utcnow)


class CoordinatorPOMDPState(Base):
    """Persistent state for cross-component coordinator POMDP."""

    __tablename__ = "coordinator_pomdp_state"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    component_beliefs_json = Column(JSON)  # Beliefs about component states
    cross_signals_json = Column(JSON)  # Detected cross-component correlations
    A_matrix_json = Column(JSON, nullable=True)
    action_history_json = Column(JSON)
    opportunities_surfaced = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow)


class UserPOMDPState(Base):
    """Per-user POMDP state for personalization."""

    __tablename__ = "user_pomdp_states"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True, unique=True)
    beliefs_json = Column(JSON)  # User preference beliefs
    A_matrix_json = Column(JSON, nullable=True)  # Learned user model
    interaction_count = Column(Integer, default=0)
    last_interaction_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class UserInteraction(Base):
    """Track user interactions for POMDP learning."""

    __tablename__ = "user_interactions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True)
    component = Column(String)  # social_pulse, demand_scan, hotelier_bets, city_desires
    action_type = Column(String)  # view, click, filter, bookmark, search
    item_id = Column(String, nullable=True)  # ID of item interacted with
    item_type = Column(String, nullable=True)  # trend, move, property, etc.
    metadata_json = Column(JSON, nullable=True)  # Additional context
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_user_interactions_user_time", "user_id", "timestamp"),
        Index("ix_user_interactions_component", "component"),
    )
