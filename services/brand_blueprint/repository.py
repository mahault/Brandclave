"""Repository for brand blueprint database operations."""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import desc
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import BrandBlueprintModel
from .schemas import (
    BrandBlueprintFull,
    BlueprintInputs,
    AlternateBrandNames,
    UnmetDesireSolved,
    TokenUsage,
)
from services.chat.schemas import (
    SignatureExperience,
    GuestJourney,
    GuestPersona,
    FnBConcept,
)

logger = logging.getLogger(__name__)


class BlueprintRepository:
    """Repository for brand blueprint CRUD operations."""

    def __init__(self, session: Session | None = None):
        """Initialize repository.

        Args:
            session: Optional SQLAlchemy session. If not provided, will use get_db_session.
        """
        self._session = session

    @property
    def session(self) -> Session:
        """Get or create a database session."""
        if self._session is None:
            self._session = get_db()
        return self._session

    def save(self, blueprint: BrandBlueprintFull) -> str:
        """Save a blueprint to the database.

        Args:
            blueprint: The blueprint to save

        Returns:
            The saved blueprint ID
        """
        model = BrandBlueprintModel(
            # Inputs
            location=blueprint.inputs.location,
            segment=blueprint.inputs.segment,
            adr=blueprint.inputs.adr,
            rooms=blueprint.inputs.rooms,
            developer_goal=blueprint.inputs.developer_goal,
            source_trend_id=blueprint.inputs.source_trend_id,
            profile_data_json=blueprint.inputs.profile_data,
            # Stage 1: Foundation
            brand_name_primary=blueprint.brand_names.primary,
            brand_name_alt_1=blueprint.brand_names.alternate_1,
            brand_name_alt_2=blueprint.brand_names.alternate_2,
            one_liner=blueprint.one_liner,
            thesis=blueprint.thesis,
            # Stage 2: Strategic
            pillars=blueprint.pillars,
            positioning_statement=blueprint.positioning_statement,
            unmet_desires_solved=[d.model_dump() for d in blueprint.unmet_desires_solved],
            # Stage 3: Experience
            guest_personas=[p.model_dump() for p in blueprint.guest_personas],
            signature_experiences=[e.model_dump() for e in blueprint.signature_experiences],
            guest_journey=blueprint.guest_journey.model_dump() if blueprint.guest_journey else None,
            # Stage 4: Atmosphere
            design_direction=blueprint.design_direction,
            fnb_concepts=[f.model_dump() for f in blueprint.fnb_concepts],
            revenue_logic=blueprint.revenue_logic,
            # Stage 5: Summary
            investor_summary=blueprint.investor_summary,
            # Metadata
            status=blueprint.status,
            confidence=blueprint.confidence,
            warnings=blueprint.warnings,
            input_tokens=blueprint.token_usage.input_tokens,
            output_tokens=blueprint.token_usage.output_tokens,
        )

        self.session.add(model)
        self.session.commit()

        logger.info(f"Saved blueprint {model.id}")
        return model.id

    def get(self, blueprint_id: str) -> BrandBlueprintFull | None:
        """Get a blueprint by ID.

        Args:
            blueprint_id: The blueprint ID

        Returns:
            The blueprint or None if not found
        """
        model = self.session.query(BrandBlueprintModel).filter(
            BrandBlueprintModel.id == blueprint_id
        ).first()

        if model is None:
            return None

        return self._model_to_schema(model)

    def list(
        self,
        limit: int = 20,
        offset: int = 0,
        location: str | None = None,
        segment: str | None = None,
    ) -> tuple[list[BrandBlueprintFull], int]:
        """List blueprints with optional filters.

        Args:
            limit: Maximum number to return
            offset: Number to skip
            location: Optional location filter
            segment: Optional segment filter

        Returns:
            Tuple of (blueprints, total_count)
        """
        query = self.session.query(BrandBlueprintModel)

        if location:
            query = query.filter(BrandBlueprintModel.location.ilike(f"%{location}%"))
        if segment:
            query = query.filter(BrandBlueprintModel.segment == segment)

        total = query.count()

        models = query.order_by(
            desc(BrandBlueprintModel.created_at)
        ).offset(offset).limit(limit).all()

        blueprints = [self._model_to_schema(m) for m in models]
        return blueprints, total

    def delete(self, blueprint_id: str) -> bool:
        """Delete a blueprint by ID.

        Args:
            blueprint_id: The blueprint ID

        Returns:
            True if deleted, False if not found
        """
        model = self.session.query(BrandBlueprintModel).filter(
            BrandBlueprintModel.id == blueprint_id
        ).first()

        if model is None:
            return False

        self.session.delete(model)
        self.session.commit()

        logger.info(f"Deleted blueprint {blueprint_id}")
        return True

    def _model_to_schema(self, model: BrandBlueprintModel) -> BrandBlueprintFull:
        """Convert database model to Pydantic schema.

        Args:
            model: The database model

        Returns:
            The Pydantic schema
        """
        # Reconstruct inputs
        inputs = BlueprintInputs(
            location=model.location,
            segment=model.segment,
            adr=model.adr,
            rooms=model.rooms,
            developer_goal=model.developer_goal,
            source_trend_id=model.source_trend_id,
            profile_data=model.profile_data_json,
        )

        # Reconstruct brand names
        brand_names = AlternateBrandNames(
            primary=model.brand_name_primary,
            alternate_1=model.brand_name_alt_1 or "",
            alternate_2=model.brand_name_alt_2 or "",
        )

        # Reconstruct unmet desires
        unmet_desires = [
            UnmetDesireSolved(**d)
            for d in (model.unmet_desires_solved or [])
        ]

        # Reconstruct personas
        guest_personas = [
            GuestPersona(**p)
            for p in (model.guest_personas or [])
        ]

        # Reconstruct experiences
        signature_experiences = [
            SignatureExperience(**e)
            for e in (model.signature_experiences or [])
        ]

        # Reconstruct journey
        guest_journey = GuestJourney(**model.guest_journey) if model.guest_journey else None

        # Reconstruct F&B
        fnb_concepts = [
            FnBConcept(**f)
            for f in (model.fnb_concepts or [])
        ]

        # Reconstruct token usage
        token_usage = TokenUsage(
            input_tokens=model.input_tokens,
            output_tokens=model.output_tokens,
            total_tokens=model.input_tokens + model.output_tokens,
            estimated_cost_usd=0.0,  # Not stored
        )

        return BrandBlueprintFull(
            id=model.id,
            inputs=inputs,
            brand_names=brand_names,
            one_liner=model.one_liner,
            thesis=model.thesis,
            pillars=model.pillars or [],
            positioning_statement=model.positioning_statement,
            unmet_desires_solved=unmet_desires,
            guest_personas=guest_personas,
            signature_experiences=signature_experiences,
            guest_journey=guest_journey,
            design_direction=model.design_direction,
            fnb_concepts=fnb_concepts,
            revenue_logic=model.revenue_logic,
            investor_summary=model.investor_summary,
            status=model.status,
            confidence=model.confidence,
            warnings=model.warnings or [],
            token_usage=token_usage,
            generated_at=model.created_at,
        )
