"""Data models for BrandClave Aggregator."""

from data_models.hotelier_move import HotelierMove, HotelierMoveCreate, MoveType
from data_models.property_features import (
    PriceSegment,
    PropertyFeatures,
    PropertyFeaturesCreate,
    PropertyType,
)
from data_models.raw_content import RawContent, RawContentCreate, SourceType
from data_models.trend_signal import AudienceSegment, TrendSignal, TrendSignalCreate

__all__ = [
    "RawContent",
    "RawContentCreate",
    "SourceType",
    "TrendSignal",
    "TrendSignalCreate",
    "AudienceSegment",
    "HotelierMove",
    "HotelierMoveCreate",
    "MoveType",
    "PropertyFeatures",
    "PropertyFeaturesCreate",
    "PropertyType",
    "PriceSegment",
]
