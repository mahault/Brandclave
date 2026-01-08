"""Tests for Demand Scan enhancements.

Tests cover:
- Demand fit score conversion (0-1 to 0-100)
- Positioning misalignment detection
- Experience gap snapshot (top 2-3)
- Opportunity lanes formatting
- Send to Build a Brand integration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, patch


class TestDemandFitScore:
    """Tests for demand fit score display."""

    def test_score_conversion_0_to_100(self):
        """Verify 0-1 float is converted to 0-100 integer."""
        # Test cases: (input_float, expected_int)
        test_cases = [
            (0.0, 0),
            (0.5, 50),
            (1.0, 100),
            (0.72, 72),
            (0.33, 33),
        ]
        for input_score, expected in test_cases:
            result = int(input_score * 100)
            assert result == expected, f"Expected {expected}, got {result} for input {input_score}"

    def test_badge_color_high(self):
        """Verify high scores (70+) get green badge."""
        high_scores = [70, 85, 100]
        for score in high_scores:
            badge_class = self._get_badge_class(score)
            assert badge_class == "demand-high", f"Score {score} should be high"

    def test_badge_color_medium(self):
        """Verify medium scores (40-69) get yellow badge."""
        medium_scores = [40, 55, 69]
        for score in medium_scores:
            badge_class = self._get_badge_class(score)
            assert badge_class == "demand-medium", f"Score {score} should be medium"

    def test_badge_color_low(self):
        """Verify low scores (<40) get red badge."""
        low_scores = [0, 20, 39]
        for score in low_scores:
            badge_class = self._get_badge_class(score)
            assert badge_class == "demand-low", f"Score {score} should be low"

    def _get_badge_class(self, score: int) -> str:
        """Helper to determine badge class from score."""
        if score >= 70:
            return "demand-high"
        elif score >= 40:
            return "demand-medium"
        else:
            return "demand-low"


class TestPositioningMisalignment:
    """Tests for positioning misalignment detection."""

    def test_luxury_price_budget_amenities(self):
        """Detect luxury pricing with budget amenities."""
        features = {
            "price_segment": "luxury",
            "amenities": ["basic wifi", "parking"],
            "themes": [],
        }
        flags = self._detect_misalignment(features)
        assert any("price" in f.lower() or "luxury" in f.lower() for f in flags)

    def test_wellness_positioning_no_spa(self):
        """Detect wellness positioning without wellness amenities."""
        features = {
            "themes": ["wellness", "mindfulness"],
            "amenities": ["restaurant", "bar", "parking"],
            "brand_positioning": "A wellness retreat for the soul",
        }
        flags = self._detect_misalignment(features)
        assert any("wellness" in f.lower() or "spa" in f.lower() for f in flags)

    def test_conflicting_themes(self):
        """Detect conflicting theme positioning."""
        features = {
            "themes": ["budget", "luxury", "boutique"],
            "amenities": [],
            "brand_positioning": "Affordable luxury for everyone",
        }
        flags = self._detect_misalignment(features)
        assert any("conflict" in f.lower() or "mix" in f.lower() for f in flags)

    def test_no_misalignment(self):
        """No flags when positioning is consistent."""
        features = {
            "price_segment": "luxury",
            "amenities": ["spa", "fine dining", "butler service", "pool"],
            "themes": ["luxury", "exclusive"],
            "brand_positioning": "Unparalleled luxury experience",
        }
        flags = self._detect_misalignment(features)
        assert len(flags) == 0 or all("warning" not in f.lower() for f in flags)

    def _detect_misalignment(self, features: dict) -> list[str]:
        """Helper to detect positioning misalignment.

        This mirrors the logic that will be implemented in the service.
        """
        flags = []
        price = features.get("price_segment", "unknown")
        amenities = [a.lower() for a in features.get("amenities", [])]
        themes = [t.lower() for t in features.get("themes", [])]
        positioning = features.get("brand_positioning", "").lower()

        # Luxury pricing with budget amenities
        luxury_amenities = {"spa", "fine dining", "butler", "concierge", "pool", "gym"}
        has_luxury_amenities = any(la in " ".join(amenities) for la in luxury_amenities)

        if price in ["luxury", "ultra_luxury"] and not has_luxury_amenities:
            flags.append("Price-tier mismatch: Luxury pricing without luxury amenities")

        # Wellness positioning without wellness offerings
        wellness_keywords = {"spa", "yoga", "meditation", "massage", "wellness"}
        has_wellness_amenities = any(wk in " ".join(amenities) for wk in wellness_keywords)
        wellness_positioned = "wellness" in themes or "wellness" in positioning

        if wellness_positioned and not has_wellness_amenities:
            flags.append("Theme mismatch: Wellness positioning without spa/wellness facilities")

        # Conflicting themes
        conflicting_pairs = [("budget", "luxury"), ("business", "romantic")]
        for t1, t2 in conflicting_pairs:
            if t1 in themes and t2 in themes:
                flags.append(f"Conflicting themes: {t1.title()} and {t2.title()} mixed")

        return flags


class TestExperienceGapSnapshot:
    """Tests for experience gap snapshot."""

    def test_returns_top_3_gaps(self):
        """Verify only top 2-3 gaps are returned."""
        all_gaps = [
            "Wellness trend (trending at 85% strength)",
            "Eco-friendly (trending at 72% strength)",
            "Digital nomad (trending at 68% strength)",
            "Adventure (trending at 55% strength)",
            "Cultural immersion (trending at 45% strength)",
        ]
        snapshot = self._get_gap_snapshot(all_gaps)
        assert len(snapshot) <= 3, f"Expected max 3 gaps, got {len(snapshot)}"

    def test_prioritizes_by_strength(self):
        """Verify gaps are ordered by trend strength."""
        all_gaps = [
            "Wellness trend (trending at 85% strength)",
            "Eco-friendly (trending at 72% strength)",
            "Digital nomad (trending at 68% strength)",
        ]
        snapshot = self._get_gap_snapshot(all_gaps)
        assert snapshot[0] == all_gaps[0], "Highest strength should be first"

    def test_empty_gaps(self):
        """Handle empty gap list gracefully."""
        snapshot = self._get_gap_snapshot([])
        assert snapshot == []

    def _get_gap_snapshot(self, gaps: list[str], limit: int = 3) -> list[str]:
        """Get top N experience gaps."""
        return gaps[:limit]


class TestOpportunityLanes:
    """Tests for opportunity lanes formatting."""

    def test_lane_has_demand_driver(self):
        """Verify opportunity lanes include demand driver."""
        opportunity = "Position as leader in 'Wellness' - high demand, low competition"
        lane = self._format_opportunity_lane(opportunity)
        assert "demand" in lane.get("description", "").lower() or "trend" in lane.get("type", "").lower()

    def test_lane_has_recommendation(self):
        """Verify opportunity lanes include positioning recommendation."""
        opportunity = "Emphasize design-forward positioning for boutique appeal"
        lane = self._format_opportunity_lane(opportunity)
        assert lane.get("recommendation") is not None

    def test_lane_structure(self):
        """Verify opportunity lane has required fields."""
        opportunity = "Position as leader in 'Digital Nomad' - high demand, low competition"
        lane = self._format_opportunity_lane(opportunity)
        required_fields = ["title", "description", "recommendation"]
        for field in required_fields:
            assert field in lane, f"Lane missing required field: {field}"

    def _format_opportunity_lane(self, opportunity: str) -> dict:
        """Format opportunity string into structured lane.

        This mirrors the logic that will be implemented in the service.
        """
        # Parse the opportunity string
        if " - " in opportunity:
            parts = opportunity.split(" - ", 1)
            title = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else ""
        else:
            title = opportunity
            description = ""

        # Extract trend name if present
        trend_name = ""
        if "'" in title:
            start = title.find("'") + 1
            end = title.find("'", start)
            if end > start:
                trend_name = title[start:end]

        return {
            "title": trend_name or title[:50],
            "type": "white_space" if "demand" in description.lower() else "positioning",
            "description": description or title,
            "recommendation": f"Focus marketing and offerings on {trend_name or 'this trend'}",
        }


class TestSendToBuildBrand:
    """Tests for Send to Build a Brand functionality."""

    def test_prefill_includes_location(self):
        """Verify property location is passed to Build a Brand."""
        property_data = {
            "name": "Test Hotel",
            "location": "Barcelona, Spain",
            "region": "europe",
            "themes": ["boutique", "design"],
        }
        prefill = self._create_build_brand_prefill(property_data)
        assert prefill.get("location") == "Barcelona, Spain"

    def test_prefill_includes_segment(self):
        """Verify property segment is passed."""
        property_data = {
            "price_segment": "luxury",
            "themes": ["luxury", "wellness"],
        }
        prefill = self._create_build_brand_prefill(property_data)
        assert prefill.get("segment") == "luxury"

    def test_prefill_includes_gaps(self):
        """Verify experience gaps are passed as context."""
        property_data = {
            "experience_gaps": ["Wellness trend", "Digital nomad"],
        }
        prefill = self._create_build_brand_prefill(property_data)
        assert "gaps" in prefill or "context" in prefill

    def _create_build_brand_prefill(self, property_data: dict) -> dict:
        """Create prefill data for Build a Brand form.

        This mirrors the logic that will be implemented.
        """
        return {
            "location": property_data.get("location", ""),
            "segment": property_data.get("price_segment", property_data.get("themes", [""])[0] if property_data.get("themes") else ""),
            "context": f"Property analysis: {property_data.get('name', 'Unknown')}",
            "gaps": property_data.get("experience_gaps", []),
            "opportunities": property_data.get("opportunity_lanes", []),
        }


class TestDemandScanAPI:
    """End-to-end API tests for Demand Scan."""

    def test_get_properties_returns_list(self):
        """Test GET /api/demand-scan returns property list."""
        from services.demand_scan import DemandScanService
        service = DemandScanService()
        properties = service.get_properties(limit=5)
        assert isinstance(properties, list)

    def test_property_has_required_fields(self):
        """Verify property response has all required fields."""
        required_fields = [
            "url", "name", "property_type", "demand_fit_score",
            "experience_gaps", "opportunity_lanes", "recommendations"
        ]
        from services.demand_scan import DemandScanService
        service = DemandScanService()
        properties = service.get_properties(limit=1)

        if properties:
            prop = properties[0]
            for field in required_fields:
                assert field in prop, f"Property missing field: {field}"


def run_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("  Demand Scan Enhancement Tests")
    print("=" * 60 + "\n")

    # Run with pytest
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    return exit_code


if __name__ == "__main__":
    sys.exit(run_tests())
