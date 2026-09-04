"""Source registry — single source of truth for every data source.

Reads configs/sources.yaml. The POMDP action space, the scheduler fallbacks,
the CLI and the API all derive their source lists from here, so adding a
source is one YAML entry plus (for content sources) a small scraper class.
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

SOURCES_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "sources.yaml"


@dataclass(frozen=True)
class SourceSpec:
    """One registered data source."""

    name: str
    class_path: str | None  # dotted import path; None for planned sources
    kind: str  # 'content' | 'metric'
    type: str  # SourceType value ('news', 'social', ...) or 'metric'
    status: str  # 'active' | 'planned' | 'blocked'
    priority: str = "medium"
    note: str = ""
    config: dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=1)
def get_registry(config_path: str | None = None) -> dict[str, SourceSpec]:
    """Load and cache the source registry."""
    path = Path(config_path) if config_path else SOURCES_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    registry: dict[str, SourceSpec] = {}
    for name, entry in (raw.get("sources") or {}).items():
        registry[name] = SourceSpec(
            name=name,
            class_path=entry.get("class"),
            kind=entry.get("kind", "content"),
            type=entry.get("type", "news"),
            status=entry.get("status", "planned"),
            priority=entry.get("priority", "medium"),
            note=entry.get("note", ""),
            config=entry.get("config") or {},
        )

    active_without_class = [s.name for s in registry.values() if s.status == "active" and not s.class_path]
    if active_without_class:
        raise ValueError(f"Active sources missing a scraper class in sources.yaml: {active_without_class}")

    return registry


def active_sources(kind: str | None = None) -> list[str]:
    """Names of sources in the scraping rotation, optionally filtered by kind."""
    return [
        s.name
        for s in get_registry().values()
        if s.status == "active" and (kind is None or s.kind == kind)
    ]


def retired_sources() -> list[str]:
    """Sources whose historical rows stay in the database but must not count.

    A blocked source (Reddit after its API closed, OTA review sites on terms of
    service) leaves content behind; trends, KPIs and listings treat it as
    archive, so the numbers describe what the platform can still observe.
    """
    return [s.name for s in get_registry().values() if s.status == "blocked"]


def runnable_sources() -> list[str]:
    """Every source with an implementation (active or blocked) — for the CLI."""
    return [s.name for s in get_registry().values() if s.class_path]


def get_source(name: str) -> SourceSpec:
    registry = get_registry()
    if name not in registry:
        raise ValueError(f"Unknown source: {name}. Known: {sorted(registry)}")
    return registry[name]


def get_scraper_class(name: str):
    """Dynamically import and return the scraper class for a source."""
    spec = get_source(name)
    if not spec.class_path:
        raise ValueError(f"Source '{name}' is {spec.status} and has no scraper implementation yet")
    module_path, class_name = spec.class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def get_source_config(name: str) -> dict[str, Any]:
    """Free-form per-source config block from sources.yaml."""
    return get_source(name).config
