"""Concept renders — turn a finished brand blueprint into a visual mood board.

The blueprint pipeline produces words: a name, a one-liner, a design direction,
F&B concepts, signature experiences. This module turns those into four
photoreal architectural visualisations with OpenAI's image models, so a client
sees the concept they built rather than reading about it.

Prompts are composed only from the blueprint's own fields (plus the property
inputs); nothing is invented outside what the client concocted. Text, logos and
people are excluded from prompts because image models render them badly and
they date the image.

Files are written under data/renders/<blueprint_id>/ with a manifest; the
render API serves them from there. On hosts with ephemeral disks the files can
vanish on redeploy and are regenerated on demand.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from config.settings import get_settings

logger = logging.getLogger(__name__)

RENDERS_ROOT = Path("data/renders")

# The four spaces every hotel concept has to answer for, in the order a guest meets them.
SCENES = [
    {
        "key": "arrival",
        "label": "Arrival & facade",
        "brief": "the street-level arrival and facade, dusk light, entrance legible from the street",
    },
    {
        "key": "lobby",
        "label": "Lobby & social heart",
        "brief": "the lobby as the social heart of the property, furniture and material palette in view",
    },
    {
        "key": "room",
        "label": "Signature guest room",
        "brief": "a signature guest room, bed and window composition, morning light",
    },
    {
        "key": "fnb",
        "label": "Food & beverage",
        "brief": "the primary food-and-beverage space at golden hour, tableware and lighting in view",
    },
]

STYLE = (
    "Photorealistic architectural visualisation, wide-angle 24mm, natural light, high dynamic range, "
    "editorial hospitality photography quality. No people, no text, no logos, no signage, no watermarks."
)


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values() if v)
    if isinstance(value, (list, tuple)):
        return "; ".join(_text(v) for v in value if v)
    return re.sub(r"\s+", " ", str(value)).strip()


def _clip(text: str, limit: int) -> str:
    text = _text(text)
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def blueprint_context(bp) -> dict:
    """Flatten the blueprint (schema object or dict) into the fields prompts use."""
    get = (lambda k, d=None: bp.get(k, d)) if isinstance(bp, dict) else (lambda k, d=None: getattr(bp, k, d))
    inputs = get("inputs") or {}
    ig = (lambda k, d=None: inputs.get(k, d)) if isinstance(inputs, dict) else (lambda k, d=None: getattr(inputs, k, d))
    names = get("brand_names") or {}
    primary = names.get("primary") if isinstance(names, dict) else getattr(names, "primary", None)
    fnb = get("fnb_concepts") or []
    first_fnb = fnb[0] if fnb else None
    experiences = get("signature_experiences") or []
    return {
        "brand": _text(primary) or get("brand_name_primary") or "the hotel",
        "one_liner": _clip(get("one_liner"), 220),
        "location": _text(ig("location")) or _text(get("location")),
        "segment": _text(ig("segment")) or _text(get("segment")),
        "rooms": ig("rooms") or get("rooms"),
        "design_direction": _clip(get("design_direction"), 700),
        "pillars": _clip(get("pillars"), 240),
        "fnb_name": _text(first_fnb.get("name") if isinstance(first_fnb, dict) else getattr(first_fnb, "name", "")) if first_fnb else "",
        "fnb_concept": _clip(
            (first_fnb.get("concept") if isinstance(first_fnb, dict) else getattr(first_fnb, "concept", "")) if first_fnb else "", 300
        ),
        "fnb_vibe": _clip((first_fnb.get("vibe") if isinstance(first_fnb, dict) else getattr(first_fnb, "vibe", "")) if first_fnb else "", 160),
        "experience": _clip(
            (experiences[0].get("description") if isinstance(experiences[0], dict) else getattr(experiences[0], "description", "")) if experiences else "", 240
        ),
    }


def compose_prompt(ctx: dict, scene: dict) -> str:
    """One scene prompt from the blueprint's own language."""
    place = f"{ctx['brand']}, a {ctx['rooms'] or ''}-key {ctx['segment'].lower() or 'lifestyle'} hotel in {ctx['location']}".replace("  ", " ")
    parts = [f"{place}: {scene['brief']}."]
    if ctx["design_direction"]:
        parts.append(f"Design direction: {ctx['design_direction']}")
    if scene["key"] == "fnb" and ctx["fnb_concept"]:
        parts.append(f"The space is {ctx['fnb_name']}: {ctx['fnb_concept']} Atmosphere: {ctx['fnb_vibe']}")
    if scene["key"] == "lobby" and ctx["experience"]:
        parts.append(f"It hosts: {ctx['experience']}")
    if ctx["one_liner"]:
        parts.append(f"Brand feeling: {ctx['one_liner']}")
    parts.append(STYLE)
    return " ".join(p.strip() for p in parts if p.strip())


def render_dir(blueprint_id: str) -> Path:
    return RENDERS_ROOT / blueprint_id


def load_manifest(blueprint_id: str) -> dict | None:
    path = render_dir(blueprint_id) / "manifest.json"
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    # Files can disappear on ephemeral disks; only report what is actually there.
    manifest["renders"] = [r for r in manifest.get("renders", []) if (render_dir(blueprint_id) / r["file"]).exists()]
    return manifest


def generate_renders(bp, blueprint_id: str, *, scenes: list[str] | None = None, quality: str = "medium", size: str = "1536x1024") -> dict:
    """Generate the mood board for a blueprint and write it to disk.

    Raises RuntimeError with a plain message when the key is missing so the API
    can return it to the UI.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; concept renders are disabled.")
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    model = settings.openai_image_model
    ctx = blueprint_context(bp)
    wanted = [s for s in SCENES if not scenes or s["key"] in scenes]

    out_dir = render_dir(blueprint_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = load_manifest(blueprint_id) or {"renders": []}
    kept = [r for r in existing.get("renders", []) if r["scene"] not in {s["key"] for s in wanted}]

    renders = []
    for scene in wanted:
        prompt = compose_prompt(ctx, scene)
        try:
            result = client.images.generate(model=model, prompt=prompt, size=size, quality=quality, n=1)
        except Exception as exc:  # surface per-scene failures, keep the rest
            logger.warning(f"Render failed for {blueprint_id}/{scene['key']}: {exc}")
            renders.append({"scene": scene["key"], "label": scene["label"], "error": str(exc)[:200], "prompt": prompt})
            continue
        data = result.data[0]
        image_bytes = base64.b64decode(data.b64_json)
        filename = f"{scene['key']}.png"
        (out_dir / filename).write_bytes(image_bytes)
        renders.append(
            {
                "scene": scene["key"],
                "label": scene["label"],
                "file": filename,
                "url": f"/api/brand-blueprint/{blueprint_id}/renders/{filename}",
                "prompt": prompt,
                "model": model,
                "size": size,
                "quality": quality,
                "bytes": len(image_bytes),
                "generated_at": datetime.utcnow().isoformat(),
            }
        )

    manifest = {
        "blueprint_id": blueprint_id,
        "brand": ctx["brand"],
        "model": model,
        "generated_at": datetime.utcnow().isoformat(),
        "renders": kept + [r for r in renders if "file" in r],
        "failures": [r for r in renders if "error" in r],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Renders for {blueprint_id}: {len(manifest['renders'])} ok, {len(manifest['failures'])} failed")
    return manifest
