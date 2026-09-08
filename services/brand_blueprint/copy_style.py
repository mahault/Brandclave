"""House style for generated brand copy.

Two halves. STYLE_CONTRACT goes into every stage's system prompt and tells the
model how to write. `polish` runs over every string the model returns (and over
saved blueprints when they are served) and removes the mechanical tells the
model produces anyway: em and en dashes, markdown, curly-quote artefacts,
doubled spaces. It never rewrites meaning; sentence-level tells such as
"not X but Y" are the prompt's job.
"""

import re
from typing import Any

STYLE_CONTRACT = (
    "\n\nHouse style, mandatory. Write like a hotel operator briefing an investor, "
    "not like a brochure. Short plain sentences, most under 20 words. One idea per "
    "sentence. Name concrete things: materials, streets, hours, prices, staff roles, "
    "menu items. Say who does what.\n"
    "Never use an em dash or an en dash; use a comma or a full stop. "
    "Never write 'not just X, it's Y', 'isn't just', 'more than a place to stay', "
    "'where X meets Y', or any sentence built on that contrast. "
    "Do not use these words: curated, seamless, elevate, elevated, immersive, "
    "sanctuary, authentic, bespoke, tapestry, testament, journey, vibrant, soul, "
    "redefine, transform, unlock, boundless, harness, leverage, delve, foster, "
    "holistic, synergy, magic, timeless, effortless, unparalleled, unforgettable.\n"
    "Do not invent statistics, market sizes, growth rates, survey results or "
    "attributions. Use a number only if it appears in the inputs or the market "
    "context you were given; otherwise describe the direction in words. "
    "No superlatives without a comparison you can defend. No exclamation marks. "
    "Fields the schema defines as scores (0 to 1) or counts are still numbers; the "
    "rule about numbers applies to prose only."
)

# Words the contract bans; polish() does not replace them (that would garble
# sentences) but they are counted so a regeneration can be judged.
BANNED_WORDS = (
    "curated", "seamless", "elevate", "elevated", "immersive", "sanctuary",
    "authentic", "bespoke", "tapestry", "testament", "journey", "vibrant", "soul",
    "redefine", "transform", "unlock", "boundless", "harness", "leverage", "delve",
    "foster", "holistic", "synergy", "magic", "timeless", "effortless",
    "unparalleled", "unforgettable",
)

_DASH_BETWEEN_WORDS = re.compile(r"\s*[—–]\s*")
_BULLET_PREFIX = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)", re.MULTILINE)
_MD_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MULTISPACE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")

_SMART = {
    "‘": "'", "’": "'", "“": '"', "”": '"', " ": " ",
    "…": "...",
}


def _dash_to_punctuation(text: str) -> str:
    """Replace em/en dashes with the punctuation the sentence needs.

    A dash followed by a lower-case continuation is a comma. A dash before a
    capitalised word that starts a new clause is a full stop. A dash used as a
    bracket pair (word — aside — word) becomes a pair of commas.
    """

    def repl(match: re.Match) -> str:
        after = text[match.end(): match.end() + 1]
        before = text[match.start() - 1: match.start()] if match.start() > 0 else ""
        if not after:
            return "."
        if before in ",.;:" or before == "":
            return " "
        if after.isupper() and not (match.end() + 1 < len(text) and text[match.end() + 1].isupper()):
            return ". "
        return ", "

    return _DASH_BETWEEN_WORDS.sub(repl, text)


def polish(text: str) -> str:
    """Mechanical cleanup of one string of generated copy."""
    if not isinstance(text, str) or not text:
        return text
    out = text
    for bad, good in _SMART.items():
        out = out.replace(bad, good)
    out = _MD_HEADING.sub("", out)
    out = _MD_EMPHASIS.sub(r"\2", out)
    out = _BULLET_PREFIX.sub("", out)
    out = _dash_to_punctuation(out)
    out = out.replace(" ,", ",").replace(",,", ",").replace("..", ".")
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    out = _MULTISPACE.sub(" ", out)
    return out.strip()


def polish_value(value: Any) -> Any:
    """polish() applied recursively through dicts and lists."""
    if isinstance(value, str):
        return polish(value)
    if isinstance(value, list):
        return [polish_value(v) for v in value]
    if isinstance(value, dict):
        return {k: polish_value(v) for k, v in value.items()}
    return value


def tell_report(value: Any) -> dict[str, int]:
    """Count the tells left in a blueprint-shaped value, for logging and tests."""
    texts: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, str):
            texts.append(v)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)

    walk(value)
    joined = "\n".join(texts)
    return {
        "dashes": joined.count("—") + joined.count("–"),
        "contrast_frames": len(re.findall(r"(?i)\b(isn't|not|more than) (just|only|merely|a place)\b", joined)),
        "banned_words": sum(len(re.findall(rf"(?i)\b{w}\b", joined)) for w in BANNED_WORDS),
        "percent_claims": len(re.findall(r"\d+(?:\.\d+)?\s?%", joined)),
        "chars": len(joined),
    }
