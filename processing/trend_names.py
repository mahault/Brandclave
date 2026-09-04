"""Trend-name normalisation shared by every consumer that must not double count.

Clustering re-discovers the same theme under small variations — with and without
quotes, singular and plural, "Getaways" versus "Travel" as the trailing noun. The
Signal Ledger, Demand Scan and any ranking that shows a user "the top N trends"
all need the same answer to "are these two the same theme?", so it lives here.
"""


def strip_markdown(text: str | None) -> str:
    """Remove the emphasis and heading markers LLMs leak into plain-text fields.

    Smaller chat models answer "**Heritage Revival Hubs**" when asked for a
    name; the asterisks would otherwise be stored and rendered literally.
    """
    import re

    cleaned = re.sub(r"\*\*|__|^#+\s*", "", text or "", flags=re.M)
    return cleaned.strip().strip('"').strip()


def normalize_trend_title(name: str) -> str:
    """Collapse quoting, case, plural and trailing-noun noise into a dedupe key.

    Keeps the first two words, except when the first word is long and specific
    enough on its own (a compound like "microadventure" or "workation"), in which
    case the trailing noun is dropped: "Micro-Adventure Getaways" and
    "Micro-Adventure Travel" are one theme.
    """
    cleaned = "".join(ch for ch in (name or "").lower() if ch.isalnum() or ch == " ")
    words = [w[:-1] if w.endswith("s") and len(w) > 4 else w for w in cleaned.split()]
    if not words:
        return ""
    if len(words[0]) >= 9:
        return words[0]
    return " ".join(words[:2])
