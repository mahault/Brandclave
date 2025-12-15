"""Text cleaning and normalization utilities."""

import re
import html
from bs4 import BeautifulSoup


def remove_html(text: str) -> str:
    """Remove HTML tags from text."""
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator=" ")


def unescape_html(text: str) -> str:
    """Unescape HTML entities."""
    return html.unescape(text)


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    return url_pattern.sub("", text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    return email_pattern.sub("", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple whitespace to single space and strip."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """Remove special characters, optionally keeping punctuation."""
    if keep_punctuation:
        # Keep letters, numbers, basic punctuation, and spaces
        pattern = r"[^a-zA-Z0-9\s.,!?;:'\"-]"
    else:
        pattern = r"[^a-zA-Z0-9\s]"
    return re.sub(pattern, "", text)


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to ASCII equivalents where possible."""
    import unicodedata

    # Normalize to NFKD form and encode to ASCII, ignoring errors
    normalized = unicodedata.normalize("NFKD", text)
    # Keep the unicode but normalize quotes and dashes
    replacements = {
        """: '"',
        """: '"',
        "'": "'",
        "'": "'",
        "–": "-",
        "—": "-",
        "…": "...",
        "\u00a0": " ",  # Non-breaking space
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def clean_text(
    text: str,
    remove_html_tags: bool = True,
    remove_url: bool = True,
    remove_email: bool = True,
    lowercase: bool = False,
    normalize_whitespace: bool = True,
    min_length: int = 0,
) -> str | None:
    """Clean text with configurable options.

    Args:
        text: Input text to clean
        remove_html_tags: Remove HTML tags
        remove_url: Remove URLs
        remove_email: Remove email addresses
        lowercase: Convert to lowercase
        normalize_whitespace: Collapse whitespace
        min_length: Minimum length after cleaning (return None if shorter)

    Returns:
        Cleaned text or None if too short
    """
    if not text:
        return None

    # Unescape HTML entities first
    text = unescape_html(text)

    if remove_html_tags:
        text = remove_html(text)

    if remove_url:
        text = remove_urls(text)

    if remove_email:
        text = remove_emails(text)

    # Normalize unicode
    text = normalize_unicode(text)

    if normalize_whitespace:
        text = remove_extra_whitespace(text)

    if lowercase:
        text = text.lower()

    # Check minimum length
    if len(text) < min_length:
        return None

    return text


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def extract_sentences(text: str, max_sentences: int | None = None) -> list[str]:
    """Extract sentences from text.

    Args:
        text: Input text
        max_sentences: Maximum number of sentences to return

    Returns:
        List of sentences
    """
    # Simple sentence splitting on common terminators
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_pattern.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if max_sentences:
        sentences = sentences[:max_sentences]

    return sentences


def get_text_stats(text: str) -> dict:
    """Get basic statistics about text."""
    words = text.split()
    sentences = extract_sentences(text)

    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
    }
