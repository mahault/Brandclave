"""Move extraction module for Hotelier Bets using LLM."""

import json
import logging
import re

from data_models.hotelier_move import MoveType
from processing.llm_utils import get_llm

logger = logging.getLogger(__name__)

# Valid move types for validation
VALID_MOVE_TYPES = {mt.value for mt in MoveType}


def extract_move_from_article(
    title: str,
    content: str,
    source_url: str,
    source_name: str,
) -> dict | None:
    """Extract structured hotelier move from a news article.

    Args:
        title: Article title
        content: Article content
        source_url: Source URL
        source_name: Source name (e.g., 'skift')

    Returns:
        Dict with extracted move data or None if no move detected
    """
    llm = get_llm()

    # Truncate content if too long
    article_text = f"{title}\n\n{content}"
    if len(article_text) > 3000:
        article_text = article_text[:3000] + "..."

    system_prompt = """You are a hospitality industry analyst extracting strategic business moves from news articles.
Extract structured information about any strategic business move by a hotel company.
If no clear strategic move is present, respond with {"has_move": false}.

You MUST respond with valid JSON only. No other text."""

    move_types_str = ", ".join(VALID_MOVE_TYPES)

    prompt = f"""Analyze this hospitality news article and extract any strategic business move:

{article_text}

Extract the following in JSON format:
{{
    "has_move": true/false,
    "company": "Company name",
    "company_type": "chain" | "independent" | "management_company" | "investor" | null,
    "move_type": one of [{move_types_str}],
    "market": "Target market/region or null",
    "investment_amount": "Investment amount if mentioned or null",
    "title": "Concise action title (5-10 words)",
    "summary": "Brief summary of the move (1-2 sentences)",
    "strategic_implications": ["implication 1", "implication 2", "implication 3"],
    "confidence": 0.0 to 1.0 (how confident you are this is a real strategic move)
}}

JSON response:"""

    try:
        response = llm.generate(prompt, system_prompt, max_tokens=500, temperature=0.3)

        # Clean response - extract JSON
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        data = json.loads(response)

        if not data.get("has_move", False):
            return None

        # Validate move_type
        move_type = data.get("move_type", "other").lower()
        if move_type not in VALID_MOVE_TYPES:
            move_type = "other"

        confidence = float(data.get("confidence", 0.5))
        if confidence < 0.3:
            logger.debug(f"Low confidence ({confidence}) for: {title}")
            return None

        return {
            "company": data.get("company", "Unknown"),
            "company_type": data.get("company_type"),
            "move_type": move_type,
            "market": data.get("market"),
            "investment_amount": data.get("investment_amount"),
            "title": data.get("title", title[:100]),
            "summary": data.get("summary", ""),
            "strategic_implications": data.get("strategic_implications", []),
            "confidence_score": confidence,
            "source_url": source_url,
            "source_name": source_name,
        }

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Error extracting move from article: {e}")
        return None


def generate_why_it_matters(
    title: str,
    summary: str,
    company: str,
    move_type: str,
    market: str | None,
) -> str:
    """Generate strategic analysis for a hotelier move.

    Args:
        title: Move title
        summary: Move summary
        company: Company name
        move_type: Type of move
        market: Target market

    Returns:
        Strategic analysis paragraph
    """
    llm = get_llm()

    system_prompt = """You are a hospitality strategy consultant advising hotel owners and brands.
Write actionable insights about why strategic moves matter for the industry.
Be specific about competitive implications, market impact, and opportunities.
Keep responses to 2-3 sentences."""

    prompt = f"""Strategic Move: {title}
Company: {company}
Move Type: {move_type}
Market: {market or 'Global'}
Summary: {summary}

Why this move matters for the hospitality industry:"""

    return llm.generate(prompt, system_prompt, max_tokens=200, temperature=0.7).strip()


def extract_competitive_impact(
    company: str,
    move_type: str,
    market: str | None,
    summary: str,
) -> str | None:
    """Generate competitive impact analysis.

    Args:
        company: Company making the move
        move_type: Type of move
        market: Target market
        summary: Move summary

    Returns:
        Competitive impact analysis or None
    """
    llm = get_llm()

    system_prompt = """You are a hospitality industry analyst.
Briefly analyze the competitive impact of strategic moves.
Focus on which competitors are affected and how.
Keep response to 1-2 sentences."""

    prompt = f"""{company} is making a {move_type} move{f' in {market}' if market else ''}.
Summary: {summary}

Competitive impact:"""

    try:
        return llm.generate(prompt, system_prompt, max_tokens=100, temperature=0.6).strip()
    except Exception:
        return None


def fallback_move_detection(title: str, content: str) -> dict | None:
    """Simple regex-based fallback for move detection when LLM fails.

    Args:
        title: Article title
        content: Article content

    Returns:
        Basic move dict or None
    """
    text = f"{title} {content}".lower()

    # Keywords for different move types
    move_patterns = {
        "acquisition": r"(acquir|acquisition|buys|bought|purchase|merger)",
        "launch": r"(launch|open|debut|unveil|introduce|new\s+hotel|new\s+property)",
        "expansion": r"(expand|expansion|enter|enters|growth|new\s+market)",
        "partnership": r"(partner|partnership|collaborat|joint\s+venture|alliance)",
        "renovation": r"(renovat|refurbish|upgrade|transform|redesign)",
        "repositioning": r"(reposition|rebrand|pivot|transform|new\s+direction)",
        "technology": r"(implement|adopt|technology|digital|tech\s+upgrade|ai\s+)",
        "sustainability": r"(sustainab|green|eco|carbon|environment|renewable)",
    }

    for move_type, pattern in move_patterns.items():
        if re.search(pattern, text):
            # Try to extract company name (very basic)
            company_patterns = [
                r"(marriott|hilton|ihg|hyatt|accor|wyndham|choice\s+hotels|best\s+western)",
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:hotel|hospitality|group|international)",
            ]

            company = "Unknown"
            for cp in company_patterns:
                match = re.search(cp, title + " " + content, re.IGNORECASE)
                if match:
                    company = match.group(1).title()
                    break

            return {
                "company": company,
                "company_type": None,
                "move_type": move_type,
                "market": None,
                "investment_amount": None,
                "title": title[:100],
                "summary": content[:200] + "..." if len(content) > 200 else content,
                "strategic_implications": [],
                "confidence_score": 0.4,  # Lower confidence for fallback
            }

    return None
