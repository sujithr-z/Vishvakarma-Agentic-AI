"""Grounded Knowledge Retriever for TERI-2021 and TARU-2015 Knowledge Base."""
import os
import json
import re
from typing import Any, Dict, List, Optional

KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "knowledge.json")


def _tokenize(text: str) -> set:
    """Normalize text into alphanumeric lowercase tokens."""
    return set(re.findall(r"\b\w{3,}\b", text.lower()))


def search_knowledge(
    query: str,
    category: Optional[str] = None,
    climate: Optional[str] = None,
    building_component: Optional[str] = None,
    top_k: int = 4,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Search grounded TERI-2021 & TARU-2015 knowledge cards using token overlap and metadata filtering.
    
    Args:
        query: User or agent question / search terms.
        category: Filter by card category (constraint, material, causal_chain, principle).
        climate: Filter by climate zone (Hot & Dry, Warm & Humid, Composite, Temperate, all).
        building_component: Filter by component (roof, wall, opening, environment, whole).
        top_k: Maximum number of knowledge items to return.
        
    Returns:
        List of relevant knowledge documents with source citations.
    """
    if not os.path.exists(KNOWLEDGE_PATH):
        return []

    try:
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return []

    query_tokens = _tokenize(query)
    # User-facing climate labels and KB labels are not identical. Normalize only
    # the filter; never invent a climate when the caller leaves it unspecified.
    climate_aliases = {
        "hot_humid": "Warm & Humid",
        "warm_humid": "Warm & Humid",
        "hot_dry": "Hot & Dry",
        "composite": "Composite",
    }
    climate_filter = climate_aliases.get(climate, climate) if climate else None

    scored_items = []
    for item in items:
        # Category filter
        if category and item.get("category") != category:
            continue
        # Climate filter
        if climate_filter and item.get("climate") not in [climate_filter, "all"]:
            continue
        # Component filter
        if building_component and item.get("building_component") not in [building_component, "whole"]:
            continue

        item_text = (
            f"{item.get('id', '')} {item.get('category', '')} {item.get('topic', '')} "
            f"{item.get('parameter', '')} {' '.join(item.get('keywords', []))} "
            f"{item.get('text', '')} {item.get('explanation', '')} {item.get('application', '')} "
            f"{item.get('climate', '')} {item.get('building_component', '')} "
            f"{item.get('rule_or_threshold', '')} {item.get('source', '')}"
        )
        item_tokens = _tokenize(item_text)

        intersection = query_tokens.intersection(item_tokens)
        score = len(intersection)

        # Keyword bonus
        for kw in item.get("keywords", []):
            if kw.lower() in query.lower():
                score += 3

        if score > 0 or not query_tokens:
            scored_items.append((score, item))

    scored_items.sort(key=lambda x: x[0], reverse=True)
    # An empty result is meaningful: returning the first cards would present
    # unrelated standards as if they supported the user's question.
    return [item for score, item in scored_items if score > 0][:top_k]
