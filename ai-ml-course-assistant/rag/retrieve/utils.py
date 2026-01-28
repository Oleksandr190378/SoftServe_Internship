"""
Utility functions for retrieval operations.

Helper functions for parsing metadata and constants.
"""

import json
import logging
from typing import List, Dict, Any


# Visual query keywords
VISUAL_KEYWORDS = [
    "show", "diagram", "architecture", "figure", "image", 
    "visualization", "chart", "graph", "draw", "display",
    "illustrate", "picture", "schema"
]


def parse_json_list(value: Any, field_name: str = "field") -> List[str]:
    """
    Parse JSON-encoded list or return list/string as-is.
    
    Args:
        value: Value to parse (could be JSON string, list, or comma-separated string)
        field_name: Field name for logging purposes
    
    Returns:
        List of string values (empty list if parsing fails)
    """
    if not value:
        return []
    
    # Already a list - return as-is
    if isinstance(value, list):
        return value
    
    # Try JSON parsing for encoded lists
    if isinstance(value, str) and value.startswith('['):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse {field_name} JSON: {e}")
            # Fallback to comma-separated parsing
            return [id.strip() for id in value.split(',') if id.strip()]
    
    # Comma-separated string
    if isinstance(value, str):
        return [id.strip() for id in value.split(',') if id.strip()]
    
    return []


def format_related_image_ids(chunk_metadata: Dict) -> List[str]:
    """
    Extract and format related image IDs from chunk metadata.
    
    Args:
        chunk_metadata: Chunk metadata dictionary
    
    Returns:
        List of image ID strings
    """
    related_ids_raw = chunk_metadata.get('related_image_ids', '')
    return parse_json_list(related_ids_raw, 'related_image_ids')


def is_visual_query(query: str) -> bool:
    """
    Check if query requests visual content.
    
    Args:
        query: User query string
    
    Returns:
        True if query contains visual keywords
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in VISUAL_KEYWORDS)
