"""
Security utilities for RAG generator.

Provides input sanitization to prevent prompt injection attacks.
"""

import logging
import re

# Query validation
MAX_QUERY_LENGTH = 500  # Maximum query length


def sanitize_query(query: str) -> str:
    """
    Sanitize user query to prevent prompt injection attacks.
    
    Security measures:
    1. Remove control characters (\x00-\x1f, \x7f-\x9f)
    2. Limit length to prevent token exhaustion
    3. Filter prompt-breaking patterns
    
    Args:
        query: Raw user query
    
    Returns:
        Sanitized query string
    """
    if not query:
        return ""
    
    # Remove control characters
    query = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query)
    
    # Limit length
    if len(query) > MAX_QUERY_LENGTH:
        logging.warning(f"Query truncated from {len(query)} to {MAX_QUERY_LENGTH} chars")
        query = query[:MAX_QUERY_LENGTH]
    
    # Filter prompt injection patterns (case-insensitive)
    injection_patterns = [
        (r'ignore\s+previous\s+instructions?', '[FILTERED]'),
        (r'ignore\s+above', '[FILTERED]'),
        (r'forget\s+(?:everything|all|previous)', '[FILTERED]'),
        (r'disregard\s+(?:previous|above)', '[FILTERED]'),
        (r'override\s+(?:system|instructions?)', '[FILTERED]'),
        (r'reveal\s+(?:system|prompt)', '[FILTERED]'),
        (r'you\s+are\s+now', '[FILTERED]'),
        (r'new\s+instructions?:', '[FILTERED]')
    ]
    
    for pattern, replacement in injection_patterns:
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    
    return query.strip()
