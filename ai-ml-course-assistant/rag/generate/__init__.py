"""
RAG Generation Package.

Provides answer generation with citations from retrieved context.
"""

from .base import RAGGenerator
from .security import sanitize_query
from .citations import (
    extract_chunk_citations,
    extract_image_citations,
    validate_and_clean_citations,
    check_answer_sources_consistency
)
from .prompts import SYSTEM_PROMPT, MAX_IMAGES_TO_CITE

__all__ = [
    'RAGGenerator',
    'sanitize_query',
    'extract_chunk_citations',
    'extract_image_citations',
    'validate_and_clean_citations',
    'check_answer_sources_consistency',
    'SYSTEM_PROMPT',
    'MAX_IMAGES_TO_CITE'
]
