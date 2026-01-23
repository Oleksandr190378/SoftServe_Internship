"""
Multimodal retrieval package.

Provides text and image retrieval with semantic verification.
"""

from .base import (
    MultimodalRetriever,
    EMBEDDING_MODEL,
    EMBEDDING_DIMS,
    DEFAULT_K_TEXT,
    DEFAULT_K_IMAGES,
    DEFAULT_MMR_LAMBDA,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW
)
from .verification import SemanticVerifier
from .image_ops import ImageRetriever
from .utils import parse_json_list, format_related_image_ids, is_visual_query, VISUAL_KEYWORDS
from config import RETRIEVAL

# Export similarity thresholds from config
SIMILARITY_THRESHOLD = RETRIEVAL.SIMILARITY_THRESHOLD
SIMILARITY_THRESHOLD_NEARBY = RETRIEVAL.SIMILARITY_THRESHOLD_NEARBY

__all__ = [
    'MultimodalRetriever',
    'SemanticVerifier',
    'ImageRetriever',
    'parse_json_list',
    'format_related_image_ids',
    'is_visual_query',
    'VISUAL_KEYWORDS',
    'EMBEDDING_MODEL',
    'EMBEDDING_DIMS',
    'DEFAULT_K_TEXT',
    'DEFAULT_K_IMAGES',
    'SIMILARITY_THRESHOLD',
    'SIMILARITY_THRESHOLD_NEARBY',
    'DEFAULT_MMR_LAMBDA',
    'CONFIDENCE_HIGH',
    'CONFIDENCE_MEDIUM',
    'CONFIDENCE_LOW'
]
