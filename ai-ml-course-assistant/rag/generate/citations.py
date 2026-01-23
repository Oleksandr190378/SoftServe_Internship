"""
Citation extraction and validation utilities.

Handles extraction, validation, and cleaning of citations in generated responses.
"""

import logging
import re
from typing import Dict, List, Tuple

# Citation validation constants
FIRST_CHUNK_INDEX = 1  # Text chunks are 1-indexed ([1], [2], [3]...)
FIRST_IMAGE_LETTER = 'A'  # Images use letters ([A], [B], [C]...)
FIRST_IMAGE_ASCII = 65  # ASCII code for 'A'


def extract_chunk_citations(
    sources_text: str, 
    text_chunks: List[Dict]
) -> Tuple[List[str], List[str]]:
    """
    Extract and validate chunk citations from sources text.
    
    Single Responsibility: Citation extraction and validation for text chunks.
    
    Args:
        sources_text: Text containing citations like [1], [2], [3]
        text_chunks: List of available text chunks
    
    Returns:
        Tuple of (chunk_ids, citation_labels)
    """
    chunk_citations = re.findall(r'\[(\d+)\]', sources_text)
    chunk_ids = []
    
    for idx in chunk_citations:
        try:
            idx_int = int(idx) - FIRST_CHUNK_INDEX
            if 0 <= idx_int < len(text_chunks):
                chunk_ids.append(text_chunks[idx_int]['chunk_id'])
        except (ValueError, KeyError, IndexError) as e:
            logging.warning(f"Invalid chunk citation [{idx}]: {e}")
            continue
    
    return chunk_ids, chunk_citations


def extract_image_citations(
    sources_text: str, 
    images: List[Dict]
) -> Tuple[List[str], List[str]]:
    """
    Extract and validate image citations from sources text.
    
    Single Responsibility: Citation extraction and validation for images.
    
    Args:
        sources_text: Text containing citations like [A], [B], [C]
        images: List of available images
    
    Returns:
        Tuple of (image_ids, citation_letters)
    """
    image_citations = re.findall(r'\[([A-Z])\]', sources_text)
    image_ids = []
    
    for letter in image_citations:
        try:
            idx = ord(letter) - FIRST_IMAGE_ASCII
            if 0 <= idx < len(images):
                image_ids.append(images[idx]['image_id'])
        except (KeyError, IndexError) as e:
            logging.warning(f"Invalid image citation [{letter}]: {e}")
            continue
    
    return image_ids, image_citations


def validate_and_clean_citations(response_text: str, num_chunks: int, num_images: int) -> str:
    """
    Validate and clean citations in generated response.
    
    Fixes:
    1. Removes "chunk [X]" format → [X]
    2. Removes citations to non-existent images/chunks
    3. Cleans up orphaned brackets
    
    Args:
        response_text: Generated answer text
        num_chunks: Number of available text chunks (1-indexed)
        num_images: Number of available images (A-indexed)
    
    Returns:
        Cleaned response text with only valid citations
    """
    original = response_text
    
    # Fix 1: Remove "chunk [X]" format
    response_text = re.sub(r',?\s*chunk\s+\[(\d+)\]', r' [\1]', response_text, flags=re.IGNORECASE)
    if response_text != original:
        logging.info("✓ Removed 'chunk [X]' format from citations")
    
    # Fix 2: Validate text chunk citations
    valid_chunk_nums = set(str(i) for i in range(1, num_chunks + 1))
    cited_chunks = set(re.findall(r'\[(\d+)\]', response_text))
    invalid_chunks = cited_chunks - valid_chunk_nums
    
    if invalid_chunks:
        logging.warning(f"Invalid chunk citations: {invalid_chunks} (valid: {valid_chunk_nums})")
        for num in invalid_chunks:
            # Remove invalid chunk citations
            response_text = re.sub(rf'\[{num}\]', '', response_text)
        logging.info(f"✓ Removed {len(invalid_chunks)} invalid chunk citation(s)")
    
    # Fix 3: Validate image citations
    valid_image_letters = set(chr(65 + i) for i in range(num_images))  # A, B, C...
    cited_images = set(re.findall(r'\[([A-Z])\]', response_text))
    invalid_images = cited_images - valid_image_letters
    
    if invalid_images:
        logging.warning(f"Invalid image citations: {invalid_images} (valid: {valid_image_letters})")
        for letter in invalid_images:
            # Remove invalid image citations
            response_text = re.sub(rf'\[{letter}\]', '', response_text)
        logging.info(f"✓ Removed {len(invalid_images)} invalid image citation(s)")
    
    # Fix 4: Clean up orphaned brackets and extra spaces
    response_text = re.sub(r'\]\s*\[', '] [', response_text)  # ][ → ] [
    response_text = re.sub(r'\s{2,}', ' ', response_text)  # multiple spaces → single
    response_text = re.sub(r'\s+([.,!?])', r'\1', response_text)  # space before punctuation
    
    if response_text != original:
        logging.info("✓ Citations validated and cleaned")
    
    return response_text


def check_answer_sources_consistency(answer: str, sources_text: str) -> None:
    """
    Check consistency between citations in Answer and Sources sections.
    
    Logs warning if inconsistencies found (log-only, doesn't modify text).
    
    Args:
        answer: Answer text with citations
        sources_text: Sources text with citations
    """
    sources_chunks = set(re.findall(r'\[(\d+)\]', sources_text))
    sources_images = set(re.findall(r'\[([A-Z])\]', sources_text))
    
    answer_chunks = set(re.findall(r'\[(\d+)\]', answer))
    answer_images = set(re.findall(r'\[([A-Z])\]', answer))
    
    orphan_chunks = answer_chunks - sources_chunks
    orphan_images = answer_images - sources_images
    
    if orphan_chunks or orphan_images:
        logging.warning(
            f"⚠️  Answer-Sources inconsistency: "
            f"Answer={orphan_chunks or 'OK'}/{orphan_images or 'OK'} missing from Sources"
        )
