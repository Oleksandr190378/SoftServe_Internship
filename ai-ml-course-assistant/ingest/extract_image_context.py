"""
Extract surrounding context for images - captures text around figures.

This module extracts text before and after each image using sentence boundaries,
along with figure captions, to provide contextual information for enriched captions.

Improvements:
- Uses sentence boundaries instead of fixed character count
- Leverages y-coordinates to identify paragraphs
- Cross-page fallback for empty context_before
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
import fitz
from utils.logging_config import setup_logging

setup_logging()

# Caption extraction constants
CAPTION_PATTERN = r'^(Figure|Fig\.|Table)\s+\d+:?\s*'
CAPTION_MAX_VERTICAL_DISTANCE = 80      # pixels - max distance from image to caption
CAPTION_MAX_LENGTH = 200                # characters - max caption length

# Context extraction constants
CONTEXT_MAX_CHARS = 250                 # characters - max context size
PARAGRAPH_Y_GAP_THRESHOLD = 5           # pixels - threshold for paragraph grouping
SENTENCE_MARKER_LENGTH = 2              # characters - length of ". " marker

SENTENCE_END_MARKERS = ['. ', '.\n', '! ', '!\n', '? ', '?\n']

FIGURE_REFERENCE_PATTERN = r'\b(Figure|Fig\.|Table)\s+\d+\b'


def _collect_text_blocks_from_page(page) -> List[Dict]:
    """
    STAGE 3: DRY - extract text collection logic.
    
    Collect all text blocks with positions from a page.
    
    Args:
        page: PyMuPDF page object
    
    Returns:
        List of text items with coordinates
    """
    try:
        text_dict = page.get_text("dict")
        blocks = text_dict.get("blocks", [])
    except Exception as e:
        logging.error(f"Failed to extract text from page: {e}")
        return []
    
    text_with_positions = []
    for block in blocks:
        if block.get("type") != 0:  # Skip non-text blocks
            continue
        
        try:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_with_positions.append({
                        "text": span["text"],
                        "y0": span["bbox"][1],
                        "y1": span["bbox"][3]
                    })
        except (KeyError, IndexError) as e:
            logging.warning(f"Failed to extract text span: {e}")
            continue
    
    return text_with_positions


def _find_line_index(block: Dict, starting_line: Dict) -> int:
    """
    STAGE 1: Validation - find line by value comparison, not reference.
    STAGE 3: DRY - extract line finding logic.
    
    Find index of starting line in block.
    
    Args:
        block: Text block containing lines
        starting_line: Line to find
    
    Returns:
        Index of line, or -1 if not found
    """
    if not isinstance(block, dict) or "lines" not in block:
        logging.warning("Invalid block structure")
        return -1
    
    try:
        for idx, line in enumerate(block.get("lines", [])):
            # Compare by line content, not reference
            if isinstance(line, dict) and "spans" in line:
                line_text = " ".join([span.get("text", "") for span in line.get("spans", [])])
                starting_text = " ".join([span.get("text", "") for span in starting_line.get("spans", [])])
                
                if line_text == starting_text:
                    return idx
        
        return -1
    except Exception as e:
        logging.warning(f"Error finding line index: {e}")
        return -1


def _extract_text_from_lines(lines: List[Dict], start_idx: int, max_length: int) -> str:
    """
    
    Extract and concatenate text from lines starting at index.
    
    Args:
        lines: List of line dicts
        start_idx: Starting line index
        max_length: Maximum characters
    
    Returns:
        Concatenated text
    """
    caption_text = ""
    
    for line_idx in range(start_idx, len(lines)):
        try:
            line = lines[line_idx]
            if isinstance(line, dict) and "spans" in line:
                line_text = " ".join([span.get("text", "") for span in line.get("spans", [])])
                caption_text += line_text + " "
                
                if len(caption_text) > max_length:
                    break
        except Exception as e:
            logging.warning(f"Error extracting text from line {line_idx}: {e}")
            continue
    
    return caption_text.strip()


def _extract_sentence_from_end(text: str, max_chars: int) -> str:
    """
    
    Extract sentence from end of text up to max_chars.
    
    Args:
        text: Source text
        max_chars: Maximum characters to consider
    
    Returns:
        Extracted text ending with sentence boundary
    """
    if not text or max_chars < 1:
        return ""
    
    chunk = text[-max_chars:] if len(text) > max_chars else text
    
    last_boundary = -1
    for end_marker in SENTENCE_END_MARKERS:
        idx = chunk.rfind(end_marker)
        if idx > last_boundary:
            last_boundary = idx
    
    if last_boundary != -1:
        return chunk[last_boundary + SENTENCE_MARKER_LENGTH:].strip()
    
    return chunk.strip()


def _extract_sentence_from_start(text: str, max_chars: int) -> str:
    """
    STAGE 3: SRP - extract sentence from start logic.
    
    Extract sentence from start of text up to max_chars.
    
    Args:
        text: Source text
        max_chars: Maximum characters to consider
    
    Returns:
        Extracted text starting with sentence boundary
    """
    if not text or max_chars < 1:
        return ""
    
    chunk = text[:max_chars] if len(text) > max_chars else text
    
    first_boundary = len(chunk)
    for end_marker in SENTENCE_END_MARKERS:
        idx = chunk.find(end_marker)
        if idx != -1 and idx < first_boundary:
            first_boundary = idx + 1  # Include the period
    
    return chunk[:first_boundary].strip()


def extract_figure_caption(text_blocks: List[Dict], image_bbox: fitz.Rect) -> Optional[str]:
    """
    Extract figure/table caption near an image.
    
    Looks for text blocks starting with "Figure", "Fig.", "Table" 
    that are positioned near the image bounding box.
    
    Args:
        text_blocks: List of text blocks with coordinates from page.get_text("dict")
        image_bbox: Bounding rectangle of the image
        
    Returns:
        Caption string or None if not found
    """
    # STAGE 1: Parameter validation
    if not isinstance(text_blocks, list):
        logging.error(f"text_blocks must be list, got {type(text_blocks).__name__}")
        return None
    
    if image_bbox is None or not hasattr(image_bbox, 'y0'):
        logging.error("image_bbox is None or invalid")
        return None
    
    try:
        for block in text_blocks:
            if not isinstance(block, dict) or 'lines' not in block:
                continue
            
            for line in block.get('lines', []):
                if not isinstance(line, dict) or 'spans' not in line:
                    continue
                    
                for span in line.get('spans', []):
                    if not isinstance(span, dict) or 'text' not in span:
                        continue
                    
                    text = span['text'].strip()

                    if re.match(CAPTION_PATTERN, text, re.IGNORECASE):
                        span_rect = fitz.Rect(span['bbox'])

                        vertical_distance = min(
                            abs(span_rect.y0 - image_bbox.y1), 
                            abs(image_bbox.y0 - span_rect.y1)  
                        )
                        
                        if vertical_distance < CAPTION_MAX_VERTICAL_DISTANCE:
                            full_caption = extract_full_caption_text(block, line)
                            return full_caption
    except Exception as e:
        logging.error(f"Error extracting figure caption: {e}")
    
    return None


def extract_full_caption_text(block: Dict, starting_line: Dict, max_length: int = CAPTION_MAX_LENGTH) -> str:
    """
    Extract complete caption text from starting line and following lines.
    
    Args:
        block: Text block containing the caption
        starting_line: Line where caption starts
        max_length: Maximum characters to extract
        
    Returns:
        Full caption text
    """
    # STAGE 1: Parameter validation
    if not isinstance(block, dict) or 'lines' not in block:
        logging.warning("Invalid block structure")
        return ""
    
    if not isinstance(starting_line, dict):
        logging.warning("Invalid starting_line")
        return ""
    
    if max_length < 1:
        logging.warning(f"Invalid max_length {max_length}")
        return ""
    
    # STAGE 3: Use helper function to find line
    start_idx = _find_line_index(block, starting_line)
    
    if start_idx == -1:
        logging.warning("Starting line not found in block")
        return ""
    
    # STAGE 3: Use helper function to extract text
    return _extract_text_from_lines(block.get('lines', []), start_idx, max_length)


def _extract_sentence_boundary_from_text(text: str, max_chars: int = CONTEXT_MAX_CHARS, from_end: bool = False) -> str:
    """
    Extract text up to nearest sentence boundary.
    
    Args:
        text: Text to extract from
        max_chars: Maximum characters to consider
        from_end: If True, extract from end (for context_before), else from start (for context_after)
    
    Returns:
        Extracted text up to sentence boundary
    """
    # STAGE 1: Parameter validation
    if not isinstance(text, str):
        logging.warning(f"text must be str, got {type(text).__name__}")
        return ""
    
    if not text:
        return ""
    
    if max_chars < 1:
        logging.warning(f"max_chars must be >= 1, got {max_chars}")
        return ""
    
    # STAGE 3: Delegate to specialized functions
    if from_end:
        return _extract_sentence_from_end(text, max_chars)
    else:
        return _extract_sentence_from_start(text, max_chars)


def _group_text_into_paragraphs(text_items: List[Dict]) -> List[Tuple[str, float, float]]:
    """
    Group text items into paragraphs based on y-coordinate proximity.
    
    Args:
        text_items: List of dicts with 'text', 'y0', 'y1' keys
    
    Returns:
        List of (paragraph_text, start_y, end_y) tuples
    """
    # STAGE 1: Parameter validation
    if not isinstance(text_items, list):
        logging.warning(f"text_items must be list, got {type(text_items).__name__}")
        return []
    
    if not text_items:
        return []
    
    # Validate first item structure
    if not isinstance(text_items[0], dict) or 'y0' not in text_items[0]:
        logging.warning("Invalid text_items structure")
        return []
    
    paragraphs = []
    current_para = []
    current_y_start = text_items[0]["y0"]
    current_y_end = text_items[0]["y1"]
    
    for i, item in enumerate(text_items):
        try:
            if not isinstance(item, dict) or 'text' not in item:
                logging.warning(f"Skipping invalid item at index {i}")
                continue
            
            if not current_para:
                # Start new paragraph
                current_para.append(item["text"])
                current_y_start = item["y0"]
                current_y_end = item["y1"]
            else:
                # Check if this item continues the paragraph
                y_gap = item["y0"] - current_y_end
                
                if y_gap < PARAGRAPH_Y_GAP_THRESHOLD:  # Same paragraph
                    current_para.append(item["text"])
                    current_y_end = item["y1"]
                else:
                    # Save current paragraph and start new one
                    para_text = " ".join(current_para)
                    paragraphs.append((para_text, current_y_start, current_y_end))
                    
                    current_para = [item["text"]]
                    current_y_start = item["y0"]
                    current_y_end = item["y1"]
        except (KeyError, TypeError) as e:
            logging.warning(f"Error processing text item {i}: {e}")
            continue
    
    # Save last paragraph
    if current_para:
        para_text = " ".join(current_para)
        paragraphs.append((para_text, current_y_start, current_y_end))
    
    return paragraphs


def _extract_context_from_previous_page(doc, page_num: int, max_chars: int = CONTEXT_MAX_CHARS) -> str:
    """
    Extract context from previous page when current page has empty context_before.
    Args:
        doc: PyMuPDF document object
        page_num: Current page number (0-based)
        max_chars: Maximum characters to extract
    
    Returns:
        Context text from previous page (last paragraph or sentence)
    """
    # STAGE 1: Parameter validation
    if page_num <= 0:
        return ""
    
    if doc is None:
        logging.warning("doc is None")
        return ""
    
    if max_chars < 1:
        logging.warning(f"max_chars must be >= 1, got {max_chars}")
        return ""
    
    try:
        prev_page = doc[page_num - 1]
        
        # STAGE 3: DRY - use helper function
        text_items = _collect_text_blocks_from_page(prev_page)
        
        if not text_items:
            return ""
        
        # Get all text from previous page
        full_text = " ".join([item["text"] for item in text_items])
        
        # Extract last sentence(s) up to max_chars
        return _extract_sentence_boundary_from_text(full_text, max_chars, from_end=True)
    
    except IndexError as e:
        logging.warning(f"Page {page_num - 1} does not exist: {e}")
        return ""
    except Exception as e:
        logging.warning(f"Error extracting context from previous page: {e}")
        return ""


def extract_surrounding_context(
    page,
    image_bbox: fitz.Rect,
    doc=None,
    page_num: int = 0,
    max_chars: int = CONTEXT_MAX_CHARS
) -> Dict[str, str]:
    """
    Extract text surrounding an image using sentence boundaries and paragraph detection.
    
    Improvements over original:
    1. Uses sentence boundaries instead of fixed character count
    2. Groups text into paragraphs based on y-coordinates
    3. Falls back to previous page if context_before is empty
    
    Args:
        page: PyMuPDF page object
        image_bbox: Bounding rectangle of the image
        doc: PyMuPDF document object (for cross-page context)
        page_num: Page number (0-based, for cross-page fallback)
        max_chars: Approximate maximum characters (actual uses sentence boundaries)
        
    Returns:
        {
            "before": "Text before image (complete sentences)",
            "after": "Text after image (complete sentences)",
            "figure_caption": "Figure X: ..." or None
        }
    """
    # STAGE 1: Parameter validation
    if page is None:
        logging.error("page is None")
        return {"before": "", "after": "", "figure_caption": None}
    
    if image_bbox is None or not hasattr(image_bbox, 'y0'):
        logging.error("image_bbox is None or invalid")
        return {"before": "", "after": "", "figure_caption": None}
    
    if max_chars < 1:
        logging.warning(f"max_chars must be >= 1, got {max_chars}")
        max_chars = CONTEXT_MAX_CHARS
    
    try:
        text_dict = page.get_text("dict")
        blocks = text_dict.get("blocks", [])
    except Exception as e:
        logging.error(f"Failed to extract text from page: {e}")
        return {"before": "", "after": "", "figure_caption": None}

    # STAGE 3: Use helper function
    figure_caption = extract_figure_caption(blocks, image_bbox)

    # STAGE 3: DRY - use helper function for text collection
    text_with_positions = _collect_text_blocks_from_page(page)
    
    if not text_with_positions:
        return {"before": "", "after": "", "figure_caption": figure_caption}

    text_with_positions.sort(key=lambda x: x["y0"])

    # Separate text into before/after image based on y-coordinates
    before_items = [item for item in text_with_positions if item["y1"] < image_bbox.y0]
    after_items = [item for item in text_with_positions if item["y0"] > image_bbox.y1]
    
    # Group into paragraphs for better context
    before_paragraphs = _group_text_into_paragraphs(before_items)
    after_paragraphs = _group_text_into_paragraphs(after_items)
    
    # Extract context_before: last paragraph(s) before image
    before_text = ""
    if before_paragraphs:
        # Take last paragraph and apply sentence boundary
        last_para = before_paragraphs[-1][0]
        before_text = _extract_sentence_boundary_from_text(last_para, max_chars, from_end=True)
    
    # Fallback: if empty and we have doc/page_num, try previous page
    if not before_text and doc is not None and page_num > 0:
        before_text = _extract_context_from_previous_page(doc, page_num, max_chars)
    
    # Extract context_after: first paragraph(s) after image
    after_text = ""
    if after_paragraphs:
        # Take first paragraph and apply sentence boundary
        first_para = after_paragraphs[0][0]
        after_text = _extract_sentence_boundary_from_text(first_para, max_chars, from_end=False)
    
    return {
        "before": before_text,
        "after": after_text,
        "figure_caption": figure_caption
    }
