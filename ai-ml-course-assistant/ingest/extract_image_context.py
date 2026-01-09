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
from typing import Dict, List, Optional, Tuple
import fitz  


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
    caption_pattern = r'^(Figure|Fig\.|Table)\s+\d+:?\s*'
    
    for block in text_blocks:
        if 'lines' not in block:
            continue
            
        for line in block['lines']:
            for span in line['spans']:
                text = span['text'].strip()

                if re.match(caption_pattern, text, re.IGNORECASE):
                    span_rect = fitz.Rect(span['bbox'])

                    vertical_distance = min(
                        abs(span_rect.y0 - image_bbox.y1), 
                        abs(image_bbox.y0 - span_rect.y1)  
                    )
                    
                    if vertical_distance < 80:
                        full_caption = extract_full_caption_text(block, line)
                        return full_caption
    
    return None


def extract_full_caption_text(block: Dict, starting_line: Dict, max_length: int = 200) -> str:
    """
    Extract complete caption text from starting line and following lines.
    
    Args:
        block: Text block containing the caption
        starting_line: Line where caption starts
        max_length: Maximum characters to extract
        
    Returns:
        Full caption text
    """
    caption_text = ""
    start_collecting = False
    
    for line in block['lines']:
        if line == starting_line:
            start_collecting = True
        
        if start_collecting:
            line_text = " ".join([span['text'] for span in line['spans']])
            caption_text += line_text + " "

            if len(caption_text) > max_length:
                break
    
    return caption_text.strip()


def _extract_sentence_boundary_from_text(text: str, max_chars: int = 250, from_end: bool = False) -> str:
    """
    Extract text up to nearest sentence boundary.
    
    Args:
        text: Text to extract from
        max_chars: Maximum characters to consider
        from_end: If True, extract from end (for context_before), else from start (for context_after)
    
    Returns:
        Extracted text up to sentence boundary
    """
    if not text:
        return ""
    
    sentence_ends = ['. ', '.\n', '! ', '!\n', '? ', '?\n']
    
    if from_end:
        # For context_before: take last max_chars and find last sentence boundary
        chunk = text[-max_chars:] if len(text) > max_chars else text
        
        last_boundary = -1
        for end_marker in sentence_ends:
            idx = chunk.rfind(end_marker)
            if idx > last_boundary:
                last_boundary = idx
        
        if last_boundary != -1:
            # Return text after the boundary marker
            return chunk[last_boundary + 2:].strip()
        return chunk.strip()
    
    else:
        # For context_after: take first max_chars and find first sentence boundary
        chunk = text[:max_chars] if len(text) > max_chars else text
        
        first_boundary = len(chunk)
        for end_marker in sentence_ends:
            idx = chunk.find(end_marker)
            if idx != -1 and idx < first_boundary:
                first_boundary = idx + 1  # Include the period
        
        return chunk[:first_boundary].strip()


def _group_text_into_paragraphs(text_items: List[Dict]) -> List[Tuple[str, float, float]]:
    """
    Group text items into paragraphs based on y-coordinate proximity.
    
    Args:
        text_items: List of dicts with 'text', 'y0', 'y1' keys
    
    Returns:
        List of (paragraph_text, start_y, end_y) tuples
    """
    if not text_items:
        return []
    
    paragraphs = []
    current_para = []
    current_y_start = text_items[0]["y0"]
    current_y_end = text_items[0]["y1"]
    
    for i, item in enumerate(text_items):
        if not current_para:
            # Start new paragraph
            current_para.append(item["text"])
            current_y_start = item["y0"]
            current_y_end = item["y1"]
        else:
            # Check if this item continues the paragraph (y-distance < 5)
            y_gap = item["y0"] - current_y_end
            
            if y_gap < 5:  # Same paragraph
                current_para.append(item["text"])
                current_y_end = item["y1"]
            else:
                # Save current paragraph and start new one
                para_text = " ".join(current_para)
                paragraphs.append((para_text, current_y_start, current_y_end))
                
                current_para = [item["text"]]
                current_y_start = item["y0"]
                current_y_end = item["y1"]
    
    # Save last paragraph
    if current_para:
        para_text = " ".join(current_para)
        paragraphs.append((para_text, current_y_start, current_y_end))
    
    return paragraphs


def _extract_context_from_previous_page(doc, page_num: int, max_chars: int = 250) -> str:
    """
    Extract context from previous page when current page has empty context_before.
    
    Args:
        doc: PyMuPDF document object
        page_num: Current page number (0-based)
        max_chars: Maximum characters to extract
    
    Returns:
        Context text from previous page (last paragraph or sentence)
    """
    if page_num == 0:
        return ""
    
    try:
        prev_page = doc[page_num - 1]
        text_dict = prev_page.get_text("dict")
        blocks = text_dict.get("blocks", [])
        
        # Collect all text from previous page
        text_items = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_items.append({
                        "text": span["text"],
                        "y0": span["bbox"][1],
                        "y1": span["bbox"][3]
                    })
        
        if not text_items:
            return ""
        
        # Get all text from previous page
        full_text = " ".join([item["text"] for item in text_items])
        
        # Extract last sentence(s) up to max_chars
        return _extract_sentence_boundary_from_text(full_text, max_chars, from_end=True)
    
    except Exception:
        return ""


def extract_surrounding_context(
    page,
    image_bbox: fitz.Rect,
    doc=None,
    page_num: int = 0,
    max_chars: int = 250
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
    text_dict = page.get_text("dict")
    blocks = text_dict.get("blocks", [])

    figure_caption = extract_figure_caption(blocks, image_bbox)

    text_with_positions = []
    for block in blocks:
        if block.get("type") != 0:  
            continue
        
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text_with_positions.append({
                    "text": span["text"],
                    "y0": span["bbox"][1],  # Top y-coordinate
                    "y1": span["bbox"][3]   # Bottom y-coordinate
                })

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


def extract_figure_references(text: str) -> List[str]:
    """
    Find references to figures/tables in text.
    
    Detects patterns like:
    - "see Figure 1"
    - "(Figure 2)"
    - "as shown in Table 3"
    - "Fig. 4 illustrates"
    
    Args:
        text: Text to search for references
        
    Returns:
        List of figure/table references (e.g., ["Figure 1", "Table 2"])
    """
    pattern = r'\b(Figure|Fig\.|Table)\s+\d+\b'
    matches = re.findall(pattern, text, re.IGNORECASE)

    normalized = []
    for match in matches:
        if match.lower().startswith("fig"):
            normalized.append(re.sub(r'^Fig\.', 'Figure', match, flags=re.IGNORECASE))
        else:
            normalized.append(match)

    seen = set()
    unique_refs = []
    for ref in normalized:
        if ref not in seen:
            seen.add(ref)
            unique_refs.append(ref)
    
    return unique_refs


if __name__ == "__main__":
    # Test with a sample PDF
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_image_context.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]

        image_list = page.get_images(full=True)
        
        for img_idx, img_info in enumerate(image_list):
            print(f"\nPage {page_num + 1}, Image {img_idx + 1}:")
            
            # For demo, use page center as fake image position
            image_bbox = fitz.Rect(100, 200, 400, 500)
            
            context = extract_surrounding_context(
                page, 
                image_bbox, 
                doc=doc, 
                page_num=page_num
            )
            
            print(f"Caption: {context['figure_caption']}")
            print(f"Before: {context['before'][:100]}...")
            print(f"After: {context['after'][:100]}...")
    
    doc.close()
