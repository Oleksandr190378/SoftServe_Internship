"""
Extract surrounding context for images - captures text around figures.

This module extracts ±200 characters of text before and after each image,
along with figure captions, to provide contextual information for enriched captions.
"""

import re
from typing import Dict, List, Optional
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
                    
                    if vertical_distance < 50:
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


def extract_surrounding_context(
    page,
    image_bbox: fitz.Rect,
    context_chars: int = 200
) -> Dict[str, str]:
    """
    Extract ±200 characters of text surrounding an image.
    
    This captures the narrative context that gives meaning to the image,
    such as "As shown in Figure 1, the model architecture..."
    
    Args:
        page: PyMuPDF page object
        image_bbox: Bounding rectangle of the image
        context_chars: Number of characters to extract before/after
        
    Returns:
        {
            "before": "Text before image (up to 200 chars)",
            "after": "Text after image (up to 200 chars)",
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

    image_center_y = (image_bbox.y0 + image_bbox.y1) / 2
    
    before_text = ""
    after_text = ""
    
    for item in text_with_positions:
        if item["y1"] < image_bbox.y0:  # Before image
            before_text += item["text"] + " "
        elif item["y0"] > image_bbox.y1:  # After image
            after_text += item["text"] + " "

    before_text = before_text.strip()
    after_text = after_text.strip()
    
    if len(before_text) > context_chars:
        before_text = "..." + before_text[-context_chars:]
    
    if len(after_text) > context_chars:
        after_text = after_text[:context_chars] + "..."
    
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
            
            context = extract_surrounding_context(page, image_bbox)
            
            print(f"Caption: {context['figure_caption']}")
            print(f"Before: {context['before'][:100]}...")
            print(f"After: {context['after'][:100]}...")
    
    doc.close()
