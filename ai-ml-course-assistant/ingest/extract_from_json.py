"""
JSON document extraction - extracts text and downloads images from RealPython/Medium JSON files.

This script handles JSON-formatted documents (RealPython, Medium/TDS) that have:
1. Text content already extracted
2. Image URLs that need to be downloaded
3. Metadata and structure information

Usage:
    from ingest.extract_from_json import extract_json_document
    result = extract_json_document("realpython_numpy-tutorial")
"""

import os
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

try:
    from PIL import Image
except ImportError:
    logging.error("Pillow not installed. Run: pip install Pillow")
    exit(1)

import io

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

DEFAULT_RAW_REALPYTHON_DIR = Path(__file__).parent.parent / "data" / "raw" / "realpython"
DEFAULT_RAW_MEDIUM_DIR = Path(__file__).parent.parent / "data" / "raw" / "medium"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed" / "images"
MIN_IMAGE_SIZE = 100


def detect_source_type(doc_id: str) -> str:
    """Detect source type from doc_id prefix."""
    if doc_id.startswith("realpython_"):
        return "realpython"
    elif doc_id.startswith("medium_"):
        return "medium"
    else:
        raise ValueError(f"Unknown source type for doc_id: {doc_id}. Expected 'realpython_' or 'medium_' prefix.")


def find_json_file(doc_id: str) -> Optional[Path]:
    """Find JSON file for given doc_id in raw/realpython or raw/medium directories."""
    source_type = detect_source_type(doc_id)
    
    if source_type == "realpython":
        base_dir = DEFAULT_RAW_REALPYTHON_DIR
    elif source_type == "medium":
        base_dir = DEFAULT_RAW_MEDIUM_DIR
    else:
        return None
    
    # Extract slug from doc_id (e.g., "realpython_numpy-tutorial" -> "numpy-tutorial")
    slug = doc_id.replace(f"{source_type}_", "", 1)
    
    # Look for directory with slug name containing JSON file
    doc_dir = base_dir / slug
    if doc_dir.exists() and doc_dir.is_dir():
        json_file = doc_dir / f"{slug}.json"
        if json_file.exists():
            return json_file
    
    logging.error(f"JSON file not found for doc_id: {doc_id} at {doc_dir}")
    return None


def download_image_from_url(url: str, output_path: Path) -> Optional[Dict]:
    """
    Download image from URL and save to output_path.
    
    Error Handling Strategy: Returns None for recoverable failures.
    - Network errors (timeout, 404) → None (skip this image, continue processing)
    - Image format errors → None (invalid image, skip it)
    - Caller should check for None and handle gracefully
    
    Args:
        url: Image URL to download
        output_path: Path to save image
    
    Returns:
        Image metadata dict (width, height, format, size) or None if download fails
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))
        width, height = image.size
        format_str = image.format.lower() if image.format else "png"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        return {
            "width": width,
            "height": height,
            "format": format_str,
            "size_bytes": len(response.content)
        }
    
    except requests.RequestException as e:
        logging.warning(f"Failed to download image from {url}: {e}")
        return None
    except Exception as e:
        logging.warning(f"Error processing image from {url}: {e}")
        return None


def _should_skip_context(caption: str, vlm_description: str = "") -> bool:
    """
    Determine if context should be skipped for this image.
    
    Skip context when:
    1. Caption indicates decorative/non-technical image ("Image by author", generic captions)
    2. VLM description indicates non-technical content
    
    Args:
        caption: Image caption/alt text
        vlm_description: Optional VLM-generated description
    
    Returns:
        True if context should be skipped, False otherwise
    """
    # Generic captions that don't add technical value
    generic_patterns = [
        "image by author",
        "photo by",
        "courtesy of",
        "source:",
        "illustration",
        "screenshot",
        "example of",
    ]
    
    caption_lower = caption.lower() if caption else ""
    
    # Check for generic captions
    for pattern in generic_patterns:
        if pattern in caption_lower:
            # If VLM says "not technical" or mentions animals/decorative, skip context
            if vlm_description:
                vlm_lower = vlm_description.lower()
                non_technical_indicators = [
                    "not a technical",
                    "not technical",
                    "photograph",
                    "kitten",
                    "kitty",
                    "cat",
                    "dog",
                    "animal",
                    "natural scene",
                    "not contain any ai/ml",
                    "does not contain",
                    "not depict"
                ]
                for indicator in non_technical_indicators:
                    if indicator in vlm_lower:
                        return True
    
    return False


def _extract_sentence_boundary(text: str, position: int, direction: str = "before") -> str:
    """
    Extract text up to nearest sentence boundary from position.
    
    Args:
        text: Full text
        position: Starting position
        direction: "before" or "after" the position
    
    Returns:
        Extracted text up to sentence boundary
    """
    sentence_ends = ['. ', '.\n', '! ', '!\n', '? ', '?\n']
    
    if direction == "before":
        # Search backwards for sentence start
        search_start = max(0, position - 250)  # Look back max 400 chars
        chunk = text[search_start:position]
        
        # Find last sentence boundary
        last_boundary = -1
        for end_marker in sentence_ends:
            idx = chunk.rfind(end_marker)
            if idx > last_boundary:
                last_boundary = idx
        
        if last_boundary != -1:
            return chunk[last_boundary + 2:].strip()  # Skip '. ' or similar
        return chunk.strip()
    
    else:  # direction == "after"
        # Search forward for sentence end
        search_end = min(len(text), position + 250)  # Look forward max 400 chars
        chunk = text[position:search_end]
        
        # Find first sentence boundary
        first_boundary = len(chunk)
        for end_marker in sentence_ends:
            idx = chunk.find(end_marker)
            if idx != -1 and idx < first_boundary:
                first_boundary = idx + 1  # Include the period
        
        return chunk[:first_boundary].strip()


def _find_position_by_keywords(full_text: str, caption: str, alt_text: str, image_index: int) -> int:
    """
    Find image position using keyword search and heuristics.
    
    Args:
        full_text: Complete document text
        caption: Image caption
        alt_text: Image alt text
        image_index: Image position in document (1-based)
    
    Returns:
        Position in text (-1 if not found)
    """
    # Try multiple search strategies
    search_texts = []
    
    if caption:
        # Extract key technical terms from caption
        caption_words = caption.split()
        if len(caption_words) > 3:
            # Try first/last few words
            search_texts.append(" ".join(caption_words[:5]))
            search_texts.append(" ".join(caption_words[-5:]))
        search_texts.append(caption.strip()[:100])
    
    if alt_text:
        search_texts.append(alt_text.strip()[:100])
    
    # Try each search text
    for search_text in search_texts:
        if not search_text:
            continue
            
        # Try exact match
        pos = full_text.find(search_text)
        if pos != -1:
            return pos
        
        # Try case-insensitive
        pos = full_text.lower().find(search_text.lower())
        if pos != -1:
            return pos
    
    # Fallback: estimate by document position
    # Split text into paragraphs and estimate which paragraph
    paragraphs = full_text.split('\n\n')
    if len(paragraphs) > image_index:
        # Find start of the paragraph that likely contains this image
        target_para = paragraphs[image_index - 1] if image_index > 0 else paragraphs[0]
        return full_text.find(target_para)
    
    # Last resort: proportional estimation
    text_length = len(full_text)
    return int((image_index / (image_index + 5)) * text_length)


def find_context_for_image(full_text: str, caption: str, alt_text: str, image_index: int, context_chars: int = 200, vlm_description: str = "") -> tuple:
    """
    Find surrounding context for image in document text with smart boundary detection.
    
    Improvements over original:
    1. Skips context for decorative images (detected via caption/VLM patterns)
    2. Uses sentence boundaries instead of fixed character count
    3. Better position finding with keyword extraction and paragraph estimation
    4. Handles "Image by author" and other generic captions intelligently
    
    Strategy:
    1. Check if context should be skipped (decorative images)
    2. Try to find caption in text using multiple strategies
    3. If not found, use image index to estimate position (divide text by number of images)
    
    Args:
        full_text: Complete document text
        caption: Image caption (if available)
        alt_text: Image alt text (if available)
        image_index: Position of image in document (1-based)
        context_chars: Approximate characters to extract (actual uses sentence boundaries)
        vlm_description: Optional VLM-generated description for relevance checking
    
    Returns:
        (context_before, context_after) tuple. Empty strings if context should be skipped.
    """
    if not full_text:
        return ("", "")
    
    # Check if context should be skipped for decorative images
    if _should_skip_context(caption, vlm_description):
        return ("", "")
    
    # Find position using improved keyword search
    position = _find_position_by_keywords(full_text, caption, alt_text, image_index)
    
    if position == -1 or position >= len(full_text):
        # If still not found, return empty context
        return ("", "")
    
    # Extract context using sentence boundaries
    context_before = _extract_sentence_boundary(full_text, position, "before")
    
    # For "after" context, skip the caption/alt text if found at this position
    skip_length = 0
    if caption:
        caption_at_pos = full_text[position:position + len(caption)]
        if caption.lower() in caption_at_pos.lower():
            skip_length = len(caption)
    
    context_after = _extract_sentence_boundary(full_text, position + skip_length, "after")
    
    return (context_before, context_after)


def extract_images_from_json(
    json_data: Dict,
    doc_id: str,
    doc_output_dir: Path,
    min_size: int = MIN_IMAGE_SIZE
) -> List[Dict]:
    """Download images from URLs in JSON and create metadata with surrounding context."""
    images_metadata = []
    
    images = json_data.get("content", {}).get("images", [])
    full_text = json_data.get("content", {}).get("text", "")
    
    for img_data in images:
        img_index = img_data.get("index", 0)
        img_url = img_data.get("url", "")
        alt_text = img_data.get("alt_text", "")
        caption = img_data.get("caption", "")
        
        if not img_url:
            continue
        
        # Determine extension from URL
        url_lower = img_url.lower()
        if ".png" in url_lower:
            ext = "png"
        elif ".jpg" in url_lower or ".jpeg" in url_lower:
            ext = "jpg"
        elif ".gif" in url_lower:
            ext = "gif"
        elif ".webp" in url_lower:
            ext = "webp"
        else:
            ext = "png"  
        
        # Generate image ID and filename
        image_id = f"{doc_id}_web_{img_index:03d}"
        image_filename = f"{image_id}.{ext}"
        image_path = doc_output_dir / image_filename
        
        # Download image
        img_info = download_image_from_url(img_url, image_path)
        
        if img_info is None:
            logging.warning(f"Skipping image {img_index} from {img_url}")
            continue
        
        # Skip if too small
        if img_info["width"] < min_size or img_info["height"] < min_size:
            logging.info(f"Skipping small image {img_index}: {img_info['width']}x{img_info['height']}")
            image_path.unlink(missing_ok=True)  # Delete downloaded file
            continue
        
        # Extract surrounding context from full text
        context_before, context_after = find_context_for_image(
            full_text=full_text,
            caption=caption,
            alt_text=alt_text,
            image_index=img_index,
            context_chars=200
        )
        
        # Create metadata entry
        metadata = {
            "image_id": image_id,
            "doc_id": doc_id,
            "filepath": str(image_path.absolute()),
            "filename": image_filename,
            "source_url": img_url,
            "image_index": img_index,
            "width": img_info["width"],
            "height": img_info["height"],
            "format": img_info["format"],
            "size_bytes": img_info["size_bytes"],
            "extraction_method": "web_download",
            "extracted_at": datetime.now().isoformat(),
            "author_caption": caption if caption else alt_text if alt_text else None,
            "context_before": context_before,
            "context_after": context_after
        }
        
        images_metadata.append(metadata)
        logging.info(f"Downloaded image {img_index}: {img_info['width']}x{img_info['height']} from {img_url}")
    
    return images_metadata


def extract_json_document(doc_id: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Dict:
    """
    Extract document from JSON file (RealPython or Medium).
    
    Args:
        doc_id: Document ID (e.g., "realpython_numpy-tutorial" or "medium_agents-plan-tasks")
        output_dir: Output directory for images (default: data/processed/images)
    
    Returns:
        Dict with keys:
            - doc_id: str
            - images_count: int
            - text_length: int
            - full_text: str (in-memory)
            - images_metadata: List[Dict]
            - document_metadata: Dict
    """
    logging.info(f"Extracting JSON document: {doc_id}")

    json_file = find_json_file(doc_id)
    if json_file is None:
        raise FileNotFoundError(f"JSON file not found for doc_id: {doc_id}")

    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    full_text = json_data.get("content", {}).get("text", "")

    doc_output_dir = output_dir / doc_id
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    images_metadata = extract_images_from_json(json_data, doc_id, doc_output_dir)

    document_metadata = {
        "doc_id": doc_id,
        "title": json_data.get("title", ""),
        "url": json_data.get("url", ""),
        "source_type": json_data.get("source_type", ""),
        "downloaded_at": json_data.get("downloaded_at", ""),
        "stats": json_data.get("stats", {})
    }
    
    logging.info(f"Extracted {len(images_metadata)} images, {len(full_text)} chars from {doc_id}")

    metadata_file = output_dir.parent / "images_metadata.json"

    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
    else:
        all_metadata = []

    all_metadata = [m for m in all_metadata if m.get("doc_id") != doc_id]
    
    # Add new metadata
    all_metadata.extend(images_metadata)
    
    # Save updated metadata
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    
    return {
        "doc_id": doc_id,
        "images_count": len(images_metadata),
        "text_length": len(full_text),
        "full_text": full_text,
        "images_metadata": images_metadata,
        "document_metadata": document_metadata
    }


def main():
    """Command-line interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract text and images from JSON documents")
    parser.add_argument("--doc-id", required=True, help="Document ID (e.g., realpython_numpy-tutorial)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for images")
    
    args = parser.parse_args()
    
    result = extract_json_document(args.doc_id, Path(args.output))
    
    print(f"\n✅ Extraction complete:")
    print(f"   - Images: {result['images_count']}")
    print(f"   - Text length: {result['text_length']} chars")
    print(f"   - Metadata saved to: {DEFAULT_OUTPUT_DIR.parent / 'images_metadata.json'}")


if __name__ == "__main__":
    main()
