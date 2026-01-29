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
import re
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    from PIL import Image
except ImportError:
    logging.error("Pillow not installed. Run: pip install Pillow")
    exit(1)

import io
from utils.logging_config import setup_logging

setup_logging()

DEFAULT_RAW_REALPYTHON_DIR = Path(__file__).parent.parent / "data" / "raw" / "realpython"
DEFAULT_RAW_MEDIUM_DIR = Path(__file__).parent.parent / "data" / "raw" / "medium"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed" / "images"

# Image processing constants
MIN_IMAGE_SIZE = 100
MIN_IMAGE_DIMENSION = 50  # Additional safety check
REQUEST_TIMEOUT_SECONDS = 10

# Context extraction constants
CONTEXT_LOOK_BACK_CHARS = 250  # Characters to search backward for sentence boundary
CONTEXT_LOOK_FORWARD_CHARS = 250  # Characters to search forward for sentence boundary
CONTEXT_DEFAULT_CHARS = 200  # Default context characters to extract
SKIP_CAPTION_MAX_LENGTH = 100  # Max length of caption to match in text
POSITION_ESTIMATE_BUFFER = 5  # Buffer for proportional position estimation

# Sentence boundary markers
SENTENCE_END_MARKERS = ['. ', '.\n', '! ', '!\n', '? ', '?\n']
SENTENCE_SPLIT_PATTERN = r'(?<=[.!?])\s+(?=[A-ZА-ЯІЄЇ])'

# Generic caption patterns that indicate non-technical/decorative images
GENERIC_CAPTION_PATTERNS = [
    "image by author",
    "photo by",
    "courtesy of",
    "source:",
    "illustration",
    "screenshot",
    "example of",
]

# Non-technical indicators from VLM descriptions
NON_TECHNICAL_INDICATORS = [
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


def detect_source_type(doc_id: str) -> str:
    """
    Detect source type from doc_id prefix.

    """
    if not doc_id or not isinstance(doc_id, str):
        raise ValueError(f"doc_id must be non-empty string, got: {type(doc_id).__name__}")
    
    if doc_id.startswith("realpython_"):
        return "realpython"
    elif doc_id.startswith("medium_"):
        return "medium"
    else:
        raise ValueError(f"Unknown source type for doc_id: {doc_id}. Expected 'realpython_' or 'medium_' prefix.")


def _validate_json_file(json_file: Path) -> Dict:
    """
    Load and validate JSON file with proper error handling.
    
    Args:
        json_file: Path to JSON file
        
    Returns:
        Parsed JSON data as dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
    """
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, OSError) as e:
        raise FileNotFoundError(f"Failed to read JSON file {json_file}: {type(e).__name__}: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {json_file}: {e}")



def find_json_file(doc_id: str) -> Path:
    """
    Find JSON file for given doc_id in raw/realpython or raw/medium directories.
    
    Args:
        doc_id: Document identifier
        
    Returns:
        Path to JSON file
        
    Raises:
        ValueError: If doc_id is invalid
        FileNotFoundError: If JSON file not found
    """
    source_type = detect_source_type(doc_id)
    
    base_dir = DEFAULT_RAW_REALPYTHON_DIR if source_type == "realpython" else DEFAULT_RAW_MEDIUM_DIR
    
    # Extract slug from doc_id (e.g., "realpython_numpy-tutorial" -> "numpy-tutorial")
    slug = doc_id.replace(f"{source_type}_", "", 1)
    
    # Look for directory with slug name containing JSON file
    doc_dir = base_dir / slug
    json_file = doc_dir / f"{slug}.json"
    
    if not json_file.exists():
        raise FileNotFoundError(
            f"JSON file not found for doc_id: {doc_id}\n"
            f"Expected path: {json_file}\n"
            f"Source type: {source_type}, Base dir: {base_dir}"
        )
    
    return json_file


def download_image_from_url(url: str, output_path: Path) -> Optional[Dict]:
    """
    Download image from URL and save to output_path.
    
    Args:
        url: Image URL to download
        output_path: Path to save image
    
    Returns:
        Image metadata dict (width, height, format, size) or None if download fails
    """
    if not url or not isinstance(url, str):
        logging.warning(f"Invalid URL: {url}")
        return None
    
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))
        width, height = image.size
        format_str = image.format.lower() if image.format else "png"

        # STAGE 1: Validation - check minimum image dimensions
        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            logging.warning(f"Image too small ({width}x{height}): {url}")
            return None

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
    if not caption or not isinstance(caption, str):
        return False
    
    caption_lower = caption.lower()
    
    # Check for generic captions
    for pattern in GENERIC_CAPTION_PATTERNS:
        if pattern in caption_lower:
            # If VLM says "not technical" or mentions animals/decorative, skip context
            if vlm_description and isinstance(vlm_description, str):
                vlm_lower = vlm_description.lower()
                for indicator in NON_TECHNICAL_INDICATORS:
                    if indicator in vlm_lower:
                        return True
    
    return False


'''def _extract_sentence_boundary(text: str, position: int, direction: str = "before") -> str:
    """
    Extract text up to nearest sentence boundary from position.
    
    Args:
        text: Full text
        position: Starting position
        direction: "before" or "after" the position
    
    Returns:
        Extracted text up to sentence boundary
    """
    if not text or not isinstance(text, str):
        return ""
    
    # STAGE 1: Ensure position is within bounds
    text_len = len(text)
    position = max(0, min(position, text_len))
    
    if direction == "before":
        # Search backwards for sentence start
        search_start = max(0, position - CONTEXT_LOOK_BACK_CHARS)
        chunk = text[search_start:position]
        
        if not chunk:
            return ""
        
        # Find last sentence boundary
        last_boundary = -1
        for end_marker in SENTENCE_END_MARKERS:
            idx = chunk.rfind(end_marker)
            if idx > last_boundary:
                last_boundary = idx
        
        if last_boundary != -1:
            return chunk[last_boundary + 2:].strip()  # Skip end marker
        return chunk.strip()
    
    else:  # direction == "after"
        # Search forward for sentence end
        search_end = min(text_len, position + CONTEXT_LOOK_FORWARD_CHARS)
        chunk = text[position:search_end]
        
        if not chunk:
            return ""
        
        # Find first sentence boundary
        first_boundary = len(chunk)
        for end_marker in SENTENCE_END_MARKERS:
            idx = chunk.find(end_marker)
            if idx != -1 and idx < first_boundary:
                first_boundary = idx + 1
        
        return chunk[:first_boundary].strip()'''



def _extract_sentence_boundary(text: str, position: int, direction: str = "before") -> str:
    if not text or not isinstance(text, str):
        return ""

    text_len = len(text)
    position = max(0, min(position, text_len))

    if direction == "before":
        start = max(0, position - CONTEXT_LOOK_BACK_CHARS)
        chunk = text[start:position]
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        if sentences:
            return sentences[-1].strip() 

    else:
        end = min(text_len, position + CONTEXT_LOOK_FORWARD_CHARS)
        chunk = text[position:end]
        match = re.search(r'[^.!?]+[.!?]', chunk)
        if match:
            return match.group().strip()
        return chunk.strip()


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
    if not full_text or not isinstance(full_text, str):
        return -1
    
    if image_index < 1:
        image_index = 1

    def _try_search(search_text: str) -> int:
        if not search_text or not isinstance(search_text, str):
            return -1
        
        search_text = search_text.strip()[:SKIP_CAPTION_MAX_LENGTH]
        if not search_text:
            return -1

        pos = full_text.find(search_text)
        if pos != -1:
            return pos

        pos = full_text.lower().find(search_text.lower())
        return pos if pos != -1 else -1

    search_texts = []
    
    if caption and isinstance(caption, str):
        caption_words = caption.split()
        if len(caption_words) > 3:
            search_texts.append(" ".join(caption_words[:5]))
            search_texts.append(" ".join(caption_words[-5:]))
        search_texts.append(caption)
    
    if alt_text and isinstance(alt_text, str):
        search_texts.append(alt_text)

    for search_text in search_texts:
        pos = _try_search(search_text)
        if pos != -1:
            return pos

    paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
    if paragraphs and len(paragraphs) > 0:
        para_index = min(image_index - 1, len(paragraphs) - 1)
        target_para = paragraphs[para_index] 
        pos = full_text.find(target_para)
        if pos != -1:
            return pos
    return -1


def find_context_for_image(
    full_text: str, 
    caption: str, 
    alt_text: str, 
    image_index: int, 
    context_chars: int = CONTEXT_DEFAULT_CHARS, 
    vlm_description: str = ""
) -> Tuple[str, str]:
    """
    Find surrounding context for image in document text with smart boundary detection.

    Improvements:
    1. Skips context for decorative images (detected via caption/VLM patterns)
    2. Uses sentence boundaries instead of fixed character count
    3. Better position finding with keyword extraction and paragraph estimation
    4. Handles "Image by author" and other generic captions intelligently
    
    Strategy:
    1. Check if context should be skipped (decorative images)
    2. Try to find caption in text using multiple strategies
    3. If not found, use image index to estimate position
    
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
    # STAGE 1: Validation
    if not full_text or not isinstance(full_text, str):
        return ("", "")
    
    if image_index < 1:
        image_index = 1
    
    # Check if context should be skipped for decorative images
    if _should_skip_context(caption, vlm_description):
        return ("", "")
    
    # Find position using improved keyword search
    position = _find_position_by_keywords(full_text, caption, alt_text, image_index)
    
    if position < 0 or position >= len(full_text):
        # If still not found, return empty context
        return ("", "")
    
    # Extract context using sentence boundaries
    context_before = _extract_sentence_boundary(full_text, position, "before")
    
    # For "after" context, skip the caption/alt text if found at this position
    skip_length = 0
    if caption and isinstance(caption, str):
        caption_part = caption[:SKIP_CAPTION_MAX_LENGTH]
        text_at_pos = full_text[position:position + len(caption_part)].lower()
        if caption_part.lower() in text_at_pos:
            skip_length = len(caption_part)
    
    context_after = _extract_sentence_boundary(full_text, position + skip_length, "after")
    
    return (context_before, context_after)


def _determine_image_extension(url: str) -> str:
    """
    Extract extension detection into helper function
    
    Args:
        url: Image URL
        
    Returns:
        File extension (without dot)
    """
    if not url or not isinstance(url, str):
        return "png"
    
    url_lower = url.lower()
    
    # Check in order to avoid matching .jpeg before .jpg
    extensions = {
        ".png": "png",
        ".webp": "webp",
        ".gif": "gif",
        ".jpeg": "jpeg",
        ".jpg": "jpg",
    }
    
    for ext_marker, ext_name in extensions.items():
        if ext_marker in url_lower:
            return ext_name
    
    return "png"  # Default


def extract_images_from_json(
    json_data: Dict,
    doc_id: str,
    doc_output_dir: Path,
    min_size: int = MIN_IMAGE_SIZE
) -> List[Dict]:
    """
    Download images from URLs in JSON and create metadata with surrounding context.
    
    Args:
        json_data: Parsed JSON data
        doc_id: Document identifier
        doc_output_dir: Output directory for images
        min_size: Minimum image size in pixels
        
    Returns:
        List of image metadata dictionaries
    """
    # STAGE 1: Validation
    if not isinstance(json_data, dict):
        raise ValueError(f"json_data must be dict, got {type(json_data).__name__}")
    
    if not doc_id or not isinstance(doc_id, str):
        raise ValueError(f"doc_id must be non-empty string")
    
    if not isinstance(doc_output_dir, Path):
        doc_output_dir = Path(doc_output_dir)
    
    if min_size < MIN_IMAGE_DIMENSION:
        min_size = MIN_IMAGE_DIMENSION
    
    images_metadata = []
    
    images = json_data.get("content", {}).get("images", [])
    full_text = json_data.get("content", {}).get("text", "")
    
    # STAGE 1: Validation - check if images is iterable
    if not images:
        return images_metadata
    
    for img_data in images:
        if not isinstance(img_data, dict):
            logging.warning(f"Skipping non-dict image data: {type(img_data).__name__}")
            continue
        
        img_index = img_data.get("index", 0)
        img_url = img_data.get("url", "")
        alt_text = img_data.get("alt_text", "")
        caption = img_data.get("caption", "")
        
        if not img_url:
            continue

        ext = _determine_image_extension(img_url)
        
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
            image_path.unlink(missing_ok=True)
            continue
        
        # Extract surrounding context from full text
        context_before, context_after = find_context_for_image(
            full_text=full_text,
            caption=caption,
            alt_text=alt_text,
            image_index=img_index,
            context_chars=CONTEXT_DEFAULT_CHARS
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


def _save_images_metadata(images_metadata: List[Dict], doc_id: str, metadata_file: Path) -> None:
    """
    Extract metadata saving into helper function
    
    Args:
        images_metadata: List of image metadata to save
        doc_id: Document ID (for filtering duplicates)
        metadata_file: Path to metadata file
    """
    if not metadata_file.parent.exists():
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing metadata with proper error handling
    all_metadata = []
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)
        except (IOError, OSError) as e:
            logging.error(f"Failed to read metadata file {metadata_file}: {type(e).__name__}: {e}")
            all_metadata = []
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in metadata file {metadata_file}: {e}")
            all_metadata = []
    
    # Remove old entries for this doc_id
    all_metadata = [m for m in all_metadata if m.get("doc_id") != doc_id]
    
    # Add new metadata
    all_metadata.extend(images_metadata)
    
    # Save with proper error handling
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)
        logging.debug(f"Metadata saved: {len(all_metadata)} total entries")
    except (IOError, OSError) as e:
        logging.error(f"Failed to save metadata file {metadata_file}: {type(e).__name__}: {e}")
        raise


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
            
    Raises:
        ValueError: If doc_id is invalid
        FileNotFoundError: If JSON file not found
    """
    if not doc_id or not isinstance(doc_id, str):
        raise ValueError(f"doc_id must be non-empty string")
    
    logging.info(f"Extracting JSON document: {doc_id}")
    
    # Find and validate JSON file
    json_file = find_json_file(doc_id)

    # Load JSON with proper error handling
    json_data = _validate_json_file(json_file)

    full_text = json_data.get("content", {}).get("text", "")
    
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

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
    _save_images_metadata(images_metadata, doc_id, metadata_file)
    
    return {
        "doc_id": doc_id,
        "images_count": len(images_metadata),
        "text_length": len(full_text),
        "full_text": full_text,
        "images_metadata": images_metadata,
        "document_metadata": document_metadata
    }
