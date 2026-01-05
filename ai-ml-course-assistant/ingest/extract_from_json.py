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
    """Download image from URL and save to output_path. Returns image metadata or None."""
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


def find_context_for_image(full_text: str, caption: str, alt_text: str, image_index: int, context_chars: int = 200) -> tuple:
    """
    Find surrounding context for an image in text.
    
    Strategy:
    1. Try to find caption in text
    2. If not found, use image index to estimate position (divide text by number of images)
    
    Args:
        full_text: Complete document text
        caption: Image caption (if available)
        alt_text: Image alt text (if available)
        image_index: Position of image in document (1-based)
        context_chars: Number of characters to extract before/after
    
    Returns:
        (context_before, context_after) tuple
    """
    if not full_text:
        return ("", "")

    search_text = caption if caption else alt_text
    position = -1
    
    if search_text:
        # Clean up search text for better matching
        search_text_clean = search_text.strip()[:100]  # Use first 100 chars for search
        
        position = full_text.find(search_text_clean)
        
        if position == -1:
            # Try case-insensitive search
            position = full_text.lower().find(search_text_clean.lower())
    
    # If caption not found, estimate position by image index
    if position == -1:
        # Assume images are roughly evenly distributed in text
        # Use image_index to estimate position
        text_length = len(full_text)
        estimated_position = int((image_index / 10) * text_length)  # Assume ~10 images max
        
        # Take a chunk around estimated position
        start = max(0, estimated_position - context_chars)
        end = min(text_length, estimated_position + context_chars)
        
        context_before = full_text[start:estimated_position].strip()
        context_after = full_text[estimated_position:end].strip()
        
        return (context_before, context_after)
    
    # Extract context before
    start = max(0, position - context_chars)
    context_before = full_text[start:position].strip()
    
    # Extract context after (skip the caption itself)
    if search_text:
        caption_end = position + len(search_text.strip()[:100])
    else:
        caption_end = position
    end = min(len(full_text), caption_end + context_chars)
    context_after = full_text[caption_end:end].strip()
    
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
