"""
Enrich images with context and Vision-LM descriptions.

This script combines:
1. generated visual descriptions
2. Author-provided figure captions
3. Surrounding text context (±200 chars)

Output: Updated images_metadata.json with enriched_caption field
"""

import json
import logging
import fitz  
from pathlib import Path
from typing import Dict, List, Optional
import sys
from dotenv import load_dotenv

load_dotenv()

from utils.logging_config import setup_logging
setup_logging()

from ingest.extract_image_context import extract_surrounding_context
from ingest.generate_captions import ImageCaptioner


# VLM and caption generation constants
VLM_MAX_CAPTION_TOKENS = 1024          # tokens - maximum length for caption generation
CONTEXT_MAX_CHARS = 250                # characters - maximum context size
CAPTION_PREVIEW_CHARS = 300            # characters - preview length for output

# Rate limiting constants (OpenAI API: 20 req/min)
RATE_LIMIT_REQUESTS_PER_MINUTE = 20    # OpenAI requests per minute limit
RATE_LIMIT_DELAY_SECONDS = 3.5         # delay between requests (1 req/3 sec)

# Extraction method constants
EXTRACTION_METHOD_PDF = 'embedded_raster'      # PDF extraction method
EXTRACTION_METHOD_WEB = 'web_download'         # Web extraction method

# Page indexing offset (PDF uses 1-based, fitz uses 0-based)
PDF_PAGE_OFFSET = 1

def _validate_image_metadata(img_meta: Dict) -> bool:
    """
    Args:
        img_meta: Image metadata dictionary
    
    Returns:
        True if metadata structure is valid, False otherwise
    """
    if not isinstance(img_meta, dict):
        logging.error(f"Image metadata must be dict, got {type(img_meta).__name__}")
        return False
    
    required_keys = ['doc_id', 'image_id']
    for key in required_keys:
        if key not in img_meta:
            logging.error(f"Image metadata missing required key: {key}")
            return False
    
    return True


def _validate_bbox_dict(bbox_dict: Optional[Dict]) -> bool:
    """
    Args:
        bbox_dict: Bounding box dictionary with x0, y0, x1, y1 keys
    
    Returns:
        True if bbox is valid, False otherwise
    """
    if not bbox_dict:
        return False
    
    if not isinstance(bbox_dict, dict):
        logging.error(f"bbox_dict must be dict, got {type(bbox_dict).__name__}")
        return False
    
    required_keys = ['x0', 'y0', 'x1', 'y1']
    for key in required_keys:
        if key not in bbox_dict:
            logging.error(f"bbox_dict missing required key: {key}")
            return False
        
        try:
            value = bbox_dict[key]
            if not isinstance(value, (int, float)):
                logging.error(f"bbox_dict[{key}] must be numeric, got {type(value).__name__}")
                return False
        except (TypeError, KeyError) as e:
            logging.error(f"Error validating bbox_dict[{key}]: {e}")
            return False
    
    return True


def _validate_page_bounds(doc: fitz.Document, page_num: int) -> bool:
    """
    Args:
        doc: PyMuPDF document
        page_num: Page number (1-based indexing)
    
    Returns:
        True if page number is valid, False otherwise
    """
    if not isinstance(page_num, int) or page_num < 1:
        logging.error(f"page_num must be positive integer, got {page_num}")
        return False
    
    if page_num > len(doc):
        logging.error(f"page_num {page_num} exceeds document length {len(doc)}")
        return False
    
    return True


def _extract_context_safely(
    page,
    bbox: fitz.Rect,
    doc: fitz.Document,
    page_num: int
) -> Dict[str, str]:
    """
     extract context extraction logic.
    
    Args:
        page: PyMuPDF page object
        bbox: Bounding box rectangle
        doc: PyMuPDF document (for cross-page fallback)
        page_num: Page number (0-based)
    
    Returns:
        Context dictionary with before/after/figure_caption keys
    """
    try:
        context = extract_surrounding_context(
            page,
            bbox,
            doc=doc,
            page_num=page_num,
            max_chars=CONTEXT_MAX_CHARS
        )
        return context if isinstance(context, dict) else {"before": "", "after": "", "figure_caption": None}
    except Exception as e:
        logging.error(f"Error extracting context: {type(e).__name__}: {e}")
        return {"before": "", "after": "", "figure_caption": None}


def _generate_vlm_description_safe(
    captioner: ImageCaptioner,
    image_path: str,
    image_id: str
) -> str:
    """
     - extract VLM generation logic.   
    Args:
        captioner: ImageCaptioner instance
        image_path: Path to image file
        image_id: Image identifier for logging
    
    Returns:
        VLM description or empty string on error
    """
    if not captioner:
        return ""
    
    try:
        description = captioner.generate_caption(image_path, max_length=VLM_MAX_CAPTION_TOKENS)
        return description if isinstance(description, str) else ""
    except Exception as e:
        logging.warning(f"VLM generation failed for {image_id}: {type(e).__name__}: {e}")
        return ""


def _assemble_enriched_caption_text(
    author_caption: str,
    vlm_description: str,
    context_before: str,
    context_after: str
) -> str:
    """
    Assemble enriched caption from multiple sources.
    
    Args:
        author_caption: Figure caption from paper
        vlm_description: Vision model generated description
        context_before: Text before image
        context_after: Text after image
    
    Returns:
        Combined enriched caption text
    """
    # STAGE 1: Validate inputs
    if not isinstance(author_caption, str):
        author_caption = ""
    if not isinstance(vlm_description, str):
        vlm_description = ""
    if not isinstance(context_before, str):
        context_before = ""
    if not isinstance(context_after, str):
        context_after = ""
    
    caption_parts = []
    
    # Add author caption
    if author_caption.strip():
        caption_parts.append(f"Figure caption: {author_caption.strip()}")
    
    # Add vision model description
    if vlm_description.strip():
        caption_parts.append(f"Visual description: {vlm_description.strip()}")
    
    # Add context
    context_text = ""
    if context_before.strip():
        context_text += context_before.strip()
    if context_after.strip():
        if context_text:
            context_text += " ... "
        context_text += context_after.strip()
    
    if context_text:
        caption_parts.append(f"Context: {context_text}")
    
    if caption_parts:
        caption_parts.append(
            "\nNote: Use only context text that is relevant to understanding this image. "
            "Ignore surrounding text if it discusses unrelated topics."
        )
        return "\n".join(caption_parts)
    
    return ""


def _load_and_validate_metadata(metadata_path: Path) -> Optional[List[Dict]]:
    """
    
    Load and validate images metadata from JSON file.
    
    Args:
        metadata_path: Path to images_metadata.json
    
    Returns:
        List of image metadata dicts, or None if loading fails
    """
    if not isinstance(metadata_path, Path):
        logging.error(f"metadata_path must be Path, got {type(metadata_path).__name__}")
        return None
    
    if not metadata_path.exists():
        logging.error(f"Metadata file not found: {metadata_path}")
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logging.error(f"Metadata root must be list, got {type(data).__name__}")
            return None
        
        logging.info(f"Loaded {len(data)} images from {metadata_path.name}")
        return data
    
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {metadata_path}: {e}")
        return None
    except IOError as e:
        logging.error(f"Failed to read {metadata_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error loading metadata: {type(e).__name__}: {e}")
        return None



def load_images_metadata(metadata_path: Path) -> Optional[List[Dict]]:
    """
    Load images metadata from JSON file.
    
    Args:
        metadata_path: Path to JSON file
    
    Returns:
        List of image metadata dicts, or None if loading fails
    """
    return _load_and_validate_metadata(metadata_path)


def save_images_metadata(metadata: List[Dict], output_path: Path):
    """
    Save updated images metadata to JSON file.
    
    STAGE 1: Validation - validate input parameters
    """
    # Parameter validation
    if not isinstance(metadata, list):
        logging.error(f"metadata must be list, got {type(metadata).__name__}")
        return
    
    if not isinstance(output_path, Path):
        logging.error(f"output_path must be Path, got {type(output_path).__name__}")
        return
    
    try:
        logging.info(f"Saving {len(metadata)} images to {output_path}")
        
        # DEBUG: Check sample before saving
        if metadata:
            sample = metadata[0]
            logging.debug(f"Sample image before save: {sample.get('image_id')}, has enriched_caption: {'enriched_caption' in sample}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved enriched metadata to {output_path.name}")
    
    except IOError as e:
        logging.error(f"Failed to save metadata to {output_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error saving metadata: {type(e).__name__}: {e}")


def enrich_single_image(
    img_meta: Dict,
    pdf_docs: Dict[str, fitz.Document],
    raw_papers_dir: Path,
    captioner=None
) -> Dict:
    """
    Enrich a single image with context and VLM description.
    
    Args:
        img_meta: Image metadata dict with bbox coordinates (optional for web images)
        pdf_docs: Cache of opened PDF documents {doc_id: Document}
        raw_papers_dir: Path to raw papers directory
        captioner: Optional ImageCaptioner instance for VLM descriptions
        
    Returns:
        Updated metadata dict with enriched fields
    """
    # STAGE 1: Parameter validation
    if not _validate_image_metadata(img_meta):
        return img_meta
    
    if not isinstance(pdf_docs, dict):
        logging.error(f"pdf_docs must be dict, got {type(pdf_docs).__name__}")
        return img_meta
    
    if not isinstance(raw_papers_dir, Path):
        logging.error(f"raw_papers_dir must be Path, got {type(raw_papers_dir).__name__}")
        return img_meta
    
    doc_id = img_meta.get('doc_id')
    extraction_method = img_meta.get('extraction_method', EXTRACTION_METHOD_PDF)
    
    # For web-downloaded images (RealPython/Medium), use stored context
    if extraction_method == EXTRACTION_METHOD_WEB:
        # Get stored context (from extract_from_json.py)
        context_before = img_meta.get('context_before', '')
        context_after = img_meta.get('context_after', '')
        author_caption = img_meta.get('author_caption', '')
        
        # Generate VLM description if requested
        vlm_description = ""
        if captioner and img_meta.get('filepath'):
            image_path = img_meta.get('filepath')
            if image_path and Path(image_path).exists():
                vlm_description = _generate_vlm_description_safe(captioner, image_path, img_meta.get('image_id', 'unknown'))
        
        # Use helper function to assemble caption
        enriched_caption = _assemble_enriched_caption_text(
            author_caption,
            vlm_description,
            context_before,
            context_after
        )
        
        if not enriched_caption:
            enriched_caption = "Image from web article (no caption or context available)"
        
        img_meta['enriched_caption'] = enriched_caption
        img_meta['vlm_description'] = vlm_description
        
        return img_meta
    
    # For PDF images, extract context from page
    page_num = img_meta.get('page_num')
    bbox_dict = img_meta.get('bbox')
    
    # Validate page_num and bbox_dict
    if not isinstance(page_num, int) or page_num < 1:
        logging.warning(f"Skipping {img_meta.get('image_id', 'unknown')} - invalid page_num: {page_num}")
        return img_meta
    
    if not _validate_bbox_dict(bbox_dict):
        logging.warning(f"Skipping {img_meta.get('image_id', 'unknown')} - invalid bbox_dict")
        return img_meta

    # Open PDF if not already cached
    if doc_id not in pdf_docs:
        pdf_path = raw_papers_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            logging.error(f"PDF not found: {pdf_path}")
            return img_meta
        
        try:
            pdf_docs[doc_id] = fitz.open(str(pdf_path))
        except Exception as e:
            logging.error(f"Failed to open PDF {pdf_path}: {type(e).__name__}: {e}")
            return img_meta
    
    doc = pdf_docs[doc_id]
    
    # Validate page bounds
    if not _validate_page_bounds(doc, page_num):
        return img_meta
    
    try:
        page = doc[page_num - PDF_PAGE_OFFSET]
        
        # Validate and create bbox
        bbox = fitz.Rect(
            float(bbox_dict['x0']),
            float(bbox_dict['y0']),
            float(bbox_dict['x1']),
            float(bbox_dict['y1'])
        )
        
        # Use helper for context extraction
        context = _extract_context_safely(page, bbox, doc, page_num - PDF_PAGE_OFFSET)
        
        # Use helper for VLM generation
        image_path = img_meta.get('filepath')
        vlm_description = ""
        if captioner and image_path:
            vlm_description = _generate_vlm_description_safe(captioner, image_path, img_meta.get('image_id', 'unknown'))
        else:
            # Preserve existing VLM description if --no-vlm is used
            vlm_description = img_meta.get('vlm_description', '')
        
        author_caption = context.get('figure_caption', '')
        context_before = context.get('before', '')
        context_after = context.get('after', '')
        
        # Use helper function to assemble caption
        enriched_caption = _assemble_enriched_caption_text(
            author_caption,
            vlm_description,
            context_before,
            context_after
        )
        
        if not enriched_caption:
            enriched_caption = "No caption or context found"
        
        img_meta.update({
            'enriched_caption': enriched_caption,
            'vlm_description': vlm_description,
            'author_caption': author_caption,
            'context_before': context_before,
            'context_after': context_after
        })
    
    except (ValueError, TypeError) as e:
        logging.error(f"Error processing image {img_meta.get('image_id', 'unknown')}: {type(e).__name__}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error enriching image: {type(e).__name__}: {e}")
    
    return img_meta


def enrich_all_images(
    images_metadata: List[Dict],
    raw_papers_dir: Path,
    captioner=None
) -> List[Dict]:
    """
    Enrich all images in the dataset. 
    Args:
        images_metadata: List of image metadata dicts
        raw_papers_dir: Path to directory with PDF files
        captioner: Optional ImageCaptioner instance
        
    Returns:
        Updated list of image metadata with enriched fields
    """
    # Parameter validation
    if not isinstance(images_metadata, list):
        logging.error(f"images_metadata must be list, got {type(images_metadata).__name__}")
        return images_metadata
    
    if not isinstance(raw_papers_dir, Path):
        logging.error(f"raw_papers_dir must be Path, got {type(raw_papers_dir).__name__}")
        return images_metadata
    
    if not raw_papers_dir.exists():
        logging.error(f"Papers directory not found: {raw_papers_dir}")
        return images_metadata
    
    enriched_images = []
    pdf_docs = {}  
    
    total = len(images_metadata)
    mode = "VLM descriptions + context" if captioner else "captions + context only"
    logging.info(f"Starting enrichment of {total} images with {mode}")
    
    if captioner:
        logging.info(f"Rate limit: {RATE_LIMIT_REQUESTS_PER_MINUTE} req/min ({RATE_LIMIT_DELAY_SECONDS} sec delay between requests)")
    
    try:
        for idx, img_meta in enumerate(images_metadata, 1):
            # Validate image metadata before processing
            if not _validate_image_metadata(img_meta):
                enriched_images.append(img_meta)
                continue
            
            image_id = img_meta.get('image_id', f'image_{idx}')
            logging.debug(f"[{idx}/{total}] Processing {image_id}")
            
            try:
                # Use helper function to enrich single image
                enriched = enrich_single_image(
                    img_meta,
                    pdf_docs,
                    raw_papers_dir,
                    captioner=captioner
                )
                enriched_images.append(enriched)
                
                # Rate limiting for API calls (20 req/min = 3.5 sec/req)
                if captioner and idx < total and img_meta.get('extraction_method', EXTRACTION_METHOD_PDF) == EXTRACTION_METHOD_PDF:
                    import time
                    time.sleep(RATE_LIMIT_DELAY_SECONDS)
                
            except Exception as e:
                logging.error(f"Error enriching image {image_id}: {type(e).__name__}: {e}")
                enriched_images.append(img_meta)  # Keep original on error
    
    finally:
        # Guarantee cleanup of PDF documents
        for doc in pdf_docs.values():
            try:
                doc.close()
            except Exception as e:
                logging.warning(f"Error closing PDF document: {type(e).__name__}")
        pdf_docs.clear()
    
    logging.info(f"Enriched {len(enriched_images)} images")
    return enriched_images

def generate_captions_for_doc(
    doc_id: str,
    metadata_path: Optional[Path] = None,
    raw_papers_dir: Optional[Path] = None,
    use_vlm: bool = True
) -> int:
    """
    Generate enriched captions for all images in a specific document.
    
    Args:
        doc_id: Document identifier
        metadata_path: Path to images_metadata.json (default: data/processed/images_metadata.json)
        raw_papers_dir: Path to raw papers directory (default: data/raw/papers)
        use_vlm: Whether to use Vision-LM for descriptions (default: True)
    
    Returns:
        Number of images captioned
    """
    # Parameter validation
    if not isinstance(doc_id, str) or not doc_id.strip():
        logging.error(f"doc_id must be non-empty string, got {type(doc_id).__name__}: {doc_id}")
        return 0
    
    project_root = Path(__file__).parent.parent
    
    if metadata_path is None:
        metadata_path = project_root / "data" / "processed" / "images_metadata.json"
    if raw_papers_dir is None:
        raw_papers_dir = project_root / "data" / "raw" / "papers"
    
    # Validate paths
    if not isinstance(metadata_path, Path):
        try:
            metadata_path = Path(metadata_path)
        except (TypeError, ValueError):
            logging.error(f"Invalid metadata_path: {metadata_path}")
            return 0
    
    if not isinstance(raw_papers_dir, Path):
        try:
            raw_papers_dir = Path(raw_papers_dir)
        except (TypeError, ValueError):
            logging.error(f"Invalid raw_papers_dir: {raw_papers_dir}")
            return 0
    
    if not metadata_path.exists():
        logging.error(f"Metadata file not found: {metadata_path}")
        return 0
    
    if not raw_papers_dir.exists():
        logging.error(f"Raw papers directory not found: {raw_papers_dir}")
        return 0
    
    # Use helper function for safe JSON loading
    all_images = _load_and_validate_metadata(metadata_path)
    if all_images is None:
        return 0
    
    # Filter images for this document
    doc_images = [img for img in all_images if img.get('doc_id') == doc_id]
    
    if not doc_images:
        logging.warning(f"No images found for document: {doc_id}")
        return 0
    
    logging.info(f"Generating captions for {len(doc_images)} images in {doc_id}")
    
    # Initialize captioner if needed
    captioner = None
    if use_vlm:
        try:
            captioner = ImageCaptioner(
                model_name="gpt-4.1-mini",
                api_key=None  # Uses OPENAI_API_KEY env variable
            )
        except ValueError as e:
            logging.warning(f"VLM not available: {e}")
            logging.info("Continuing with captions + context only")
    
    # Enrich images for this document
    enriched_doc_images = enrich_all_images(
        doc_images,
        raw_papers_dir,
        captioner=captioner
    ) 
    # Update metadata for enriched images
    enriched_dict = {img['image_id']: img for img in enriched_doc_images}
    updated_count = 0
    for i, img in enumerate(all_images):
        if img['image_id'] in enriched_dict:
            all_images[i] = enriched_dict[img['image_id']]
            updated_count += 1
    
    logging.info(f"Updated {updated_count} images in metadata (out of {len(all_images)} total)")
    
    # Save updated metadata
    save_images_metadata(all_images, metadata_path)
    
    return len(enriched_doc_images)
