"""
Smart image extraction - extracts only diagrams/figures, not full pages.

This script  finds and extracts:
1. Embedded raster images (PNG, JPG)
2. Vector graphics regions (diagrams, charts)
3. Figures and tables (detected by captions)

Usage:
    python extract_images_smart.py --input data/raw/papers --output data/processed/images
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from datetime import datetime

try:
    import fitz  
except ImportError:
    logging.error("PyMuPDF not installed. Run: pip install PyMuPDF")
    exit(1)

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

# ============================================================================
# CONFIGURATION CONSTANTS - STAGE 2: Constants instead of magic numbers
# ============================================================================
DEFAULT_INPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "papers"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed" / "images"

# Image processing constants
MIN_IMAGE_SIZE = 100
DPI = 200  # Resolution for vector graphics extraction

# Vector graphics extraction constants
MIN_VECTOR_REGION_SIZE = 100  # Minimum width/height for vector regions (pixels)
MERGE_THRESHOLD = 50  # Distance threshold for merging nearby rectangles (pixels)
VECTOR_REGION_PADDING = 20  # Padding around vector regions when cropping (pixels)

# PDF processing constants
PDF_DPI_BASE = 72  # Base DPI for PDF coordinate system

# Text extraction constants
TEXT_EXTRACTION_FORMAT = "blocks"  # Format for text extraction from PDF
PNG_FORMAT = "png"  # Image format for saved vector graphics

# Figure detection constants
FIGURE_CAPTION_KEYWORDS = ["figure", "fig.", "table", "diagram"]


# ============================================================================
# HELPER FUNCTIONS - STAGE 3: Extract DRY/SRP principles
# ============================================================================

def _create_bbox_from_rect(rect) -> Optional[Dict]:
    """
    STAGE 3: DRY - extract bbox creation into helper function.
    
    Creates bbox dictionary from fitz.Rect object.
    
    Args:
        rect: fitz.Rect object with x0, y0, x1, y1 attributes
    
    Returns:
        Dictionary with bbox coordinates or None if rect is None
    """
    if rect is None:
        return None
    
    try:
        return {
            "x0": rect.x0,
            "y0": rect.y0,
            "x1": rect.x1,
            "y1": rect.y1
        }
    except Exception as e:
        logging.warning(f"Failed to create bbox from rect: {e}")
        return None


def _create_image_metadata(
    image_id: str,
    doc_id: str,
    image_path: Path,
    page_num: int,
    width: int,
    height: int,
    image_format: str,
    size_bytes: int,
    extraction_method: str,
    bbox: Optional[Dict] = None,
    dpi: Optional[int] = None,
    region_index: Optional[int] = None
) -> Dict:
    """
    STAGE 3: DRY/SRP - consolidate metadata creation into single function.
    
    Creates metadata dictionary for extracted image.
    
    Args:
        image_id: Unique image identifier
        doc_id: Document identifier
        image_path: Path where image was saved
        page_num: Page number (1-based)
        width: Image width in pixels
        height: Image height in pixels
        image_format: Image format (png, jpg, etc.)
        size_bytes: File size in bytes
        extraction_method: Method used for extraction (embedded_raster, vector_graphics)
        bbox: Bounding box dictionary (optional)
        dpi: DPI for vector graphics (optional)
        region_index: Region index for vector graphics (optional)
    
    Returns:
        Dictionary with image metadata
    """
    metadata = {
        "image_id": image_id,
        "doc_id": doc_id,
        "filepath": str(image_path.absolute()),
        "filename": image_path.name,
        "page_num": page_num,
        "bbox": bbox,
        "width": width,
        "height": height,
        "format": image_format,
        "size_bytes": size_bytes,
        "extraction_method": extraction_method,
        "extracted_at": datetime.now().isoformat(),
        "enriched_caption": None,
        "vlm_description": None,
        "author_caption": None,
        "context_before": None,
        "context_after": None,
    }
    
    # Add optional fields
    if region_index is not None:
        metadata["region_index"] = region_index
    
    if dpi is not None:
        metadata["dpi"] = dpi
    
    return metadata


def _load_papers_metadata(metadata_path: Path) -> Dict[str, Dict]:
    """
    STAGE 3: DRY - extract JSON loading logic into helper function.
    
    Load papers metadata from JSON file.
    
    Args:
        metadata_path: Path to papers_metadata.json file
    
    Returns:
        Dictionary mapping doc_id to metadata, or empty dict if file doesn't exist/error
    """
    if not metadata_path.exists():
        return {}
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            papers_list = json.load(f)
            if isinstance(papers_list, list):
                return {p["doc_id"]: p for p in papers_list if isinstance(p, dict) and "doc_id" in p}
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {metadata_path}: {e}")
    except (IOError, OSError) as e:
        logging.error(f"Failed to read {metadata_path}: {type(e).__name__}: {e}")
    
    return {}


def _save_images_metadata_file(images_metadata_path: Path, all_images_metadata: List[Dict]) -> bool:
    """
    STAGE 3: DRY - extract JSON saving logic into helper function.
    
    Save images metadata to JSON file.
    
    Args:
        images_metadata_path: Path to images_metadata.json file
        all_images_metadata: List of image metadata dictionaries
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(images_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(all_images_metadata, f, indent=2, ensure_ascii=False)
        return True
    except (IOError, OSError) as e:
        logging.error(f"Failed to save {images_metadata_path}: {type(e).__name__}: {e}")
        return False
    except TypeError as e:
        logging.error(f"Failed to serialize metadata to JSON: {e}")
        return False


def extract_embedded_images(
    pdf_document,
    doc_output_dir: Path,
    doc_id: str,
    min_size: int = MIN_IMAGE_SIZE
) -> Tuple[List[Dict], int]:
    """
    Extract embedded raster images (PNG, JPG) with bbox coordinates.
    
    STAGE 1: Validation - check all parameters and handle errors gracefully
    """
    # STAGE 1: Parameter validation
    if pdf_document is None:
        logging.error("pdf_document is None")
        return [], 0
    
    if not isinstance(doc_output_dir, Path):
        logging.error(f"doc_output_dir must be Path, got {type(doc_output_dir).__name__}")
        return [], 0
    
    if not doc_id or not isinstance(doc_id, str):
        logging.error(f"doc_id must be non-empty string, got {type(doc_id).__name__}")
        return [], 0
    
    if min_size < 1:
        logging.warning(f"min_size must be >= 1, got {min_size}, using MIN_IMAGE_SIZE")
        min_size = MIN_IMAGE_SIZE
    
    images_metadata = []
    image_count = 0
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = pdf_document.extract_image(xref)
                
                # STAGE 1: Handle missing keys in dictionary
                if not base_image or "image" not in base_image or "ext" not in base_image:
                    logging.warning(f"Invalid image data at page {page_num + 1}, image {img_index}")
                    continue
                
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                image = Image.open(io.BytesIO(image_bytes))
                width, height = image.size
                
                if width < min_size or height < min_size:
                    continue

                image_rects = page.get_image_rects(xref)
                bbox = None
                if image_rects:
                    rect = image_rects[0]
                    bbox = _create_bbox_from_rect(rect)
                
                image_count += 1
                image_id = f"{doc_id}_embedded_{image_count:03d}"
                image_filename = f"{image_id}.{image_ext}"
                image_path = doc_output_dir / image_filename
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                metadata = _create_image_metadata(
                    image_id=image_id,
                    doc_id=doc_id,
                    image_path=image_path,
                    page_num=page_num + 1,
                    width=width,
                    height=height,
                    image_format=image_ext,
                    size_bytes=len(image_bytes),
                    extraction_method="embedded_raster",
                    bbox=bbox
                )
                
                images_metadata.append(metadata)
                
            except (IOError, OSError) as e:
                logging.warning(f"Failed to save image at page {page_num + 1}, image {img_index}: {type(e).__name__}: {e}")
                continue
            except Exception as e:
                logging.warning(f"Error processing image at page {page_num + 1}, image {img_index}: {type(e).__name__}: {e}")
                continue
    
    return images_metadata, image_count


def detect_vector_graphics_regions(page) -> List[fitz.Rect]:
    """
    Detect regions with vector graphics (drawings, paths).
    Returns list of bounding rectangles.
    
    STAGE 1: Validation - handle None/empty safely
    """
    # STAGE 1: Parameter validation
    if page is None:
        logging.warning("page is None in detect_vector_graphics_regions")
        return []
    
    try:
        drawings = page.get_drawings()
        
        # STAGE 1: Check if drawings exists and is iterable
        if not drawings:
            return []
        
        rects = []
        for drawing in drawings:
            if not isinstance(drawing, dict):
                continue
            
            rect = drawing.get("rect")
            if rect:
                try:
                    rects.append(fitz.Rect(rect))
                except Exception as e:
                    logging.warning(f"Failed to create Rect from drawing: {e}")
                    continue
        
        if not rects:
            return []

        merged_rects = merge_nearby_rectangles(rects, threshold=MERGE_THRESHOLD)
        
        return merged_rects
    
    except Exception as e:
        logging.warning(f"Error detecting vector graphics: {type(e).__name__}: {e}")
        return []


def merge_nearby_rectangles(rects: List[fitz.Rect], threshold: float = 50) -> List[fitz.Rect]:
    """
    Merge rectangles that are close to each other.
    This groups drawing elements that belong to the same figure.
    """
    if not rects:
        return []
    
    merged = []
    used = set()
    
    for i, rect1 in enumerate(rects):
        if i in used:
            continue
        
        current = fitz.Rect(rect1)
        used.add(i)

        changed = True
        while changed:
            changed = False
            for j, rect2 in enumerate(rects):
                if j in used:
                    continue

                if are_rectangles_close(current, rect2, threshold):
                    current = current | rect2  
                    used.add(j)
                    changed = True
        
        merged.append(current)
    
    return merged


def are_rectangles_close(rect1: fitz.Rect, rect2: fitz.Rect, threshold: float) -> bool:
    """Check if two rectangles are within threshold distance."""
    
    if rect1.intersects(rect2):
        return True

    horizontal_distance = max(0, rect2.x0 - rect1.x1, rect1.x0 - rect2.x1)
    vertical_distance = max(0, rect2.y0 - rect1.y1, rect1.y0 - rect2.y1)
    
    distance = (horizontal_distance ** 2 + vertical_distance ** 2) ** 0.5
    
    return distance < threshold


def detect_figure_captions(page) -> List[Tuple[fitz.Rect, str]]:
    """
    Detect figure/table captions in text.
    Returns list of (region_rect, caption_text).
    """
    text_blocks = page.get_text(TEXT_EXTRACTION_FORMAT)
    
    figures = []
    
    for block in text_blocks:
        text = block[4] if len(block) > 4 else ""

        if any(keyword in text.lower() for keyword in FIGURE_CAPTION_KEYWORDS):

            rect = fitz.Rect(block[:4])
            figures.append((rect, text.strip()))
    
    return figures


def extract_vector_graphics(
    pdf_document,
    doc_output_dir: Path,
    doc_id: str,
    dpi: int = DPI,
    min_size: int = MIN_IMAGE_SIZE
) -> List[Dict]:
    """
    Extract vector graphics regions as images.
    Only captures areas with actual graphics, not text.
    
    STAGE 1: Validation - validate DPI, min_size, and handle errors gracefully
    """
    # STAGE 1: Parameter validation
    if pdf_document is None:
        logging.error("pdf_document is None")
        return []
    
    if not isinstance(doc_output_dir, Path):
        logging.error(f"doc_output_dir must be Path, got {type(doc_output_dir).__name__}")
        return []
    
    if not doc_id or not isinstance(doc_id, str):
        logging.error(f"doc_id must be non-empty string")
        return []
    
    # STAGE 1: Validate DPI - cannot be 0 or negative
    if dpi <= 0:
        logging.warning(f"Invalid DPI {dpi}, using default {DPI}")
        dpi = DPI
    
    if min_size < 1:
        logging.warning(f"min_size must be >= 1, got {min_size}, using MIN_IMAGE_SIZE")
        min_size = MIN_IMAGE_SIZE
    
    images_metadata = []
    image_count = 0
    
    zoom = dpi / PDF_DPI_BASE
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(len(pdf_document)):
        try:
            page = pdf_document[page_num]

            graphics_regions = detect_vector_graphics_regions(page)
            
            if not graphics_regions:
                continue

            significant_regions = [
                rect for rect in graphics_regions
                if rect.width > MIN_VECTOR_REGION_SIZE and rect.height > MIN_VECTOR_REGION_SIZE
            ]
            
            if not significant_regions:
                continue
   
            try:
                pix = page.get_pixmap(matrix=matrix)
                page_img = Image.open(io.BytesIO(pix.tobytes("png")))
            except Exception as e:
                logging.warning(f"Failed to render page {page_num + 1}: {type(e).__name__}: {e}")
                continue

            for region_idx, region_rect in enumerate(significant_regions):

                scaled_rect = fitz.Rect(
                    region_rect.x0 * zoom,
                    region_rect.y0 * zoom,
                    region_rect.x1 * zoom,
                    region_rect.y1 * zoom
                )

                crop_box = (
                    max(0, int(scaled_rect.x0 - VECTOR_REGION_PADDING)),
                    max(0, int(scaled_rect.y0 - VECTOR_REGION_PADDING)),
                    min(page_img.width, int(scaled_rect.x1 + VECTOR_REGION_PADDING)),
                    min(page_img.height, int(scaled_rect.y1 + VECTOR_REGION_PADDING))
                )

                cropped_img = page_img.crop(crop_box)

                if cropped_img.width < min_size or cropped_img.height < min_size:
                    continue

                image_count += 1
                image_id = f"{doc_id}_vector_{page_num + 1:03d}_{region_idx + 1:02d}"
                image_filename = f"{image_id}.{PNG_FORMAT}"
                image_path = doc_output_dir / image_filename
                
                try:
                    cropped_img.save(image_path, PNG_FORMAT.upper())
                except (IOError, OSError) as e:
                    logging.warning(f"Failed to save vector graphic at page {page_num + 1}: {type(e).__name__}: {e}")
                    continue

                bbox = _create_bbox_from_rect(region_rect)
                
                metadata = _create_image_metadata(
                    image_id=image_id,
                    doc_id=doc_id,
                    image_path=image_path,
                    page_num=page_num + 1,
                    width=cropped_img.width,
                    height=cropped_img.height,
                    image_format=PNG_FORMAT,
                    size_bytes=image_path.stat().st_size if image_path.exists() else 0,
                    extraction_method="vector_graphics",
                    bbox=bbox,
                    dpi=dpi,
                    region_index=region_idx + 1
                )
                
                images_metadata.append(metadata)
        
        except Exception as e:
            logging.warning(f"Error processing page {page_num + 1}: {type(e).__name__}: {e}")
            continue
    
    return images_metadata


def extract_images_smart(
    pdf_path: Path,
    output_dir: Path,
    doc_id: str,
    min_size: int = MIN_IMAGE_SIZE,
    dpi: int = DPI
) -> List[Dict]:
    """
    Smart extraction: embedded images + vector graphics regions only.
    
    Error Handling Strategy: Logs errors and returns partial results.
    - PDF corruption or parse errors → Logs error, returns empty list
    - Individual page errors → Logs warning, continues with other pages
    - Caller should check for empty list but not expect exceptions
    
    For critical errors that should halt processing, caller should validate
    PDF existence/readability before calling this function.
    
    STAGE 1: Validation - proper try-finally to ensure PDF is closed
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory for extracted images
        doc_id: Document identifier
        min_size: Minimum image dimension (pixels)
        dpi: DPI for vector graphics rasterization
    
    Returns:
        List of image metadata dicts (may be empty if extraction fails)
    """
    # STAGE 1: Parameter validation
    if not isinstance(pdf_path, Path):
        logging.error(f"pdf_path must be Path, got {type(pdf_path).__name__}")
        return []
    
    if not isinstance(output_dir, Path):
        logging.error(f"output_dir must be Path, got {type(output_dir).__name__}")
        return []
    
    if not doc_id or not isinstance(doc_id, str):
        logging.error(f"doc_id must be non-empty string")
        return []
    
    all_images_metadata = []
    pdf_document = None
    
    try:
        doc_output_dir = output_dir / doc_id
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            pdf_document = fitz.open(pdf_path)
        except Exception as e:
            logging.error(f"Failed to open PDF {pdf_path.name}: {type(e).__name__}: {e}")
            return []
        
        logging.info(f"  📄 Processing: {pdf_path.name}")
        logging.info(f"  📑 Pages: {len(pdf_document)}")

        embedded_images, embedded_count = extract_embedded_images(
            pdf_document, doc_output_dir, doc_id, min_size
        )
        all_images_metadata.extend(embedded_images)
        logging.info(f"  📦 Embedded images: {embedded_count}")

        logging.info(f"  🎨 Detecting vector graphics regions...")
        vector_images = extract_vector_graphics(
            pdf_document, doc_output_dir, doc_id, dpi, min_size
        )
        all_images_metadata.extend(vector_images)
        logging.info(f"  ✅ Vector graphics: {len(vector_images)}")
        
        logging.info(f"  ✅ Total images: {len(all_images_metadata)}")
        
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_path.name}: {type(e).__name__}: {e}")
    
    finally:
        # STAGE 1: Ensure PDF is always closed
        if pdf_document is not None:
            try:
                pdf_document.close()
            except Exception as e:
                logging.warning(f"Error closing PDF: {type(e).__name__}: {e}")
    
    return all_images_metadata


def extract_text_from_pdf(pdf_path: Path) -> Tuple[str, List[Dict]]:
    """
    Extract text from PDF.
    
    STAGE 1: Validation - proper try-finally to ensure PDF is closed
    """
    # STAGE 1: Parameter validation
    if not isinstance(pdf_path, Path):
        logging.error(f"pdf_path must be Path, got {type(pdf_path).__name__}")
        return "", []
    
    pdf_document = None
    
    try:
        try:
            pdf_document = fitz.open(pdf_path)
        except Exception as e:
            logging.error(f"Failed to open PDF {pdf_path}: {type(e).__name__}: {e}")
            return "", []
        
        full_text = ""
        pages_text = []
        
        for page_num in range(len(pdf_document)):
            try:
                page = pdf_document[page_num]
                page_text = page.get_text()
                full_text += page_text
                
                pages_text.append({
                    "page_num": page_num + 1,
                    "text": page_text,
                    "char_count": len(page_text)
                })
            except Exception as e:
                logging.warning(f"Error extracting text from page {page_num + 1}: {type(e).__name__}: {e}")
                continue
        
        return full_text, pages_text
    
    finally:
        # STAGE 1: Ensure PDF is always closed
        if pdf_document is not None:
            try:
                pdf_document.close()
            except Exception as e:
                logging.warning(f"Error closing PDF: {type(e).__name__}: {e}")





def process_all_papers(
    input_dir: Path,
    output_dir: Path,
    min_image_size: int = MIN_IMAGE_SIZE,
    dpi: int = DPI
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process all papers with smart extraction.
    
    STAGE 1: Validation - check parameters and handle JSON errors
    """
    # STAGE 1: Parameter validation
    if not isinstance(input_dir, Path):
        logging.error(f"input_dir must be Path, got {type(input_dir).__name__}")
        return [], []
    
    if not isinstance(output_dir, Path):
        logging.error(f"output_dir must be Path, got {type(output_dir).__name__}")
        return [], []
    
    if not input_dir.exists():
        logging.error(f"input_dir does not exist: {input_dir}")
        return [], []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_images_metadata = []
    documents_metadata = []
    
    metadata_path = input_dir / "papers_metadata.json"
    papers_info = _load_papers_metadata(metadata_path)
    
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logging.warning(f"No PDF files found in {input_dir}")
        return [], []
    
    logging.info(f"Found {len(pdf_files)} PDF files")
    logging.info(f"Smart extraction: embedded + vector graphics regions only")
    logging.info(f"DPI: {dpi}")
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        logging.info(f"[{idx}/{len(pdf_files)}] Processing: {pdf_path.name}")
        
        doc_id = pdf_path.stem

        images_metadata = extract_images_smart(
            pdf_path, output_dir, doc_id, min_image_size, dpi
        )
        all_images_metadata.extend(images_metadata)

        # Extract text stats (but don't save to disk)
        full_text, pages_text = extract_text_from_pdf(pdf_path)
        logging.info(f"  Text extracted: {len(full_text.split())} words")
        
        doc_metadata = {
            "doc_id": doc_id,
            "filename": pdf_path.name,
            "filepath": str(pdf_path.absolute()),
            "num_images": len(images_metadata),
            "num_pages": len(pages_text),
            "char_count": len(full_text),
            "word_count": len(full_text.split()),
            "processed_at": datetime.now().isoformat(),
        }
        
        if doc_id in papers_info:
            doc_metadata.update(papers_info[doc_id])
        
        documents_metadata.append(doc_metadata)
    
    return all_images_metadata, documents_metadata


def extract_document(
    doc_id: str,
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    min_image_size: int = MIN_IMAGE_SIZE,
    dpi: int = DPI
) -> Dict:
    """
    Extract images and text from a single document.
    
    Args:
        doc_id: Document identifier (PDF filename without extension)
        input_dir: Directory containing PDF file (default: data/raw/papers)
        output_dir: Output directory for extracted data (default: data/processed/images)
        min_image_size: Minimum image size in pixels
        dpi: Resolution for vector graphics extraction
    
    Returns:
        Dictionary with extraction results:
        {
            "doc_id": str,
            "images_count": int,
            "text_length": int,
            "images_metadata": List[Dict],
            "document_metadata": Dict
        }
    """
    if input_dir is None:
        input_dir = DEFAULT_INPUT_DIR
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = input_dir / f"{doc_id}.pdf"
    if not pdf_path.exists():
        logging.error(f"PDF not found: {pdf_path}")
        return {
            "doc_id": doc_id,
            "images_count": 0,
            "text_length": 0,
            "images_metadata": [],
            "document_metadata": None,
            "error": "PDF file not found"
        }
    
    logging.info(f"Extracting document: {doc_id}")

    images_metadata = extract_images_smart(
        pdf_path, output_dir, doc_id, min_image_size, dpi
    )
    
    # Extract text (in-memory only)
    full_text, pages_text = extract_text_from_pdf(pdf_path)
    
    # Build document metadata
    doc_metadata = {
        "doc_id": doc_id,
        "filename": pdf_path.name,
        "filepath": str(pdf_path.absolute()),
        "num_images": len(images_metadata),
        "num_pages": len(pages_text),
        "char_count": len(full_text),
        "word_count": len(full_text.split()),
        "processed_at": datetime.now().isoformat(),
    }
    
    # Load paper metadata if available
    metadata_path = input_dir / "papers_metadata.json"
    papers_info = _load_papers_metadata(metadata_path)
    if doc_id in papers_info:
        doc_metadata.update(papers_info[doc_id])
    
    logging.info(f"  Images: {len(images_metadata)}, Text: {len(full_text.split())} words")
    
    # Save or update images_metadata.json for this document
    images_metadata_path = output_dir.parent / "images_metadata.json"
    all_images_metadata = []
    
    if images_metadata_path.exists():
        try:
            # Load existing metadata
            with open(images_metadata_path, 'r', encoding='utf-8') as f:
                all_images_metadata = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {images_metadata_path}: {e}")
            all_images_metadata = []
        except (IOError, OSError) as e:
            logging.error(f"Failed to read {images_metadata_path}: {type(e).__name__}: {e}")
            all_images_metadata = []
    
    # Remove old metadata for this doc_id
    all_images_metadata = [img for img in all_images_metadata if img.get("doc_id") != doc_id]
    
    # Add new metadata
    all_images_metadata.extend(images_metadata)
    
    # Save updated metadata using helper function
    _save_images_metadata_file(images_metadata_path, all_images_metadata)
    
    return {
        "doc_id": doc_id,
        "images_count": len(images_metadata),
        "text_length": len(full_text),
        "full_text": full_text,  
        "images_metadata": images_metadata,
        "document_metadata": doc_metadata
    }


def save_metadata(
    images_metadata: List[Dict],
    documents_metadata: List[Dict],
    output_dir: Path
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Save metadata to JSON files.
    
    STAGE 1: Validation - handle JSON errors properly
    STAGE 3: Uses helper functions for DRY principle
    
    Args:
        images_metadata: List of image metadata dictionaries
        documents_metadata: List of document metadata dictionaries
        output_dir: Output directory for JSON files
    
    Returns:
        Tuple of (images_json_path, docs_json_path) or (None, None) on error
    """
    # STAGE 1: Parameter validation
    if not isinstance(output_dir, Path):
        logging.error(f"output_dir must be Path, got {type(output_dir).__name__}")
        return None, None
    
    if not isinstance(images_metadata, list):
        logging.error(f"images_metadata must be list, got {type(images_metadata).__name__}")
        return None, None
    
    if not isinstance(documents_metadata, list):
        logging.error(f"documents_metadata must be list, got {type(documents_metadata).__name__}")
        return None, None
    
    images_json_path = output_dir.parent / "images_metadata.json"
    docs_json_path = output_dir.parent / "documents_metadata.json"
    
    # STAGE 3: Use helper function for saving images metadata
    success = _save_images_metadata_file(images_json_path, images_metadata)
    if not success:
        return None, docs_json_path
    
    # Save documents metadata
    try:
        with open(docs_json_path, "w", encoding="utf-8") as f:
            json.dump(documents_metadata, f, indent=2, ensure_ascii=False)
    except (IOError, OSError) as e:
        logging.error(f"Failed to save {docs_json_path}: {type(e).__name__}: {e}")
        return images_json_path, None
    except TypeError as e:
        logging.error(f"Failed to serialize documents metadata: {e}")
        return images_json_path, None
    
    return images_json_path, docs_json_path


