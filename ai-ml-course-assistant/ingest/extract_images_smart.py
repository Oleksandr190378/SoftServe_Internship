"""
Smart image extraction - extracts only diagrams/figures, not full pages.

This script intelligently finds and extracts:
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


DEFAULT_INPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "papers"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed" / "images"
MIN_IMAGE_SIZE = 100
DPI = 200  # Resolution for vector graphics extraction


def extract_embedded_images(
    pdf_document,
    doc_output_dir: Path,
    doc_id: str,
    min_size: int = MIN_IMAGE_SIZE
) -> Tuple[List[Dict], int]:
    """Extract embedded raster images (PNG, JPG) with bbox coordinates."""
    images_metadata = []
    image_count = 0
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = pdf_document.extract_image(xref)
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
                    bbox = {
                        "x0": rect.x0,
                        "y0": rect.y0,
                        "x1": rect.x1,
                        "y1": rect.y1
                    }
                
                image_count += 1
                image_id = f"{doc_id}_embedded_{image_count:03d}"
                image_filename = f"{image_id}.{image_ext}"
                image_path = doc_output_dir / image_filename
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                metadata = {
                    "image_id": image_id,
                    "doc_id": doc_id,
                    "filepath": str(image_path.absolute()),
                    "filename": image_filename,
                    "page_num": page_num + 1,
                    "bbox": bbox,  
                    "width": width,
                    "height": height,
                    "format": image_ext,
                    "size_bytes": len(image_bytes),
                    "extraction_method": "embedded_raster",
                    "extracted_at": datetime.now().isoformat(),
                    "enriched_caption": None,
                    "vlm_description": None,
                    "author_caption": None,
                    "context_before": None,
                    "context_after": None,
                }
                
                images_metadata.append(metadata)
                
            except Exception as e:
                continue
    
    return images_metadata, image_count


def detect_vector_graphics_regions(page) -> List[fitz.Rect]:
    """
    Detect regions with vector graphics (drawings, paths).
    Returns list of bounding rectangles.
    """
    try:
        drawings = page.get_drawings()
        
        if not drawings:
            return []
        
        rects = []
        for drawing in drawings:
            rect = drawing.get("rect")
            if rect:
                rects.append(fitz.Rect(rect))
        
        if not rects:
            return []

        merged_rects = merge_nearby_rectangles(rects, threshold=50)
        
        return merged_rects
    
    except Exception as e:
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
    text_blocks = page.get_text("blocks")
    
    figures = []
    
    for block in text_blocks:
        text = block[4] if len(block) > 4 else ""

        if any(keyword in text.lower() for keyword in ["figure", "fig.", "table", "diagram"]):

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
    """
    images_metadata = []
    image_count = 0
    
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(len(pdf_document)):
        try:
            page = pdf_document[page_num]

            graphics_regions = detect_vector_graphics_regions(page)
            
            if not graphics_regions:
                continue

            significant_regions = [
                rect for rect in graphics_regions
                if rect.width > 100 and rect.height > 100
            ]
            
            if not significant_regions:
                continue
   
            pix = page.get_pixmap(matrix=matrix)
            page_img = Image.open(io.BytesIO(pix.tobytes("png")))

            for region_idx, region_rect in enumerate(significant_regions):

                scaled_rect = fitz.Rect(
                    region_rect.x0 * zoom,
                    region_rect.y0 * zoom,
                    region_rect.x1 * zoom,
                    region_rect.y1 * zoom
                )

                padding = 20
                crop_box = (
                    max(0, int(scaled_rect.x0 - padding)),
                    max(0, int(scaled_rect.y0 - padding)),
                    min(page_img.width, int(scaled_rect.x1 + padding)),
                    min(page_img.height, int(scaled_rect.y1 + padding))
                )

                cropped_img = page_img.crop(crop_box)

                if cropped_img.width < min_size or cropped_img.height < min_size:
                    continue

                image_count += 1
                image_id = f"{doc_id}_vector_{page_num + 1:03d}_{region_idx + 1:02d}"
                image_filename = f"{image_id}.png"
                image_path = doc_output_dir / image_filename
                
                cropped_img.save(image_path, "PNG")

                bbox = {
                    "x0": region_rect.x0,
                    "y0": region_rect.y0,
                    "x1": region_rect.x1,
                    "y1": region_rect.y1
                }
                
                metadata = {
                    "image_id": image_id,
                    "doc_id": doc_id,
                    "filepath": str(image_path.absolute()),
                    "filename": image_filename,
                    "page_num": page_num + 1,
                    "region_index": region_idx + 1,
                    "bbox": bbox,  # Original vector graphics region coordinates
                    "width": cropped_img.width,
                    "height": cropped_img.height,
                    "format": "png",
                    "size_bytes": image_path.stat().st_size,
                    "extraction_method": "vector_graphics",
                    "dpi": dpi,
                    "extracted_at": datetime.now().isoformat(),
                    "enriched_caption": None,
                    "vlm_description": None,
                    "author_caption": None,
                    "context_before": None,
                    "context_after": None,
                }
                
                images_metadata.append(metadata)
        
        except Exception as e:
            print(f"    ⚠️  Error processing page {page_num + 1}: {e}")
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
    """
    all_images_metadata = []
    
    doc_output_dir = output_dir / doc_id
    doc_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        pdf_document = fitz.open(pdf_path)
        
        print(f"  📄 Processing: {pdf_path.name}")
        print(f"  📑 Pages: {len(pdf_document)}")

        embedded_images, embedded_count = extract_embedded_images(
            pdf_document, doc_output_dir, doc_id, min_size
        )
        all_images_metadata.extend(embedded_images)
        print(f"  📦 Embedded images: {embedded_count}")

        print(f"  🎨 Detecting vector graphics regions...")
        vector_images = extract_vector_graphics(
            pdf_document, doc_output_dir, doc_id, dpi, min_size
        )
        all_images_metadata.extend(vector_images)
        print(f"  ✅ Vector graphics: {len(vector_images)}")
        
        pdf_document.close()
        
        print(f"  ✅ Total images: {len(all_images_metadata)}")
        
    except Exception as e:
        print(f"  ❌ Error processing PDF {pdf_path.name}: {e}")
    
    return all_images_metadata


def extract_text_from_pdf(pdf_path: Path) -> Tuple[str, List[Dict]]:
    """Extract text from PDF."""
    try:
        pdf_document = fitz.open(pdf_path)
        full_text = ""
        pages_text = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_text = page.get_text()
            full_text += page_text
            
            pages_text.append({
                "page_num": page_num + 1,
                "text": page_text,
                "char_count": len(page_text)
            })
        
        pdf_document.close()
        return full_text, pages_text
    
    except Exception as e:
        return "", []


# Text is no longer saved to disk - used in-memory by pipeline
# extract_text_from_pdf() returns text directly for chunking


def process_all_papers(
    input_dir: Path,
    output_dir: Path,
    min_image_size: int = MIN_IMAGE_SIZE,
    dpi: int = DPI
) -> Tuple[List[Dict], List[Dict]]:
    """Process all papers with smart extraction."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_images_metadata = []
    documents_metadata = []
    
    metadata_path = input_dir / "papers_metadata.json"
    papers_info = {}
    
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            papers_list = json.load(f)
            papers_info = {p["doc_id"]: p for p in papers_list}
    
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
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            papers_list = json.load(f)
            papers_info = {p["doc_id"]: p for p in papers_list}
            if doc_id in papers_info:
                doc_metadata.update(papers_info[doc_id])
    
    logging.info(f"  Images: {len(images_metadata)}, Text: {len(full_text.split())} words")
    
    # Save or update images_metadata.json for this document
    images_metadata_path = output_dir.parent / "images_metadata.json"
    all_images_metadata = []
    
    if images_metadata_path.exists():
        # Load existing metadata
        with open(images_metadata_path, 'r', encoding='utf-8') as f:
            all_images_metadata = json.load(f)
        # Remove old metadata for this doc_id
        all_images_metadata = [img for img in all_images_metadata if img['doc_id'] != doc_id]
    
    # Add new metadata
    all_images_metadata.extend(images_metadata)
    
    # Save updated metadata
    with open(images_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_images_metadata, f, indent=2, ensure_ascii=False)
    
    logging.debug(f"Saved {len(images_metadata)} images to {images_metadata_path.name}")
    
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
):
    """Save metadata to JSON files."""
    images_json_path = output_dir.parent / "images_metadata.json"
    with open(images_json_path, "w", encoding="utf-8") as f:
        json.dump(images_metadata, f, indent=2, ensure_ascii=False)
    
    docs_json_path = output_dir.parent / "documents_metadata.json"
    with open(docs_json_path, "w", encoding="utf-8") as f:
        json.dump(documents_metadata, f, indent=2, ensure_ascii=False)
    
    return images_json_path, docs_json_path


def main():
    parser = argparse.ArgumentParser(
        description="Smart image extraction (diagrams/figures only)"
    )
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_INPUT_DIR)
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT_DIR)
    )
    parser.add_argument(
        "--min-size", type=int, default=MIN_IMAGE_SIZE
    )
    parser.add_argument(
        "--dpi", type=int, default=DPI
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    print("=" * 70)
    print("🧠 Smart Image Extractor - AI/ML Course Assistant")
    print("=" * 70)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Min size: {args.min_size}px")
    print(f"DPI: {args.dpi}")
    print("=" * 70)
    print()
    
    images_metadata, documents_metadata = process_all_papers(
        input_dir, output_dir, args.min_size, args.dpi
    )
    
    if images_metadata or documents_metadata:
        images_json_path, docs_json_path = save_metadata(
            images_metadata, documents_metadata, output_dir
        )
        
        print("=" * 70)
        print(f"✅ Processing complete!")
        print("=" * 70)
        print()
        print("📊 Summary:")
        print(f"  - Documents: {len(documents_metadata)}")
        print(f"  - Images: {len(images_metadata)}")
        
        embedded = sum(1 for img in images_metadata 
                      if img.get("extraction_method") == "embedded_raster")
        vector = sum(1 for img in images_metadata 
                    if img.get("extraction_method") == "vector_graphics")
        
        print(f"    • Embedded raster: {embedded}")
        print(f"    • Vector graphics regions: {vector}")
        print(f"  - Images saved to: {output_dir}")
        print()
        
        if images_metadata:
            total_size_mb = sum(img["size_bytes"] for img in images_metadata) / (1024 * 1024)
            print(f"  - Total size: {total_size_mb:.2f} MB")
        
        print()
        print("🔜 Next: python ingest/test_extraction.py")
        print("=" * 70)


if __name__ == "__main__":
    main()
