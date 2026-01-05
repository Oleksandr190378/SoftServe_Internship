"""
Enrich images with context and Vision-LM descriptions.

This script combines:
1. BLIP-2 generated visual descriptions
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
import time
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

from ingest.extract_image_context import extract_surrounding_context
from ingest.generate_captions import ImageCaptioner


VISION_LM_PROMPT = """Describe this academic figure in technical detail for a machine learning researcher.

Focus on identifying and describing:

**If it's a neural network architecture:**
- Name specific layer types (Conv, LSTM, Transformer, Attention, etc.)
- Describe data flow and connections between components
- Mention input/output dimensions if visible
- Identify skip connections, residual blocks, or special structures

**If it's a graph or chart:**
- Identify axis labels and what they measure (accuracy, loss, epochs, etc.)
- Describe trends (increasing, decreasing, converging, plateauing)
- Compare multiple lines/bars if present
- Mention specific values or ranges if readable

**If it's a table:**
- Describe what metrics are being compared
- Identify the best/worst performing methods
- Mention specific numerical results if clearly visible

**If it's a diagram or flowchart:**
- Describe the process or algorithm flow
- Identify main components and their relationships
- Explain what each step does

**If it's a visualization (attention maps, feature maps, etc.):**
- Describe what is being visualized
- Mention patterns or interesting regions
- Explain what different colors/intensities represent

**Always mention:**
- Any text labels, legends, or annotations visible in the image
- Mathematical notation or formulas if present
- Specific model names, dataset names, or method names shown

Be specific and technical. Avoid generic descriptions like "interesting image" or "shows results"."""


def load_images_metadata(metadata_path: Path) -> List[Dict]:
    """Load images metadata from JSON file."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_images_metadata(metadata: List[Dict], output_path: Path):
    """Save updated images metadata to JSON file."""
    logging.info(f"Saving {len(metadata)} images to {output_path}")
    
    # DEBUG: Check sample before saving
    if metadata:
        sample = metadata[0]
        logging.debug(f"Sample image before save: {sample.get('image_id')}, has enriched_caption: {'enriched_caption' in sample}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved enriched metadata to {output_path.name}")


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
    doc_id = img_meta['doc_id']
    extraction_method = img_meta.get('extraction_method', 'embedded_raster')
    
    # For web-downloaded images (RealPython/Medium), use stored context
    if extraction_method == 'web_download':
        # Get stored context (from extract_from_json.py)
        context_before = img_meta.get('context_before', '')
        context_after = img_meta.get('context_after', '')
        author_caption = img_meta.get('author_caption', '')
        
        # Generate VLM description if requested
        vlm_description = ""
        if captioner and img_meta.get('filepath'):
            image_path = Path(img_meta['filepath'])
            if image_path.exists():
                try:
                    vlm_description = captioner.generate_caption(str(image_path))
                except Exception as e:
                    logging.warning(f"VLM failed for {img_meta['image_id']}: {e}")
        
        # Build enriched caption with all available information
        caption_parts = []
        
        if author_caption:
            caption_parts.append(f"Figure caption: {author_caption}")
        
        if context_before or context_after:
            context_text = f"Context: {context_before} ... {context_after}"
            caption_parts.append(context_text)
        
        if vlm_description:
            caption_parts.append(f"Visual description: {vlm_description}")
        
        if caption_parts:
            enriched_caption = "\n".join(caption_parts)
            enriched_caption += "\n\nNote: Use only context text that is relevant to understanding this image. Ignore surrounding text if it discusses unrelated topics."
        else:
            enriched_caption = "Image from web article (no caption or context available)"
        
        img_meta['enriched_caption'] = enriched_caption
        img_meta['vlm_description'] = vlm_description
        # context_before and context_after already stored
        
        return img_meta
    
    # For PDF images, extract context from page
    page_num = img_meta.get('page_num')
    bbox_dict = img_meta.get('bbox')
    
    if not page_num or not bbox_dict:
        logging.warning(f"Skipping {img_meta['image_id']} - no page_num or bbox")
        return img_meta

    if doc_id not in pdf_docs:
        pdf_path = raw_papers_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            print(f"  ⚠️  PDF not found: {pdf_path}")
            return img_meta
        pdf_docs[doc_id] = fitz.open(pdf_path)
    
    doc = pdf_docs[doc_id]
    page = doc[page_num - 1]  

    bbox = fitz.Rect(
        bbox_dict['x0'],
        bbox_dict['y0'],
        bbox_dict['x1'],
        bbox_dict['y1']
    )

    context = extract_surrounding_context(page, bbox, context_chars=200)

    vlm_description = ""
    if captioner:
        image_path = img_meta['filepath']
        print(f"  🎨 Generating caption for {img_meta['filename']}...")
        vlm_description = captioner.generate_caption(image_path, max_length=1024)

    author_caption = context.get('figure_caption', '')
    context_before = context.get('before', '')
    context_after = context.get('after', '')

    enriched_parts = []
    
    if author_caption:
        enriched_parts.append(f"Figure caption: {author_caption}")
    
    if vlm_description and vlm_description != "Error generating caption":
        enriched_parts.append(f"Visual description: {vlm_description}")
    
    context_text = ""
    if context_before:
        context_text += context_before
    if context_after:
        if context_text:
            context_text += " ... "
        context_text += context_after
    
    if context_text:
        enriched_parts.append(f"Context: {context_text}")

    if enriched_parts:
        enriched_parts.append(
            "\nNote: Use only context text that is relevant to understanding this image. "
            "Ignore surrounding text if it discusses unrelated topics."
        )
    
    enriched_caption = "\n".join(enriched_parts) if enriched_parts else "No caption or context found"

    img_meta.update({
        'enriched_caption': enriched_caption,
        'vlm_description': vlm_description,
        'author_caption': author_caption,
        'context_before': context_before,
        'context_after': context_after
    })
    
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
    enriched_images = []
    pdf_docs = {}  
    
    total = len(images_metadata)
    mode = "VLM descriptions + context" if captioner else "captions + context only"
    print(f"\n🎨 Enriching {total} images with {mode}...\n")
    
    if captioner:
        print("⏱️  Rate limit: 20 req/min (3 sec delay between requests)\n")
    
    for idx, img_meta in enumerate(images_metadata, 1):
        print(f"[{idx}/{total}] {img_meta['image_id']}")
        
        try:
            enriched = enrich_single_image(
                img_meta,
                pdf_docs,
                raw_papers_dir,
                captioner=captioner
            )
            enriched_images.append(enriched)
            
            # Rate limiting for Cohere API (20 req/min = 1 req every 3 sec)
            if captioner and idx < total:
                time.sleep(3.5)  # 3.5 sec to be safe
            
        except Exception as e:
            logging.error(f"  Error enriching image: {e}")
            enriched_images.append(img_meta)  # Keep original
    
 
    for doc in pdf_docs.values():
        doc.close()
    
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
    project_root = Path(__file__).parent.parent
    
    if metadata_path is None:
        metadata_path = project_root / "data" / "processed" / "images_metadata.json"
    if raw_papers_dir is None:
        raw_papers_dir = project_root / "data" / "raw" / "papers"
    
    if not metadata_path.exists():
        logging.error(f"Metadata file not found: {metadata_path}")
        return 0
    
    if not raw_papers_dir.exists():
        logging.error(f"Raw papers directory not found: {raw_papers_dir}")
        return 0
    
    # Load all images metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        all_images = json.load(f)
    
    # Filter images for this document
    doc_images = [img for img in all_images if img['doc_id'] == doc_id]
    
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
    
    # DEBUG: Check if enriched_caption exists
    for img_id, enriched_img in enriched_dict.items():
        has_caption = 'enriched_caption' in enriched_img
        logging.debug(f"  {img_id}: enriched_caption={has_caption}")
    
    # Save updated metadata
    save_images_metadata(all_images, metadata_path)
    
    return len(enriched_doc_images)

def main():
    project_root = Path(__file__).parent.parent
    metadata_path = project_root / "data" / "processed" / "images_metadata.json"
    raw_papers_dir = project_root / "data" / "raw" / "papers"
    
    if not metadata_path.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        print("Run extract_images_smart.py first!")
        return
    
    if not raw_papers_dir.exists():
        print(f"❌ Raw papers directory not found: {raw_papers_dir}")
        return
    
    print("=" * 70)
    print("🎨 Image Enrichment - AI/ML Course Assistant")
    print("=" * 70)
    print(f"Metadata: {metadata_path}")
    print(f"Papers: {raw_papers_dir}")
    print("=" * 70)
 
    print("\n📖 Loading images metadata...")
    images_metadata = load_images_metadata(metadata_path)
    print(f"Found {len(images_metadata)} images")

    already_enriched = sum(
        1 for img in images_metadata 
        if img.get('enriched_caption') is not None
    )
    
    if already_enriched > 0:
        print(f"⚠️  {already_enriched} images already enriched")
        response = input("Re-enrich all images? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    print("\n🤖 Loading OpenAI GPT-4.1-mini Vision API...")
    try:
        captioner = ImageCaptioner(
            model_name="gpt-4.1-mini",
            api_key=None  # Uses OPENAI_API_KEY env variable
        )
    except ValueError as e:
        print(f"⚠️  {e}")
        print("Continuing without VLM (captions + context only)")
        captioner = None

    enriched_metadata = enrich_all_images(
        images_metadata,
        raw_papers_dir,
        captioner=captioner
    )

    save_images_metadata(enriched_metadata, metadata_path)
    
    print("\n" + "=" * 70)
    print("✅ Image enrichment complete!")
    print("=" * 70)
    print("\n📊 Enrichment Summary:")
 
    has_caption = sum(1 for img in enriched_metadata if img.get('author_caption'))
    has_context = sum(1 for img in enriched_metadata if img.get('context_before') or img.get('context_after'))
    has_vlm = sum(1 for img in enriched_metadata if img.get('vlm_description'))
    
    print(f"  - Images with author captions: {has_caption}/{len(enriched_metadata)}")
    print(f"  - Images with text context: {has_context}/{len(enriched_metadata)}")
    print(f"  - Images with VLM descriptions: {has_vlm}/{len(enriched_metadata)}")

    if enriched_metadata:
        print(f"\n📝 Example enriched caption:")
        example = enriched_metadata[0]
        print(f"Image: {example['image_id']}")
        enriched_caption = example.get('enriched_caption', '')
        if enriched_caption:
            preview = enriched_caption[:300] + "..." if len(enriched_caption) > 300 else enriched_caption
            print(f"\n{preview}")
    
    print("\n🔜 Next steps:")
    print("  1. python index/chunk_documents.py")
    print("  2. python index/generate_embeddings.py")
    print("  3. python index/build_index.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
