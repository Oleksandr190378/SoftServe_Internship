"""
Chunk documents with image tracking for multimodal retrieval.

This module implements smart chunking that:
1. Splits text into ~500 token chunks (≈1800 characters for English)
2. Tracks which chunks contain image references
3. Links chunks to nearby images via page_num (PDF) or position (JSON)

Supports both document types:
- PDF documents (arXiv): Page-based image linking (page_num)
- JSON documents (RealPython/Medium): Position-based linking (image_index)

Embedding model: OpenAI text-embedding-3-small (1536 dims)
Chunk size: ~500 tokens = ~1800 chars (3.5 chars/token for English text)
Overlap: ~60 tokens = ~200 chars
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional
from pathlib import Path
import json
import re
import logging

logger = logging.getLogger(__name__)


def extract_figure_references(text: str) -> List[str]:
    """
    Find references to figures/tables in chunk text.
    
    Args:
        text: Chunk text to search
        
    Returns:
        List of figure/table references (e.g., ["Figure 1", "Table 2"])
    """
    pattern = r'\b(Figure|Fig\.|Table)\s+\d+\b'
    matches = re.findall(pattern, text, re.IGNORECASE)

    normalized = []
    for match in matches:
        if 'fig' in match.lower():
            normalized.append(re.sub(r'Fig\.', 'Figure', match, flags=re.IGNORECASE))
        else:
            normalized.append(match)

    return list(set(normalized))


def estimate_page_number(
    chunk_index: int,
    total_chunks: int,
    total_pages: int
) -> int:
    """
    Estimate page number for a chunk based on its position (for PDF documents only).
    
    Simple linear approximation: chunk 0 → page 1, last chunk → last page
    
    Args:
        chunk_index: Index of chunk (0-based)
        total_chunks: Total number of chunks in document
        total_pages: Total pages in original PDF
        
    Returns:
        Estimated page number (1-based), or None if not applicable
    """
    if total_chunks == 0 or total_pages == 0:
        return None

    page_num = int((chunk_index / total_chunks) * total_pages) + 1

    return max(1, min(page_num, total_pages))


def chunk_document_with_image_tracking(
    doc_id: str,
    full_text: str,
    total_pages: int,
    images_metadata: List[Dict],
    chunk_size: int = 1800,  
    chunk_overlap: int = 200  
) -> List[Dict]:
    """
    Split document into chunks and track image relationships.
    
    Supports two document types:
    1. PDF documents (arXiv): Use page_num for image linking
    2. JSON documents (RealPython/Medium): Use image_index for sequential linking
    
    Anti-hallucination strategy:
    - has_figure_references: True if chunk explicitly mentions "Figure X" or "Table Y"
    - related_image_ids: 
      * PDF: Images on EXACT same page (strong link)
      * JSON: Images estimated to be in same text region (strong link)
    - nearby_image_ids: 
      * PDF: Images on ±1 page (weak link)
      * JSON: Images in adjacent text regions (weak link)
    
    Args:
        doc_id: Document identifier
        full_text: Complete text of document
        total_pages: Number of pages in PDF (0 for JSON documents)
        images_metadata: List of image metadata with page_num (PDF) or image_index (JSON)
        chunk_size: Target chunk size in characters (~1800 = ~500 tokens)
        chunk_overlap: Overlap between chunks in characters (~200 = ~60 tokens)
        
    Returns:
        List of chunk metadata dicts with explicit image linking fields
        
    Note:
        Chunk size calibrated for OpenAI text-embedding-3-small:
        - English text: ~3.5 chars/token
        - 500 tokens ≈ 1800 characters
        - 60 tokens overlap ≈ 200 characters
    """

    is_pdf = total_pages > 0
    is_json = any(img.get('extraction_method') == 'web_download' for img in images_metadata)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  
        separators=["\n\n", "\n", ". ", " ", ""]
    )
 
    chunks = splitter.create_documents([full_text])

    chunks_with_metadata = []
    
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.page_content
        
        # Find figure/table references in chunk
        image_refs = extract_figure_references(chunk_text)
        
        if is_pdf:
            # PDF documents: Use page-based linking
            page_num = estimate_page_number(i, len(chunks), total_pages)
            
            # Find related images (exact same page only)
            related_image_ids = [
                img['image_id'] for img in images_metadata 
                if img.get('page_num', 0) == page_num
            ]
            
            # Find nearby images (±1 page for optional context)
            nearby_image_ids = [
                img['image_id'] for img in images_metadata 
                if abs(img.get('page_num', 0) - page_num) == 1
            ]
            
        else:
            # JSON documents: Use position-based linking (no pages)
            page_num = None
            
            # Estimate chunk position in document (0.0 to 1.0)
            chunk_start_pos = sum(len(chunks[j].page_content) for j in range(i))
            chunk_mid_pos = chunk_start_pos + len(chunk_text) / 2
            chunk_relative_pos = chunk_mid_pos / len(full_text) if len(full_text) > 0 else 0
            
            # For JSON docs, estimate image position based on image_index
            # Assume images are roughly evenly distributed
            total_images = len(images_metadata)
            
            related_image_ids = []
            nearby_image_ids = []
            
            if total_images > 0:
                for img in images_metadata:
                    img_index = img.get('image_index', 1)
                    # Estimate image relative position (0.0 to 1.0)
                    img_relative_pos = (img_index - 0.5) / total_images
                    
                    # Distance between chunk and image (in relative document position)
                    distance = abs(chunk_relative_pos - img_relative_pos)
                    
                    # Related: within ~15% of document length (relaxed for shorter docs)
                    if distance < 0.15:
                        related_image_ids.append(img['image_id'])
                    # Nearby: within ~30% of document length (but not already related)
                    elif distance < 0.30:
                        nearby_image_ids.append(img['image_id'])
        
        # Create metadata with explicit linking fields
        chunk_meta = {
            "chunk_id": f"{doc_id}_chunk_{i:04d}",
            "doc_id": doc_id,
            "text": chunk_text,
            "chunk_index": i,
            "page_num": page_num,  # None for JSON documents
            "char_count": len(chunk_text),
            "word_count": len(chunk_text.split()),
            "has_figure_references": len(image_refs) > 0,
            "image_references": image_refs,
            "related_image_ids": related_image_ids,
            "nearby_image_ids": nearby_image_ids,
            "extraction_method": "pdf" if is_pdf else "json"
        }
        
        chunks_with_metadata.append(chunk_meta)
    
    return chunks_with_metadata


# Removed chunk_all_documents() - replaced by integration into run_pipeline.py
# This function expected files on disk, but we process in-memory in the pipeline


if __name__ == "__main__":
    # Simple test with mock data
    logging.basicConfig(level=logging.INFO)
    
    # Test JSON document (no pages)
    test_text = "Sample text for testing chunking. " * 200  # ~6000 chars
    test_images = [
        {"image_id": "img_001", "image_index": 2, "extraction_method": "web_download"},
        {"image_id": "img_002", "image_index": 5, "extraction_method": "web_download"}
    ]
    
    logger.info("Testing JSON document chunking (no pages)...")
    chunks = chunk_document_with_image_tracking(
        doc_id="test_json",
        full_text=test_text,
        total_pages=0,  # JSON has no pages
        images_metadata=test_images,
        chunk_size=1800,
        chunk_overlap=150
    )
    
    logger.info(f"Created {len(chunks)} chunks")
    logger.info(f"Chunks with related images: {sum(1 for c in chunks if len(c['related_image_ids']) > 0)}")
    
    if chunks:
        sample = chunks[0]
        logger.info(f"Sample chunk: {sample['chunk_id']}, page={sample['page_num']}, method={sample['extraction_method']}")
