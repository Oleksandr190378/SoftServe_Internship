"""
Unified pipeline for processing documents - AI/ML Course Assistant

This pipeline orchestrates all stages of document processing:
1. Extract: Images and text from PDFs
2. Caption: Generate enriched captions for images
3. Chunk: Split text into semantic chunks
4. Embed: Generate embeddings for chunks and images
5. Index: Store in ChromaDB

Features:
- Incremental processing (skip already processed docs)
- Resume after interruption
- Force reprocessing if needed
- Registry tracking with status and costs

Usage:
    # Process all documents (skip completed)
    python run_pipeline.py process --all
    
    # Process specific document
    python run_pipeline.py process --doc-id arxiv_1706_03762
    
    # Force reprocess (ignore registry)
    python run_pipeline.py process --doc-id arxiv_1706_03762 --force
    
    # Process only new documents
    python run_pipeline.py process --new-only
    
    # Show processing status
    python run_pipeline.py status
"""

import json
import logging
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

from ingest.extract_images_smart import extract_document as extract_pdf_document
from ingest.extract_from_json import extract_json_document
from ingest.enrich_images import generate_captions_for_doc
from index.chunk_documents import chunk_document_with_image_tracking
from index.embedding_utils import embed_document

# ChromaDB imports for Stage 5
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

PROJECT_ROOT = Path(__file__).parent
REGISTRY_PATH = PROJECT_ROOT / "data" / "processed_docs.json"
RAW_PAPERS_DIR = PROJECT_ROOT / "data" / "raw" / "papers"
IMAGES_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "images"

# ChromaDB configuration
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536



def load_registry(registry_path: Path = REGISTRY_PATH) -> Dict:
    """Load processing registry from JSON file."""
    if registry_path.exists():
        with open(registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_registry(registry: Dict, registry_path: Path = REGISTRY_PATH):
    """Save processing registry to JSON file."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    logging.debug(f"Registry saved: {len(registry)} documents")


def update_registry(doc_id: str, updates: Dict, registry_path: Path = REGISTRY_PATH):
    """
    Update registry for a specific document.
    
    Args:
        doc_id: Document identifier
        updates: Dictionary with updates (supports nested keys like "stages.extract")
        registry_path: Path to registry file
    
    Example:
        update_registry("arxiv_1706_03762", {
            "status": "in_progress",
            "stages.extract": "completed",
            "stats.images_count": 8
        })
    """
    registry = load_registry(registry_path)
    
    if doc_id not in registry:
        registry[doc_id] = {
            "doc_id": doc_id,
            "status": "pending",
            "stages": {},
            "stats": {},
            "cost": {}
        }

    for key, value in updates.items():
        if '.' in key:  # Nested key like "stages.extract"
            parts = key.split('.')
            target = registry[doc_id]
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        else:
            registry[doc_id][key] = value
    
    save_registry(registry, registry_path)


def is_stage_completed(doc_id: str, stage: str, registry_path: Path = REGISTRY_PATH) -> bool:
    """Check if a processing stage is completed for a document."""
    registry = load_registry(registry_path)
    if doc_id not in registry:
        return False
    return registry[doc_id].get("stages", {}).get(stage) == "completed"


def is_document_completed(doc_id: str, registry_path: Path = REGISTRY_PATH) -> bool:
    """Check if document is fully processed."""
    registry = load_registry(registry_path)
    if doc_id not in registry:
        return False
    return registry[doc_id].get("status") == "completed"



def index_to_chromadb(
    doc_id: str,
    chunks_with_embeddings: List[Dict],
    images_with_embeddings: List[Dict]
) -> Dict:
    """
    Index chunks and images to ChromaDB collections.
    
    Creates/updates two collections:
    - text_chunks: Text chunks with embeddings and metadata
    - image_captions: Image captions with embeddings and metadata
    
    Supports incremental indexing - skips already indexed items.
    
    Args:
        doc_id: Document identifier
        chunks_with_embeddings: List of chunks with embeddings
        images_with_embeddings: List of images with embeddings
        
    Returns:
        Dict with indexing statistics
    """
    logging.info(f"📊 Indexing {doc_id} to ChromaDB")
    
    # Ensure ChromaDB directory exists
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMS
    )
    
    # Statistics
    stats = {
        "text_chunks_added": 0,
        "text_chunks_skipped": 0,
        "images_added": 0,
        "images_skipped": 0
    }
    
    # ========================================================================
    # Index text chunks
    # ========================================================================
    logging.info(f"Indexing {len(chunks_with_embeddings)} text chunks...")
    
    # Load or create text_chunks collection
    try:
        text_store = Chroma(
            collection_name="text_chunks",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR / "text_chunks")
        )
        
        # Check existing IDs to avoid duplicates
        try:
            existing_collection = text_store._collection
            existing_ids = set(existing_collection.get()["ids"])
        except:
            existing_ids = set()
            
    except Exception as e:
        logging.info(f"Creating new text_chunks collection")
        text_store = Chroma(
            collection_name="text_chunks",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR / "text_chunks")
        )
        existing_ids = set()
    
    # Prepare new chunks (skip existing)
    new_chunks_texts = []
    new_chunks_metadatas = []
    new_chunks_ids = []
    new_chunks_embeddings = []
    
    for chunk in chunks_with_embeddings:
        chunk_id = chunk['chunk_id']
        
        if chunk_id in existing_ids:
            stats["text_chunks_skipped"] += 1
            continue
        
        new_chunks_texts.append(chunk['text'])
        new_chunks_ids.append(chunk_id)
        new_chunks_embeddings.append(chunk['embedding'])
        
        # Metadata (convert lists to comma-separated strings for ChromaDB)
        metadata = {
            'chunk_id': chunk['chunk_id'],
            'doc_id': chunk['doc_id'],
            'chunk_index': chunk['chunk_index'],
            'page_num': chunk['page_num'],
            'char_count': chunk['char_count'],
            'word_count': chunk['word_count'],
            'has_figure_references': chunk['has_figure_references'],
            'image_references': ','.join(chunk['image_references']),
            'related_image_ids': ','.join(chunk['related_image_ids']),
            'nearby_image_ids': ','.join(chunk['nearby_image_ids']),
            'extraction_method': chunk.get('extraction_method', 'unknown')
        }
        new_chunks_metadatas.append(metadata)
    
    # Add new chunks to collection
    if new_chunks_texts:
        text_store.add_texts(
            texts=new_chunks_texts,
            metadatas=new_chunks_metadatas,
            ids=new_chunks_ids,
            embeddings=new_chunks_embeddings
        )
        stats["text_chunks_added"] = len(new_chunks_texts)
        logging.info(f"✅ Added {len(new_chunks_texts)} new text chunks")
    else:
        logging.info(f"⏭️  All {len(chunks_with_embeddings)} text chunks already indexed")
    
    # ========================================================================
    # Index image captions
    # ========================================================================
    logging.info(f"Indexing {len(images_with_embeddings)} image captions...")
    
    # Load or create image_captions collection
    try:
        image_store = Chroma(
            collection_name="image_captions",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR / "image_captions")
        )
        
        # Check existing IDs
        try:
            existing_collection = image_store._collection
            existing_ids = set(existing_collection.get()["ids"])
        except:
            existing_ids = set()
            
    except Exception as e:
        logging.info(f"Creating new image_captions collection")
        image_store = Chroma(
            collection_name="image_captions",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR / "image_captions")
        )
        existing_ids = set()
    
    # Prepare new images (skip existing)
    new_images_texts = []
    new_images_metadatas = []
    new_images_ids = []
    new_images_embeddings = []
    
    for img in images_with_embeddings:
        image_id = img['image_id']
        
        if image_id in existing_ids:
            stats["images_skipped"] += 1
            continue
        
        new_images_texts.append(img['caption_for_embedding'])
        new_images_ids.append(image_id)
        new_images_embeddings.append(img['embedding'])
        
        # Metadata
        metadata = {
            'image_id': img['image_id'],
            'doc_id': img['doc_id'],
            'filename': img['filename'],
            'page_num': img.get('page_num', img.get('image_index', 0)),  # PDF uses page_num, JSON uses image_index
            'region_index': img.get('region_index', 0),
            'width': img.get('width', 0),
            'height': img.get('height', 0),
            'format': img.get('format', ''),
            'author_caption': img.get('author_caption', ''),
            'extraction_method': img.get('extraction_method', '')
        }
        new_images_metadatas.append(metadata)
    
    # Add new images to collection
    if new_images_texts:
        image_store.add_texts(
            texts=new_images_texts,
            metadatas=new_images_metadatas,
            ids=new_images_ids,
            embeddings=new_images_embeddings
        )
        stats["images_added"] = len(new_images_texts)
        logging.info(f"✅ Added {len(new_images_texts)} new image captions")
    else:
        logging.info(f"⏭️  All {len(images_with_embeddings)} images already indexed")
    
    logging.info(f"✅ Indexing complete: {stats['text_chunks_added']} chunks + {stats['images_added']} images added")
    
    return stats


def detect_document_type(doc_id: str) -> str:
    """Detect document type based on doc_id prefix."""
    if doc_id.startswith("arxiv_"):
        return "pdf"
    elif doc_id.startswith("realpython_") or doc_id.startswith("medium_"):
        return "json"
    else:
        raise ValueError(f"Unknown document type for doc_id: {doc_id}")


def process_document(
    doc_id: str,
    force: bool = False,
    use_vlm: bool = True
) -> Dict:
    """
    Process a single document through all stages.
    
    Args:
        doc_id: Document identifier (PDF filename without extension)
        force: If True, reprocess even if already completed
        use_vlm: If True, use Vision-LM for image captions (costs API calls)
    
    Returns:
        Processing result with stats and costs
    """
    logging.info(f"{'='*70}")
    logging.info(f"Processing document: {doc_id}")
    logging.info(f"{'='*70}")

    if is_document_completed(doc_id) and not force:
        logging.info(f"⏭️  Skipping - already processed")
        registry = load_registry()
        return registry[doc_id]
    
    if force:
        logging.info(f"🔄 Force reprocessing enabled")

    update_registry(doc_id, {
        "status": "in_progress",
        "started_at": datetime.now().isoformat()
    })
    
    result = {
        "doc_id": doc_id,
        "stages_completed": [],
        "errors": []
    }
    
    try:
        doc_type = detect_document_type(doc_id)
        logging.info(f"📋 Document type: {doc_type.upper()}")
        
        # ====================================================================
        # Stage 1: Extract images and text (PDF or JSON)
        # ====================================================================
        if not is_stage_completed(doc_id, "extract") or force:
            logging.info(f"\n📄 Stage 1: Extracting images and text...")
            
            if doc_type == "pdf":
                extract_result = extract_pdf_document(
                    doc_id=doc_id,
                    input_dir=RAW_PAPERS_DIR,
                    output_dir=IMAGES_OUTPUT_DIR
                )
            elif doc_type == "json":
                extract_result = extract_json_document(
                    doc_id=doc_id,
                    output_dir=IMAGES_OUTPUT_DIR
                )
            else:
                raise ValueError(f"Unsupported document type: {doc_type}")
            
            if "error" in extract_result:
                raise Exception(f"Extraction failed: {extract_result['error']}")

            # Store extract_result in memory for Stage 3 (NOT in registry - too large)
            result["extract_result"] = extract_result
            
            # Update registry with stats only
            update_registry(doc_id, {
                "stages.extract": "completed",
                "stats.images_count": extract_result["images_count"],
                "stats.text_length": extract_result["text_length"]
            })
            
            result["stages_completed"].append("extract")
            logging.info(f"✅ Extract: {extract_result['images_count']} images, "
                        f"{extract_result['text_length']} chars")
        else:
            logging.info(f"⏭️  Stage 1: Extract - already completed")
            # Stage 1 completed - need full pipeline restart with --force
            logging.warning(f"⚠️  extract_result not available (stage completed). "
                          f"Use --force for full pipeline restart.")
        
      
        if not is_stage_completed(doc_id, "caption") or force:
            logging.info(f"\n🎨 Stage 2: Generating enriched captions...")

            registry = load_registry()
            images_count = registry[doc_id].get("stats", {}).get("images_count", 0)
            
            if images_count > 0:
                captioned_count = generate_captions_for_doc(
                    doc_id=doc_id,
                    use_vlm=use_vlm
                )
                
                # Estimate cost (OpenAI GPT-4.1-mini Vision: ~$0.015 per image)
                caption_cost = captioned_count * 0.015 if use_vlm else 0.0
                
                update_registry(doc_id, {
                    "stages.caption": "completed",
                    "cost.captions": round(caption_cost, 3)
                })
                
                result["stages_completed"].append("caption")
                logging.info(f"✅ Caption: {captioned_count} images enriched "
                           f"(cost: ${caption_cost:.3f})")
            else:
                logging.info(f"⏭️  No images to caption")
                update_registry(doc_id, {"stages.caption": "completed"})
        else:
            logging.info(f"⏭️  Stage 2: Caption - already completed")
        
        # ====================================================================
        # Stage 3: Chunk text (IN-MEMORY)
        # ====================================================================
        if not is_stage_completed(doc_id, "chunk") or force:
            logging.info(f"\n📝 Stage 3: Chunking text...")
            
            # Use extract_result from Stage 1 (in-memory only, not in registry)
            if "extract_result" not in result:
                logging.error(f"❌ No extract_result in memory for {doc_id}")
                logging.error(f"   Stage 1 must complete in same pipeline run")
                logging.error(f"   Use --force to restart full pipeline")
                return result
            
            extract_result = result["extract_result"]
            
            # Load images metadata
            images_metadata_path = PROJECT_ROOT / "data" / "processed" / "images_metadata.json"
            if images_metadata_path.exists():
                with open(images_metadata_path, 'r', encoding='utf-8') as f:
                    all_images = json.load(f)
                doc_images = [img for img in all_images if img['doc_id'] == doc_id]
            else:
                doc_images = []
            
            # Determine total_pages (0 for JSON documents)
            total_pages = extract_result.get("document_metadata", {}).get("num_pages", 0)
            
            # Chunk document
            chunks = chunk_document_with_image_tracking(
                doc_id=doc_id,
                full_text=extract_result["full_text"],
                total_pages=total_pages,
                images_metadata=doc_images,
                chunk_size=1700,  # ~500 tokens
                chunk_overlap=150  # ~60 tokens
            )
            
            # Store chunks in result for next stages (in-memory only)
            result["chunks"] = chunks
            
            # Update registry with stats only (chunks go to ChromaDB)
            update_registry(doc_id, {
                "stages.chunk": "completed",
                "stats.chunks_count": len(chunks)
            })
            
            # Statistics
            with_refs = sum(1 for c in chunks if c['has_figure_references'])
            with_related = sum(1 for c in chunks if len(c['related_image_ids']) > 0)
            
            result["stages_completed"].append("chunk")
            logging.info(f"✅ Chunk: {len(chunks)} chunks created")
            logging.info(f"   - {with_refs} with figure references")
            logging.info(f"   - {with_related} with related images")
        else:
            logging.info(f"⏭️  Stage 3: Chunk - already completed")
            # If chunk stage already completed, need to re-chunk or use --force
            logging.warning(f"⚠️  Chunks not available (stage already completed). "
                          f"Use --force to regenerate chunks for embedding.")
            return result
        
        # ====================================================================
        # Stage 4: Generate embeddings (IN-MEMORY)
        # ====================================================================
        if not is_stage_completed(doc_id, "embed") or force:
            logging.info(f"\n🔢 Stage 4: Generating embeddings...")
            
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            client = OpenAI(api_key=api_key)
            
            # Load images metadata for this document
            images_metadata_path = PROJECT_ROOT / "data" / "processed" / "images_metadata.json"
            if images_metadata_path.exists():
                with open(images_metadata_path, 'r', encoding='utf-8') as f:
                    all_images = json.load(f)
                doc_images = [img for img in all_images if img['doc_id'] == doc_id]
            else:
                doc_images = []
            
            # Generate embeddings
            embed_result = embed_document(
                doc_id=doc_id,
                chunks=chunks,
                images=doc_images,
                client=client
            )
            
            # Store embeddings in result for Stage 5
            result["chunks_with_embeddings"] = embed_result["chunks_with_embeddings"]
            result["images_with_embeddings"] = embed_result["images_with_embeddings"]
            
            # Update registry
            update_registry(doc_id, {
                "stages.embed": "completed",
                "cost.embeddings": embed_result["cost"]
            })
            
            result["stages_completed"].append("embed")
            logging.info(f"✅ Embed: {embed_result['stats']['chunks_embedded']} chunks, "
                        f"{embed_result['stats']['images_embedded']} images")
        else:
            logging.info(f"⏭️  Stage 4: Embed - already completed")
            # If already completed, we need to load embeddings for Stage 5
            # For now, we'll skip and require re-running with --force
            logging.warning(f"⚠️  Embeddings not loaded (stage already completed). "
                          f"Use --force to regenerate.")
        
        # ====================================================================
        # Stage 5: Index to ChromaDB
        # ====================================================================
        if not is_stage_completed(doc_id, "index") or force:
            logging.info(f"\n💾 Stage 5: Indexing to ChromaDB...")
            
            # Check if we have embeddings in memory
            if "chunks_with_embeddings" not in result or "images_with_embeddings" not in result:
                raise ValueError(
                    f"Cannot index {doc_id}: embeddings not found in memory. "
                    f"Use --force to regenerate embeddings."
                )
            
            # Index to ChromaDB
            index_stats = index_to_chromadb(
                doc_id,
                result["chunks_with_embeddings"],
                result["images_with_embeddings"]
            )
            
            logging.info(f"✅ Index: {index_stats['text_chunks_added']} chunks, "
                        f"{index_stats['images_added']} images added")
            
            update_registry(doc_id, {
                "stages.index": "completed",
                "stats.indexed_chunks": index_stats['text_chunks_added'] + index_stats['text_chunks_skipped'],
                "stats.indexed_images": index_stats['images_added'] + index_stats['images_skipped']
            })
            
            result["stages_completed"].append("index")
        else:
            logging.info(f"⏭️  Stage 5: Index - already completed")
        
        # ====================================================================
        # Mark as completed
        # ====================================================================
        registry = load_registry()
        total_cost = sum(registry[doc_id].get("cost", {}).values())
        
        update_registry(doc_id, {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "cost.total": round(total_cost, 3)
        })
        
        logging.info(f"\n{'='*70}")
        logging.info(f"✅ Document processing completed!")
        logging.info(f"Total cost: ${total_cost:.3f}")
        logging.info(f"{'='*70}")
        
        return registry[doc_id]
        
    except Exception as e:
        logging.error(f"\n❌ Error processing {doc_id}: {e}")
        update_registry(doc_id, {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })
        result["errors"].append(str(e))
        raise


def process_all_documents(
    force: bool = False,
    new_only: bool = False,
    use_vlm: bool = True
):
    """
    Process all documents in data/raw/papers/.
    
    Args:
        force: Reprocess all documents
        new_only: Only process documents not in registry
        use_vlm: Use Vision-LM for captions
    """
    # Find all PDFs
    pdf_files = list(RAW_PAPERS_DIR.glob("*.pdf"))
    
    if not pdf_files:
        logging.error(f"No PDF files found in {RAW_PAPERS_DIR}")
        return
    
    doc_ids = [pdf.stem for pdf in pdf_files]
    
    # Filter based on registry
    registry = load_registry()
    
    if new_only:
        doc_ids = [doc_id for doc_id in doc_ids if doc_id not in registry]
        logging.info(f"Processing {len(doc_ids)} new documents (not in registry)")
    else:
        logging.info(f"Processing {len(doc_ids)} documents total")
    
    if not doc_ids:
        logging.info("No documents to process")
        return
    
    # Process each document
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for idx, doc_id in enumerate(doc_ids, 1):
        logging.info(f"\n{'='*70}")
        logging.info(f"Document [{idx}/{len(doc_ids)}]: {doc_id}")
        logging.info(f"{'='*70}")
        
        try:
            if is_document_completed(doc_id) and not force:
                logging.info(f"⏭️  Already completed - skipping")
                skipped_count += 1
                continue
            
            process_document(doc_id, force=force, use_vlm=use_vlm)
            success_count += 1
            
        except Exception as e:
            logging.error(f"❌ Failed: {e}")
            failed_count += 1
            continue
    
    # Summary
    logging.info(f"\n{'='*70}")
    logging.info(f"📊 Processing Summary")
    logging.info(f"{'='*70}")
    logging.info(f"Total documents: {len(doc_ids)}")
    logging.info(f"✅ Successful: {success_count}")
    logging.info(f"⏭️  Skipped: {skipped_count}")
    logging.info(f"❌ Failed: {failed_count}")
    logging.info(f"{'='*70}")


def show_status():
    """Show processing status for all documents."""
    registry = load_registry()
    
    if not registry:
        logging.info("Registry is empty - no documents processed yet")
        return
    
    logging.info(f"\n{'='*70}")
    logging.info(f"📊 Processing Status")
    logging.info(f"{'='*70}\n")
    
    completed = [d for d in registry.values() if d.get("status") == "completed"]
    in_progress = [d for d in registry.values() if d.get("status") == "in_progress"]
    failed = [d for d in registry.values() if d.get("status") == "failed"]
    
    logging.info(f"Total documents in registry: {len(registry)}")
    logging.info(f"  ✅ Completed: {len(completed)}")
    logging.info(f"  🔄 In Progress: {len(in_progress)}")
    logging.info(f"  ❌ Failed: {len(failed)}")
    
    # Calculate total costs
    total_cost = sum(d.get("cost", {}).get("total", 0) for d in completed)
    logging.info(f"\n💰 Total cost: ${total_cost:.3f}")
    
    # Show in-progress documents
    if in_progress:
        logging.info(f"\n🔄 In Progress:")
        for doc in in_progress:
            stages = doc.get("stages", {})
            completed_stages = [s for s, status in stages.items() if status == "completed"]
            logging.info(f"  - {doc['doc_id']}: {len(completed_stages)}/5 stages")
    
    # Show failed documents
    if failed:
        logging.info(f"\n❌ Failed:")
        for doc in failed:
            error = doc.get("error", "Unknown error")
            logging.info(f"  - {doc['doc_id']}: {error}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified document processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all documents
  python run_pipeline.py process --all
  
  # Process specific document
  python run_pipeline.py process --doc-id arxiv_1706_03762
  
  # Force reprocess
  python run_pipeline.py process --doc-id arxiv_1706_03762 --force
  
  # Process only new documents
  python run_pipeline.py process --new-only
  
  # Show status
  python run_pipeline.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process documents")
    process_parser.add_argument("--doc-id", type=str, help="Process specific document")
    process_parser.add_argument("--all", action="store_true", help="Process all documents")
    process_parser.add_argument("--new-only", action="store_true", help="Process only new documents")
    process_parser.add_argument("--force", action="store_true", help="Force reprocess")
    process_parser.add_argument("--no-vlm", action="store_true", help="Skip Vision-LM captions (save costs)")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show processing status")
    
    args = parser.parse_args()
    
    if args.command == "process":
        if args.doc_id:
            # Process single document
            process_document(args.doc_id, force=args.force, use_vlm=not args.no_vlm)
        elif args.all or args.new_only:
            # Process multiple documents
            process_all_documents(force=args.force, new_only=args.new_only, use_vlm=not args.no_vlm)
        else:
            logging.error("Specify --doc-id, --all, or --new-only")
            return
    
    elif args.command == "status":
        show_status()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
