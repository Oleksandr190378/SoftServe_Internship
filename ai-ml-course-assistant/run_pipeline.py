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
from index.build_index import index_documents_to_chromadb

# Load environment variables
load_dotenv()

# Configure logging
from utils.logging_config import setup_logging
setup_logging()

# Import centralized configuration
from config import CHUNKING, BASE_DIR

PROJECT_ROOT = BASE_DIR

# Processing configuration constants (from config)
CHUNK_SIZE = CHUNKING.CHUNK_SIZE
CHUNK_OVERLAP = CHUNKING.CHUNK_OVERLAP

# Vision API cost tracking
API_COST_PER_IMAGE = 0.015  # OpenAI GPT-4o-mini Vision cost
IMAGES_METADATA_FILE = "images_metadata.json"


def validate_environment():
    """
    Validate required environment variables before pipeline execution.
    
    Raises:
        EnvironmentError: If required variables are missing with actionable guidance
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "\n❌ OPENAI_API_KEY not found.\n\n"
            "Required for embeddings and VLM captions.\n"
            "Please create a .env file in the project root with:\n\n"
            "  OPENAI_API_KEY=your_key_here\n\n"
            "Or set the environment variable:\n"
            "  Windows: set OPENAI_API_KEY=your_key_here\n"
            "  Linux/Mac: export OPENAI_API_KEY=your_key_here\n"
        )
    logging.info("✅ Environment validation passed")

REGISTRY_PATH = PROJECT_ROOT / "data" / "processed_docs.json"
RAW_PAPERS_DIR = PROJECT_ROOT / "data" / "raw" / "papers"
IMAGES_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "images"


def load_registry(registry_path: Path = REGISTRY_PATH) -> Dict:
    """Load processing registry from JSON file."""
    if registry_path.exists():
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, OSError) as e:
            logging.error(f"Failed to read registry: {type(e).__name__}: {e}")
            return {}
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in registry: {e}")
            return {}
    return {}


def save_registry(registry: Dict, registry_path: Path = REGISTRY_PATH):
    """Save processing registry to JSON file."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        logging.debug(f"Registry saved: {len(registry)} documents")
    except (IOError, OSError) as e:
        logging.error(f"Failed to save registry: {type(e).__name__}: {e}")
        raise


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



def detect_document_type(doc_id: str) -> str:
    """Detect document type based on doc_id prefix."""
    if doc_id.startswith("arxiv_"):
        return "pdf"
    elif doc_id.startswith("realpython_") or doc_id.startswith("medium_"):
        return "json"
    else:
        raise ValueError(f"Unknown document type for doc_id: {doc_id}")


def validate_doc_id(doc_id: str) -> None:
    """
    Validate document ID and check if source file exists.
    
    Args:
        doc_id: Document identifier
        
    Raises:
        ValueError: If doc_id is invalid
        FileNotFoundError: If source file doesn't exist
    """
    if not doc_id or not doc_id.strip():
        raise ValueError("doc_id cannot be empty")
    
    doc_type = detect_document_type(doc_id)
    
    if doc_type == "pdf":
        pdf_path = RAW_PAPERS_DIR / f"{doc_id}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found for doc_id '{doc_id}'. "
                f"Expected at: {pdf_path}"
            )
    elif doc_type == "json":
        # Check if JSON document exists in medium or realpython directories
        if doc_id.startswith("medium_"):
            json_dir = PROJECT_ROOT / "data" / "raw" / "medium"
        else:  # realpython
            json_dir = PROJECT_ROOT / "data" / "raw" / "realpython"
        
        slug = doc_id.split("_", 1)[1]  # Remove prefix
        json_path = json_dir / slug / f"{slug}.json"
        
        if not json_path.exists():
            raise FileNotFoundError(
                f"JSON not found for doc_id '{doc_id}'. "
                f"Expected at: {json_path}"
            )



class ProcessingStage:
    """Base class for document processing stages."""
    
    def __init__(self, stage_name: str):
        self.stage_name = stage_name
    
    def is_completed(self, doc_id: str) -> bool:
        """Check if stage is already completed for document."""
        return is_stage_completed(doc_id, self.stage_name)
    
    def mark_completed(self, doc_id: str, stats: Dict):
        """Mark stage as completed and save stats."""
        updates = {f"stages.{self.stage_name}": "completed"}
        updates.update(stats)
        update_registry(doc_id, updates)
    
    def execute(self, doc_id: str, context: Dict, force: bool) -> Dict:
        """Execute the stage. Must be implemented by subclasses."""
        raise NotImplementedError


class ExtractStage(ProcessingStage):
    """Stage 1: Extract images and text from documents."""
    
    def __init__(self):
        super().__init__("extract")
    
    def execute(self, doc_id: str, context: Dict, force: bool) -> Dict:
        if self.is_completed(doc_id) and not force:
            logging.info(f"⏭️  Stage 1: Extract - already completed")
            logging.warning(f"⚠️  extract_result not available (stage completed). "
                          f"Use --force for full pipeline restart.")
            return context
        
        logging.info(f"\n📄 Stage 1: Extracting images and text...")
        
        doc_type = detect_document_type(doc_id)
        
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
        
        # Store in context for next stages
        context["extract_result"] = extract_result
        
        # Save stats to registry
        self.mark_completed(doc_id, {
            "stats.images_count": extract_result["images_count"],
            "stats.text_length": extract_result["text_length"]
        })
        
        logging.info(f"✅ Extract: {extract_result['images_count']} images, "
                    f"{extract_result['text_length']} chars")
        
        return context


class CaptionStage(ProcessingStage):
    """Stage 2: Generate enriched captions for images."""
    
    def __init__(self, use_vlm: bool = True):
        super().__init__("caption")
        self.use_vlm = use_vlm
    
    def execute(self, doc_id: str, context: Dict, force: bool) -> Dict:
        if self.is_completed(doc_id) and not force:
            logging.info(f"⏭️  Stage 2: Caption - already completed")
            return context
        
        logging.info(f"\n🎨 Stage 2: Generating enriched captions...")
        
        registry = load_registry()
        images_count = registry[doc_id].get("stats", {}).get("images_count", 0)
        
        if images_count > 0:
            captioned_count = generate_captions_for_doc(
                doc_id=doc_id,
                use_vlm=self.use_vlm
            )
            
            caption_cost = captioned_count * API_COST_PER_IMAGE if self.use_vlm else 0.0
            
            self.mark_completed(doc_id, {
                "cost.captions": round(caption_cost, 3)
            })
            
            logging.info(f"✅ Caption: {captioned_count} images enriched "
                       f"(cost: ${caption_cost:.3f})")
        else:
            logging.info(f"⏭️  No images to caption")
            self.mark_completed(doc_id, {})
        
        return context


class ChunkStage(ProcessingStage):
    """Stage 3: Chunk text into semantic pieces."""
    
    def __init__(self):
        super().__init__("chunk")
    
    def execute(self, doc_id: str, context: Dict, force: bool) -> Dict:
        if self.is_completed(doc_id) and not force:
            logging.info(f"⏭️  Stage 3: Chunk - already completed")
            logging.warning(f"⚠️  Chunks not available (stage already completed). "
                          f"Use --force to regenerate chunks for embedding.")
            return context
        
        logging.info(f"\n📝 Stage 3: Chunking text...")
        
        # Require extract_result from Stage 1
        if "extract_result" not in context:
            logging.error(f"❌ No extract_result in memory for {doc_id}")
            logging.error(f"   Stage 1 must complete in same pipeline run")
            logging.error(f"   Use --force to restart full pipeline")
            raise ValueError("extract_result not found in context")
        
        extract_result = context["extract_result"]
        
        # Load images metadata
        images_metadata_path = PROJECT_ROOT / "data" / "processed" / IMAGES_METADATA_FILE
        if images_metadata_path.exists():
            try:
                with open(images_metadata_path, 'r', encoding='utf-8') as f:
                    all_images = json.load(f)
                doc_images = [img for img in all_images if img['doc_id'] == doc_id]
            except (IOError, OSError, json.JSONDecodeError) as e:
                logging.warning(f"Failed to load images metadata: {type(e).__name__}: {e}")
                doc_images = []
        else:
            doc_images = []
        
        total_pages = extract_result.get("document_metadata", {}).get("num_pages", 0)
        
        chunks = chunk_document_with_image_tracking(
            doc_id=doc_id,
            full_text=extract_result["full_text"],
            total_pages=total_pages,
            images_metadata=doc_images,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Store in context for next stages
        context["chunks"] = chunks
        
        # Save stats
        self.mark_completed(doc_id, {
            "stats.chunks_count": len(chunks)
        })
        
        # Statistics
        with_refs = sum(1 for c in chunks if c['has_figure_references'])
        with_related = sum(1 for c in chunks if len(c['related_image_ids']) > 0)
        
        logging.info(f"✅ Chunk: {len(chunks)} chunks created")
        logging.info(f"   - {with_refs} with figure references")
        logging.info(f"   - {with_related} with related images")
        
        return context


class EmbedStage(ProcessingStage):
    """Stage 4: Generate embeddings for chunks and images."""
    
    def __init__(self):
        super().__init__("embed")
    
    def execute(self, doc_id: str, context: Dict, force: bool) -> Dict:
        if self.is_completed(doc_id) and not force:
            logging.info(f"⏭️  Stage 4: Embed - already completed")
            logging.warning(f"⚠️  Embeddings not loaded (stage already completed). "
                          f"Use --force to regenerate.")
            return context
        
        logging.info(f"\n🔢 Stage 4: Generating embeddings...")
        
        # Require chunks from Stage 3
        if "chunks" not in context:
            raise ValueError("chunks not found in context. Stage 3 must complete first.")
        
        chunks = context["chunks"]
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        client = OpenAI(api_key=api_key)
        
        # Load images metadata
        images_metadata_path = PROJECT_ROOT / "data" / "processed" / IMAGES_METADATA_FILE
        if images_metadata_path.exists():
            try:
                with open(images_metadata_path, 'r', encoding='utf-8') as f:
                    all_images = json.load(f)
                doc_images = [img for img in all_images if img['doc_id'] == doc_id]
            except (IOError, OSError, json.JSONDecodeError) as e:
                logging.warning(f"Failed to load images metadata: {type(e).__name__}: {e}")
                doc_images = []
        else:
            doc_images = []
        
        # Generate embeddings
        embed_result = embed_document(
            doc_id=doc_id,
            chunks=chunks,
            images=doc_images,
            client=client
        )
        
        # Store in context for Stage 5
        context["chunks_with_embeddings"] = embed_result["chunks_with_embeddings"]
        context["images_with_embeddings"] = embed_result["images_with_embeddings"]
        
        # Save stats
        self.mark_completed(doc_id, {
            "cost.embeddings": embed_result["cost"]
        })
        
        logging.info(f"✅ Embed: {embed_result['stats']['chunks_embedded']} chunks, "
                    f"{embed_result['stats']['images_embedded']} images")
        
        return context


class IndexStage(ProcessingStage):
    """Stage 5: Index to ChromaDB."""
    
    def __init__(self):
        super().__init__("index")
    
    def execute(self, doc_id: str, context: Dict, force: bool) -> Dict:
        if self.is_completed(doc_id) and not force:
            logging.info(f"⏭️  Stage 5: Index - already completed")
            return context
        
        logging.info(f"\n💾 Stage 5: Indexing to ChromaDB...")
        
        # Require embeddings from Stage 4
        if "chunks_with_embeddings" not in context or "images_with_embeddings" not in context:
            raise ValueError(
                f"Cannot index {doc_id}: embeddings not found in memory. "
                f"Use --force to regenerate embeddings."
            )
        
        index_stats = index_documents_to_chromadb(
            doc_id,
            context["chunks_with_embeddings"],
            context["images_with_embeddings"]
        )
        
        logging.info(f"✅ Index: {index_stats['text_chunks_added']} chunks, "
                    f"{index_stats['images_added']} images added")
        
        self.mark_completed(doc_id, {
            "stats.indexed_chunks": index_stats['text_chunks_added'] + index_stats['text_chunks_skipped'],
            "stats.indexed_images": index_stats['images_added'] + index_stats['images_skipped']
        })
        
        return context


def process_document(
    doc_id: str,
    force: bool = False,
    use_vlm: bool = True
) -> Dict:
    """
    Process a single document through all stages using Stage classes.
    
    Args:
        doc_id: Document identifier (PDF filename without extension)
        force: If True, reprocess even if already completed
        use_vlm: If True, use Vision-LM for image captions (costs API calls)
    
    Returns:
        Processing result with stats and costs
        
    Raises:
        ValueError: If doc_id is invalid
        FileNotFoundError: If source document doesn't exist
    """
    # Validate doc_id before processing
    validate_doc_id(doc_id)
    
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
    
    # Context stores data passed between stages
    context = {
        "doc_id": doc_id,
        "stages_completed": [],
        "errors": []
    }
    
    try:
        doc_type = detect_document_type(doc_id)
        logging.info(f"📋 Document type: {doc_type.upper()}")
        
        # Initialize processing stages
        stages = [
            ExtractStage(),
            CaptionStage(use_vlm=use_vlm),
            ChunkStage(),
            EmbedStage(),
            IndexStage()
        ]
        
        # Execute stages in sequence
        for stage in stages:
            context = stage.execute(doc_id, context, force)
            context["stages_completed"].append(stage.stage_name)
        
        # Mark as completed
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
        logging.error(f"\n❌ Error processing {doc_id}: {type(e).__name__}: {e}")
        update_registry(doc_id, {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })
        context["errors"].append(str(e))
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
    # Validate environment before any processing
    try:
        validate_environment()
    except EnvironmentError as e:
        logging.error(str(e))
        return 1
    
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
    process_parser.add_argument("--doc-id", type=str, nargs='+', help="Process specific document(s)")
    process_parser.add_argument("--all", action="store_true", help="Process all documents")
    process_parser.add_argument("--new-only", action="store_true", help="Process only new documents")
    process_parser.add_argument("--force", action="store_true", help="Force reprocess")
    process_parser.add_argument("--no-vlm", action="store_true", help="Skip Vision-LM captions (save costs)")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show processing status")
    
    args = parser.parse_args()
    
    if args.command == "process":
        if args.doc_id:
            # Process single or multiple documents
            for doc_id in args.doc_id:
                process_document(doc_id, force=args.force, use_vlm=not args.no_vlm)
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
