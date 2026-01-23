"""
Build ChromaDB vector store for text chunks and image captions.

Creates two collections:
1. text_chunks - Text chunks with embeddings and metadata
2. image_captions - Image captions with embeddings and metadata

Both collections use OpenAI text-embedding-3-small (1536 dims) embeddings.

Features:
- Cross-platform file locking to prevent race conditions
- Incremental indexing (skip duplicates)
- Native ChromaDB API (no langchain wrapper)
- Single Responsibility Principle compliance
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
import chromadb
from chromadb.config import Settings

# File locking imports (cross-platform)
if sys.platform == 'win32':
    import msvcrt
else:
    import fcntl

# Configure logging
from utils.logging_config import setup_logging
setup_logging()  # Centralized logging configuration

# Import centralized configuration
from config import EMBEDDING, CHROMA_DIR, CHUNKS_BACKUP_DIR, BASE_DIR

# Constants (from config)
EMBEDDING_MODEL = EMBEDDING.MODEL
EMBEDDING_DIMS = EMBEDDING.DIMENSIONS
MAX_ENRICHED_CAPTION_LENGTH = 1000
MAX_AUTHOR_CAPTION_LENGTH = 500

# Default paths (configurable via IndexConfig)
DEFAULT_BASE_DIR = BASE_DIR
DEFAULT_CHROMA_DIR = CHROMA_DIR
DEFAULT_CHUNKS_BACKUP_DIR = CHUNKS_BACKUP_DIR

# Legacy globals for backwards compatibility
BASE_DIR = DEFAULT_BASE_DIR
CHROMA_DIR = DEFAULT_CHROMA_DIR
CHROMA_LOCK_FILE = CHROMA_DIR / ".chroma.lock"
CHUNKS_BACKUP_DIR = DEFAULT_CHUNKS_BACKUP_DIR
CHUNKS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)


class ChromaDBLock:
    """Cross-platform file-based lock for ChromaDB operations."""
    
    def __init__(self, lock_file: Path):
        self.lock_file = lock_file
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = None
    
    def __enter__(self):
        """Acquire exclusive lock on ChromaDB directory."""
        self.file_handle = open(self.lock_file, 'w')
        
        if sys.platform == 'win32':
            # Windows: Use msvcrt
            try:
                msvcrt.locking(self.file_handle.fileno(), msvcrt.LK_NBLCK, 1)
            except IOError:
                self.file_handle.close()
                raise RuntimeError(
                    f"ChromaDB is locked by another process. "
                    f"Wait for other process to finish or remove {self.lock_file}"
                )
        else:
            # Unix/Linux/Mac: Use fcntl
            try:
                fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except IOError:
                self.file_handle.close()
                raise RuntimeError(
                    f"ChromaDB is locked by another process. "
                    f"Wait for other process to finish or remove {self.lock_file}"
                )
        
        logging.info(f"Acquired ChromaDB lock: {self.lock_file}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Release lock and cleanup.
        
        Best-effort cleanup: exceptions during unlock/cleanup are logged
        but don't mask original exceptions from the with-block.
        """
        if self.file_handle:
            # Unlock file (platform-specific)
            if sys.platform == 'win32':
                try:
                    msvcrt.locking(self.file_handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError as e:
                    # Best-effort unlock; log and continue to avoid masking original errors
                    logging.debug(f"Failed to release Windows file lock: {e}")
            else:
                try:
                    fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_UN)
                except OSError as e:
                    # Best-effort unlock; log and continue to avoid masking original errors
                    logging.debug(f"Failed to release POSIX file lock: {e}")
            
            # Close file handle
            try:
                self.file_handle.close()
            except OSError as e:
                logging.debug(f"Failed to close lock file handle: {e}")
            
            # Remove lock file (cleanup)
            try:
                self.lock_file.unlink()
            except OSError as e:
                # Lock file might already be removed or permission issue
                logging.debug(f"Failed to remove lock file {self.lock_file}: {e}")
            
            logging.info("Released ChromaDB lock")


def _save_chunks_backup(
    doc_id: str,
    chunks_with_embeddings: List[Dict],
    images_with_embeddings: List[Dict]
) -> None:
    """
    Save chunks and images to JSON backup files.
    
    This ensures data can be recovered even if ChromaDB index corrupts.
    Each document gets separate backup files for easy recovery.
    
    Args:
        doc_id: Document identifier
        chunks_with_embeddings: Text chunks with embeddings
        images_with_embeddings: Image captions with embeddings
        
    Raises:
        OSError: If backup directory cannot be created or written to
        ValueError: If doc_id is empty
    """
    if not doc_id or not isinstance(doc_id, str):
        raise ValueError("doc_id must be a non-empty string")
    
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save text chunks
    if chunks_with_embeddings:
        chunks_file = CHUNKS_BACKUP_DIR / f"{doc_id}_chunks_{timestamp}.json"
        backup_data = {
            'doc_id': doc_id,
            'timestamp': timestamp,
            'count': len(chunks_with_embeddings),
            'chunks': chunks_with_embeddings
        }
        
        try:
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 Backed up {len(chunks_with_embeddings)} chunks to {chunks_file.name}")
        except (OSError, IOError) as e:
            raise OSError(f"Failed to save chunks backup to {chunks_file}: {e}") from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid JSON data in chunks: {e}") from e
    
    # Save images
    if images_with_embeddings:
        images_file = CHUNKS_BACKUP_DIR / f"{doc_id}_images_{timestamp}.json"
        backup_data = {
            'doc_id': doc_id,
            'timestamp': timestamp,
            'count': len(images_with_embeddings),
            'images': images_with_embeddings
        }
        
        try:
            with open(images_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 Backed up {len(images_with_embeddings)} images to {images_file.name}")
        except (OSError, IOError) as e:
            raise OSError(f"Failed to save images backup to {images_file}: {e}") from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid JSON data in images: {e}") from e


def _get_existing_ids(collection) -> Set[str]:
    """
    Get set of existing document IDs from ChromaDB collection.
    
    Single responsibility: Query existing IDs for duplicate detection.
    
    Args:
        collection: ChromaDB collection object
        
    Returns:
        Set of existing document IDs
    """
    try:
        result = collection.get()
        return set(result['ids']) if result and 'ids' in result else set()
    except Exception as e:
        logging.warning(f"Could not fetch existing IDs: {e}")
        return set()


def _prepare_text_chunks_for_indexing(
    chunks_with_embeddings: List[Dict],
    existing_ids: Set[str]
) -> Tuple[List[str], List[str], List[List[float]], List[Dict], int]:
    """
    Prepare text chunks for ChromaDB indexing (skip duplicates).
    
    Single responsibility: Filter and format text chunks for indexing.
    
    Args:
        chunks_with_embeddings: List of chunks with embeddings
        existing_ids: Set of already indexed IDs
        
    Returns:
        Tuple of (ids, documents, embeddings, metadatas, skipped_count)
    """
    # Validate input
    if not isinstance(chunks_with_embeddings, list):
        logging.warning(f"Expected list, got {type(chunks_with_embeddings)}")
        return [], [], [], [], 0
    
    if not isinstance(existing_ids, set):
        existing_ids = set()  # Fallback to empty set
    
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    skipped_count = 0
    
    for chunk in chunks_with_embeddings:
        # Validate chunk structure
        if not isinstance(chunk, dict):
            logging.warning(f"Skipping non-dict chunk: {type(chunk)}")
            continue
        
        if 'chunk_id' not in chunk:
            logging.warning(f"Skipping chunk without chunk_id")
            continue
        
        chunk_id = chunk['chunk_id']
        
        # Skip if already indexed
        if chunk_id in existing_ids:
            skipped_count += 1
            continue
        
        # Validate required fields
        if 'text' not in chunk or 'embedding' not in chunk:
            logging.warning(f"Skipping chunk {chunk_id}: missing text or embedding")
            continue
        
        ids.append(chunk_id)
        documents.append(chunk['text'])
        embeddings.append(chunk['embedding'])
        
        # Metadata (convert lists to JSON strings)
        metadata = {
            'doc_id': chunk['doc_id'],
            'chunk_index': chunk['chunk_index'],
            'char_count': chunk['char_count'],
            'word_count': chunk['word_count'],
            'has_figure_references': chunk['has_figure_references'],
            'image_references': json.dumps(chunk['image_references']),
            'related_image_ids': json.dumps(chunk['related_image_ids']),
            'nearby_image_ids': json.dumps(chunk['nearby_image_ids']),
            'extraction_method': chunk.get('extraction_method', 'unknown')
        }
        
        # Add page_num only if not None (ChromaDB may not store None)
        if chunk.get('page_num') is not None:
            metadata['page_num'] = chunk['page_num']
        
        metadatas.append(metadata)
    
    return ids, documents, embeddings, metadatas, skipped_count


def _prepare_images_for_indexing(
    images_with_embeddings: List[Dict],
    existing_ids: Set[str]
) -> Tuple[List[str], List[str], List[List[float]], List[Dict], int]:
    """
    Prepare image captions for ChromaDB indexing (skip duplicates).
    
    Single responsibility: Filter and format image captions for indexing.
    
    Args:
        images_with_embeddings: List of images with embeddings
        existing_ids: Set of already indexed IDs
        
    Returns:
        Tuple of (ids, documents, embeddings, metadatas, skipped_count)
    """
    # Validate input
    if not isinstance(images_with_embeddings, list):
        logging.warning(f"Expected list, got {type(images_with_embeddings)}")
        return [], [], [], [], 0
    
    if not isinstance(existing_ids, set):
        existing_ids = set()  # Fallback to empty set
    
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    skipped_count = 0
    
    for img in images_with_embeddings:
        # Validate image structure
        if not isinstance(img, dict):
            logging.warning(f"Skipping non-dict image: {type(img)}")
            continue
        
        if 'image_id' not in img:
            logging.warning(f"Skipping image without image_id")
            continue
        
        image_id = img['image_id']
        
        # Skip if already indexed
        if image_id in existing_ids:
            skipped_count += 1
            continue
        
        # Validate required fields
        if 'caption_for_embedding' not in img or 'embedding' not in img:
            logging.warning(f"Skipping image {image_id}: missing caption or embedding")
            continue
        
        ids.append(image_id)
        documents.append(img['caption_for_embedding'])
        embeddings.append(img['embedding'])
        
        # Metadata (CRITICAL: include image_id for retrieval by ID)
        metadata = {
            'image_id': image_id,  # REQUIRED for fetch_images_by_ids()
            'doc_id': img['doc_id'],
            'filename': img['filename'],
            'width': img.get('width', 0),
            'height': img.get('height', 0),
            'format': img.get('format', ''),
            'enriched_caption': img['enriched_caption'][:MAX_ENRICHED_CAPTION_LENGTH],  # Truncate using constant
            'author_caption': img.get('author_caption', '')[:MAX_AUTHOR_CAPTION_LENGTH] if img.get('author_caption') else '',
            'extraction_method': img.get('extraction_method', '')
        }
        
        # Add page_num (handle both PDF and JSON documents)
        page_num = img.get('page_num', img.get('image_index', 0))
        if page_num is not None:
            metadata['page_num'] = page_num
        
        metadatas.append(metadata)
    
    return ids, documents, embeddings, metadatas, skipped_count


def _get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    description: str
):
    """
    Get existing or create new ChromaDB collection (DRY helper).
    
    Args:
        client: ChromaDB client
        collection_name: Name of collection
        description: Collection description
        
    Returns:
        ChromaDB collection object
    """
    try:
        collection = client.get_collection(collection_name)
        logging.debug(f"Using existing {collection_name} collection")
        return collection
    except ValueError:
        # Collection doesn't exist - create new one
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "description": description,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dims": EMBEDDING_DIMS
            }
        )
        logging.info(f"Created new {collection_name} collection")
        return collection


def index_documents_to_chromadb(
    doc_id: str,
    chunks_with_embeddings: List[Dict],
    images_with_embeddings: List[Dict]
) -> Dict:
    """
    Index text chunks and image captions to ChromaDB with locking.
    
    Features:
    - File-based locking prevents concurrent write race conditions
    - Incremental indexing (skips duplicates)
    - Native ChromaDB API (no langchain wrapper)
    - Single Responsibility: orchestrate indexing with locking
    
    Args:
        doc_id: Document identifier (for logging)
        chunks_with_embeddings: List of text chunks with embeddings
        images_with_embeddings: List of image captions with embeddings
        
    Returns:
        Dict with indexing statistics:
        - text_chunks_added: Number of new chunks indexed
        - text_chunks_skipped: Number of duplicate chunks skipped
        - images_added: Number of new images indexed
        - images_skipped: Number of duplicate images skipped
        
    Raises:
        ValueError: If doc_id is invalid or data structure is incorrect
        TypeError: If inputs are not of expected types
    """
    logging.info(f"📊 Indexing {doc_id} to ChromaDB")
    
    # STAGE 1: Critical validation
    if not doc_id or not isinstance(doc_id, str):
        raise ValueError("doc_id must be a non-empty string")
    
    if not isinstance(chunks_with_embeddings, list):
        raise TypeError("chunks_with_embeddings must be a list")
    
    if not isinstance(images_with_embeddings, list):
        raise TypeError("images_with_embeddings must be a list")
    
    # Validate chunk structure
    for idx, chunk in enumerate(chunks_with_embeddings):
        if not isinstance(chunk, dict):
            raise TypeError(f"Chunk {idx} must be a dict, got {type(chunk)}")
        required_keys = ['chunk_id', 'text', 'embedding', 'doc_id']
        missing = [k for k in required_keys if k not in chunk]
        if missing:
            raise ValueError(f"Chunk {idx} missing required keys: {missing}")
    
    # Validate image structure
    for idx, img in enumerate(images_with_embeddings):
        if not isinstance(img, dict):
            raise TypeError(f"Image {idx} must be a dict, got {type(img)}")
        required_keys = ['image_id', 'embedding', 'doc_id']
        missing = [k for k in required_keys if k not in img]
        if missing:
            raise ValueError(f"Image {idx} missing required keys: {missing}")
    
    # Save backup (STAGE 2: Error handling added)
    _save_chunks_backup(doc_id, chunks_with_embeddings, images_with_embeddings)
    
    # Ensure ChromaDB directory exists
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {
        "text_chunks_added": 0,
        "text_chunks_skipped": 0,
        "images_added": 0,
        "images_skipped": 0
    }
    
    # Use file-based locking to prevent concurrent writes
    with ChromaDBLock(CHROMA_LOCK_FILE):
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )
        
        logging.info(f"Indexing {len(chunks_with_embeddings)} text chunks...")
        
        # Get or create text_chunks collection (DRY: using helper)
        text_collection = _get_or_create_collection(
            client, "text_chunks", "Text chunks from documents"
        )
        
        # Get existing IDs to avoid duplicates
        existing_text_ids = _get_existing_ids(text_collection)
        
        # Prepare new chunks
        chunk_ids, chunk_docs, chunk_embeds, chunk_metas, text_skipped = \
            _prepare_text_chunks_for_indexing(chunks_with_embeddings, existing_text_ids)
        
        # Add new chunks to collection
        if chunk_ids:
            text_collection.add(
                ids=chunk_ids,
                documents=chunk_docs,
                embeddings=chunk_embeds,
                metadatas=chunk_metas
            )
            stats["text_chunks_added"] = len(chunk_ids)
            logging.info(f"✅ Added {len(chunk_ids)} new text chunks")
        
        stats["text_chunks_skipped"] = text_skipped
        if text_skipped > 0:
            logging.info(f"⏭️  Skipped {text_skipped} existing text chunks")
        
        # ====================================================================
        # Index image captions
        # ====================================================================
        logging.info(f"Indexing {len(images_with_embeddings)} image captions...")
        
        # Get or create image_captions collection (DRY: using helper)
        image_collection = _get_or_create_collection(
            client, "image_captions", "Enriched image captions from documents"
        )
        
        # Get existing IDs to avoid duplicates
        existing_image_ids = _get_existing_ids(image_collection)
        
        # Prepare new images
        image_ids, image_docs, image_embeds, image_metas, images_skipped = \
            _prepare_images_for_indexing(images_with_embeddings, existing_image_ids)
        
        # Add new images to collection
        if image_ids:
            image_collection.add(
                ids=image_ids,
                documents=image_docs,
                embeddings=image_embeds,
                metadatas=image_metas
            )
            stats["images_added"] = len(image_ids)
            logging.info(f"✅ Added {len(image_ids)} new image captions")
        
        stats["images_skipped"] = images_skipped
        if images_skipped > 0:
            logging.info(f"⏭️  Skipped {images_skipped} existing images")
    
    logging.info(
        f"✅ Indexing complete: "
        f"{stats['text_chunks_added']} chunks + {stats['images_added']} images added"
    )
    
    return stats
