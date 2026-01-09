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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Paths
BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
CHROMA_LOCK_FILE = CHROMA_DIR / ".chroma.lock"


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
        """Release lock."""
        if self.file_handle:
            if sys.platform == 'win32':
                try:
                    msvcrt.locking(self.file_handle.fileno(), msvcrt.LK_UNLCK, 1)
                except:
                    pass
            else:
                try:
                    fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_UN)
                except:
                    pass
            
            self.file_handle.close()
            
            # Remove lock file
            try:
                self.lock_file.unlink()
            except:
                pass
            
            logging.info("Released ChromaDB lock")


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
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    skipped_count = 0
    
    for chunk in chunks_with_embeddings:
        chunk_id = chunk['chunk_id']
        
        # Skip if already indexed
        if chunk_id in existing_ids:
            skipped_count += 1
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
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    skipped_count = 0
    
    for img in images_with_embeddings:
        image_id = img['image_id']
        
        # Skip if already indexed
        if image_id in existing_ids:
            skipped_count += 1
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
            'enriched_caption': img['enriched_caption'][:1000],  # Truncate
            'author_caption': img.get('author_caption', '')[:500] if img.get('author_caption') else '',
            'extraction_method': img.get('extraction_method', '')
        }
        
        # Add page_num (handle both PDF and JSON documents)
        page_num = img.get('page_num', img.get('image_index', 0))
        if page_num is not None:
            metadata['page_num'] = page_num
        
        metadatas.append(metadata)
    
    return ids, documents, embeddings, metadatas, skipped_count


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
    """
    logging.info(f"📊 Indexing {doc_id} to ChromaDB")
    
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
        
        # ====================================================================
        # Index text chunks
        # ====================================================================
        logging.info(f"Indexing {len(chunks_with_embeddings)} text chunks...")
        
        # Get or create text_chunks collection
        try:
            text_collection = client.get_collection("text_chunks")
        except:
            text_collection = client.create_collection(
                name="text_chunks",
                metadata={
                    "description": "Text chunks from documents",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dims": 1536
                }
            )
            logging.info("Created new text_chunks collection")
        
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
        
        # Get or create image_captions collection
        try:
            image_collection = client.get_collection("image_captions")
        except:
            image_collection = client.create_collection(
                name="image_captions",
                metadata={
                    "description": "Enriched image captions from documents",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dims": 1536
                }
            )
            logging.info("Created new image_captions collection")
        
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
