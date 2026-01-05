"""
Build ChromaDB vector store for text chunks and image captions.

Creates two collections:
1. text_chunks - 104 text chunks with embeddings
2. image_captions - 9 image captions with embeddings

Both collections use OpenAI text-embedding-3-small (1536 dims) embeddings.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Paths
BASE_DIR = Path(__file__).parent.parent
EMBEDDINGS_DIR = BASE_DIR / "data" / "processed" / "embeddings"
CHUNKS_FILE = EMBEDDINGS_DIR / "chunks_with_embeddings.json"
IMAGES_FILE = EMBEDDINGS_DIR / "images_with_embeddings.json"
CHROMA_DIR = BASE_DIR / "data" / "vector_store"


def load_data() -> tuple:
    """Load chunks and images with embeddings."""
    logging.info(f"Loading data from {EMBEDDINGS_DIR}")
    
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    with open(IMAGES_FILE, 'r', encoding='utf-8') as f:
        images = json.load(f)
    
    logging.info(f"Loaded {len(chunks)} chunks and {len(images)} images")
    return chunks, images


def create_chroma_client() -> chromadb.PersistentClient:
    """Initialize ChromaDB persistent client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Initializing ChromaDB at {CHROMA_DIR}")
    
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    return client


def build_text_chunks_collection(client: chromadb.PersistentClient, chunks: List[Dict]):
    """
    Build text_chunks collection with embeddings and metadata.
    
    Metadata includes:
    - chunk_id, doc_id, chunk_index, page_num
    - has_figure_references, image_references
    - related_image_ids (same page), nearby_image_ids (±1 page)
    """
    logging.info("Building text_chunks collection")
    
    # Delete if exists
    try:
        client.delete_collection("text_chunks")
        logging.info("Deleted existing text_chunks collection")
    except:
        pass
    
    # Create collection
    collection = client.create_collection(
        name="text_chunks",
        metadata={
            "description": "Text chunks from academic papers",
            "embedding_model": "text-embedding-3-small",
            "embedding_dims": 1536
        }
    )
    
    # Prepare data
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for chunk in chunks:
        ids.append(chunk['chunk_id'])
        embeddings.append(chunk['embedding'])
        documents.append(chunk['text'])
        
        # Metadata (exclude large fields like 'text' and 'embedding')
        metadata = {
            'doc_id': chunk['doc_id'],
            'chunk_index': chunk['chunk_index'],
            'page_num': chunk['page_num'],
            'char_count': chunk['char_count'],
            'word_count': chunk['word_count'],
            'has_figure_references': chunk['has_figure_references'],
            # Store lists as JSON strings (ChromaDB metadata limitation)
            'image_references': json.dumps(chunk['image_references']),
            'related_image_ids': json.dumps(chunk['related_image_ids']),
            'nearby_image_ids': json.dumps(chunk['nearby_image_ids'])
        }
        metadatas.append(metadata)
    
    # Add to collection
    logging.info(f"Adding {len(ids)} chunks to collection")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    logging.info(f"✅ text_chunks collection created with {collection.count()} documents")
    return collection


def build_image_captions_collection(client: chromadb.PersistentClient, images: List[Dict]):
    """
    Build image_captions collection with embeddings and metadata.
    
    Metadata includes:
    - image_id, doc_id, page_num, filename
    - enriched_caption (full text used for embedding)
    - author_caption, vlm_description
    """
    logging.info("Building image_captions collection")
    
    # Delete if exists
    try:
        client.delete_collection("image_captions")
        logging.info("Deleted existing image_captions collection")
    except:
        pass
    
    # Create collection
    collection = client.create_collection(
        name="image_captions",
        metadata={
            "description": "Enriched image captions from academic papers",
            "embedding_model": "text-embedding-3-small",
            "embedding_dims": 1536
        }
    )
    
    # Prepare data
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for img in images:
        ids.append(img['image_id'])
        embeddings.append(img['embedding'])
        
        # Document is the caption used for embedding
        documents.append(img['caption_for_embedding'])
        
        # Metadata
        metadata = {
            'doc_id': img['doc_id'],
            'page_num': img['page_num'],
            'filename': img['filename'],
            'width': img['width'],
            'height': img['height'],
            'format': img['format'],
            # Store captions separately for reference
            'enriched_caption': img['enriched_caption'][:1000],  # Truncate if too long
            'author_caption': img.get('author_caption', '')[:500] if img.get('author_caption') else ''
        }
        metadatas.append(metadata)
    
    # Add to collection
    logging.info(f"Adding {len(ids)} images to collection")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    logging.info(f"✅ image_captions collection created with {collection.count()} documents")
    return collection


def verify_collections(client: chromadb.PersistentClient):
    """Verify collections and show statistics."""
    logging.info("\nVerifying collections...")
    
    collections = client.list_collections()
    logging.info(f"Total collections: {len(collections)}")
    
    for coll in collections:
        logging.info(f"\nCollection: {coll.name}")
        logging.info(f"  Count: {coll.count()} documents")
        logging.info(f"  Metadata: {coll.metadata}")
        
        # Sample query
        result = coll.peek(limit=1)
        if result['ids']:
            logging.info(f"  Sample ID: {result['ids'][0]}")
            logging.info(f"  Embedding dims: {len(result['embeddings'][0])}")


def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("🗄️  ChromaDB Index Building")
    print("=" * 70)
    print()
    
    # Load data
    chunks, images = load_data()
    
    # Create ChromaDB client
    client = create_chroma_client()
    
    # Build collections
    text_collection = build_text_chunks_collection(client, chunks)
    image_collection = build_image_captions_collection(client, images)
    
    # Verify
    verify_collections(client)
    
    # Final statistics
    print()
    print("=" * 70)
    print("📊 Index Statistics")
    print("=" * 70)
    print(f"Vector store location: {CHROMA_DIR}")
    print(f"Text chunks: {text_collection.count()}")
    print(f"Image captions: {image_collection.count()}")
    print(f"Total documents: {text_collection.count() + image_collection.count()}")
    print(f"Embedding model: text-embedding-3-small (1536 dims)")
    print()
    print("=" * 70)
    print("✅ ChromaDB index built successfully!")
    print("=" * 70)
    print()
    print("Next step: Build retriever (rag/retriever.py)")


if __name__ == "__main__":
    main()
