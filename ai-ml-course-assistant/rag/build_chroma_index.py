"""
Build ChromaDB vector store with text chunks and image captions.

Creates two separate collections:


Uses OpenAI text-embedding-3-small (1536 dimensions) for both.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

BASE_DIR = Path(__file__).parent.parent
CHUNKS_FILE = BASE_DIR / "data" / "processed" / "embeddings" / "chunks_with_embeddings.json"
IMAGES_FILE = BASE_DIR / "data" / "processed" / "embeddings" / "images_with_embeddings.json"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536


def load_chunks_with_embeddings() -> List[Dict]:
    """Load text chunks with embeddings."""
    logging.info(f"Loading chunks from {CHUNKS_FILE}")
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logging.info(f"Loaded {len(chunks)} chunks")
    return chunks


def load_images_with_embeddings() -> List[Dict]:
    """Load image captions with embeddings."""
    logging.info(f"Loading images from {IMAGES_FILE}")
    with open(IMAGES_FILE, 'r', encoding='utf-8') as f:
        images = json.load(f)
    logging.info(f"Loaded {len(images)} images")
    return images


def create_text_chunks_collection(chunks: List[Dict]) -> Chroma:
    """
    Create ChromaDB collection for text chunks.
    
    Args:
        chunks: List of chunks with embeddings
        
    Returns:
        Chroma vector store instance
    """
    logging.info("Creating text_chunks collection in ChromaDB")
    
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMS
    )
    
    texts = []
    metadatas = []
    ids = []
    embeddings_list = []
    
    for chunk in chunks:
        texts.append(chunk['text'])
        
        # Metadata for filtering and retrieval
        metadata = {
            'chunk_id': chunk['chunk_id'],
            'doc_id': chunk['doc_id'],
            'chunk_index': chunk['chunk_index'],
            'page_num': chunk['page_num'],
            'char_count': chunk['char_count'],
            'word_count': chunk['word_count'],
            'has_figure_references': chunk['has_figure_references'],
            'image_references': ','.join(chunk['image_references']),  # Store as comma-separated string
            'related_image_ids': ','.join(chunk['related_image_ids']),
            'nearby_image_ids': ','.join(chunk['nearby_image_ids'])
        }
        metadatas.append(metadata)
        ids.append(chunk['chunk_id'])
        embeddings_list.append(chunk['embedding'])
    
    # Create vector store
    vector_store = Chroma(
        collection_name="text_chunks",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR / "text_chunks")
    )
    
    # Add documents with pre-computed embeddings
    logging.info(f"Adding {len(texts)} text chunks to ChromaDB")
    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings_list
    )
    
    logging.info(f"Text chunks collection created: {len(texts)} documents")
    return vector_store


def create_image_captions_collection(images: List[Dict]) -> Chroma:
    """
    Create ChromaDB collection for image captions.
    
    Args:
        images: List of images with embeddings
        
    Returns:
        Chroma vector store instance
    """
    logging.info("Creating image_captions collection in ChromaDB")
    
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMS
    )
    
    # Prepare documents and metadata
    texts = []
    metadatas = []
    ids = []
    embeddings_list = []
    
    for img in images:
        # Use caption_for_embedding (enriched + author caption)
        texts.append(img['caption_for_embedding'])
        
        # Metadata for filtering and retrieval
        metadata = {
            'image_id': img['image_id'],
            'doc_id': img['doc_id'],
            'filename': img['filename'],
            'page_num': img['page_num'],
            'region_index': img.get('region_index', 0),
            'width': img.get('width', 0),
            'height': img.get('height', 0),
            'format': img.get('format', ''),
            'author_caption': img.get('author_caption', ''),
            'extraction_method': img.get('extraction_method', '')
        }
        metadatas.append(metadata)
        ids.append(img['image_id'])
        embeddings_list.append(img['embedding'])
    
    # Create vector store
    vector_store = Chroma(
        collection_name="image_captions",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR / "image_captions")
    )
    
    # Add documents with pre-computed embeddings
    logging.info(f"Adding {len(texts)} image captions to ChromaDB")
    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings_list
    )
    
    logging.info(f"Image captions collection created: {len(texts)} documents")
    return vector_store


def verify_collections(text_store: Chroma, image_store: Chroma):
    """Verify collections are created correctly."""
    logging.info("\nVerifying ChromaDB collections...")
    
    # Test text chunks collection
    text_results = text_store.similarity_search(
        "convolutional neural network architecture",
        k=3
    )
    logging.info(f"Text chunks test query returned {len(text_results)} results")
    if text_results:
        logging.info(f"  Sample result: {text_results[0].metadata['chunk_id']}")
    
    # Test image captions collection
    image_results = image_store.similarity_search(
        "neural network diagram",
        k=3
    )
    logging.info(f"Image captions test query returned {len(image_results)} results")
    if image_results:
        logging.info(f"  Sample result: {image_results[0].metadata['image_id']}")


def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("🗄️  ChromaDB Index Building Pipeline")
    print("=" * 70)
    print()
    
    # Create chroma directory
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    chunks = load_chunks_with_embeddings()
    images = load_images_with_embeddings()
    
    # Create collections
    text_store = create_text_chunks_collection(chunks)
    image_store = create_image_captions_collection(images)
    
    # Verify
    verify_collections(text_store, image_store)
    
    # Final statistics
    print()
    print("=" * 70)
    print("📊 ChromaDB Statistics")
    print("=" * 70)
    print(f"Text chunks collection: {len(chunks)} documents")
    print(f"Image captions collection: {len(images)} documents")
    print(f"Total documents: {len(chunks) + len(images)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding dimensions: {EMBEDDING_DIMS}")
    print(f"Persist directory: {CHROMA_DIR}")
    print()
    print("=" * 70)
    print("✅ ChromaDB index built successfully!")
    print("=" * 70)
    print()
    print("Next step: Create retriever in rag/retriever.py")


if __name__ == "__main__":
    main()
