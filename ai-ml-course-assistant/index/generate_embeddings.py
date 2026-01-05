"""
Generate embeddings for text chunks and image captions using OpenAI text-embedding-3-small.

This script:
1. Loads text chunks and image metadata
2. Generates embeddings for all text chunks (104)
3. Generates embeddings for all enriched image captions (9)
4. Saves embeddings to separate JSON files

Embedding model: OpenAI text-embedding-3-small (1536 dimensions)
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Paths
BASE_DIR = Path(__file__).parent.parent
CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks" / "chunks_metadata.json"
IMAGES_FILE = BASE_DIR / "data" / "processed" / "images_metadata.json"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "embeddings"

# Embedding parameters
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
BATCH_SIZE = 100  # OpenAI allows up to 2048 per request, but we use smaller batches


def load_chunks() -> List[Dict]:
    """Load text chunks metadata."""
    logging.info(f"Loading text chunks from {CHUNKS_FILE}")
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logging.info(f"Loaded {len(chunks)} chunks")
    return chunks


def load_images() -> List[Dict]:
    """Load images metadata."""
    logging.info(f"Loading images metadata from {IMAGES_FILE}")
    with open(IMAGES_FILE, 'r', encoding='utf-8') as f:
        images = json.load(f)
    logging.info(f"Loaded {len(images)} images")
    return images


def generate_embeddings_batch(texts: List[str], client: OpenAI) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts.
    
    Args:
        texts: List of text strings to embed
        client: OpenAI client instance
        
    Returns:
        List of embedding vectors (1536-dimensional)
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float"
    )
    
    # Extract embeddings in the same order as input
    embeddings = [item.embedding for item in response.data]
    return embeddings


def process_chunks(chunks: List[Dict], client: OpenAI) -> List[Dict]:
    """
    Generate embeddings for all text chunks.
    
    Args:
        chunks: List of chunk metadata dicts
        client: OpenAI client
        
    Returns:
        List of chunks with embeddings added
    """
    logging.info(f"Generating embeddings for {len(chunks)} text chunks")
    logging.info(f"Model: {EMBEDDING_MODEL} ({EMBEDDING_DIMS} dims), Batch size: {BATCH_SIZE}")
    
    chunks_with_embeddings = []
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        
        logging.info(f"Batch [{batch_num}/{total_batches}]: Processing {len(batch_chunks)} chunks")
        
        # Extract texts
        texts = [chunk['text'] for chunk in batch_chunks]
        
        # Generate embeddings
        try:
            embeddings = generate_embeddings_batch(texts, client)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(batch_chunks, embeddings):
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding['embedding'] = embedding
                chunk_with_embedding['embedding_model'] = EMBEDDING_MODEL
                chunk_with_embedding['embedding_dims'] = EMBEDDING_DIMS
                chunks_with_embeddings.append(chunk_with_embedding)
            
            logging.info(f"Generated {len(embeddings)} embeddings for batch {batch_num}")
            
        except Exception as e:
            logging.error(f"Error in batch {batch_num}: {e}")
            raise
        
        # Rate limiting: small delay between batches
        if batch_idx + BATCH_SIZE < len(chunks):
            time.sleep(0.5)
    
    logging.info(f"Total text embeddings generated: {len(chunks_with_embeddings)}")
    return chunks_with_embeddings


def process_images(images: List[Dict], client: OpenAI) -> List[Dict]:
    """
    Generate embeddings for all enriched image captions.
    
    Args:
        images: List of image metadata dicts
        client: OpenAI client
        
    Returns:
        List of images with embeddings added
    """
    logging.info(f"Generating embeddings for {len(images)} image captions")
    
    # Extract enriched captions (use enriched_caption field from OpenAI Vision)
    texts = []
    for img in images:
        # Combine enriched caption with author caption for richer context
        caption = img.get('enriched_caption', '')
        author_caption = img.get('author_caption', '')
        
        # Format: "Enriched: ... Author caption: ..."
        if author_caption:
            full_text = f"{caption}\n\nAuthor caption: {author_caption}"
        else:
            full_text = caption
        
        texts.append(full_text)
    
    try:
        embeddings = generate_embeddings_batch(texts, client)
        
        # Add embeddings to images
        images_with_embeddings = []
        for img, embedding, caption_text in zip(images, embeddings, texts):
            img_with_embedding = img.copy()
            img_with_embedding['embedding'] = embedding
            img_with_embedding['embedding_model'] = EMBEDDING_MODEL
            img_with_embedding['embedding_dims'] = EMBEDDING_DIMS
            img_with_embedding['caption_for_embedding'] = caption_text
            images_with_embeddings.append(img_with_embedding)
        
        logging.info(f"Generated {len(embeddings)} image caption embeddings")
        
    except Exception as e:
        logging.error(f"Error generating image embeddings: {e}")
        raise
    
    return images_with_embeddings


def save_embeddings(chunks_with_embeddings: List[Dict], images_with_embeddings: List[Dict]):
    """Save embeddings to JSON files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save chunks
    chunks_output = OUTPUT_DIR / "chunks_with_embeddings.json"
    logging.info(f"Saving chunk embeddings to {chunks_output}")
    with open(chunks_output, 'w', encoding='utf-8') as f:
        json.dump(chunks_with_embeddings, f, indent=2, ensure_ascii=False)
    
    file_size_mb = chunks_output.stat().st_size / (1024 * 1024)
    logging.info(f"Saved {len(chunks_with_embeddings)} chunks ({file_size_mb:.2f} MB)")
    
    # Save images
    images_output = OUTPUT_DIR / "images_with_embeddings.json"
    logging.info(f"Saving image embeddings to {images_output}")
    with open(images_output, 'w', encoding='utf-8') as f:
        json.dump(images_with_embeddings, f, indent=2, ensure_ascii=False)
    
    file_size_mb = images_output.stat().st_size / (1024 * 1024)
    logging.info(f"Saved {len(images_with_embeddings)} images ({file_size_mb:.2f} MB)")


def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("🔢 Embedding Generation Pipeline")
    print("=" * 70)
    print()
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Load data
    chunks = load_chunks()
    images = load_images()
    
    # Generate embeddings
    chunks_with_embeddings = process_chunks(chunks, client)
    images_with_embeddings = process_images(images, client)
    
    # Save results
    save_embeddings(chunks_with_embeddings, images_with_embeddings)
    
    # Final statistics
    print()
    print("=" * 70)
    print("📊 Embedding Statistics")
    print("=" * 70)
    print(f"Text chunks embedded: {len(chunks_with_embeddings)}")
    print(f"Image captions embedded: {len(images_with_embeddings)}")
    print(f"Total embeddings: {len(chunks_with_embeddings) + len(images_with_embeddings)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding dimensions: {EMBEDDING_DIMS}")
    
    # Cost estimation
    total_tokens = sum(c['char_count'] for c in chunks_with_embeddings) / 4
    image_tokens = sum(len(img.get('enriched_caption', '')) for img in images_with_embeddings) / 4
    total_tokens_est = total_tokens + image_tokens
    cost_estimate = (total_tokens_est / 1_000_000) * 0.02  # $0.02 per 1M tokens
    
    print(f"Estimated cost: ${cost_estimate:.4f}")
    print()
    print("=" * 70)
    print("✅ Embedding generation complete!")
    print("=" * 70)
    print()
    print("Next step: python build_index.py")


if __name__ == "__main__":
    main()
