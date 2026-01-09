"""
Embedding utilities for document chunks and image captions.

This module provides functions to generate embeddings for:
- Text chunks from documents
- Image captions (enriched by Vision LLM)

Used by run_pipeline.py Stage 4.
"""

import logging
import time
from typing import List, Dict, Tuple
from openai import OpenAI
import openai


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
BATCH_SIZE = 100  


def generate_embeddings_batch(texts: List[str], client: OpenAI) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts using OpenAI.
    
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


def embed_chunks(
    chunks: List[Dict],
    client: OpenAI,
    batch_size: int = BATCH_SIZE
) -> Tuple[List[Dict], float]:
    """
    Generate embeddings for text chunks.
    
    Args:
        chunks: List of chunk dictionaries with 'text' field
        client: OpenAI client
        batch_size: Number of chunks per batch
        
    Returns:
        Tuple of (chunks_with_embeddings, cost_usd)
    """
    logging.info(f"Generating embeddings for {len(chunks)} chunks")
    logging.info(f"Model: {EMBEDDING_MODEL} ({EMBEDDING_DIMS} dims)")
    
    chunks_with_embeddings = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    total_tokens = 0
    
    for batch_idx in range(0, len(chunks), batch_size):
        batch_chunks = chunks[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        
        logging.debug(f"Batch [{batch_num}/{total_batches}]: {len(batch_chunks)} chunks")
        
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
            
            # Track tokens (approximate: 1 token ≈ 4 chars)
            batch_tokens = sum(len(text) for text in texts) / 4
            total_tokens += batch_tokens
            
        except openai.AuthenticationError:
            logging.error(f"Batch {batch_num}: OpenAI authentication failed. Check API key.")
            raise
        except openai.RateLimitError:
            logging.error(f"Batch {batch_num}: Rate limit exceeded. Consider reducing batch size or adding delays.")
            raise
        except openai.APIError as e:
            logging.error(f"Batch {batch_num}: OpenAI API error: {type(e).__name__}")
            raise
        except Exception as e:
            logging.error(f"Batch {batch_num}: Unexpected error: {type(e).__name__}")
            raise
        
        # Rate limiting
        if batch_idx + batch_size < len(chunks):
            time.sleep(0.3)
    
    # Calculate cost: text-embedding-3-small is $0.02 per 1M tokens
    cost_usd = (total_tokens / 1_000_000) * 0.02
    
    logging.info(f"✅ Generated {len(chunks_with_embeddings)} chunk embeddings")
    logging.info(f"   Tokens: ~{int(total_tokens):,}, Cost: ${cost_usd:.4f}")
    
    return chunks_with_embeddings, cost_usd


def embed_images(
    images: List[Dict],
    client: OpenAI
) -> Tuple[List[Dict], float]:
    """
    Generate embeddings for image captions.
    
    Args:
        images: List of image metadata dicts with 'enriched_caption'
        client: OpenAI client
        
    Returns:
        Tuple of (images_with_embeddings, cost_usd)
    """
    if not images:
        logging.info("No images to embed")
        return [], 0.0
    
    logging.info(f"Generating embeddings for {len(images)} image captions")
    
    # Build caption texts (combine enriched + author caption)
    texts = []
    for img in images:
        caption = img.get('enriched_caption', '')
        author_caption = img.get('author_caption', '')
        
        # Format: "Caption. Author: ..."
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
        
        # Calculate cost
        total_tokens = sum(len(text) for text in texts) / 4
        cost_usd = (total_tokens / 1_000_000) * 0.02
        
        logging.info(f"✅ Generated {len(images_with_embeddings)} image embeddings")
        logging.info(f"   Tokens: ~{int(total_tokens):,}, Cost: ${cost_usd:.4f}")
        
        return images_with_embeddings, cost_usd
        
    except openai.AuthenticationError:
        logging.error("Image embeddings: OpenAI authentication failed. Check API key.")
        raise
    except openai.RateLimitError:
        logging.error("Image embeddings: Rate limit exceeded. Consider adding delays between requests.")
        raise
    except openai.APIError as e:
        logging.error(f"Image embeddings: OpenAI API error: {type(e).__name__}")
        raise
    except Exception as e:
        logging.error(f"Image embeddings: Unexpected error: {type(e).__name__}")
        raise


def embed_document(
    doc_id: str,
    chunks: List[Dict],
    images: List[Dict],
    client: OpenAI
) -> Dict:
    """
    Generate embeddings for a document's chunks and images.
    
    Args:
        doc_id: Document identifier
        chunks: List of text chunks
        images: List of image metadata (with enriched captions)
        client: OpenAI client
        
    Returns:
        Dict with:
            - chunks_with_embeddings: List of chunks with embeddings
            - images_with_embeddings: List of images with embeddings
            - cost: Total cost in USD
    """
    logging.info(f"📊 Embedding document: {doc_id}")
    
    # Embed chunks
    chunks_with_embeddings, chunks_cost = embed_chunks(chunks, client)
    
    # Embed images (only those with enriched captions)
    images_to_embed = [img for img in images if img.get('enriched_caption')]
    images_with_embeddings, images_cost = embed_images(images_to_embed, client)
    
    total_cost = chunks_cost + images_cost
    
    return {
        "chunks_with_embeddings": chunks_with_embeddings,
        "images_with_embeddings": images_with_embeddings,
        "cost": total_cost,
        "stats": {
            "chunks_embedded": len(chunks_with_embeddings),
            "images_embedded": len(images_with_embeddings)
        }
    }
