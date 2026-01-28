"""
Image-specific operations for retrieval.

Handles image fetching by IDs and strict metadata-based retrieval.
"""

import logging
from typing import List, Tuple
from langchain_core.documents import Document

from config import RETRIEVAL
from .utils import parse_json_list


class ImageRetriever:
    """Handle image retrieval and ID-based fetching."""
    
    def __init__(self, image_store, text_store):
        """
        Initialize image retriever.
        
        Args:
            image_store: ChromaDB image collection
            text_store: ChromaDB text collection (for strict retrieval)
        """
        self.image_store = image_store
        self.text_store = text_store
        
        # Config constants
        self.default_k_text = RETRIEVAL.DEFAULT_K_TEXT
        self.default_k_images = RETRIEVAL.DEFAULT_K_IMAGES
        self.mmr_lambda = RETRIEVAL.MMR_LAMBDA
    
    def fetch_images_by_ids(self, image_ids: List[str]) -> List[Document]:
        """
        Fetch images by exact image_id match (no semantic search).
        
        Args:
            image_ids: List of image IDs to fetch
        
        Returns:
            List of image Documents
        """
        if not image_ids:
            return []
        
        images = []
        for img_id in image_ids:
            try:
                results = self.image_store.get(
                    where={"image_id": img_id}
                )
                if results and 'documents' in results:
                    for i, doc_text in enumerate(results['documents']):
                        metadata = results['metadatas'][i] if 'metadatas' in results else {}
                        images.append(Document(page_content=doc_text, metadata=metadata))
            except Exception as e:
                logging.warning(f"Failed to fetch image {img_id}: {type(e).__name__} - {e}")
                continue
        
        logging.info(f"Fetched {len(images)} images by ID")
        return images
    
    def retrieve_images(
        self,
        query: str,
        k: int = None,
        filter_dict: dict = None
    ) -> List[Document]:
        """
        Retrieve relevant image captions using similarity search.
        
        Note: Uses similarity search (not MMR) because:
        - Images are naturally diverse (different diagrams/figures)
        - Small number of images per document (2-14)
        - Maximum relevance is more important than diversity for visual content
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters
        
        Returns:
            List of Document objects with captions and metadata
        """
        if k is None:
            k = self.default_k_images
        
        try:
            results = self.image_store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            logging.info(f"Retrieved {len(results)} images")
        except Exception as e:
            logging.error(f"Image search failed: {type(e).__name__} - {e}")
            results = []
        
        return results
    
    def retrieve_with_strict_images(
        self,
        query: str,
        k_text: int = None,
        mmr_lambda: float = None
    ) -> Tuple[List[Document], List[Document]]:
        """
        Retrieve text chunks and ONLY images explicitly referenced in those chunks.
        
        Strict metadata-driven image retrieval:
        - NO semantic search for images
        - Images ONLY from related_image_ids (same page) and nearby_image_ids (±1 page)
        - nearby_image_ids included ONLY if chunk has figure references
        
        Args:
            query: Search query
            k_text: Number of text chunks
            mmr_lambda: MMR lambda parameter
        
        Returns:
            (text_chunks, images) where images are strictly metadata-linked
        """
        if k_text is None:
            k_text = self.default_k_text
        if mmr_lambda is None:
            mmr_lambda = self.mmr_lambda
        
        logging.info(f"Strict multimodal retrieval for query: '{query}'")
        
        # Retrieve text chunks using MMR
        try:
            text_chunks = self.text_store.max_marginal_relevance_search(
                query=query,
                k=k_text,
                fetch_k=k_text * RETRIEVAL.MMR_FETCH_MULTIPLIER,
                lambda_mult=mmr_lambda
            )
            logging.info(f"Retrieved {len(text_chunks)} text chunks (MMR, λ={mmr_lambda})")
        except Exception as e:
            logging.error(f"Text retrieval failed: {type(e).__name__} - {e}")
            text_chunks = []
        
        # Collect image IDs from metadata with priority tracking
        image_ids_by_priority = {}  # chunk_rank -> list of image_ids
        
        for chunk_rank, chunk in enumerate(text_chunks):
            # Always add same-page images
            related_raw = chunk.metadata.get('related_image_ids', '')
            related_list = parse_json_list(related_raw, 'related_image_ids')
            
            if related_list:
                if chunk_rank not in image_ids_by_priority:
                    image_ids_by_priority[chunk_rank] = []
                image_ids_by_priority[chunk_rank].extend(related_list)
            
            # Add nearby images ONLY if chunk mentions figures
            if chunk.metadata.get('has_figure_references', False):
                nearby_raw = chunk.metadata.get('nearby_image_ids', '')
                nearby_list = parse_json_list(nearby_raw, 'nearby_image_ids')
                if nearby_list:
                    if chunk_rank not in image_ids_by_priority:
                        image_ids_by_priority[chunk_rank] = []
                    image_ids_by_priority[chunk_rank].extend(nearby_list)
        
        # Build ordered list: prioritize by chunk rank (MMR order)
        # This ensures most relevant chunks' images come first
        all_image_ids = []
        for chunk_rank in sorted(image_ids_by_priority.keys()):
            all_image_ids.extend(image_ids_by_priority[chunk_rank])
        
        # Fetch images by ID (no semantic search!)
        if all_image_ids:
            images = self.fetch_images_by_ids(all_image_ids)
            num_strong = sum(len(ids) for ids in image_ids_by_priority.values())
            logging.info(f"Strict retrieval: {len(images)} images from metadata (prioritized by chunk rank)")
        else:
            images = []
            logging.info("Strict retrieval: No images found in chunk metadata")
        
        return text_chunks, images
    
    def rerank_by_figure_refs(self, results: List[Document]) -> List[Document]:
        """
        Re-rank results to prioritize chunks with explicit figure references.
        
        Args:
            results: List of Document objects
        
        Returns:
            Re-ranked list (chunks with has_figure_references=True first)
        """
        with_refs = [doc for doc in results if doc.metadata.get('has_figure_references', False)]
        without_refs = [doc for doc in results if not doc.metadata.get('has_figure_references', False)]
        
        reranked = with_refs + without_refs
        logging.info(f"Re-ranked: {len(with_refs)} with figure refs, {len(without_refs)} without")
        return reranked
