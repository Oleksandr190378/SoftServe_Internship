"""
Semantic verification for image-text associations.

Handles confidence scoring, cosine similarity, and batch embedding.
"""

import logging
import re
from typing import List, Dict, Tuple
import numpy as np
from langchain_core.documents import Document

from config import CONFIDENCE, RETRIEVAL
from .utils import parse_json_list


class SemanticVerifier:
    """Semantic verification for image-text associations."""
    
    def __init__(self, embeddings, image_store):
        """
        Initialize verifier with embeddings function.
        
        Args:
            embeddings: OpenAI embeddings instance
            image_store: ChromaDB image collection for fallback search
        """
        self.embeddings = embeddings
        self.image_store = image_store
        
        # Thresholds from config
        self.threshold = RETRIEVAL.SIMILARITY_THRESHOLD
        self.threshold_nearby = RETRIEVAL.SIMILARITY_THRESHOLD_NEARBY
        
        # Confidence levels
        self.confidence_high = CONFIDENCE.HIGH
        self.confidence_medium = CONFIDENCE.MEDIUM
        self.confidence_low = CONFIDENCE.LOW
        
        # Fallback settings
        self.fallback_images_per_doc = RETRIEVAL.FALLBACK_IMAGES_PER_DOC
        self.fallback_threshold = RETRIEVAL.FALLBACK_SIMILARITY_THRESHOLD
        self.max_images_per_query = RETRIEVAL.MAX_IMAGES_PER_QUERY
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Returns 0.0 if vectors are invalid or would cause division by zero.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not vec1 or not vec2:
            logging.warning("Empty vector provided to cosine_similarity")
            return 0.0
        
        if len(vec1) != len(vec2):
            logging.warning(f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
            return 0.0
        
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # Prevent division by zero
            if norm1 == 0.0 or norm2 == 0.0:
                logging.warning("Zero norm vector in cosine_similarity")
                return 0.0
            
            return np.dot(vec1, vec2) / (norm1 * norm2)
        except Exception as e:
            logging.error(f"Error computing cosine similarity: {e}")
            return 0.0
    
    def verify_semantic_match(
        self,
        image: Document,
        text_chunks: List[Document],
        threshold: float = None,
        chunk_embeddings: Dict[str, List[float]] = None,
        image_embedding: List[float] = None
    ) -> Tuple[bool, float, str]:
        """
        Verify if image is semantically related to any text chunk.
        
        Args:
            image: Image document with embedding
            text_chunks: List of text chunk documents
            threshold: Minimum similarity score (uses default if None)
            chunk_embeddings: Pre-computed chunk embeddings (optional)
            image_embedding: Pre-computed image embedding (optional, saves 1 API call)
        
        Returns:
            (is_match, best_similarity, matched_chunk_id)
        """
        if threshold is None:
            threshold = self.threshold
        
        # Get image embedding (use cached if provided)
        if image_embedding is not None:
            img_embedding = image_embedding
        else:
            try:
                img_embedding = self.embeddings.embed_query(image.page_content)
            except Exception as e:
                logging.error(f"Failed to embed image caption: {type(e).__name__} - {e}")
                return False, 0.0, "error"
        
        best_similarity = 0.0
        best_chunk_id = ""
        
        for chunk in text_chunks:
            chunk_id = chunk.metadata.get('chunk_id', 'unknown')
            
            # Use cached embedding if available
            if chunk_embeddings and chunk_id in chunk_embeddings:
                chunk_embedding = chunk_embeddings[chunk_id]
            else:
                try:
                    chunk_embedding = self.embeddings.embed_query(chunk.page_content)
                    if chunk_embeddings is not None:
                        chunk_embeddings[chunk_id] = chunk_embedding
                except Exception as e:
                    logging.warning(f"Failed to embed chunk {chunk_id}: {type(e).__name__} - {e}")
                    continue
            
            similarity = self.cosine_similarity(img_embedding, chunk_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_chunk_id = chunk_id
        
        is_match = best_similarity >= threshold
        return is_match, best_similarity, best_chunk_id
    
    def batch_embed_chunks(self, chunks: List[Document]) -> Dict[str, List[float]]:
        """
        Batch embed all text chunks and return embedding mapping.
        
        Args:
            chunks: List of text chunk documents
        
        Returns:
            Dict mapping chunk_id to embedding vector
        """
        chunk_embeddings = {}
        
        if not chunks:
            logging.warning("No chunks provided for batch embedding")
            return chunk_embeddings
        
        chunk_texts = [chunk.page_content for chunk in chunks if chunk.page_content]
        
        if not chunk_texts:
            logging.warning("All chunks have empty content")
            return chunk_embeddings
        
        try:
            # Batch embed all chunks at once (1 API call instead of N)
            embeddings_list = self.embeddings.embed_documents(chunk_texts)
            
            if not embeddings_list or len(embeddings_list) != len(chunk_texts):
                logging.error(f"Embedding mismatch: got {len(embeddings_list)} embeddings for {len(chunk_texts)} chunks")
                return chunk_embeddings
            
            for i, chunk in enumerate(chunks):
                if chunk.page_content:  # Only add chunks with content
                    chunk_id = chunk.metadata.get('chunk_id', f'chunk_{i}')
                    if i < len(embeddings_list):
                        chunk_embeddings[chunk_id] = embeddings_list[i]
        except Exception as e:
            logging.error(f"Error batch embedding chunks: {type(e).__name__} - {e}")
        
        return chunk_embeddings
    
    def batch_embed_images(self, images: List[Document]) -> Dict[str, List[float]]:
        """
        Batch embed all image captions and return embedding mapping.
        
        Args:
            images: List of image documents
        
        Returns:
            Dict mapping image_id to embedding vector
        """
        image_embeddings = {}
        
        if not images:
            return image_embeddings
        
        image_texts = [img.page_content for img in images if img.page_content]
        
        if not image_texts:
            logging.warning("All images have empty captions")
            return image_embeddings
        
        try:
            embeddings_list = self.embeddings.embed_documents(image_texts)
            
            if not embeddings_list or len(embeddings_list) != len(image_texts):
                logging.error(f"Image embedding mismatch: got {len(embeddings_list)} for {len(image_texts)} images")
                return image_embeddings
            
            for i, img in enumerate(images):
                if img.page_content:  # Only add images with content
                    img_id = img.metadata.get('image_id', f'img_{i}')
                    if i < len(embeddings_list):
                        image_embeddings[img_id] = embeddings_list[i]
            
            logging.info(f"Batch embedded {len(image_embeddings)} images in 1 API call")
        except Exception as e:
            logging.error(f"Error batch embedding images: {type(e).__name__} - {e}")
        
        return image_embeddings
    
    def verify_metadata_images(
        self,
        metadata_images: List[Document],
        text_chunks: List[Document],
        chunk_embeddings: Dict[str, List[float]],
        image_embeddings: Dict[str, List[float]]
    ) -> List[Dict]:
        """
        Verify metadata images with confidence scoring.
        
        Args:
            metadata_images: Candidate images from metadata linking
            text_chunks: Retrieved text chunks for context
            chunk_embeddings: Pre-computed chunk embeddings
            image_embeddings: Pre-computed image embeddings
        
        Returns:
            List of verified images with confidence scores and reasons
        """
        verified_images = []
        seen_image_ids = set()  # For deduplication by image_id
        seen_figure_nums = {}  # For deduplication by Figure number: {fig_num: (image_obj, verified_index)}
        
        for img in metadata_images:
            img_id = img.metadata.get('image_id', '')
            
            # Skip duplicates by image_id
            if img_id in seen_image_ids:
                logging.info(f"  {img_id}: Skipped (duplicate image_id)")
                continue
            
            # Extract Figure number from caption (if available)
            caption = img.page_content or ""
            fig_match = re.search(r'Figure\s+(\d+)', caption, re.IGNORECASE)
            fig_num = fig_match.group(1) if fig_match else None
            
            # Check if chunk explicitly mentions figure FIRST (before dedup check)
            has_explicit_ref = False
            for chunk in text_chunks:
                if chunk.metadata.get('has_figure_references', False):
                    related_ids_raw = chunk.metadata.get('related_image_ids', '')
                    related_ids = parse_json_list(related_ids_raw, 'related_image_ids')
                    
                    # Check if image_id is in the list
                    if img_id in related_ids:
                        has_explicit_ref = True
                        break
            
            # Handle duplicates by Figure number
            if fig_num and fig_num in seen_figure_nums:
                prev_img, prev_index = seen_figure_nums[fig_num]
                prev_img_id = prev_img.metadata.get('image_id', 'unknown')
                
                if has_explicit_ref:
                    # Current image is HIGH confidence - REPLACE previous
                    logging.info(f"  {img_id}: HIGH confidence (explicit ref) - REPLACING {prev_img_id}")
                    verified_images.pop(prev_index)
                    # Update indices in seen_figure_nums for remaining items
                    for key in seen_figure_nums:
                        stored_img, stored_idx = seen_figure_nums[key]
                        if stored_idx > prev_index:
                            seen_figure_nums[key] = (stored_img, stored_idx - 1)
                else:
                    # Current image is not HIGH confidence - SKIP (keep previous)
                    logging.info(f"  {img_id}: Skipped (duplicate Figure {fig_num}, keeping {prev_img_id})")
                    continue
            
            # Track seen items
            seen_image_ids.add(img_id)
            
            if has_explicit_ref:
                # HIGH confidence: explicit figure reference
                verified_images.append({
                    'image': img,
                    'confidence': self.confidence_high,
                    'similarity': 1.0,
                    'reason': 'Explicit figure reference in text'
                })
                if fig_num:
                    seen_figure_nums[fig_num] = (img, len(verified_images) - 1)
                logging.info(f"  {img_id}: {self.confidence_high} confidence (explicit ref)")
            else:
                # Semantic verification with cached embeddings
                img_embedding = image_embeddings.get(img_id)  # Use pre-computed
                
                is_match, similarity, chunk_id = self.verify_semantic_match(
                    img, text_chunks,
                    chunk_embeddings=chunk_embeddings,
                    image_embedding=img_embedding
                )
                
                if is_match:
                    verified_images.append({
                        'image': img,
                        'confidence': self.confidence_medium,
                        'similarity': similarity,
                        'reason': f'Semantic match with {chunk_id} (sim={similarity:.2f})'
                    })
                    if fig_num:
                        seen_figure_nums[fig_num] = (img, len(verified_images) - 1)
                    logging.info(f"  {img_id}: {self.confidence_medium} confidence (sim={similarity:.2f})")
                else:
                    logging.info(f"  {img_id}: Rejected (sim={similarity:.2f} < threshold)")
            
            # Hard cap to prevent excessive images
            if len(verified_images) >= self.max_images_per_query:
                logging.info(f"  Reached MAX_IMAGES_PER_QUERY ({self.max_images_per_query}), stopping verification")
                break
        
        # Log summary of verified images
        high_confidence = sum(1 for img in verified_images if img['confidence'] == self.confidence_high)
        medium_confidence = sum(1 for img in verified_images if img['confidence'] == self.confidence_medium)
        low_confidence = sum(1 for img in verified_images if img['confidence'] == self.confidence_low)
        logging.info(f"Verification summary: {len(verified_images)} total ({high_confidence} HIGH, {medium_confidence} MEDIUM, {low_confidence} LOW)")
        
        return verified_images
    
    def fallback_visual_search(
        self,
        query: str,
        text_chunks: List[Document],
        chunk_embeddings: Dict[str, List[float]],
        seen_image_ids: set
    ) -> List[Dict]:
        """
        Fallback semantic caption search for visual queries.
        
        Args:
            query: User query
            text_chunks: Retrieved text chunks
            chunk_embeddings: Pre-computed chunk embeddings
            seen_image_ids: Already processed image IDs (for deduplication)
        
        Returns:
            List of fallback images with LOW confidence
        """
        fallback_images = []
        
        logging.info("  No metadata images, fallback to semantic caption search")
        
        # Get unique document IDs from text chunks to filter fallback search
        chunk_doc_ids = set(chunk.metadata.get('doc_id', '') for chunk in text_chunks)
        logging.info(f"  Filtering fallback to documents: {chunk_doc_ids}")
        
        # Retrieve images with document filter
        semantic_images = []
        for doc_id in chunk_doc_ids:
            doc_images = self.image_store.similarity_search(
                query=query,
                k=self.fallback_images_per_doc,
                filter={'doc_id': doc_id}
            )
            semantic_images.extend(doc_images)
        
        for img in semantic_images:
            img_id = img.metadata.get('image_id', '')
            
            # Skip if already processed
            if img_id in seen_image_ids:
                continue
            seen_image_ids.add(img_id)
            
            is_match, similarity, chunk_id = self.verify_semantic_match(
                img, text_chunks,
                threshold=self.fallback_threshold,
                chunk_embeddings=chunk_embeddings
            )
            
            fallback_images.append({
                'image': img,
                'confidence': self.confidence_low,
                'similarity': similarity,
                'reason': f'Visual query fallback (sim={similarity:.2f})'
            })
            logging.info(f"  {img_id}: {self.confidence_low} confidence (fallback)")
        
        return fallback_images
