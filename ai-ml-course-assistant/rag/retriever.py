"""
Multimodal Retriever for text chunks and image captions.

Retrieves relevant text chunks and images from ChromaDB based on query.
Uses anti-hallucination metadata to prioritize accurate image-text associations.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple
import os
import numpy as np
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536

VISUAL_KEYWORDS = [
    "show", "diagram", "architecture", "figure", "image", 
    "visualization", "chart", "graph", "draw", "display",
    "illustrate", "picture", "schema"
]

# Similarity thresholds for semantic image verification
# Calibrated via pilot evaluation (eval/results/pilot_3docs.md):
#   - Achieved 87.5% image hit rate (4/5 queries) with 80% precision
#   - Same-page images: 0.5 balances recall (finds relevant images) vs precision (rejects irrelevant)
#   - Nearby images (±1 page): 0.65 requires stronger semantic match (less structurally related)
# Can be overridden via environment variables for tuning without code changes
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))  # Same-page images
SIMILARITY_THRESHOLD_NEARBY = float(os.getenv("SIMILARITY_THRESHOLD_NEARBY", "0.65"))  # ±1 page images

DEFAULT_MMR_LAMBDA = 0.7  # Balance between relevance (1.0) and diversity (0.0)
MMR_FETCH_MULTIPLIER = 2  # Fetch k*2 candidates for better diversity

DEFAULT_K_TEXT = 3  # Default number of text chunks to retrieve
DEFAULT_K_IMAGES = 3  # Default number of images to retrieve
DEFAULT_CONTEXT_CHUNKS = 2  # Default chunks for image context
MAX_IMAGES_PER_QUERY = 8  # Hard limit on total images returned (prevents Query #7: 12 images)

FALLBACK_IMAGES_PER_DOC = 1  # Images to retrieve per document in fallback (lowered from 2)
FALLBACK_SIMILARITY_THRESHOLD = 0.5  # Lower threshold for visual query fallback

CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW = "LOW"

DEFAULT_TEXT_COLLECTION = "text_chunks"
DEFAULT_IMAGE_COLLECTION = "image_captions"


class MultimodalRetriever:
    """
    Retriever for both text chunks and image captions.
    Features:
    - Searches both text and image collections
    - Merges and re-ranks results
    - Uses anti-hallucination metadata (has_figure_references, related_image_ids)
    - Supports filtering by document, page, or figure references
    """
    
    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        text_collection: str = DEFAULT_TEXT_COLLECTION,
        image_collection: str = DEFAULT_IMAGE_COLLECTION
    ):
        """
        Initialize retriever with ChromaDB collections.    
        Args:
            chroma_dir: Path to ChromaDB directory
            text_collection: Name of text chunks collection
            image_collection: Name of image captions collection
        """
        logging.info(f"Initializing MultimodalRetriever from {chroma_dir}")

        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMS
        )       
        # Cache for embeddings to avoid re-computation
        self._chunk_embeddings_cache = {}  # chunk_id -> embedding       
        # Load ChromaDB collections with error handling
        # IMPORTANT: persist_directory points to the PARENT directory containing collections,
        # NOT the collection-specific subdirectory. This matches how chromadb.PersistentClient works.
        try:
            self.text_store = Chroma(
                collection_name=text_collection,
                embedding_function=self.embeddings,
                persist_directory=str(chroma_dir)  # Fixed: use parent dir, not subdirectory
            )
            logging.info(f"✅ Loaded text collection: {text_collection}")
        except Exception as e:
            logging.error(f"Failed to load text collection '{text_collection}': {type(e).__name__} - {e}")
            raise RuntimeError(f"Cannot initialize text collection: {e}") from e
        
        try:
            self.image_store = Chroma(
                collection_name=image_collection,
                embedding_function=self.embeddings,
                persist_directory=str(chroma_dir)  # Fixed: use parent dir, not subdirectory
            )
            logging.info(f"✅ Loaded image collection: {image_collection}")
        except Exception as e:
            logging.error(f"Failed to load image collection '{image_collection}': {type(e).__name__} - {e}")
            raise RuntimeError(f"Cannot initialize image collection: {e}") from e
    
    def _parse_json_list(self, value: any, field_name: str = "field") -> List[str]:
        """
        Parse JSON-encoded list or return list/string as-is.
        Args:
            value: Value to parse (could be JSON string, list, or comma-separated string)
            field_name: Field name for logging purposes
        
        Returns:
            List of string values (empty list if parsing fails)
        """
        if not value:
            return []
        
        # Already a list - return as-is
        if isinstance(value, list):
            return value
        
        # Try JSON parsing for encoded lists
        if isinstance(value, str) and value.startswith('['):
            import json
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse {field_name} JSON: {e}")
                # Fallback to comma-separated parsing
                return [id.strip() for id in value.split(',') if id.strip()]
        
        # Comma-separated string
        if isinstance(value, str):
            return [id.strip() for id in value.split(',') if id.strip()]
        
        return []
    
    def _format_related_image_ids(self, chunk_metadata: Dict) -> List[str]:
        """
        Extract and format related image IDs from chunk metadata.               
        Args:
            chunk_metadata: Chunk metadata dictionary
        
        Returns:
            List of image ID strings
        """
        related_ids_raw = chunk_metadata.get('related_image_ids', '')
        return self._parse_json_list(related_ids_raw, 'related_image_ids')
    
    def retrieve_text_chunks(
        self,
        query: str,
        k: int = DEFAULT_K_TEXT,
        filter_dict: Dict = None,
        search_type: str = "mmr",
        mmr_lambda: float = DEFAULT_MMR_LAMBDA
    ) -> List[Document]:
        """
        Retrieve relevant text chunks using similarity or MMR search.        
        MMR (Maximal Marginal Relevance) balances relevance and diversity:
        - lambda=1.0: Maximum relevance (same as similarity search)
        - lambda=0.5-0.7: Balanced (recommended for RAG - diverse context)
        - lambda=0.0: Maximum diversity        
        Default uses MMR to avoid redundant chunks from same section.        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters (e.g., {'doc_id': 'arxiv_1234'})
            search_type: "similarity" or "mmr" (default: "mmr" for diversity)
            mmr_lambda: MMR lambda parameter (0.0-1.0), only used if search_type="mmr"
            
        Returns:
            List of Document objects with text and metadata
        """
        if search_type == "mmr":
            try:
                results = self.text_store.max_marginal_relevance_search(
                    query=query,
                    k=k,
                    fetch_k=k * MMR_FETCH_MULTIPLIER,  # Fetch more candidates for better diversity
                    lambda_mult=mmr_lambda,
                    filter=filter_dict
                )
                logging.info(f"Retrieved {len(results)} text chunks (MMR, λ={mmr_lambda})")
            except Exception as e:
                logging.error(f"MMR search failed: {type(e).__name__} - {e}")
                results = []
        else:
            try:
                results = self.text_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
                logging.info(f"Retrieved {len(results)} text chunks (similarity)")
            except Exception as e:
                logging.error(f"Similarity search failed: {type(e).__name__} - {e}")
                results = []       
        return results
    
    def retrieve_images(
        self,
        query: str,
        k: int = DEFAULT_K_IMAGES,
        filter_dict: Dict = None
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
    
    def retrieve_multimodal(
        self,
        query: str,
        k_text: int = DEFAULT_K_TEXT,
        k_images: int = DEFAULT_K_IMAGES,
        prioritize_figure_refs: bool = True
    ) -> Tuple[List[Document], List[Document]]:
        """
        Retrieve both text chunks and images for a query.       
        Anti-hallucination strategy:
        - If prioritize_figure_refs=True, boosts text chunks with explicit figure references
        - Links retrieved images to their related text chunks via metadata
        
        Args:
            query: Search query
            k_text: Number of text chunks to retrieve
            k_images: Number of images to retrieve
            prioritize_figure_refs: Whether to prioritize chunks with figure references
            
        Returns:
            (text_chunks, images) tuple
        """
        logging.info(f"Multimodal retrieval for query: '{query}'")

        text_results = self.retrieve_text_chunks(query, k=k_text)
        image_results = self.retrieve_images(query, k=k_images)

        if prioritize_figure_refs:
            text_results = self._rerank_by_figure_refs(text_results)
        
        return text_results, image_results
    
    def _rerank_by_figure_refs(self, results: List[Document]) -> List[Document]:
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
    
    def get_related_chunks_for_image(
        self,
        image_doc: Document,
        k: int = DEFAULT_CONTEXT_CHUNKS
    ) -> List[Document]:
        """
        Get text chunks related to a specific image.       
        Uses anti-hallucination metadata:
        - Searches for chunks with matching page_num
        - Prioritizes chunks with related_image_ids containing this image
        
        Args:
            image_doc: Image Document object
            k: Number of chunks to return
            
        Returns:
            List of related text chunks
        """
        image_id = image_doc.metadata['image_id']
        page_num = image_doc.metadata['page_num']
        filter_dict = {'page_num': page_num}
        
        try:
            chunks = self.text_store.similarity_search(
                query=image_doc.page_content,
                k=k,
                filter=filter_dict
            )
        except Exception as e:
            logging.error(f"Failed to get related chunks for image {image_id}: {type(e).__name__} - {e}")
            return []

        chunks_with_link = []
        chunks_without_link = []
        
        for chunk in chunks:
            related_ids_raw = chunk.metadata.get('related_image_ids', '')
            related_ids = self._parse_json_list(related_ids_raw, 'related_image_ids')
            
            # Check if image_id is in the list
            if image_id in related_ids:
                chunks_with_link.append(chunk)
            else:
                chunks_without_link.append(chunk)
        
        result = chunks_with_link + chunks_without_link
        logging.info(f"Found {len(result)} chunks related to image {image_id}")
        return result[:k]
    
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
    
    def retrieve_with_strict_images(
        self,
        query: str,
        k_text: int = DEFAULT_K_TEXT
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
            
        Returns:
            (text_chunks, images) where images are strictly metadata-linked
        """
        logging.info(f"Strict multimodal retrieval for query: '{query}'")
        
        text_chunks = self.retrieve_text_chunks(query, k=k_text)

        image_ids_strong = set()  # related_image_ids (same page - always include)
        image_ids_weak = set()    # nearby_image_ids (±1 page - only if has figure refs)
        
        for chunk in text_chunks:
            # Always add same-page images
            related_raw = chunk.metadata.get('related_image_ids', '')
            related_list = self._parse_json_list(related_raw, 'related_image_ids')
            image_ids_strong.update(related_list)
            
            # Add nearby images ONLY if chunk mentions figures
            if chunk.metadata.get('has_figure_references', False):
                nearby_raw = chunk.metadata.get('nearby_image_ids', '')
                nearby_list = self._parse_json_list(nearby_raw, 'nearby_image_ids')
                image_ids_weak.update(nearby_list)
        
        # 3. Combine image IDs (strong links first)
        all_image_ids = list(image_ids_strong) + list(image_ids_weak)
        
        # 4. Fetch images by ID (no semantic search!)
        if all_image_ids:
            images = self.fetch_images_by_ids(all_image_ids)
            logging.info(f"Strict retrieval: {len(images)} images from metadata ({len(image_ids_strong)} strong, {len(image_ids_weak)} weak links)")
        else:
            images = []
            logging.info("Strict retrieval: No images found in chunk metadata")
        return text_chunks, images
    
    def is_visual_query(self, query: str) -> bool:
        """Check if query requests visual content."""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in VISUAL_KEYWORDS)
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Returns 0.0 if vectors are invalid or would cause division by zero.
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
        threshold: float = SIMILARITY_THRESHOLD,
        chunk_embeddings: Dict[str, List[float]] = None,
        image_embedding: List[float] = None
    ) -> Tuple[bool, float, str]:
        """
        Verify if image is semantically related to any text chunk.     
        Args:
            image: Image document with embedding
            text_chunks: List of text chunk documents
            threshold: Minimum similarity score
            chunk_embeddings: Pre-computed chunk embeddings (optional)
            image_embedding: Pre-computed image embedding (optional, saves 1 API call)
            
        Returns:
            (is_match, best_similarity, matched_chunk_id)
        """
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
    
    def _batch_embed_chunks(self, chunks: List[Document]) -> Dict[str, List[float]]:
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
    
    def _batch_embed_images(self, images: List[Document]) -> Dict[str, List[float]]:
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
    
    def _verify_metadata_images(
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
        seen_figure_nums = {}  # For deduplication by Figure number: {fig_num: image_obj}
        
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
            
            # Skip duplicates by Figure number (prefer first occurrence)
            if fig_num and fig_num in seen_figure_nums:
                prev_img_id = seen_figure_nums[fig_num].metadata.get('image_id', 'unknown')
                logging.info(f"  {img_id}: Skipped (duplicate Figure {fig_num}, keeping {prev_img_id})")
                continue
            
            # Track seen items
            seen_image_ids.add(img_id)
            if fig_num:
                seen_figure_nums[fig_num] = img
            
            # Check if chunk explicitly mentions figure (skip verification)
            has_explicit_ref = False
            for chunk in text_chunks:
                if chunk.metadata.get('has_figure_references', False):
                    related_ids_raw = chunk.metadata.get('related_image_ids', '')
                    related_ids = self._parse_json_list(related_ids_raw, 'related_image_ids')
                    
                    # Check if image_id is in the list
                    if img_id in related_ids:
                        has_explicit_ref = True
                        break
            
            if has_explicit_ref:
                # HIGH confidence: explicit figure reference
                verified_images.append({
                    'image': img,
                    'confidence': CONFIDENCE_HIGH,
                    'similarity': 1.0,
                    'reason': 'Explicit figure reference in text'
                })
                logging.info(f"  {img_id}: {CONFIDENCE_HIGH} confidence (explicit ref)")
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
                        'confidence': CONFIDENCE_MEDIUM,
                        'similarity': similarity,
                        'reason': f'Semantic match with {chunk_id} (sim={similarity:.2f})'
                    })
                    logging.info(f"  {img_id}: {CONFIDENCE_MEDIUM} confidence (sim={similarity:.2f})")
                else:
                    logging.info(f"  {img_id}: Rejected (sim={similarity:.2f} < threshold)")
            
            # Hard cap to prevent excessive images (Query #7: 12 images -> 2/5 citation score)
            if len(verified_images) >= MAX_IMAGES_PER_QUERY:
                logging.info(f"  Reached MAX_IMAGES_PER_QUERY ({MAX_IMAGES_PER_QUERY}), stopping verification")
                break
        
        return verified_images
    
    def _fallback_visual_search(
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
            doc_images = self.retrieve_images(query, k=FALLBACK_IMAGES_PER_DOC, filter_dict={'doc_id': doc_id})
            semantic_images.extend(doc_images)
        
        for img in semantic_images:
            img_id = img.metadata.get('image_id', '')
            
            # Skip if already processed
            if img_id in seen_image_ids:
                continue
            seen_image_ids.add(img_id)
            
            is_match, similarity, chunk_id = self.verify_semantic_match(
                img, text_chunks, threshold=FALLBACK_SIMILARITY_THRESHOLD, chunk_embeddings=chunk_embeddings
            )
            fallback_images.append({
                'image': img,
                'confidence': CONFIDENCE_LOW,
                'similarity': similarity,
                'reason': f'Visual query fallback (sim={similarity:.2f})'
            })
            logging.info(f"  {img_id}: {CONFIDENCE_LOW} confidence (fallback)")
        
        return fallback_images
    
    def retrieve_with_verification(
        self,
        query: str,
        k_text: int = DEFAULT_K_TEXT
    ) -> Tuple[List[Document], List[Dict]]:
        """
        Retrieve with semantic verification and adaptive image strategy.
    
        Strategy:
        1. Text retrieval (always semantic)
        2. Batch embed chunks and images (1+1 API calls)
        3. Verify metadata images with confidence scoring
        4. Visual query fallback if needed
        
        Args:
            query: Search query
            k_text: Number of text chunks
            
        Returns:
            (text_chunks, verified_images) where each image has confidence score
        """
        logging.info(f"Adaptive retrieval with verification for: '{query}'")

        # 1. Retrieve text chunks
        text_chunks = self.retrieve_text_chunks(query, k=k_text)
        
        # 2. Batch embed chunks (1 API call)
        chunk_embeddings = self._batch_embed_chunks(text_chunks)
        
        # 3. Collect metadata candidate images
        text_chunks_strict, metadata_images = self.retrieve_with_strict_images(query, k_text)
        
        # 4. Batch embed images (1 API call)
        image_embeddings = self._batch_embed_images(metadata_images)
        
        # 5. Verify metadata images with confidence scoring
        verified_images = self._verify_metadata_images(
            metadata_images, 
            text_chunks, 
            chunk_embeddings, 
            image_embeddings
        )
        
        # 6. Visual query fallback (if no verified images and query is visual)
        if len(verified_images) == 0 and self.is_visual_query(query):
            seen_image_ids = set(img.metadata.get('image_id', '') for img in metadata_images)
            fallback_images = self._fallback_visual_search(
                query, 
                text_chunks, 
                chunk_embeddings, 
                seen_image_ids
            )
            verified_images.extend(fallback_images)
        
        logging.info(f"Final: {len(verified_images)} verified images (deduplicated)")
        return text_chunks, verified_images
    
    def prepare_for_llm(
        self,
        query: str,
        text_chunks: List[Document],
        verified_images: List[Dict]
    ) -> Dict:
        """
        Format retrieval results for LLM input.
        
        Returns structured data with:
        - Query
        - Text chunks with metadata (figure refs, image links)
        - Images with captions, confidence, similarity
        
        Args:
            query: User query
            text_chunks: Retrieved text chunks
            verified_images: Verified images with confidence scores
            
        Returns:
            Dict formatted for LLM consumption
        """
        # Format text chunks
        formatted_chunks = []
        for chunk in text_chunks:
            formatted_chunks.append({
                'chunk_id': chunk.metadata.get('chunk_id', 'unknown'),
                'text': chunk.page_content,
                'page': chunk.metadata.get('page_num', 0),
                'source': chunk.metadata.get('source', 'unknown'),
                'has_figure_references': chunk.metadata.get('has_figure_references', False),
                'image_references': chunk.metadata.get('image_references', []),
                'related_image_ids': self._format_related_image_ids(chunk.metadata)
            })
        
        # Format images
        formatted_images = []
        for img_data in verified_images:
            img = img_data['image']
            formatted_images.append({
                'image_id': img.metadata.get('image_id', 'unknown'),
                'filename': img.metadata.get('filename', 'unknown'),
                'page': img.metadata.get('page_num', 0),
                'caption': img.page_content,  # Enriched caption from OpenAI Vision
                'confidence': img_data['confidence'],
                'similarity': round(img_data['similarity'], 3),
                'reason': img_data['reason']
            })
        
        return {
            'query': query,
            'text_chunks': formatted_chunks,
            'images': formatted_images,
            'metadata': {
                'num_text_chunks': len(formatted_chunks),
                'num_images': len(formatted_images),
                'high_confidence_images': sum(1 for img in formatted_images if img['confidence'] == CONFIDENCE_HIGH),
                'medium_confidence_images': sum(1 for img in formatted_images if img['confidence'] == CONFIDENCE_MEDIUM),
                'low_confidence_images': sum(1 for img in formatted_images if img['confidence'] == CONFIDENCE_LOW)
            }
        }
    
    def retrieve_with_context(
        self,
        query: str,
        k_text: int = DEFAULT_K_TEXT,
        k_images: int = DEFAULT_CONTEXT_CHUNKS,
        include_image_context: bool = True
    ) -> Dict:
        """
        Retrieve text and images with additional context.
        
        Returns structured result with:
        - text_chunks: Top-k relevant text chunks
        - images: Top-k relevant images
        - image_contexts: For each image, related text chunks
        
        Args:
            query: Search query
            k_text: Number of text chunks
            k_images: Number of images
            include_image_context: Whether to fetch related chunks for each image
            
        Returns:
            Dict with 'text_chunks', 'images', 'image_contexts'
        """
        # Main retrieval
        text_chunks, images = self.retrieve_multimodal(
            query=query,
            k_text=k_text,
            k_images=k_images
        )
        
        result = {
            'query': query,
            'text_chunks': text_chunks,
            'images': images,
            'image_contexts': {}
        }
        
        # Get context for each image
        if include_image_context:
            for img in images:
                img_id = img.metadata['image_id']
                context_chunks = self.get_related_chunks_for_image(img, k=DEFAULT_CONTEXT_CHUNKS)
                result['image_contexts'][img_id] = context_chunks
        
        return result


def test_retriever():
    """Test retriever with semantic verification and LLM formatting."""
    print("=" * 70)
    print("🔍 Testing Adaptive Retriever with Semantic Verification + LLM Format")
    print("=" * 70)
    print()
    
    # Initialize retriever
    retriever = MultimodalRetriever()
    
    # Test query: visual request for encoder/decoder
    query = "show encoder decoder"
    
    print(f"📝 Query: {query}")
    print(f"   Visual query: {retriever.is_visual_query(query)}")
    print("-" * 70)
    
    # Retrieve with verification
    text_chunks, verified_images = retriever.retrieve_with_verification(
        query=query,
        k_text=3
    )
    
    # Display retrieval results
    print(f"\n📄 Text Chunks ({len(text_chunks)}):")
    for i, doc in enumerate(text_chunks, 1):
        print(f"  [{i}] {doc.metadata['chunk_id']} (page {doc.metadata['page_num']})")
        print(f"      has_figure_refs: {doc.metadata.get('has_figure_references', False)}")
        print(f"      related_images: {doc.metadata.get('related_image_ids', 'none')}")
        preview = doc.page_content[:80].replace('\n', ' ')
        print(f"      Preview: {preview}...")
    
    print(f"\n🖼️  Verified Images ({len(verified_images)}):")
    if verified_images:
        for i, img_data in enumerate(verified_images, 1):
            img = img_data['image']
            print(f"  [{i}] {img.metadata['image_id']} - {img.metadata['filename']}")
            print(f"      Page: {img.metadata['page_num']}")
            print(f"      Confidence: {img_data['confidence']}")
            print(f"      Similarity: {img_data['similarity']:.3f}")
            print(f"      Reason: {img_data['reason']}")
    else:
        print("  ❌ No verified images")
    
    # Format for LLM
    print("\n" + "=" * 70)
    print("🤖 LLM Input Format")
    print("=" * 70)
    
    llm_input = retriever.prepare_for_llm(query, text_chunks, verified_images)
    
    print(f"\n📊 Metadata:")
    print(f"   Total text chunks: {llm_input['metadata']['num_text_chunks']}")
    print(f"   Total images: {llm_input['metadata']['num_images']}")
    print(f"   HIGH confidence: {llm_input['metadata']['high_confidence_images']}")
    print(f"   MEDIUM confidence: {llm_input['metadata']['medium_confidence_images']}")
    print(f"   LOW confidence: {llm_input['metadata']['low_confidence_images']}")
    
    print(f"\n📄 Formatted Text Chunks:")
    for i, chunk in enumerate(llm_input['text_chunks'], 1):
        print(f"  [{i}] {chunk['chunk_id']} (page {chunk['page']})")
        print(f"      Source: {chunk['source']}")
        print(f"      Has figure refs: {chunk['has_figure_references']}")
        if chunk['related_image_ids']:
            print(f"      Related images: {', '.join(chunk['related_image_ids'])}")
        print(f"      Text: {chunk['text'][:100].replace(chr(10), ' ')}...")
    
    print(f"\n🖼️  Formatted Images:")
    for i, img in enumerate(llm_input['images'], 1):
        print(f"  [{i}] {img['image_id']} (page {img['page']})")
        print(f"      Confidence: {img['confidence']} | Similarity: {img['similarity']}")
        print(f"      Reason: {img['reason']}")
        print(f"      Caption: {img['caption'][:150].replace(chr(10), ' ')}...")
    
    print("\n" + "=" * 70)
    print("✅ Test complete! Ready for Generator implementation")
    print("=" * 70)


if __name__ == "__main__":
    test_retriever()
