"""
Multimodal Retriever for text chunks and image captions.

Main retrieval class that coordinates text/image retrieval with semantic verification.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

from config import (
    EMBEDDING,
    RETRIEVAL,
    COLLECTIONS,
    CONFIDENCE,
    CHROMA_DIR
)
from utils.logging_config import setup_logging
from .verification import SemanticVerifier
from .image_ops import ImageRetriever
from .utils import parse_json_list, format_related_image_ids, is_visual_query

load_dotenv()
setup_logging()

# Use config constants
EMBEDDING_MODEL = EMBEDDING.MODEL
EMBEDDING_DIMS = EMBEDDING.DIMENSIONS

DEFAULT_K_TEXT = RETRIEVAL.DEFAULT_K_TEXT
DEFAULT_K_IMAGES = RETRIEVAL.DEFAULT_K_IMAGES
DEFAULT_CONTEXT_CHUNKS = RETRIEVAL.DEFAULT_CONTEXT_CHUNKS
DEFAULT_MMR_LAMBDA = RETRIEVAL.MMR_LAMBDA
MMR_FETCH_MULTIPLIER = RETRIEVAL.MMR_FETCH_MULTIPLIER

# Collection names from config
DEFAULT_TEXT_COLLECTION = COLLECTIONS.TEXT_CHUNKS
DEFAULT_IMAGE_COLLECTION = COLLECTIONS.IMAGE_CAPTIONS

# Confidence levels
CONFIDENCE_HIGH = CONFIDENCE.HIGH
CONFIDENCE_MEDIUM = CONFIDENCE.MEDIUM
CONFIDENCE_LOW = CONFIDENCE.LOW


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
        
        # Initialize specialized components
        self.verifier = SemanticVerifier(self.embeddings, self.image_store)
        self.image_retriever = ImageRetriever(self.image_store, self.text_store)
    
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
        
        Delegates to ImageRetriever for actual retrieval.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters
        
        Returns:
            List of Document objects with captions and metadata
        """
        return self.image_retriever.retrieve_images(query, k, filter_dict)
    
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
            text_results = self.image_retriever.rerank_by_figure_refs(text_results)
        
        return text_results, image_results
    
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
            related_ids = parse_json_list(related_ids_raw, 'related_image_ids')
            
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
        
        Delegates to ImageRetriever.
        
        Args:
            image_ids: List of image IDs to fetch
        
        Returns:
            List of image Documents
        """
        return self.image_retriever.fetch_images_by_ids(image_ids)
    
    def retrieve_with_strict_images(
        self,
        query: str,
        k_text: int = DEFAULT_K_TEXT
    ) -> Tuple[List[Document], List[Document]]:
        """
        Retrieve text chunks and ONLY images explicitly referenced in those chunks.
        
        Delegates to ImageRetriever for strict metadata-driven retrieval.
        
        Args:
            query: Search query
            k_text: Number of text chunks
        
        Returns:
            (text_chunks, images) where images are strictly metadata-linked
        """
        return self.image_retriever.retrieve_with_strict_images(query, k_text)
    
    def is_visual_query(self, query: str) -> bool:
        """
        Check if query requests visual content.
        
        Delegates to utils function.
        """
        return is_visual_query(query)
    
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
        import time
        start_time = time.perf_counter()
        
        logging.info(f"Adaptive retrieval with verification for: '{query}'")
        
        # 1. Retrieve text chunks
        text_chunks = self.retrieve_text_chunks(query, k=k_text)
        
        # 2. Batch embed chunks (1 API call)
        chunk_embeddings = self.verifier.batch_embed_chunks(text_chunks)
        
        # 3. Collect metadata candidate images
        text_chunks_strict, metadata_images = self.retrieve_with_strict_images(query, k_text)
        
        # 4. Batch embed images (1 API call)
        image_embeddings = self.verifier.batch_embed_images(metadata_images)
        
        # 5. Verify metadata images with confidence scoring
        verified_images = self.verifier.verify_metadata_images(
            metadata_images,
            text_chunks,
            chunk_embeddings,
            image_embeddings
        )
        
        # 6. Visual query fallback (if no verified images and query is visual)
        if len(verified_images) == 0 and self.is_visual_query(query):
            seen_image_ids = set(img.metadata.get('image_id', '') for img in metadata_images)
            fallback_images = self.verifier.fallback_visual_search(
                query,
                text_chunks,
                chunk_embeddings,
                seen_image_ids
            )
            verified_images.extend(fallback_images)
        
        retrieval_time = time.perf_counter() - start_time
        logging.info(f"Retrieved in {retrieval_time:.3f}s: {len(text_chunks)} chunks, {len(verified_images)} images")
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
                'related_image_ids': format_related_image_ids(chunk.metadata)
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
