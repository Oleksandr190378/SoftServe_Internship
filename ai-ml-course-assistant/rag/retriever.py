"""
Multimodal Retriever for text chunks and image captions.

Retrieves relevant text chunks and images from ChromaDB based on query.
Uses anti-hallucination metadata to prioritize accurate image-text associations.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple
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

# Embedding configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536

# Visual query keywords
VISUAL_KEYWORDS = [
    "show", "diagram", "architecture", "figure", "image", 
    "visualization", "chart", "graph", "draw", "display",
    "illustrate", "picture", "schema"
]


SIMILARITY_THRESHOLD = 0.5  # For same-page images (lowered for visual queries)
SIMILARITY_THRESHOLD_NEARBY = 0.65  # For ±1 page images


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
        text_collection: str = "text_chunks",
        image_collection: str = "image_captions"
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
        
        # Load ChromaDB collections
        self.text_store = Chroma(
            collection_name=text_collection,
            embedding_function=self.embeddings,
            persist_directory=str(chroma_dir / text_collection)
        )
        
        self.image_store = Chroma(
            collection_name=image_collection,
            embedding_function=self.embeddings,
            persist_directory=str(chroma_dir / image_collection)
        )
        
        logging.info(f"✅ Loaded collections: {text_collection}, {image_collection}")
    
    def retrieve_text_chunks(
        self,
        query: str,
        k: int = 3,
        filter_dict: Dict = None,
        search_type: str = "mmr",
        mmr_lambda: float = 0.7
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
            results = self.text_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=k*2,  # Fetch more candidates for better diversity
                lambda_mult=mmr_lambda,
                filter=filter_dict
            )
            logging.info(f"Retrieved {len(results)} text chunks (MMR, λ={mmr_lambda})")
        else:
            results = self.text_store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            logging.info(f"Retrieved {len(results)} text chunks (similarity)")
        
        return results
    
    def retrieve_images(
        self,
        query: str,
        k: int = 3,
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
        results = self.image_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
        logging.info(f"Retrieved {len(results)} images")
        return results
    
    def retrieve_multimodal(
        self,
        query: str,
        k_text: int = 3,
        k_images: int = 3,
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
        k: int = 3
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
        chunks = self.text_store.similarity_search(
            query=image_doc.page_content,
            k=k,
            filter=filter_dict
        )

        chunks_with_link = []
        chunks_without_link = []
        
        for chunk in chunks:
            related_ids = chunk.metadata.get('related_image_ids', '')
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
            results = self.image_store.get(
                where={"image_id": img_id}
            )
            if results and 'documents' in results:
                for i, doc_text in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if 'metadatas' in results else {}
                    images.append(Document(page_content=doc_text, metadata=metadata))
        
        logging.info(f"Fetched {len(images)} images by ID")
        return images
    
    def retrieve_with_strict_images(
        self,
        query: str,
        k_text: int = 3
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
            related = chunk.metadata.get('related_image_ids', '')
            if related:
                image_ids_strong.update([id.strip() for id in related.split(',') if id.strip()])
            
            # Add nearby images ONLY if chunk mentions figures
            if chunk.metadata.get('has_figure_references', False):
                nearby = chunk.metadata.get('nearby_image_ids', '')
                if nearby:
                    image_ids_weak.update([id.strip() for id in nearby.split(',') if id.strip()])
        
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
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def verify_semantic_match(
        self,
        image: Document,
        text_chunks: List[Document],
        threshold: float = SIMILARITY_THRESHOLD,
        chunk_embeddings: Dict[str, List[float]] = None
    ) -> Tuple[bool, float, str]:
        """
        Verify if image is semantically related to any text chunk.
        
        Args:
            image: Image document with embedding
            text_chunks: List of text chunk documents
            threshold: Minimum similarity score
            chunk_embeddings: Pre-computed chunk embeddings (optional)
            
        Returns:
            (is_match, best_similarity, matched_chunk_id)
        """
        # Get image embedding
        img_embedding = self.embeddings.embed_query(image.page_content)
        
        best_similarity = 0.0
        best_chunk_id = ""
        
        for chunk in text_chunks:
            chunk_id = chunk.metadata.get('chunk_id', 'unknown')
            
            # Use cached embedding if available
            if chunk_embeddings and chunk_id in chunk_embeddings:
                chunk_embedding = chunk_embeddings[chunk_id]
            else:
                chunk_embedding = self.embeddings.embed_query(chunk.page_content)
                if chunk_embeddings is not None:
                    chunk_embeddings[chunk_id] = chunk_embedding
            
            similarity = self.cosine_similarity(img_embedding, chunk_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_chunk_id = chunk_id
        
        is_match = best_similarity >= threshold
        return is_match, best_similarity, best_chunk_id
    
    def retrieve_with_verification(
        self,
        query: str,
        k_text: int = 3
    ) -> Tuple[List[Document], List[Dict]]:
        """
        Retrieve with semantic verification and adaptive image strategy.
        
        Strategy:
        1. Text retrieval (always semantic)
        2. Metadata candidates (structural linking)
        3. Semantic verification (similarity check with caching)
        4. Visual query fallback (if no metadata images)
        
        Optimizations:
        - Batch embeddings for all chunks at once
        - Cache chunk embeddings to avoid re-computation
        - Deduplicate images by image_id
        
        Args:
            query: Search query
            k_text: Number of text chunks
            
        Returns:
            (text_chunks, verified_images) where each image has confidence score
        """
        logging.info(f"Adaptive retrieval with verification for: '{query}'")

        text_chunks = self.retrieve_text_chunks(query, k=k_text)
        
        # 2. Pre-compute chunk embeddings in batch
        chunk_embeddings = {}
        chunk_texts = [chunk.page_content for chunk in text_chunks]
        if chunk_texts:
            # Batch embed all chunks at once (1 API call instead of N)
            embeddings_list = self.embeddings.embed_documents(chunk_texts)
            for i, chunk in enumerate(text_chunks):
                chunk_id = chunk.metadata.get('chunk_id', f'chunk_{i}')
                chunk_embeddings[chunk_id] = embeddings_list[i]
        
        # 3. Collect metadata candidate images
        text_chunks_strict, metadata_images = self.retrieve_with_strict_images(query, k_text)
        
        # 4. Semantic verification for metadata images
        verified_images = []
        seen_image_ids = set()  # For deduplication
        
        for img in metadata_images:
            img_id = img.metadata.get('image_id', '')
            
            # Skip duplicates
            if img_id in seen_image_ids:
                logging.info(f"  {img_id}: Skipped (duplicate)")
                continue
            seen_image_ids.add(img_id)
            
            # Check if chunk explicitly mentions figure (skip verification)
            has_explicit_ref = any(
                img_id in chunk.metadata.get('related_image_ids', '')
                for chunk in text_chunks
                if chunk.metadata.get('has_figure_references', False)
            )
            
            if has_explicit_ref:
                # HIGH confidence: explicit figure reference
                verified_images.append({
                    'image': img,
                    'confidence': 'HIGH',
                    'similarity': 1.0,
                    'reason': 'Explicit figure reference in text'
                })
                logging.info(f"  {img_id}: HIGH confidence (explicit ref)")
            else:
                # Semantic verification with cached embeddings
                is_match, similarity, chunk_id = self.verify_semantic_match(
                    img, text_chunks, chunk_embeddings=chunk_embeddings
                )
                
                if is_match:
                    verified_images.append({
                        'image': img,
                        'confidence': 'MEDIUM',
                        'similarity': similarity,
                        'reason': f'Semantic match with {chunk_id} (sim={similarity:.2f})'
                    })
                    logging.info(f"  {img_id}: MEDIUM confidence (sim={similarity:.2f})")
                else:
                    logging.info(f"  {img_id}: Rejected (sim={similarity:.2f} < threshold)")
        
        # 5. Visual query fallback (if no verified images and query is visual)
        if len(verified_images) == 0 and self.is_visual_query(query):
            logging.info("  No metadata images, fallback to semantic caption search")
            
            # Get unique document IDs from text chunks to filter fallback search
            chunk_doc_ids = set(chunk.metadata.get('doc_id', '') for chunk in text_chunks)
            logging.info(f"  Filtering fallback to documents: {chunk_doc_ids}")
            
            # Retrieve images with document filter
            semantic_images = []
            for doc_id in chunk_doc_ids:
                doc_images = self.retrieve_images(query, k=2, filter_dict={'doc_id': doc_id})
                semantic_images.extend(doc_images)
            
            for img in semantic_images:
                img_id = img.metadata.get('image_id', '')
                
                # Skip if already processed
                if img_id in seen_image_ids:
                    continue
                seen_image_ids.add(img_id)
                
                is_match, similarity, chunk_id = self.verify_semantic_match(
                    img, text_chunks, threshold=0.5, chunk_embeddings=chunk_embeddings
                )
                verified_images.append({
                    'image': img,
                    'confidence': 'LOW',
                    'similarity': similarity,
                    'reason': f'Visual query fallback (sim={similarity:.2f})'
                })
                logging.info(f"  {img_id}: LOW confidence (fallback)")
        
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
                'related_image_ids': [
                    id.strip() 
                    for id in chunk.metadata.get('related_image_ids', '').split(',') 
                    if id.strip()
                ]
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
                'high_confidence_images': sum(1 for img in formatted_images if img['confidence'] == 'HIGH'),
                'medium_confidence_images': sum(1 for img in formatted_images if img['confidence'] == 'MEDIUM'),
                'low_confidence_images': sum(1 for img in formatted_images if img['confidence'] == 'LOW')
            }
        }
    
    def retrieve_with_context(
        self,
        query: str,
        k_text: int = 5,
        k_images: int = 2,
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
                context_chunks = self.get_related_chunks_for_image(img, k=2)
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
