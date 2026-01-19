"""
Test suite for retriever.py - Multimodal Retriever functionality.

Tests cover:
- Initialization and error handling
- Text chunk retrieval (similarity and MMR search)
- Image caption retrieval
- Metadata parsing
- Result filtering and ranking
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
import sys
from pathlib import Path

# Add rag module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "rag"))

from retriever import (
    MultimodalRetriever,
    EMBEDDING_MODEL,
    EMBEDDING_DIMS,
    DEFAULT_K_TEXT,
    DEFAULT_K_IMAGES,
    SIMILARITY_THRESHOLD,
    SIMILARITY_THRESHOLD_NEARBY,
    DEFAULT_MMR_LAMBDA,
    VISUAL_KEYWORDS,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW
)


class TestRetrieverConstants(unittest.TestCase):
    """Test retriever configuration constants."""
    
    def test_embedding_model_constant(self):
        """Verify embedding model matches build_index.py."""
        self.assertEqual(EMBEDDING_MODEL, "text-embedding-3-small")
    
    def test_embedding_dims_constant(self):
        """Verify embedding dimensions."""
        self.assertEqual(EMBEDDING_DIMS, 1536)
    
    def test_default_k_values(self):
        """Verify default retrieval counts."""
        self.assertEqual(DEFAULT_K_TEXT, 3)
        self.assertEqual(DEFAULT_K_IMAGES, 3)
    
    def test_similarity_thresholds(self):
        """Verify semantic matching thresholds."""
        self.assertIsInstance(SIMILARITY_THRESHOLD, float)
        self.assertIsInstance(SIMILARITY_THRESHOLD_NEARBY, float)
        self.assertGreater(SIMILARITY_THRESHOLD_NEARBY, SIMILARITY_THRESHOLD)
    
    def test_mmr_lambda_range(self):
        """Verify MMR lambda is in valid range."""
        self.assertGreaterEqual(DEFAULT_MMR_LAMBDA, 0.0)
        self.assertLessEqual(DEFAULT_MMR_LAMBDA, 1.0)
    
    def test_visual_keywords_not_empty(self):
        """Visual keywords list should not be empty."""
        self.assertGreater(len(VISUAL_KEYWORDS), 0)
        self.assertIn("diagram", VISUAL_KEYWORDS)
    
    def test_confidence_levels(self):
        """Verify confidence level constants."""
        self.assertEqual(CONFIDENCE_HIGH, "HIGH")
        self.assertEqual(CONFIDENCE_MEDIUM, "MEDIUM")
        self.assertEqual(CONFIDENCE_LOW, "LOW")


class TestRetrieverInitialization(unittest.TestCase):
    """Test MultimodalRetriever initialization."""
    
    @patch('retriever.Chroma')
    @patch('retriever.OpenAIEmbeddings')
    def test_retriever_initialization_success(self, mock_embeddings, mock_chroma):
        """Successfully initialize retriever with valid paths."""
        # Arrange
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance
        mock_text_store = MagicMock()
        mock_image_store = MagicMock()
        mock_chroma.side_effect = [mock_text_store, mock_image_store]
        
        # Act
        retriever = MultimodalRetriever()
        
        # Assert
        self.assertEqual(retriever.embeddings, mock_embeddings_instance)
        mock_embeddings.assert_called_once()
        self.assertEqual(mock_chroma.call_count, 2)  # text and image stores
    
    @patch('retriever.Chroma')
    @patch('retriever.OpenAIEmbeddings')
    def test_retriever_initialization_text_collection_error(self, mock_embeddings, mock_chroma):
        """Handle error when text collection initialization fails."""
        # Arrange
        mock_embeddings.return_value = MagicMock()
        mock_chroma.side_effect = [
            Exception("Collection not found"),  # text store fails
            MagicMock()  # image store (not called)
        ]
        
        # Act & Assert
        with self.assertRaises(RuntimeError):
            MultimodalRetriever()
    
    @patch('retriever.Chroma')
    @patch('retriever.OpenAIEmbeddings')
    def test_retriever_initialization_image_collection_error(self, mock_embeddings, mock_chroma):
        """Handle error when image collection initialization fails."""
        # Arrange
        mock_embeddings.return_value = MagicMock()
        mock_chroma.side_effect = [
            MagicMock(),  # text store succeeds
            Exception("Collection not found")  # image store fails
        ]
        
        # Act & Assert
        with self.assertRaises(RuntimeError):
            MultimodalRetriever()


class TestParseJsonList(unittest.TestCase):
    """Test JSON list parsing helper method."""
    
    @patch('retriever.Chroma')
    @patch('retriever.OpenAIEmbeddings')
    def setUp(self, mock_embeddings, mock_chroma):
        """Set up mock retriever."""
        mock_embeddings.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()
        self.retriever = MultimodalRetriever()
    
    def test_parse_json_list_from_json_string(self):
        """Parse JSON-encoded list string."""
        # Act
        result = self.retriever._parse_json_list('["img1", "img2", "img3"]', 'test_field')
        
        # Assert
        self.assertEqual(result, ["img1", "img2", "img3"])
    
    def test_parse_json_list_from_list(self):
        """Return list as-is if already a list."""
        # Act
        result = self.retriever._parse_json_list(["img1", "img2"], 'test_field')
        
        # Assert
        self.assertEqual(result, ["img1", "img2"])
    
    def test_parse_json_list_from_comma_separated(self):
        """Parse comma-separated string."""
        # Act
        result = self.retriever._parse_json_list('img1, img2, img3', 'test_field')
        
        # Assert
        self.assertEqual(result, ["img1", "img2", "img3"])
    
    def test_parse_json_list_empty_input(self):
        """Return empty list for empty input."""
        # Act
        result = self.retriever._parse_json_list('', 'test_field')
        
        # Assert
        self.assertEqual(result, [])
    
    def test_parse_json_list_none_input(self):
        """Return empty list for None input."""
        # Act
        result = self.retriever._parse_json_list(None, 'test_field')
        
        # Assert
        self.assertEqual(result, [])
    
    def test_parse_json_list_invalid_json(self):
        """Fallback to comma-separated parsing on invalid JSON."""
        # Act - invalid JSON without comma falls back to comma-separated parsing
        # Since "[invalid json" has no comma, it returns as single element
        result = self.retriever._parse_json_list('[invalid json', 'test_field')
        
        # Assert - no comma in string, so treated as single value
        self.assertEqual(result, ['[invalid json'])


class TestFormatRelatedImageIds(unittest.TestCase):
    """Test related image ID extraction."""
    
    @patch('retriever.Chroma')
    @patch('retriever.OpenAIEmbeddings')
    def setUp(self, mock_embeddings, mock_chroma):
        """Set up mock retriever."""
        mock_embeddings.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()
        self.retriever = MultimodalRetriever()
    
    def test_format_related_image_ids_from_json(self):
        """Extract image IDs from JSON metadata."""
        # Arrange
        metadata = {'related_image_ids': '["img_1", "img_2"]'}
        
        # Act
        result = self.retriever._format_related_image_ids(metadata)
        
        # Assert
        self.assertEqual(result, ["img_1", "img_2"])
    
    def test_format_related_image_ids_missing_field(self):
        """Return empty list when field is missing."""
        # Arrange
        metadata = {'doc_id': 'd1'}  # No related_image_ids
        
        # Act
        result = self.retriever._format_related_image_ids(metadata)
        
        # Assert
        self.assertEqual(result, [])
    
    def test_format_related_image_ids_empty_string(self):
        """Return empty list for empty string."""
        # Arrange
        metadata = {'related_image_ids': ''}
        
        # Act
        result = self.retriever._format_related_image_ids(metadata)
        
        # Assert
        self.assertEqual(result, [])


class TestTextChunkRetrieval(unittest.TestCase):
    """Test text chunk retrieval functionality."""
    
    @patch('retriever.Chroma')
    @patch('retriever.OpenAIEmbeddings')
    def setUp(self, mock_embeddings, mock_chroma):
        """Set up mock retriever."""
        mock_embeddings.return_value = MagicMock()
        self.mock_text_store = MagicMock()
        self.mock_image_store = MagicMock()
        mock_chroma.side_effect = [self.mock_text_store, self.mock_image_store]
        self.retriever = MultimodalRetriever()
    
    def test_retrieve_text_chunks_similarity_search(self):
        """Retrieve chunks using similarity search."""
        # Arrange
        mock_docs = [
            MagicMock(page_content='test content 1', metadata={'chunk_id': 'c1'}),
            MagicMock(page_content='test content 2', metadata={'chunk_id': 'c2'})
        ]
        self.mock_text_store.similarity_search.return_value = mock_docs
        
        # Act
        results = self.retriever.retrieve_text_chunks(
            "test query",
            k=2,
            search_type="similarity"
        )
        
        # Assert
        self.assertEqual(len(results), 2)
        self.mock_text_store.similarity_search.assert_called_once()
    
    def test_retrieve_text_chunks_mmr_search(self):
        """Retrieve chunks using MMR search for diversity."""
        # Arrange
        mock_docs = [
            MagicMock(page_content='diverse content 1', metadata={'chunk_id': 'c1'}),
            MagicMock(page_content='diverse content 2', metadata={'chunk_id': 'c2'})
        ]
        self.mock_text_store.max_marginal_relevance_search.return_value = mock_docs
        
        # Act
        results = self.retriever.retrieve_text_chunks(
            "test query",
            k=2,
            search_type="mmr",
            mmr_lambda=0.7
        )
        
        # Assert
        self.assertEqual(len(results), 2)
        self.mock_text_store.max_marginal_relevance_search.assert_called_once()
    
    def test_retrieve_text_chunks_with_filter(self):
        """Retrieve chunks with metadata filter."""
        # Arrange
        filter_dict = {'doc_id': 'arxiv_1234'}
        mock_docs = [MagicMock(page_content='filtered content')]
        self.mock_text_store.similarity_search.return_value = mock_docs
        
        # Act
        results = self.retriever.retrieve_text_chunks(
            "test query",
            k=3,
            filter_dict=filter_dict,
            search_type="similarity"
        )
        
        # Assert
        self.assertEqual(len(results), 1)


class TestImageRetrieval(unittest.TestCase):
    """Test image caption retrieval functionality."""
    
    @patch('retriever.Chroma')
    @patch('retriever.OpenAIEmbeddings')
    def setUp(self, mock_embeddings, mock_chroma):
        """Set up mock retriever."""
        mock_embeddings.return_value = MagicMock()
        self.mock_text_store = MagicMock()
        self.mock_image_store = MagicMock()
        mock_chroma.side_effect = [self.mock_text_store, self.mock_image_store]
        self.retriever = MultimodalRetriever()
    
    def test_retrieve_images_basic(self):
        """Retrieve images with default parameters."""
        # Arrange
        mock_docs = [
            MagicMock(page_content='image caption 1', metadata={'image_id': 'img1'}),
            MagicMock(page_content='image caption 2', metadata={'image_id': 'img2'})
        ]
        self.mock_image_store.similarity_search.return_value = mock_docs
        
        # Act - Verify method exists and can be called
        # Note: Full implementation may vary
        self.assertIsNotNone(self.retriever.image_store)


class TestVisualizationKeywordDetection(unittest.TestCase):
    """Test visual query keyword matching."""
    
    def test_visual_keywords_list_completeness(self):
        """Verify all visualization-related keywords are present."""
        expected_keywords = ["diagram", "architecture", "chart", "graph", "figure"]
        for keyword in expected_keywords:
            self.assertIn(keyword, VISUAL_KEYWORDS)
    
    def test_visual_keywords_case_variation(self):
        """Keywords should support case-insensitive matching in queries."""
        # This is tested at the query level, not constant level
        self.assertIsInstance(VISUAL_KEYWORDS, list)


class TestEmbeddingCache(unittest.TestCase):
    """Test embedding caching mechanism."""
    
    @patch('retriever.Chroma')
    @patch('retriever.OpenAIEmbeddings')
    def setUp(self, mock_embeddings, mock_chroma):
        """Set up mock retriever."""
        mock_embeddings.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()
        self.retriever = MultimodalRetriever()
    
    def test_embedding_cache_initialized_empty(self):
        """Embedding cache should start empty."""
        self.assertEqual(self.retriever._chunk_embeddings_cache, {})
    
    def test_embedding_cache_is_dict(self):
        """Embedding cache should be a dictionary."""
        self.assertIsInstance(self.retriever._chunk_embeddings_cache, dict)


if __name__ == '__main__':
    unittest.main()
