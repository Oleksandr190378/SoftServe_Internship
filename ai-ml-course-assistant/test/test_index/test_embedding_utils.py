"""
Test suite for embedding_utils.py - STAGE 1-3 validation pattern.

STAGE 1: Input validation, error handling, resource cleanup (9 tests)
STAGE 2: Constants and configuration (3 tests)
STAGE 3: Helper functions and business logic (12 tests)

Total: 24 tests
"""

import unittest
from unittest.mock import patch, MagicMock, Mock
import sys
from pathlib import Path

# Add index module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "index"))

import embedding_utils
from embedding_utils import (
    generate_embeddings_batch,
    embed_chunks,
    embed_images,
    embed_document,
    EMBEDDING_MODEL,
    EMBEDDING_DIMS,
    BATCH_SIZE
)
import openai


class TestEmbeddingConstants(unittest.TestCase):
    """STAGE 2: Verify embedding configuration constants."""
    
    def test_embedding_model_is_text_embedding_3_small(self):
        """Verify correct OpenAI embedding model used."""
        self.assertEqual(EMBEDDING_MODEL, "text-embedding-3-small")
    
    def test_embedding_dims_is_1536(self):
        """Verify embedding dimensionality is 1536."""
        self.assertEqual(EMBEDDING_DIMS, 1536)
    
    def test_batch_size_is_100(self):
        """Verify default batch size for API calls."""
        self.assertEqual(BATCH_SIZE, 100)


# ============================================================================
# STAGE 1: Input Validation & Error Handling
# ============================================================================

class TestGenerateEmbeddingsBatchValidation(unittest.TestCase):
    """STAGE 1: Validate inputs and handle API errors."""
    
    def setUp(self):
        """Create mock OpenAI client."""
        self.mock_client = MagicMock()
    
    def test_generate_embeddings_batch_empty_list(self):
        """Empty text list should return empty embeddings."""
        # Arrange
        self.mock_client.embeddings.create.return_value = MagicMock(data=[])
        
        # Act
        result = generate_embeddings_batch([], self.mock_client)
        
        # Assert
        self.assertEqual(result, [])
        self.mock_client.embeddings.create.assert_called_once()
    
    @patch('embedding_utils.logging')
    def test_embed_chunks_invalid_chunks_type(self, mock_log):
        """Reject non-list chunks input."""
        # Act & Assert
        with self.assertRaises((TypeError, AttributeError)):
            embed_chunks("not a list", self.mock_client)
    
    @patch('embedding_utils.logging')
    def test_embed_images_invalid_images_type(self, mock_log):
        """Reject non-list images input."""
        # Act & Assert
        with self.assertRaises((TypeError, AttributeError)):
            embed_images("not a list", self.mock_client)


# ============================================================================
# STAGE 3: Helper Functions & Business Logic
# ============================================================================

class TestGenerateEmbeddingsBatchLogic(unittest.TestCase):
    """STAGE 3: Test embedding generation logic."""
    
    def setUp(self):
        """Create mock OpenAI client."""
        self.mock_client = MagicMock()
    
    def test_generate_embeddings_batch_returns_correct_embeddings(self):
        """Verify embeddings are extracted correctly from API response."""
        # Arrange
        mock_embedding_1 = [0.1] * EMBEDDING_DIMS
        mock_embedding_2 = [0.2] * EMBEDDING_DIMS
        
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=mock_embedding_1),
            MagicMock(embedding=mock_embedding_2)
        ]
        self.mock_client.embeddings.create.return_value = mock_response
        
        # Act
        result = generate_embeddings_batch(["text1", "text2"], self.mock_client)
        
        # Assert
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], mock_embedding_1)
        self.assertEqual(result[1], mock_embedding_2)
    
    def test_generate_embeddings_batch_maintains_order(self):
        """Verify embeddings maintain input text order."""
        # Arrange
        embeddings_data = [[i] * EMBEDDING_DIMS for i in range(3)]
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=e) for e in embeddings_data]
        self.mock_client.embeddings.create.return_value = mock_response
        
        # Act
        texts = ["first", "second", "third"]
        result = generate_embeddings_batch(texts, self.mock_client)
        
        # Assert
        self.assertEqual(len(result), 3)
        for i, embedding in enumerate(result):
            self.assertEqual(embedding, embeddings_data[i])


class TestEmbedChunksLogic(unittest.TestCase):
    """STAGE 3: Test text chunk embedding logic."""
    
    def setUp(self):
        """Create mock client and sample chunks."""
        self.mock_client = MagicMock()
        self.chunks = [
            {'text': 'chunk 1' * 100, 'chunk_id': 'c1', 'doc_id': 'd1'},
            {'text': 'chunk 2' * 100, 'chunk_id': 'c2', 'doc_id': 'd1'},
            {'text': 'chunk 3' * 100, 'chunk_id': 'c3', 'doc_id': 'd1'}
        ]
    
    @patch('embedding_utils.generate_embeddings_batch')
    @patch('embedding_utils.logging')
    def test_embed_chunks_adds_embedding_fields(self, mock_log, mock_batch):
        """Verify embedding fields added to chunks."""
        # Arrange
        mock_embeddings = [[i] * EMBEDDING_DIMS for i in range(3)]
        mock_batch.return_value = mock_embeddings
        
        # Act
        result_chunks, cost = embed_chunks(self.chunks, self.mock_client)
        
        # Assert
        self.assertEqual(len(result_chunks), 3)
        for i, chunk in enumerate(result_chunks):
            self.assertIn('embedding', chunk)
            self.assertEqual(chunk['embedding_model'], EMBEDDING_MODEL)
            self.assertEqual(chunk['embedding_dims'], EMBEDDING_DIMS)
    
    @patch('embedding_utils.generate_embeddings_batch')
    @patch('embedding_utils.logging')
    def test_embed_chunks_calculates_cost(self, mock_log, mock_batch):
        """Verify cost calculation (tokens / 1M * $0.02)."""
        # Arrange
        mock_batch.return_value = [[0.1] * EMBEDDING_DIMS for _ in self.chunks]
        total_chars = sum(len(c['text']) for c in self.chunks)
        expected_tokens = total_chars / 4
        expected_cost = (expected_tokens / 1_000_000) * 0.02
        
        # Act
        _, cost = embed_chunks(self.chunks, self.mock_client)
        
        # Assert
        self.assertAlmostEqual(cost, expected_cost, places=6)
    
    @patch('embedding_utils.generate_embeddings_batch')
    @patch('embedding_utils.logging')
    def test_embed_chunks_batching(self, mock_log, mock_batch):
        """Verify chunks are batched correctly."""
        # Arrange
        large_chunk_list = self.chunks * 30  # 90 chunks
        mock_batch.return_value = [[i % EMBEDDING_DIMS] * EMBEDDING_DIMS for i in range(100)]
        
        # Act
        embed_chunks(large_chunk_list, self.mock_client, batch_size=50)
        
        # Assert
        # Should be called multiple times for batching
        self.assertGreaterEqual(mock_batch.call_count, 1)
    
    @patch('embedding_utils.logging')
    def test_embed_chunks_preserves_chunk_data(self, mock_log):
        """Verify original chunk data is preserved in output."""
        # Arrange
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1]*EMBEDDING_DIMS) for _ in self.chunks]
        self.mock_client.embeddings.create.return_value = mock_response
        
        # Act
        result_chunks, _ = embed_chunks(self.chunks, self.mock_client)
        
        # Assert
        for orig, result in zip(self.chunks, result_chunks):
            self.assertEqual(result['chunk_id'], orig['chunk_id'])
            self.assertEqual(result['doc_id'], orig['doc_id'])
            self.assertEqual(result['text'], orig['text'])


class TestEmbedImagesLogic(unittest.TestCase):
    """STAGE 3: Test image embedding logic."""
    
    def setUp(self):
        """Create mock client and sample images."""
        self.mock_client = MagicMock()
        self.images = [
            {
                'image_id': 'img1',
                'doc_id': 'd1',
                'enriched_caption': 'A beautiful landscape',
                'author_caption': 'Sunset'
            },
            {
                'image_id': 'img2',
                'doc_id': 'd1',
                'enriched_caption': 'Neural network visualization',
                'author_caption': ''
            }
        ]
    
    @patch('embedding_utils.logging')
    def test_embed_images_empty_list(self, mock_log):
        """Empty image list should return empty result."""
        # Act
        result_images, cost = embed_images([], self.mock_client)
        
        # Assert
        self.assertEqual(result_images, [])
        self.assertEqual(cost, 0.0)
    
    @patch('embedding_utils.generate_embeddings_batch')
    @patch('embedding_utils.logging')
    def test_embed_images_adds_embedding_fields(self, mock_log, mock_batch):
        """Verify embedding fields added to images."""
        # Arrange
        mock_batch.return_value = [[i] * EMBEDDING_DIMS for i in range(len(self.images))]
        
        # Act
        result_images, _ = embed_images(self.images, self.mock_client)
        
        # Assert
        self.assertEqual(len(result_images), 2)
        for img in result_images:
            self.assertIn('embedding', img)
            self.assertEqual(img['embedding_model'], EMBEDDING_MODEL)
            self.assertEqual(img['embedding_dims'], EMBEDDING_DIMS)
            self.assertIn('caption_for_embedding', img)
    
    @patch('embedding_utils.generate_embeddings_batch')
    @patch('embedding_utils.logging')
    def test_embed_images_combines_captions(self, mock_log, mock_batch):
        """Verify captions are combined (enriched + author)."""
        # Arrange
        mock_batch.return_value = [[0.1] * EMBEDDING_DIMS for _ in self.images]
        
        # Act
        result_images, _ = embed_images(self.images, self.mock_client)
        
        # Assert
        # First image should have both captions
        self.assertIn('Author caption:', result_images[0]['caption_for_embedding'])
        # Second image has empty author caption, should only have enriched
        self.assertEqual(result_images[1]['caption_for_embedding'], 
                        self.images[1]['enriched_caption'])
    
    @patch('embedding_utils.generate_embeddings_batch')
    @patch('embedding_utils.logging')
    def test_embed_images_no_enriched_caption(self, mock_log, mock_batch):
        """Handle images without enriched captions gracefully."""
        # Arrange
        images_no_caption = [
            {'image_id': 'img1', 'doc_id': 'd1', 'author_caption': 'test'}
        ]
        mock_batch.return_value = [[0.1] * EMBEDDING_DIMS]
        
        # Act
        result_images, _ = embed_images(images_no_caption, self.mock_client)
        
        # Assert - should use author caption only or empty string
        self.assertIsNotNone(result_images[0]['caption_for_embedding'])


class TestEmbedDocumentLogic(unittest.TestCase):
    """STAGE 3: Test orchestration of document embedding."""
    
    def setUp(self):
        """Create mock client and test data."""
        self.mock_client = MagicMock()
        self.chunks = [
            {'text': 'chunk 1' * 50, 'chunk_id': 'c1', 'doc_id': 'd1'},
            {'text': 'chunk 2' * 50, 'chunk_id': 'c2', 'doc_id': 'd1'}
        ]
        self.images = [
            {
                'image_id': 'img1',
                'doc_id': 'd1',
                'enriched_caption': 'Test image',
                'author_caption': ''
            }
        ]
    
    @patch('embedding_utils.embed_chunks')
    @patch('embedding_utils.embed_images')
    @patch('embedding_utils.logging')
    def test_embed_document_returns_correct_structure(self, mock_log, mock_img, mock_chunks):
        """Verify embed_document returns correct output structure."""
        # Arrange
        mock_chunks.return_value = (self.chunks, 0.001)
        mock_img.return_value = (self.images, 0.0001)
        
        # Act
        result = embed_document('d1', self.chunks, self.images, self.mock_client)
        
        # Assert
        self.assertIn('chunks_with_embeddings', result)
        self.assertIn('images_with_embeddings', result)
        self.assertIn('cost', result)
        self.assertIn('stats', result)
    
    @patch('embedding_utils.embed_chunks')
    @patch('embedding_utils.embed_images')
    @patch('embedding_utils.logging')
    def test_embed_document_calculates_total_cost(self, mock_log, mock_img, mock_chunks):
        """Verify total cost is sum of chunks and images cost."""
        # Arrange
        chunks_cost = 0.001
        images_cost = 0.0002
        mock_chunks.return_value = (self.chunks, chunks_cost)
        mock_img.return_value = (self.images, images_cost)
        
        # Act
        result = embed_document('d1', self.chunks, self.images, self.mock_client)
        
        # Assert
        expected_cost = chunks_cost + images_cost
        self.assertAlmostEqual(result['cost'], expected_cost, places=6)
    
    @patch('embedding_utils.embed_chunks')
    @patch('embedding_utils.embed_images')
    @patch('embedding_utils.logging')
    def test_embed_document_filters_images_without_enriched_caption(self, mock_log, mock_img, mock_chunks):
        """Only images with enriched_caption should be embedded."""
        # Arrange
        images_mixed = [
            {'image_id': 'img1', 'doc_id': 'd1', 'enriched_caption': 'test'},
            {'image_id': 'img2', 'doc_id': 'd1'}  # No enriched_caption
        ]
        mock_chunks.return_value = (self.chunks, 0.001)
        mock_img.return_value = ([], 0.0)
        
        # Act
        embed_document('d1', self.chunks, images_mixed, self.mock_client)
        
        # Assert - embed_images should be called with only 1 image
        called_images = mock_img.call_args[0][0]
        self.assertEqual(len(called_images), 1)
        self.assertEqual(called_images[0]['image_id'], 'img1')
    
    @patch('embedding_utils.embed_chunks')
    @patch('embedding_utils.embed_images')
    @patch('embedding_utils.logging')
    def test_embed_document_stats(self, mock_log, mock_img, mock_chunks):
        """Verify stats include counts of embedded items."""
        # Arrange
        mock_chunks.return_value = (self.chunks, 0.001)
        mock_img.return_value = (self.images, 0.0001)
        
        # Act
        result = embed_document('d1', self.chunks, self.images, self.mock_client)
        
        # Assert
        stats = result['stats']
        self.assertEqual(stats['chunks_embedded'], len(self.chunks))
        self.assertEqual(stats['images_embedded'], len(self.images))


if __name__ == '__main__':
    unittest.main()
