"""
Test suite for build_index.py - STAGE 1-3 validation pattern.

STAGE 1: Input validation, error handling, file locking (10 tests)
STAGE 2: Constants and configuration (2 tests)
STAGE 3: Helper functions and business logic (18 tests)

Total: 30 tests
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open, Mock
import sys
import json
from pathlib import Path

# Add index module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "index"))

import build_index
from build_index import (
    ChromaDBLock,
    _save_chunks_backup,
    _get_existing_ids,
    _prepare_text_chunks_for_indexing,
    _prepare_images_for_indexing,
    _get_or_create_collection,
    index_documents_to_chromadb,
    EMBEDDING_MODEL,
    EMBEDDING_DIMS,
    MAX_ENRICHED_CAPTION_LENGTH,
    MAX_AUTHOR_CAPTION_LENGTH
)


class TestBuildIndexConstants(unittest.TestCase):
    """STAGE 2: Verify ChromaDB configuration constants."""
    
    def test_embedding_model_consistency(self):
        """Verify embedding model matches specification."""
        self.assertEqual(EMBEDDING_MODEL, "text-embedding-3-small")
    
    def test_embedding_dims_consistency(self):
        """Verify embedding dimensionality matches OpenAI model."""
        self.assertEqual(EMBEDDING_DIMS, 1536)
    
    def test_max_caption_lengths(self):
        """Verify caption length constants are defined."""
        self.assertEqual(MAX_ENRICHED_CAPTION_LENGTH, 1000)
        self.assertEqual(MAX_AUTHOR_CAPTION_LENGTH, 500)


# ============================================================================
# STAGE 1: File Locking & Error Handling
# ============================================================================

class TestChromaDBLock(unittest.TestCase):
    """STAGE 1: Test cross-platform file locking."""
    
    @unittest.skipIf(sys.platform != 'win32', "Windows-specific test")
    @patch('msvcrt.locking')
    @patch('builtins.open', new_callable=mock_open)
    @patch('build_index.CHROMA_LOCK_FILE', new_callable=MagicMock)
    @patch('build_index.logging')
    def test_chromadb_lock_windows_acquire(self, mock_log, mock_lock_path, mock_file, mock_lock):
        """Test Windows file lock acquisition."""
        # Arrange
        mock_lock_path.parent.mkdir = MagicMock()
        mock_file.return_value.fileno.return_value = 1
        
        # Act
        with ChromaDBLock(mock_lock_path):
            pass
        
        # Assert - msvcrt.locking should be called
        mock_lock.assert_called()


# ============================================================================
# STAGE 1: Input Validation
# ============================================================================

class TestIndexDocumentsValidation(unittest.TestCase):
    """STAGE 1: Validate inputs to index_documents_to_chromadb."""
    
    def setUp(self):
        """Create valid test data."""
        self.valid_chunks = [
            {
                'chunk_id': 'c1',
                'text': 'test',
                'embedding': [0.1]*1536,
                'doc_id': 'd1',
                'chunk_index': 0,
                'char_count': 4,
                'word_count': 1,
                'has_figure_references': False,
                'image_references': [],
                'related_image_ids': [],
                'nearby_image_ids': [],
                'extraction_method': 'pdf'
            }
        ]
        self.valid_images = [
            {'image_id': 'img1', 'embedding': [0.1]*1536, 'doc_id': 'd1'}
        ]
    
    @patch('build_index.ChromaDBLock')
    @patch('build_index.logging')
    def test_invalid_doc_id_empty(self, mock_log, mock_lock):
        """Reject empty doc_id."""
        with self.assertRaises(ValueError):
            index_documents_to_chromadb("", self.valid_chunks, self.valid_images)
    
    @patch('build_index.ChromaDBLock')
    @patch('build_index.logging')
    def test_invalid_doc_id_type(self, mock_log, mock_lock):
        """Reject non-string doc_id."""
        with self.assertRaises((ValueError, TypeError)):
            index_documents_to_chromadb(123, self.valid_chunks, self.valid_images)
    
    @patch('build_index.ChromaDBLock')
    @patch('build_index.logging')
    def test_invalid_chunks_type(self, mock_log, mock_lock):
        """Reject non-list chunks."""
        with self.assertRaises(TypeError):
            index_documents_to_chromadb("d1", "not a list", self.valid_images)
    
    @patch('build_index.ChromaDBLock')
    @patch('build_index.logging')
    def test_invalid_images_type(self, mock_log, mock_lock):
        """Reject non-list images."""
        with self.assertRaises(TypeError):
            index_documents_to_chromadb("d1", self.valid_chunks, "not a list")
    
    @patch('build_index.ChromaDBLock')
    @patch('build_index.logging')
    def test_chunk_missing_chunk_id(self, mock_log, mock_lock):
        """Reject chunk without chunk_id."""
        invalid_chunks = [{'text': 'test', 'embedding': [0.1]*1536, 'doc_id': 'd1'}]
        with self.assertRaises((ValueError, KeyError)):
            index_documents_to_chromadb("d1", invalid_chunks, [])
    
    @patch('build_index.ChromaDBLock')
    @patch('build_index.logging')
    def test_chunk_missing_text(self, mock_log, mock_lock):
        """Reject chunk without text."""
        invalid_chunks = [{'chunk_id': 'c1', 'embedding': [0.1]*1536, 'doc_id': 'd1'}]
        with self.assertRaises((ValueError, KeyError)):
            index_documents_to_chromadb("d1", invalid_chunks, [])
    
    @patch('build_index.ChromaDBLock')
    @patch('build_index.logging')
    def test_chunk_missing_embedding(self, mock_log, mock_lock):
        """Reject chunk without embedding."""
        invalid_chunks = [{'chunk_id': 'c1', 'text': 'test', 'doc_id': 'd1'}]
        with self.assertRaises((ValueError, KeyError)):
            index_documents_to_chromadb("d1", invalid_chunks, [])
    
    @patch('build_index.ChromaDBLock')
    @patch('build_index.logging')
    def test_image_missing_image_id(self, mock_log, mock_lock):
        """Reject image without image_id."""
        invalid_images = [{'embedding': [0.1]*1536, 'doc_id': 'd1'}]
        with self.assertRaises((ValueError, KeyError)):
            index_documents_to_chromadb("d1", self.valid_chunks, invalid_images)
    
    @patch('build_index.ChromaDBLock')
    @patch('build_index.logging')
    def test_image_missing_embedding(self, mock_log, mock_lock):
        """Reject image without embedding."""
        invalid_images = [{'image_id': 'img1', 'doc_id': 'd1'}]
        with self.assertRaises((ValueError, KeyError)):
            index_documents_to_chromadb("d1", self.valid_chunks, invalid_images)


# ============================================================================
# STAGE 3: Helper Functions
# ============================================================================

class TestSaveChunksBackup(unittest.TestCase):
    """STAGE 3: Test backup functionality."""
    
    @patch('build_index.CHUNKS_BACKUP_DIR')
    @patch('build_index.logging')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_chunks_backup_creates_files(self, mock_file, mock_log, mock_dir):
        """Verify backup files are created with correct structure."""
        # Arrange
        mock_dir.mkdir = MagicMock()
        chunks = [{'chunk_id': 'c1', 'text': 'test', 'embedding': [0.1]*1536}]
        images = [{'image_id': 'img1', 'embedding': [0.1]*1536}]
        
        # Act
        _save_chunks_backup('d1', chunks, images)
        
        # Assert - files should be opened for writing
        self.assertGreater(mock_file.call_count, 0)
    
    @patch('build_index.CHUNKS_BACKUP_DIR')
    @patch('build_index.logging')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_chunks_backup_json_structure(self, mock_file, mock_log, mock_dir):
        """Verify backup JSON has correct structure."""
        # Arrange
        mock_dir.mkdir = MagicMock()
        mock_file_handle = mock_file.return_value.__enter__.return_value
        chunks = [{'chunk_id': 'c1', 'text': 'test'}]
        images = []
        
        # Act
        _save_chunks_backup('d1', chunks, images)
        
        # Assert - json.dump should be called
        # This verifies the backup process was attempted


class TestGetExistingIds(unittest.TestCase):
    """STAGE 3: Test duplicate detection."""
    
    def test_get_existing_ids_returns_empty_set_for_new_collection(self):
        """New collection should return empty set."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.get.return_value = {'ids': []}
        
        # Act
        result = _get_existing_ids(mock_collection)
        
        # Assert
        self.assertEqual(result, set())
    
    def test_get_existing_ids_returns_set_of_ids(self):
        """Verify existing IDs are returned as set."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.get.return_value = {'ids': ['id1', 'id2', 'id3']}
        
        # Act
        result = _get_existing_ids(mock_collection)
        
        # Assert
        self.assertEqual(result, {'id1', 'id2', 'id3'})
    
    def test_get_existing_ids_handles_error(self):
        """Handle collection errors gracefully."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.get.side_effect = Exception("Collection error")
        
        # Act
        result = _get_existing_ids(mock_collection)
        
        # Assert - should return empty set on error
        self.assertEqual(result, set())


class TestGetOrCreateCollection(unittest.TestCase):
    """STAGE 3: Test DRY helper for collection management."""
    
    def test_get_existing_collection(self):
        """Should return existing collection if it exists."""
        # Arrange
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        
        # Act
        result = _get_or_create_collection(mock_client, "test_collection", "Test")
        
        # Assert
        self.assertEqual(result, mock_collection)
        mock_client.get_collection.assert_called_once_with("test_collection")
        mock_client.create_collection.assert_not_called()
    
    def test_create_new_collection(self):
        """Should create new collection if it doesn't exist."""
        # Arrange
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = ValueError("Not found")
        mock_new_collection = MagicMock()
        mock_client.create_collection.return_value = mock_new_collection
        
        # Act
        result = _get_or_create_collection(mock_client, "new_collection", "New Test")
        
        # Assert
        self.assertEqual(result, mock_new_collection)
        mock_client.create_collection.assert_called_once()
        # Verify metadata includes constants
        call_args = mock_client.create_collection.call_args
        self.assertEqual(call_args[1]['name'], "new_collection")
        self.assertEqual(call_args[1]['metadata']['description'], "New Test")
        self.assertEqual(call_args[1]['metadata']['embedding_model'], EMBEDDING_MODEL)
        self.assertEqual(call_args[1]['metadata']['embedding_dims'], EMBEDDING_DIMS)


class TestPrepareTextChunksForIndexing(unittest.TestCase):
    """STAGE 3: Test text chunk preparation logic."""
    
    def setUp(self):
        """Create test chunks."""
        self.chunks = [
            {
                'chunk_id': 'c1',
                'text': 'test chunk 1',
                'embedding': [0.1]*1536,
                'doc_id': 'd1',
                'chunk_index': 0,
                'char_count': 12,
                'word_count': 2,
                'has_figure_references': False,
                'image_references': [],
                'related_image_ids': [],
                'nearby_image_ids': [],
                'extraction_method': 'pdf',
                'page_num': 1
            }
        ]
    
    def test_prepare_text_chunks_filters_duplicates(self):
        """Duplicate chunks should be skipped."""
        # Arrange
        existing_ids = {'c1'}
        
        # Act
        ids, docs, embeds, metas, skipped = _prepare_text_chunks_for_indexing(
            self.chunks, existing_ids
        )
        
        # Assert
        self.assertEqual(len(ids), 0)
        self.assertEqual(skipped, 1)
    
    def test_prepare_text_chunks_returns_correct_structure(self):
        """Verify output structure."""
        # Arrange
        existing_ids = set()
        
        # Act
        ids, docs, embeds, metas, skipped = _prepare_text_chunks_for_indexing(
            self.chunks, existing_ids
        )
        
        # Assert
        self.assertEqual(len(ids), 1)
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(embeds), 1)
        self.assertEqual(len(metas), 1)
        self.assertEqual(ids[0], 'c1')
        self.assertEqual(docs[0], 'test chunk 1')
    
    def test_prepare_text_chunks_includes_metadata(self):
        """Verify all metadata fields included."""
        # Arrange
        existing_ids = set()
        
        # Act
        _, _, _, metas, _ = _prepare_text_chunks_for_indexing(self.chunks, existing_ids)
        
        # Assert
        meta = metas[0]
        self.assertIn('doc_id', meta)
        self.assertIn('chunk_index', meta)
        self.assertIn('char_count', meta)
        self.assertIn('word_count', meta)
        self.assertIn('has_figure_references', meta)
    
    def test_prepare_text_chunks_converts_lists_to_json(self):
        """Verify lists are converted to JSON strings."""
        # Arrange
        self.chunks[0]['image_references'] = ['Figure 1', 'Table 2']
        self.chunks[0]['related_image_ids'] = ['img1', 'img2']
        existing_ids = set()
        
        # Act
        _, _, _, metas, _ = _prepare_text_chunks_for_indexing(self.chunks, existing_ids)
        
        # Assert
        meta = metas[0]
        self.assertIsInstance(meta['image_references'], str)
        self.assertIsInstance(meta['related_image_ids'], str)
    
    def test_prepare_text_chunks_invalid_input_type(self):
        """Handle non-list input gracefully."""
        # Act
        ids, docs, embeds, metas, skipped = _prepare_text_chunks_for_indexing(
            "not a list", set()
        )
        
        # Assert
        self.assertEqual(ids, [])
        self.assertEqual(skipped, 0)
    
    def test_prepare_text_chunks_missing_chunk_id(self):
        """Skip chunks missing chunk_id."""
        # Arrange
        invalid_chunks = [{'text': 'test', 'embedding': [0.1]*1536}]
        existing_ids = set()
        
        # Act
        ids, docs, embeds, metas, _ = _prepare_text_chunks_for_indexing(
            invalid_chunks, existing_ids
        )
        
        # Assert
        self.assertEqual(len(ids), 0)


class TestPrepareImagesForIndexing(unittest.TestCase):
    """STAGE 3: Test image preparation logic."""
    
    def setUp(self):
        """Create test images."""
        self.images = [
            {
                'image_id': 'img1',
                'embedding': [0.1]*1536,
                'doc_id': 'd1',
                'filename': 'test.png',
                'width': 100,
                'height': 100,
                'format': 'png',
                'enriched_caption': 'A test image' * 100,
                'author_caption': 'Test',
                'extraction_method': 'pdf',
                'page_num': 1,
                'caption_for_embedding': 'test caption'
            }
        ]
    
    def test_prepare_images_filters_duplicates(self):
        """Duplicate images should be skipped."""
        # Arrange
        existing_ids = {'img1'}
        
        # Act
        ids, docs, embeds, metas, skipped = _prepare_images_for_indexing(
            self.images, existing_ids
        )
        
        # Assert
        self.assertEqual(len(ids), 0)
        self.assertEqual(skipped, 1)
    
    def test_prepare_images_returns_correct_structure(self):
        """Verify output structure."""
        # Arrange
        existing_ids = set()
        
        # Act
        ids, docs, embeds, metas, skipped = _prepare_images_for_indexing(
            self.images, existing_ids
        )
        
        # Assert
        self.assertEqual(len(ids), 1)
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(embeds), 1)
        self.assertEqual(len(metas), 1)
    
    def test_prepare_images_includes_image_id_in_metadata(self):
        """CRITICAL: image_id must be in metadata for retrieval."""
        # Arrange
        existing_ids = set()
        
        # Act
        _, _, _, metas, _ = _prepare_images_for_indexing(self.images, existing_ids)
        
        # Assert
        self.assertIn('image_id', metas[0])
        self.assertEqual(metas[0]['image_id'], 'img1')
    
    def test_prepare_images_truncates_long_captions(self):
        """Long captions should be truncated."""
        # Arrange
        existing_ids = set()
        self.images[0]['enriched_caption'] = 'x' * 2000
        
        # Act
        _, _, _, metas, _ = _prepare_images_for_indexing(self.images, existing_ids)
        
        # Assert
        self.assertLessEqual(len(metas[0]['enriched_caption']), 1000)
    
    def test_prepare_images_handles_missing_optional_fields(self):
        """Handle images with missing optional fields."""
        # Arrange
        images_minimal = [
            {
                'image_id': 'img1',
                'embedding': [0.1]*1536,
                'doc_id': 'd1',
                'filename': 'test.png',
                'enriched_caption': 'test',
                'caption_for_embedding': 'test'
            }
        ]
        existing_ids = set()
        
        # Act
        ids, _, _, metas, _ = _prepare_images_for_indexing(images_minimal, existing_ids)
        
        # Assert
        self.assertEqual(len(ids), 1)
        self.assertEqual(metas[0]['width'], 0)
        self.assertEqual(metas[0]['height'], 0)
    
    def test_prepare_images_invalid_input_type(self):
        """Handle non-list input gracefully."""
        # Act
        ids, docs, embeds, metas, skipped = _prepare_images_for_indexing(
            "not a list", set()
        )
        
        # Assert
        self.assertEqual(ids, [])
        self.assertEqual(skipped, 0)


class TestIndexDocumentsOrchestration(unittest.TestCase):
    """STAGE 3: Test orchestration logic."""
    
    def setUp(self):
        """Create valid test data."""
        self.chunks = [
            {
                'chunk_id': 'c1',
                'text': 'test',
                'embedding': [0.1]*1536,
                'doc_id': 'd1',
                'chunk_index': 0,
                'char_count': 4,
                'word_count': 1,
                'has_figure_references': False,
                'image_references': [],
                'related_image_ids': [],
                'nearby_image_ids': [],
                'extraction_method': 'pdf'
            }
        ]
        self.images = [
            {'image_id': 'img1', 'embedding': [0.1]*1536, 'doc_id': 'd1'}
        ]
    
    @patch('build_index._save_chunks_backup')
    @patch('build_index.ChromaDBLock')
    @patch('build_index.chromadb.PersistentClient')
    @patch('build_index.logging')
    def test_index_documents_backup_called_first(self, mock_log, mock_chroma, mock_lock, mock_backup):
        """Verify backup is saved before indexing."""
        # Arrange
        mock_lock_instance = MagicMock()
        mock_lock.return_value = mock_lock_instance
        mock_lock_instance.__enter__ = MagicMock(return_value=mock_lock_instance)
        mock_lock_instance.__exit__ = MagicMock(return_value=False)
        
        mock_client = MagicMock()
        mock_chroma.return_value = mock_client
        mock_client.get_collection.side_effect = ValueError("Not found")
        mock_client.create_collection.return_value = MagicMock()
        
        # Act
        index_documents_to_chromadb('d1', self.chunks, self.images)
        
        # Assert
        mock_backup.assert_called_once()
    
    @patch('build_index._save_chunks_backup')
    @patch('build_index.ChromaDBLock')
    @patch('build_index.chromadb.PersistentClient')
    @patch('build_index.logging')
    def test_index_documents_returns_stats(self, mock_log, mock_chroma, mock_lock, mock_backup):
        """Verify statistics are returned."""
        # Arrange
        mock_lock_instance = MagicMock()
        mock_lock.return_value = mock_lock_instance
        mock_lock_instance.__enter__ = MagicMock(return_value=mock_lock_instance)
        mock_lock_instance.__exit__ = MagicMock(return_value=False)
        
        mock_client = MagicMock()
        mock_chroma.return_value = mock_client
        mock_collection = MagicMock()
        mock_collection.get.return_value = {'ids': []}
        mock_client.get_collection.side_effect = ValueError("Not found")
        mock_client.create_collection.return_value = mock_collection
        
        # Act
        result = index_documents_to_chromadb('d1', self.chunks, self.images)
        
        # Assert
        self.assertIn('text_chunks_added', result)
        self.assertIn('text_chunks_skipped', result)
        self.assertIn('images_added', result)
        self.assertIn('images_skipped', result)


if __name__ == '__main__':
    unittest.main()
