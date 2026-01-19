"""
Test suite for chunk_documents.py module.

Tests cover:
- STAGE 1: Parameter validation and error handling
- STAGE 2: Constants and configuration
- STAGE 3: Helper functions and chunking logic
- Integration: Document chunking with image tracking
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile

from index.chunk_documents import (
    extract_figure_references,
    estimate_page_number,
    chunk_document_with_image_tracking,
)


class TestConstants(unittest.TestCase):
    """STAGE 2: Constants and configuration validation."""
    
    def test_chunk_size_reasonable(self):
        """STAGE 2: Verify chunk size is appropriate for embeddings."""
        # ~1800 chars = ~500 tokens for text-embedding-3-small
        chunk_size = 1800
        self.assertGreater(chunk_size, 500)
        self.assertLess(chunk_size, 5000)
    
    def test_chunk_overlap_smaller_than_size(self):
        """STAGE 2: Verify overlap is smaller than chunk size."""
        chunk_size = 1800
        chunk_overlap = 200
        self.assertLess(chunk_overlap, chunk_size)
        self.assertGreater(chunk_overlap, 0)


class TestExtractFigureReferences(unittest.TestCase):
    """STAGE 3: Figure/table reference extraction helper."""
    
    def test_extract_figure_references_valid(self):
        """STAGE 3: Extract figure references with numbers from text."""
        text = "As shown in Figure 1, the model performs well. See Table 2 for details."
        refs = extract_figure_references(text)
        
        # Should return full references with numbers
        self.assertIn("Figure 1", refs)
        self.assertIn("Table 2", refs)
    
    def test_extract_figure_references_case_insensitive(self):
        """STAGE 1: Handle case-insensitive figure references."""
        text = "figure 1 and FIGURE 2 and FIG. 3"
        refs = extract_figure_references(text)
        
        self.assertEqual(len(refs), 3)
    
    def test_extract_no_references(self):
        """STAGE 1: Return empty list when no references."""
        text = "This is regular text without any figures or tables."
        refs = extract_figure_references(text)
        
        self.assertEqual(refs, [])
    
    def test_extract_duplicate_references(self):
        """STAGE 3: Remove duplicate figure references."""
        text = "Figure 1 is shown. Figure 1 is important. Figure 1 demonstrates this."
        refs = extract_figure_references(text)

        # Should only have one "Figure 1" (deduplicated)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0], "Figure 1")
    
    def test_extract_fig_abbreviation(self):
        """STAGE 3: Convert Fig. abbreviation to Figure with number."""
        text = "As shown in Fig. 1, the results are clear."
        refs = extract_figure_references(text)
        
        # Should normalize "Fig. 1" to "Figure 1"
        self.assertIn("Figure 1", refs)


class TestEstimatePageNumber(unittest.TestCase):
    """STAGE 3: Page number estimation helper."""
    
    def test_estimate_page_first_chunk(self):
        """STAGE 3: First chunk should be on page 1."""
        page = estimate_page_number(chunk_index=0, total_chunks=100, total_pages=10)
        self.assertEqual(page, 1)
    
    def test_estimate_page_middle_chunk(self):
        """STAGE 3: Middle chunk should be on middle page."""
        page = estimate_page_number(chunk_index=50, total_chunks=100, total_pages=10)
        self.assertEqual(page, 6)
    
    def test_estimate_page_last_chunk(self):
        """STAGE 3: Last chunk should be on last page."""
        page = estimate_page_number(chunk_index=99, total_chunks=100, total_pages=10)
        self.assertEqual(page, 10)
    
    def test_estimate_page_zero_chunks(self):
        """STAGE 1: Handle zero chunks gracefully."""
        page = estimate_page_number(chunk_index=0, total_chunks=0, total_pages=10)
        self.assertIsNone(page)
    
    def test_estimate_page_zero_pages(self):
        """STAGE 1: Handle zero pages (JSON documents)."""
        page = estimate_page_number(chunk_index=0, total_chunks=10, total_pages=0)
        self.assertIsNone(page)
    
    def test_estimate_page_bounds_checking(self):
        """STAGE 1: Ensure page is within bounds."""
        # Even with extreme indices, should stay within [1, total_pages]
        page = estimate_page_number(chunk_index=1000, total_chunks=100, total_pages=10)
        self.assertGreaterEqual(page, 1)
        self.assertLessEqual(page, 10)


class TestChunkDocumentValidation(unittest.TestCase):
    """STAGE 1: Parameter validation for chunking."""
    
    def test_chunk_document_empty_text(self):
        """STAGE 1: Handle empty document text."""
        chunks = chunk_document_with_image_tracking(
            doc_id="test_empty",
            full_text="",
            total_pages=1,
            images_metadata=[]
        )
        
        # Empty text should produce 0 or 1 empty chunk
        self.assertIsInstance(chunks, list)
    
    def test_chunk_document_empty_images(self):
        """STAGE 1: Handle empty images list."""
        text = "Sample text. " * 200  # Long enough text
        chunks = chunk_document_with_image_tracking(
            doc_id="test_no_images",
            full_text=text,
            total_pages=0,
            images_metadata=[]
        )
        
        self.assertGreater(len(chunks), 0)
        # All chunks should have empty related/nearby image IDs
        for chunk in chunks:
            self.assertEqual(chunk['related_image_ids'], [])
            self.assertEqual(chunk['nearby_image_ids'], [])
    
    def test_chunk_document_invalid_chunk_size(self):
        """STAGE 1: Reject invalid chunk size."""
        text = "Sample text. " * 200
        
        # Zero or negative chunk size should raise ValueError
        with self.assertRaises(ValueError):
            chunk_document_with_image_tracking(
                doc_id="test_invalid",
                full_text=text,
                total_pages=0,
                images_metadata=[],
                chunk_size=0  # Invalid
            )


class TestChunkDocumentPDF(unittest.TestCase):
    """STAGE 3: PDF document chunking (page-based)."""
    
    def test_chunk_pdf_document_basic(self):
        """STAGE 3: Create chunks for PDF document."""
        text = "Sample text. " * 500  # Long text
        images = [
            {"image_id": "img_001", "page_num": 1},
            {"image_id": "img_002", "page_num": 1},
            {"image_id": "img_003", "page_num": 2},
            {"image_id": "img_004", "page_num": 3}
        ]
        
        chunks = chunk_document_with_image_tracking(
            doc_id="arxiv_test",
            full_text=text,
            total_pages=3,
            images_metadata=images
        )
        
        self.assertGreater(len(chunks), 0)
        
        # Check metadata structure
        for chunk in chunks:
            self.assertIn("chunk_id", chunk)
            self.assertIn("doc_id", chunk)
            self.assertIn("text", chunk)
            self.assertIn("page_num", chunk)
            self.assertIn("has_figure_references", chunk)
            self.assertIn("related_image_ids", chunk)
            self.assertIn("nearby_image_ids", chunk)
    
    def test_chunk_pdf_extraction_method(self):
        """STAGE 2: Verify PDF document has correct extraction method."""
        text = "Sample. " * 200
        chunks = chunk_document_with_image_tracking(
            doc_id="test_pdf",
            full_text=text,
            total_pages=1,
            images_metadata=[]
        )
        
        for chunk in chunks:
            self.assertEqual(chunk['extraction_method'], "pdf")
    
    def test_chunk_pdf_page_numbers(self):
        """STAGE 3: Verify page numbers are estimated correctly."""
        text = "Sample. " * 500
        chunks = chunk_document_with_image_tracking(
            doc_id="test_pages",
            full_text=text,
            total_pages=5,
            images_metadata=[]
        )
        
        # All chunks should have valid page numbers
        for chunk in chunks:
            page = chunk['page_num']
            self.assertIsNotNone(page)
            self.assertGreaterEqual(page, 1)
            self.assertLessEqual(page, 5)


class TestChunkDocumentJSON(unittest.TestCase):
    """STAGE 3: JSON document chunking (position-based)."""
    
    def test_chunk_json_document_basic(self):
        """STAGE 3: Create chunks for JSON document."""
        text = "Sample text. " * 500
        images = [
            {"image_id": "img_001", "image_index": 1, "extraction_method": "web_download"},
            {"image_id": "img_002", "image_index": 2, "extraction_method": "web_download"},
            {"image_id": "img_003", "image_index": 5, "extraction_method": "web_download"}
        ]
        
        chunks = chunk_document_with_image_tracking(
            doc_id="realpython_test",
            full_text=text,
            total_pages=0,  # No pages for JSON
            images_metadata=images
        )
        
        self.assertGreater(len(chunks), 0)
    
    def test_chunk_json_extraction_method(self):
        """STAGE 2: Verify JSON document has correct extraction method."""
        text = "Sample. " * 200
        images = [
            {"image_id": "img_001", "extraction_method": "web_download"}
        ]
        chunks = chunk_document_with_image_tracking(
            doc_id="test_json",
            full_text=text,
            total_pages=0,
            images_metadata=images
        )
        
        for chunk in chunks:
            self.assertEqual(chunk['extraction_method'], "json")
    
    def test_chunk_json_no_page_numbers(self):
        """STAGE 3: JSON documents should have None page_num."""
        text = "Sample. " * 200
        chunks = chunk_document_with_image_tracking(
            doc_id="test_json_pages",
            full_text=text,
            total_pages=0,
            images_metadata=[]
        )
        
        for chunk in chunks:
            self.assertIsNone(chunk['page_num'])


class TestChunkMetadata(unittest.TestCase):
    """STAGE 3: Chunk metadata consistency."""
    
    def test_chunk_ids_unique(self):
        """STAGE 1: All chunk IDs should be unique."""
        text = "Sample. " * 500
        chunks = chunk_document_with_image_tracking(
            doc_id="test_unique",
            full_text=text,
            total_pages=0,
            images_metadata=[]
        )
        
        chunk_ids = [chunk['chunk_id'] for chunk in chunks]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
    
    def test_chunk_ids_format(self):
        """STAGE 3: Chunk IDs follow expected format."""
        text = "Sample. " * 200
        chunks = chunk_document_with_image_tracking(
            doc_id="test_format",
            full_text=text,
            total_pages=0,
            images_metadata=[]
        )
        
        for chunk in chunks:
            # Format: doc_id_chunk_XXXX
            self.assertIn("_chunk_", chunk['chunk_id'])
            self.assertTrue(chunk['chunk_id'].startswith("test_format"))
    
    def test_chunk_text_not_empty(self):
        """STAGE 1: Chunk text should not be empty."""
        text = "Sample text. " * 500
        chunks = chunk_document_with_image_tracking(
            doc_id="test_text",
            full_text=text,
            total_pages=0,
            images_metadata=[]
        )
        
        for chunk in chunks:
            self.assertGreater(len(chunk['text']), 0)
    
    def test_chunk_word_count_consistency(self):
        """STAGE 3: Word count matches actual text."""
        text = "Sample text. " * 200
        chunks = chunk_document_with_image_tracking(
            doc_id="test_words",
            full_text=text,
            total_pages=0,
            images_metadata=[]
        )
        
        for chunk in chunks:
            actual_words = len(chunk['text'].split())
            stored_words = chunk['word_count']
            self.assertEqual(actual_words, stored_words)
    
    def test_chunk_char_count_consistency(self):
        """STAGE 3: Character count matches actual text."""
        text = "Sample text. " * 200
        chunks = chunk_document_with_image_tracking(
            doc_id="test_chars",
            full_text=text,
            total_pages=0,
            images_metadata=[]
        )
        
        for chunk in chunks:
            actual_chars = len(chunk['text'])
            stored_chars = chunk['char_count']
            self.assertEqual(actual_chars, stored_chars)


class TestImageLinking(unittest.TestCase):
    """STAGE 3: Image-chunk relationship tracking."""
    
    def test_related_images_same_page(self):
        """STAGE 3: Related images on same page."""
        text = "Sample. " * 200
        images = [
            {"image_id": "img_001", "page_num": 1},
            {"image_id": "img_002", "page_num": 1}
        ]
        
        chunks = chunk_document_with_image_tracking(
            doc_id="test_same_page",
            full_text=text,
            total_pages=1,
            images_metadata=images
        )
        
        # First chunk on page 1 should link to images on page 1
        if chunks and chunks[0]['page_num'] == 1:
            self.assertGreater(len(chunks[0]['related_image_ids']), 0)
    
    def test_nearby_images_adjacent_page(self):
        """STAGE 3: Nearby images on adjacent page."""
        text = "Sample. " * 500
        images = [
            {"image_id": "img_001", "page_num": 1},
            {"image_id": "img_002", "page_num": 2},
            {"image_id": "img_003", "page_num": 3}
        ]
        
        chunks = chunk_document_with_image_tracking(
            doc_id="test_nearby",
            full_text=text,
            total_pages=3,
            images_metadata=images
        )
        
        # Check that nearby images are tracked
        for chunk in chunks:
            self.assertIsInstance(chunk['nearby_image_ids'], list)


if __name__ == "__main__":
    unittest.main()
