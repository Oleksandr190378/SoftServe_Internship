"""
Test suite for download_arxiv.py module.

Tests cover:
- STAGE 1: Parameter validation and error handling
- STAGE 2: Constants and configuration
- STAGE 3: Helper functions and metadata creation
- Integration: Download workflows
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
from datetime import datetime
import tempfile

from ingest.download_arxiv import (
    _create_paper_metadata,
    _download_and_save_pdf,
    download_curated_papers_by_ids,
    download_curated_papers,
    search_and_download_papers,
    download_papers,
    DEFAULT_CATEGORIES,
    DEFAULT_NUM_PAPERS,
    DOWNLOAD_DELAY_SECONDS,
    CURATED_PAPERS
)


class TestConstants(unittest.TestCase):
    """STAGE 2: Constants validation."""
    
    def test_default_categories_list(self):
        """Verify default categories are configured."""
        self.assertIsInstance(DEFAULT_CATEGORIES, list)
        self.assertGreater(len(DEFAULT_CATEGORIES), 0)
        self.assertIn("cs.LG", DEFAULT_CATEGORIES)
    
    def test_default_num_papers_positive(self):
        """Verify default num papers is positive."""
        self.assertGreater(DEFAULT_NUM_PAPERS, 0)
    
    def test_download_delay_positive(self):
        """Verify download delay is reasonable."""
        self.assertGreater(DOWNLOAD_DELAY_SECONDS, 0)
        self.assertLess(DOWNLOAD_DELAY_SECONDS, 10)
    
    def test_curated_papers_populated(self):
        """Verify curated papers list is populated."""
        self.assertIsInstance(CURATED_PAPERS, list)
        self.assertGreater(len(CURATED_PAPERS), 0)


class TestCreatePaperMetadata(unittest.TestCase):
    """STAGE 3: Paper metadata creation helper."""
    
    def setUp(self):
        """Create mock paper object."""
        self.mock_paper = MagicMock()
        self.mock_paper.get_short_id.return_value = "1706.03762"
        self.mock_paper.title = "Attention Is All You Need"
        self.mock_paper.authors = [
            MagicMock(name="Vaswani"),
            MagicMock(name="Shazeer")
        ]
        self.mock_paper.summary = "Transformer architecture..."
        self.mock_paper.published = datetime(2017, 6, 12)
        self.mock_paper.categories = ["cs.CL", "cs.LG"]
        self.mock_paper.pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
    
    def test_create_metadata_valid_paper(self):
        """STAGE 3: Create metadata for valid paper."""
        metadata = _create_paper_metadata(self.mock_paper, "1706.03762")
        
        self.assertIn("doc_id", metadata)
        self.assertTrue(metadata["doc_id"].startswith("arxiv_"))
        self.assertIn("arxiv_id", metadata)
        self.assertEqual(metadata["title"], "Attention Is All You Need")
        self.assertEqual(len(metadata["authors"]), 2)
    
    def test_create_metadata_safe_id_format(self):
        """STAGE 1: Verify doc_id uses safe format."""
        metadata = _create_paper_metadata(self.mock_paper, "1706.03762")
        
        # Should use underscores, not dots
        self.assertIn("_", metadata["doc_id"])
        self.assertNotIn(".", metadata["doc_id"])
    
    def test_create_metadata_includes_timestamp(self):
        """STAGE 3: Verify metadata includes download timestamp."""
        metadata = _create_paper_metadata(self.mock_paper, "1706.03762")
        
        self.assertIn("downloaded_at", metadata)
        self.assertIsInstance(metadata["downloaded_at"], str)
    
    def test_create_metadata_source_type(self):
        """STAGE 3: Verify source_type is arxiv."""
        metadata = _create_paper_metadata(self.mock_paper, "1706.03762")
        
        self.assertEqual(metadata["source_type"], "arxiv")


class TestDownloadAndSavePdf(unittest.TestCase):
    """STAGE 3: PDF download and save helper."""
    
    def setUp(self):
        """Create mock objects for PDF download."""
        self.mock_paper = MagicMock()
        self.mock_paper.title = "Test Paper"
        
        self.metadata = {
            "doc_id": "arxiv_1706_03762",
            "title": "Test Paper"
        }
    
    @patch('ingest.download_arxiv.Path.exists')
    def test_pdf_already_exists(self, mock_exists):
        """STAGE 1: Handle already existing PDF."""
        mock_exists.return_value = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = _download_and_save_pdf(
                self.mock_paper, self.metadata, output_dir, 1, 5
            )
        
        self.assertTrue(result)
        self.assertIn("pdf_path", self.metadata)  # Always added
    
    @patch('ingest.download_arxiv.time.sleep')
    @patch('ingest.download_arxiv.Path.exists')
    def test_pdf_download_success(self, mock_exists, mock_sleep):
        """STAGE 3: Successfully download PDF."""
        mock_exists.return_value = False
        self.mock_paper.download_pdf = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = _download_and_save_pdf(
                self.mock_paper, self.metadata, output_dir, 1, 5
            )
        
        self.assertTrue(result)
        mock_sleep.assert_called()
    
    @patch('ingest.download_arxiv.Path.exists')
    def test_pdf_download_failure_handling(self, mock_exists):
        """STAGE 1: Handle PDF download errors."""
        mock_exists.return_value = False
        self.mock_paper.download_pdf = MagicMock(side_effect=Exception("Network error"))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = _download_and_save_pdf(
                self.mock_paper, self.metadata, output_dir, 1, 5
            )
        
        self.assertFalse(result)


class TestDownloadCuratedPapersById(unittest.TestCase):
    """STAGE 1-3: Download papers by specific IDs."""
    
    def setUp(self):
        """Create mock paper object."""
        self.mock_paper = MagicMock()
        self.mock_paper.title = "Test Paper"
        self.mock_paper.authors = [MagicMock(name="Author")]
        self.mock_paper.summary = "Abstract..."
        self.mock_paper.published = datetime(2017, 1, 1)
        self.mock_paper.categories = ["cs.LG"]
        self.mock_paper.pdf_url = "https://arxiv.org/pdf/test.pdf"
        self.mock_paper.get_short_id.return_value = "1706.03762"
    
    def test_empty_paper_ids_raises_error(self):
        """STAGE 1: Reject empty paper IDs list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                download_curated_papers_by_ids([], Path(tmpdir))
    
    @patch('ingest.download_arxiv.arxiv.Search')
    @patch('ingest.download_arxiv._download_and_save_pdf')
    def test_download_valid_papers(self, mock_save, mock_search):
        """STAGE 3: Download valid papers."""
        mock_search_instance = MagicMock()
        mock_search_instance.results.return_value = iter([self.mock_paper])
        mock_search.return_value = mock_search_instance
        mock_save.return_value = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            papers = download_curated_papers_by_ids(["1706.03762"], Path(tmpdir))
        
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["title"], "Test Paper")
    
    @patch('ingest.download_arxiv.arxiv.Search')
    def test_handle_paper_not_found(self, mock_search):
        """STAGE 1: Handle paper not found error."""
        mock_search_instance = MagicMock()
        mock_search_instance.results.return_value = iter([])  # Empty iterator
        mock_search.return_value = mock_search_instance
        
        with tempfile.TemporaryDirectory() as tmpdir:
            papers = download_curated_papers_by_ids(["nonexistent"], Path(tmpdir))
        
        self.assertEqual(len(papers), 0)


class TestDownloadCuratedPapers(unittest.TestCase):
    """STAGE 1-3: Download curated papers."""
    
    @patch('ingest.download_arxiv.download_curated_papers_by_ids')
    def test_curated_papers_num_limit(self, mock_download):
        """STAGE 1: Respect num_papers limit."""
        mock_download.return_value = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            download_curated_papers(Path(tmpdir), num_papers=5)
        
        mock_download.assert_called_once()
        call_args = mock_download.call_args[0][0]
        self.assertEqual(len(call_args), 5)
    
    def test_curated_papers_invalid_num(self):
        """STAGE 1: Reject invalid num_papers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                download_curated_papers(Path(tmpdir), num_papers=0)
            
            with self.assertRaises(ValueError):
                download_curated_papers(Path(tmpdir), num_papers=-1)


class TestSearchAndDownloadPapers(unittest.TestCase):
    """STAGE 1-3: Search and download papers."""
    
    def test_empty_query_raises_error(self):
        """STAGE 1: Reject empty query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                search_and_download_papers("", ["cs.LG"], Path(tmpdir))
    
    def test_empty_categories_raises_error(self):
        """STAGE 1: Reject empty categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                search_and_download_papers("neural network", [], Path(tmpdir))
    
    def test_invalid_max_results_raises_error(self):
        """STAGE 1: Reject invalid max_results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                search_and_download_papers("test", ["cs.LG"], Path(tmpdir), max_results=0)
    
    @patch('ingest.download_arxiv.arxiv.Search')
    def test_search_with_valid_params(self, mock_search):
        """STAGE 3: Execute search with valid parameters."""
        mock_search_instance = MagicMock()
        mock_search_instance.results.return_value = []
        mock_search.return_value = mock_search_instance
        
        with tempfile.TemporaryDirectory() as tmpdir:
            search_and_download_papers("neural network", ["cs.LG"], Path(tmpdir), max_results=5)
        
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        self.assertEqual(call_kwargs["max_results"], 5)


class TestDownloadPapers(unittest.TestCase):
    """STAGE 1-3: Main download API."""
    
    def test_invalid_mode_raises_error(self):
        """STAGE 1: Reject invalid mode."""
        with self.assertRaises(ValueError):
            download_papers(mode="invalid")
    
    def test_invalid_max_papers_raises_error(self):
        """STAGE 1: Reject invalid max_papers."""
        with self.assertRaises(ValueError):
            download_papers(max_papers=0)
        
        with self.assertRaises(ValueError):
            download_papers(max_papers=-5)
    
    @patch('ingest.download_arxiv.download_curated_papers')
    def test_curated_mode_default(self, mock_curated):
        """STAGE 3: Use curated mode by default."""
        mock_curated.return_value = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            download_papers(output_dir=Path(tmpdir), max_papers=5)
        
        mock_curated.assert_called_once()
    
    @patch('ingest.download_arxiv.download_curated_papers_by_ids')
    def test_curated_mode_with_specific_ids(self, mock_download):
        """STAGE 3: Download specific paper IDs in curated mode."""
        mock_download.return_value = []
        paper_ids = ["1706.03762", "1512.03385"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            download_papers(
                paper_ids=paper_ids,
                mode="curated",
                output_dir=Path(tmpdir)
            )
        
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        self.assertEqual(call_kwargs['paper_ids'], paper_ids)
    
    @patch('ingest.download_arxiv.search_and_download_papers')
    def test_search_mode(self, mock_search):
        """STAGE 3: Use search mode when specified."""
        mock_search.return_value = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            download_papers(
                mode="search",
                query="transformer",
                output_dir=Path(tmpdir),
                max_papers=5
            )
        
        mock_search.assert_called_once()
    
    @patch('ingest.download_arxiv.download_curated_papers')
    def test_returns_doc_ids_list(self, mock_curated):
        """STAGE 3: Return list of doc_ids."""
        mock_curated.return_value = [
            {"doc_id": "arxiv_1706_03762", "title": "Paper 1"},
            {"doc_id": "arxiv_1512_03385", "title": "Paper 2"}
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_ids = download_papers(output_dir=Path(tmpdir), max_papers=2)
        
        self.assertEqual(len(doc_ids), 2)
        self.assertIn("arxiv_1706_03762", doc_ids)


class TestDownloadPapersIntegration(unittest.TestCase):
    """Integration tests for download workflow."""
    
    @patch('ingest.download_arxiv._download_and_save_pdf')
    @patch('ingest.download_arxiv.arxiv.Search')
    def test_full_download_workflow(self, mock_search, mock_save):
        """STAGE 3: Test complete download workflow."""
        # Setup mock paper
        mock_paper = MagicMock()
        mock_paper.title = "Test Paper"
        mock_paper.authors = [MagicMock(name="Author")]
        mock_paper.summary = "Abstract"
        mock_paper.published = datetime(2017, 1, 1)
        mock_paper.categories = ["cs.LG"]
        mock_paper.pdf_url = "https://arxiv.org/pdf/test.pdf"
        mock_paper.get_short_id.return_value = "1706.03762"
        
        mock_search_instance = MagicMock()
        mock_search_instance.results.return_value = iter([mock_paper])
        mock_search.return_value = mock_search_instance
        mock_save.return_value = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_ids = download_papers(
                paper_ids=["1706.03762"],
                mode="curated",
                output_dir=Path(tmpdir)
            )
        
        self.assertGreater(len(doc_ids), 0)
        self.assertTrue(doc_ids[0].startswith("arxiv_"))


class TestErrorHandling(unittest.TestCase):
    """STAGE 1: Error handling and edge cases."""
    
    @patch('ingest.download_arxiv.arxiv.Search')
    def test_arxiv_api_error_handling(self, mock_search):
        """STAGE 1: Handle arXiv API errors gracefully."""
        mock_search.side_effect = Exception("API Error")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise, but return empty list
            papers = download_curated_papers_by_ids(["1706.03762"], Path(tmpdir))
        
        self.assertEqual(len(papers), 0)
    
    def test_metadata_none_handling(self):
        """STAGE 1: Handle None values in metadata safely."""
        mock_paper = MagicMock()
        mock_paper.authors = []  # Empty list
        mock_paper.get_short_id.side_effect = Exception("No ID")
        mock_paper.title = "Test"
        mock_paper.summary = "Abstract"
        mock_paper.published = datetime(2017, 1, 1)
        mock_paper.categories = ["cs.LG"]
        mock_paper.pdf_url = "https://example.com/pdf"
        
        metadata = _create_paper_metadata(mock_paper, "1706.03762")
        
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["authors"], [])


if __name__ == "__main__":
    unittest.main()
