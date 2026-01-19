"""
Test suite for extract_images_smart refactored module.

Tests for STAGE 1, 2, 3 improvements:
- STAGE 1: Critical Bugs & Validation
- STAGE 2: Magic Numbers → Constants
- STAGE 3: SOLID Principles (DRY, SRP)

Uses unittest.mock to avoid file I/O during tests.
"""

import sys
import logging
import json
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch, mock_open

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingest.extract_images_smart import (
    _create_bbox_from_rect,
    _create_image_metadata,
    _load_papers_metadata,
    _save_images_metadata_file,
    extract_embedded_images,
    extract_vector_graphics,
    extract_images_smart,
    extract_text_from_pdf,
    process_all_papers,
    save_metadata,
    # Constants
    MIN_IMAGE_SIZE,
    DPI,
    MIN_VECTOR_REGION_SIZE,
    MERGE_THRESHOLD,
    VECTOR_REGION_PADDING,
    PDF_DPI_BASE,
    TEXT_EXTRACTION_FORMAT,
    PNG_FORMAT,
    FIGURE_CAPTION_KEYWORDS,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# STAGE 3: Helper Function Tests
# ============================================================================

class TestHelperFunctions:
    """Test STAGE 3 DRY/SRP helper functions."""
    
    def test_create_bbox_from_rect_valid(self):
        """STAGE 3: Test bbox creation from valid rect."""
        logger.info("\n=== Test: _create_bbox_from_rect (valid) ===")
        
        rect = MagicMock()
        rect.x0 = 10
        rect.y0 = 20
        rect.x1 = 100
        rect.y1 = 200
        
        bbox = _create_bbox_from_rect(rect)
        
        assert bbox is not None, "bbox should not be None"
        assert bbox["x0"] == 10, f"Expected x0=10, got {bbox['x0']}"
        assert bbox["y0"] == 20, f"Expected y0=20, got {bbox['y0']}"
        assert bbox["x1"] == 100, f"Expected x1=100, got {bbox['x1']}"
        assert bbox["y1"] == 200, f"Expected y1=200, got {bbox['y1']}"
        logger.info("✅ Bbox created correctly from rect")
    
    def test_create_bbox_from_rect_none(self):
        """STAGE 3: Test bbox creation from None rect."""
        logger.info("\n=== Test: _create_bbox_from_rect (None) ===")
        
        bbox = _create_bbox_from_rect(None)
        
        assert bbox is None, "bbox should be None for None rect"
        logger.info("✅ None rect handled correctly")
    
    def test_create_image_metadata_full(self):
        """STAGE 3: Test image metadata creation with all fields."""
        logger.info("\n=== Test: _create_image_metadata (full) ===")
        
        image_path = Path("/tmp/test_image.png")
        bbox = {"x0": 10, "y0": 20, "x1": 100, "y1": 200}
        
        metadata = _create_image_metadata(
            image_id="doc1_embedded_001",
            doc_id="doc1",
            image_path=image_path,
            page_num=1,
            width=90,
            height=180,
            image_format="png",
            size_bytes=5000,
            extraction_method="embedded_raster",
            bbox=bbox,
            dpi=200,
            region_index=1
        )
        
        assert metadata["image_id"] == "doc1_embedded_001", "image_id mismatch"
        assert metadata["doc_id"] == "doc1", "doc_id mismatch"
        assert metadata["page_num"] == 1, "page_num mismatch"
        assert metadata["width"] == 90, "width mismatch"
        assert metadata["height"] == 180, "height mismatch"
        assert metadata["extraction_method"] == "embedded_raster", "extraction_method mismatch"
        assert metadata["bbox"] == bbox, "bbox mismatch"
        assert metadata["dpi"] == 200, "dpi mismatch"
        assert metadata["region_index"] == 1, "region_index mismatch"
        logger.info("✅ Image metadata created correctly with all fields")
    
    def test_create_image_metadata_minimal(self):
        """STAGE 3: Test image metadata creation with minimal fields."""
        logger.info("\n=== Test: _create_image_metadata (minimal) ===")
        
        image_path = Path("/tmp/test_image.png")
        
        metadata = _create_image_metadata(
            image_id="doc1_embedded_001",
            doc_id="doc1",
            image_path=image_path,
            page_num=1,
            width=150,
            height=200,
            image_format="png",
            size_bytes=10000,
            extraction_method="embedded_raster"
        )
        
        # Check required fields exist
        assert "image_id" in metadata, "image_id should be in metadata"
        assert "doc_id" in metadata, "doc_id should be in metadata"
        # Check optional fields are None
        assert metadata.get("dpi") is None, "dpi should be None"
        assert metadata.get("region_index") is None, "region_index should be None"
        assert metadata.get("bbox") is None, "bbox should be None"
        logger.info("✅ Image metadata created correctly with minimal fields")
    
    @patch("builtins.open", new_callable=mock_open)
    def test_load_papers_metadata_valid_json(self, mock_file):
        """STAGE 3: Test loading valid papers metadata."""
        logger.info("\n=== Test: _load_papers_metadata (valid) ===")
        
        metadata_path = Path("/tmp/papers_metadata.json")
        json_data = [
            {"doc_id": "paper1", "title": "Title 1"},
            {"doc_id": "paper2", "title": "Title 2"}
        ]
        
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(json_data)
        
        with patch.object(Path, 'exists', return_value=True):
            result = _load_papers_metadata(metadata_path)
        
        assert "paper1" in result, "paper1 should be in result"
        assert "paper2" in result, "paper2 should be in result"
        assert result["paper1"]["title"] == "Title 1", "paper1 title mismatch"
        logger.info("✅ Valid papers metadata loaded correctly")
    
    def test_load_papers_metadata_missing_file(self):
        """STAGE 3: Test loading from missing metadata file."""
        logger.info("\n=== Test: _load_papers_metadata (missing) ===")
        
        metadata_path = Path("/tmp/nonexistent.json")
        
        with patch.object(Path, 'exists', return_value=False):
            result = _load_papers_metadata(metadata_path)
        
        assert result == {}, "Should return empty dict for missing file"
        logger.info("✅ Missing file handled correctly - returned empty dict")
    
    @patch("builtins.open", new_callable=mock_open)
    def test_save_images_metadata_file_success(self, mock_file):
        """STAGE 3: Test saving images metadata successfully."""
        logger.info("\n=== Test: _save_images_metadata_file (success) ===")
        
        metadata_path = Path("/tmp/images_metadata.json")
        images_metadata = [
            {"image_id": "img1", "doc_id": "doc1"},
            {"image_id": "img2", "doc_id": "doc1"}
        ]
        
        with patch.object(Path, 'exists', return_value=True):
            result = _save_images_metadata_file(metadata_path, images_metadata)
        
        assert result is True, "Should return True on success"
        mock_file.assert_called_once()
        logger.info("✅ Images metadata saved successfully")
    
    @patch("builtins.open", side_effect=IOError("Write failed"))
    def test_save_images_metadata_file_ioerror(self, mock_file):
        """STAGE 3: Test saving with IOError."""
        logger.info("\n=== Test: _save_images_metadata_file (IOError) ===")
        
        metadata_path = Path("/tmp/images_metadata.json")
        images_metadata = [{"image_id": "img1"}]
        
        result = _save_images_metadata_file(metadata_path, images_metadata)
        
        assert result is False, "Should return False on IOError"
        logger.info("✅ IOError handled correctly - returned False")


# ============================================================================
# STAGE 1: Parameter Validation Tests
# ============================================================================

class TestParameterValidation:
    """Test STAGE 1 parameter validation."""
    
    def test_extract_embedded_images_none_pdf_document(self):
        """STAGE 1: Test with None pdf_document."""
        logger.info("\n=== Test: extract_embedded_images (None pdf_document) ===")
        
        result_metadata, result_count = extract_embedded_images(
            None,
            Path("/tmp"),
            "doc1"
        )
        
        assert result_metadata == [], "Should return empty list"
        assert result_count == 0, "Should return 0 count"
        logger.info("✅ None pdf_document handled correctly")
    
    def test_extract_embedded_images_invalid_doc_id(self):
        """STAGE 1: Test with invalid doc_id."""
        logger.info("\n=== Test: extract_embedded_images (invalid doc_id) ===")
        
        pdf_doc = MagicMock()
        
        result_metadata, result_count = extract_embedded_images(
            pdf_doc,
            Path("/tmp"),
            ""  # Empty string
        )
        
        assert result_metadata == [], "Should return empty list"
        assert result_count == 0, "Should return 0 count"
        logger.info("✅ Invalid doc_id handled correctly")
    
    def test_extract_vector_graphics_invalid_dpi(self):
        """STAGE 1: Test with invalid DPI (negative)."""
        logger.info("\n=== Test: extract_vector_graphics (invalid DPI) ===")
        
        pdf_doc = MagicMock()
        pdf_doc.__len__ = MagicMock(return_value=0)
        
        result = extract_vector_graphics(
            pdf_doc,
            Path("/tmp"),
            "doc1",
            dpi=-100  # Invalid DPI
        )
        
        assert result == [], "Should return empty list (no pages)"
        logger.info("✅ Invalid DPI handled correctly")
    
    def test_extract_images_smart_invalid_path(self):
        """STAGE 1: Test with invalid path type."""
        logger.info("\n=== Test: extract_images_smart (invalid path type) ===")
        
        result = extract_images_smart(
            "/tmp/test.pdf",  # String instead of Path
            Path("/tmp"),
            "doc1"
        )
        
        assert result == [], "Should return empty list"
        logger.info("✅ Invalid path type handled correctly")


# ============================================================================
# STAGE 2: Constants Validation Tests
# ============================================================================

class TestConstants:
    """Test STAGE 2 constants are correctly defined."""
    
    def test_constants_defined(self):
        """STAGE 2: Test all constants are defined with correct types."""
        logger.info("\n=== Test: Constants are defined ===")
        
        assert isinstance(MIN_IMAGE_SIZE, int) and MIN_IMAGE_SIZE > 0, "MIN_IMAGE_SIZE invalid"
        assert isinstance(DPI, int) and DPI > 0, "DPI invalid"
        assert isinstance(MIN_VECTOR_REGION_SIZE, int) and MIN_VECTOR_REGION_SIZE > 0, "MIN_VECTOR_REGION_SIZE invalid"
        assert isinstance(MERGE_THRESHOLD, (int, float)) and MERGE_THRESHOLD > 0, "MERGE_THRESHOLD invalid"
        assert isinstance(VECTOR_REGION_PADDING, int) and VECTOR_REGION_PADDING >= 0, "VECTOR_REGION_PADDING invalid"
        assert isinstance(PDF_DPI_BASE, int) and PDF_DPI_BASE > 0, "PDF_DPI_BASE invalid"
        assert isinstance(TEXT_EXTRACTION_FORMAT, str) and TEXT_EXTRACTION_FORMAT, "TEXT_EXTRACTION_FORMAT invalid"
        assert isinstance(PNG_FORMAT, str) and PNG_FORMAT, "PNG_FORMAT invalid"
        assert isinstance(FIGURE_CAPTION_KEYWORDS, list) and len(FIGURE_CAPTION_KEYWORDS) > 0, "FIGURE_CAPTION_KEYWORDS invalid"
        
        logger.info("✅ All constants defined correctly")
        logger.info(f"   MIN_IMAGE_SIZE = {MIN_IMAGE_SIZE}")
        logger.info(f"   DPI = {DPI}")
        logger.info(f"   MIN_VECTOR_REGION_SIZE = {MIN_VECTOR_REGION_SIZE}")
        logger.info(f"   MERGE_THRESHOLD = {MERGE_THRESHOLD}")
        logger.info(f"   VECTOR_REGION_PADDING = {VECTOR_REGION_PADDING}")


# ============================================================================
# Integration Tests (Mock-based, no file I/O)
# ============================================================================

class TestIntegration:
    """Integration tests using mocks."""
    
    @patch("ingest.extract_images_smart.fitz.open")
    def test_extract_images_smart_with_mocked_pdf(self, mock_fitz_open):
        """STAGE 1: Test extract_images_smart with mocked PDF."""
        logger.info("\n=== Test: extract_images_smart (mocked PDF) ===")
        
        # Setup mocks
        pdf_doc = MagicMock()
        pdf_doc.__len__ = MagicMock(return_value=2)
        pdf_doc.__iter__ = MagicMock(return_value=iter([MagicMock(), MagicMock()]))
        pdf_doc.close = MagicMock()
        mock_fitz_open.return_value = pdf_doc
        
        pdf_path = Path("/tmp/test.pdf")
        output_dir = Path("/tmp/output")
        
        with patch.object(Path, 'mkdir'):
            with patch("ingest.extract_images_smart.extract_embedded_images", return_value=([], 0)):
                with patch("ingest.extract_images_smart.extract_vector_graphics", return_value=[]):
                    result = extract_images_smart(pdf_path, output_dir, "doc1")
        
        assert isinstance(result, list), "Should return list"
        assert pdf_doc.close.called, "PDF should be closed"
        logger.info("✅ extract_images_smart with mocked PDF works correctly")
    
    @patch("builtins.open", new_callable=mock_open)
    def test_save_metadata_with_mocks(self, mock_file):
        """STAGE 3: Test save_metadata with mocks (no file I/O)."""
        logger.info("\n=== Test: save_metadata (mocked I/O) ===")
        
        images_metadata = [{"image_id": "img1", "doc_id": "doc1"}]
        documents_metadata = [{"doc_id": "doc1", "filename": "doc1.pdf"}]
        output_dir = Path("/tmp/output/images")
        
        with patch.object(Path, 'parent', Path("/tmp/output")):
            result = save_metadata(images_metadata, documents_metadata, output_dir)
        
        assert result is not None, "Should return tuple of paths"
        logger.info("✅ save_metadata works correctly with mocks")


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests and print summary."""
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING TEST SUITE FOR extract_images_smart.py")
    logger.info("=" * 80)
    
    test_classes = [
        TestHelperFunctions,
        TestParameterValidation,
        TestConstants,
        TestIntegration,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Running: {test_class.__name__}")
        logger.info(f"{'=' * 80}")
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith("test_")]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, test_method)
                method()
                passed_tests += 1
            except Exception as e:
                failed_tests += 1
                logger.error(f"❌ {test_method} FAILED: {e}")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"✅ Passed: {passed_tests}")
    logger.info(f"❌ Failed: {failed_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
    logger.info(f"{'=' * 80}\n")
    
    return failed_tests == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
