"""
Test suite for extract_image_context refactored module.

Tests for STAGE 1, 2, 3 improvements:
- STAGE 1: Critical Bugs & Validation
- STAGE 2: Magic Numbers → Constants
- STAGE 3: SOLID Principles (SRP, DRY)
"""

import sys
import logging
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingest.extract_image_context import (
    extract_figure_caption,
    extract_full_caption_text,
    extract_surrounding_context,
    _extract_sentence_boundary_from_text,
    _group_text_into_paragraphs,
    _extract_context_from_previous_page,
    _collect_text_blocks_from_page,
    _find_line_index,
    _extract_text_from_lines,
    _extract_sentence_from_end,
    _extract_sentence_from_start,
    # Constants
    CAPTION_MAX_VERTICAL_DISTANCE,
    CAPTION_MAX_LENGTH,
    CONTEXT_MAX_CHARS,
    PARAGRAPH_Y_GAP_THRESHOLD,
    SENTENCE_END_MARKERS,
)

from utils.logging_config import enable_test_mode
enable_test_mode()  # Simple format for test output
logger = logging.getLogger(__name__)


# ============================================================================
# STAGE 3: Helper Function Tests
# ============================================================================

class TestHelperFunctions:
    """Test STAGE 3 DRY/SRP helper functions."""
    
    def test_collect_text_blocks_from_page_valid(self):
        """STAGE 3: Test text collection from valid page."""
        logger.info("\n=== Test: _collect_text_blocks_from_page (valid) ===")
        
        page = MagicMock()
        page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Hello",
                                    "bbox": [0, 10, 50, 20]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        result = _collect_text_blocks_from_page(page)
        
        assert len(result) == 1, "Should collect 1 text item"
        assert result[0]["text"] == "Hello", "Text should be 'Hello'"
        assert result[0]["y0"] == 10, "y0 should be 10"
        assert result[0]["y1"] == 20, "y1 should be 20"
        logger.info("✅ Text blocks collected correctly")
    
    def test_find_line_index_valid(self):
        """STAGE 3 & STAGE 1: Test finding line by content."""
        logger.info("\n=== Test: _find_line_index (valid) ===")
        
        block = {
            "lines": [
                {"spans": [{"text": "Line 1"}]},
                {"spans": [{"text": "Line 2"}]},
                {"spans": [{"text": "Line 3"}]}
            ]
        }
        starting_line = {"spans": [{"text": "Line 2"}]}
        
        idx = _find_line_index(block, starting_line)
        
        assert idx == 1, f"Expected index 1, got {idx}"
        logger.info("✅ Line found by content correctly")
    
    def test_extract_text_from_lines_valid(self):
        """STAGE 3: Test text extraction from lines."""
        logger.info("\n=== Test: _extract_text_from_lines (valid) ===")
        
        lines = [
            {"spans": [{"text": "Start"}]},
            {"spans": [{"text": "Middle"}]},
            {"spans": [{"text": "End"}]}
        ]
        
        result = _extract_text_from_lines(lines, 1, 50)
        
        assert "Middle" in result, "Should contain 'Middle'"
        assert "End" in result, "Should contain 'End'"
        assert "Start" not in result, "Should not contain 'Start'"
        logger.info("✅ Text extracted from lines correctly")
    
    def test_extract_sentence_from_end_valid(self):
        """STAGE 3: Test sentence extraction from end."""
        logger.info("\n=== Test: _extract_sentence_from_end (valid) ===")
        
        text = "This is sentence one. This is sentence two. This is sentence three."
        result = _extract_sentence_from_end(text, 40)
        
        assert result, "Should return text"
        assert "sentence" in result.lower(), "Should contain 'sentence'"
        logger.info(f"✅ Sentence extracted: '{result}'")
    
    def test_extract_sentence_from_start_valid(self):
        """STAGE 3: Test sentence extraction from start."""
        logger.info("\n=== Test: _extract_sentence_from_start (valid) ===")
        
        text = "First sentence. Second sentence. Third sentence."
        result = _extract_sentence_from_start(text, 40)
        
        assert result, "Should return text"
        assert "First" in result, "Should start with 'First'"
        logger.info(f"✅ Sentence extracted: '{result}'")


# ============================================================================
# STAGE 1: Parameter Validation Tests
# ============================================================================

class TestParameterValidation:
    """Test STAGE 1 parameter validation."""
    
    def test_extract_figure_caption_none_text_blocks(self):
        """STAGE 1: Test with None text_blocks."""
        logger.info("\n=== Test: extract_figure_caption (None text_blocks) ===")
        
        image_bbox = MagicMock()
        image_bbox.y0 = 100
        image_bbox.y1 = 200
        
        result = extract_figure_caption(None, image_bbox)
        
        assert result is None, "Should return None"
        logger.info("✅ None text_blocks handled correctly")
    
    def test_extract_figure_caption_none_image_bbox(self):
        """STAGE 1: Test with None image_bbox."""
        logger.info("\n=== Test: extract_figure_caption (None image_bbox) ===")
        
        text_blocks = []
        result = extract_figure_caption(text_blocks, None)
        
        assert result is None, "Should return None"
        logger.info("✅ None image_bbox handled correctly")
    
    def test_extract_surrounding_context_none_page(self):
        """STAGE 1: Test with None page."""
        logger.info("\n=== Test: extract_surrounding_context (None page) ===")
        
        image_bbox = MagicMock()
        result = extract_surrounding_context(None, image_bbox)
        
        assert result is not None, "Should return dict"
        assert result["before"] == "", "Should have empty before"
        assert result["figure_caption"] is None, "Should have None caption"
        logger.info("✅ None page handled correctly")
    


class TestConstants:
    """Test STAGE 2 constants are correctly defined."""
    
    def test_constants_defined(self):
        """STAGE 2: Test all constants are defined with correct types."""
        logger.info("\n=== Test: Constants are defined ===")
        
        assert isinstance(CAPTION_MAX_VERTICAL_DISTANCE, int) and CAPTION_MAX_VERTICAL_DISTANCE > 0
        assert isinstance(CAPTION_MAX_LENGTH, int) and CAPTION_MAX_LENGTH > 0
        assert isinstance(CONTEXT_MAX_CHARS, int) and CONTEXT_MAX_CHARS > 0
        assert isinstance(PARAGRAPH_Y_GAP_THRESHOLD, int) and PARAGRAPH_Y_GAP_THRESHOLD > 0
        assert isinstance(SENTENCE_END_MARKERS, list) and len(SENTENCE_END_MARKERS) > 0
        
        logger.info("✅ All constants defined correctly")
        logger.info(f"   CAPTION_MAX_VERTICAL_DISTANCE = {CAPTION_MAX_VERTICAL_DISTANCE}")
        logger.info(f"   CAPTION_MAX_LENGTH = {CAPTION_MAX_LENGTH}")
        logger.info(f"   CONTEXT_MAX_CHARS = {CONTEXT_MAX_CHARS}")
        logger.info(f"   PARAGRAPH_Y_GAP_THRESHOLD = {PARAGRAPH_Y_GAP_THRESHOLD}")


# ============================================================================
# Integration Tests (Mock-based)
# ============================================================================

class TestIntegration:
    """Integration tests using mocks."""
    
    def test_extract_figure_caption_with_mocks(self):
        """STAGE 1-3: Test caption extraction with mocks."""
        logger.info("\n=== Test: extract_figure_caption (mocked) ===")
        
        image_bbox = MagicMock()
        image_bbox.y0 = 100
        image_bbox.y1 = 200
        
        text_blocks = [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "Figure 1: Test caption",
                                "bbox": [0, 210, 100, 220]
                            }
                        ]
                    }
                ]
            }
        ]
        
        result = extract_figure_caption(text_blocks, image_bbox)
        
        assert result is not None, "Should find caption"
        assert "Figure 1" in result, "Should contain 'Figure 1'"
        logger.info(f"✅ Caption found: '{result}'")
    
    def test_group_text_into_paragraphs_with_mocks(self):
        """STAGE 3: Test paragraph grouping with mocks."""
        logger.info("\n=== Test: _group_text_into_paragraphs (mocked) ===")
        
        text_items = [
            {"text": "Line 1", "y0": 0, "y1": 10},
            {"text": "Line 2", "y0": 12, "y1": 22},  # Close (gap=2)
            {"text": "Line 3", "y0": 50, "y1": 60},  # Far (gap=28)
        ]
        
        result = _group_text_into_paragraphs(text_items)
        
        assert len(result) == 2, f"Expected 2 paragraphs, got {len(result)}"
        assert "Line 1" in result[0][0] and "Line 2" in result[0][0], "First para should have Line 1-2"
        assert "Line 3" in result[1][0], "Second para should have Line 3"
        logger.info(f"✅ Grouped into {len(result)} paragraphs correctly")
    


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests and print summary."""
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING TEST SUITE FOR extract_image_context.py")
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
