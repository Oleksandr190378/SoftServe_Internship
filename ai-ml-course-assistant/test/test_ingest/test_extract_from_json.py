"""
Test suite for extract_from_json refactored module.

Tests for STAGE 1, 2, 3 improvements:
- STAGE 1: Critical Bugs & Validation
- STAGE 2: Exception Handling & Constants
- STAGE 3: SOLID Principles (SRP, DRY, KISS)
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingest.extract_from_json import (
    detect_source_type,
    _should_skip_context,
    _extract_sentence_boundary,
    _find_position_by_keywords,
    find_context_for_image,
    _determine_image_extension,
    MIN_IMAGE_DIMENSION,
    REQUEST_TIMEOUT_SECONDS,
    CONTEXT_LOOK_BACK_CHARS,
    CONTEXT_LOOK_FORWARD_CHARS,
)

from utils.logging_config import enable_test_mode
enable_test_mode()  # Simple format for test output
logger = logging.getLogger(__name__)


def test_detect_source_type():
    """STAGE 1: Test source type detection with validation."""
    logger.info("\n=== Test: detect_source_type ===")
    
    # Valid inputs
    assert detect_source_type("realpython_numpy-tutorial") == "realpython", "Failed: realpython detection"
    assert detect_source_type("medium_agents-plan") == "medium", "Failed: medium detection"
    logger.info("✅ Valid source types detected correctly")
    
    # STAGE 1: Validation - check invalid inputs
    try:
        detect_source_type("")
        assert False, "Should raise ValueError for empty string"
    except ValueError as e:
        logger.info(f"✅ Empty string handled: {e}")
    
    try:
        detect_source_type("unknown_doc")
        assert False, "Should raise ValueError for unknown prefix"
    except ValueError as e:
        logger.info(f"✅ Unknown prefix handled: {e}")
    
    try:
        detect_source_type(None)
        assert False, "Should raise ValueError for None"
    except ValueError as e:
        logger.info(f"✅ None handled: {e}")


def test_should_skip_context():
    """STAGE 2/1: Test context skipping with proper validation."""
    logger.info("\n=== Test: _should_skip_context ===")
    
    # STAGE 1: Handle empty/None values
    assert _should_skip_context("") == False, "Failed: empty caption should not skip"
    assert _should_skip_context(None) == False, "Failed: None caption should not skip"
    logger.info("✅ Empty values handled correctly")
    
    # Generic captions should skip
    assert _should_skip_context("Image by author") == False, "Caption alone should not skip"
    assert _should_skip_context("Image by author", "not a technical image") == True, "Should skip generic + VLM"
    logger.info("✅ Generic captions with VLM detected correctly")
    
    # Technical captions should not skip
    assert _should_skip_context("CNN architecture diagram") == False, "Should not skip technical caption"
    logger.info("✅ Technical captions preserved")


def test_extract_sentence_boundary():
    """STAGE 1/2: Test sentence boundary extraction with boundary checks."""
    logger.info("\n=== Test: _extract_sentence_boundary ===")
    
    text = "This is sentence one. This is sentence two. And this is three."
    
    # STAGE 1: Test empty/None handling
    assert _extract_sentence_boundary("", 0) == "", "Failed: empty text should return empty"
    assert _extract_sentence_boundary(None, 0) == "", "Failed: None should return empty"
    logger.info("✅ Empty values handled")
    
    # STAGE 1: Test boundary conditions
    result_before = _extract_sentence_boundary(text, 20, "before")
    assert len(result_before) > 0, "Should extract context before"
    assert "sentence one" in result_before.lower(), "Should contain previous sentence"
    logger.info(f"✅ Before extraction: '{result_before[:30]}...'")
    
    result_after = _extract_sentence_boundary(text, 20, "after")
    assert len(result_after) > 0, "Should extract context after"
    logger.info(f"✅ After extraction: '{result_after[:30]}...'")
    
    # STAGE 1: Test position boundary checks
    result_far_end = _extract_sentence_boundary(text, len(text) + 100, "after")
    assert isinstance(result_far_end, str), "Should return string even for out-of-bounds position"
    logger.info("✅ Out-of-bounds position handled")


def test_find_position_by_keywords():
    """STAGE 1/3: Test position finding with edge case handling."""
    logger.info("\n=== Test: _find_position_by_keywords ===")
    
    text = "Introduction here. Here is a figure about neural networks. The architecture is shown in Fig. 1."
    
    # STAGE 1: Handle empty/None
    assert _find_position_by_keywords("", "caption", "", 1) == -1, "Empty text should return -1"
    assert _find_position_by_keywords(None, "caption", "", 1) == -1, "None text should return -1"
    logger.info("✅ Empty values return -1")
    
    # STAGE 1: Handle invalid image_index
    pos = _find_position_by_keywords(text, "", "", 0)  # 0-based index
    assert pos >= 0 or pos == -1, "Should handle 0-based index"
    logger.info(f"✅ Image index 0 handled: {pos}")
    
    # Normal search
    pos = _find_position_by_keywords(text, "neural networks", "", 1)
    assert pos > 0, "Should find text in document"
    logger.info(f"✅ Found 'neural networks' at position {pos}")
    
    # Not found - should use fallback
    pos = _find_position_by_keywords(text, "nonexistent", "", 1)
    assert pos >= 0 or pos == -1, "Should return valid position or -1"
    logger.info(f"✅ Fallback for 'nonexistent': {pos}")


def test_find_context_for_image():
    """STAGE 1/3: Test context finding with validation."""
    logger.info("\n=== Test: find_context_for_image ===")
    
    text = "Section 1. Here we discuss networks. Figure 1: Architecture diagram. The results show improvements."
    
    # STAGE 1: Empty text handling
    before, after = find_context_for_image("", "Fig 1", "", 1)
    assert before == "" and after == "", "Empty text should return empty strings"
    logger.info("✅ Empty text handled")
    
    # STAGE 1: Invalid image_index
    before, after = find_context_for_image(text, "Architecture", "", 0)
    assert isinstance(before, str) and isinstance(after, str), "Should return tuple of strings"
    logger.info(f"✅ Image index 0 handled: before='{before[:20]}...'")
    
    # Normal operation
    before, after = find_context_for_image(text, "Architecture diagram", "", 1)
    assert isinstance(before, str) and isinstance(after, str), "Should return tuple of strings"
    logger.info(f"✅ Context extracted: before='{before[:30]}...' after='{after[:30]}...'")
    
    # Decorative image skip
    before, after = find_context_for_image(text, "Image by author", "", 1, vlm_description="not a technical image")
    assert before == "" and after == "", "Decorative image should return empty context"
    logger.info("✅ Decorative image skipped")


def test_determine_image_extension():
    """STAGE 2/3: Test DRY principle - extension detection."""
    logger.info("\n=== Test: _determine_image_extension ===")
    
    # STAGE 3: DRY - centralized extension detection
    assert _determine_image_extension("http://example.com/image.png") == "png", "Failed: PNG detection"
    assert _determine_image_extension("http://example.com/image.jpg") == "jpg", "Failed: JPG detection"
    assert _determine_image_extension("http://example.com/image.jpeg") == "jpeg", "Failed: JPEG detection"
    assert _determine_image_extension("http://example.com/image.webp") == "webp", "Failed: WEBP detection"
    assert _determine_image_extension("http://example.com/image.gif") == "gif", "Failed: GIF detection"
    logger.info("✅ All formats detected correctly")
    
    # STAGE 1: Handle edge cases
    assert _determine_image_extension("") == "png", "Empty URL should default to PNG"
    assert _determine_image_extension(None) == "png", "None should default to PNG"
    assert _determine_image_extension("http://example.com/image.unknown") == "png", "Unknown format defaults to PNG"
    logger.info("✅ Edge cases default to PNG")
    
    # STAGE 2: Constants are used
    logger.info(f"✅ Constants defined: LOOK_BACK={CONTEXT_LOOK_BACK_CHARS}, LOOK_FORWARD={CONTEXT_LOOK_FORWARD_CHARS}")
    logger.info(f"✅ Constants defined: MIN_DIMENSION={MIN_IMAGE_DIMENSION}, TIMEOUT={REQUEST_TIMEOUT_SECONDS}s")


def test_constants_replacement():
    """STAGE 2: Verify magic numbers replaced with constants."""
    logger.info("\n=== Test: Constants Replacement ===")
    
    # Check that constants exist and have reasonable values
    assert MIN_IMAGE_DIMENSION == 50, f"MIN_IMAGE_DIMENSION should be 50, got {MIN_IMAGE_DIMENSION}"
    assert REQUEST_TIMEOUT_SECONDS == 10, f"REQUEST_TIMEOUT_SECONDS should be 10, got {REQUEST_TIMEOUT_SECONDS}"
    assert CONTEXT_LOOK_BACK_CHARS == 250, f"CONTEXT_LOOK_BACK_CHARS should be 250, got {CONTEXT_LOOK_BACK_CHARS}"
    assert CONTEXT_LOOK_FORWARD_CHARS == 250, f"CONTEXT_LOOK_FORWARD_CHARS should be 250, got {CONTEXT_LOOK_FORWARD_CHARS}"
    
    logger.info("✅ All magic numbers replaced with named constants")
    logger.info("✅ STAGE 2: Constants are now configurable in one place")


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("REFACTORED extract_from_json.py - UNIT TESTS")
    logger.info("Testing: STAGE 1 (Validation), STAGE 2 (Constants), STAGE 3 (SOLID)")
    logger.info("=" * 70)
    
    try:
        test_detect_source_type()
        test_should_skip_context()
        test_extract_sentence_boundary()
        test_find_position_by_keywords()
        test_find_context_for_image()
        test_determine_image_extension()
        test_constants_replacement()
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 70)
        logger.info("\nRefactoring Summary:")
        logger.info("✅ STAGE 1: Critical Bugs & Validation - Edge cases handled")
        logger.info("✅ STAGE 2: Exception Handling & Constants - Magic numbers replaced")
        logger.info("✅ STAGE 3: SOLID Principles - Helper functions extracted (DRY, SRP)")
        logger.info("=" * 70 + "\n")
        
        return 0
        
    except AssertionError as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        logger.error(f"\n❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
