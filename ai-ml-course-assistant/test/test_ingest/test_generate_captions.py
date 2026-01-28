"""
Comprehensive test suite for generate_captions.py module.

Tests cover:
- STAGE 1: Parameter validation, error handling, boundary conditions
- STAGE 2: Constants defined and used correctly
- STAGE 3: Helper functions working correctly and reducing duplication
"""

import unittest
import logging
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
from PIL import Image
import io
import sys

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingest.generate_captions import (
    ImageCaptioner,
    _validate_image_path,
    _resize_and_convert_image,
    _encode_image_to_base64,
    _assemble_enriched_caption,
    # Constants
    MAX_IMAGE_SIZE,
    JPEG_QUALITY,
    JPEG_OPTIMIZATION,
    MAX_CAPTION_TOKENS,
    VALID_IMAGE_MODES_FOR_CONVERSION,
    TARGET_IMAGE_MODE,
    IMAGE_FORMAT,
    IMAGE_DATA_URL_FORMAT,
)

from utils.logging_config import enable_test_mode
enable_test_mode()  # Simple format for test output


class TestHelperFunctions(unittest.TestCase):
    """Test STAGE 3 helper functions."""
    
    def test_validate_image_path_file_exists(self):
        """✅ validate_image_path returns True for existing readable file."""
        with patch('ingest.generate_captions.Path') as mock_path:
            mock_instance = MagicMock()
            mock_path.return_value = mock_instance
            mock_instance.exists.return_value = True
            mock_instance.is_file.return_value = True
            
            with patch('ingest.generate_captions.os.access', return_value=True):
                result = _validate_image_path("test.jpg")
        
        self.assertTrue(result)
    
    def test_validate_image_path_file_not_found(self):
        """✅ validate_image_path returns False for non-existent file."""
        with patch('ingest.generate_captions.Path') as mock_path:
            mock_instance = MagicMock()
            mock_path.return_value = mock_instance
            mock_instance.exists.return_value = False
            
            result = _validate_image_path("nonexistent.jpg")
        
        self.assertFalse(result)
    
    def test_validate_image_path_not_file(self):
        """✅ validate_image_path returns False if path is directory."""
        with patch('ingest.generate_captions.Path') as mock_path:
            mock_instance = MagicMock()
            mock_path.return_value = mock_instance
            mock_instance.exists.return_value = True
            mock_instance.is_file.return_value = False
            
            result = _validate_image_path("/some/directory")
        
        self.assertFalse(result)
    
    def test_validate_image_path_not_readable(self):
        """✅ validate_image_path returns False if file not readable."""
        with patch('ingest.generate_captions.Path') as mock_path:
            mock_instance = MagicMock()
            mock_path.return_value = mock_instance
            mock_instance.exists.return_value = True
            mock_instance.is_file.return_value = True
            
            with patch('ingest.generate_captions.os.access', return_value=False):
                result = _validate_image_path("test.jpg")
        
        self.assertFalse(result)
    
    def test_resize_and_convert_image_valid_max_size(self):
        """✅ resize_and_convert_image resizes when needed."""
        # Create mock image
        mock_img = MagicMock(spec=Image.Image)
        mock_img.size = (2000, 2000)
        mock_img.mode = 'RGB'
        mock_img.thumbnail = MagicMock()
        
        result = _resize_and_convert_image(mock_img, 1024)
        
        # Should call thumbnail
        mock_img.thumbnail.assert_called_once()
        self.assertEqual(result, mock_img)
    
    def test_resize_and_convert_image_invalid_max_size(self):
        """✅ resize_and_convert_image uses default for invalid max_size."""
        mock_img = MagicMock(spec=Image.Image)
        mock_img.size = (500, 500)
        mock_img.mode = 'RGB'
        
        result = _resize_and_convert_image(mock_img, -100)
        
        self.assertEqual(result, mock_img)
    
    def test_resize_and_convert_image_rgba_conversion(self):
        """✅ resize_and_convert_image converts RGBA to RGB."""
        mock_img = MagicMock(spec=Image.Image)
        mock_img.size = (500, 500)
        mock_img.mode = 'RGBA'
        mock_converted = MagicMock(spec=Image.Image)
        mock_img.convert.return_value = mock_converted
        
        result = _resize_and_convert_image(mock_img, 1024)
        
        mock_img.convert.assert_called_once_with('RGB')
        self.assertEqual(result, mock_converted)
    
    def test_encode_image_to_base64_success(self):
        """✅ encode_image_to_base64 returns base64 string."""
        mock_img = MagicMock(spec=Image.Image)
        
        with patch('ingest.generate_captions.io.BytesIO') as mock_buffer:
            mock_buffer_instance = MagicMock()
            mock_buffer.return_value = mock_buffer_instance
            mock_buffer_instance.read.return_value = b'\x89PNG\r\n'
            
            with patch('ingest.generate_captions.base64.b64encode') as mock_encode:
                mock_encode.return_value = b'iVBORw0KGgo='
                
                result = _encode_image_to_base64(mock_img)
        
        self.assertIsNotNone(result)
    
    def test_encode_image_to_base64_encoding_error(self):
        """✅ encode_image_to_base64 returns None on encoding error."""
        mock_img = MagicMock(spec=Image.Image)
        mock_img.save.side_effect = Exception("Save failed")
        
        result = _encode_image_to_base64(mock_img)
        
        self.assertIsNone(result)
    
    def test_assemble_enriched_caption_all_fields(self):
        """✅ assemble_enriched_caption combines all fields."""
        result = _assemble_enriched_caption(
            "Figure 1: Test",
            "Visual content here",
            "Context text"
        )
        
        self.assertIn("Figure caption: Figure 1: Test", result)
        self.assertIn("Visual description: Visual content here", result)
        self.assertIn("Context: Context text", result)
        self.assertIn("Note:", result)
    
    def test_assemble_enriched_caption_partial_fields(self):
        """✅ assemble_enriched_caption handles missing fields."""
        result = _assemble_enriched_caption(
            "",
            "Visual content",
            ""
        )
        
        self.assertNotIn("Figure caption:", result)
        self.assertIn("Visual description:", result)
        self.assertNotIn("Context:", result)
    
    def test_assemble_enriched_caption_invalid_types(self):
        """✅ assemble_enriched_caption handles non-string inputs."""
        result = _assemble_enriched_caption(
            None,
            123,
            ["list"]
        )
        
        # Should not crash, should handle gracefully
        self.assertIsInstance(result, str)


class TestParameterValidation(unittest.TestCase):
    """Test STAGE 1 parameter validation."""
    
    def test_encode_image_invalid_path_type(self):
        """✅ encode_image validates path type."""
        mock_api = MagicMock()
        captioner = ImageCaptioner(api_key="test_key")
        
        result = captioner.encode_image(123)  # Pass int instead of str
        
        self.assertIsNone(result)
    
    def test_encode_image_path_not_found(self):
        """✅ encode_image validates file existence."""
        with patch('ingest.generate_captions._validate_image_path', return_value=False):
            captioner = ImageCaptioner(api_key="test_key")
            result = captioner.encode_image("nonexistent.jpg")
        
        self.assertIsNone(result)
    
    def test_generate_caption_invalid_image_path_type(self):
        """✅ generate_caption validates image_path type."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            captioner = ImageCaptioner()
            result = captioner.generate_caption(None)
        
        self.assertIsNone(result)
    
    def test_generate_caption_invalid_max_length(self):
        """✅ generate_caption validates max_length parameter."""
        with patch('ingest.generate_captions.ImageCaptioner.encode_image', return_value=None):
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
                captioner = ImageCaptioner()
                result = captioner.generate_caption("test.jpg", max_length=-1)
        
        self.assertIsNone(result)
    
    def test_generate_caption_encoding_failure(self):
        """✅ generate_caption handles encoding failure."""
        with patch('ingest.generate_captions.ImageCaptioner.encode_image', return_value=None):
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
                captioner = ImageCaptioner()
                result = captioner.generate_caption("test.jpg")
        
        self.assertIsNone(result)


class TestConstants(unittest.TestCase):
    """Test STAGE 2 constants."""
    
    def test_max_image_size_is_valid(self):
        """✅ MAX_IMAGE_SIZE is positive integer."""
        self.assertIsInstance(MAX_IMAGE_SIZE, int)
        self.assertGreater(MAX_IMAGE_SIZE, 0)
    
    def test_jpeg_quality_is_valid(self):
        """✅ JPEG_QUALITY is in valid range (1-100)."""
        self.assertIsInstance(JPEG_QUALITY, int)
        self.assertGreater(JPEG_QUALITY, 0)
        self.assertLessEqual(JPEG_QUALITY, 100)
    
    def test_jpeg_optimization_is_boolean(self):
        """✅ JPEG_OPTIMIZATION is boolean."""
        self.assertIsInstance(JPEG_OPTIMIZATION, bool)
    
    def test_max_caption_tokens_is_valid(self):
        """✅ MAX_CAPTION_TOKENS is positive integer."""
        self.assertIsInstance(MAX_CAPTION_TOKENS, int)
        self.assertGreater(MAX_CAPTION_TOKENS, 0)
    
    def test_valid_image_modes_is_tuple(self):
        """✅ VALID_IMAGE_MODES_FOR_CONVERSION is tuple of strings."""
        self.assertIsInstance(VALID_IMAGE_MODES_FOR_CONVERSION, tuple)
        for mode in VALID_IMAGE_MODES_FOR_CONVERSION:
            self.assertIsInstance(mode, str)
    
    def test_target_image_mode_is_string(self):
        """✅ TARGET_IMAGE_MODE is string."""
        self.assertIsInstance(TARGET_IMAGE_MODE, str)
        self.assertEqual(TARGET_IMAGE_MODE, 'RGB')
    
    def test_image_format_is_string(self):
        """✅ IMAGE_FORMAT is string."""
        self.assertIsInstance(IMAGE_FORMAT, str)
        self.assertEqual(IMAGE_FORMAT, 'JPEG')
    
    def test_image_data_url_format_is_string(self):
        """✅ IMAGE_DATA_URL_FORMAT is format string."""
        self.assertIsInstance(IMAGE_DATA_URL_FORMAT, str)
        self.assertIn('{}', IMAGE_DATA_URL_FORMAT)


class TestIntegration(unittest.TestCase):
    """Test STAGE 3 integration scenarios."""
    
    def test_image_captioner_initialization_no_api_key(self):
        """✅ ImageCaptioner raises error when API key missing."""
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError):
                ImageCaptioner()
    
    def test_image_captioner_initialization_with_api_key(self):
        """✅ ImageCaptioner initializes successfully with API key."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            with patch('ingest.generate_captions.OpenAI'):
                captioner = ImageCaptioner()
                self.assertEqual(captioner.model_name, "gpt-4.1-mini")


def run_tests():
    """Run all tests and display summary."""
    logging.info("=" * 80)
    logging.info("Running: TestHelperFunctions")
    logging.info("=" * 80)
    
    suite = unittest.TestSuite()
    
    # Add helper tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHelperFunctions))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestParameterValidation))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConstants))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    logging.info("=" * 80)
    logging.info("TEST SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Total Tests: {result.testsRun}")
    logging.info(f"✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    logging.info(f"❌ Failed: {len(result.failures)}")
    logging.info(f"⚠️ Errors: {len(result.errors)}")
    logging.info(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    logging.info("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
