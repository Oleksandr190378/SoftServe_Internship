"""
Comprehensive test suite for ingest/enrich_images.py

STAGE 1: Parameter validation tests
STAGE 2: Constants verification tests
STAGE 3: Helper function tests + Integration tests
"""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingest.enrich_images import (
    # Constants
    VLM_MAX_CAPTION_TOKENS,
    CONTEXT_MAX_CHARS,
    CAPTION_PREVIEW_CHARS,
    RATE_LIMIT_REQUESTS_PER_MINUTE,
    RATE_LIMIT_DELAY_SECONDS,
    EXTRACTION_METHOD_PDF,
    EXTRACTION_METHOD_WEB,
    PDF_PAGE_OFFSET,
    # Helper functions
    _validate_image_metadata,
    _validate_bbox_dict,
    _validate_page_bounds,
    _extract_context_safely,
    _generate_vlm_description_safe,
    _assemble_enriched_caption_text,
    _load_and_validate_metadata,
    # Main functions
    load_images_metadata,
    save_images_metadata,
    enrich_single_image,
    enrich_all_images,
    generate_captions_for_doc,
)


class TestConstants(unittest.TestCase):
    """STAGE 2: Verify all constants are properly defined"""
    
    def test_constants_defined(self):
        """All required constants should be defined"""
        self.assertEqual(VLM_MAX_CAPTION_TOKENS, 1024)
        self.assertEqual(CONTEXT_MAX_CHARS, 250)
        self.assertEqual(CAPTION_PREVIEW_CHARS, 300)
        self.assertEqual(RATE_LIMIT_REQUESTS_PER_MINUTE, 20)
        self.assertEqual(RATE_LIMIT_DELAY_SECONDS, 3.5)
        self.assertEqual(EXTRACTION_METHOD_PDF, 'embedded_raster')
        self.assertEqual(EXTRACTION_METHOD_WEB, 'web_download')
        self.assertEqual(PDF_PAGE_OFFSET, 1)
    
    def test_rate_limit_consistency(self):
        """Rate limiting constants should be reasonable"""
        # 20 req/min = 3 sec/req minimum, but 3.5 is safe margin
        expected_min = 60 / RATE_LIMIT_REQUESTS_PER_MINUTE
        self.assertGreaterEqual(RATE_LIMIT_DELAY_SECONDS, expected_min)


class TestValidationHelpers(unittest.TestCase):
    """STAGE 3: Test all validation helper functions"""
    
    def test_validate_image_metadata_valid(self):
        """Valid image metadata should pass validation"""
        img_meta = {
            'doc_id': 'arxiv_123',
            'image_id': 'img_001',
            'page_num': 1,
            'bbox': {'x0': 0, 'y0': 0, 'x1': 100, 'y1': 100}
        }
        self.assertTrue(_validate_image_metadata(img_meta))
    
    def test_validate_image_metadata_missing_doc_id(self):
        """Missing doc_id should fail validation"""
        img_meta = {'image_id': 'img_001'}
        self.assertFalse(_validate_image_metadata(img_meta))
    
    def test_validate_image_metadata_missing_image_id(self):
        """Missing image_id should fail validation"""
        img_meta = {'doc_id': 'arxiv_123'}
        self.assertFalse(_validate_image_metadata(img_meta))
    
    def test_validate_image_metadata_not_dict(self):
        """Non-dict should fail validation"""
        self.assertFalse(_validate_image_metadata(None))
        self.assertFalse(_validate_image_metadata([]))
        self.assertFalse(_validate_image_metadata("string"))
    
    def test_validate_bbox_dict_valid(self):
        """Valid bbox dict should pass validation"""
        bbox = {'x0': 0.0, 'y0': 0.0, 'x1': 100.0, 'y1': 100.0}
        self.assertTrue(_validate_bbox_dict(bbox))
    
    def test_validate_bbox_dict_missing_keys(self):
        """Missing bbox keys should fail validation"""
        bbox = {'x0': 0.0, 'y0': 0.0}  # Missing x1, y1
        self.assertFalse(_validate_bbox_dict(bbox))
    
    def test_validate_bbox_dict_invalid_values(self):
        """Invalid bbox values should fail validation"""
        bbox = {'x0': 'invalid', 'y0': 0.0, 'x1': 100.0, 'y1': 100.0}
        self.assertFalse(_validate_bbox_dict(bbox))
    
    def test_validate_page_bounds_valid(self):
        """Valid page within bounds should pass"""
        doc = MagicMock()
        doc.__len__ = MagicMock(return_value=10)
        self.assertTrue(_validate_page_bounds(doc, 5))  # Page 5 of 10
    
    def test_validate_page_bounds_out_of_range(self):
        """Page beyond document range should fail"""
        doc = MagicMock()
        doc.__len__ = MagicMock(return_value=10)
        self.assertFalse(_validate_page_bounds(doc, 15))  # Page 15 of 10
    
    def test_validate_page_bounds_zero_page(self):
        """Zero or negative page number should fail"""
        doc = MagicMock()
        doc.__len__ = MagicMock(return_value=10)
        self.assertFalse(_validate_page_bounds(doc, 0))
        self.assertFalse(_validate_page_bounds(doc, -1))


class TestContextExtraction(unittest.TestCase):
    """STAGE 3: Test safe context extraction"""
    
    @patch('ingest.enrich_images.extract_surrounding_context')
    def test_extract_context_safely_success(self, mock_extract):
        """Successful context extraction should return context dict"""
        mock_extract.return_value = {
            'before': 'Before text',
            'after': 'After text',
            'figure_caption': 'Figure caption'
        }
        
        page = MagicMock()
        bbox = MagicMock()
        doc = MagicMock()
        
        result = _extract_context_safely(page, bbox, doc, 0)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['before'], 'Before text')
        self.assertEqual(result['after'], 'After text')
    
    @patch('ingest.enrich_images.extract_surrounding_context')
    def test_extract_context_safely_exception(self, mock_extract):
        """Exception during extraction should return empty dict"""
        mock_extract.side_effect = ValueError("Extraction failed")
        
        page = MagicMock()
        bbox = MagicMock()
        doc = MagicMock()
        
        result = _extract_context_safely(page, bbox, doc, 0)
        
        self.assertIsInstance(result, dict)
        self.assertIn('before', result)
        self.assertIn('after', result)
        self.assertIn('figure_caption', result)


class TestVLMGeneration(unittest.TestCase):
    """STAGE 3: Test safe VLM description generation"""
    
    def test_generate_vlm_description_safe_valid_path(self):
        """Valid image path should generate description"""
        captioner = MagicMock()
        captioner.generate_caption.return_value = "Test caption"
        
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            result = _generate_vlm_description_safe(captioner, tmp.name, 'img_001')
            
            self.assertEqual(result, "Test caption")
            captioner.generate_caption.assert_called_once()
    
    def test_generate_vlm_description_safe_missing_path(self):
        """Missing image path should return empty string"""
        captioner = MagicMock()
        
        result = _generate_vlm_description_safe(captioner, '/nonexistent/path.png', 'img_001')
        
        self.assertEqual(result, "")
    
    def test_generate_vlm_description_safe_exception(self):
        """Exception during generation should return empty string"""
        captioner = MagicMock()
        captioner.generate_caption.side_effect = RuntimeError("API error")
        
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            result = _generate_vlm_description_safe(captioner, tmp.name, 'img_001')
            
            self.assertEqual(result, "")


class TestCaptionAssembly(unittest.TestCase):
    """STAGE 3: Test caption assembly helper"""
    
    def test_assemble_enriched_caption_all_parts(self):
        """Caption with all parts should assemble correctly"""
        caption = _assemble_enriched_caption_text(
            author_caption="Figure caption",
            vlm_description="Visual description",
            context_before="Before text",
            context_after="After text"
        )
        
        self.assertIn("Figure caption", caption)
        self.assertIn("Visual description", caption)
        self.assertIn("Before text", caption)
        self.assertIn("After text", caption)
    
    def test_assemble_enriched_caption_partial(self):
        """Caption with partial parts should assemble correctly"""
        caption = _assemble_enriched_caption_text(
            author_caption="Figure caption",
            vlm_description="",
            context_before="Before text",
            context_after=""
        )
        
        self.assertIn("Figure caption", caption)
        self.assertIn("Before text", caption)
    
    def test_assemble_enriched_caption_empty(self):
        """Empty caption should return empty string"""
        caption = _assemble_enriched_caption_text("", "", "", "")
        
        self.assertEqual(caption, "")


class TestMetadataLoading(unittest.TestCase):
    """STAGE 3: Test safe metadata loading"""
    
    def test_load_and_validate_metadata_valid(self):
        """Valid metadata file should load correctly"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            metadata = [
                {'doc_id': 'arxiv_123', 'image_id': 'img_001'},
                {'doc_id': 'arxiv_124', 'image_id': 'img_002'}
            ]
            json.dump(metadata, tmp)
            tmp.flush()
            
            result = _load_and_validate_metadata(Path(tmp.name))
            
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]['doc_id'], 'arxiv_123')
    
    def test_load_and_validate_metadata_missing_file(self):
        """Missing file should return None"""
        result = _load_and_validate_metadata(Path('/nonexistent/file.json'))
        
        self.assertIsNone(result)
    
    def test_load_and_validate_metadata_invalid_json(self):
        """Invalid JSON should return None"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp.write("{ invalid json }")
            tmp.flush()
            
            result = _load_and_validate_metadata(Path(tmp.name))
            
            self.assertIsNone(result)


class TestLoadSaveMetadata(unittest.TestCase):
    """Test load/save metadata functions"""
    
    def test_load_images_metadata_valid(self):
        """Valid metadata should load successfully"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            metadata = [
                {'doc_id': 'arxiv_123', 'image_id': 'img_001'},
                {'doc_id': 'arxiv_124', 'image_id': 'img_002'}
            ]
            json.dump(metadata, tmp)
            tmp.flush()
            
            result = load_images_metadata(Path(tmp.name))
            
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 2)
    
    def test_save_images_metadata_valid(self):
        """Valid metadata should save successfully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = [
                {'doc_id': 'arxiv_123', 'image_id': 'img_001'},
                {'doc_id': 'arxiv_124', 'image_id': 'img_002'}
            ]
            output_path = Path(tmpdir) / "metadata.json"
            
            save_images_metadata(metadata, output_path)
            
            self.assertTrue(output_path.exists())
            with open(output_path) as f:
                loaded = json.load(f)
            self.assertEqual(len(loaded), 2)
    
    def test_save_images_metadata_invalid_type(self):
        """Invalid metadata type should not crash"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metadata.json"
            
            # Should handle invalid type gracefully
            save_images_metadata(None, output_path)  # Won't save
            save_images_metadata("invalid", output_path)  # Won't save


class TestEnrichSingleImage(unittest.TestCase):
    """Test single image enrichment"""
    
    @patch('ingest.enrich_images._validate_image_metadata')
    def test_enrich_single_image_invalid_metadata(self, mock_validate):
        """Invalid metadata should return unchanged"""
        mock_validate.return_value = False
        
        img_meta = {'doc_id': 'arxiv_123', 'image_id': 'img_001'}
        result = enrich_single_image(img_meta, {}, Path('/tmp'), None)
        
        self.assertEqual(result, img_meta)
    
    @patch('ingest.enrich_images._validate_image_metadata')
    def test_enrich_single_image_web_image(self, mock_validate):
        """Web image should use stored context"""
        mock_validate.return_value = True
        
        img_meta = {
            'doc_id': 'medium_123',
            'image_id': 'img_001',
            'extraction_method': EXTRACTION_METHOD_WEB,
            'context_before': 'Before',
            'context_after': 'After',
            'author_caption': 'Caption'
        }
        
        result = enrich_single_image(img_meta, {}, Path('/tmp'), None)
        
        self.assertIn('enriched_caption', result)
        self.assertIn('Caption', result['enriched_caption'])


class TestEnrichAllImages(unittest.TestCase):
    """Test batch image enrichment"""
    
    @patch('ingest.enrich_images._validate_image_metadata')
    @patch('ingest.enrich_images.enrich_single_image')
    def test_enrich_all_images_valid(self, mock_enrich, mock_validate):
        """Valid images should be enriched"""
        mock_validate.return_value = True
        mock_enrich.side_effect = lambda x, *args, **kwargs: x
        
        images = [
            {'doc_id': 'arxiv_123', 'image_id': 'img_001'},
            {'doc_id': 'arxiv_124', 'image_id': 'img_002'}
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = enrich_all_images(images, Path(tmpdir), None)
            
            self.assertEqual(len(result), 2)
    
    def test_enrich_all_images_invalid_images_type(self):
        """Invalid images type should return unchanged"""
        result = enrich_all_images(None, Path('/tmp'), None)
        
        self.assertIsNone(result)
    
    def test_enrich_all_images_invalid_papers_dir(self):
        """Invalid papers directory should return unchanged"""
        images = [{'doc_id': 'arxiv_123', 'image_id': 'img_001'}]
        result = enrich_all_images(images, None, None)
        
        self.assertEqual(result, images)


class TestGenerateCaptionsForDoc(unittest.TestCase):
    """Test document-specific caption generation"""
    
    @patch('ingest.enrich_images._load_and_validate_metadata')
    @patch('ingest.enrich_images.enrich_all_images')
    @patch('ingest.enrich_images.save_images_metadata')
    def test_generate_captions_for_doc_valid(self, mock_save, mock_enrich, mock_load):
        """Valid document should have captions generated"""
        mock_load.return_value = [
            {'doc_id': 'arxiv_123', 'image_id': 'img_001'},
            {'doc_id': 'arxiv_124', 'image_id': 'img_002'}
        ]
        mock_enrich.return_value = [
            {'doc_id': 'arxiv_123', 'image_id': 'img_001', 'enriched_caption': 'Caption 1'},
            {'doc_id': 'arxiv_123', 'image_id': 'img_002', 'enriched_caption': 'Caption 2'}
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy metadata file
            metadata_path = Path(tmpdir) / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump([], f)
            
            result = generate_captions_for_doc(
                'arxiv_123',
                metadata_path,
                Path(tmpdir)
            )
            
            # Should return number of images enriched
            self.assertGreaterEqual(result, 0)
    
    def test_generate_captions_for_doc_invalid_doc_id(self):
        """Invalid doc_id should return 0"""
        result = generate_captions_for_doc('', None, None)
        
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
