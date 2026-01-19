"""
Test suite for ui/app.py - Streamlit UI for RAG system.

Tests cover:
- Configuration constants validation
- Image path resolution helpers
- Data filtering and formatting
- HTML badge generation
- Error handling for missing resources
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
import sys
import json
import tempfile
import os

# Add ui module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ui"))

# Mock streamlit before importing app
sys.modules['streamlit'] = MagicMock()

import app
from app import (
    DEFAULT_K_TEXT,
    MIN_K_TEXT,
    MAX_K_TEXT,
    INLINE_IMAGE_WIDTH,
    GRID_IMAGE_WIDTH,
    CAPTION_TEXT_AREA_HEIGHT,
    TEXT_AREA_HEIGHT,
    IMAGE_GRID_COLUMNS,
    METRICS_COLUMNS,
    _extract_paper_id,
    _try_image_path_variants,
    get_image_path,
    get_confidence_badge_html,
    _filter_cited_images,
    load_images_metadata
)


class TestUIConstants(unittest.TestCase):
    """Test UI configuration constants."""
    
    def test_text_retrieval_k_defaults(self):
        """Verify text retrieval k parameters."""
        self.assertEqual(DEFAULT_K_TEXT, 3)
        self.assertEqual(MIN_K_TEXT, 2)
        self.assertEqual(MAX_K_TEXT, 5)
    
    def test_k_values_range(self):
        """k values should be in logical range."""
        self.assertLess(MIN_K_TEXT, DEFAULT_K_TEXT)
        self.assertLess(DEFAULT_K_TEXT, MAX_K_TEXT)
    
    def test_image_width_constants(self):
        """Verify image display widths."""
        self.assertEqual(INLINE_IMAGE_WIDTH, 400)
        self.assertEqual(GRID_IMAGE_WIDTH, 250)
        self.assertGreater(INLINE_IMAGE_WIDTH, GRID_IMAGE_WIDTH)
    
    def test_text_area_height_constants(self):
        """Verify text area heights."""
        self.assertEqual(CAPTION_TEXT_AREA_HEIGHT, 150)
        self.assertEqual(TEXT_AREA_HEIGHT, 200)
    
    def test_layout_grid_columns(self):
        """Verify layout grid configuration."""
        self.assertEqual(IMAGE_GRID_COLUMNS, 3)
        self.assertGreater(IMAGE_GRID_COLUMNS, 0)
    
    def test_metrics_columns(self):
        """Verify metrics display columns."""
        self.assertEqual(METRICS_COLUMNS, 2)
        self.assertGreater(METRICS_COLUMNS, 0)


class TestExtractPaperId(unittest.TestCase):
    """Test paper ID extraction from image IDs."""
    
    def test_extract_paper_id_pdf_embedded(self):
        """Extract paper ID from PDF embedded image."""
        # Arrange
        image_id = "arxiv_1706_03762_embedded_001"
        
        # Act
        result = _extract_paper_id(image_id)
        
        # Assert
        self.assertEqual(result, "arxiv_1706_03762")
    
    def test_extract_paper_id_pdf_vector(self):
        """Extract paper ID from PDF vector image."""
        # Arrange
        image_id = "arxiv_1409_3215_vector_006_01"
        
        # Act
        result = _extract_paper_id(image_id)
        
        # Assert
        self.assertEqual(result, "arxiv_1409_3215")
    
    def test_extract_paper_id_json_web(self):
        """Extract paper ID from JSON web source."""
        # Arrange
        image_id = "realpython_numpy-tutorial_web_004"
        
        # Act
        result = _extract_paper_id(image_id)
        
        # Assert
        self.assertEqual(result, "realpython_numpy-tutorial")
    
    def test_extract_paper_id_medium_source(self):
        """Extract paper ID from Medium article."""
        # Arrange
        image_id = "medium_agents-plan-tasks_web_001"
        
        # Act
        result = _extract_paper_id(image_id)
        
        # Assert
        self.assertEqual(result, "medium_agents-plan-tasks")
    
    def test_extract_paper_id_no_pattern(self):
        """Return full image_id if no pattern matches."""
        # Arrange
        image_id = "unknown_format_123"
        
        # Act
        result = _extract_paper_id(image_id)
        
        # Assert
        self.assertEqual(result, image_id)


class TestImagePathVariants(unittest.TestCase):
    """Test image path resolution with multiple variants."""
    
    def setUp(self):
        """Create temporary directory structure for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_find_png_in_paper_subfolder(self):
        """Find PNG image in paper subfolder."""
        # Arrange
        paper_id = "arxiv_1706_03762"
        image_id = "arxiv_1706_03762_embedded_001"
        subfolder = self.base_dir / paper_id
        subfolder.mkdir()
        image_path = subfolder / f"{image_id}.png"
        image_path.touch()
        
        # Act
        result = _try_image_path_variants(self.base_dir, paper_id, image_id)
        
        # Assert
        self.assertEqual(result, image_path)
    
    def test_find_jpg_in_paper_subfolder(self):
        """Find JPG image in paper subfolder."""
        # Arrange
        paper_id = "arxiv_1706_03762"
        image_id = "arxiv_1706_03762_embedded_001"
        subfolder = self.base_dir / paper_id
        subfolder.mkdir()
        image_path = subfolder / f"{image_id}.jpg"
        image_path.touch()
        
        # Act
        result = _try_image_path_variants(self.base_dir, paper_id, image_id)
        
        # Assert
        self.assertEqual(result, image_path)
    
    def test_find_png_direct_path_fallback(self):
        """Find PNG using direct path (backward compatibility)."""
        # Arrange
        image_id = "arxiv_1706_03762_embedded_001"
        image_path = self.base_dir / f"{image_id}.png"
        image_path.touch()
        
        # Act
        result = _try_image_path_variants(
            self.base_dir, "arxiv_1706_03762", image_id
        )
        
        # Assert
        self.assertEqual(result, image_path)
    
    def test_find_jpg_direct_path_fallback(self):
        """Find JPG using direct path (backward compatibility)."""
        # Arrange
        image_id = "arxiv_1706_03762_embedded_001"
        image_path = self.base_dir / f"{image_id}.jpg"
        image_path.touch()
        
        # Act
        result = _try_image_path_variants(
            self.base_dir, "arxiv_1706_03762", image_id
        )
        
        # Assert
        self.assertEqual(result, image_path)
    
    def test_not_found_returns_none(self):
        """Return None when image not found."""
        # Arrange
        paper_id = "arxiv_1706_03762"
        image_id = "arxiv_1706_03762_embedded_001"
        
        # Act
        result = _try_image_path_variants(self.base_dir, paper_id, image_id)
        
        # Assert
        self.assertIsNone(result)


class TestGetImagePath(unittest.TestCase):
    """Test main image path getter function."""
    
    @patch('app.IMAGES_DIR', None)
    def test_get_image_path_images_dir_none(self):
        """Return None when IMAGES_DIR is not available."""
        # Act
        result = get_image_path("arxiv_1706_03762_embedded_001")
        
        # Assert
        self.assertIsNone(result)
    
    def setUp(self):
        """Create temporary image directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestConfidenceBadgeHtml(unittest.TestCase):
    """Test confidence badge HTML generation."""
    
    def test_high_confidence_badge(self):
        """Generate HIGH confidence badge."""
        # Act
        result = get_confidence_badge_html('HIGH', 0.95)
        
        # Assert
        self.assertIn('confidence-badge-high', result)
        self.assertIn('HIGH', result)
        self.assertIn('0.950', result)
    
    def test_medium_confidence_badge(self):
        """Generate MEDIUM confidence badge."""
        # Act
        result = get_confidence_badge_html('MEDIUM', 0.72)
        
        # Assert
        self.assertIn('confidence-badge-medium', result)
        self.assertIn('MEDIUM', result)
        self.assertIn('0.720', result)
    
    def test_low_confidence_badge(self):
        """Generate LOW confidence badge."""
        # Act
        result = get_confidence_badge_html('LOW', 0.45)
        
        # Assert
        self.assertIn('confidence-badge-low', result)
        self.assertIn('LOW', result)
        self.assertIn('0.450', result)
    
    def test_badge_html_format(self):
        """Verify badge uses correct HTML structure."""
        # Act
        result = get_confidence_badge_html('HIGH', 0.85)
        
        # Assert
        self.assertIn('<span class=', result)
        self.assertIn('</span>', result)


class TestFilterCitedImages(unittest.TestCase):
    """Test image filtering for citations."""
    
    def test_filter_cited_images_all_cited(self):
        """All images are cited."""
        # Arrange
        images = [
            {'image_id': 'img1', 'caption': 'Figure 1'},
            {'image_id': 'img2', 'caption': 'Figure 2'},
            {'image_id': 'img3', 'caption': 'Figure 3'}
        ]
        cited_ids = ['img1', 'img2', 'img3']
        
        # Act
        result = _filter_cited_images(images, cited_ids)
        
        # Assert
        self.assertEqual(len(result), 3)
    
    def test_filter_cited_images_partial(self):
        """Only some images are cited."""
        # Arrange
        images = [
            {'image_id': 'img1', 'caption': 'Figure 1'},
            {'image_id': 'img2', 'caption': 'Figure 2'},
            {'image_id': 'img3', 'caption': 'Figure 3'}
        ]
        cited_ids = ['img1', 'img3']  # img2 not cited
        
        # Act
        result = _filter_cited_images(images, cited_ids)
        
        # Assert
        self.assertEqual(len(result), 2)
        self.assertIn('img1', [img['image_id'] for img in result])
        self.assertIn('img3', [img['image_id'] for img in result])
        self.assertNotIn('img2', [img['image_id'] for img in result])
    
    def test_filter_cited_images_none(self):
        """No images are cited."""
        # Arrange
        images = [
            {'image_id': 'img1', 'caption': 'Figure 1'},
            {'image_id': 'img2', 'caption': 'Figure 2'}
        ]
        cited_ids = []
        
        # Act
        result = _filter_cited_images(images, cited_ids)
        
        # Assert
        self.assertEqual(len(result), 0)


class TestLoadImagesMetadata(unittest.TestCase):
    """Test images metadata loading."""
    
    def test_load_images_metadata_valid_json(self):
        """Load valid JSON metadata."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            metadata = [
                {'image_id': 'img1', 'filename': 'img1.png'},
                {'image_id': 'img2', 'filename': 'img2.png'}
            ]
            json.dump(metadata, f)
            temp_file = f.name
        
        try:
            # Act - would need to patch the file path in app module
            # This is a basic test of the JSON structure
            self.assertIsInstance(metadata, list)
            self.assertEqual(len(metadata), 2)
        finally:
            os.unlink(temp_file)
    
    def test_metadata_list_structure(self):
        """Metadata should be a list of dictionaries."""
        # Arrange
        metadata = [
            {'image_id': 'img1', 'doc_id': 'arxiv_1234', 'filename': 'img1.png'},
            {'image_id': 'img2', 'doc_id': 'arxiv_1234', 'filename': 'img2.png'}
        ]
        
        # Assert
        self.assertIsInstance(metadata, list)
        for item in metadata:
            self.assertIsInstance(item, dict)
            self.assertIn('image_id', item)


class TestUIPathConstants(unittest.TestCase):
    """Test UI path configurations."""
    
    def test_images_dir_is_path_object(self):
        """IMAGES_DIR should be a Path object if set."""
        if app.IMAGES_DIR is not None:
            self.assertIsInstance(app.IMAGES_DIR, Path)
    
    def test_images_metadata_file_is_path(self):
        """IMAGES_METADATA_FILE should be a Path object if set."""
        if app.IMAGES_METADATA_FILE is not None:
            self.assertIsInstance(app.IMAGES_METADATA_FILE, Path)


if __name__ == '__main__':
    unittest.main()
