"""
Generate enriched captions for images using OpenAI GPT-4.1-mini  API.

This module combines:
1. Vision-LM generated descriptions (what's visually in the image)
2. Author-provided captions (if available)  
3. Surrounding text context (narrative that explains the image)

The enriched captions are then embedded as text for unified retrieval.
"""

import base64
import logging
import os
from pathlib import Path
from typing import Dict, Optional
from openai import OpenAI
import openai
from PIL import Image
import io

# Import centralized configuration
from config import VISION


# ============================================================================
# CONFIGURATION CONSTANTS - STAGE 2: Constants instead of magic numbers
# ============================================================================

# Image processing constants
MAX_IMAGE_SIZE = 1024               # pixels - maximum width/height for image encoding
JPEG_QUALITY = 85                   # JPEG compression quality (1-100)
JPEG_OPTIMIZATION = True            # Optimize JPEG encoding for smaller file size

# API response constants
MAX_CAPTION_TOKENS = 1024          # tokens - maximum length for caption generation

# Valid image modes for conversion
VALID_IMAGE_MODES_FOR_CONVERSION = ('RGBA', 'LA', 'P')  # Modes requiring RGB conversion
TARGET_IMAGE_MODE = 'RGB'          # Target mode after conversion

# Image format
IMAGE_FORMAT = 'JPEG'              # Image format for encoding
IMAGE_DATA_URL_FORMAT = 'data:image/jpeg;base64,{}'  # Data URL format for base64


# ============================================================================
# HELPER FUNCTIONS - STAGE 3: Extract DRY/SRP principles
# ============================================================================

def _validate_image_path(image_path: str) -> bool:
    """
    STAGE 1: Validation - check if image file exists and is readable.
    STAGE 3: DRY - extract path validation logic.
    
    Args:
        image_path: Path to image file
    
    Returns:
        True if file exists and is readable, False otherwise
    """
    try:
        path = Path(image_path)
        if not path.exists():
            logging.error(f"Image file not found: {image_path}")
            return False
        if not path.is_file():
            logging.error(f"Path is not a file: {image_path}")
            return False
        if not os.access(path, os.R_OK):
            logging.error(f"Image file is not readable: {image_path}")
            return False
        return True
    except Exception as e:
        logging.error(f"Error validating image path {image_path}: {e}")
        return False


def _resize_and_convert_image(img: Image.Image, max_size: int) -> Image.Image:
    """
    
    Resize image and convert to RGB if needed.
    
    Args:
        img: PIL Image object
        max_size: Maximum width/height in pixels
    
    Returns:
        Resized and converted PIL Image
    """
    # STAGE 1: Validate max_size
    if not isinstance(max_size, int) or max_size <= 0:
        logging.warning(f"Invalid max_size {max_size}, using default {MAX_IMAGE_SIZE}")
        max_size = MAX_IMAGE_SIZE
    
    # Resize if needed
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed
    if img.mode in VALID_IMAGE_MODES_FOR_CONVERSION:
        img = img.convert(TARGET_IMAGE_MODE)
    
    return img


def _encode_image_to_base64(img: Image.Image) -> Optional[str]:
    """
    Compress image and encode to base64.
    
    Args:
        img: PIL Image object (should be in RGB mode)
    
    Returns:
        Base64 encoded string, or None if encoding fails
    """
    try:
        buffer = io.BytesIO()
        img.save(
            buffer,
            format=IMAGE_FORMAT,
            quality=JPEG_QUALITY,
            optimize=JPEG_OPTIMIZATION
        )
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Failed to encode image to base64: {e}")
        return None


def _assemble_enriched_caption(
    author_caption: str,
    vlm_description: str,
    context_text: str
) -> str:
    """
    
    Assemble enriched caption from multiple sources.
    
    Args:
        author_caption: Figure caption from paper
        vlm_description: Vision model generated description
        context_text: Surrounding text context
    
    Returns:
        Combined enriched caption text
    """
    # STAGE 1: Validate inputs
    if not isinstance(author_caption, str):
        author_caption = ""
    if not isinstance(vlm_description, str):
        logging.warning("vlm_description is not a string")
        vlm_description = ""
    if not isinstance(context_text, str):
        context_text = ""
    
    enriched = ""
    
    # Add author caption
    if author_caption.strip():
        enriched += f"Figure caption: {author_caption.strip()}\n"
    
    # Add vision model description
    if vlm_description.strip():
        enriched += f"Visual description: {vlm_description.strip()}\n"
    
    # Add context
    if context_text.strip():
        enriched += f"Context: {context_text.strip()}\n"
    
    # Add disclaimer
    enriched += "\nNote: Use only context text that is relevant to understanding this image. Ignore surrounding text if it discusses unrelated topics."
    
    return enriched.strip()


class ImageCaptioner:
    """
    Vision-Language Model for generating detailed image descriptions.
    
    Uses OpenAI GPT-4.1-mini  to describe diagrams, charts, 
    neural network architectures, formulas, and other visual content in AI/ML papers.
    
    Model: gpt-4.1-mini
    Specializes in: charts, graphs, diagrams, tables, OCR, document Q&A
    """
    
    def __init__(
        self, 
        model_name: str = VISION.MODEL,
        api_key: Optional[str] = None
    ):
        """
        Initialize OpenAI Vision API client.
        
        Args:
            model_name: OpenAI model identifier
                - "gpt-4.1-mini" 
            api_key: OpenAI API key (or set OPENAI_API_KEY env variable)
        """
        self.model_name = model_name

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            # This should be caught by validate_environment() at startup
            raise ValueError(
                "OpenAI API key not found. "
                "This should have been validated at pipeline startup."
            )
        
        self.client = OpenAI(api_key=api_key)
        logging.info(f"✅ OpenAI Vision API initialized: {model_name}")

        self.default_prompt = """Describe this technical image in detail for AI/ML developers and data scientists.

        Identify the image type and describe accordingly:

        **Neural Network Architecture:**
        - Name specific components (Transformer, Conv layers, LSTM, Attention, etc.)
        - Describe data flow and connections between blocks
        - Mention input/output dimensions if visible
        - Note skip connections, residual blocks, or special structures

        **Charts & Graphs (training curves, comparisons):**
        - Identify what's being measured (accuracy, loss, performance metrics)
        - Describe axis labels and scales
        - Note trends (increasing, converging, plateauing)
        - Compare multiple lines/bars if present
        - Mention specific values or ranges if readable

        **Tables (experiment results, benchmarks):**
        - Describe what metrics are compared
        - Identify best/worst performing methods
        - Note dataset names or test conditions
        - Mention key numerical results if clearly visible

        **Process Diagrams & Workflows:**
        - Describe the overall process or pipeline flow
        - Name main components and their relationships
        - Explain what each step does
        - Identify input/output of each stage

        **Code Screenshots & IDE Interfaces:**
        - Describe the tool/environment shown (Jupyter, VS Code, Repl.it, etc.)
        - Mention visible UI elements (panels, buttons, menus)
        - Note any code snippets visible and what they demonstrate
        - Identify file names, tabs, or project structure if shown

        **Data Visualizations (plots, heatmaps, feature maps):**
        - Describe what data is being visualized
        - Explain what colors/intensities represent
        - Note patterns or interesting regions
        - Mention color scales or legends

        **System Architecture & Design Diagrams:**
        - Describe main components and services
        - Explain data flow between components
        - Note databases, APIs, or external systems
        - Identify communication patterns

        **Always include:**
        - Any text labels, legends, or annotations visible
        - Tool names, library names, or framework names shown
        - Mathematical notation or formulas if present
        - Code snippets or terminal output if visible
        - Specific model/dataset/method names mentioned

        Be specific and technical. Focus on details useful for searching and understanding the content."""
    
    def encode_image(self, image_path: str, max_size: int = MAX_IMAGE_SIZE) -> Optional[str]:
        """
        Encode image to base64 with compression if needed.
      
        Args:
            image_path: Path to image file
            max_size: Maximum width/height in pixels (default MAX_IMAGE_SIZE)
            
        Returns:
            Base64 encoded image string, or None if encoding fails
        """
        # STAGE 1: Validate image path
        if not isinstance(image_path, str):
            logging.error(f"image_path must be str, got {type(image_path).__name__}")
            return None
        
        if not _validate_image_path(image_path):
            return None
        
        try:
            # STAGE 3: Use helper for path validation
            img = Image.open(image_path)
            
            # STAGE 3: Use helper for resize and convert
            img = _resize_and_convert_image(img, max_size)
            
            # STAGE 3: Use helper for base64 encoding
            base64_image = _encode_image_to_base64(img)
            
            return base64_image
        
        except IOError as e:
            logging.error(f"Failed to open image {image_path}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error encoding image {image_path}: {type(e).__name__}: {e}")
            return None
    
    def generate_caption(
        self, 
        image_path: str, 
        max_length: int = MAX_CAPTION_TOKENS,
        prompt: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate detailed description of an image using OpenAI Vision API.
        
        STAGE 1: Validation - validate all parameters
        STAGE 2: Use constants instead of magic numbers
        
        Args:
            image_path: Path to image file
            max_length: Maximum tokens in description
            prompt: Optional custom prompt (uses default_prompt if None)
            
        Returns:
            Detailed text description of the image, or None if generation fails
        """
        # STAGE 1: Validate inputs
        if not isinstance(image_path, str):
            logging.error(f"image_path must be str, got {type(image_path).__name__}")
            return None
        
        if not isinstance(max_length, int) or max_length <= 0:
            logging.warning(f"Invalid max_length {max_length}, using default {MAX_CAPTION_TOKENS}")
            max_length = MAX_CAPTION_TOKENS
        
        # STAGE 2: Use constant for max_size
        image_base64 = self.encode_image(image_path, max_size=MAX_IMAGE_SIZE)
        
        if image_base64 is None:
            logging.error(f"Failed to encode image {image_path}")
            return None

        # Use provided prompt or default
        if prompt is None:
            used_prompt = self.default_prompt
        elif not isinstance(prompt, str):
            logging.warning("Custom prompt is not a string, using default")
            used_prompt = self.default_prompt
        else:
            used_prompt = prompt

        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": used_prompt},
                            {
                                "type": "input_image",
                                "image_url": IMAGE_DATA_URL_FORMAT.format(image_base64)
                            }
                        ]
                    }
                ]
            )
            
            caption = response.output_text
            return caption.strip() if isinstance(caption, str) else None
            
        except openai.AuthenticationError:
            logging.error("OpenAI authentication failed. Check API key validity.")
            return None
        except openai.RateLimitError:
            logging.error("OpenAI rate limit exceeded. Retry later or reduce request frequency.")
            return None
        except openai.APIConnectionError:
            logging.error("OpenAI API connection error. Check network connectivity.")
            return None
        except openai.APIError as e:
            logging.error(f"OpenAI API error: {type(e).__name__}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error generating caption: {type(e).__name__}: {e}")
            return None


def create_enriched_caption(
    image_id: str,
    image_path: str,
    context_dict: Dict[str, Optional[str]],
    captioner: ImageCaptioner
) -> Optional[Dict[str, str]]:
    """
    Create enriched caption combining multiple sources.
    
    STAGE 1: Validation - validate all parameters
    STAGE 3: DRY - use helper function for caption assembly
    
    Args:
        image_id: Unique identifier for the image
        image_path: Path to image file
        context_dict: {
            "before": "Text before image",
            "after": "Text after image",  
            "figure_caption": "Figure X: ..." or None
        }
        captioner: ImageCaptioner instance
        
    Returns:
        {
            "image_id": "arxiv_1706_03762_vector_003_01",
            "enriched_caption": "Combined caption for embedding",
            "vlm_description": "OpenAI generated description",
            "author_caption": "Figure caption from paper",
            "context_text": "Surrounding text"
        }
        Returns None if caption generation fails
    """
    # STAGE 1: Validate inputs
    if not isinstance(image_id, str) or not image_id.strip():
        logging.error("image_id must be non-empty string")
        return None
    
    if not isinstance(image_path, str):
        logging.error(f"image_path must be str, got {type(image_path).__name__}")
        return None
    
    if not isinstance(context_dict, dict):
        logging.error(f"context_dict must be dict, got {type(context_dict).__name__}")
        return None
    
    if not isinstance(captioner, ImageCaptioner):
        logging.error("captioner must be ImageCaptioner instance")
        return None
    
    try:
        logging.info(f"Generating caption for {Path(image_path).name}...")
        vlm_description = captioner.generate_caption(image_path)
        
        if vlm_description is None:
            logging.warning(f"Failed to generate caption for {image_path}")
            vlm_description = ""

        # Extract context fields with type validation
        author_caption = context_dict.get("figure_caption", "")
        if not isinstance(author_caption, str):
            author_caption = ""
        
        before_text = context_dict.get("before", "")
        if not isinstance(before_text, str):
            before_text = ""
        
        after_text = context_dict.get("after", "")
        if not isinstance(after_text, str):
            after_text = ""

        # Assemble context text
        context_text = ""
        if before_text.strip():
            context_text += before_text.strip()
        if after_text.strip():
            if context_text:
                context_text += " ... "
            context_text += after_text.strip()

        # STAGE 3: Use helper function to assemble caption
        enriched_caption = _assemble_enriched_caption(
            author_caption,
            vlm_description,
            context_text
        )
        
        return {
            "image_id": image_id,
            "enriched_caption": enriched_caption,
            "vlm_description": vlm_description,
            "author_caption": author_caption,
            "context_text": context_text
        }
    
    except Exception as e:
        logging.error(f"Error creating enriched caption for {image_id}: {type(e).__name__}: {e}")
        return None


if __name__ == "__main__":
    # Test caption generation
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_captions.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Initialize captioner
    captioner = ImageCaptioner()
    
    # Generate caption
    caption = captioner.generate_caption(image_path)
    
    print(f"\nGenerated Caption:")
    print(caption)
    
    # Create enriched caption with mock context
    mock_context = {
        "figure_caption": "Figure 1: The Transformer model architecture",
        "before": "The model uses self-attention mechanisms to process sequences.",
        "after": "As shown in the figure, the encoder stack consists of 6 layers."
    }
    
    enriched = create_enriched_caption(
        "test_image_001",
        image_path,
        mock_context,
        captioner
    )
    
    print(f"\nEnriched Caption:")
    print(enriched['enriched_caption'])
