"""
Generate enriched captions for images using OpenAI GPT-4.1-mini  API.

This module combines:
1. Vision-LM generated descriptions (what's visually in the image)
2. Author-provided captions (if available)  
3. Surrounding text context (narrative that explains the image)

The enriched captions are then embedded as text for unified retrieval.
"""

import base64
import os
from pathlib import Path
from typing import Dict, Optional
from openai import OpenAI
from PIL import Image
import io


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
        model_name: str = "gpt-4.1-mini",
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
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=api_key)
        print(f"✅ OpenAI Vision API initialized: {model_name}")

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
    
    def encode_image(self, image_path: str, max_size: int = 1024) -> str:
        """
        Encode image to base64 with compression if needed.
        
        Args:
            image_path: Path to image file
            max_size: Maximum width/height in pixels (default 1024)
            
        Returns:
            Base64 encoded image string
        """
        img = Image.open(image_path)

        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85, optimize=True)
        buffer.seek(0)
        
        return base64.b64encode(buffer.read()).decode("utf-8")
    
    def generate_caption(
        self, 
        image_path: str, 
        max_length: int = 1024,
        prompt: Optional[str] = None
    ) -> str:
        """
        Generate detailed description of an image using OpenAI Vision API.
        
        Args:
            image_path: Path to image file
            max_length: Maximum tokens in description
            prompt: Optional custom prompt (uses default_prompt if None)
            
        Returns:
            Detailed text description of the image
        """
        image_base64 = self.encode_image(image_path, max_size=1024)

        used_prompt = prompt if prompt is not None else self.default_prompt

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
                                "image_url": f"data:image/png;base64,{image_base64}"
                            }
                        ]
                    }
                ]
            )
            
            caption = response.output_text
            return caption.strip()
            
        except Exception as e:
            print(f"  ⚠️  OpenAI API error: {e}")
            return "Error generating caption"


def create_enriched_caption(
    image_id: str,
    image_path: str,
    context_dict: Dict[str, Optional[str]],
    captioner: ImageCaptioner
) -> Dict[str, str]:
    """
    Create enriched caption combining multiple sources.
    
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
    """
    print(f"  Generating caption for {Path(image_path).name}...")
    vlm_description = captioner.generate_caption(image_path)

    author_caption = context_dict.get("figure_caption", "")
    before_text = context_dict.get("before", "")
    after_text = context_dict.get("after", "")

    context_text = ""
    if before_text:
        context_text += before_text
    if after_text:
        if context_text:
            context_text += " ... "
        context_text += after_text

    enriched_caption = ""
    
    if author_caption:
        enriched_caption += f"Figure caption: {author_caption}\n"
    
    enriched_caption += f"Visual description: {vlm_description}\n"
    
    if context_text:
        enriched_caption += f"Context: {context_text}\n"

    enriched_caption += "\nNote: Use only context text that is relevant to understanding this image. Ignore surrounding text if it discusses unrelated topics."
    
    return {
        "image_id": image_id,
        "enriched_caption": enriched_caption,
        "vlm_description": vlm_description,
        "author_caption": author_caption or "",
        "context_text": context_text
    }


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
