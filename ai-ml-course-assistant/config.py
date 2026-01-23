"""
Centralized configuration for AI/ML Course Assistant.

This module contains all shared constants and configuration settings
used across multiple modules. Module-specific constants should remain
in their respective files.

Last Updated: 2026-01-21
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# === Base Paths ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_BACKUP_DIR = DATA_DIR / "chunks_backup"


# === Embedding Configuration ===
@dataclass
class EmbeddingConfig:
    """Configuration for text embeddings (used across indexing, retrieval, generation)."""
    
    MODEL: str = "text-embedding-3-small"
    """OpenAI embedding model for all text and image captions."""
    
    DIMENSIONS: int = 1536
    """Embedding vector dimensions."""
    
    BATCH_SIZE: int = 100
    """Batch size for embedding API calls (cost optimization)."""


# === LLM Models ===
@dataclass
class LLMModels:
    """OpenAI model names (centralized for easy version updates)."""
    
    GENERATOR: str = "gpt-5-mini"
    """Main generation model for answering user queries (high quality, 128K context)."""
    
    QUERY_ANALYZER: str = "gpt-5-nano"
    """Fast model for query classification and structured output."""
    
    VISION: str = "gpt-4.1-mini"
    """Vision model for image caption generation (has vision capabilities)."""
    
    JUDGE: str = "gpt-5-mini"
    """Model for faithfulness evaluation."""


# === Generator Configuration ===
@dataclass
class GeneratorConfig:
    """Configuration for answer generation (rag/generator.py)."""
    
    MODEL: str = LLMModels.GENERATOR
    """Model for generation."""
    
    TEMPERATURE: float = 0.0
    """Zero temperature for maximum determinism (anti-hallucination)."""
    
    MAX_TOKENS: int = 120000
    """Maximum output tokens (gpt-5-mini supports 128K, leaving 8K buffer)."""


# === Query Analyzer Configuration ===
@dataclass
class QueryAnalyzerConfig:
    """Configuration for query analysis (ui/query_analyzer.py)."""
    
    MODEL: str = LLMModels.QUERY_ANALYZER
    """Fast model for query classification."""
    
    TEMPERATURE: float = 0.0
    """Deterministic predictions for consistent k_text values."""
    
    MAX_TOKENS: int = 500
    """Small output for structured JSON only."""


# === Vision Configuration ===
@dataclass
class VisionConfig:
    """Configuration for image caption generation (ingest/generate_captions.py)."""
    
    MODEL: str = LLMModels.VISION
    """Vision model for image analysis."""
    
    MAX_TOKENS: int = 300
    """Maximum caption length in tokens."""
    
    TEMPERATURE: float = 0.3
    """Slight creativity for natural captions."""


# === Retrieval Configuration ===
@dataclass
class RetrievalConfig:
    """Configuration for multimodal retrieval (rag/retriever.py)."""
    
    # Text retrieval
    DEFAULT_K_TEXT: int = 3
    """Default number of text chunks to retrieve."""
    
    # Image retrieval
    DEFAULT_K_IMAGES: int = 3
    """Default number of images to retrieve."""
    
    MAX_IMAGES_PER_QUERY: int = 6
    """Hard limit on total images returned (prevents overload)."""
    
    # Similarity thresholds
    SIMILARITY_THRESHOLD: float = 0.45
    """Same-page images semantic match threshold."""
    
    SIMILARITY_THRESHOLD_NEARBY: float = 0.65
    """±1 page images semantic match threshold (stricter)."""
    
    FALLBACK_SIMILARITY_THRESHOLD: float = 0.5
    """Visual query fallback threshold."""
    
    # MMR parameters
    MMR_LAMBDA: float = 0.7
    """Balance between relevance (1.0) and diversity (0.0)."""
    
    MMR_FETCH_MULTIPLIER: int = 2
    """Fetch k*2 candidates for better diversity."""
    
    # Context
    DEFAULT_CONTEXT_CHUNKS: int = 2
    """Default chunks for image context."""
    
    FALLBACK_IMAGES_PER_DOC: int = 1
    """Images per document in fallback mode."""


# === Chunking Configuration ===
@dataclass
class ChunkingConfig:
    """Configuration for document chunking (index/chunk_documents.py)."""
    
    CHUNK_SIZE: int = 1700
    """Chunk size in characters (~500 tokens)."""
    
    CHUNK_OVERLAP: int = 200
    """Overlap between chunks for context preservation."""


# === Collection Names ===
@dataclass
class Collections:
    """ChromaDB collection names."""
    
    TEXT_CHUNKS: str = "text_chunks"
    """Collection for text chunk embeddings."""
    
    IMAGE_CAPTIONS: str = "image_captions"
    """Collection for image caption embeddings."""


# === Confidence Levels ===
class ConfidenceLevels:
    """Image verification confidence levels."""
    
    HIGH: str = "HIGH"
    """Explicit figure reference in text."""
    
    MEDIUM: str = "MEDIUM"
    """Semantic match with retrieved chunks."""
    
    LOW: str = "LOW"
    """Visual query fallback."""


# === Global Instances (for convenience) ===
# These can be imported directly: from config import EMBEDDING, GENERATOR, etc.
EMBEDDING = EmbeddingConfig()
GENERATOR = GeneratorConfig()
QUERY_ANALYZER = QueryAnalyzerConfig()
VISION = VisionConfig()
RETRIEVAL = RetrievalConfig()
CHUNKING = ChunkingConfig()
COLLECTIONS = Collections()
CONFIDENCE = ConfidenceLevels()


# === Helper Functions ===
def get_chroma_dir() -> Path:
    """Get ChromaDB directory path."""
    return CHROMA_DIR


def get_data_dir() -> Path:
    """Get data directory path."""
    return DATA_DIR


def validate_paths() -> bool:
    """
    Validate that all required directories exist.
    
    Returns:
        True if all paths exist, False otherwise.
    """
    required_dirs = [DATA_DIR, CHROMA_DIR, PROCESSED_DIR, RAW_DIR]
    missing = [d for d in required_dirs if not d.exists()]
    
    if missing:
        print(f"⚠️  Missing directories: {[str(d) for d in missing]}")
        return False
    
    return True


if __name__ == "__main__":
    """Quick configuration check."""
    print("=" * 70)
    print("📋 AI/ML Course Assistant - Configuration")
    print("=" * 70)
    print()
    print("📂 Paths:")
    print(f"   BASE_DIR: {BASE_DIR}")
    print(f"   DATA_DIR: {DATA_DIR}")
    print(f"   CHROMA_DIR: {CHROMA_DIR}")
    print()
    print("🤖 Models:")
    print(f"   Generator: {GENERATOR.MODEL}")
    print(f"   Query Analyzer: {QUERY_ANALYZER.MODEL}")
    print(f"   Vision: {VISION.MODEL}")
    print()
    print("🔢 Embedding:")
    print(f"   Model: {EMBEDDING.MODEL}")
    print(f"   Dimensions: {EMBEDDING.DIMENSIONS}")
    print()
    print("⚙️  Retrieval:")
    print(f"   Default k_text: {RETRIEVAL.DEFAULT_K_TEXT}")
    print(f"   Default k_images: {RETRIEVAL.DEFAULT_K_IMAGES}")
    print(f"   Max images per query: {RETRIEVAL.MAX_IMAGES_PER_QUERY}")
    print()
    print("✂️  Chunking:")
    print(f"   Chunk size: {CHUNKING.CHUNK_SIZE} chars")
    print(f"   Chunk overlap: {CHUNKING.CHUNK_OVERLAP} chars")
    print()
    print("✅ Validating paths...")
    if validate_paths():
        print("   All required directories exist!")
    else:
        print("   Some directories are missing (run setup.bat)")
    print("=" * 70)
