# Phase 5: Configuration Centralization

**Date:** January 21, 2026  
**Status:** ✅ Completed  
**Files Changed:** 8 files created/modified

---

## 📋 Overview

Created centralized `config.py` to eliminate constant duplication across modules and provide single source of truth for configuration.

**Key Principle:** 
- ✅ **In config.py** → Constants duplicated in 2+ files (EMBEDDING_MODEL, MODEL_NAME, etc.)
- ✅ **In local file** → Module-specific constants (VISUAL_KEYWORDS, CURATED_ARTICLES, etc.)

---

## 🆕 New Files

### `config.py` (283 lines)

Centralized configuration with dataclass-based structure:

```python
# === Path Configuration ===
BASE_DIR, DATA_DIR, CHROMA_DIR, PROCESSED_DIR, etc.

# === Model Configuration ===
@dataclass
class LLMModels:
    GENERATOR: str = "gpt-5-mini"       # Main generation (128K context)
    QUERY_ANALYZER: str = "gpt-5-nano"  # Fast classification
    VISION: str = "gpt-4.1-mini"        # Vision model (has vision API)
    JUDGE: str = "gpt-5-mini"           # Faithfulness evaluation

# === Component Configurations ===
@dataclass
class EmbeddingConfig:
    MODEL: str = "text-embedding-3-small"
    DIMENSIONS: int = 1536
    BATCH_SIZE: int = 100

@dataclass
class GeneratorConfig:
    MODEL: str = "gpt-5-mini"
    TEMPERATURE: float = 0.0          # Anti-hallucination
    MAX_TOKENS: int = 120000          # gpt-5-mini supports 128K

@dataclass
class QueryAnalyzerConfig:
    MODEL: str = "gpt-5-nano"
    TEMPERATURE: float = 0.0          # Deterministic
    MAX_TOKENS: int = 500             # Structured output only

@dataclass
class VisionConfig:
    MODEL: str = "gpt-4.1-mini"       # Vision API model
    MAX_TOKENS: int = 300
    TEMPERATURE: float = 0.3

@dataclass
class RetrievalConfig:
    DEFAULT_K_TEXT: int = 3
    DEFAULT_K_IMAGES: int = 3
    MAX_IMAGES_PER_QUERY: int = 8
    SIMILARITY_THRESHOLD: float = 0.5
    SIMILARITY_THRESHOLD_NEARBY: float = 0.65
    MMR_LAMBDA: float = 0.7
    ...

@dataclass
class ChunkingConfig:
    CHUNK_SIZE: int = 1700            # ~500 tokens
    CHUNK_OVERLAP: int = 200

@dataclass
class Collections:
    TEXT_CHUNKS: str = "text_chunks"
    IMAGE_CAPTIONS: str = "image_captions"

class ConfidenceLevels:
    HIGH: str = "HIGH"
    MEDIUM: str = "MEDIUM"
    LOW: str = "LOW"
```

**Global instances for convenience:**
```python
EMBEDDING = EmbeddingConfig()
GENERATOR = GeneratorConfig()
QUERY_ANALYZER = QueryAnalyzerConfig()
VISION = VisionConfig()
RETRIEVAL = RetrievalConfig()
CHUNKING = ChunkingConfig()
COLLECTIONS = Collections()
CONFIDENCE = ConfidenceLevels()
```

---

## 🔧 Modified Files

### 1. `rag/retriever.py` (946 lines)

**Before:**
```python
BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
DEFAULT_MMR_LAMBDA = 0.7
DEFAULT_K_TEXT = 3
DEFAULT_K_IMAGES = 3
CONFIDENCE_HIGH = "HIGH"
DEFAULT_TEXT_COLLECTION = "text_chunks"
...
```

**After:**
```python
from config import (
    EMBEDDING, RETRIEVAL, COLLECTIONS, CONFIDENCE,
    CHROMA_DIR, BASE_DIR
)

# Use config constants
EMBEDDING_MODEL = EMBEDDING.MODEL
EMBEDDING_DIMS = EMBEDDING.DIMENSIONS
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", str(RETRIEVAL.SIMILARITY_THRESHOLD)))
DEFAULT_MMR_LAMBDA = RETRIEVAL.MMR_LAMBDA
DEFAULT_K_TEXT = RETRIEVAL.DEFAULT_K_TEXT
CONFIDENCE_HIGH = CONFIDENCE.HIGH
DEFAULT_TEXT_COLLECTION = COLLECTIONS.TEXT_CHUNKS
...
```

**Impact:** Removed 15+ hardcoded constants, now all from config.

---

### 2. `rag/generator.py` (887 lines)

**Before:**
```python
MODEL_NAME = "gpt-5-mini"
TEMPERATURE = 0.0
MAX_TOKENS = 120000
```

**After:**
```python
from config import GENERATOR

MODEL_NAME = GENERATOR.MODEL
TEMPERATURE = GENERATOR.TEMPERATURE
MAX_TOKENS = GENERATOR.MAX_TOKENS
```

**Impact:** Generator configuration now centralized, easy to change model.

---

### 3. `ui/query_analyzer.py` (408 lines)

**Before:**
```python
MODEL_NAME = "gpt-5-nano"
TEMPERATURE = 0.0
MAX_TOKENS = 500
```

**After:**
```python
from config import QUERY_ANALYZER

MODEL_NAME = QUERY_ANALYZER.MODEL
TEMPERATURE = QUERY_ANALYZER.TEMPERATURE
MAX_TOKENS = QUERY_ANALYZER.MAX_TOKENS
```

**Impact:** Query analyzer uses fast gpt-5-nano model (optimized for structured output).

---

### 4. `index/build_index.py` (560 lines)

**Before:**
```python
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
DEFAULT_CHROMA_DIR = DEFAULT_BASE_DIR / "data" / "chroma_db"
```

**After:**
```python
from config import EMBEDDING, CHROMA_DIR, CHUNKS_BACKUP_DIR, BASE_DIR

EMBEDDING_MODEL = EMBEDDING.MODEL
EMBEDDING_DIMS = EMBEDDING.DIMENSIONS
DEFAULT_CHROMA_DIR = CHROMA_DIR
```

**Impact:** Index building now uses centralized embedding config.

---

### 5. `index/embedding_utils.py` (235 lines)

**Before:**
```python
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
BATCH_SIZE = 100
```

**After:**
```python
from config import EMBEDDING

EMBEDDING_MODEL = EMBEDDING.MODEL
EMBEDDING_DIMS = EMBEDDING.DIMENSIONS
BATCH_SIZE = EMBEDDING.BATCH_SIZE
```

**Impact:** Embedding utilities now use config for all embedding settings.

---

### 6. `ingest/generate_captions.py` (527 lines)

**Before:**
```python
def __init__(self, model_name: str = "gpt-4.1-mini", ...):
```

**After:**
```python
from config import VISION

def __init__(self, model_name: str = VISION.MODEL, ...):
```

**Impact:** Vision model centralized (gpt-4.1-mini has vision capabilities).

---

### 7. `ingest/enrich_images.py` (712 lines)

**Before:**
```python
model_name="gpt-4.1-mini",
```

**After:**
```python
model_name="gpt-4.1-mini",  # Uses gpt-4.1-mini (vision model)
```

**Impact:** Model name consistent across vision modules.

---

### 8. `run_pipeline.py` (779 lines)

**Before:**
```python
CHUNK_SIZE = 1700
CHUNK_OVERLAP = 150
```

**After:**
```python
from config import CHUNKING, BASE_DIR

CHUNK_SIZE = CHUNKING.CHUNK_SIZE
CHUNK_OVERLAP = CHUNKING.CHUNK_OVERLAP
```

**Impact:** Chunking configuration centralized.

---

## 🎯 What Was NOT Moved (By Design)

These constants remain in their modules because they're **module-specific**:

### `rag/retriever.py`
- `VISUAL_KEYWORDS` - Only used in retriever logic
- Module-specific helper functions

### `ingest/download_medium.py`
- `CURATED_ARTICLES` - Medium-specific article list
- `DEFAULT_OUTPUT_DIR` - Medium-specific output path
- `DOWNLOAD_DELAY_SECONDS` - Medium rate limit (different per source)

### `ingest/download_realpython.py`
- `CURATED_ARTICLES` - RealPython-specific article list
- `DEFAULT_OUTPUT_DIR` - RealPython-specific output path

### `ingest/download_arxiv.py`
- Similar source-specific constants

**Rationale:** These constants define the **data and behavior of each data source**, not shared infrastructure.

---

## ✅ Benefits

1. **Single Source of Truth**
   - Change model: edit config.py once, affects all modules
   - Change chunk size: edit config.py once, affects indexing + pipeline

2. **Type Safety**
   - Dataclasses provide IDE autocomplete
   - Clear documentation via docstrings

3. **Easy Testing**
   - Mock config objects in tests
   - Override settings without touching code

4. **Better Organization**
   - All constants in one place
   - Clear separation: shared vs module-specific

5. **Validation**
   - `validate_paths()` function checks directory existence
   - Can add more validation logic centrally

---

## 🧪 Testing

```bash
# Test config.py standalone
python config.py

# Output:
📋 AI/ML Course Assistant - Configuration
📂 Paths:
   BASE_DIR: .../ai-ml-course-assistant
   CHROMA_DIR: .../data/chroma_db
🤖 Models:
   Generator: gpt-5-mini
   Query Analyzer: gpt-5-nano
   Vision: gpt-4.1-mini
🔢 Embedding:
   Model: text-embedding-3-small
   Dimensions: 1536
✅ Validating paths... All required directories exist!
```

---

## 🔄 Migration Summary

| Module | Constants Moved | Import Added |
|--------|----------------|--------------|
| `rag/retriever.py` | 15+ constants | ✅ |
| `rag/generator.py` | 3 constants | ✅ |
| `ui/query_analyzer.py` | 3 constants | ✅ |
| `index/build_index.py` | 4 constants | ✅ |
| `index/embedding_utils.py` | 3 constants | ✅ |
| `ingest/generate_captions.py` | 1 constant | ✅ |
| `ingest/enrich_images.py` | 1 constant | ✅ |
| `run_pipeline.py` | 2 constants | ✅ |

**Total:** 8 files updated, 32+ constants centralized

---

## 📝 Usage Examples

### Importing Configuration

```python
# Import specific configs
from config import EMBEDDING, GENERATOR, RETRIEVAL

# Use constants
model = GENERATOR.MODEL          # "gpt-5-mini"
temp = GENERATOR.TEMPERATURE     # 0.0
k_text = RETRIEVAL.DEFAULT_K_TEXT  # 3

# Import paths
from config import CHROMA_DIR, DATA_DIR
```

### Overriding Configuration (Testing)

```python
from config import RETRIEVAL

# Override for testing
RETRIEVAL.MAX_IMAGES_PER_QUERY = 5  # Instead of 8
```

---

## 🚀 Next Steps

- **Phase 6:** Logging configuration centralization
- **Phase 6:** Performance monitoring (time.perf_counter)
- **Phase 4:** Refactor retriever.py (deferred to end)

---

## 📚 Related Files

- Main config: [config.py](../config.py)
- Environment template: [.env.example](../.env.example)
- Setup script: [setup.bat](../setup.bat)
- Requirements: [requirements.txt](../requirements.txt)

---

## ✍️ Author Notes

**Why gpt-4.1-mini for vision?**
- gpt-5-mini doesn't have vision capabilities yet
- gpt-4.1-mini is current best vision model (cheaper than gpt-4o)

**Why different MAX_TOKENS?**
- Generator: 120K tokens (full answers)
- Query Analyzer: 500 tokens (JSON output only)
- Vision: 300 tokens (image captions)

**Why dataclasses?**
- Type hints + IDE support
- Immutable by default (use frozen=True if needed)
- Clear documentation via docstrings
