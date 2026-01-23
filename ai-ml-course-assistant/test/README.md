# 🧪 Test Suite

## Overview

Comprehensive test suite with **331 tests** covering all modules: ingest, index, rag, and ui.

Tests follow SOLID principles and validate critical functionality, error handling, and edge cases.

## 🚀 Quick Start

```bash
# Run all tests (331 tests)
pytest test/ -v

# Run specific module
pytest test/test_ingest/ -v    # Ingest tests
pytest test/test_index/ -v     # Index tests
pytest test/test_rag/ -v       # RAG tests
pytest test/test_ui/ -v        # UI tests

# Run with markers
pytest test/ -m stage1 -v      # Critical validation tests
pytest test/ -m "ingest and stage1" -v

# Stop on first failure
pytest test/ -x -v

# Show detailed output
pytest test/ -vv --tb=short
```

**Note:** Configuration is in [pytest.ini](../pytest.ini) in project root.

---

## 📁 Directory Structure

```
test/
├── conftest.py                          # Pytest configuration & shared fixtures
├── README.md                            # This file
│
├── test_ingest/                         # Document ingestion tests (88 tests)
│   ├── test_download_arxiv.py
│   ├── test_download_medium.py
│   ├── test_download_realpython.py
│   ├── test_enrich_images.py
│   ├── test_extract_from_json.py
│   ├── test_extract_image_context.py
│   ├── test_extract_images_smart.py
│   └── test_generate_captions.py
│
├── test_index/                          # Indexing & chunking tests (83 tests)
│   ├── test_build_index.py
│   ├── test_chunk_documents.py
│   └── test_embedding_utils.py
│
├── test_rag/                            # RAG module tests (70 tests)
│   ├── test_retrieve_base.py            # Text retrieval tests (NEW)
│   ├── test_retrieve_verification.py    # Verification + dedup tests (NEW)
│   ├── test_retrieve_image_ops.py       # Image ops + ranking tests (NEW)
│   ├── test_generate_base.py            # Answer generation tests (NEW)
│   ├── test_generate_citations.py       # Citation extraction tests (NEW)
│   ├── test_generate_security.py        # Input sanitization tests (NEW)
│   └── test_retriever.py                # Legacy retriever tests (for reference)
│
└── test_ui/                             # UI application tests (90 tests)
    └── test_app.py
```

---

## 📊 Test Statistics (Jan 23, 2026)

| Module | Test File | Tests | Status |
|--------|-----------|-------|--------|
| **test_ingest/** | 8 files | 88 | ✅ All passing |
| **test_index/** | 3 files | 83 | ✅ All passing |
| **test_rag/** | 7 files | **70** | ✅ All passing |
| **test_ui/** | 1 file | 90 | ✅ All passing |
| **TOTAL** | **19 files** | **331** | ✅ **100% passing** |

### Phase 4 Part 2 Refactoring Impact

**Before:** Monolithic generators + retrievers
- Single test_retriever.py covering old monolithic code
- Single test_generator.py with minimal modular coverage

**After:** Modular packages with comprehensive coverage
- `test_retrieve_base.py` - Text retrieval pipeline (base.py) 
- `test_retrieve_verification.py` - Confidence scoring, deduplication with HIGH confidence fix
- `test_retrieve_image_ops.py` - Image retrieval with dict-based ranking (chunk_rank preservation)
- `test_generate_base.py` - UNION citation logic (Answer + Sources combined)
- `test_generate_citations.py` - Citation extraction and validation
- `test_generate_security.py` - Input sanitization (8 injection patterns)

**Test Coverage Improvements:**
- ✅ UNION citation logic: Tests verify citations from both Answer and Sources sections
- ✅ Conditional deduplication: Tests verify HIGH confidence images preserved (Query 9 case)
- ✅ Chunk rank ordering: Tests verify dict-based image prioritization (Query 4 case)
- ✅ Per-session caching: Tests verify isolated retriever/generator instances
- ✅ Clear button logic: Tests verify session_state.clear() with text input key="user_query"

## 🎯 Test Markers

Tests are organized with markers (defined in [conftest.py](conftest.py)):

- `stage1` - Critical validation & edge cases
- `stage2` - Exception handling & constants  
- `stage3` - SOLID principles (SRP, DRY, KISS)
- `ingest` - Ingest module tests
- `index` - Index module tests
- `rag` - RAG module tests
- `ui` - UI module tests

**Usage:**
```bash
pytest test/ -m stage1           # Run only critical tests
pytest test/ -m "rag and stage2" # RAG exception handling tests
```

---

## 🧩 Key Test Coverage

### Phase 4 Part 2: RAG Refactoring Tests (70 tests)

#### Retriever Package (test_retrieve_*.py)

**test_retrieve_base.py** - Text Retrieval Pipeline
- `retrieve_text_chunks()` basic retrieval
- Empty query handling
- Chunk ordering and relevance
- Metadata preservation
- Error handling for missing collections

**test_retrieve_verification.py** - Confidence & Deduplication
- HIGH confidence image detection (explicit figure references)
- MEDIUM confidence (same-page semantic match)
- LOW confidence (fallback visual query)
- **Deduplication logic:** 
  - ✅ HIGH confidence preserved (line 259: explicit ref check FIRST)
  - ✅ Conditional skip (line 275: only skip if not explicit ref)
  - ✅ Query 9 test case: Image Recall 0→1 after fix
- Figure numbering validation
- Confidence score normalization

**test_retrieve_image_ops.py** - Image Retrieval & Ranking
- `retrieve_images()` basic image search
- `retrieve_with_strict_images()` document-filtered retrieval
- **Dict-based ranking:**
  - ✅ Chunk rank preservation (line 151-176: dict with chunk_rank keys)
  - ✅ Query 4 test case: Image Recall 0→1 after fix
- Similarity threshold testing (0.45, 0.5, 0.65)
- Document filter validation (prevents cross-doc pollution)
- Empty result handling

#### Generator Package (test_generate_*.py)

**test_generate_base.py** - RAG Pipeline & Citations
- `generate()` full pipeline: sanitize → format → LLM → parse → validate
- **UNION citation logic (lines 320-335):**
  - ✅ Citations from Answer section extracted
  - ✅ Citations from Sources section extracted
  - ✅ Both sets combined (union, not either-or)
  - ✅ Prevents "citation loss" where LLM forgets Sources
- Answer validation (no hallucinations, grounded in sources)
- Confidence scoring
- Error handling and fallbacks

**test_generate_citations.py** - Citation Extraction
- `extract_chunk_citations()` - Parse [1], [2], [3] format
- `extract_image_citations()` - Parse [A], [B], [C] format
- `validate_and_clean_citations()` - Remove duplicates, validate IDs
- `check_answer_sources_consistency()` - Verify citations in both sections
- Edge cases: Missing citations, invalid IDs, format variations

**test_generate_security.py** - Input Sanitization
- `sanitize_query()` function validation
- **8 injection pattern blocks:**
  1. SQL injection (`' OR '1'='1`)
  2. Command injection (`;rm -rf /`)
  3. Python code injection (`__import__`)
  4. Template injection (`{{ }}`)
  5. XML/XXE injection (`<?xml>`)
  6. Path traversal (`../../../`)
  7. Special characters (control chars)
  8. Length limits (MAX_QUERY_LENGTH=500)
- Query size limits
- Special character handling

### Ingest Module (88 tests)
- Document downloading (arXiv, Medium, RealPython)
- Image extraction and caption generation
- Context enrichment from surrounding text
- Caption generation with GPT-4.1-mini Vision

### Index Module (83 tests)
- Document chunking with semantic boundaries
- Embedding generation with OpenAI
- ChromaDB indexing with metadata validation

### UI Module (90 tests)
- Configuration constants validation
- Image path resolution with fallback patterns
- Citation filtering and display
- Metadata loading and session state management

---

## ⚙️ Configuration

Test configuration is in:
- **[pytest.ini](../pytest.ini)** - Main pytest configuration
- **[conftest.py](conftest.py)** - Shared fixtures and markers

---

## 📝 Test Principles

All tests follow:
1. ✅ **STAGE 1**: Critical validation & edge cases
2. ✅ **STAGE 2**: Exception handling & constants
3. ✅ **STAGE 3**: SOLID principles (SRP, DRY, KISS)

---

## 🔧 Running Tests with Coverage

```bash
# Generate coverage report
pytest test/ --cov=ingest,index,rag,ui --cov-report=html

# View coverage in browser
# Open htmlcov/index.html
```

---

## 📚 Test Results & Progress

**Latest Run (Jan 23, 2026):**
- ✅ 331/331 tests passing (100%)
- ✅ Phase 4 Part 2 refactoring validated
- ✅ All modular packages tested
- ✅ UNION citation logic verified
- ✅ Deduplication fixes verified (Query 9)
- ✅ Image ranking fixes verified (Query 4)
- ✅ Per-session caching verified
- ✅ Clear button logic verified

**Performance:**
- Total runtime: ~45-60 seconds (depends on API calls)
- Fast tests (<100ms): 280+ tests
- Slow tests (>1s): Integration tests with API calls

---

**Last Updated:** January 23, 2026  
**Total Tests:** 331  
**Success Rate:** 100% ✅
