# Test Suite Structure

## Overview

Tests are organized by module, mirroring the project structure for easy navigation and maintenance.

## Directory Structure

```
test/
├── __init__.py                          # Test package initialization
├── conftest.py                          # Shared pytest configuration (optional)
│
├── test_ingest/                         # Tests for document extraction
│   ├── __init__.py
│   ├── test_extract_from_json.py        # Tests for JSON extraction with refactoring validation
│   ├── test_download_arxiv.py           # (TODO) ArXiv downloader tests
│   ├── test_download_medium.py          # (TODO) Medium downloader tests
│   └── test_download_realpython.py      # (TODO) RealPython downloader tests
│
├── test_index/                          # Tests for indexing and chunking
│   ├── __init__.py
│   ├── test_chunk_documents.py          # (TODO) Document chunking tests
│   ├── test_embedding_utils.py          # (TODO) Embedding utility tests
│   ├── test_generate_embeddings.py      # (TODO) Embedding generation tests
│   └── test_build_index.py              # ✅ COMPLETE - 34 tests (validation, error handling, DRY helpers)
│
├── test_rag/                            # Tests for retrieval and generation
│   ├── __init__.py
│   ├── test_retriever.py                # ✅ COMPLETE - 27 tests (initialization, retrieval, caching)
│   └── test_generator.py                # ✅ COMPLETE - 33+5 subtests (security, citations, formatting)
│
└── test_ui/                             # Tests for UI application
    ├── __init__.py
    └── test_app.py                      # ✅ COMPLETE - 28 tests (constants, path resolution, filtering)
```

## Running Tests

### Run all tests:
```bash
pytest test/
```

### Run specific module tests:
```bash
pytest test/test_ingest/
pytest test/test_index/
pytest test/test_rag/
pytest test/test_ui/
```

### Run specific test file:
```bash
pytest test/test_ingest/test_extract_from_json.py
```

### Run specific test function:
```bash
pytest test/test_ingest/test_extract_from_json.py::test_detect_source_type
```

### Run with verbose output:
```bash
pytest test/ -v
```

### Run with coverage:
```bash
pytest test/ --cov=. --cov-report=html
```

## Test Categories

### ✅ COMPLETE: index/build_index.py (34 tests)
- **STAGE 1**: Critical validation (empty lists, None, type checking)
- **STAGE 2**: Constants & exception handling (magic numbers → constants, try-except)
- **STAGE 3**: SOLID principles (DRY helper: `_get_or_create_collection()`)
- **Coverage**: Initialization, validation, helper functions, error handling

### ✅ COMPLETE: rag/generator.py (33+5 subtests)
- **Security**: Query sanitization, injection prevention
- **Citations**: Hallucination prevention, format validation
- **Response**: Format enforcement, section parsing
- **Coverage**: Constants, security patterns, citation rules, model config

### ✅ COMPLETE: rag/retriever.py (27 tests)
- **Initialization**: ChromaDB setup, error handling
- **Retrieval**: Text/image search, MMR vs similarity
- **Metadata**: JSON parsing, list extraction, related IDs
- **Caching**: Embedding cache management
- **Coverage**: Constants, initialization, retrieval, parsing, filtering

### ✅ COMPLETE: ui/app.py (28 tests)
- **Configuration**: UI constants (dimensions, grid layout)
- **Path Resolution**: Paper ID extraction, image path variants
- **Filtering**: Citation-based filtering, metadata loading
- **HTML**: Confidence badge generation
- **Coverage**: Constants, helpers, filtering, error handling

### STAGE 1: Critical Bugs & Validation ✅
- Edge case handling (empty lists, None, zero division)
- Parameter validation
- Boundary condition checks
- Error messages

Files:
- `test/test_ingest/test_extract_from_json.py`

### STAGE 2: Exception Handling & Constants ✅
- File I/O error handling
- Constants instead of magic numbers
- Proper error messages

Files:
- `test/test_ingest/test_extract_from_json.py`

### STAGE 3: SOLID Principles ✅
- Single Responsibility Principle (SRP)
- Don't Repeat Yourself (DRY)
- Keep It Simple, Stupid (KISS)

Files:
- `test/test_ingest/test_extract_from_json.py`

## Adding New Tests

1. Create test file in appropriate subdirectory
2. Follow naming convention: `test_*.py`
3. Use descriptive test function names: `test_*_with_*_should_*`
4. Include docstrings explaining what stage/feature is being tested
5. Use logging for clear output

Example:
```python
def test_extract_sentence_boundary():
    """STAGE 1/2: Test sentence boundary extraction with boundary checks."""
    logger.info("\n=== Test: _extract_sentence_boundary ===")
    
    # STAGE 1: Test empty/None handling
    assert _extract_sentence_boundary("", 0) == ""
    logger.info("✅ Empty values handled")
```

## Test Dependencies

### Required:
- `pytest` - Test framework
- `pytest-cov` - Coverage plugin

### Optional:
- `pytest-xdist` - Parallel test execution
- `pytest-timeout` - Test timeout handling
- `pytest-mock` - Mocking utilities

Install with:
```bash
pip install pytest pytest-cov pytest-xdist pytest-timeout pytest-mock
```

## CI/CD Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run tests
  run: pytest test/ --cov=. --cov-report=xml
```

## Notes

- Each test file corresponds to a source module
- Tests validate SOLID principles and best practices
- Edge cases and error handling are prioritized
- Use logging for clear test output

## Test Statistics (Updated)

### Overall Results: ✅ **122 tests + 5 subtests PASSED**

| Module | File | Tests | Status |
|--------|------|-------|--------|
| Index | test_build_index.py | 34 | ✅ COMPLETE |
| Generator | test_generator.py | 33+5 | ✅ COMPLETE |
| Retriever | test_retriever.py | 27 | ✅ COMPLETE |
| UI | test_app.py | 28 | ✅ COMPLETE |

All tests follow 5-stage refactoring plan (5 étapes universelles):
1. ✅ Critical Bugs & Validation
2. ✅ Exception Handling & Constants  
3. ✅ SOLID Principles (SRP, DRY, KISS)
4. ✅ Type Safety & Dataclasses
5. ✅ Dependency Injection & Configuration
- Maintain test coverage above 80%

---

**Last Updated**: 2026-01-17  
**Status**: Structure created, extraction tests complete
