"""
Pytest configuration and fixtures for all tests.

This file is automatically discovered by pytest and provides:
- Shared fixtures
- Global configuration
- Test markers
"""

import pytest
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Markers for categorizing tests
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "stage1: STAGE 1 - Critical Bugs & Validation tests"
    )
    config.addinivalue_line(
        "markers",
        "stage2: STAGE 2 - Exception Handling & Constants tests"
    )
    config.addinivalue_line(
        "markers",
        "stage3: STAGE 3 - SOLID Principles tests"
    )
    config.addinivalue_line(
        "markers",
        "ingest: Ingest module tests"
    )
    config.addinivalue_line(
        "markers",
        "index: Index module tests"
    )
    config.addinivalue_line(
        "markers",
        "rag: RAG module tests"
    )
    config.addinivalue_line(
        "markers",
        "ui: UI module tests"
    )


# ============================================================================
# Shared Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir():
    """Return data directory."""
    return PROJECT_ROOT / "data"


@pytest.fixture(scope="session")
def raw_data_dir(data_dir):
    """Return raw data directory."""
    return data_dir / "raw"


@pytest.fixture(scope="session")
def processed_data_dir(data_dir):
    """Return processed data directory."""
    return data_dir / "processed"


@pytest.fixture(scope="session")
def realpython_dir(raw_data_dir):
    """Return RealPython raw data directory."""
    return raw_data_dir / "realpython"


@pytest.fixture(scope="session")
def medium_dir(raw_data_dir):
    """Return Medium raw data directory."""
    return raw_data_dir / "medium"


@pytest.fixture(scope="session")
def papers_dir(raw_data_dir):
    """Return papers (ArXiv) raw data directory."""
    return raw_data_dir / "papers"


# ============================================================================
# Helper functions for common test operations
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Auto-apply markers based on test file location."""
    for item in items:
        # Apply module markers
        if "test_ingest" in str(item.fspath):
            item.add_marker(pytest.mark.ingest)
        elif "test_index" in str(item.fspath):
            item.add_marker(pytest.mark.index)
        elif "test_rag" in str(item.fspath):
            item.add_marker(pytest.mark.rag)
        elif "test_ui" in str(item.fspath):
            item.add_marker(pytest.mark.ui)
        
        # Apply stage markers based on docstring
        if item.obj and item.obj.__doc__:
            doc = item.obj.__doc__.lower()
            if "stage 1" in doc:
                item.add_marker(pytest.mark.stage1)
            elif "stage 2" in doc:
                item.add_marker(pytest.mark.stage2)
            elif "stage 3" in doc:
                item.add_marker(pytest.mark.stage3)


# ============================================================================
# Pytest configuration
# ============================================================================

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )


# ============================================================================
# Usage Examples
# ============================================================================
"""
Run tests with markers:

    # Run all STAGE 1 tests
    pytest test/ -m stage1
    
    # Run all ingest module tests
    pytest test/ -m ingest
    
    # Run STAGE 2 ingest tests
    pytest test/ -m "stage2 and ingest"
    
    # Run all tests except slow tests
    pytest test/ -m "not slow"
    
    # Run with verbose output and markers
    pytest test/ -v -m stage1
    
    # Run with coverage
    pytest test/ --cov=. --cov-report=html
    
    # Run specific test file
    pytest test/test_ingest/test_extract_from_json.py -v
"""
