# Test Commands Reference

## Quick Start

```bash
# Run all tests
python -m pytest test/ -v

# Run specific test file
python -m pytest test/test_ingest/test_extract_from_json.py -v

# Run specific test function
python -m pytest test/test_ingest/test_extract_from_json.py::test_detect_source_type -v
```

## By Module

```bash
# Ingest module tests
python -m pytest test/test_ingest/ -v

# Index module tests
python -m pytest test/test_index/ -v

# RAG module tests
python -m pytest test/test_rag/ -v

# UI module tests
python -m pytest test/test_ui/ -v
```

## By STAGE

```bash
# STAGE 1: Validation tests
python -m pytest test/ -m stage1 -v

# STAGE 2: Exception Handling & Constants tests
python -m pytest test/ -m stage2 -v

# STAGE 3: SOLID Principles tests
python -m pytest test/ -m stage3 -v
```

## Combined Markers

```bash
# Ingest module STAGE 1 tests
python -m pytest test/ -m "ingest and stage1" -v

# All modules STAGE 2 tests
python -m pytest test/ -m "stage2" -v

# Ingest and Index modules
python -m pytest test/ -m "ingest or index" -v
```

## Coverage Reports

```bash
# Generate coverage report (HTML)
python -m pytest test/ --cov=. --cov-report=html --cov-report=term

# Show coverage for specific module
python -m pytest test/test_ingest/ --cov=ingest --cov-report=term-missing

# Coverage with branches
python -m pytest test/ --cov=. --cov-report=html --cov-branch
```

## Advanced Options

```bash
# Stop on first failure
python -m pytest test/ -x -v

# Show local variables on failure
python -m pytest test/ -l -v

# Verbose output + print statements
python -m pytest test/ -v -s

# Run tests in parallel (requires pytest-xdist)
python -m pytest test/ -n auto

# Run with timeout (requires pytest-timeout)
python -m pytest test/ --timeout=300

# Run last failed tests only
python -m pytest test/ --lf -v

# Run failed tests first
python -m pytest test/ --ff -v

# Run specific number of tests
python -m pytest test/ --maxfail=3 -v

# Show slowest tests
python -m pytest test/ --durations=10 -v
```

## Direct Python Execution

```bash
# Run test file directly
python test/test_ingest/test_extract_from_json.py

# Run test module with main()
cd ai-ml-course-assistant
python test/test_ingest/test_extract_from_json.py
```

## Debugging

```bash
# Drop into debugger on failure (requires pdb)
python -m pytest test/ --pdb

# Drop into debugger on failure or error
python -m pytest test/ --pdbcls=IPython.terminal.debugger:Pdb

# Show print statements and logs
python -m pytest test/ -s

# Increase log level
python -m pytest test/ --log-cli-level=DEBUG
```

## CI/CD Integration

```bash
# Exit code will be 0 if all tests pass, 1 otherwise
python -m pytest test/ --tb=short

# Generate JUnit XML (for GitHub Actions, Jenkins, etc.)
python -m pytest test/ --junit-xml=test-results.xml

# Generate JSON report
python -m pytest test/ --json-report --json-report-file=report.json
```

## Environment Setup

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-xdist pytest-timeout pytest-mock

# Install with development extras (if configured in setup.py)
pip install -e .[dev]

# Verify pytest is installed
pytest --version
```

## Notes

- Tests assume the current working directory is the project root
- `conftest.py` automatically provides fixtures and markers
- All test functions must start with `test_` prefix
- Use `@pytest.mark.stage1/2/3` or module markers for filtering
- Coverage reports are generated in `htmlcov/index.html`

---

**Last Updated**: 2026-01-17
