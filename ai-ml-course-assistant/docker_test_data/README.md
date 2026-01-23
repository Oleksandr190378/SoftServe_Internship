# Docker Test Data Directory

This directory is used for Docker testing to avoid affecting production data.

## Purpose

- **Isolated testing:** Test Docker container without modifying real data
- **Safe experimentation:** Try different configurations
- **Clean state:** Easy to reset by deleting this folder

## Structure

```
docker_test_data/
├── chroma_db/          # ChromaDB test database (created by container)
├── processed/          # Processed documents (created by container)
├── raw/               # Raw input files (copy sample files here)
└── README.md          # This file
```

## Usage

### Option 1: Start with Empty Database
```bash
# Container will create empty ChromaDB
docker-compose -f docker-compose.test.yml up -d
```

### Option 2: Copy Sample Data
```bash
# Copy some sample documents for testing
Copy-Item data/raw/papers/*.pdf docker_test_data/raw/papers/ -Recurse

# Run pipeline inside container to process
docker-compose -f docker-compose.test.yml exec streamlit-test python run_pipeline.py
```

### Option 3: Copy Existing Database
```bash
# Copy entire ChromaDB for testing (WARNING: may be large)
Copy-Item data/chroma_db docker_test_data/ -Recurse
```

## Reset Test Environment

```bash
# Stop container
docker-compose -f docker-compose.test.yml down

# Delete test data
Remove-Item docker_test_data/* -Recurse -Force

# Start fresh
docker-compose -f docker-compose.test.yml up -d
```

## Production Data Safety

✅ This directory is SEPARATE from `./data/`
✅ Production data in `./data/` is NOT affected by Docker tests
✅ Test container uses port 8502 (production uses 8501)
