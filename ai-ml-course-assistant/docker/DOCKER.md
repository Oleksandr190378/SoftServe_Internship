# 🐳 Docker Setup Guide

## Overview

This project includes Docker and Docker Compose configuration for containerized deployment. The setup ensures consistent environments across development, testing, and production.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- `.env` file with API keys (see below)

## Quick Start

### 1. Prepare Environment

Create a `.env` file in the project root:

```bash
# Copy from example
cp .env.example .env

# Edit .env with your credentials
# Required:
# - OPENAI_API_KEY or GROQ_API_KEY
# - HUGGINGFACE_TOKEN (for embeddings)
```

### 2. Choose Testing or Production

**For Safe Testing (Recommended for First Time):**
```bash
# From docker/ directory
# Uses separate docker_test_data/ directory
# Port 8502 (won't conflict with local development on 8501)
docker-compose -f docker-compose.test.yml build
docker-compose -f docker-compose.test.yml up -d

# Access at: http://localhost:8502
```

**For Production:**
```bash
# From docker/ directory
# Uses ./data/ directory (your actual data)
# Port 8501
docker-compose build
docker-compose up -d

# Access at: http://localhost:8501
```

### 3. Build and Run (Testing Environment)

```bash
# Navigate to docker directory
cd docker

# Build the Docker image for testing
docker-compose -f docker-compose.test.yml build

# Start the test container
docker-compose -f docker-compose.test.yml up

# Run in background
docker-compose -f docker-compose.test.yml up -d

# View logs
docker-compose -f docker-compose.test.yml logs -f streamlit-test

# Stop the application
docker-compose -f docker-compose.test.yml down
```

### 4. Access the Application

- **Streamlit UI:** http://localhost:8501
- **Health Check:** curl http://localhost:8501/_stcore/health

## Project Structure in Container

```
/app/
├── data/
│   ├── chroma_db/        # Vector store (persisted)
│   ├── processed/        # Processed documents (persisted)
│   └── raw/              # Raw documents (persisted)
├── ui/
│   └── app.py            # Streamlit application
├── rag/
│   ├── generator.py
│   └── retriever.py
├── index/
│   └── build_index.py
├── requirements.txt      # Python dependencies
└── .env                  # Environment variables (mounted)
```

## Volume Mounts

### Production (docker-compose.yml)

| Host Path | Container Path | Purpose |
|-----------|-----------------|---------|
| `./data` | `/app/data` | Persists vector store and documents (PRODUCTION) |
| `./.env` | `/app/.env` (ro) | Configuration (read-only) |
| `./logs` | `/app/logs` | Application logs |

### Testing (docker-compose.test.yml)

| Host Path | Container Path | Purpose |
|-----------|-----------------|---------|
| `./docker_test_data` | `/app/data` | Isolated test data (SAFE) |
| `./.env` | `/app/.env` (ro) | Configuration (read-only) |
| `./docker_test_logs` | `/app/logs` | Test logs |

**Key Differences:**
- **Test:** Uses `docker_test_data/` - safe to delete/reset
- **Test:** Port 8502 (no conflict with local dev)
- **Production:** Uses `data/` - your actual data
- **Production:** Port 8501

## Environment Variables

### Required

```env
# API Keys - choose at least ONE
OPENAI_API_KEY=sk-...  # For OpenAI embeddings
GROQ_API_KEY=gsk-...   # For Groq LLM

# Embeddings
HUGGINGFACE_TOKEN=hf_...  # For Hugging Face models
```

### Optional

```env
GROQ_MODEL=llama-3.1-70b-versatile
TEXT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
IMAGE_EMBEDDING_MODEL=openai/clip-vit-base-patch32
```

## Common Commands

### Development

```bash
# Rebuild after code changes
docker-compose build --no-cache

# Run with interactive terminal
docker-compose run --rm streamlit bash

# Run pytest inside container
docker-compose run --rm streamlit pytest test/ -v

# Run with custom command
docker-compose run --rm streamlit python run_pipeline.py
```

### Production

```bash
# Start as daemon
docker-compose up -d

# View real-time logs
docker-compose logs -f streamlit

# Restart container
docker-compose restart streamlit

# Stop gracefully
docker-compose stop streamlit

# Remove all containers and volumes
docker-compose down -v
```

### Debugging

```bash
# Access container shell
docker-compose exec streamlit bash

# Run Python commands
docker-compose exec streamlit python -c "import chromadb; print('OK')"

# View container details
docker inspect ai-ml-course-assistant

# Check resource usage
docker stats ai-ml-course-assistant
```

## Image Size Optimization

The Dockerfile uses multi-stage build to minimize image size:

- **Stage 1 (Builder):** Compiles dependencies with build tools
- **Stage 2 (Runtime):** Only runtime libraries (≈2.5GB final size)

### Estimated Image Size

- Base (python:3.10-slim): 150 MB
- Dependencies: ~2.3 GB
- Application code: ~50 MB
- **Total: ~2.5 GB**

## Health Checks

The container includes automated health checks:

```bash
# Check status
docker ps

# Expected output includes:
# STATUS: Up 2 minutes (healthy)
```

## Network Configuration

- **Port 8501:** Streamlit web interface
- **Driver:** Bridge network (default)
- **Isolation:** Full container isolation

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs streamlit

# Common issues:
# - Missing .env file
# - Invalid API keys
# - Port 8501 already in use
```

### Port Already in Use

```bash
# Change port in docker-compose.yml:
# ports:
#   - "8502:8501"  # Map to different port

# Or kill existing process
docker-compose down
```

### Data Persistence Issues

```bash
# Verify volume is mounted
docker inspect ai-ml-course-assistant | grep -A 5 Mounts

# Check volume contents
docker-compose exec streamlit ls -la data/chroma_db/
```

### API Key Issues

```bash
# Verify environment variables inside container
docker-compose exec streamlit env | grep API_KEY

# Debug API connectivity
docker-compose exec streamlit python -c "import openai; print(openai.api_key)"
```

## Performance Tuning

### Resource Limits (in docker-compose.yml)

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 2G
    reservations:
      cpus: '1'
      memory: 1G
```

### Streamlit Configuration

```env
# Reduce startup time
STREAMLIT_CLIENT_LOGGER_LEVEL=warning

# Cache settings
STREAMLIT_CLIENT_CACHE_CONTROL_HEADER=max-age=3600
```

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Build Docker image
  run: docker-compose build

- name: Run tests
  run: docker-compose run --rm streamlit pytest test/ -v

- name: Push to registry
  run: docker push registry.example.com/ai-ml-assistant:latest
```

## Security Considerations

✅ **Implemented:**
- `.env` mounted as read-only
- Non-root user execution (implied by slim image)
- Minimal attack surface (slim base image)

⚠️ **Additional Recommendations:**
- Use secrets management for production (AWS Secrets, HashiCorp Vault)
- Enable container scanning for vulnerabilities
- Set resource limits
- Use private container registry

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## Testing Docker Setup

### 1. Validate Environment File

Before building, ensure `.env` file exists and contains required keys:

```bash
# Check if .env exists
if (Test-Path .env) { Write-Host "✅ .env file exists" } else { Write-Host "❌ .env file missing" }

# Validate API keys (without exposing values)
Get-Content .env | Select-String -Pattern "OPENAI_API_KEY|GROQ_API_KEY" | ForEach-Object {
    $key = $_ -split "=" | Select-Object -First 1
    $value = $_ -split "=" | Select-Object -Last 1
    if ($value -match "^sk-" -or $value -match "^gsk-") {
        Write-Host "✅ $key is set"
    } else {
        Write-Host "❌ $key is missing or invalid"
    }
}
```

**Required keys:**
```env
OPENAI_API_KEY=sk-...  # Must start with 'sk-'
GROQ_API_KEY=gsk-...   # Must start with 'gsk-' (optional if using OpenAI)
```

### 2. Build Docker Image

```bash
# Clean build (recommended for first time)
docker-compose build --no-cache

# Quick build (uses cache)
docker-compose build

# Check build success
docker images | Select-String "ai-ml-course-assistant"
```

**Expected output:**
```
ai-ml-course-assistant   latest   abc123def456   2 minutes ago   2.5GB
```

### 3. Test Container Startup

```bash
# Start container in foreground (see logs in real-time)
docker-compose up

# Start in background
docker-compose up -d

# Check container status (should show "healthy" after ~30s)
docker ps -a | Select-String "ai-ml-course-assistant"
```

**Expected status progression:**
```
STATUS
Up 5 seconds (health: starting)
Up 35 seconds (healthy)  ← Should reach this
```

### 4. Verify Data Persistence

Test that ChromaDB data persists across container restarts:

```bash
# Method 1: Check volume mounts
docker inspect ai-ml-course-assistant | Select-String -Pattern "Mounts" -Context 0,15

# Method 2: Access container and check directory
docker-compose exec streamlit ls -la /app/data/chroma_db/

# Method 3: Test persistence
# a) Create test data
docker-compose exec streamlit python -c "
import chromadb
client = chromadb.PersistentClient(path='/app/data/chroma_db')
print('ChromaDB collections:', len(client.list_collections()))
"

# b) Restart container
docker-compose restart streamlit

# c) Verify data still exists
docker-compose exec streamlit python -c "
import chromadb
client = chromadb.PersistentClient(path='/app/data/chroma_db')
print('ChromaDB collections after restart:', len(client.list_collections()))
"
```

**Expected output:**
- Same number of collections before and after restart
- Files persist in `./data/chroma_db/` on host

### 5. Validate Environment Variables

Check that API keys are loaded correctly inside container:

```bash
# Method 1: Check environment variables (safe - doesn't show values)
docker-compose exec streamlit env | Select-String -Pattern "API_KEY"

# Method 2: Validate API connectivity
docker-compose exec streamlit python -c "
import os
import sys

# Check if keys exist
openai_key = os.getenv('OPENAI_API_KEY', '')
groq_key = os.getenv('GROQ_API_KEY', '')

if openai_key.startswith('sk-'):
    print('✅ OPENAI_API_KEY loaded correctly')
elif groq_key.startswith('gsk-'):
    print('✅ GROQ_API_KEY loaded correctly')
else:
    print('❌ No valid API key found')
    sys.exit(1)
"

# Method 3: Test OpenAI connection
docker-compose exec streamlit python -c "
from openai import OpenAI
import os

try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.embeddings.create(
        model='text-embedding-3-small',
        input='test'
    )
    print('✅ OpenAI API connection successful')
    print(f'   Embedding dimension: {len(response.data[0].embedding)}')
except Exception as e:
    print(f'❌ OpenAI API failed: {str(e)}')
"
```

### 6. Test Application Functionality

```bash
# Access the UI
Start-Process "http://localhost:8501"

# Test retrieval system
docker-compose exec streamlit python -c "
from rag.retrieve import MultimodalRetriever

try:
    retriever = MultimodalRetriever(
        chroma_dir='/app/data/chroma_db',
        text_collection='text_chunks',
        image_collection='image_captions'
    )
    results = retriever.retrieve_with_verification('What is a Transformer?', k_text=3)
    print('✅ Retrieval system working')
    print(f'   Retrieved: {len(results[\"chunks\"])} chunks, {len(results[\"images\"])} images')
except Exception as e:
    print(f'❌ Retrieval failed: {str(e)}')
"

# Check health endpoint
curl http://localhost:8501/_stcore/health
```

### 7. Monitor Logs

```bash
# Real-time logs
docker-compose logs -f streamlit

# Look for success indicators:
# ✅ "INFO - Initializing MultimodalRetriever"
# ✅ "INFO - ✅ Loaded text collection: text_chunks"
# ✅ "INFO - ✅ Loaded image collection: image_captions"

# Check for errors:
# ❌ "ERROR" messages
# ❌ Python exceptions
# ❌ API connection failures
```

### 8. Performance Testing

```bash
# Check resource usage
docker stats ai-ml-course-assistant

# Expected ranges:
# CPU: 1-5% idle, 50-100% during inference
# Memory: 500MB-1.5GB
# Network: varies with queries
```

## Troubleshooting Docker Issues

### Build Failures

```bash
# Clear Docker cache
docker system prune -a

# Rebuild from scratch
docker-compose build --no-cache --pull

# Check disk space
docker system df
```

### Container Won't Start

```bash
# Check detailed logs
docker-compose logs --tail=100 streamlit

# Common issues:
# 1. Missing .env file → Create .env with API keys
# 2. Invalid API key → Check key format (sk-... or gsk-...)
# 3. Port conflict → Change port in docker-compose.yml
# 4. ChromaDB corruption → Delete data/chroma_db/ and rebuild index
```

### Data Not Persisting

```bash
# Verify volume mount
docker inspect ai-ml-course-assistant --format='{{json .Mounts}}' | ConvertFrom-Json

# Expected Source: /absolute/path/to/data
# Expected Destination: /app/data

# Fix: Check docker-compose.yml volumes section
# - ./data:/app/data  ← Must be relative path
```

### API Connection Failures

```bash
# Test outside container first
python -c "
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
print('API Key:', os.getenv('OPENAI_API_KEY')[:10] + '...')
response = client.embeddings.create(model='text-embedding-3-small', input='test')
print('✅ Connection successful')
"

# If works outside but not inside:
# 1. Check .env is mounted: docker-compose config | Select-String ".env"
# 2. Verify .env is in project root
# 3. Restart container: docker-compose restart streamlit
```

## Support

For issues or questions:

1. Check logs: `docker-compose logs streamlit`
2. Verify configuration: `docker-compose config`
3. Run health check: `curl http://localhost:8501/_stcore/health`
4. Test environment: Follow "Testing Docker Setup" section above
