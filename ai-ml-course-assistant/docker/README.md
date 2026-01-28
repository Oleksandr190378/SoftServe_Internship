# Docker Configuration

This directory contains all Docker-related files for containerized deployment.

## � Architecture Overview

### ✅ Current Architecture (Phase 4 Part 2 - Single Container)

**Current Deployment:**
```
┌─────────────────────────────────────┐
│   Docker Container (Streamlit)      │
│  ┌──────────────────────────────┐  │
│  │   Streamlit UI (ui/app.py)   │  │
│  │  - Query interface           │  │
│  │  - Retrieval & Generation    │  │
│  │  - Citation display          │  │
│  └──────────────────────────────┘  │
│        ↓ (semantic search)          │
│  ┌──────────────────────────────┐  │
│  │  ChromaDB (data/chroma_db/)  │  │
│  │  - 905 text chunks indexed   │  │
│  │  - 294 image captions indexed│  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
     ✅ Production-ready for serving
```

**⚠️ Important Assumption:**
- Documents are **already processed and indexed** in `../data/chroma_db/`
- ChromaDB is **pre-populated** with 905 chunks + 294 images
- Docker container is for **UI + retrieval only** (no processing)

### 🔄 Document Processing (Outside Docker - for now)

Document preparation happens on host machine:
```bash
# 1. Download documents (Optional - already have 54 docs)
python ingest/download_arxiv.py

# 2. Extract images and create enriched captions
python run_pipeline.py process --doc-id <doc_id>

# 3. Generate embeddings and index to ChromaDB
# (run_pipeline.py handles this)

# Result: data/chroma_db/ is populated and ready
```

This is a **temporary setup** before multi-container deployment.

---

## �📁 Structure

```
docker/
├── Dockerfile               # Multi-stage build definition
├── docker-compose.yml       # Production configuration
├── docker-compose.test.yml  # Testing configuration (safe, isolated)
├── DOCKER.md               # Complete Docker documentation
└── README.md               # This file
```

## 🚀 Quick Start

### Testing (Recommended First)

Safe testing without affecting your production data:

```bash
# From project root
cd docker
docker-compose -f docker-compose.test.yml build
docker-compose -f docker-compose.test.yml up -d

# Access at: http://localhost:8502
```

**Test configuration:**
- Uses `../docker_test_data/` directory (isolated)
- Port 8502 (no conflict)
- Container name: `ai-ml-course-assistant-test`

### Production

Uses your actual data in `../data/`:

```bash
# From docker/ directory
docker-compose build
docker-compose up -d

# Access at: http://localhost:8501
```

## 📖 Documentation

See [DOCKER.md](DOCKER.md) for:
- Complete setup guide
- Testing procedures
- Environment variables
- Troubleshooting
- Performance tuning

## 🔒 Safety Features

✅ Separate test configuration  
✅ Isolated test data directory  
✅ Different ports (8501 prod, 8502 test)  
✅ .env mounted read-only  
✅ Health checks enabled  

## 🛠️ Common Commands

```bash
# Build
docker-compose -f docker-compose.test.yml build --no-cache

# Start/Stop
docker-compose -f docker-compose.test.yml up -d
docker-compose -f docker-compose.test.yml down

# Logs
docker-compose -f docker-compose.test.yml logs -f streamlit-test

# Shell access
docker-compose -f docker-compose.test.yml exec streamlit-test bash
```

## 📊 Configuration Comparison

| Feature | Test | Production |
|---------|------|------------|
| Data directory | `../docker_test_data/` | `../data/` |
| Port | 8502 | 8501 |
| Container name | `ai-ml-course-assistant-test` | `ai-ml-course-assistant` |
| Network | `ai-ml-test-network` | `ai-ml-network` |
| Use case | Safe testing | Real deployment |
