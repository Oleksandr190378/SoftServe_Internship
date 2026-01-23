# 🎓 AI/ML Course Assistant - Multimodal RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** system designed to help students and educators learn AI/ML concepts through intelligent retrieval of text and images (diagrams, architectures, equations) from academic papers and tutorials.

**🚀 Status:** Production Ready - All Phases Complete ✅

---

## ✨ Key Features

| Feature | Status | Details |
|---------|--------|---------|
| **Multi-source Document Ingestion** | ✅ | arXiv papers, Medium articles, RealPython tutorials |
| **Semantic Text Chunking** | ✅ | LangChain RecursiveCharacterTextSplitter with figure detection |
| **Automated Image Captioning** | ✅ | GPT-4o-mini Vision descriptions (~$0.015/image) |
| **Vector Embeddings** | ✅ | OpenAI `text-embedding-3-small` (1536-dimensional) |
| **ChromaDB Vector Store** | ✅ | 2 collections: text chunks + image captions |
| **MMR + Semantic Retrieval** | ✅ | Diversity for text (λ=0.7), confidence verification for images |
| **LLM Answer Generation** | ✅ | OpenAI GPT-5 Mini with citation grounding (TEMPERATURE=0.0) |
| **Anti-Hallucination Protection** | ✅ | Metadata validation, few-shot prompting, zero temperature |
| **Streamlit Web Interface** | ✅ | Interactive query interface with inline image display |
| **Comprehensive Testing** | ✅ | Unit tests, retrieval evaluation, ground truth validation |

---

## 📊 Performance Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Recall@5 (text)** | ≥70% | **95.0%** | ✅ |
| **Image Hit Rate** | ≥60% | **88.9%** | ✅ |
| **Mean Reciprocal Rank** | ≥0.70 | **1.000** | ✅ |
| **Indexed Documents** | 54 | **54** | ✅ |
| **Processed Documents** | - | **19** | ✅ |
| **Total Text Chunks** | - | **369** | ✅ |
| **Total Images** | - | **142** | ✅ |

---

## 🎯 Who Should Use This?

- **Students** - Learning AI/ML concepts with visual explanations
- **Educators** - Building RAG-based educational tools
- **Developers** - Implementing production RAG systems
- **Researchers** - Experimenting with retrieval & generation techniques

---

## 📋 Prerequisites

- **Python 3.11** or higher
- **OpenAI API key** (for embeddings, image captioning, and LLM)
- **2-3 GB** disk space (for documents and vector store)
- **Internet connection** (for downloading documents from sources)

---

## 🚀 Installation

### Quick Setup (Recommended)

**Windows users:** Run the automated setup script:
```bash
.\setup.bat
```

This will:
- ✅ Check Python version (3.10+)
- ✅ Create virtual environment
- ✅ Copy `.env.example` to `.env`
- ✅ Create data directories
- ✅ Install all dependencies

### Manual Setup

**1. Clone or Navigate to Project**

```bash
cd c:\Users\Користувач\Documents\python_1\softserve\SoftServe_Internship\ai-ml-course-assistant
```

**2. Create Virtual Environment**

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables**

Create `.env` file from template:
```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```env
OPENAI_API_KEY=your_openai_api_key_here
CHROMA_DB_PATH=./data/chroma_db
DATA_DIR=./data
```

**Get your API key:** https://platform.openai.com/api-keys

---

## ⚡ Quick Start (3 Steps)

### Step 1: Download Documents

Choose one or more data sources:

**Option A: Download arXiv Papers**
```bash
cd ingest
python download_arxiv.py
```
Downloads ~30 arXiv papers on deep learning and machine learning.

**Option B: Download Medium/TDS Articles**
```bash
python download_medium.py
```
Downloads ~10 tutorial articles with explanations and examples.

**Option C: Download RealPython Tutorials**
```bash
python download_realpython.py
```
Downloads ~9 Python ML tutorials with code examples.

**Download All Sources**
```bash
# From project root
cd ingest
python download_arxiv.py && python download_medium.py && python download_realpython.py
```

### Step 2: Process Documents

```bash
cd ..  # Back to project root

# Check processing status
python run_pipeline.py status

# Process all unprocessed documents (fast, no VLM cost)
python run_pipeline.py process --all --no-vlm

# Or process with image captioning (includes GPT-4o-mini descriptions)
python run_pipeline.py process --all

# Process specific document
python run_pipeline.py process --doc-id arxiv_1706_03762

# Force reprocess document
python run_pipeline.py process --doc-id arxiv_1706_03762 --force
```

**Processing Output:**
- Text chunks stored in ChromaDB
- Images extracted and stored in `data/processed/images/`
- Metadata saved to `data/processed/images_metadata.json`
- Processing tracked in `data/processed_docs.json`

### Step 3: Run the Web Interface

```bash
# Option 1: Using batch file (Windows)
.\run_app.bat

# Option 2: Direct Streamlit command
streamlit run ui/app.py
```

**Access the app:**
- Opens at `http://localhost:8501`
- Enter questions like:
  - "Show Transformer encoder-decoder architecture"
  - "Explain residual connections in ResNet"
  - "What is attention mechanism"

---

## 📁 Project Structure

```
ai-ml-course-assistant/
├── data/
│   ├── raw/                          # Downloaded documents (3 subdirs: arxiv, medium, realpython)
│   ├── processed/
│   │   ├── processed_docs.json       # Registry of processed documents
│   │   ├── images_metadata.json      # Image captions and metadata
│   │   └── images/                   # Extracted image files
│   └── chroma_db/                    # ChromaDB vector store
│       ├── text_chunks/              # Text embedding collection
│       └── image_captions/           # Image caption embedding collection
│
├── ingest/                           # Stage 1-2: Download & Extract
│   ├── download_arxiv.py             # Download arXiv papers
│   ├── download_medium.py            # Download Medium/TDS articles
│   ├── download_realpython.py        # Download RealPython tutorials
│   └── utils.py                      # Shared utilities
│
├── index/                            # Stage 3-5: Chunk, Embed, Index
│   ├── build_index.py                # Main indexing pipeline
│   ├── chunk_documents.py            # Semantic text chunking
│   ├── embedding_utils.py            # OpenAI embeddings wrapper
│   └── extract_image_context.py      # Image context extraction
│
├── rag/                              # Retrieval & Generation
│   ├── retriever.py                  # MMR search + semantic verification ✅
│   └── generator.py                  # GPT-5 Nano answer generation ✅
│
├── ui/
│   ├── app.py                        # Streamlit web interface ✅
│   └── assets/                       # UI images/icons
│
├── test/                             # Testing & Validation (334 tests)
│   ├── conftest.py                   # Pytest configuration & fixtures
│   ├── README.md                     # Test suite documentation
│   ├── test_ingest/                  # Ingest module tests (88 tests)
│   ├── test_index/                   # Index module tests (83 tests)
│   ├── test_rag/                     # RAG module tests (63 tests)
│   └── test_ui/                      # UI tests (100 tests)
│
├── eval/                             # Evaluation Framework
│   ├── evaluate_retrieval.py         # Metrics: Recall@k, Precision@k, MRR ✅
│   ├── validate_ground_truth.py      # Ground truth validation ✅
│   ├── ground_truth.json             # 10 test queries with annotations
│   ├── test_queries.json             # 30 test queries for evaluation
│   └── results/                      # Evaluation outputs
│
├── docs/                             # Documentation
│   ├── QUICKSTART.md                 # 5-minute setup guide
│   ├── PIPELINE_GUIDE.md             # Detailed document processing guide
│   ├── PRD.md                        # Product requirements
│   ├── data_sources.md               # Data sources catalog
│   ├── retrieval_strategy_analysis.md# MMR vs similarity analysis
│   └── ROADMAP.md                    # Development roadmap
│
├── run_pipeline.py                   # Main CLI for document processing ✅
├── run_app.bat                       # Batch script to run Streamlit UI
├── setup.bat                         # Automated setup script (Windows) ✅
├── requirements.txt                  # Python dependencies
├── pytest.ini                        # Pytest configuration ✅
├── .env                              # Environment variables (API keys)
├── .env.example                      # Environment template
├── .gitignore                        # Git ignore rules
├── .github/
│   └── workflows/
│       └── test.yml                  # CI/CD automated testing ✅
└── README.md                         # This file
```

---

## 🧪 Testing & Evaluation

### Run Unit Tests (334 tests)

Test configuration is in [pytest.ini](pytest.ini). All tests are organized by module and follow SOLID principles.

```bash
# All tests (334 tests)
pytest test/ -v

# Specific module
pytest test/test_ingest/ -v    # Ingest tests (88)
pytest test/test_index/ -v     # Index tests (83)
pytest test/test_rag/ -v       # RAG tests (63)
pytest test/test_ui/ -v        # UI tests (100)

# By test category
pytest test/ -m stage1 -v      # Critical validation tests
pytest test/ -m "rag and stage2" -v  # RAG exception handling

# Stop on first failure
pytest test/ -x -v
```

**Test Statistics:**
- ✅ **334 tests** total
- ✅ **100% passing**
- ✅ Covers all modules: ingest, index, rag, ui

For detailed test documentation, see [test/README.md](test/README.md).

### Validate Ground Truth

Verify all ground truth queries reference existing documents/images:

```bash
python eval/validate_ground_truth.py

# Expected output:
# ✅ Ground truth structure is valid
# ✅ All images have valid metadata
# ✅ All document and image references are valid
```

### Run Retrieval Evaluation

Evaluate retrieval quality against 10 test queries:

```bash
python eval/evaluate_retrieval.py
```

**Output includes:**
- Recall@3, Recall@5, Recall@10
- Precision@k metrics
- Mean Reciprocal Rank (MRR)
- Image retrieval statistics
- Results saved to `eval/results/retrieval_eval_<timestamp>.json`

### Continuous Integration

Automated testing runs on every push via [GitHub Actions](.github/workflows/test.yml):
- ✅ Tests on Ubuntu & Windows
- ✅ Python 3.10, 3.11, 3.12
- ✅ Code quality checks (flake8, black, isort)

---

## 📖 Documentation

For detailed information, see:

| Document | Purpose |
|----------|---------|
| **[QUICKSTART.md](docs/QUICKSTART.md)** | 5-minute setup and basic usage |
| **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Deep dive: Retriever & Generator modules, anti-hallucination mechanisms |
| **[PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md)** | Detailed document processing pipeline |
| **[PRD.md](docs/PRD.md)** | Product requirements and success metrics |
| **[data_sources.md](docs/data_sources.md)** | Complete catalog of 54 data sources |
| **[retrieval_strategy_analysis.md](docs/retrieval_strategy_analysis.md)** | MMR vs similarity search comparison |
| **[ROADMAP.md](docs/ROADMAP.md)** | Development phases and milestones |

---

## 🔧 Utilities & Maintenance

### Delete Document from Index

Remove a processed document from ChromaDB:

```bash
python delete_doc_from_chromadb.py arxiv_1706_03762
```

### Check Index Status

```bash
python run_pipeline.py status
```

---

## 🏗️ How It Works

### 1. Retrieval Pipeline

1. **Query Input** → User asks a question
2. **MMR Semantic Search** → OpenAI embeddings + Maximal Marginal Relevance (diversity-aware ranking)
3. **Batch Embedding Generation** → Efficient multi-item embedding in single API call
4. **Multi-Level Image Verification** → 3-tier confidence (HIGH/MEDIUM/LOW based on metadata, semantics, proximity)
5. **Result Ranking** → By similarity, confidence, and diversity

**For details, see:** [ARCHITECTURE.md - Retriever Module](docs/ARCHITECTURE.md#part-1-retriever-module-search--verification)

### 2. Generation Pipeline

1. **Context Preparation** → Format retrieved text + images into structured LLM input
2. **Grounded LLM Inference** → GPT-5 Nano with TEMPERATURE=0.0 (no randomness)
3. **Few-Shot Prompting** → 2 complete examples showing correct citation behavior
4. **Answer Formatting** → Structured output with [1][2][A][B] citations
5. **Citation Validation** → Remove hallucinated references, ensure Answer-Sources sync

**For details, see:** [ARCHITECTURE.md - Generator Module](docs/ARCHITECTURE.md#part-2-generator-module-intelligent-processing--answers)

### 3. Anti-Hallucination Measures

- ✅ **Zero Temperature** - No randomness, deterministic token selection
- ✅ **Few-Shot Prompting** - 2 complete working examples in system prompt
- ✅ **Citation Grounding** - Every statement tied to [source] with validation
- ✅ **Metadata Validation** - Cross-check images belong to cited documents
- ✅ **Negative Examples** - Explicit forbidden patterns (wrong citations, external knowledge)
- ✅ **Output Parsing** - Automatic removal of hallucinated references

**For technical deep dive, see:** [ARCHITECTURE.md - Anti-Hallucination Safeguards](docs/ARCHITECTURE.md#5-anti-hallucination-safeguards)

---

## 📊 Technology Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| **Language** | Python 3.11+ | Modern syntax, type hints |
| **CLI/Orchestration** | LangChain | RAG pipeline, document processing |
| **Vector Database** | ChromaDB | Text + image embeddings |
| **Embeddings** | OpenAI `text-embedding-3-small` | 1536-dimensional, $0.02/1M tokens |
| **Vision/Captioning** | OpenAI `gpt-4o-mini` | Image descriptions, $0.015/image |
| **LLM** | OpenAI `gpt-5-mini` | Answer generation, $0.150/1M in |
| **PDF Processing** | PyMuPDF (fitz) | Text + image extraction from PDFs |
| **Image Processing** | Pillow (PIL) | WebP conversion, metadata |
| **Web UI** | Streamlit | Interactive query interface |
| **Testing** | pytest | Unit and integration tests |

---

## 🎯 Next Steps

1. **First Time Setup:**
   ```bash
   pip install -r requirements.txt
   python ingest/download_arxiv.py      # ~2 min
   python run_pipeline.py process --all --no-vlm  # ~5 min
   streamlit run ui/app.py
   ```

2. **Explore Features:**
   - Try sample queries in the web interface
   - Check retrieval with debug view enabled
   - Review inline images and citations

3. **Understand the System:**
   - Read [PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md) for processing details
   - Review [PRD.md](docs/PRD.md) for success metrics
   - Check [data_sources.md](docs/data_sources.md) for available documents

4. **Extend the System:**
   - Add more data sources in `ingest/`
   - Customize LLM prompts in `rag/generator.py`
   - Add evaluation metrics in `eval/`

---

## 🤝 Contributing

This is an educational project demonstrating production RAG systems. 

**Key Design Principles:**
- ✅ Pure retrieval-augmented generation
- ✅ Grounded answers with citations
- ✅ Anti-hallucination safeguards
- ✅ Comprehensive testing and evaluation
- ✅ Educational focus on explainability

---

## 📞 Project Information

**Project:** AI/ML Course Assistant  
**Domain:** Multimodal RAG System  
**Status:** Production Ready ✅  
**Last Updated:** January 21, 2026  

**Key Achievements:**
- ✅ 54 data sources curated
- ✅ 19 documents fully processed (369 chunks, 142 images)
- ✅ Retrieval: Recall@5=95%, Image Hit Rate=88.9%
- ✅ Generation: Zero hallucinations with citation grounding
- ✅ Comprehensive evaluation framework
- ✅ Production Streamlit interface

---

## 📝 License

Educational project for learning purposes.

---

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| "OpenAI API key not found" | Add OPENAI_API_KEY to `.env` file |
| "ChromaDB path mismatch" | Verify CHROMA_DB_PATH in `.env` matches actual location |
| "Images not displaying" | Check `data/processed/images/` directory exists with image files |
| "Processing fails" | Run `python run_pipeline.py status` to check document status |
| "Streamlit connection refused" | Verify port 8501 is not in use, or specify different port |

---

**Ready to use! Start with [Quick Start](#-quick-start-3-steps) section above.** 🚀


---

##  Future: Multi-Container Architecture (Phase 6)

**Planned upgrade for production deployment:**

`

   Docker Compose Network (ai-ml-net)       

  Processing   Indexing      Streamlit    
  Container    Container     Containers   
                            (x2-3 scale)  
 - Download    - Chunk                    
 - Extract     - Embed      - Query       
 - Caption     - Index      - Retrieve    

  Shared Volume: ../data/ (ChromaDB)        
  Shared Network: ai-ml-net                 

`

**Benefits of Multi-Container:**
-  **Security:** Processing isolation from UI
-  **Efficiency:** UI startup ~15 sec (no processing delay)
-  **Scalability:** Multiple UI containers, single processor
-  **Reliability:** Process failure won't crash UI
-  **Best practices:** Microservices architecture

**Implementation plan:**
- [ ] Create `docker/Dockerfile.processing` (ingest + index stages)
- [ ] Create `docker/Dockerfile.ui` (lightweight Streamlit only)
- [ ] Update `docker-compose.yml` with orchestration
- [ ] Add health checks between containers
- [ ] Document CI/CD pipeline integration

**Estimated timeframe:** Phase 6 (post-mentor review)

