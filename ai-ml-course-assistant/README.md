# 🎓 AI/ML Course Assistant - Multimodal RAG System

A production-ready multimodal Retrieval-Augmented Generation (RAG) system that helps students learn AI/ML concepts by retrieving relevant text and images (diagrams, architectures, formulas) from academic papers and tutorials.

## 📋 Project Overview

**End-to-end RAG pipeline with:**
- ✅ Multi-source ingestion (PDF papers, JSON articles)
- ✅ GPT-4o-mini  for image captioning
- ✅ Semantic text chunking with figure detection
- ✅ OpenAI embeddings (text-embedding-3-small, 1536d)
- ✅ ChromaDB vector store with metadata anti-hallucination
- ✅ MMR retrieval for diversity + semantic verification
- ✅ Automated evaluation framework
- ⏳ LLM answer generation (next phase)

**Status:** Phases 1-4 Complete (Retrieval Ready) ✅

---

## 🎯 Current Capabilities

**Working queries (retrieval tested):**
1. "What is LSTM?" → 3 sequential text chunks from arxiv paper
2. "Show LSTM architecture" → 2 relevant diagrams with MEDIUM confidence
3. "How do AI agents plan tasks?" → Diverse chunks from multiple documents
4. "Explain sequence to sequence model with diagram" → Text + images with HIGH confidence

**Test Results:** 87.5% image hit rate, MMR diversity validated ✅

---

## 📂 Project Structure

├── data/
│   ├── raw_data/                  # 54 documents (PDFs + JSON articles)
│   ├── processed/                 # Extracted text, images, metadata
│   │   ├── processed_docs.json    # Registry (3/54 documents indexed)
│   │   ├── images_metadata.json   # Image captions (14 VLM-described)
│   │   └── images/                # Extracted images (18 total)
│   └── chroma_db/                 # Vector store (73 chunks, 14 images)
│
├── ingest/                        # Stage 1-2: Extraction + Captioning
│   ├── pdf_extractor.py           # PyMuPDF extraction
│   └── json_extractor.py          # Web article extraction
│
├── index/                         # Stage 3-5: Chunking + Embedding + Indexing
│   ├── semantic_chunker.py        # LangChain RecursiveCharacterTextSplitter
│   ├── embedding_utils.py         # OpenAI embeddings wrapper
│   └── chromadb_indexer.py        # ChromaDB operations
│
├── rag/                           # Retrieval (Stage 6-7 next)
│   ├── retriever.py               # MMR + semantic verification ✅
│   └── generator.py               # LLM answer generation (TODO)
│
├── eval/                          # Evaluation framework
│   ├── test_retrieval_indexed.py  # 8-query test suite ✅
│   ├── test_queries.json          # 30 test queries (10 text, 10 visual, 10 hybrid)
│   └── results/                   # Test logs + JSON summaries
│
├── docs/                          # Documentation
│   ├── PRD.md                     # Requirements ✅
│   ├── ROADMAP.md                 # 7-phase plan ✅
│   ├── PIPELINE_GUIDE.md          # User guide (70+ KB) ✅
│   ├── retrieval_strategy_analysis.md  # MMR vs similarity analysis ✅
│   ├── data_sources.md            # 54 documents catalog ✅
│   └── QUICKSTART.md              # 5-min setup guide ✅
│
├── run_pipeline.py                # Main CLI (process, status) ✅
├── delete_doc_from_chromadb.py    # Utility for cleanup ✅
└── requirements.txt               # Dependencies ✅e rules
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (or Ollama for local LLM)
- 2-3 GB disk space for data

### Installation

1. **Clone/Navigate to project:**
```bash
cd c:\Users\Користувач\Documents\python_1\softserve\ai-ml-course-assistant
```

2. **Create virtual environment:**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key (or configure Ollama)
```

### Processing Documents

See **[📖 Pipeline User Guide](docs/PIPELINE_GUIDE.md)** for comprehensive documentation.

**Quick commands:**
```bash
# Show status
python run_pipeline.py status

# Process specific document
python run_pipeline.py process --doc-id arxiv_1706_03762

# Process all documents
python run_pipeline.py process --all --no-vlm  # Cost: ~$0.01
python run_pipeline.py process --all           # Cost: ~$1.50-3.00 (with VLM)
```

---

## 📊 Development Roadmap

### ✅ Phase 0: Planning (COMPLETED)
- [x] Choose application domain (AI/ML education)
- [x] Write PRD document
- [x] Define success metrics
- [x] Select technical stack

### 🔄 Phase 1: Data Ingestion (NEXT - Week 1-2)
- [ ] Download 30-50 arXiv papers (CS.LG, CS.AI)
- [ ] Scrape 20-30 technical blog articles
- [ ] Extract images from PDFs
- [ ] Create structured metadata (documents.json, images.json)

### ⏳ Phase 2: IndProgress

### ✅ Phase A: Planning & Setup (COMPLETED)
- [x] PRD document with success metrics
- [x] Technical stack selection
- [x] 54 documents curated (arXiv, Medium, RealPython)

### ✅ Phase B: Data Pipeline (COMPLETED)
- [x] PDF/JSON extractors (Stage 1)
- [x] GPT-4o-mini Vision captioning (Stage 2) - $0.015/image
- [x] Semantic chunking with figure detection (Stage 3)
- [x] OpenAI embeddings (Stage 4) - text-embedding-3-small
- [x] ChromaDB indexing (Stage 5)
- [x] Registry system for incremental processing

### ✅ Phase C: Retrieval (COMPLETED)
- [x] MMR search for text diversity (λ=0.7)
- [x] Semantic verification for images
- [x] Anti-hallucination metadata (has_figure_references, related_image_ids)
- [x] Document-filtered fallback search
- [x] 87.5% image hit rate validated

### ✅ Phase D1: Evaluation Details |
|-----------|-----------|---------|
| **Language** | Python 3.11+ | Type hints, modern syntax |
| **Orchestration** | LangChain | RAG pipeline, document loaders |
| **Vector DB** | ChromaDB | 2 collections (text_chunks, image_captions) |
| **Embeddings** | OpenAI `text-embedding-3-small` | 1536d, $0.00002/1K tokens |
| **Vision** | OpenAI `gpt-4o-mini-2024-07-18` | Image captioning, $0.015/image |
| **LLM** | OpenAI `gpt-4o-mini` (planned) | Answer generation, $0.150/1M in |
| **PDF Processing** | PyMuPDF (fitz) | Text + image extraction |
| **Image Processing** | Pillow (PIL) | WebP conversion, metadata |
| **Retrieval** | MMR + Similarity | Diversity for text, relevance for images
- [ ] Citation system
- [ ] Hallucination prevention ("I don't know")
- [ ] evaluate_answers.py (Faithfulness, Citation Accuracy)


### Process Documents
```bash
# Check status
python run_pipeline.py status

# Process single document (fast, no VLM cost)
python run_pipeline.py process --doc-id arxiv_1706_03762 --no-vlm

# Process all new documents (51 remaining, ~$0.01)
python run_pipeline.py process --all --no-vlm

# Force reprocess with VLM ($0.015 per image)
python run_pipeline.py process --doc-id arxiv_1706_03762 --force
```

**See [docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md) for full documentation.**

### Test Retrieval
```bash
# Run 8-query test suite
python eval/test_retrieval_indexed.py

# Results in eval/results/retrieval_test_<timestamp>.txt
```

### Utilities
```bash
# Delete document from ChromaDB
python delete_doc_from_chromadb.py arxiv_1706_03762

# Clean registry for re-indexing
python clean_registrycess.py
```

### Step 2: Build Index
```bash
python index/build_index.py
```

### Step 3: Run UI
```bash
streamlit run ui/app.py Notes |
|--------|--------|--------|-------|
| **Image Hit Rate** | ≥ 60% | ✅ **87.5%** | 7/8 test queries retrieved images |
| **Text Diversity** | Sequential coherence | ✅ **Validated** | MMR retrieves chunks 4→5→6 (not 4→5→10) |
| Recall@5 (text) | ≥ 70% | ⏳ Pending | Ground truth labeling needed |
| Faithfulness | ≥ 80% | ⏳ Phase E | Answer generation not yet implemented |
| Citation Accuracy | ≥ 85% | ⏳ Phase E | Generator needed |
| Query Latency | < 5 sec | ⏳ Phase G | Latency profiling pending |

**Current Focus:** (54 Documents Curated)

1. **arXiv Papers (24 PDFs)**
   - Deep Learning, Neural Networks, Transformers
   - Examples: Attention Is All You Need, LSTM, BERT
   
2. **Medium Articles (16 JSON)**
   - AI/ML tutorials and explanations
   - Topics: Agents, transformers, optimization

3. **RealPython Tutorials (14 JSON)**
   - Python ML libraries (NumPy, Pandas, Matplotlib)

**Processed:** 3 documents (73 chunks, 14 images in ChromaDB)  
**Remaining:** 51 documents (~$0.01 to process with --no-vlm)

**See [docs/data_sources.md](docs/data_sources.md) for full catalog

1. **arXiv Papers** (30-50 papers)
   - Categories: cs.LG, cs.AI, cs.CV
   - Topics: Deep Learning, Neural Networks, Optimization
   
2. **Technical Blogs** (20-30 articles)
   - Medium, Towards Data Science
   - Topics: Tutorials, architecture explanations
   
3. **Wikipedia** (15-25 articles)
   - ML/AI concept pages with diagrams

4. **Official Documentation** (10-15 pages)
   - PyTorch, TensorFlow, Scikit-learn docs

**See [docs/data_sources.md](docs/data_sources.md) for detailed list.**

---

## 🤝 Contributing

This is an educational project. Key constraints:
- ✅ Pure RAG (retrieval + generation)
- ❌ No agentic behavior
- ❌ No real-time tool calling
- ❌ Preprocessing scripts are OK

---

## 📝 License

Educational project for learning purposes. 
- a**Create ground truth** for 30 test queries → Recall@5, MRR metrics
2. **Implement generator.py** → LLM answer generation with citations
3. **Full pipeline run** → Process 51 remaining documents (~4-25 min)
4. **Evaluation** → Faithfulness, Citation Accuracy metrics
5. **Streamlit UI** → Interactive query interface

**Current milestone:** Phase D2 (Ground Truth Creation) ⏳

---

## 📚 Documentation

- **[QUICKSTART.md](docs/QUICKSTART.md)** - 5-minute setup guide
- **[PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md)** - Full user guide (70+ KB)
- **[ROADMAP.md](docs/ROADMAP.md)** - 7-phase development plan
- **[retrieval_strategy_analysis.md](docs/retrieval_strategy_analysis.md)** - MMR vs similarity comparison
- **[PRD.md](docs/PRD.md)** - Product requirements
- **[data_sources.md](docs/data_sources.md)** - Document catalog

## 📞 Contact

**Project:** AI/ML Course Assistant  
**Status:** In Development (Step 0 Complete)  
**Documentation:** See [docs/PRD.md](docs/PRD.md)

---

## 🎯 Next Steps

1. Review the PRD: [docs/PRD.md](docs/PRD.md)
2. Begin Phase 1: Data ingestion scripts
3. Test with small dataset (5-10 papers) first

**Ready to start coding!** 🚀
