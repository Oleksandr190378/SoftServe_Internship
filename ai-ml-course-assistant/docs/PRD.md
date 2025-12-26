# Product Requirements Document (PRD)
# AI/ML Course Assistant - Multimodal RAG System

**Version:** 1.0  
**Date:** December 24, 2025  
**Project Type:** Multimodal Retrieval-Augmented Generation (RAG) Application  
**Status:** POC / MVP Phase

---

## 1. Problem & Users

### 1.1 Target Users
- **Primary:** Students learning AI/Machine Learning concepts (undergraduate to graduate level)
- **Secondary:** Self-taught developers transitioning into AI/ML
- **Tertiary:** Technical professionals needing quick reference to AI/ML concepts

### 1.2 Problem Statement
Students and learners often struggle with:
- **Information overload:** Too many resources scattered across papers, tutorials, and documentation
- **Visual understanding:** Many AI/ML concepts (neural network architectures, optimization algorithms, mathematical formulations) are best understood through diagrams and visualizations
- **Context switching:** Constantly switching between text explanations, code examples, and visual representations
- **Finding relevant examples:** Difficulty locating specific diagrams, formulas, or code snippets when studying

### 1.3 Solution
A multimodal RAG system that allows users to:
- Query natural language questions about AI/ML topics
- Receive text answers grounded in authoritative sources
- View relevant diagrams, architecture visualizations, and formula images
- Access both conceptual explanations and visual representations in one interface

---

## 2. MVP Scope

### 2.1 In Scope (v1.0)
✅ **Core Features:**
- Text-based query input (natural language)
- Retrieval from curated AI/ML educational content (papers, tutorials, documentation)
- Multimodal retrieval (text chunks + images)
- Answer generation with source citations
- Display of retrieved text sources
- Display of relevant images (diagrams, architectures, formulas)
- Simple web UI (Streamlit)

✅ **Content Types:**
- Academic papers (arXiv CS.LG, CS.CV, CS.AI categories)
- Technical blog posts (Medium, Towards Data Science)
- Official documentation (PyTorch, TensorFlow, Scikit-learn)
- Wikipedia articles on ML topics

✅ **Query Types:**
- Conceptual questions ("What is backpropagation?")
- Visual queries ("Show me the architecture of ResNet")
- Formula/equation queries ("What is the gradient descent update rule?")
- Comparison queries ("Difference between RNN and LSTM?")

### 2.2 Out of Scope (Explicitly NOT included)
❌ **Agentic behavior:** No autonomous planning, multi-step reasoning chains
❌ **Real-time tool calling:** No web scraping, API calls, or browsing at query time
❌ **Code execution:** No running code snippets or notebooks
❌ **Interactive learning:** No quizzes, flashcards, or progress tracking
❌ **User accounts:** No authentication or personalization
❌ **Multi-turn conversations:** No memory of previous queries (stateless)
❌ **Content generation:** No creating new diagrams or visualizations

---

## 3. Content & Data

### 3.1 Data Sources

| Source | Type | Approximate Volume | Access Method |
|--------|------|-------------------|---------------|
| **arXiv papers** | PDF (CS.LG, CS.AI) | 30-50 papers | arXiv API + manual curation |
| **Medium/TDS articles** | Web articles (HTML) | 20-30 articles | Web scraping (offline) |
| **Wikipedia** | Structured text + images | 15-25 articles | Wikipedia API |
| **Official docs** | Markdown/HTML | 10-15 documentation pages | Manual download |

### 3.2 Content Composition
- **Total documents:** 75-120 documents
- **Total images:** 150-400 images
  - Architecture diagrams (CNNs, RNNs, Transformers, etc.)
  - Algorithm flowcharts
  - Mathematical formula images
  - Training curves / performance graphs
  - Conceptual illustrations

### 3.3 Image-Text Relationship
Images are linked to text through:
- **Document co-location:** Images extracted from the same PDF/article
- **Metadata linkage:** `doc_id` field connects images to parent documents
- **Caption association:** Alt-text, figure captions, or generated descriptions
- **Folder structure:** `/data/raw/{doc_id}/images/`

### 3.4 Data Licensing
- arXiv: Open access (various licenses, mostly permissive)
- Wikipedia: CC BY-SA 3.0
- Medium/blogs: Fair use for educational purposes (non-commercial POC)
- Official documentation: Check individual project licenses (most are Apache 2.0, MIT)

**Privacy:** No personal data, no user-generated content in corpus.

---

## 4. Example Queries

### 4.1 Text-Focused Queries (5 examples)
1. "What is the difference between supervised and unsupervised learning?"
2. "Explain how gradient descent optimization works"
3. "What are the main types of neural network layers?"
4. "How does dropout regularization prevent overfitting?"
5. "What is the vanishing gradient problem?"

### 4.2 Image-Required Queries (5 examples)
6. **"Show me the architecture diagram of a Convolutional Neural Network"**
7. **"What does the Transformer model architecture look like?"**
8. **"Display the formula for cross-entropy loss"**
9. **"Show me a visualization of how backpropagation flows through layers"**
10. **"Find diagrams comparing different activation functions"**

### 4.3 Hybrid Queries (2 examples)
11. "Explain ResNet and show its skip connection architecture"
12. "What is attention mechanism? Show the calculation formula"

---

## 5. Success Metrics

### 5.1 Retrieval Quality Metrics
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Recall@5 (text)** | ≥ 70% | Manual evaluation on 30 queries |
| **Image Hit Rate** | ≥ 60% | % of image queries returning ≥1 relevant image in top-5 |
| **MRR (Mean Reciprocal Rank)** | ≥ 0.5 | Position of first relevant result |

### 5.2 Answer Quality Metrics
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Faithfulness** | ≥ 80% | % of answers supported by retrieved sources (manual check) |
| **Citation Accuracy** | ≥ 85% | % of citations actually present and relevant |
| **"I don't know" correctness** | 100% | System says "I don't know" when context insufficient |

### 5.3 Performance Metrics
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Query latency** | < 5 seconds | End-to-end time (retrieval + generation) |
| **UI responsiveness** | < 1 second | Time to display loading state |

### 5.4 User Experience (Qualitative)
- Answers are clear and concise (not overly verbose)
- Images are visually readable (not pixelated)
- Sources are traceable (user can verify claims)
- Interface is intuitive (no training needed)

---

## 6. UI Expectations

### 6.1 Main Interface Components

**Query Input:**
- Text input box (supports multi-line)
- Submit button
- Optional: Sample query suggestions (clickable)

**Results Display:**
- **Answer Panel:** 
  - Generated answer with citations (e.g., [1], [2])
  - Confidence indicator (optional: "High confidence" / "Limited information")
  
- **Text Sources Panel:**
  - List of retrieved documents with:
    - Document title (linked)
    - Relevant text snippet (highlighted chunk)
    - Source type (paper/article/docs)
    - Relevance score (optional)

- **Image Gallery Panel:**
  - Thumbnail view of relevant images (3-6 images)
  - Click to expand full-size
  - Caption/alt-text below each image
  - Source document reference

**Status Indicators:**
- Loading spinner during retrieval/generation
- Progress messages ("Searching documents...", "Generating answer...")
- Error messages if query fails

### 6.2 Optional/Nice-to-Have
- Toggle for retrieval mode (text-only vs. multimodal)
- Adjustable top-k slider (5, 10, 15 results)
- Download citations (BibTeX format)
- Dark mode toggle

---

## 7. Technical Choices

### 7.1 LLM Selection
**Primary:** Groq API (Llama 3.1 70B or Mixtral 8x7B)
- Extremely fast inference (up to 750 tokens/sec)
- Free tier available (generous limits)
- Open source models (Llama, Mixtral)
- Good instruction-following for citation requirements

**Fallback:** Local Ollama (llama3.1:8b)
- No API costs
- Full privacy
- Slower but acceptable for demo

### 7.2 Embedding Models
**Text Embeddings:**
- Model: `sentence-transformers/all-MiniLM-L6-v2` (via Hugging Face)
- Dimension: 384
- Speed: ~1000 sentences/sec
- Rationale: Small, fast, sufficient for 100-200 docs
- Access: Hugging Face Transformers library

**Image Embeddings:**
- Model: `openai/clip-vit-base-patch32` (open source, via Hugging Face)
- Alternative: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` (LAION trained)
- Dimension: 512
- Rationale: CLIP enables joint text-image embedding space
- License: MIT (fully open source)

### 7.3 Vector Store
**Choice:** ChromaDB
- Local, lightweight, easy setup
- Supports metadata filtering
- Built-in persistence
- Multi-collection support (separate text/image collections)

**Alternative considered:** FAISS (faster, but more setup complexity)

### 7.4 Chunking Strategy
- **Chunk size:** 512 tokens (~400 words)
- **Overlap:** 50 tokens
- **Splitting logic:** 
  - Prefer section boundaries (Markdown headers)
  - Fallback: sentence boundaries
  - Keep formulas/code blocks intact

### 7.5 Retrieval Approach
**Hybrid Strategy:**
1. Text retrieval: Semantic search over text chunks (top-10)
2. Image retrieval: 
   - **Approach A (primary):** CLIP text encoder on query → retrieve images directly
   - **Approach B (fallback):** Retrieve text first, then fetch linked images
3. Re-ranking: Optional (may use cross-encoder for top-5)

### 7.6 Tech Stack Summary
| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Orchestration | LangChain |
| Vector DB | ChromaDB |
| Text Embeddings | Hugging Face (sentence-transformers/all-MiniLM-L6-v2) |
| Image Embeddings | CLIP (openai/clip-vit-base-patch32, open source) |
| LLM | Groq API (Llama 3.1 / Mixtral) / Ollama fallback |
| UI | Streamlit |
| PDF Processing | PyMuPDF (fitz) |
| Image Processing | Pillow (PIL) |
| Data Format | JSON (metadata), local files (images) |

---

## 8. Limitations & Known Constraints

### 8.1 Current Limitations
- **Static corpus:** No real-time updates to knowledge base
- **No multimodal LLM:**
- **English-only:** All content in English
- **Limited scope:** AI/ML topics only (narrow domain)
- **No code execution:** Can't run or debug code snippets

### 8.2 Future Improvements (Post-MVP)
- Add vision-capable LLM (GPT-4V, LLaVA) to directly "read" retrieved images
- Expand corpus to 500+ documents
- Add hybrid search (BM25 + semantic)
- Implement query expansion for better recall
- Add caching layer for common queries
- Support PDF/image uploads (query your own notes)

---

## 9. Next Steps (Implementation Phases)

### Phase 1: Data Ingestion (Week 1-2)
- [ ] Download 30-50 arXiv papers
- [ ] Scrape 20-30 blog articles
- [ ] Extract and organize images
- [ ] Create structured metadata

### Phase 2: Indexing (Week 2-3)
- [ ] Implement chunking pipeline
- [ ] Generate embeddings
- [ ] Populate ChromaDB
- [ ] Test retrieval quality

### Phase 3: RAG Pipeline (Week 3-4)
- [ ] Build LangChain retrieval chain
- [ ] Design prompts for grounded answers
- [ ] Implement citation logic
- [ ] Test on example queries

### Phase 4: UI + Evaluation (Week 4-5)
- [ ] Build Streamlit interface
- [ ] Create evaluation dataset (30-50 queries)
- [ ] Run metrics
- [ ] Iterate on failures

---

## 10. Success Criteria for Step 0 (PRD Completion)

✅ **This PRD is approved when:**
1. Application domain is clearly defined (AI/ML education)
2. User personas and problem statement are explicit
3. MVP scope has clear boundaries (RAG-only, no agents)
4. Data sources are identified and accessible
5. 12 example queries are provided (5+ image-related)
6. Success metrics are measurable
7. Technical stack is justified
8. Limitations are acknowledged

---

**Document Status:** ✅ READY FOR IMPLEMENTATION

**Prepared by:** AI/ML Course Assistant Team  
**Reviewed by:** [Pending instructor/mentor review]

---

**Appendix A: Glossary**
- **RAG:** Retrieval-Augmented Generation
- **CLIP:** Contrastive Language-Image Pre-training
- **ChromaDB:** Open-source embedding database
- **arXiv:** Open-access archive for scientific papers
- **MRR:** Mean Reciprocal Rank (evaluation metric)
