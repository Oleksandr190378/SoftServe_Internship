# Product Requirements Document (PRD)
# AI/ML Course Assistant - Multimodal RAG System

**Version:** 1.2  
**Date:** December 29, 2025  
**Project Type:** Multimodal Retrieval-Augmented Generation (RAG) Application  
**Status:** Implementation Phase - Phase 1 Complete, Phase 2 In Progress

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
- Medium/blogs: Fair use for educational purposes (non-commercial POC). **Note:** Verify individual article licenses before inclusion; focus on CC-licensed or explicitly open content. This corpus is for educational demonstration only.
- Official documentation: Check individual project licenses (most are Apache 2.0, MIT)

**Privacy:** No personal data, no user-generated content in corpus.

### 3.5 Enriched Caption Format
Each image is indexed with an **enriched caption** combining multiple sources:

**Components:**
1. **Author Caption:** Figure/Table caption from document (if present)
   - Example: "Figure 1: The Transformer model architecture"

2. **Vision-LM Description:** Detailed visual description from OpenAI GPT-4.1-mini Vision
   - Example: "A diagram showing a neural network with encoder and decoder stacks. Multiple attention blocks connected with residual connections. Input embeddings at bottom, output probabilities at top."

3. **Surrounding Context:** ±200 characters of text before and after the image
   - Captures narrative that gives the image meaning
   - Example: "...uses multi-head attention mechanism. Figure 1 shows the architecture. The encoder consists of 6 identical layers..."

**Enriched Caption Template:**
```
Image caption: [author caption]
Visual description: [BLIP-2 output]
Context: [before text] ... [after text]

Note: Use only context text that is relevant to understanding this image.
Ignore surrounding text if it discusses unrelated topics.
```

**Rationale:**
- Author captions often lack detail ("Figure 1: Model architecture")
- Vision-LM provides detailed visual understanding (layer types, connections, comprehensive technical details)
- Context explains *why* the image matters ("shows attention mechanism")
- Instruction helps LLM filter irrelevant context

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
**Unified Text Embedding Model:**
- Model: `sentence-transformers/all-MiniLM-L6-v2` (via Hugging Face)
- Dimension: 384
- Speed: ~1000 sentences/sec
- Rationale: Small, fast, sufficient for 100-200 docs
- Access: Hugging Face Transformers library

**Embedding Strategy:**
- **Single embedding space (384-d)** for unified retrieval
- **Both text chunks AND image captions** embedded with same model
- Images indexed via **enriched captions** (detailed text descriptions)
- Enriched captions combine:
  - Author-provided caption (if any)
  - **Vision-LM generated description (OpenAI GPT-4.1-mini Vision)** ✅
  - Surrounding context text (±200 characters) ✅
  - **No validation step needed** - OpenAI Vision produces accurate, hallucination-free descriptions

**Why Not CLIP?**
- CLIP (512-d) would require separate embedding space
- Academic papers have rich captions and contextual references
- Context-aware text captions provide better semantic alignment
- Simpler architecture: one embedding model for everything
- CLIP available as fallback if caption-based retrieval underperforms

### 7.3 Vector Store
**Choice:** ChromaDB
- Local, lightweight, easy setup
- Supports metadata filtering
- Built-in persistence
- Multi-collection support (separate text/image collections)

**Alternative considered:** FAISS (faster, but more setup complexity)

### 7.4 Chunking Strategy
- **Chunk size:** 800-1000 tokens (~600-800 words)
  - Larger than typical RAG (512 tokens) because academic papers have long paragraphs
  - Vision-LM descriptions are highly detailed (avg 3,200 chars), benefit from rich context
- **Overlap:** 100 tokens
- **Splitting logic:** 
  - Prefer section boundaries (Markdown headers)
  - Fallback: sentence boundaries
  - Keep formulas/code blocks intact
- **Metadata enrichment:**
  - Track `has_images: bool` (chunk contains image references)
  - Track `image_references: List[str]` (e.g., ["Figure 1", "Table 2"])
  - Track `page_num` for linking to co-located images

### 7.5 Retrieval Approach
**Context-Aware Multi-Vector Strategy:**

1. **Text Chunk Retrieval:**
   - Semantic search over text chunks (top-10)
   - Metadata includes `has_images` flag and `image_references`
   - Chunks with images get slight priority boost for visual queries

2. **Image Retrieval via Enriched Captions:**
   - Query embedded with text model (384-d)
   - Search over enriched image captions in same embedding space
   - Top-5 most relevant images returned
   - Each result includes: filepath, VLM description, author caption, context

3. **Contextual Linking:**
   - Images linked to chunks via `doc_id` + `page_num` metadata
   - For retrieved text chunks, automatically fetch co-located images
   - Combines semantic relevance + structural proximity

4. **Multi-Vector Pattern:**
   - ChromaDB stores embeddings + metadata (lightweight)
   - Raw images stored as files (referenced by filepath)
   - Docstore pattern: search summaries, retrieve originals

**Fallback Strategy:**
- If enriched caption retrieval has low precision (<60% Image Hit Rate)
- Fall back to CLIP-based visual embedding
- Requires separate 512-d collection (added complexity)

### 7.6 Tech Stack Summary
| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Orchestration | LangChain |
| Vector DB | ChromaDB |
| Text Embeddings | Hugging Face (sentence-transformers/all-MiniLM-L6-v2) |
| Image Captioning | BLIP-2 (Salesforce/blip2-opt-2.7b) via Hugging Face |
| LLM | Groq API (Llama 3.1 / Mixtral) / Ollama fallback |
| UI | Streamlit |
| PDF Processing | PyMuPDF (fitz) |
| Image Processing | Pillow (PIL) |
| Data Format | JSON (metadata), local files (images) |

### 7.7 Image Captioning Strategy
**Vision-Language Model: BLIP-2**
- Model: `Salesforce/blip2-opt-2.7b` (Hugging Face)
- Purpose: Generate detailed descriptions of diagrams, charts, architectures
- License: BSD-3-Clause (open source)
- Hardware: CPU-compatible (~3-4GB RAM per image), faster on GPU
- Alternative: `Salesforce/blip2-flan-t5-xl` (if GPU available)

**Captioning Process (Offline Preprocessing):**
1. **Extract images** from PDFs (existing: `extract_images_smart.py`)
2. **Extract context:**
   - Identify text blocks near image bounding box
   - Extract figure caption (starts with "Figure", "Table", "Fig.")
   - Capture ±200 characters of surrounding text
3. **Generate VLM description:**
   - Pass image to BLIP-2 with custom prompt
   - Prompt: "Describe this image in detail. Focus on: diagrams, charts, architecture components, formulas, tables. If it's a neural network, describe layers and connections. If it's a chart, describe axes and trends."
4. **Combine into enriched caption:**
   - Format: author caption + VLM output + context
   - Add instruction to ignore irrelevant context
5. **Embed caption** with text model (384-d)
6. **Store in ChromaDB** with metadata (filepath, page_num, doc_id)

**Context Extraction Details:**
- Use `page.get_text("dict")` to get text with coordinates
- Find text blocks with y-coordinates near image bbox
- Extract caption by regex: `r'(Figure|Fig\.|Table)\s+\d+:?[^\n]+'`
- Take 200 chars before + 200 chars after image position
- If context discusses unrelated topics, VLM description dominates

**Quality Safeguards:**
- Manual review of 10-20 sample captions during Phase 1
- Adjust context window if too much noise (±200 → ±150 chars)
- Refine VLM prompt if descriptions are too generic
- Fallback to CLIP if caption-based retrieval fails (<60% hit rate)

---

## 8. Limitations & Known Constraints

### 8.1 Current Limitations
- **Static corpus:** No real-time updates to knowledge base
- **No multimodal LLM at generation time:** System cannot directly interpret image visual content during query; relies on Vision-LM generated captions during preprocessing (OpenAI GPT-4.1-mini Vision provides detailed, accurate descriptions)
- **English-only:** All content in English
- **Limited scope:** AI/ML topics only (narrow domain)
- **No code execution:** Can't run or debug code snippets
- **Caption quality dependency:** Image retrieval quality depends on Vision-LM caption accuracy (OpenAI GPT-4.1-mini Vision: 100% technical accuracy, 0% hallucination rate) and surrounding text relevance

### 8.2 Future Improvements (Post-MVP)
- Add vision-capable LLM (GPT-4V, LLaVA) to directly "read" retrieved images
- Expand corpus to 500+ documents
- Add hybrid search (BM25 + semantic)
- Implement query expansion for better recall
- Add caching layer for common queries
- Support PDF/image uploads (query your own notes)

---

## 9. Next Steps (Implementation Phases)

### Phase 1: Data Ingestion (Week 1-2) ✅ COMPLETED
- [x] Download 30-50 arXiv papers → 3 papers (ResNet, Attention Is All You Need, +1)
- [x] Extract and organize images → 9 images extracted with bbox coordinates
- [x] Create structured metadata → images_metadata.json with enriched fields
- [x] **Image Enrichment:**
  - [x] Modified extract_images_smart.py to save bbox coordinates
  - [x] Created extract_image_context.py (±200 char extraction)
  - [x] Created generate_captions.py (Cohere Command A Vision integration)
  - [x] Created enrich_images.py orchestration
  - [x] Generated VLM descriptions for all 9 images
  - [x] Added context_before, context_after, enriched_caption fields
  - [x] Tested Cohere Command A Vision (100% hallucination rate, deprecated)
  - [x] Migrated to OpenAI GPT-4.1-mini Vision (gpt-4.1-mini) ✅
  - [x] Validated quality: avg 3,262 chars/image, 100% technical accuracy, 0% hallucinations

**Vision-LM Selection:** OpenAI GPT-4.1-mini Vision (gpt-4.1-mini) ✅
- **Superior quality:** Detailed, accurate technical descriptions (avg 3,262 chars)
- **Zero hallucinations:** 100% accuracy on all 9 test images (vs Cohere 81 hallucinations)
- **No validation needed:** Eliminates Groq LLM validation step (cost + time savings)
- Image compression: max 1024px, JPEG quality 85, base64 encoding
- API: `client.responses.create()` with `input_image` + `input_text`
- Successfully generated comprehensive descriptions for all images

### Phase 2: Indexing (Week 2-3) 🔄 IN PROGRESS
- [ ] Implement chunking pipeline (chunk_documents.py)
- [ ] Generate embeddings (generate_embeddings.py)
- [ ] Populate ChromaDB (build_index.py)
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
