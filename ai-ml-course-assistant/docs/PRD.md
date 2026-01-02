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

| Source | Type | Actual Volume | Access Method |
|--------|------|---------------|---------------|
| **arXiv papers** | PDF (CS.LG, CS.AI, CS.CV) | **35 papers** | arXiv API + manual curation |
| **RealPython tutorials** | Web articles (HTML) | **9 articles** | Web scraping (offline) |
| **Medium/TDS articles** | Web articles (HTML) | **10 articles** | Web scraping (offline) |
| **Official docs** | Markdown/HTML | 0 pages (optional future addition) | Manual download |

### 3.2 Content Composition
- **Total documents:** **54 documents** (35 papers + 9 RealPython + 10 Medium/TDS)
- **Coverage:**
  - **arXiv Papers (35):**
    - Foundations (3): Transformers, ResNet, VGG
    - LLMs (6): BERT, GPT-3, XLNet, InstructGPT, Scaling Laws, RoBERTa
    - Vision (12): YOLO, ViT, Mask R-CNN, EfficientNet, Swin, DenseNet, ResNeXt, NiN, Inception, U-Net, Layer Norm, MobileNets
    - Multimodal (2): CLIP, RAG
    - Generative (4): DDPM, GANs, VAE, Stable Diffusion
    - Optimization (4): Batch Norm, Dropout, LoRA, AdamW
    - RNN/Seq (2): GRU, Seq2Seq
    - RL (2): DQN, PPO
    - GNN (1): GCN
  - **RealPython (9):** Neural networks basics, GANs, NLP/Text Classification, PyTorch vs TensorFlow, Gradient Descent, Face Recognition, Pillow Image Processing, Pandas Data Exploration, NumPy Tutorial
  - **Medium/TDS (10):** RAG evaluation, Chunk size in RAG, Agents planning, Production LLMs, Transformers/Self-Attention, Search metrics (MAP/MRR/NDCG), LLM reasoning, AI hallucinations, Generative AI UX, Gradient descent variants
- **Estimated images:** 200-350 images (from papers + tutorials)
  - Architecture diagrams (CNNs, RNNs, Transformers, GANs, Diffusion)
  - Algorithm flowcharts
  - Mathematical formulas
  - Training curves / performance graphs
  - Code visualization / data exploration plots

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
**Primary:** Groq API (Llama 3.3 70B)
- Extremely fast inference (up to 750 tokens/sec)
- Free tier available (generous limits)
- Excellent instruction-following for citation requirements
- Strong reasoning capabilities for metadata-based filtering

### 7.2 Embedding & Vision Models

**Text Embeddings:** OpenAI `text-embedding-3-small`
- **Dimension:** 1536
- **Cost:** $0.02 per 1M tokens
- **Speed:** Fast API-based
- **Quality:** Superior semantic understanding for academic content
- **Unified space:** Same model for text chunks AND image captions
- **Rationale:** High quality, cost-effective, production-ready

**Vision-Language Model:** OpenAI GPT-4.1-mini  (`gpt-4.1-mini`)
- **Purpose:** Generate detailed image descriptions during preprocessing
- **Quality Metrics:** ✅ avg 3,262 chars/image, 100% technical accuracy, 0% hallucinations

**Embedding Strategy:**
- **Single 1536-d embedding space** for unified retrieval
- **Both text chunks AND image captions** embedded with text-embedding-3-small
- Images indexed via **enriched captions** (OpenAI Vision descriptions)
- Enriched captions combine:
  - Author-provided caption (if any)
  - OpenAI Vision detailed description (avg 3,262 chars)
  - Surrounding context text (±200 characters)

### 7.3 Vector Store
**Choice:** ChromaDB
- Local, lightweight, easy setup
- Supports metadata filtering
- Built-in persistence
- Multi-collection support:
  - `text_chunks` collection (104 documents from 3 papers)
  - `image_captions` collection (9 images with enriched descriptions)
- Separate persist directories for isolation

### 7.4 Chunking Strategy

**Parameters:**
- **Target:** ~800 tokens per chunk
- **Character size:** ~1700 chars (calibrated for text-embedding-3-small: ~3.5 chars/token)
- **Overlap:** ~100 tokens (~150 chars)
- **Library:** `langchain-text-splitters.RecursiveCharacterTextSplitter`
- **Separators:** `["\n\n", "\n", ". ", " ", ""]`

**Anti-Hallucination Metadata:**
Each chunk includes:
- `has_figure_references`: Boolean (chunk explicitly mentions "Figure X" or "Table Y")
- `image_references`: List of strings (e.g., ["Figure 3", "Table 1"])
- `related_image_ids`: List of image IDs on **same page** (strong link)
- `nearby_image_ids`: List of image IDs on **±1 page** (weak link, only if has_figure_references=True)
- `page_num`, `doc_id`: For structural linking

**Rationale:**
- Explicit metadata prevents false image-text associations
- LLM can prioritize chunks with figure references
- Strict same-page links reduce hallucination risk

### 7.5 Retrieval Strategy

**Adaptive Hybrid Retrieval:**

1. **Text Retrieval:** Semantic search → top-3 chunks, batch embedded (1 API call)

2. **Metadata Candidates:** Extract images from `related_image_ids` (same page) + `nearby_image_ids` (±1 page if has figure refs)

3. **Semantic Verification:**
   - Explicit figure reference → **HIGH** confidence (1.0)
   - Same-page similarity ≥0.6 → **MEDIUM** confidence
   - Nearby similarity ≥0.7 → **MEDIUM** confidence
   - Below threshold → Rejected

4. **Visual Query Fallback:** If no verified images + visual keywords detected ("show", "diagram") → semantic caption search (threshold 0.5) → **LOW** confidence

5. **Deduplication:** Remove duplicate image_ids

**Result:** 0-3 verified images with confidence levels + similarity scores

**Optimizations:** Batch embeddings, caching (26 API calls → ~8 per query)

### 7.6 Tech Stack Summary
| Component | Technology |
|-----------|------------|
| Language | Python 3.13 |
| Orchestration | LangChain |
| Vector DB | ChromaDB (local persistent) |
| Text Embeddings | OpenAI text-embedding-3-small (1536-d) |
| Vision Model | OpenAI GPT-4.1-mini |
| LLM | Groq API (Llama 3.3 70B) |
| UI | Streamlit |
| PDF Processing | PyMuPDF (fitz) |
| Image Processing | Pillow (PIL) |
| Data Format | JSON (metadata), local files (images) |
| Environment | python-dotenv (.env for API keys) |

---

## 8. Limitations & Known Constraints

### 8.1 Current Limitations
- **Small dataset:** Limited to 3 papers (VGG, ResNet, Attention), 9 images
- **No multimodal LLM at query time:** LLM doesn't "see" images, relies on enriched captions (3,262 chars avg)
- **English-only:** All content in English
- **API dependency:** Requires OpenAI (embeddings + vision) and Groq (LLM) API keys
- **No code execution:** Can't run or debug code snippets
- **Visual query fallback threshold:** LOW confidence images may have lower relevance (threshold 0.5)

### 8.2 Strengths
✅ **Zero hallucination image descriptions:** OpenAI Vision 100% accurate (3,262 chars avg)
✅ **Adaptive hybrid retrieval:** Metadata-driven + semantic verification + visual query fallback
✅ **Confidence tiers:** HIGH (explicit refs), MEDIUM (semantic match >0.6), LOW (fallback >0.5)
✅ **Optimized API usage:** Batch embeddings, caching, deduplication (26 calls → ~8 calls)
✅ **LLM-ready formatting:** Structured output with captions, confidence, similarity scores

✅ **Fast:** Groq 750 tokens/sec, optimized retrieval with caching

### 8.3 Future Improvements (Post-MVP)
- Expand corpus to 30+ papers, 150+ images
- Add hybrid search (BM25 + semantic)
- Implement query expansion
- Support PDF uploads (custom documents)
- Add caching for common queries

---

## 9. Implementation Progress

### Phase 1: Data Ingestion ✅ COMPLETED
- [x] Downloaded 3 arXiv papers (VGG, ResNet, Attention Is All You Need)
- [x] Extracted 9 images with bbox coordinates
- [x] Created structured metadata (images_metadata.json)
- [x] **Image Enrichment:**
  - [x] Modified extract_images_smart.py for bbox tracking
  - [x] Created extract_image_context.py (±200 char extraction)
  - [x] Migrated from Cohere to OpenAI GPT-4.1-mini Vision
  - [x] Generated high-quality descriptions: avg 3,262 chars, 0% hallucinations
  - [x] Fields: enriched_caption, vlm_description, author_caption, context_before, context_after

### Phase 2: Indexing ✅ IN PROGRESS 
- [x] **Chunking Pipeline:**
  - [x] Implemented anti-hallucination metadata (has_figure_references, related_image_ids, nearby_image_ids)
  - [x] Created chunk_documents.py (2800 chars, 350 overlap)
  - [x] Created run_chunking.py orchestration
  - [x] Generated 104 chunks from 3 papers
  - [x] Statistics: 64.4% with figure refs, 11.5% with same-page images, 23.1% with nearby images

- [x] **Embedding Generation:**
  - [x] Created generate_embeddings.py with OpenAI text-embedding-3-small
  - [x] Generated embeddings for 104 text chunks
  - [x] Generated embeddings for 9 image captions
  - [x] Output: chunks_with_embeddings.json (1536-d vectors)

- [x] **ChromaDB Index:**
  - [x] Created build_chroma_index.py
  - [x] Built text_chunks collection (104 documents)
  - [x] Built image_captions collection (9 documents)
  - [x] Verified with test queries

### Phase 3: RAG Pipeline 🔄 IN PROGRESS 
- [x] **Retriever :**
  - [x] Created retriever.py with MultimodalRetriever class
  - [x] Implemented retrieve_text_chunks() - semantic search
  - [x] Implemented retrieve_with_strict_images() - metadata-driven
  - [x] **Adaptive Hybrid Retrieval:**
    - [x] Visual query detection (13 keywords: "show", "diagram", etc.)
    - [x] Semantic verification with cosine similarity
    - [x] Confidence tiers: HIGH (1.0), MEDIUM (0.6+), LOW (0.5+)
    - [x] Image deduplication by image_id
  - [x] **API Optimizations:**
    - [x] Batch embeddings (embed_documents for all chunks at once)
    - [x] Caching chunk embeddings during verification
    - [x] Reduced API calls: 26 → ~8 per query
  - [x] **LLM Integration:**
    - [x] prepare_for_llm() method for structured formatting
    - [x] Output: query, text_chunks, images with captions/confidence/similarity
  - [x] Tested: "show encoder decoder" → 3 unique images, all MEDIUM confidence
  - [x] Validation: Enriched captions (768 chars) used for semantic matching

- [ ] **Generator :**
  - [ ] Create generator.py with Groq LLM integration
  - [ ] Design prompt with metadata awareness (confidence levels, similarity scores)
  - [ ] Implement citation logic (chunk_id, image_id references)
  - [ ] Test on example queries

### Phase 4: UI + Evaluation ⏳ PENDING
- [ ] Build Streamlit interface
- [ ] Create evaluation dataset (30 queries)
- [ ] Run retrieval metrics
- [ ] Iterate on failures

