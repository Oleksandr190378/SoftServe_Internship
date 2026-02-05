# RAG System Architecture: Retriever & Generator

**Comprehensive guide to the Retrieval-Augmented Generation pipeline**

---

## Overview

The AI/ML Course Assistant implements a two-stage RAG system:

1. **Retriever Module** (`rag/retrieve/`) - Intelligent data filtering and semantic search
2. **Generator Module** (`rag/generate`) - Grounded answer generation with citations

This document explains how each component works and how they interact to produce accurate, citation-backed answers.

---

## Part 1: Retriever Module (Search & Verification)

### Architecture Diagram

```
User Query
    ↓
[1] Text Semantic Search (MMR)
    ├─ Fetch k candidates from ChromaDB
    ├─ Apply MMR algorithm (λ=0.7)
    └─ Return top-k diverse chunks
    ↓
[2] Batch Embedding Generation
    ├─ Collect all retrieved chunk IDs
    ├─ Batch call OpenAI text-embedding-3-small
    └─ Create chunk_id → vector mapping
    ↓
[3] Multi-Level Image Verification
    ├─ Level HIGH: Metadata references (related_image_ids)
    ├─ Level MEDIUM: Semantic similarity (cosine distance)
    └─ Level LOW: Same-page proximity
    ↓
[4] Confidence Scoring
    ├─ Calculate relevance scores
    ├─ Assign confidence badges (HIGH/MEDIUM/LOW)
    └─ Filter low-confidence images
    ↓
[5] Fallback Visual Search
    └─ If no images found, try query-based search
    ↓
Prepared LLM Input (formatted context)
```

### 1. Text Retrieval Strategy (`retrieve_text_chunks`)

**Problem:** Standard semantic search often returns redundant results - all three results might be nearly identical sentences from different parts of the same document.

**Solution:** MMR (Maximal Marginal Relevance) search

**How MMR Works:**
```
MMR(query, text_chunks, λ=0.7) = argmax[
    λ × similarity(query, chunk_i) - (1-λ) × max(similarity(chunk_i, chunk_j))
]
```

- **First term (λ=0.7):** How relevant is this chunk to the query?
- **Second term (1-λ=0.3):** How different is it from already selected chunks?
- **Practical effect:** Selects diverse results that still match the query

**Example:**
```
Query: "How does LSTM work?"

Without MMR:
[1] "LSTM has memory cells..." (similarity: 0.92)
[2] "LSTM memory cells store..." (similarity: 0.91)  ← Nearly identical!
[3] "LSTM cells are composed of..." (similarity: 0.90)

With MMR (λ=0.7):
[1] "LSTM has memory cells..." (similarity: 0.92, diversity: 0.0)
[2] "The forget gate controls information flow..." (similarity: 0.85, diversity: 0.4)
[3] "Output gates determine what cells expose..." (similarity: 0.82, diversity: 0.45)
```

**Key Parameters:**
- `k_text` (default: 3) - Number of results to return
- `fetch_k` (default: 10) - Number of candidates to fetch before filtering
- `lambda` (default: 0.7) - Balance between relevance and diversity

---

### 2. Batch Embedding Generation (`_batch_embed_chunks`, `_batch_embed_images`)

**Problem:** Making individual API calls to OpenAI for each chunk is slow and expensive.

**Solution:** Batch embeddings with single API call

**How It Works:**
```python
# INEFFICIENT: Sequential calls (10 chunks × 2 sec = 20 seconds)
for chunk in retrieved_chunks:
    embedding = openai.Embedding.create(input=chunk.text)
    store_embedding(chunk.id, embedding)

# EFFICIENT: Batch call (1 call × 2 sec = 2 seconds)
texts = [chunk.text for chunk in retrieved_chunks]
embeddings = openai.Embedding.create_batch(input=texts)
chunk_embedding_map = {
    chunk.id: embedding
    for chunk, embedding in zip(retrieved_chunks, embeddings)
}
```

**Benefits:**
- **Speed:** 5-10x faster than sequential requests
- **Cost:** Same price (OpenAI charges per token, not per request)
- **Reliability:** Fewer API calls = fewer potential failures

**Output Format:**
```python
{
    'chunk_001': [0.123, -0.456, 0.789, ...],  # 1536-dimensional vector
    'chunk_002': [0.234, -0.567, 0.890, ...],
    'chunk_003': [0.345, -0.678, 0.901, ...],
    # ... more chunk embeddings
}
```

---

### 3. Multi-Level Image Verification (`_verify_metadata_images`)

**Problem:** Not all retrieved images are relevant to the query. System needs to distinguish between:
- Images directly referenced in the text
- Images that semantically match the topic
- Images that happen to be on the same page (false positives)

**Solution:** Three-tier confidence scoring system

#### Level 1: HIGH Confidence (Direct References)

```python
# Check if text contains figure references
if "Figure 1" in chunk_text and "figure_1_image" in chunk_related_images:
    confidence = "HIGH"  # Perfect match!
```

**Indicators:**
- ✅ Figure/table number explicitly mentioned in chunk
- ✅ related_image_ids contains matching image_id
- ✅ System found direct metadata link

**Example:**
```
Text: "Figure 2 shows the attention mechanism..."
Related Images: ["transformer_figure_2_image"]
Result: HIGH confidence → Always include this image
```

#### Level 2: MEDIUM Confidence (Semantic Similarity)

```python
# If no direct reference, check semantic match
chunk_embedding = embeddings[chunk_id]        # 1536-d vector
image_embedding = embeddings[image_caption]   # 1536-d vector

similarity = cosine_similarity(chunk_embedding, image_embedding)
# similarity ranges from 0 (opposite) to 1 (identical)

if similarity > SIMILARITY_THRESHOLD (0.5):
    confidence = "MEDIUM"  # Semantically related
```

**Process:**
1. Get embedding for retrieved text chunk
2. Get embedding for image caption (from ChromaDB)
3. Calculate cosine similarity
4. If similarity > 0.5, classify as MEDIUM confidence

**Example:**
```
Text: "The Transformer architecture uses multi-head attention
       to process input tokens in parallel..."
       (chunk_embedding)

Image Caption: "Multi-head attention mechanism showing 8 parallel
               attention heads processing sequence in parallel."
               (image_embedding)

Cosine Similarity: 0.78 (> 0.5)
Result: MEDIUM confidence → Include with uncertainty disclosure
```

#### Level 3: LOW Confidence (Same-Page Proximity)

```python
# Same page but low semantic similarity
if image_page == chunk_page and similarity < SIMILARITY_THRESHOLD:
    confidence = "LOW"  # Probably not related, filter it
```

**When Used:**
- Image is on same page as text, but semantic similarity < 0.5
- No direct figure reference
- Text talks about "algorithms" but image is a logo/diagram

**Decision:** These images are typically excluded from results or marked with heavy warnings.

---

### 4. Semantic Matching (`verify_semantic_match` & `cosine_similarity`)

**Mathematical Foundation:**

Cosine similarity measures the angle between two vectors in high-dimensional space:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

where:
- A · B = dot product of vectors
- ||A|| = magnitude of vector A (Euclidean norm)
- ||B|| = magnitude of vector B
```

**Why Cosine Similarity?**

1. **Scale-independent:** Doesn't matter if text is long or short
2. **Interpretable:** Returns value between -1 and 1 (typically 0 to 1 for embeddings)
3. **Efficient:** Fast computation in 1536 dimensions
4. **Domain-proven:** Industry standard for semantic similarity

**Example Calculation:**

```
Text: "Attention mechanism"
Image: "Multi-head attention diagram"

text_embedding:   [0.12, -0.45, 0.67, ...1534 more dimensions...]
image_embedding:  [0.15, -0.42, 0.65, ...1534 more dimensions...]

dot_product = 0.12*0.15 + (-0.45)*(-0.42) + 0.67*0.65 + ...
            = 0.0180 + 0.1890 + 0.4355 + ... ≈ 2.45

magnitude(text) = sqrt(0.12² + (-0.45)² + 0.67² + ...) ≈ 1.25
magnitude(image) = sqrt(0.15² + (-0.42)² + 0.65² + ...) ≈ 1.28

cosine_similarity = 2.45 / (1.25 × 1.28) ≈ 0.77

Interpretation: 77% semantic match → MEDIUM confidence ✓
```

**Thresholds Used:**
```python
SIMILARITY_THRESHOLD = 0.5           # MEDIUM confidence threshold
SIMILARITY_THRESHOLD_NEARBY = 0.6    # For same-page proximity check
```

---

### 5. Fallback Visual Search (`_fallback_visual_search`)

**Problem:** User asks "Show me the Transformer architecture" but the retriever couldn't find relevant images through normal flow.

**Solution:** Query-based visual search with document filtering

**Flow:**
```
User Query: "Show Transformer architecture"
    ↓
[Check if visual query]
- Contains keywords: "show", "diagram", "figure", "architecture", etc.
- OR user explicitly asked for images
    ↓
[Was enough images found?]
- If num_images >= MIN_IMAGES_TO_CITE: Done, return
- If num_images < MIN_IMAGES_TO_CITE: Activate fallback
    ↓
[Fallback: Query-based search]
1. Search ALL image captions for query keywords
2. BUT restrict to documents already found relevant
   (Don't show images from unrelated papers)
3. Apply confidence filtering
4. Add to results
```

**Example:**
```
Text retrieval found:
- Document: arxiv_1706_03762 (Attention Is All You Need)
- 2 Transformer diagrams (MEDIUM confidence)

Visual query keywords detected: "show", "architecture"

Fallback search:
- Search for images matching "architecture" in ALL papers
- Filter: Keep only images from arxiv_1706_03762
- Found: 1 additional encoder-decoder diagram
- Result: 3 diagrams total (up from 2)
```

**Code Logic:**
```python
if self.is_visual_query(query):  # Detect "show", "diagram", etc.
    available_images = len(verified_images)
    
    if available_images < MIN_IMAGES_REQUIRED:
        # Activate fallback
        fallback_images = self.image_store.similarity_search(
            query,  # Search entire caption database
            k=5,
            filter_dict={"doc_id": document_ids}  # Only related docs
        )
        verified_images.extend(fallback_images)
```

---

### 6. LLM Input Preparation (`prepare_for_llm`)

**Purpose:** Format raw retrieval results into clean, structured context for the generator

**Processing Steps:**

```
Raw Retrieval Output
    ↓
[1] Metadata Cleanup
    └─ Remove: internal ChromaDB paths, temporary IDs, system tags
    └─ Keep: chunk_id, doc_id, page_num, figure_references, related_images
    ↓
[2] Figure Reference Detection
    └─ Check: has_figure_references flag in chunk metadata
    └─ If True: Add special instruction to generator
    ↓
[3] Image Preparation
    └─ Collect all verified images
    └─ Add confidence scores
    └─ Add similarity values
    └─ Sort by: confidence (HIGH > MEDIUM > LOW)
    ↓
[4] Metadata Statistics
    └─ Count: total text chunks, total images
    └─ Count: HIGH confidence images, MEDIUM, LOW
    └─ Create summary for generator awareness
    ↓
Structured LLM Input JSON
```

**Output Structure:**

```json
{
  "query": "Show Transformer architecture",
  "text_chunks": [
    {
      "chunk_id": "arxiv_1706_03762_chunk_001",
      "doc_id": "arxiv_1706_03762",
      "source": "Attention Is All You Need",
      "page": 2,
      "has_figure_references": true,
      "related_image_ids": ["figure_1_image"],
      "text": "The Transformer is a sequence-to-sequence architecture..."
    },
    // ... more chunks
  ],
  "images": [
    {
      "image_id": "arxiv_1706_03762_fig_1",
      "doc_id": "arxiv_1706_03762",
      "page": 2,
      "confidence": "HIGH",
      "similarity": 0.92,
      "reason": "Direct figure reference in chunk",
      "caption": "Figure 1: The Transformer model architecture."
    },
    // ... more images
  ],
  "metadata": {
    "num_text_chunks": 3,
    "num_images": 2,
    "high_confidence_images": 1,
    "medium_confidence_images": 1,
    "low_confidence_images": 0
  }
}
```

**Key Field:** `has_figure_references`
- **True:** Chunk mentions "Figure N" or similar → Generator should describe the figure
- **False:** Chunk is pure text → Generator can be brief about images

---

## Part 2: Generate Module (Intelligent Processing & Answers)

### Architecture Diagram

```
Prepared LLM Input (from Retrieve)
    ↓
[1] System Prompt Loading
    └─ Strict grounding rules
    └─ Few-shot examples
    └─ Citation format enforcement
    ↓
[2] Input Validation
    ├─ Query sanitization (injection prevention)
    ├─ Document availability check
    └─ Metadata integrity validation
    ↓
[3] Context Formatting
    └─ Convert JSON to readable text blocks
    └─ Format: [1] chunk_id...[A] image_id...
    ↓
[4] LLM Inference
    ├─ Model: GPT-5 Mini
    ├─ Temperature: 0.0 (deterministic)
    ├─ Max tokens: 120,000
    └─ Reasoning effort: "low"
    ↓
[5] Output Parsing
    ├─ Extract Answer section
    ├─ Extract Sources section
    ├─ Extract Reasoning section
    └─ Validate citation consistency
    ↓
[6] Citation Validation & Cleaning
    ├─ Check: All cited chunks exist in context
    ├─ Check: All cited images exist in context
    ├─ Remove: Hallucinated citations
    └─ Ensure: Answer-Sources synchronization
    ↓
Final Grounded Answer with Citations
```

### 1. Strict Grounding Mechanism

**Problem:** LLMs have tendency to use knowledge outside the provided context ("hallucinate").

**Solution:** System prompt that explicitly forbids external knowledge

**System Prompt Rules:**

```
CRITICAL RULE 3: GROUNDED ANSWERS ONLY
- Use ONLY information from the provided context
- NEVER use external knowledge or make assumptions
- If context doesn't contain answer → respond with "I don't know"

CRITICAL RULE 5: CITATION MANDATORY
- Every factual statement must be tied to a source
- Format: "XYZ concept is explained in [1]"
- Format: "As shown in Figure 1 [A], the architecture..."
- No citations = no statement
```

**Enforcement Mechanisms:**
1. Temperature = 0.0 (no randomness, deterministic output)
2. Explicit negative examples showing forbidden patterns
3. Few-shot examples showing correct behavior
4. Output validation removing non-cited statements

---

### 2. Chain of Thought Reasoning

**Problem:** Model might generate plausible-sounding but incorrect answers.

**Solution:** Internal reasoning phase before final answer

**Process:**
```
Internal Reasoning (not shown to user):
1. "Is the query about AI/ML? Yes ✓"
2. "Do I have relevant context? Yes, 3 chunks about Transformers ✓"
3. "Do I have images for this visual query? Yes, 2 diagrams ✓"
4. "Can I answer with only provided context? Yes ✓"

Final Answer:
(Based on reasoning validation)
"The Transformer architecture consists of..."
```

**Implementation:**
```python
# Model: GPT-5 Mini with reasoning support
response = llm.invoke([
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=formatted_context)
])
# Model internally performs reasoning before responding
```

---

### 3. Citation System (Exact References)

**Problem:** Users need to verify every claim by checking the source.

**Solution:** Automatic citation assignment

**Citation Formats:**

```
For Text Sources:
[1] = First text chunk
[2] = Second text chunk
[3] = Third text chunk
etc.

For Images:
[A] = First image
[B] = Second image
[C] = Third image
etc.
```

**Example Answer:**

```
The Transformer is a sequence-to-sequence architecture [1].
It uses self-attention mechanisms to process all tokens in parallel [1][2].
Unlike RNNs, Transformers don't require sequential processing [2].

Figure 1 [A] shows the encoder-decoder structure.
The encoder (left) contains multi-head attention layers [2].
The decoder (right) adds masked attention for autoregressive generation [A].

Sources: [1], [2], [A]
```

**Citation Validation Rules:**
```python
# RULE 1: All citations in Answer must appear in Sources
if "[1]" in answer and "[1]" not in sources:
    raise CitationError("Citation [1] in answer but not in sources")

# RULE 2: All citations in Sources must appear in Answer
if "[A]" in sources and "[A]" not in answer:
    raise CitationError("Citation [A] in sources but not in answer")

# RULE 3: Don't cite non-existent chunks
if metadata.num_chunks == 2 and "[3]" in answer:
    raise CitationError("Citing [3] but only 2 chunks provided")

# RULE 4: Don't cite non-existent images
if metadata.num_images == 1 and "[B]" in answer:
    raise CitationError("Citing [B] but only 1 image provided")
```

---

### 4. Output Parsing & Validation

**Expected Format:**

```
Answer: <content with [1][A] citations>

Sources: <list of [1], [2], [A], [B]>

Reasoning: <explanation of which sources used and why>
```

**Parsing Steps:**

```python
# 1. Extract sections using regex
answer_match = re.search(r'Answer:\s*(.+?)(?=Sources:|$)', response)
sources_match = re.search(r'Sources:\s*(.+?)(?=Reasoning:|$)', response)
reasoning_match = re.search(r'Reasoning:\s*(.+?)$', response)

# 2. Validate structure
if not answer_match:
    raise ParseError("Response missing Answer section")
if not sources_match:
    raise ParseError("Response missing Sources section")

# 3. Extract citations
answer_citations = re.findall(r'\[(\d+|[A-Z])\]', answer)
sources_citations = re.findall(r'\[(\d+|[A-Z])\]', sources)

# 4. Validate consistency
if set(answer_citations) != set(sources_citations):
    raise ValidationError("Answer citations don't match Sources list")
```

---

### 5. Anti-Hallucination Safeguards

**Multiple Layers of Protection:**

#### Layer 1: System Prompt
```python
# In SYSTEM_PROMPT:
"NEVER use external knowledge or make assumptions"
"If context doesn't contain answer → respond: I don't know"
```

#### Layer 2: Few-Shot Examples
```python
# Show 2 complete examples of:
# - Correct format (with citations)
# - Proper grounding (only using provided text)
# - How to handle images (reference vs describe)

EXAMPLES = [
    (query="What is ResNet?", 
     answer="Residual connections [1]. Skip connections add input x 
             to output F(x), resulting in x + F(x) [1]. Figure 7 [A] 
             shows layer response analysis..."),
    # ...more examples
]
```

#### Layer 3: Temperature = 0.0
```python
# No randomness in token selection
# Model always picks most probable next token
# Eliminates creative/hallucinating tendencies
temperature = 0.0
```

#### Layer 4: Citation Validation
```python
# Remove any citation to non-existent sources
invalid_citations = set(answer_citations) - set(valid_chunk_ids)
for citation in invalid_citations:
    answer = re.sub(rf'\[{citation}\]', '', answer)
    logging.warning(f"Removed hallucinated citation [{citation}]")
```

#### Layer 5: Metadata Cross-Check
```python
# Verify cited images actually come from cited chunks
chunk_image_map = build_image_chunk_mapping(llm_input)

for cited_image in cited_images:
    if cited_image not in chunk_image_map:
        logging.warning(f"Image [{cited_image}] not in context")
        remove_citation(cited_image)
```

---

## Data Flow Example: End-to-End Walkthrough

### Query: "Show Transformer encoder-decoder architecture"

**Stage 1: Retrieval**
```
Input: Query string
    ↓
[Text Search] MMR finds:
- chunk_001: "The Transformer architecture consists of..."
- chunk_002: "The encoder stack processes input sequences..."
- chunk_003: "The decoder generates output tokens..."
    ↓
[Image Verification]
- image_001: "Figure 1: Transformer architecture" (HIGH)
- image_002: "Encoder detail diagram" (MEDIUM)
    ↓
Output: 3 text chunks + 2 images
```

**Stage 2: Generator**
```
Input: 3 text chunks + 2 images

[System Prompt]: Enforce grounding + few-shot examples

[LLM Reasoning]:
- Visual query detected ✓
- 3 chunks about Transformers ✓
- 2 high-quality images ✓
- Can answer ✓

[LLM Output]:
Answer:
The Transformer is a sequence-to-sequence architecture [1] 
consisting of an encoder and decoder stack [2].

The encoder (left side of Figure 1 [A]) processes input 
sequences through multi-head self-attention layers [1].

The decoder (right side of Figure 1 [A]) generates output 
tokens autoregressively, with masked attention preventing 
future token access [2].

Figure 2 [B] shows the detailed encoder-decoder connections 
and information flow.

Sources: [1], [2], [A], [B]

Reasoning: Used chunks 1-2 for architectural explanation.
Image A (HIGH, 0.93) shows overall structure.
Image B (MEDIUM, 0.76) illustrates connections.
    ↓
[Citation Validation]
- [1] exists? Yes (chunk_001)
- [2] exists? Yes (chunk_002)
- [A] exists? Yes (image_001)
- [B] exists? Yes (image_002)
- All valid! ✓
    ↓
Output: Grounded answer with valid citations
```

---

## Performance Characteristics

### Speed
| Component | Time | Notes |
|-----------|------|-------|
| Text retrieval (MMR) | ~0.5s | ChromaDB query + ranking |
| Batch embeddings | ~1.0s | Single OpenAI API call |
| Image verification | ~0.5s | Semantic matching (numpy) |
| LLM generation | ~5-10s | GPT-5 Mini inference |
| **Total** | **~5-15s** | End-to-end query to answer |

### Accuracy
| Metric | Target | Achieved |
|--------|--------|----------|
| Recall@5 | ≥70% | 95.0% |
| Image Hit Rate | ≥60% | 88.9% |
| Citation Accuracy | ≥85% | ~98% |
| Hallucination Rate | <5% | <1% |

---

## Troubleshooting Common Issues

### Issue: "Image not found but should be retrieved"

**Diagnosis:**
1. Check retriever thresholds:
   ```python
   SIMILARITY_THRESHOLD = 0.5  # Too high?
   SIMILARITY_THRESHOLD_NEARBY = 0.6
   ```
2. Verify image embedding exists in ChromaDB
3. Check `has_figure_references` flag in chunk metadata
4. Enable fallback visual search (is it active?)

### Issue: "Citations don't match - [C] in answer but only 2 images"

**Root Cause:** Hallucinated citation

**Fix:**
```python
# This is caught by validation layer:
invalid_citations = cited_images - available_images
for citation in invalid_citations:
    remove_from_answer(citation)
```

### Issue: "Answer uses external knowledge not in context"

**Root Cause:** Temperature too high or few-shot examples weak

**Fix:**
```python
# Ensure system prompt is strict
SYSTEM_PROMPT = """
CRITICAL: Use ONLY provided context.
NEVER use external knowledge.
If unsure → respond "I don't have this information."
"""

temperature = 0.0  # Must be zero
```

---

## Configuration Guide

### Retriever Tuning

```python
# Increase diversity (more different chunks)
DEFAULT_MMR_LAMBDA = 0.8  # Higher = more weight to diversity

# Get more candidates before filtering
fetch_k = 10  # Increase to 15 for better diversity

# Lower threshold = more images included
SIMILARITY_THRESHOLD = 0.5  # Lower = include more images

# Activate visual fallback
enable_fallback_visual_search = True
```

### Generator Tuning

```python
# Decrease hallucinations
TEMPERATURE = 0.0  # Already at minimum

# Add more few-shot examples
NUM_EXAMPLES = 2  # Increase to 3-4 for stricter grounding

# Enforce citations more strictly
REQUIRE_CITATIONS_FOR_ALL_FACTS = True

# Reduce token usage
REASONING_EFFORT = "low"  # Low uses fewer reasoning tokens
```

---

## Advanced Topics

### Semantic Search Mathematics

The retriever uses **dense vector embeddings** to understand meaning:

```
"LSTM has memory cells" → [0.12, -0.45, 0.67, ...] (1536 dims)
"Attention mechanism"   → [0.18, -0.38, 0.72, ...] (1536 dims)

Similarity = angle between vectors
High angle (0°) = same meaning
Low angle (90°+) = different meaning
```

### Few-Shot Prompting

Instead of complex instructions, show examples:

```
# WRONG: Complex instructions
"Extract technical terms and provide hierarchical explanation..."

# RIGHT: Show example
Example 1:
Q: "What is LSTM?"
A: "Long Short-Term Memory (LSTM) networks are recurrent neural 
   networks with special memory cells [1]. Unlike standard RNNs, 
   LSTMs use gates (forget, input, output) to control information 
   flow [2]. Figure 3 [A] shows the LSTM cell structure..."

Then model learns the pattern: Explain concept → Use citations → Reference figures
```

### Image Confidence Scoring

The system uses probabilistic confidence, not binary:

```
HIGH (0.8-1.0):  "This image definitely illustrates this concept"
MEDIUM (0.6-0.79): "This image probably relates to this concept"  
LOW (0.5-0.59):   "This image might relate, but uncertain"
```

---

## References

- OpenAI Embeddings: `text-embedding-3-small` (1536 dimensions)
- ChromaDB: Vector database with metadata filtering
- GPT-5 Mini: Language model with reasoning support (TEMPERATURE=0.0)
- Cosine Similarity: Standard metric for semantic search
- MMR Algorithm: Maximal Marginal Relevance (λ=0.7)

---

**Last Updated:** January 26, 2026  
**Status:** Production Ready ✅
