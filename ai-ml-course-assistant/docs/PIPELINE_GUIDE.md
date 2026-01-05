# Document Processing Pipeline - User Guide

**Version:** 1.0  
**Last Updated:** January 4, 2026

## Overview

The `run_pipeline.py` script is a unified document processing pipeline that orchestrates all stages from raw documents to indexed, searchable content in ChromaDB.

### Processing Stages

1. **Extract** - Extract images and text from PDFs or JSON documents
2. **Caption** - Generate enriched captions for images using Vision-LM (optional)
3. **Chunk** - Split text into semantic chunks with image tracking
4. **Embed** - Generate embeddings for chunks and images (OpenAI text-embedding-3-small)
5. **Index** - Store in ChromaDB for semantic search

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-key-here"  # Linux/Mac
# or
$env:OPENAI_API_KEY="your-key-here"   # Windows PowerShell
```

### Basic Commands

```bash
# Show processing status
python run_pipeline.py status

# Process specific document
python run_pipeline.py process --doc-id arxiv_1706_03762

# Process all documents
python run_pipeline.py process --all

# Process only new documents (not in registry)
python run_pipeline.py process --new-only
```

---

## Command Reference

### `status` - Show Processing Status

Displays current processing status for all documents in the registry.

```bash
python run_pipeline.py status
```

**Output:**
- Total documents in registry
- Completed, in-progress, and failed counts
- Total API costs
- Details of in-progress and failed documents

**Example Output:**
```
📊 Processing Status
======================================================================

Total documents in registry: 4
  ✅ Completed: 3
  🔄 In Progress: 0
  ❌ Failed: 1

💰 Total cost: $0.180

❌ Failed:
  - numpy-tutorial: Unknown document type for doc_id: numpy-tutorial
```

---

### `process` - Process Documents

Process one or more documents through all 5 stages.

#### Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--doc-id <id>` | Process specific document | None |
| `--all` | Process all documents in data/raw/papers/ | None |
| `--new-only` | Process only documents not in registry | None |
| `--force` | Force reprocess (ignore registry) | False |
| `--no-vlm` | Skip Vision-LM captions (use existing or none) | False |

**Note:** Must specify one of: `--doc-id`, `--all`, or `--new-only`.

---

## Common Use Cases

### 1. Process a Single Document (First Time)

Process a new document with Vision-LM captions:

```bash
python run_pipeline.py process --doc-id arxiv_1706_03762
```

**Cost:** ~$0.01-0.05 (varies by document size and image count)
- Embeddings: ~$0.0002/document
- VLM captions: ~$0.015/image

**Time:** 15-60 seconds (depends on image count)

---

### 2. Process Without Vision-LM (Cost Saving)

Skip VLM captions to save costs (uses author captions only):

```bash
python run_pipeline.py process --doc-id arxiv_1706_03762 --no-vlm
```

**Cost:** ~$0.0002-0.0010 (embeddings only)
**Time:** 5-15 seconds

**When to use:**
- ✅ Testing pipeline functionality
- ✅ Documents with good author captions (PDFs with figure captions)
- ✅ Budget constraints
- ❌ Documents without captions (web JSON)
- ❌ Complex diagrams requiring detailed descriptions

---

### 3. Process All Documents

Process entire corpus (skips already completed):

```bash
python run_pipeline.py process --all
```

**Cost Estimate (54 documents):**
- With VLM: $1.50-$3.00 (depending on image count)
- Without VLM (`--no-vlm`): $0.01-0.02

**Time:** 10-30 minutes

**Progress:**
- Pipeline shows progress: `Document [3/54]: arxiv_1706_03762`
- Can interrupt (Ctrl+C) and resume later
- Completed documents won't reprocess

---

### 4. Process Only New Documents

Process documents not in registry:

```bash
python run_pipeline.py process --new-only
```

**Use case:** Added new PDFs to data/raw/papers/

---

### 5. Force Reprocess (Re-index with New Metadata)

Reprocess completed document:

```bash
python run_pipeline.py process --doc-id arxiv_1409_3215 --force
```

**When to use:**
- Updated chunking algorithm (need new metadata)
- Re-generate VLM captions
- Fix errors in processed document
- Update ChromaDB with new metadata fields

**Warning:** Will regenerate all stages, including VLM captions (unless `--no-vlm` specified)

---

### 6. Re-index Without Re-captioning

Re-process with existing VLM captions:

```bash
python run_pipeline.py process --doc-id arxiv_1409_3215 --force --no-vlm
```

**Use case:**
- VLM captions already exist in `data/processed/images_metadata.json`
- Updated chunking logic (new figure references)
- Fix ChromaDB indexing issues
- Save cost by reusing existing captions

**Cost:** Only embeddings (~$0.0002)

---

## Document Types

### Supported Formats

#### 1. PDF Documents (Academic Papers)

**Naming:** `arxiv_<paper_id>.pdf`

**Example:** `arxiv_1706_03762.pdf`

**Location:** `data/raw/papers/`

**Features:**
- Extracts embedded images and vector graphics
- Detects figure captions
- Preserves page numbers
- Links chunks to images via figure references

---

#### 2. JSON Documents (Web Content)

**Naming:** `<source>_<doc_name>.json`

**Examples:**
- `realpython_numpy-tutorial.json`
- `medium_agents-plan-tasks.json`

**Location:** `data/raw/`

**JSON Structure:**
```json
{
  "doc_id": "realpython_numpy-tutorial",
  "text": "Full article text...",
  "images": [
    {
      "url": "https://...",
      "alt_text": "Image caption"
    }
  ]
}
```

**Features:**
- Downloads images from URLs
- Uses alt_text as author captions
- No page numbers (uses image_index)

---

## Understanding the Registry

### Registry File: `data/processed_docs.json`

Tracks processing status and statistics for all documents.

### Document Status

| Status | Meaning |
|--------|---------|
| `pending` | Not started |
| `in_progress` | Currently processing |
| `completed` | All 5 stages done |
| `failed` | Error occurred |

### Stage Status

Each document tracks completion of 5 stages:

```json
{
  "arxiv_1706_03762": {
    "doc_id": "arxiv_1706_03762",
    "status": "completed",
    "stages": {
      "extract": "completed",
      "caption": "completed",
      "chunk": "completed",
      "embed": "completed",
      "index": "completed"
    },
    "stats": {
      "images_count": 8,
      "text_length": 45672,
      "chunks_count": 28,
      "indexed_chunks": 28,
      "indexed_images": 8
    },
    "cost": {
      "captions": 0.120,
      "embeddings": 0.0003,
      "total": 0.120
    }
  }
}
```

---

## Pipeline Behavior

### Incremental Processing

✅ **Skips completed stages by default**

```bash
# If document already completed, this skips:
python run_pipeline.py process --doc-id arxiv_1706_03762
# Output: ⏭️  Skipping - already processed
```

### In-Memory Pipeline Requirement

**Important:** Stages 3-5 must run in the same pipeline execution.

**Why?**
- Stage 3 (Chunk) output kept in memory (not saved to disk)
- Stage 4 (Embed) needs chunks from Stage 3
- Stage 5 (Index) needs embeddings from Stage 4

**Example of incomplete run:**

```bash
# Run interrupted after Stage 2
python run_pipeline.py process --doc-id arxiv_1706_03762
# Ctrl+C pressed

# Resume fails at Stage 3:
python run_pipeline.py process --doc-id arxiv_1706_03762
# ❌ Error: Chunks not available (stage already completed)
#    Use --force to restart full pipeline
```

**Solution:** Use `--force` to restart from Stage 1:

```bash
python run_pipeline.py process --doc-id arxiv_1706_03762 --force
```

---

## Cost Management

### Cost Breakdown by Stage

| Stage | Service | Cost | Notes |
|-------|---------|------|-------|
| 1. Extract | Local | $0.00 | PyMuPDF |
| 2. Caption (VLM) | OpenAI GPT-4.1-mini Vision | ~$0.015/image | Optional |
| 2. Caption (no VLM) | None | $0.00 | Uses author captions |
| 4. Embed | OpenAI text-embedding-3-small | ~$0.0001/chunk | ~500 tokens/chunk |
| 4. Embed | OpenAI text-embedding-3-small | ~$0.0001/image | ~300 tokens/caption |
| 5. Index | Local | $0.00 | ChromaDB |

### Cost Examples

**Single Document:**
- With VLM: $0.03-0.15 (8 images × $0.015 + embeddings)
- Without VLM: $0.0002-0.0010 (embeddings only)

**54 Documents Corpus:**
- With VLM: $1.50-$3.00 (estimated)
- Without VLM: $0.01-0.02 (embeddings only)

### Cost Optimization Strategies

1. **Initial Testing:** Use `--no-vlm` for pipeline testing
2. **Selective VLM:** Process key documents with VLM, others without
3. **Batch Processing:** Process multiple documents in one `--all` run

---

## Output Directories

```
data/
├── raw/                          # Input documents
│   └── papers/                   # PDF papers
│       └── arxiv_*.pdf
├── processed/
│   ├── images/                   # Extracted images
│   │   └── arxiv_1706_03762/
│   │       ├── arxiv_1706_03762_vector_003_01.png
│   │       └── ...
│   ├── images_metadata.json      # Image metadata + captions
│   └── processed_docs.json       # Registry
└── chroma_db/                    # ChromaDB storage
    ├── text_chunks/              # Text collection
    └── image_captions/           # Image collection
```

---

## Troubleshooting

### Common Issues

#### 1. "Chunks not available - use --force"

**Problem:** Pipeline interrupted after Stage 2

**Solution:**
```bash
python run_pipeline.py process --doc-id <doc_id> --force
```

---

#### 2. "Unknown document type for doc_id"

**Problem:** Document naming doesn't match expected pattern

**Valid patterns:**
- `arxiv_<id>.pdf` → PDF
- `realpython_<name>.json` → JSON
- `medium_<name>.json` → JSON

**Solution:** Rename file to match pattern

---

#### 3. ChromaDB Shows Old Metadata

**Problem:** Updated chunking but ChromaDB has old metadata

**Solution:**
```bash
# Delete old records
python delete_doc_from_chromadb.py <doc_id>

# Clean registry
# Edit data/processed_docs.json: remove "index" stage

# Re-index
python run_pipeline.py process --doc-id <doc_id> --force --no-vlm
```

---

#### 4. Rate Limit Errors (429)

**Problem:** OpenAI API rate limits exceeded

**Current limits:**
- Embeddings: 20 req/min (VLM has 3-second delay built-in)
- Embeddings API: Higher tier (500 req/min for tier 2+)

**Solution:**
- Wait 1 minute and resume
- Pipeline will continue from last completed stage

---

## Advanced Usage

### Viewing Detailed Logs

Pipeline outputs detailed logs to console. Capture for analysis:

```bash
# Windows PowerShell
python run_pipeline.py process --all > pipeline_log.txt 2>&1

# Linux/Mac
python run_pipeline.py process --all &> pipeline_log.txt
```

---

### Checking ChromaDB Contents

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

# Load collections
text_store = Chroma(
    collection_name="text_chunks",
    embedding_function=embeddings,
    persist_directory="data/chroma_db/text_chunks"
)

# Get counts
collection = text_store._collection
print(f"Total chunks: {collection.count()}")

# Get all doc_ids
all_data = collection.get()
doc_ids = set(meta['doc_id'] for meta in all_data['metadatas'])
print(f"Documents: {doc_ids}")
```

---

### Deleting Document from ChromaDB

```bash
python delete_doc_from_chromadb.py <doc_id>
```

**Use cases:**
- Remove test documents
- Clean up before re-indexing with new metadata
- Fix corrupted entries

---

## Performance Benchmarks

### Processing Times (Single Document)

| Stage | Time | Bottleneck |
|-------|------|------------|
| 1. Extract | 1-2s | CPU (PyMuPDF) |
| 2. Caption (VLM) | 3s/image | API calls + rate limit delay |
| 2. Caption (no VLM) | <1s | None |
| 3. Chunk | 1-2s | CPU |
| 4. Embed | 2-5s | API calls |
| 5. Index | 1-2s | Disk I/O |

**Total:**
- With VLM (8 images): ~35-45 seconds
- Without VLM: ~5-10 seconds

### Batch Processing (54 Documents)

**With VLM:**
- Time: 10-30 minutes
- Cost: $1.50-$3.00

**Without VLM:**
- Time: 4-15 minutes
- Cost: $0.01-0.02

---

## Best Practices

### 1. Test Before Batch Processing

```bash
# Test on 1-2 documents first
python run_pipeline.py process --doc-id arxiv_test --no-vlm
python run_pipeline.py status
```

### 2. Use --no-vlm for Development

```bash
# Fast iteration during development
python run_pipeline.py process --all --no-vlm
```

### 3. Selective VLM for Production

```python
# Process key documents with VLM
important_docs = ['arxiv_1706_03762', 'arxiv_1409_3215']
for doc in important_docs:
    # With VLM
    !python run_pipeline.py process --doc-id {doc}

# Process remaining with --no-vlm
!python run_pipeline.py process --new-only --no-vlm
```

### 4. Monitor Costs

```bash
# Check after each batch
python run_pipeline.py status
# Look at: "💰 Total cost: $X.XXX"
```

### 5. Resume After Interruption

```bash
# Pipeline stopped? Just run again:
python run_pipeline.py process --all
# Automatically resumes from incomplete documents
```

---

## FAQ

**Q: Can I process documents in parallel?**  
A: No, pipeline is sequential. ChromaDB concurrent writes not supported yet.

**Q: What happens if API call fails?**  
A: Document marked as "failed" in registry. Check status and re-run with `--force`.

**Q: Can I change embedding model?**  
A: Yes, edit `EMBEDDING_MODEL` and `EMBEDDING_DIMS` in run_pipeline.py. Requires reprocessing all documents.

**Q: How to update chunking algorithm?**  
A: Edit `index/chunk_documents.py`, then reprocess: `python run_pipeline.py process --all --force --no-vlm`

**Q: Can I process non-English documents?**  
A: Yes, text-embedding-3-small supports 100+ languages. VLM also multilingual.

---

## Next Steps

After processing documents:

1. **Test Retrieval:**
   ```bash
   python eval/test_retrieval_indexed.py
   ```

2. **Full System Evaluation:**
   ```bash
   python eval/evaluate_retrieval.py
   python eval/evaluate_answers.py
   ```

3. **Deploy RAG System:**
   - Connect retriever.py to LLM
   - Implement generator.py
   - Build web interface

---

## Support

**Issues:**
- Check logs: `python run_pipeline.py status`
- View registry: `data/processed_docs.json`
- Test single document: `--doc-id <id> --force`

**Contact:** See project README.md

---

**Last Updated:** January 4, 2026  
**Version:** 1.0
