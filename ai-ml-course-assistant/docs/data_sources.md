# Data Sources Documentation

This document describes the three data sources used in the AI/ML Course Assistant.

---

## 📚 Overview

The system ingests content from three complementary sources to provide comprehensive AI/ML knowledge:

- **arXiv**: Academic papers with research depth and mathematical rigor
- **RealPython**: Practical tutorials with working code examples  
- **Medium/TowardsDataScience**: Industry insights and best practices

**Current Dataset:** 54 documents, 369 text chunks, 142 images

---

## 1. arXiv Papers (Academic Source)

### Description
Research papers from arXiv.org focusing on Machine Learning, AI, and Computer Vision.

### Categories
- `cs.LG` - Machine Learning
- `cs.AI` - Artificial Intelligence  
- `cs.CV` - Computer Vision

### Content Characteristics
- ✅ **Text:** Abstract, methodology, experimental results, mathematical formulations
- ✅ **Images:** Architecture diagrams, training curves, algorithm flowcharts, mathematical formulas
- ✅ **License:** Open access (various permissive licenses)

### Download Script
**Location:** `ingest/download_arxiv.py`

**Usage:**
```bash
python download_arxiv.py --num-papers 10 --categories cs.LG,cs.AI
```

**Key Features:**
- Downloads PDFs directly from arXiv API
- Extracts metadata (title, authors, abstract, categories)
- Saves to `data/raw/arxiv/`

### Example Papers
- "Attention Is All You Need" (Transformers)
- "Deep Residual Learning for Image Recognition" (ResNet)
- "Adam: A Method for Stochastic Optimization"

---

## 2. RealPython Tutorials (Practical Source)

### Description
Practical ML/DL tutorials with code examples, visualizations, and step-by-step guides. Focuses on Python implementation details and best practices.

### Content Characteristics
- ✅ **Text:** Code examples, implementation guides, debugging tips, performance optimization
- ✅ **Images:** Code screenshots, output visualizations, workflow diagrams
- ✅ **License:** Fair use for educational purposes

### Download Script
**Location:** `ingest/download_realpython.py`

**Usage:**
```bash
python download_realpython.py --num-articles 10
```

**Key Features:**
- Scrapes article HTML from realpython.com
- Extracts code blocks and inline images
- Preserves code formatting and syntax highlighting
- Saves to `data/raw/realpython/`

### Example Tutorials
- "Python AI: How to Build a Neural Network"
- "PyTorch Tutorial: Getting Started with Deep Learning"
- "Building a Neural Network with Keras"

---

## 3. Medium Articles (Industry Source)

### Description
Practical ML/DL articles from Medium and TowardsDataScience focusing on real-world use cases, industry insights, common pitfalls, and project walkthroughs.

### Content Characteristics
- ✅ **Text:** Project case studies, concept explanations, best practices, troubleshooting guides
- ✅ **Images:** Custom diagrams, infographics, result comparisons, architecture illustrations
- ✅ **License:** Fair use for educational purposes

### Download Script
**Location:** `ingest/download_medium.py`

**Usage:**
```bash
python download_medium.py --num-articles 7
```

**Key Features:**
- Scrapes article content via Medium API or direct HTML parsing
- Handles paywalled articles (when accessible)
- Extracts embedded images and code snippets
- Saves to `data/raw/medium/`

### Example Articles
- "Understanding Transformers in NLP"
- "How Agents Plan Complex Tasks"
- "The Math Behind Neural Networks"

---

## 📊 Dataset Summary

| Source | Documents | Key Strength | License |
|--------|-----------|--------------|---------|
| **arXiv** | 35 | Research depth, mathematical rigor | Open access |
| **RealPython** | 9 | Code examples, implementation details | Fair use |
| **Medium/TDS** | 10 | Industry insights, real-world cases | Fair use |
| **TOTAL** | **54** | Comprehensive coverage | - |

### Content Statistics
- **Text Chunks:** 369 (avg 500 tokens)
- **Images:** 142 (with VLM captions)
- **Topics Covered:** Neural networks, CNNs, RNNs, Transformers, optimization, regularization

---

## 🚀 Running the Pipeline

### Download All Sources
```bash
cd ingest

# Download arXiv papers
python download_arxiv.py --num-papers 35 --categories cs.LG,cs.AI,cs.CV

# Download RealPython tutorials  
python download_realpython.py --num-articles 9

# Download Medium articles
python download_medium.py --num-articles 10
```

### Process Documents
```bash
# Run full ingestion pipeline
cd ..
python run_pipeline.py
```

This will:
1. Extract text and images from downloaded content
2. Generate enriched captions for images using VLM
3. Chunk documents for optimal retrieval
4. Create embeddings and build vector index

---

## 🔒 Legal & Ethical Considerations

### Licensing
- **arXiv:** Open access, cite papers appropriately
- **RealPython/Medium:** Fair use for non-commercial educational purposes, always cite sources

### Best Practices
1. Cite original sources in UI responses
2. Use only for educational/research purposes  
3. Respect rate limits and robots.txt
4. Do not redistribute raw content

---

## 📝 Example Data Structure

### Downloaded Content
```
data/raw/
├── arxiv/
│   ├── 1706.03762.pdf           # "Attention Is All You Need"
│   └── 1512.03385.pdf           # "Deep Residual Learning"
├── realpython/
│   ├── python-ai-neural-network/
│   │   ├── content.json
│   │   └── images/
│   └── pytorch-tutorial/
│       └── content.json
└── medium/
    ├── agents-plan-tasks/
    │   ├── content.json
    │   └── images/
    └── understanding-transformers/
        └── content.json
```

### Processed Output
```
data/processed/
├── arxiv_1706_03762_chunks.json      # Text chunks
├── arxiv_1706_03762_images.json      # Image metadata + captions
└── processed_docs.json                # Document index
```

---

## 🎯 Topic Coverage

The combined dataset covers:

### Core Concepts
- Neural networks fundamentals
- Backpropagation & gradient descent
- Loss functions & optimization
- Regularization (dropout, L1/L2)

### Architectures  
- Feedforward networks
- CNNs (Convolutional Neural Networks)
- RNNs & LSTMs
- Transformers & Attention

### Advanced Topics
- Transfer learning
- Batch normalization
- Residual connections
- Multi-head attention

---

**Last Updated:** January 28, 2026  
**Status:** ✅ All sources integrated and production-ready

