# Data Sources Documentation

This document provides detailed information about all data sources for the AI/ML Course Assistant.

---

## 📚 Overview

**Total Target:**
- 75-120 documents
- 150-400 images (diagrams, architectures, formulas)

---

## 1. arXiv Papers (Primary Source)

### Description
Academic papers from arXiv.org, focusing on Machine Learning and AI topics.

### Access Method
- **API:** arXiv API (Python library: `arxiv`)
- **License:** Open access (various, mostly permissive)
- **Cost:** Free

### Categories to Scrape
| Category | Description | Expected Papers |
|----------|-------------|-----------------|
| `cs.LG` | Machine Learning | 20-25 papers |
| `cs.AI` | Artificial Intelligence | 10-15 papers |
| `cs.CV` | Computer Vision | 10-15 papers |
| `cs.NE` | Neural Networks (optional) | 5-10 papers |

### Sample Papers to Include
1. "Attention Is All You Need" (Transformers)
2. "Deep Residual Learning for Image Recognition" (ResNet)
3. "Adam: A Method for Stochastic Optimization"
4. "Batch Normalization: Accelerating Deep Network Training"
5. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
6. Recent survey papers on deep learning

### Content Types
- ✅ Text: Abstracts, methodology, results
- ✅ Images: Architecture diagrams, algorithm flowcharts, training curves, formula images

### Download Script
Location: `ingest/download_arxiv.py`

**Example usage:**
```python
import arxiv

# Search for papers
search = arxiv.Search(
    query="cat:cs.LG AND (neural network OR deep learning)",
    max_results=50,
    sort_by=arxiv.SortCriterion.Relevance
)

for paper in search.results():
    paper.download_pdf(dirpath="./data/raw/papers")
```

---

## 2. Technical Blog Articles (Secondary Source)

### Description
Tutorial articles and explanations from popular ML/AI blogs.

### Target Websites
| Website | Topics | Expected Articles |
|---------|--------|-------------------|
| **Towards Data Science** | Neural networks, tutorials | 10-15 articles |
| **Distill.pub** | Visual explanations | 5-10 articles |
| **Jay Alammar's Blog** | Transformers, attention | 3-5 articles |
| **colah.github.io** | LSTMs, CNNs | 3-5 articles |

### Sample Articles
1. "The Illustrated Transformer" (Jay Alammar)
2. "Understanding LSTM Networks" (Christopher Olah)
3. "A Gentle Introduction to Neural Networks" (TDS)
4. "Visualizing Neural Networks with Activation Atlases" (Distill)

### Access Method
- Web scraping (BeautifulSoup, requests)
- Manual download for sites with restrictive policies
- **License:** Fair use for educational purposes (cite sources)

### Content Types
- ✅ Text: Explanations, step-by-step guides
- ✅ Images: Custom diagrams, animations (as static images), code visualizations

### Download Script
Location: `ingest/scrape_articles.py`

---

## 3. Wikipedia (Tertiary Source)

### Description
Structured articles on ML/AI concepts with illustrations.

### Access Method
- **API:** Wikipedia API (Python library: `wikipedia-api`)
- **License:** CC BY-SA 3.0
- **Cost:** Free

### Sample Articles
| Topic | Why Include | Has Images |
|-------|-------------|------------|
| Neural network | Foundational concept | ✅ Architecture diagrams |
| Convolutional neural network | CNN visualization | ✅ Convolution examples |
| Recurrent neural network | RNN structure | ✅ Unrolled diagrams |
| Transformer (machine learning) | Attention mechanism | ✅ Architecture |
| Backpropagation | Algorithm flowchart | ✅ Computation graph |
| Gradient descent | Optimization visual | ✅ 3D loss surface |
| Overfitting | Concept illustration | ✅ Training curves |

### Download Script
Location: `ingest/download_wikipedia.py`

**Example usage:**
```python
import wikipediaapi

wiki = wikipediaapi.Wikipedia('en')
page = wiki.page('Neural_network')

# Get text
text = page.text

# Get images (requires parsing HTML)
# Use requests + BeautifulSoup to download images
```

---

## 4. Official Documentation (Quaternary Source)

### Description
Official documentation pages from popular ML frameworks.

### Target Frameworks
| Framework | Pages to Include | Focus |
|-----------|------------------|-------|
| **PyTorch** | Tutorials, nn.Module docs | Architecture examples |
| **TensorFlow** | Keras guides | Layer diagrams |
| **Scikit-learn** | Algorithm explanations | Decision boundaries |
| **Hugging Face** | Transformers guide | Model cards |

### Sample Pages
1. PyTorch: "Neural Networks Tutorial"
2. TensorFlow: "Convolutional Neural Networks"
3. Scikit-learn: "Comparing different clustering algorithms"
4. Hugging Face: "BERT model documentation"

### Access Method
- Manual download (most sites allow scraping)
- Some sites provide data dumps
- **License:** Varies (Apache 2.0, MIT, BSD) - check each

### Content Types
- ✅ Text: API docs, conceptual guides
- ✅ Images: Code output visualizations, architecture diagrams

---

## 📊 Data Collection Summary

| Source | Documents | Images | License | Difficulty |
|--------|-----------|--------|---------|------------|
| arXiv | 40-50 | 100-200 | Open access | ⭐⭐ Medium |
| Blogs | 20-30 | 40-100 | Fair use | ⭐⭐⭐ Easy |
| Wikipedia | 15-25 | 30-60 | CC BY-SA | ⭐⭐⭐ Easy |
| Docs | 10-15 | 20-40 | Varies | ⭐ Hard |
| **TOTAL** | **85-120** | **190-400** | - | - |

---

## 🔍 Specific Topics to Cover

To ensure comprehensive coverage of AI/ML concepts:

### Core Concepts (Must Have)
- [ ] Neural Networks (basics)
- [ ] Backpropagation
- [ ] Gradient Descent
- [ ] Overfitting & Regularization
- [ ] Activation Functions

### Architectures (Must Have)
- [ ] Convolutional Neural Networks (CNNs)
- [ ] Recurrent Neural Networks (RNNs)
- [ ] Long Short-Term Memory (LSTMs)
- [ ] Transformers
- [ ] ResNet / Skip Connections

### Advanced Topics (Nice to Have)
- [ ] Attention Mechanisms
- [ ] Batch Normalization
- [ ] Dropout
- [ ] Transfer Learning
- [ ] Generative Adversarial Networks (GANs)
- [ ] Variational Autoencoders (VAEs)

---

## 📥 Download Instructions

### Step 1: arXiv Papers
```bash
cd ingest
python download_arxiv.py --categories cs.LG,cs.AI --max-results 50 --output ../data/raw/papers
```

### Step 2: Blog Articles
```bash
python scrape_articles.py --sources tds,distill,alammar --output ../data/raw/articles
```

### Step 3: Wikipedia
```bash
python download_wikipedia.py --topics neural_network,cnn,rnn,transformer --output ../data/raw/wiki
```

### Step 4: Extract Images
```bash
python extract_images.py --input ../data/raw --output ../data/processed/images
```

---

## 🔒 Legal & Ethical Considerations

### Licensing Compliance
- ✅ **arXiv:** Cite papers, respect author rights
- ✅ **Wikipedia:** Attribute under CC BY-SA 3.0
- ⚠️ **Blogs:** Fair use for non-commercial educational POC (cite sources)
- ⚠️ **Docs:** Check each framework's terms (most allow educational use)

### Best Practices
1. Always cite original sources in UI
2. Do not redistribute raw datasets
3. Use only for educational/research purposes
4. Respect robots.txt for web scraping
5. Rate-limit API calls

### Privacy
- No personal data collected
- No user-generated content in corpus
- Publicly available content only

---

## 📝 Metadata Schema

### Document Metadata (`documents.json`)
```json
{
  "doc_id": "arxiv_1706.03762",
  "title": "Attention Is All You Need",
  "source_type": "arxiv",
  "source_uri": "https://arxiv.org/abs/1706.03762",
  "authors": ["Vaswani et al."],
  "year": 2017,
  "abstract": "...",
  "topics": ["transformer", "attention", "nlp"],
  "has_images": true,
  "image_ids": ["img_001", "img_002"],
  "created_at": "2025-01-15T10:30:00Z"
}
```

### Image Metadata (`images.json`)
```json
{
  "image_id": "img_001",
  "doc_id": "arxiv_1706.03762",
  "filepath": "data/processed/images/arxiv_1706.03762/img_001.png",
  "caption": "Multi-Head Attention architecture",
  "alt_text": "Diagram showing Q, K, V inputs to scaled dot-product attention",
  "page_num": 3,
  "source_uri": "https://arxiv.org/abs/1706.03762",
  "width": 800,
  "height": 600,
  "format": "png"
}
```

---

## 🚀 Next Steps

1. ✅ Review this document
2. ⏳ Implement download scripts (`ingest/`)
3. ⏳ Test with small dataset (5-10 papers)
4. ⏳ Expand to full corpus (75-120 docs)

---

**Status:** Ready for implementation  
**Last Updated:** December 24, 2025
