# Evaluation Directory

This directory contains evaluation scripts and results for Phase D: System Evaluation.

## Structure

```
eval/
├── README.md                          # This file
├── test_queries.json                  # 30 test queries (10 text, 10 visual, 10 hybrid)
├── ground_truth.json                  # Manual ground truth labels (to be created)
├── test_retrieval_indexed.py          # Test retriever on 3 indexed docs
├── evaluate_retrieval.py              # Full evaluation on 30 queries (to be created)
├── evaluate_answers.py                # Answer quality evaluation (to be created)
└── results/                           # Test outputs
    ├── retrieval_test_YYYYMMDD_HHMMSS.txt      # Detailed test logs
    ├── retrieval_summary_YYYYMMDD_HHMMSS.json  # Metrics summary
    └── ...
```

## Current Status

### ✅ Completed
- `test_queries.json` - 30 test queries created (ROADMAP Phase D1)
- `test_retrieval_indexed.py` - Retrieval test on 3 indexed documents
- `results/` - Output directory for logs and metrics

### ⏳ TODO
- [ ] Create `ground_truth.json` with manual labels for 30 queries
- [ ] Create `evaluate_retrieval.py` for full retrieval metrics
- [ ] Create `evaluate_answers.py` for answer quality metrics
- [ ] Add latency profiling

## Usage

### Test Retrieval on Indexed Documents

Tests retriever performance on 3 already-indexed documents:
- arxiv_1409_3215 - Seq2Seq Learning
- medium_agents-plan-tasks - AI Agents
- realpython_numpy-tutorial - NumPy Tutorial

```bash
python eval/test_retrieval_indexed.py
```

**Outputs:**
- Console: Detailed results for each query
- `results/retrieval_test_<timestamp>.txt` - Full log
- `results/retrieval_summary_<timestamp>.json` - Metrics summary

### Full Evaluation (30 Queries)

*To be implemented after all 54 documents are indexed*

```bash
python eval/evaluate_retrieval.py
```

## Metrics

### Retrieval Metrics (D2)
- **Recall@5**: % of relevant chunks in top-5 (target ≥70%)
- **Image Hit Rate**: % of visual queries with ≥1 relevant image (target ≥60%)
- **MRR (Mean Reciprocal Rank)**: Average 1/rank of first relevant result

### Answer Quality Metrics (D3)
- **Faithfulness**: % of answers supported by sources (target ≥80%)
- **Citation Accuracy**: % of citations actually relevant (target ≥85%)
- **"I don't know" Correctness**: System refuses when context insufficient (target 100%)

### Latency Metrics (D4)
- Text retrieval time (semantic search)
- Image retrieval time (metadata + verification)
- Total retrieval time
- Generation time (reasoning + answer)
- End-to-end latency (target <60s for medium reasoning)

## Test Queries

### Text-focused (10)
Questions about concepts, definitions, explanations without explicit visual requests.

### Visual (10)
Queries explicitly requesting diagrams, figures, architectures, visualizations.

### Hybrid (10)
Queries combining explanations with visual requests ("Explain X and show Y").

See `test_queries.json` for full list.
