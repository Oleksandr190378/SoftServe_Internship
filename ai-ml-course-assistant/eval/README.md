# Evaluation Directory

This directory contains evaluation scripts and results for Phase E: Production Validation (Completed Jan 19, 2026).

## Structure

```
eval/
├── README.md                          # This file
├── test_queries.json                  # 30 test queries (10 text, 10 visual, 10 hybrid)
├── ground_truth.json                  # Manual ground truth labels with relevance scores
├── evaluate_retrieval.py              # Full retrieval evaluation (Recall@5, MRR, Image Hit Rate)
├── faithfulness_judge.py              # LLM-based faithfulness evaluator
├── validate_ground_truth.py           # Validates ground truth format and coverage
└── results/                           # Evaluation outputs (.gitignored)
    ├── retrieval_eval_YYYYMMDD_HHMMSS.json    # Retrieval metrics
    ├── faithfulness_eval_YYYYMMDD_HHMMSS.json # Faithfulness scores
    └── ...
```

## Current Status (Phase E - Complete)

### ✅ Completed
- `test_queries.json` - 30 test queries (10 text, 10 visual, 10 hybrid)
- `ground_truth.json` - Manual relevance labels for all 30 queries
- `evaluate_retrieval.py` - Full evaluation pipeline for retrieval metrics
- `faithfulness_judge.py` - LLM-based faithfulness validation
- `validate_ground_truth.py` - Ground truth format validation
- All evaluations run and results stored in `results/`

### 📊 Final Metrics (Jan 19, 2026)
- **Recall@5**: 95% (target: ≥70%) ✅
- **Image Hit Rate**: 88.9% (target: ≥60%) ✅
- **Faithfulness**: 4.525/5.0 (target: ≥80%) ✅
- **MRR**: 1.0 (perfect ranking) ✅

## Usage

### Run Full Evaluation

Evaluates all 30 test queries against the indexed document collection (54 documents, 369 chunks):

```bash
python eval/evaluate_retrieval.py
```

**Outputs:**
- `results/retrieval_eval_<timestamp>.json` - Retrieval metrics (Recall@5, MRR, Image Hit Rate)
- Console: Detailed per-query results and aggregated statistics

### Run Faithfulness Evaluation

Validates answer faithfulness using LLM-based judgment:

```bash
python eval/faithfulness_judge.py
```

**Outputs:**
- `results/faithfulness_eval_<timestamp>.json` - Faithfulness scores (0-5 scale)
- Console: Per-query faithfulness analysis

### Validate Ground Truth

Checks that ground truth file has correct format and covers all queries:

```bash
python eval/validate_ground_truth.py
```

## Metrics

### Retrieval Metrics (D2)
- **Recall@5**: % of relevant chunks in top-5 (target ≥70%)
- **Image Hit Rate**: % of visual queries with ≥1 relevant image (target ≥60%)
- **MRR (Mean Reciprocal Rank)**: Average 1/rank of first relevant result

### Answer Quality Metrics
- **Faithfulness**: Answer support level (0-5 scale, actual: 4.525/5.0)
- **Citation Accuracy**: % of citations matching retrieved content
- **Context Utilization**: Quality of document context in answers

### Latency Metrics
- Text retrieval time (semantic search)
- Image retrieval time (metadata + verification)
- Total retrieval time
- Generation time (reasoning + answer)
- End-to-end latency

## Test Queries

### Text-focused (10)
Questions about concepts, definitions, explanations without explicit visual requests.

### Visual (10)
Queries explicitly requesting diagrams, figures, architectures, visualizations.

### Hybrid (10)
Queries combining explanations with visual requests ("Explain X and show Y").

See `test_queries.json` for full list.
