"""
Retrieval Quality Evaluation Script.

Evaluates MultimodalRetriever performance against ground truth annotations.
Computes standard retrieval metrics:
- Recall@k (k=3,5,10)
- Precision@k
- MRR (Mean Reciprocal Rank)
- Image Recall

Usage:
    python eval/evaluate_retrieval.py

Outputs:
    - Console: Summary metrics
    - eval/results/retrieval_eval_<timestamp>.json: Detailed results
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, asdict
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retrieve import MultimodalRetriever

# Configuration constants
DEFAULT_CHROMA_DIR = "data/chroma_db"
DEFAULT_GROUND_TRUTH_PATH = "eval/ground_truth.json"
DEFAULT_OUTPUT_DIR = "eval/results"
DEFAULT_K_VALUES = [3, 5, 10]

# Evaluation targets (from PRD spec)
TARGET_RECALL_AT_5 = 0.70
TARGET_IMAGE_RECALL = 0.60  
TARGET_MRR = 0.5  

# Result storage limits
MAX_STORED_DOC_IDS = 10
MAX_STORED_IMAGE_IDS = 5


# Dataclasses for type-safe structured data
@dataclass
class MetricStats:
    """Statistics for a single metric (avg, min, max)."""
    average: float
    min: float
    max: float


@dataclass
class MetricsByK:
    """Recall and Precision metrics at specific k."""
    k: int
    avg_recall: float
    avg_precision: float
    min_recall: float
    max_recall: float
    min_precision: float
    max_precision: float


@dataclass
class ImageMetrics:
    """Image retrieval metrics."""
    visual_queries: int
    avg_recall: float
    min_recall: float
    max_recall: float


@dataclass
class QueryTypeMetrics:
    """Metrics aggregated by query type."""
    count: int
    avg_recall_at_5: float
    avg_precision_at_5: float
    avg_mrr: float


@dataclass
class QueryMetrics:
    """Metrics for a single query evaluation."""
    query_id: int
    query: str
    query_type: str
    category: str
    retrieved_chunks: int
    retrieved_images: int
    expected_docs: int
    expected_images: int
    # Dynamic recall@k and precision@k added as dict
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    mrr: float
    image_recall: float
    retrieved_doc_ids: List[str]
    retrieved_image_ids: List[str]
    relevant_doc_ids: List[str]
    relevant_image_ids: List[str]
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization with flattened recall@k."""
        result = {
            'query_id': self.query_id,
            'query': self.query,
            'query_type': self.query_type,
            'category': self.category,
            'retrieved_chunks': self.retrieved_chunks,
            'retrieved_images': self.retrieved_images,
            'expected_docs': self.expected_docs,
            'expected_images': self.expected_images,
            'mrr': self.mrr,
            'image_recall': self.image_recall,
            'retrieved_doc_ids': self.retrieved_doc_ids,
            'retrieved_image_ids': self.retrieved_image_ids,
            'relevant_doc_ids': self.relevant_doc_ids,
            'relevant_image_ids': self.relevant_image_ids
        }
        # Flatten recall@k and precision@k
        for k, recall in self.recall_at_k.items():
            result[f'recall@{k}'] = recall
        for k, precision in self.precision_at_k.items():
            result[f'precision@{k}'] = precision
        return result


@dataclass
class EvaluationSummary:
    """Summary statistics for entire evaluation."""
    total_queries: int
    by_k: Dict[int, MetricsByK]
    mrr: MetricStats
    images: Optional[ImageMetrics]
    by_query_type: Dict[str, QueryTypeMetrics]
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            'total_queries': self.total_queries,
            'by_k': {k: asdict(v) for k, v in self.by_k.items()},
            'mrr': asdict(self.mrr),
            'images': asdict(self.images) if self.images else None,
            'by_query_type': {k: asdict(v) for k, v in self.by_query_type.items()}
        }


class RetrievalEvaluator:
    """Evaluate retrieval quality against ground truth."""
    
    def __init__(
        self, 
        ground_truth_path: str = DEFAULT_GROUND_TRUTH_PATH,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        retriever: MultimodalRetriever = None
    ):
        """Initialize evaluator with configurable paths and optional retriever injection."""
        self.ground_truth = self.load_ground_truth(ground_truth_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or inject retriever
        if retriever is None:
            print("Initializing retriever...")
            self.retriever = MultimodalRetriever(chroma_dir=Path(chroma_dir))
        else:
            self.retriever = retriever
        
        self.results = []
        
    def load_ground_truth(self, path: str) -> dict:
        """Load ground truth annotations with error handling."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Validate required keys
            if 'queries' not in data:
                raise ValueError(f"Ground truth file missing 'queries' key: {path}")
            
            return data
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Ground truth file not found: {path}\n"
                f"Expected location: {Path(path).resolve()}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in ground truth file {path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load ground truth from {path}: {e}")
    
    def evaluate_query(self, query_data: dict, k_values: List[int] = [3, 5, 10]) -> QueryMetrics:
        """
        Orchestrate single query evaluation .
        
        Args:
            query_data: Ground truth query data
            k_values: List of k values for Recall@k and Precision@k
            
        Returns:
            QueryMetrics dataclass with all metrics
        """
        # Extract query data
        query = query_data['query']
        query_id = query_data['id']
        query_type = query_data['query_type']
        relevant_doc_ids = set(query_data['relevant_docs'])
        relevant_image_ids = set(query_data['relevant_images'])
        
        # 1. Perform retrieval (SRP)
        text_chunks, verified_images = self._perform_retrieval(query, max(k_values))
        
        # 2. Extract IDs (SRP)
        retrieved_doc_ids, retrieved_image_ids = self._extract_retrieved_ids(text_chunks, verified_images)
        
        # 3. Compute metrics (SRP)
        text_metrics = self._compute_text_metrics(retrieved_doc_ids, relevant_doc_ids, k_values)
        image_recall = self._compute_image_metrics(retrieved_image_ids, relevant_image_ids)
        
        # 4. Build QueryMetrics dataclass
        metrics = QueryMetrics(
            query_id=query_id,
            query=query,
            query_type=query_type,
            category=query_data['category'],
            retrieved_chunks=len(text_chunks),
            retrieved_images=len(verified_images),
            expected_docs=len(relevant_doc_ids),
            expected_images=len(relevant_image_ids),
            recall_at_k={k: text_metrics[f'recall@{k}'] for k in k_values},
            precision_at_k={k: text_metrics[f'precision@{k}'] for k in k_values},
            mrr=text_metrics['mrr'],
            image_recall=image_recall,
            retrieved_doc_ids=retrieved_doc_ids[:MAX_STORED_DOC_IDS],
            retrieved_image_ids=retrieved_image_ids[:MAX_STORED_IMAGE_IDS],
            relevant_doc_ids=list(relevant_doc_ids),
            relevant_image_ids=list(relevant_image_ids)
        )
        
        # 5. Log results (SRP)
        self._log_query_results(
            query_id, query, query_type, query_data['category'],
            text_metrics, image_recall, k_values,
            len(text_chunks), len(verified_images),
            len(relevant_doc_ids), len(relevant_image_ids)
        )
        
        return metrics
    
    def _aggregate_single_metric(self, values: List[float]) -> MetricStats:
        """DRY helper: compute avg/min/max for a metric."""
        if not values:
            return MetricStats(average=0.0, min=0.0, max=0.0)
        return MetricStats(
            average=sum(values) / len(values),
            min=min(values),
            max=max(values)
        )
    
    def _perform_retrieval(self, query: str, k_text: int) -> Tuple[List, List]:
        """SRP: Handle retrieval operation only."""
        return self.retriever.retrieve_with_verification(
            query=query,
            k_text=k_text
        )
    
    def _extract_retrieved_ids(self, text_chunks: List, verified_images: List) -> Tuple[List[str], List[str]]:
        """SRP: Extract IDs from retrieval results."""
        retrieved_doc_ids = [chunk.metadata.get('doc_id') for chunk in text_chunks]
        retrieved_image_ids = [img['image'].metadata.get('image_id') for img in verified_images]
        return retrieved_doc_ids, retrieved_image_ids
    
    def _compute_text_metrics(self, retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k_values: List[int]) -> dict:
        """SRP: Compute text retrieval metrics (Recall@k, Precision@k, MRR)."""
        metrics = {}
        
        # Metrics at different k values
        for k in k_values:
            recall_k = self.calc_recall(retrieved_doc_ids[:k], relevant_doc_ids)
            precision_k = self.calc_precision(retrieved_doc_ids[:k], relevant_doc_ids)
            metrics[f'recall@{k}'] = recall_k
            metrics[f'precision@{k}'] = precision_k
        
        # MRR
        metrics['mrr'] = self.calc_mrr(retrieved_doc_ids, relevant_doc_ids)
        
        return metrics
    
    def _compute_image_metrics(self, retrieved_image_ids: List[str], relevant_image_ids: Set[str]) -> float:
        """SRP: Compute image retrieval metrics."""
        if relevant_image_ids:
            return len(set(retrieved_image_ids) & relevant_image_ids) / len(relevant_image_ids)
        return 0.0
    
    def _log_query_results(self, query_id: int, query: str, query_type: str, category: str,
                           metrics: dict, image_recall: float, k_values: List[int], 
                           num_chunks: int, num_images: int,
                           num_expected_docs: int, num_expected_images: int) -> None:
        """SRP: Handle console logging only."""
        print(f"\n{'='*80}")
        print(f"Query {query_id}: {query}")
        print(f"Type: {query_type} | Category: {category}")
        print('-'*80)
        print(f"Retrieved: {num_chunks} chunks, {num_images} images")
        print(f"Expected: {num_expected_docs} docs, {num_expected_images} images")
        
        for k in k_values:
            print(f"  @k={k}: Recall={metrics[f'recall@{k}']:.2f}, Precision={metrics[f'precision@{k}']:.2f}")
        
        print(f"  MRR: {metrics['mrr']:.3f}")
        
        if num_expected_images > 0:
            print(f"  Image Recall: {image_recall:.2f}")
        else:
            print(f"  No images expected")
    
    def calc_recall(self, retrieved: List[str], relevant: Set[str]) -> float:
        """
        Recall@k = |retrieved ∩ relevant| / |relevant|
        
        Measures: What fraction of relevant docs were retrieved?
        """
        if not relevant:
            return 0.0  # No relevant docs defined - indicates data quality issue
        
        retrieved_set = set(retrieved)
        intersection = retrieved_set & relevant
        return len(intersection) / len(relevant)
    
    def calc_precision(self, retrieved: List[str], relevant: Set[str]) -> float:
        """
        Precision@k = |retrieved ∩ relevant| / k
        
        Measures: What fraction of retrieved docs are relevant?
        """
        if not retrieved:
            return 0.0
        
        retrieved_set = set(retrieved)
        intersection = retrieved_set & relevant
        return len(intersection) / len(retrieved)
    
    def calc_mrr(self, retrieved: List[str], relevant: Set[str]) -> float:
        """
        MRR (Mean Reciprocal Rank) = 1 / rank_of_first_relevant
        
        Measures: How quickly do we find the first relevant doc?
        """
        if not relevant:
            return 0.0  # No relevant docs defined - indicates data quality issue
        
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        
        return 0.0  # No relevant doc found
    
    def evaluate_all(self, k_values: List[int] = None) -> EvaluationSummary:
        """
        Run evaluation on all queries in ground truth.
        
        Args:
            k_values: List of k values for metrics (default: [3, 5, 10])
            
        Returns:
            EvaluationSummary dataclass with aggregated statistics
        """
        if k_values is None:
            k_values = DEFAULT_K_VALUES
            
        print("\n" + "="*80)
        print("RETRIEVAL EVALUATION")
        print("="*80)
        print(f"Total queries: {len(self.ground_truth['queries'])}")
        print(f"Evaluating at k={k_values}")
        
        all_results = []
        
        for query_data in self.ground_truth['queries']:
            metrics = self.evaluate_query(query_data, k_values)
            all_results.append(metrics)
        
        # Aggregate metrics
        summary = self.aggregate_metrics(all_results, k_values)
        
        # Save results with exception handling
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"retrieval_eval_{timestamp}.json"
        
        full_results = {
            'timestamp': timestamp,
            'summary': summary.to_dict(),
            'per_query': [r.to_dict() for r in all_results],
            'ground_truth_stats': self.ground_truth.get('query_distribution', {})
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Results saved to: {output_file}")
        except IOError as e:
            print(f"\n⚠️  Warning: Could not save results to {output_file}: {e}")
            print("Evaluation completed but results not persisted.")
        
        return summary
    
    def aggregate_metrics(self, results: List[QueryMetrics], k_values: List[int]) -> EvaluationSummary:
        """Aggregate metrics across all queries (returns type-safe EvaluationSummary)."""
        # Convert to dicts for backward compatibility with aggregation logic
        result_dicts = [r.to_dict() for r in results]
        
        # Aggregate by k
        by_k = {}
        for k in k_values:
            recall_stats = self._aggregate_single_metric([r[f'recall@{k}'] for r in result_dicts])
            precision_stats = self._aggregate_single_metric([r[f'precision@{k}'] for r in result_dicts])
            
            by_k[k] = MetricsByK(
                k=k,
                avg_recall=recall_stats.average,
                avg_precision=precision_stats.average,
                min_recall=recall_stats.min,
                max_recall=recall_stats.max,
                min_precision=precision_stats.min,
                max_precision=precision_stats.max
            )
        
        # MRR
        mrr = self._aggregate_single_metric([r['mrr'] for r in result_dicts])
        
        # Image metrics
        visual_queries = [r for r in result_dicts if r['expected_images'] > 0]
        if visual_queries:
            image_stats = self._aggregate_single_metric([r['image_recall'] for r in visual_queries])
            images = ImageMetrics(
                visual_queries=len(visual_queries),
                avg_recall=image_stats.average,
                min_recall=image_stats.min,
                max_recall=image_stats.max
            )
        else:
            images = None
        
        # By query type
        by_query_type = {}
        for query_type in ['text_focused', 'visual', 'hybrid']:
            type_results = [r for r in result_dicts if r['query_type'] == query_type]
            if type_results:
                by_query_type[query_type] = QueryTypeMetrics(
                    count=len(type_results),
                    avg_recall_at_5=self._aggregate_single_metric([r['recall@5'] for r in type_results]).average,
                    avg_precision_at_5=self._aggregate_single_metric([r['precision@5'] for r in type_results]).average,
                    avg_mrr=self._aggregate_single_metric([r['mrr'] for r in type_results]).average
                )
        
        return EvaluationSummary(
            total_queries=len(results),
            by_k=by_k,
            mrr=mrr,
            images=images,
            by_query_type=by_query_type
        )
    
    def print_summary(self, summary: EvaluationSummary):
        """Print summary in human-readable format (now type-safe with dataclass)."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\n📊 Overall Performance:")
        print(f"  Total queries: {summary.total_queries}")
        
        # Metrics by k
        print(f"\n📈 Retrieval Metrics:")
        for k, metrics in summary.by_k.items():
            print(f"\n  @k={k}:")
            print(f"    Recall:    {metrics.avg_recall:.3f} (min: {metrics.min_recall:.2f}, max: {metrics.max_recall:.2f})")
            print(f"    Precision: {metrics.avg_precision:.3f} (min: {metrics.min_precision:.2f}, max: {metrics.max_precision:.2f})")
        
        # MRR
        print(f"\n  MRR: {summary.mrr.average:.3f} (min: {summary.mrr.min:.2f}, max: {summary.mrr.max:.2f})")
        
        # Image metrics
        if summary.images:
            print(f"\n🖼️  Image Retrieval:")
            print(f"    Visual queries: {summary.images.visual_queries}")
            print(f"    Avg Recall: {summary.images.avg_recall:.3f} (min: {summary.images.min_recall:.2f}, max: {summary.images.max_recall:.2f})")
        
        # By query type
        print(f"\n📋 By Query Type:")
        for qtype, metrics in summary.by_query_type.items():
            print(f"\n  {qtype} ({metrics.count} queries):")
            print(f"    Recall@5: {metrics.avg_recall_at_5:.3f}")
            print(f"    Precision@5: {metrics.avg_precision_at_5:.3f}")
            print(f"    MRR: {metrics.avg_mrr:.3f}")
        
        # Targets comparison
        print(f"\n🎯 Target Comparison:")
        recall_5 = summary.by_k[5].avg_recall
        print(f"    Recall@5: {recall_5:.1%} (target: ≥{TARGET_RECALL_AT_5:.0%}) {'✅' if recall_5 >= TARGET_RECALL_AT_5 else '❌'}")
        
        if summary.images:
            image_recall = summary.images.avg_recall
            print(f"    Image Recall: {image_recall:.1%} (target: ≥{TARGET_IMAGE_RECALL:.0%}) {'✅' if image_recall >= TARGET_IMAGE_RECALL else '❌'}")
        
        mrr = summary.mrr.average
        print(f"    MRR: {mrr:.3f} (target: ≥{TARGET_MRR}) {'✅' if mrr >= TARGET_MRR else '❌'}")
        
        print("\n" + "="*80)


def main():
    """Run retrieval evaluation."""
    try:
        evaluator = RetrievalEvaluator(
            ground_truth_path=DEFAULT_GROUND_TRUTH_PATH,
            output_dir=DEFAULT_OUTPUT_DIR,
            chroma_dir=DEFAULT_CHROMA_DIR
        )
        
        # Run evaluation
        summary = evaluator.evaluate_all(k_values=DEFAULT_K_VALUES)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        return None
    
    # Print summary
    evaluator.print_summary(summary)
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
