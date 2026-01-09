"""
Retrieval Quality Evaluation Script.

Evaluates MultimodalRetriever performance against ground truth annotations.
Computes standard retrieval metrics:
- Recall@k (k=3,5,10)
- Precision@k
- MRR (Mean Reciprocal Rank)
- Image Hit Rate

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
from typing import List, Dict, Set, Tuple
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retriever import MultimodalRetriever


class RetrievalEvaluator:
    """Evaluate retrieval quality against ground truth."""
    
    def __init__(self, ground_truth_path: str, output_dir: str = "eval/results"):
        self.ground_truth = self.load_ground_truth(ground_truth_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize retriever
        print("Initializing retriever...")
        self.retriever = MultimodalRetriever(
            chroma_dir=Path("data/chroma_db")
        )
        
        self.results = []
        
    def load_ground_truth(self, path: str) -> dict:
        """Load ground truth annotations."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def evaluate_query(self, query_data: dict, k_values: List[int] = [3, 5, 10]) -> dict:
        """
        Evaluate single query and compute metrics.
        
        Args:
            query_data: Ground truth query data
            k_values: List of k values for Recall@k and Precision@k
            
        Returns:
            Dictionary with metrics for this query
        """
        query = query_data['query']
        query_id = query_data['id']
        query_type = query_data['query_type']
        
        print(f"\n{'='*80}")
        print(f"Query {query_id}: {query}")
        print(f"Type: {query_type} | Category: {query_data['category']}")
        print('-'*80)
        
        # Retrieve
        text_chunks, verified_images = self.retriever.retrieve_with_verification(
            query=query,
            k_text=max(k_values)  # Retrieve enough for largest k
        )
        
        # Extract retrieved IDs
        retrieved_doc_ids = [chunk.metadata.get('doc_id') for chunk in text_chunks]
        retrieved_image_ids = [img['image'].metadata.get('image_id') for img in verified_images]
        
        # Ground truth IDs
        relevant_doc_ids = set(query_data['relevant_docs'])
        relevant_image_ids = set(query_data['relevant_images'])
        
        print(f"Retrieved: {len(text_chunks)} chunks, {len(verified_images)} images")
        print(f"Expected: {len(relevant_doc_ids)} docs, {len(relevant_image_ids)} images")
        
        # Calculate metrics for each k
        metrics = {
            'query_id': query_id,
            'query': query,
            'query_type': query_type,
            'category': query_data['category'],
            'retrieved_chunks': len(text_chunks),
            'retrieved_images': len(verified_images),
            'expected_docs': len(relevant_doc_ids),
            'expected_images': len(relevant_image_ids)
        }
        
        # Compute metrics at different k values
        for k in k_values:
            # Text chunk metrics
            recall_k = self.calc_recall(retrieved_doc_ids[:k], relevant_doc_ids)
            precision_k = self.calc_precision(retrieved_doc_ids[:k], relevant_doc_ids)
            
            metrics[f'recall@{k}'] = recall_k
            metrics[f'precision@{k}'] = precision_k
            
            print(f"  @k={k}: Recall={recall_k:.2f}, Precision={precision_k:.2f}")
        
        # MRR (Mean Reciprocal Rank)
        mrr = self.calc_mrr(retrieved_doc_ids, relevant_doc_ids)
        metrics['mrr'] = mrr
        print(f"  MRR: {mrr:.3f}")
        
        # Image Hit Rate
        image_hit = 1.0 if (len(verified_images) > 0 and len(relevant_image_ids) > 0) else 0.0
        metrics['image_hit'] = image_hit
        
        # Image recall (if images expected)
        if relevant_image_ids:
            img_recall = len(set(retrieved_image_ids) & relevant_image_ids) / len(relevant_image_ids)
            metrics['image_recall'] = img_recall
            print(f"  Image Hit: {image_hit} | Image Recall: {img_recall:.2f}")
        else:
            metrics['image_recall'] = None
            print(f"  No images expected")
        
        # Store retrieved items for analysis
        metrics['retrieved_doc_ids'] = retrieved_doc_ids[:10]  # Top 10
        metrics['retrieved_image_ids'] = retrieved_image_ids[:5]  # Top 5
        metrics['relevant_doc_ids'] = list(relevant_doc_ids)
        metrics['relevant_image_ids'] = list(relevant_image_ids)
        
        return metrics
    
    def calc_recall(self, retrieved: List[str], relevant: Set[str]) -> float:
        """
        Recall@k = |retrieved ∩ relevant| / |relevant|
        
        Measures: What fraction of relevant docs were retrieved?
        """
        if not relevant:
            return 1.0  # No relevant docs, perfect recall
        
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
            return 1.0
        
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        
        return 0.0  # No relevant doc found
    
    def evaluate_all(self, k_values: List[int] = [3, 5, 10]) -> dict:
        """
        Run evaluation on all queries in ground truth.
        
        Returns:
            Summary statistics and per-query results
        """
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
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"retrieval_eval_{timestamp}.json"
        
        full_results = {
            'timestamp': timestamp,
            'summary': summary,
            'per_query': all_results,
            'ground_truth_stats': self.ground_truth.get('query_distribution', {})
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        return summary
    
    def aggregate_metrics(self, results: List[dict], k_values: List[int]) -> dict:
        """Aggregate metrics across all queries."""
        
        # Overall metrics
        summary = {
            'total_queries': len(results),
            'by_k': {}
        }
        
        # Aggregate by k
        for k in k_values:
            recall_values = [r[f'recall@{k}'] for r in results]
            precision_values = [r[f'precision@{k}'] for r in results]
            
            summary['by_k'][k] = {
                'avg_recall': sum(recall_values) / len(recall_values),
                'avg_precision': sum(precision_values) / len(precision_values),
                'min_recall': min(recall_values),
                'max_recall': max(recall_values),
                'min_precision': min(precision_values),
                'max_precision': max(precision_values)
            }
        
        # MRR
        mrr_values = [r['mrr'] for r in results]
        summary['mrr'] = {
            'average': sum(mrr_values) / len(mrr_values),
            'min': min(mrr_values),
            'max': max(mrr_values)
        }
        
        # Image metrics
        visual_queries = [r for r in results if r['expected_images'] > 0]
        if visual_queries:
            image_hit_rate = sum(r['image_hit'] for r in visual_queries) / len(visual_queries)
            image_recall_values = [r['image_recall'] for r in visual_queries if r['image_recall'] is not None]
            avg_image_recall = sum(image_recall_values) / len(image_recall_values) if image_recall_values else 0.0
            
            summary['images'] = {
                'visual_queries': len(visual_queries),
                'hit_rate': image_hit_rate,
                'avg_recall': avg_image_recall
            }
        else:
            summary['images'] = None
        
        # By query type
        summary['by_query_type'] = {}
        for qtype in ['text_focused', 'visual', 'hybrid']:
            type_results = [r for r in results if r['query_type'] == qtype]
            if type_results:
                summary['by_query_type'][qtype] = {
                    'count': len(type_results),
                    'avg_recall@5': sum(r['recall@5'] for r in type_results) / len(type_results),
                    'avg_precision@5': sum(r['precision@5'] for r in type_results) / len(type_results),
                    'avg_mrr': sum(r['mrr'] for r in type_results) / len(type_results)
                }
        
        return summary
    
    def print_summary(self, summary: dict):
        """Print summary in human-readable format."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\n📊 Overall Performance:")
        print(f"  Total queries: {summary['total_queries']}")
        
        # Metrics by k
        print(f"\n📈 Retrieval Metrics:")
        for k, metrics in summary['by_k'].items():
            print(f"\n  @k={k}:")
            print(f"    Recall:    {metrics['avg_recall']:.3f} (min: {metrics['min_recall']:.2f}, max: {metrics['max_recall']:.2f})")
            print(f"    Precision: {metrics['avg_precision']:.3f} (min: {metrics['min_precision']:.2f}, max: {metrics['max_precision']:.2f})")
        
        # MRR
        print(f"\n  MRR: {summary['mrr']['average']:.3f} (min: {summary['mrr']['min']:.2f}, max: {summary['mrr']['max']:.2f})")
        
        # Image metrics
        if summary['images']:
            print(f"\n🖼️  Image Retrieval:")
            print(f"    Visual queries: {summary['images']['visual_queries']}")
            print(f"    Hit Rate: {summary['images']['hit_rate']:.1%}")
            print(f"    Avg Recall: {summary['images']['avg_recall']:.3f}")
        
        # By query type
        print(f"\n📋 By Query Type:")
        for qtype, metrics in summary['by_query_type'].items():
            print(f"\n  {qtype} ({metrics['count']} queries):")
            print(f"    Recall@5: {metrics['avg_recall@5']:.3f}")
            print(f"    Precision@5: {metrics['avg_precision@5']:.3f}")
            print(f"    MRR: {metrics['avg_mrr']:.3f}")
        
        # Targets comparison
        print(f"\n🎯 Target Comparison:")
        recall_5 = summary['by_k'][5]['avg_recall']
        print(f"    Recall@5: {recall_5:.1%} (target: ≥70%) {'✅' if recall_5 >= 0.70 else '❌'}")
        
        if summary['images']:
            hit_rate = summary['images']['hit_rate']
            print(f"    Image Hit Rate: {hit_rate:.1%} (target: ≥60%) {'✅' if hit_rate >= 0.60 else '❌'}")
        
        mrr = summary['mrr']['average']
        print(f"    MRR: {mrr:.3f} (target: ≥0.70) {'✅' if mrr >= 0.70 else '❌'}")
        
        print("\n" + "="*80)


def main():
    """Run retrieval evaluation."""
    evaluator = RetrievalEvaluator(
        ground_truth_path="eval/ground_truth.json",
        output_dir="eval/results"
    )
    
    # Run evaluation
    summary = evaluator.evaluate_all(k_values=[3, 5, 10])
    
    # Print summary
    evaluator.print_summary(summary)
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
