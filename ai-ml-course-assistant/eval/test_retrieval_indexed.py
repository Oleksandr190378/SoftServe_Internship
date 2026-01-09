"""
Retrieval Quality Testing for Indexed Documents.

Part of Phase D: System Evaluation - D2. Retrieval Metrics

Tests retriever performance on 3 indexed documents:
1. arxiv_1409_3215 - Seq2Seq Learning with Neural Networks
2. medium_agents-plan-tasks - AI Agents with Planning  
3. realpython_numpy-tutorial - NumPy Tutorial with Image Processing

Outputs:
- Console: Detailed results for each query
- eval/results/retrieval_test_<timestamp>.txt: Full test log
- eval/results/retrieval_summary_<timestamp>.json: Metrics summary
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retriever import MultimodalRetriever


class RetrievalTester:
    """Test retriever and log results to files."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"retrieval_test_{timestamp}.txt"
        self.summary_file = self.output_dir / f"retrieval_summary_{timestamp}.json"
        
        self.results = []
        
    def log(self, message: str, console=True, file=True):
        """Log message to console and/or file."""
        if console:
            print(message)
        if file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
    
    def test_query(self, retriever, query: str, description: str, query_type: str):
        """Test a single query and record results."""
        separator = "=" * 80
        self.log(f"\n{separator}")
        self.log(f"🔍 TEST: {description}")
        self.log(separator)
        self.log(f"Query: {query}")
        self.log(f"Query Type: {query_type}")
        self.log(f"Visual query detected: {retriever.is_visual_query(query)}")
        self.log("-" * 80)
        
        # Retrieve
        text_chunks, verified_images = retriever.retrieve_with_verification(
            query=query,
            k_text=3
        )
        
        # ===== ASSERTIONS: Validate retrieval quality =====
        # P5 Fix: Add automated validation instead of silent logging
        
        # Assertion 1: Text chunks should always be retrieved
        assert len(text_chunks) > 0, f"FAIL: No text chunks retrieved for query: '{query}'"
        self.log(f"✅ Assertion passed: {len(text_chunks)} text chunks retrieved")
        
        # Assertion 2: Visual queries should retrieve images
        if query_type == "visual" or retriever.is_visual_query(query):
            assert len(verified_images) > 0, f"FAIL: No images for visual query: '{query}'"
            self.log(f"✅ Assertion passed: {len(verified_images)} images for visual query")
        
        # Assertion 3: Check metadata integrity for text chunks
        for i, chunk in enumerate(text_chunks, 1):
            assert 'chunk_id' in chunk.metadata, f"FAIL: Chunk {i} missing 'chunk_id'"
            assert 'doc_id' in chunk.metadata, f"FAIL: Chunk {i} missing 'doc_id'"
            # page_num is optional (None for JSON docs, number for PDF docs)
            if 'page_num' in chunk.metadata:
                assert chunk.metadata['page_num'] is None or isinstance(chunk.metadata['page_num'], int), \
                    f"FAIL: Chunk {i} page_num must be None or int"
            assert len(chunk.page_content) > 0, f"FAIL: Chunk {i} has empty content"
        self.log(f"✅ Assertion passed: All chunks have required metadata")
        
        # Assertion 4: Check metadata integrity for images
        for i, img_data in enumerate(verified_images, 1):
            img = img_data['image']
            assert 'image_id' in img.metadata, f"FAIL: Image {i} missing 'image_id'"
            assert 'doc_id' in img.metadata, f"FAIL: Image {i} missing 'doc_id'"
            assert 'confidence' in img_data, f"FAIL: Image {i} missing 'confidence' score"
            assert img_data['confidence'] in ['HIGH', 'MEDIUM', 'LOW'], \
                f"FAIL: Image {i} invalid confidence: {img_data['confidence']}"
            assert 'similarity' in img_data, f"FAIL: Image {i} missing 'similarity' score"
            assert 0 <= img_data['similarity'] <= 1, \
                f"FAIL: Image {i} similarity out of range: {img_data['similarity']}"
        if verified_images:
            self.log(f"✅ Assertion passed: All images have valid metadata and scores")
        
        # Assertion 5: Confidence scores should match similarity thresholds
        for img_data in verified_images:
            if img_data['confidence'] == 'HIGH':
                assert img_data['similarity'] >= 0.9, \
                    f"FAIL: HIGH confidence but low similarity: {img_data['similarity']}"
            elif img_data['confidence'] == 'MEDIUM':
                # MEDIUM should have reasonable similarity
                assert img_data['similarity'] >= 0.5, \
                    f"FAIL: MEDIUM confidence but similarity < threshold: {img_data['similarity']}"
        if verified_images:
            self.log(f"✅ Assertion passed: Confidence scores consistent with similarity")
        
        self.log("\n" + "-" * 80)
        self.log("📊 RETRIEVAL RESULTS:")
        self.log("-" * 80)
        
        # Extract results
        chunk_results = []
        for i, doc in enumerate(text_chunks, 1):
            chunk_data = {
                'rank': i,
                'chunk_id': doc.metadata.get('chunk_id', 'unknown'),
                'doc_id': doc.metadata.get('doc_id', 'unknown'),
                'page': doc.metadata.get('page_num', 0),
                'has_figure_refs': doc.metadata.get('has_figure_references', False),
                'related_images': doc.metadata.get('related_image_ids', ''),
                'text_preview': doc.page_content[:120].replace('\n', ' ')
            }
            chunk_results.append(chunk_data)
            
            self.log(f"\n  [{i}] {chunk_data['chunk_id']}")
            self.log(f"      Doc: {chunk_data['doc_id']} | Page: {chunk_data['page']}")
            self.log(f"      Figure refs: {chunk_data['has_figure_refs']} | Related: {chunk_data['related_images'] or 'none'}")
            self.log(f"      Text: {chunk_data['text_preview']}...")
        
        # Extract image results
        image_results = []
        self.log(f"\n🖼️  Verified Images ({len(verified_images)}):")
        if verified_images:
            for i, img_data in enumerate(verified_images, 1):
                img = img_data['image']
                img_record = {
                    'rank': i,
                    'image_id': img.metadata.get('image_id', 'unknown'),
                    'doc_id': img.metadata.get('doc_id', 'unknown'),
                    'page': img.metadata.get('page_num', 0),
                    'confidence': img_data['confidence'],
                    'similarity': float(img_data['similarity']),
                    'reason': img_data['reason'],
                    'caption_preview': img.page_content[:100].replace('\n', ' ')
                }
                image_results.append(img_record)
                
                self.log(f"  [{i}] {img_record['image_id']}")
                self.log(f"      Doc: {img_record['doc_id']} | Page: {img_record['page']}")
                self.log(f"      Confidence: {img_record['confidence']} | Similarity: {img_record['similarity']:.3f}")
                self.log(f"      Reason: {img_record['reason']}")
                self.log(f"      Caption: {img_record['caption_preview']}...")
        else:
            self.log("  ❌ No images found")
        
        # Record result summary
        result_summary = {
            'query': query,
            'description': description,
            'query_type': query_type,
            'is_visual_query': retriever.is_visual_query(query),
            'text_chunks_count': len(text_chunks),
            'images_count': len(verified_images),
            'images_by_confidence': {
                'HIGH': sum(1 for img in image_results if img['confidence'] == 'HIGH'),
                'MEDIUM': sum(1 for img in image_results if img['confidence'] == 'MEDIUM'),
                'LOW': sum(1 for img in image_results if img['confidence'] == 'LOW')
            },
            'documents_retrieved': list(set(c['doc_id'] for c in chunk_results)),
            'chunks': chunk_results,
            'images': image_results
        }
        self.results.append(result_summary)
        
        return result_summary
    
    def generate_summary(self):
        """Generate summary statistics and save to JSON."""
        total_queries = len(self.results)
        text_focused = [r for r in self.results if r['query_type'] == 'text']
        visual = [r for r in self.results if r['query_type'] == 'visual']
        hybrid = [r for r in self.results if r['query_type'] == 'hybrid']
        
        summary = {
            'test_date': datetime.now().isoformat(),
            'total_queries': total_queries,
            'query_breakdown': {
                'text_focused': len(text_focused),
                'visual': len(visual),
                'hybrid': len(hybrid)
            },
            'overall_stats': {
                'avg_text_chunks_per_query': sum(r['text_chunks_count'] for r in self.results) / total_queries,
                'avg_images_per_query': sum(r['images_count'] for r in self.results) / total_queries,
                'queries_with_images': sum(1 for r in self.results if r['images_count'] > 0),
                'image_hit_rate': sum(1 for r in self.results if r['images_count'] > 0) / total_queries * 100,
                'total_images_retrieved': sum(r['images_count'] for r in self.results),
                'images_by_confidence': {
                    'HIGH': sum(r['images_by_confidence']['HIGH'] for r in self.results),
                    'MEDIUM': sum(r['images_by_confidence']['MEDIUM'] for r in self.results),
                    'LOW': sum(r['images_by_confidence']['LOW'] for r in self.results)
                }
            },
            'by_query_type': {},
            'detailed_results': self.results
        }
        
        # Stats by query type
        for query_type, queries in [('text', text_focused), ('visual', visual), ('hybrid', hybrid)]:
            if queries:
                summary['by_query_type'][query_type] = {
                    'count': len(queries),
                    'avg_images': sum(r['images_count'] for r in queries) / len(queries),
                    'image_hit_rate': sum(1 for r in queries if r['images_count'] > 0) / len(queries) * 100,
                    'queries': [r['query'] for r in queries]
                }
        
        # Save to JSON
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print summary to console and log
        self.log("\n" + "=" * 80)
        self.log("📊 SUMMARY STATISTICS")
        self.log("=" * 80)
        self.log(f"\nTotal Queries: {total_queries}")
        self.log(f"  - Text-focused: {len(text_focused)}")
        self.log(f"  - Visual: {len(visual)}")
        self.log(f"  - Hybrid: {len(hybrid)}")
        
        self.log(f"\n📈 Overall Performance:")
        self.log(f"  - Avg text chunks/query: {summary['overall_stats']['avg_text_chunks_per_query']:.1f}")
        self.log(f"  - Avg images/query: {summary['overall_stats']['avg_images_per_query']:.1f}")
        self.log(f"  - Image hit rate: {summary['overall_stats']['image_hit_rate']:.1f}%")
        self.log(f"  - Total images retrieved: {summary['overall_stats']['total_images_retrieved']}")
        
        self.log(f"\n🎯 Image Confidence Distribution:")
        for conf, count in summary['overall_stats']['images_by_confidence'].items():
            self.log(f"  - {conf}: {count}")
        
        self.log(f"\n📁 Results saved to:")
        self.log(f"  - Log: {self.log_file}")
        self.log(f"  - Summary: {self.summary_file}")
        
        return summary


def main():
    """Run retrieval tests on indexed documents."""
    print("=" * 80)
    print("🧪 RETRIEVAL QUALITY TEST - Indexed Documents")
    print("=" * 80)
    print("\nDocuments in ChromaDB:")
    print("  1. arxiv_1409_3215 - Seq2Seq with Neural Networks")
    print("  2. medium_agents-plan-tasks - AI Agent Planning")
    print("  3. realpython_numpy-tutorial - NumPy Tutorial")
    print()
    
    # Initialize
    output_dir = Path(__file__).parent / "results"
    tester = RetrievalTester(output_dir)
    retriever = MultimodalRetriever()
    
    # Test queries for indexed documents
    test_cases = [
        # Text-focused (3)
        ("What is LSTM?", "LSTM neural network explanation", "text"),
        ("How do AI agents plan tasks?", "AI agent planning mechanism", "text"),
        ("How to use NumPy arrays?", "NumPy array basics", "text"),
        
        # Visual (3)
        ("Show LSTM architecture", "LSTM architecture diagram", "visual"),
        ("Show planning agent workflow", "Agent planning workflow diagram", "visual"),
        ("Show image processing example", "NumPy image processing visual", "visual"),
        
        # Hybrid (2)
        ("Explain sequence to sequence model with diagram", "Seq2seq with diagram", "hybrid"),
        ("How does agent planning work? Show example", "Agent planning with example", "hybrid"),
    ]
    
    # Run tests
    tester.log("=" * 80, file=False)
    tester.log("🏁 Starting Retrieval Tests", file=False)
    tester.log("=" * 80, file=False)
    
    for query, description, query_type in test_cases:
        tester.test_query(retriever, query, description, query_type)
    
    # Generate summary
    tester.generate_summary()
    
    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
