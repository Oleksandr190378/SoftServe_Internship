"""
Faithfulness Judge - LLM-as-a-Judge for RAG Answer Quality.

Evaluates generated answers using gpt-5-mini as a judge.
Measures 4 dimensions:
- Relevance: Does answer address the query?
- Completeness: Are all important aspects covered?
- Accuracy: Are facts correct and grounded in context?
- Citation Quality: Are sources properly cited?

Target: ≥4.0/5.0 average faithfulness score

Usage:
    python eval/faithfulness_judge.py
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retriever import MultimodalRetriever
from rag.generator import RAGGenerator
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration constants
DEFAULT_GROUND_TRUTH_PATH = "eval/ground_truth.json"
DEFAULT_OUTPUT_DIR = "eval/results"
DEFAULT_CHROMA_DIR = "data/chroma_db"
DEFAULT_K_TEXT = 3  # Number of text chunks to retrieve

# LLM Judge configuration
JUDGE_MODEL = "gpt-5-mini"  # Cost-effective for evaluation
JUDGE_TEMPERATURE = 0.0  # Deterministic scoring

# Evaluation target (from PRD spec)
TARGET_FAITHFULNESS = 4.0  # ≥4.0/5.0 average faithfulness score


# Dataclasses for type-safe structured data
@dataclass
class ScoreDimension:
    """Score for a single evaluation dimension."""
    score: int
    justification: str


@dataclass
class JudgeScores:
    """All scores from LLM judge evaluation."""
    relevance: int
    completeness: int
    accuracy: int
    citation_quality: int
    overall_score: float
    relevance_justification: str
    completeness_justification: str
    accuracy_justification: str
    citation_justification: str
    summary: str
    parse_success: bool
    raw_response: str
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class DimensionStats:
    """Statistics for a single dimension (avg, min, max)."""
    average: float
    min: float
    max: float


@dataclass
class QueryTypeStats:
    """Statistics for a specific query type."""
    count: int
    avg_relevance: float
    avg_completeness: float
    avg_accuracy: float
    avg_citation: float
    avg_overall: float


@dataclass
class QueryEvaluation:
    """Evaluation results for a single query."""
    query_id: int
    query: str
    query_type: str
    category: str
    answer: str
    cited_chunks: List[str]
    cited_images: List[str]
    sources_text: str
    reasoning: str
    num_chunks_retrieved: int
    num_images_retrieved: int
    scores: JudgeScores
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        result = asdict(self)
        # Ensure scores are properly serialized
        result['scores'] = self.scores.to_dict()
        return result


@dataclass
class EvaluationSummary:
    """Summary statistics for entire evaluation."""
    total_queries: int
    average_scores: Dict[str, float]
    min_scores: Dict[str, float]
    max_scores: Dict[str, float]
    by_query_type: Dict[str, QueryTypeStats]
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            'total_queries': self.total_queries,
            'average_scores': self.average_scores,
            'min_scores': self.min_scores,
            'max_scores': self.max_scores,
            'by_query_type': {k: asdict(v) for k, v in self.by_query_type.items()}
        }


# Evaluation prompt
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems.
Your task is to evaluate the quality of generated answers on 4 dimensions.

EVALUATION CRITERIA:

1. RELEVANCE (1-5):
   - Does the answer directly address the query?
   - Is it on-topic and focused?
   - 5: Perfectly addresses query
   - 4: Addresses most aspects
   - 3: Partially relevant
   - 2: Tangentially related
   - 1: Off-topic

2. COMPLETENESS (1-5):
   - Are all important aspects covered?
   - Is the answer comprehensive?
   - 5: Fully comprehensive
   - 4: Covers most key points
   - 3: Missing some details
   - 2: Incomplete
   - 1: Very superficial

3. ACCURACY (1-5):
   - Are facts correct based on context?
   - No hallucinations or unsupported claims?
   - 5: Perfectly accurate, all claims supported
   - 4: Mostly accurate, minor issues
   - 3: Some inaccuracies
   - 2: Several errors
   - 1: Mostly incorrect

4. CITATION QUALITY (1-5):
   - Are sources properly cited?
   - Are citations accurate and sufficient?
   - 5: All claims cited, accurate labels
   - 4: Most claims cited
   - 3: Some citations missing
   - 2: Sparse citations
   - 1: No citations or wrong labels

RESPONSE FORMAT:
Provide scores and brief justifications in this exact format:

Relevance: <score>/5
Justification: <1-2 sentences>

Completeness: <score>/5
Justification: <1-2 sentences>

Accuracy: <score>/5
Justification: <1-2 sentences>

Citation Quality: <score>/5
Justification: <1-2 sentences>

Overall Faithfulness: <average score>/5.0
Summary: <2-3 sentences overall assessment>
"""


class FaithfulnessJudge:
    """LLM-as-a-Judge for evaluating RAG answer quality."""
    
    def __init__(
        self, 
        ground_truth_path: str = DEFAULT_GROUND_TRUTH_PATH, 
        output_dir: str = DEFAULT_OUTPUT_DIR,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        retriever: MultimodalRetriever = None,
        generator: RAGGenerator = None
    ):
        """
        Initialize judge with retriever, generator, and evaluation LLM.
        
        Args:
            ground_truth_path: Path to ground truth queries
            output_dir: Directory for saving results
            chroma_dir: Path to ChromaDB directory
            retriever: Optional retriever instance (for testing/injection)
            generator: Optional generator instance (for testing/injection)
        """
        self.ground_truth = self.load_ground_truth(ground_truth_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or inject RAG components
        if retriever is None:
            logging.info("Initializing retriever...")
            self.retriever = MultimodalRetriever(chroma_dir=Path(chroma_dir))
        else:
            self.retriever = retriever
        
        if generator is None:
            logging.info("Initializing generator...")
            self.generator = RAGGenerator()
        else:
            self.generator = generator
        
        # Initialize judge LLM
        logging.info(f"Initializing judge: {JUDGE_MODEL}")
        self.judge = ChatOpenAI(
            model=JUDGE_MODEL,
            temperature=JUDGE_TEMPERATURE
        )
        
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
    
    def _validate_score_range(self, score: int, dimension: str) -> int:
        """
        Validate score is in valid range (1-5).
        
        Args:
            score: Raw score value
            dimension: Name of dimension for logging
        
        Returns:
            Validated score (clamped to 1-5) or None if invalid
        """
        if score is None or score == 0:
            logging.warning(f"Missing {dimension} score - judge failed to parse")
            return None
        
        if not (1 <= score <= 5):
            logging.warning(f"Invalid {dimension} score: {score} (expected 1-5) - clamping")
            return max(1, min(5, score))
        
        return score
    
    def format_evaluation_prompt(self, query: str, answer: str, 
                                 context: str, cited_sources: str) -> str:
        """
        Format prompt for LLM judge.
        
        Args:
            query: Original user query
            answer: Generated answer to evaluate
            context: Retrieved context used for generation
            cited_sources: Sources cited in answer
        
        Returns:
            Formatted prompt for judge
        """
        prompt = f"""QUERY:
{query}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

CITED SOURCES:
{cited_sources}

Please evaluate this answer on the 4 dimensions (Relevance, Completeness, Accuracy, Citation Quality).
Focus on whether the answer is grounded in the provided context and properly cites sources."""
        
        return prompt
    
    def parse_judge_response(self, response: str) -> JudgeScores:
        """
        Parse judge LLM response into structured scores.
        
        Expected format:
            Relevance: 5/5
            Justification: ...
            
            Completeness: 4/5
            ...
        
        Returns:
            JudgeScores dataclass with all scores and justifications
        """
        import re
        
        # Extract scores using regex
        relevance = re.search(r'Relevance:\s*(\d+)/5', response)
        completeness = re.search(r'Completeness:\s*(\d+)/5', response)
        accuracy = re.search(r'Accuracy:\s*(\d+)/5', response)
        citation = re.search(r'Citation Quality:\s*(\d+)/5', response)
        overall = re.search(r'Overall Faithfulness:\s*([\d.]+)/5', response)
        
        # Extract justifications (text after "Justification:" until next section)
        rel_just = re.search(r'Relevance:.*?Justification:\s*(.+?)(?=\n\n|\nCompleteness:)', 
                            response, re.DOTALL)
        comp_just = re.search(r'Completeness:.*?Justification:\s*(.+?)(?=\n\n|\nAccuracy:)', 
                             response, re.DOTALL)
        acc_just = re.search(r'Accuracy:.*?Justification:\s*(.+?)(?=\n\n|\nCitation Quality:)', 
                            response, re.DOTALL)
        cit_just = re.search(r'Citation Quality:.*?Justification:\s*(.+?)(?=\n\n|\nOverall)', 
                            response, re.DOTALL)
        summary = re.search(r'Summary:\s*(.+?)$', response, re.DOTALL)
        
        # Extract raw scores
        rel_score = int(relevance.group(1)) if relevance else None
        comp_score = int(completeness.group(1)) if completeness else None
        acc_score = int(accuracy.group(1)) if accuracy else None
        cit_score = int(citation.group(1)) if citation else None
        overall_score = float(overall.group(1)) if overall else None
        
        # Validate score ranges (1-5)
        rel_score = self._validate_score_range(rel_score, "Relevance")
        comp_score = self._validate_score_range(comp_score, "Completeness")
        acc_score = self._validate_score_range(acc_score, "Accuracy")
        cit_score = self._validate_score_range(cit_score, "Citation Quality")
        
        # Check if any critical scores are missing
        if any(s is None for s in [rel_score, comp_score, acc_score, cit_score]):
            logging.error("Judge response parsing failed - missing critical scores")
            logging.debug(f"Raw response: {response[:200]}...")  # Log first 200 chars
        
        # Validate overall score
        if overall_score and not (1.0 <= overall_score <= 5.0):
            logging.warning(f"Invalid overall score: {overall_score} (expected 1.0-5.0)")
            overall_score = max(1.0, min(5.0, overall_score))
        
        return JudgeScores(
            relevance=rel_score if rel_score else 0,
            completeness=comp_score if comp_score else 0,
            accuracy=acc_score if acc_score else 0,
            citation_quality=cit_score if cit_score else 0,
            overall_score=overall_score if overall_score else 0.0,
            relevance_justification=rel_just.group(1).strip() if rel_just else "[Parse failed]",
            completeness_justification=comp_just.group(1).strip() if comp_just else "[Parse failed]",
            accuracy_justification=acc_just.group(1).strip() if acc_just else "[Parse failed]",
            citation_justification=cit_just.group(1).strip() if cit_just else "[Parse failed]",
            summary=summary.group(1).strip() if summary else "[Parse failed]",
            raw_response=response,
            parse_success=all(s is not None for s in [rel_score, comp_score, acc_score, cit_score])
        )
    
    def _aggregate_single_dimension(self, scores: List[float]) -> Dict:
        """DRY helper: compute avg/min/max for a single dimension."""
        if not scores:
            return {'average': 0.0, 'min': 0, 'max': 0}
        return {
            'average': round(sum(scores) / len(scores), 2),
            'min': min(scores),
            'max': max(scores)
        }
    
    def _retrieve_context(self, query: str) -> tuple:
        """SRP: Handle retrieval operation only."""
        logging.info("Step 1/3: Retrieving context...")
        text_chunks, verified_images = self.retriever.retrieve_with_verification(
            query=query,
            k_text=DEFAULT_K_TEXT
        )
        
        llm_input = self.retriever.prepare_for_llm(query, text_chunks, verified_images)
        logging.info(f"Retrieved: {len(text_chunks)} chunks, {len(verified_images)} images")
        
        return text_chunks, verified_images, llm_input
    
    def _generate_answer(self, llm_input: Dict) -> Dict:
        """SRP: Handle answer generation only."""
        logging.info("Step 2/3: Generating answer...")
        gen_result = self.generator.generate(llm_input)
        
        answer = gen_result['answer']
        cited_sources = gen_result['sources_text']
        
        logging.info(f"Answer generated: {len(answer)} chars")
        logging.info(f"Citations: {len(gen_result['cited_chunks'])} chunks, {len(gen_result['cited_images'])} images")
        
        return gen_result
    
    def _evaluate_with_judge(self, query: str, answer: str, llm_input: Dict, cited_sources: str) -> JudgeScores:
        """SRP: Handle LLM judge evaluation only."""
        logging.info("Step 3/3: Evaluating with LLM judge...")
        
        # Format context for judge
        context = self.generator.format_context_for_llm(llm_input)
        
        # Create evaluation prompt
        eval_prompt = self.format_evaluation_prompt(
            query=query,
            answer=answer,
            context=context,
            cited_sources=cited_sources
        )
        
        # Call judge LLM
        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=eval_prompt)
        ]
        
        try:
            response = self.judge.invoke(messages)
            judge_response = response.content if hasattr(response, 'content') else str(response)
            
            # Parse scores
            scores = self.parse_judge_response(judge_response)
            
            return scores
            
        except Exception as e:
            logging.error(f"Error calling judge: {e}")
            return JudgeScores(
                relevance=0,
                completeness=0,
                accuracy=0,
                citation_quality=0,
                overall_score=0.0,
                relevance_justification=f"[Error: {e}]",
                completeness_justification=f"[Error: {e}]",
                accuracy_justification=f"[Error: {e}]",
                citation_justification=f"[Error: {e}]",
                summary=f"[Error: {e}]",
                raw_response="",
                parse_success=False
            )
    
    def _log_evaluation_results(self, scores: JudgeScores) -> None:
        """SRP: Handle console logging only."""
        logging.info(f"Scores: Relevance={scores.relevance}, "
                    f"Completeness={scores.completeness}, "
                    f"Accuracy={scores.accuracy}, "
                    f"Citation={scores.citation_quality}")
        logging.info(f"Overall Faithfulness: {scores.overall_score:.2f}/5.0")
    
    def evaluate_query(self, query_data: dict) -> QueryEvaluation:
        """
        Orchestrate single query evaluation (refactored for SRP).
        
        Steps:
        1. Run retrieval
        2. Generate answer
        3. Use LLM judge to evaluate
        
        Args:
            query_data: Ground truth query data
        
        Returns:
            QueryEvaluation dataclass with all results
        """
        query = query_data['query']
        query_id = query_data['id']
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Evaluating {query_id}: {query}")
        logging.info(f"{'='*80}")
        
        # 1. Retrieve context (SRP)
        text_chunks, verified_images, llm_input = self._retrieve_context(query)
        
        # 2. Generate answer (SRP)
        gen_result = self._generate_answer(llm_input)
        answer = gen_result['answer']
        cited_sources = gen_result['sources_text']
        
        # 3. Evaluate with judge (SRP)
        scores = self._evaluate_with_judge(query, answer, llm_input, cited_sources)
        
        # 4. Log results (SRP)
        self._log_evaluation_results(scores)
        
        # 5. Build QueryEvaluation dataclass
        return QueryEvaluation(
            query_id=query_id,
            query=query,
            query_type=query_data['query_type'],
            category=query_data['category'],
            answer=answer,
            cited_chunks=gen_result['cited_chunks'],
            cited_images=gen_result['cited_images'],
            sources_text=cited_sources,
            reasoning=gen_result.get('reasoning', ''),
            num_chunks_retrieved=len(text_chunks),
            num_images_retrieved=len(verified_images),
            scores=scores
        )
    
    def evaluate_all(self, max_queries: int = None) -> EvaluationSummary:
        """
        Run evaluation on all queries in ground truth.
        
        Args:
            max_queries: Maximum number of queries to evaluate (None = all)
        
        Returns:
            EvaluationSummary dataclass with aggregated statistics
        """
        queries_to_eval = self.ground_truth['queries'][:max_queries] if max_queries else self.ground_truth['queries']
        
        logging.info("\n" + "="*80)
        logging.info("FAITHFULNESS EVALUATION")
        logging.info("="*80)
        logging.info(f"Total queries: {len(queries_to_eval)}")
        logging.info(f"Judge model: {JUDGE_MODEL}")
        
        all_results = []
        
        for query_data in queries_to_eval:
            result = self.evaluate_query(query_data)
            all_results.append(result)
        
        # Aggregate scores
        summary = self.aggregate_scores(all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"faithfulness_eval_{timestamp}.json"
        
        full_results = {
            'timestamp': timestamp,
            'judge_model': JUDGE_MODEL,
            'summary': summary.to_dict(),
            'per_query': [r.to_dict() for r in all_results]
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False)
            logging.info(f"\n✅ Results saved to: {output_file}")
        except IOError as e:
            logging.error(f"\n⚠️  Warning: Could not save results to {output_file}: {e}")
            logging.info("Evaluation completed but results not persisted.")
        
        return summary
    
    def aggregate_scores(self, results: List[QueryEvaluation]) -> EvaluationSummary:
        """Aggregate scores across all queries (returns type-safe EvaluationSummary)."""
        
        # Guard against empty results
        if not results:
            logging.warning("No results to aggregate - returning empty summary")
            return EvaluationSummary(
                total_queries=0,
                average_scores={
                    'relevance': 0.0,
                    'completeness': 0.0,
                    'accuracy': 0.0,
                    'citation_quality': 0.0,
                    'overall_faithfulness': 0.0
                },
                min_scores={'relevance': 0, 'completeness': 0, 'accuracy': 0, 'citation_quality': 0, 'overall_faithfulness': 0.0},
                max_scores={'relevance': 0, 'completeness': 0, 'accuracy': 0, 'citation_quality': 0, 'overall_faithfulness': 0.0},
                by_query_type={}
            )
        
        # Extract all scores from dataclasses
        relevance_scores = [r.scores.relevance for r in results]
        completeness_scores = [r.scores.completeness for r in results]
        accuracy_scores = [r.scores.accuracy for r in results]
        citation_scores = [r.scores.citation_quality for r in results]
        overall_scores = [r.scores.overall_score for r in results]
        
        # Use DRY helper for aggregation
        rel_stats = self._aggregate_single_dimension(relevance_scores)
        comp_stats = self._aggregate_single_dimension(completeness_scores)
        acc_stats = self._aggregate_single_dimension(accuracy_scores)
        cit_stats = self._aggregate_single_dimension(citation_scores)
        overall_stats = self._aggregate_single_dimension(overall_scores)
        
        summary = {
            'total_queries': len(results),
            'average_scores': {
                'relevance': rel_stats['average'],
                'completeness': comp_stats['average'],
                'accuracy': acc_stats['average'],
                'citation_quality': cit_stats['average'],
                'overall_faithfulness': overall_stats['average']
            },
            'min_scores': {
                'relevance': rel_stats['min'],
                'completeness': comp_stats['min'],
                'accuracy': acc_stats['min'],
                'citation_quality': cit_stats['min'],
                'overall_faithfulness': overall_stats['min']
            },
            'max_scores': {
                'relevance': rel_stats['max'],
                'completeness': comp_stats['max'],
                'accuracy': acc_stats['max'],
                'citation_quality': cit_stats['max'],
                'overall_faithfulness': overall_stats['max']
            }
        }
        
        # By query type - create QueryTypeStats dataclasses
        by_query_type = {}
        for qtype in ['text_focused', 'visual', 'hybrid']:
            type_results = [r for r in results if r.query_type == qtype]
            if type_results:
                type_rel = [r.scores.relevance for r in type_results]
                type_comp = [r.scores.completeness for r in type_results]
                type_acc = [r.scores.accuracy for r in type_results]
                type_cit = [r.scores.citation_quality for r in type_results]
                type_overall = [r.scores.overall_score for r in type_results]
                
                by_query_type[qtype] = QueryTypeStats(
                    count=len(type_results),
                    avg_relevance=round(sum(type_rel) / len(type_rel), 2),
                    avg_completeness=round(sum(type_comp) / len(type_comp), 2),
                    avg_accuracy=round(sum(type_acc) / len(type_acc), 2),
                    avg_citation=round(sum(type_cit) / len(type_cit), 2),
                    avg_overall=round(sum(type_overall) / len(type_overall), 2)
                )
        
        return EvaluationSummary(
            total_queries=len(results),
            average_scores={
                'relevance': rel_stats['average'],
                'completeness': comp_stats['average'],
                'accuracy': acc_stats['average'],
                'citation_quality': cit_stats['average'],
                'overall_faithfulness': overall_stats['average']
            },
            min_scores={
                'relevance': rel_stats['min'],
                'completeness': comp_stats['min'],
                'accuracy': acc_stats['min'],
                'citation_quality': cit_stats['min'],
                'overall_faithfulness': overall_stats['min']
            },
            max_scores={
                'relevance': rel_stats['max'],
                'completeness': comp_stats['max'],
                'accuracy': acc_stats['max'],
                'citation_quality': cit_stats['max'],
                'overall_faithfulness': overall_stats['max']
            },
            by_query_type=by_query_type
        )
    
    def print_summary(self, summary: EvaluationSummary):
        """Print summary in human-readable format (now type-safe with dataclass)."""
        print("\n" + "="*80)
        print("FAITHFULNESS EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\n📊 Overall Performance:")
        print(f"  Total queries: {summary.total_queries}")
        
        avg = summary.average_scores
        print(f"\n📈 Average Scores (out of 5):")
        print(f"  Relevance:        {avg['relevance']:.2f}")
        print(f"  Completeness:     {avg['completeness']:.2f}")
        print(f"  Accuracy:         {avg['accuracy']:.2f}")
        print(f"  Citation Quality: {avg['citation_quality']:.2f}")
        print(f"  Overall Faithfulness: {avg['overall_faithfulness']:.2f}")
        
        # Target comparison
        overall = avg['overall_faithfulness']
        status = "✅" if overall >= TARGET_FAITHFULNESS else "❌"
        
        print(f"\n🎯 Target Comparison:")
        print(f"  Overall Faithfulness: {overall:.2f}/5.0 (target: ≥{TARGET_FAITHFULNESS}) {status}")
        
        # By query type
        print(f"\n📋 By Query Type:")
        for qtype, metrics in summary.by_query_type.items():
            print(f"\n  {qtype} ({metrics.count} queries):")
            print(f"    Relevance: {metrics.avg_relevance:.2f}")
            print(f"    Completeness: {metrics.avg_completeness:.2f}")
            print(f"    Accuracy: {metrics.avg_accuracy:.2f}")
            print(f"    Citation: {metrics.avg_citation:.2f}")
            print(f"    Overall: {metrics.avg_overall:.2f}")
        
        # Score ranges
        min_scores = summary.min_scores
        max_scores = summary.max_scores
        print(f"\n📊 Score Ranges:")
        print(f"  Overall: {min_scores['overall_faithfulness']:.1f} - {max_scores['overall_faithfulness']:.1f}")
        
        print("\n" + "="*80)


def main():
    """Run faithfulness evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate RAG answer faithfulness')
    parser.add_argument('--max-queries', type=int, default=None,
                        help='Maximum number of queries to evaluate (default: all)')
    args = parser.parse_args()
    
    try:
        judge = FaithfulnessJudge(
            ground_truth_path=DEFAULT_GROUND_TRUTH_PATH,
            output_dir=DEFAULT_OUTPUT_DIR,
            chroma_dir=DEFAULT_CHROMA_DIR
        )
        
        # Run evaluation
        summary = judge.evaluate_all(max_queries=args.max_queries)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print summary
    judge.print_summary(summary)
    
    print("\n✅ Faithfulness evaluation complete!")


if __name__ == '__main__':
    main()
