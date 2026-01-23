"""
Query Complexity Analyzer for Dynamic k_text Selection.

Uses LLM to analyze query complexity and determine optimal number of text chunks.
Integrates with RAG pipeline to improve retrieval quality.
"""

import logging
import re
from typing import Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import centralized configuration
from config import QUERY_ANALYZER
from utils.logging_config import setup_logging

load_dotenv()
setup_logging()  # Centralized logging configuration

# Query Analyzer configuration (from config)
MODEL_NAME = QUERY_ANALYZER.MODEL
TEMPERATURE = QUERY_ANALYZER.TEMPERATURE
MAX_TOKENS = QUERY_ANALYZER.MAX_TOKENS

# Complexity Levels & k_text Mapping
COMPLEXITY_SIMPLE = "simple"
COMPLEXITY_MODERATE = "moderate"
COMPLEXITY_COMPLEX = "complex"

K_TEXT_SIMPLE = 3  # Simple queries (single concept, definition)
K_TEXT_MODERATE = 4  # Moderate queries (comparisons, explanations)
K_TEXT_COMPLEX = 5  # Complex queries (multi-step, deep analysis)

# Query Validation
MAX_QUERY_LENGTH = 500  # Maximum query length 
MIN_QUERY_LENGTH = 3  # Minimum meaningful query length

# Default Fallback
DEFAULT_K_TEXT = K_TEXT_SIMPLE  # Safe default if analysis fails
DEFAULT_COMPLEXITY = COMPLEXITY_SIMPLE


# ═══════════════════════════════════════════════════════════════════════
# PYDANTIC MODEL FOR STRUCTURED OUTPUT
# ═══════════════════════════════════════════════════════════════════════

class QueryComplexityOutput(BaseModel):
    """Structured output for query complexity analysis."""
    
    is_ai_ml_related: bool = Field(
        ...,
        description="Whether the query is about AI/ML/DL/Data Science topics (true) or off-topic (false)"
    )
    complexity: str = Field(
        ...,
        description="Query complexity level: 'simple', 'moderate', or 'complex' (only if is_ai_ml_related=true)"
    )
    k_text: int = Field(
        ...,
        description="Number of text chunks to retrieve (3, 4, or 5) (only if is_ai_ml_related=true)"
    )
    reasoning: str = Field(
        ...,
        description="Brief 1-sentence explanation of the classification"
    )

# ═══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT - QUERY COMPLEXITY CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Analyze query complexity for AI/ML RAG system.

TASK: Return JSON with is_ai_ml_related, complexity, k_text, reasoning

AI/ML TOPICS (is_ai_ml_related=true): Deep Learning, Neural Networks, CNNs, RNNs, Transformers, ML algorithms, Computer Vision, NLP, Data Science, PyTorch, TensorFlow

OFF-TOPIC (is_ai_ml_related=false): Cooking, Sports, Politics, History, General programming, General math

COMPLEXITY RULES:
- SIMPLE (k=3): Single concept "what is X", "show X", "define X"
- MODERATE (k=4): Binary comparison "compare X and Y", "how does X work"
- COMPLEX (k=5): Multi-concept "compare X, Y, Z", "why", "analyze"

For OFF-TOPIC queries, set is_ai_ml_related=false with complexity=simple, k_text=3, reasoning="Query is not about AI/ML/Data Science".

Provide only the JSON output."""


class QueryComplexityAnalyzer:
    """
    Analyzes query complexity to determine optimal k_text for retrieval.
    
    Features:
    - LLM-based classification (gpt-4o-mini)
    - Three complexity levels (simple, moderate, complex)
    - Automatic k_text selection (3, 4, 5)
    - Input validation and sanitization
    - Robust error handling with fallback
    - Dependency injection for configurability
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = MODEL_NAME,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS
    ):
        """
        Initialize query complexity analyzer.
        
        Args:
            openai_api_key: Optional API key (uses .env if not provided)
            model_name: OpenAI model to use
            temperature: Temperature for LLM (0.0 for deterministic)
            max_tokens: Maximum tokens for LLM response
        
        Raises:
            ValueError: If model configuration is invalid
        """
        logging.info(f"Initializing QueryComplexityAnalyzer with {model_name}")
        
        # Validate configuration
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError(f"Invalid temperature: {temperature}. Must be 0.0-2.0")
        
        if max_tokens < 50:
            raise ValueError(f"Invalid max_tokens: {max_tokens}. Must be ≥50")
        
        try:
            base_llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=openai_api_key
            )
            # Use structured output for reliable parsing
            self.llm = base_llm.with_structured_output(QueryComplexityOutput)
            self.system_prompt = SYSTEM_PROMPT
            logging.info("Analyzer ready with structured output")
        
        except ValueError as e:
            logging.error(f"Invalid LLM configuration: {e}")
            raise
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}") from e
    
    def _validate_query(self, query: str) -> str:
        """
        Validate and sanitize query input.
        
        Single Responsibility: Input validation.
        
        Args:
            query: Raw user query
        
        Returns:
            Sanitized query string
        
        Raises:
            ValueError: If query is invalid
        """
        if not query:
            raise ValueError("Query cannot be empty")
        
        # Remove control characters
        query = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query)
        query = query.strip()
        
        if len(query) < MIN_QUERY_LENGTH:
            raise ValueError(f"Query too short: minimum {MIN_QUERY_LENGTH} characters")
        
        if len(query) > MAX_QUERY_LENGTH:
            logging.warning(f"Query truncated from {len(query)} to {MAX_QUERY_LENGTH} chars")
            query = query[:MAX_QUERY_LENGTH]
        
        return query
    
    def _validate_output(self, output: QueryComplexityOutput) -> Dict[str, any]:
        """
        Validate structured output from LLM.
        
        Single Responsibility: Output validation.
        
        Args:
            output: Pydantic model output from LLM
        
        Returns:
            Dict with validated is_ai_ml_related, complexity, k_text, reasoning
        
        Raises:
            ValueError: If output values are invalid
        """
        # is_ai_ml_related is always valid (bool)
        
        # If off-topic, don't validate complexity/k_text strictly
        if not output.is_ai_ml_related:
            return {
                'is_ai_ml_related': False,
                'complexity': output.complexity,
                'k_text': output.k_text,
                'reasoning': output.reasoning
            }
        
        # Validate complexity (only for AI/ML queries)
        valid_complexities = {COMPLEXITY_SIMPLE, COMPLEXITY_MODERATE, COMPLEXITY_COMPLEX}
        if output.complexity not in valid_complexities:
            raise ValueError(
                f"Invalid complexity '{output.complexity}'. "
                f"Must be one of: {valid_complexities}"
            )
        
        # Validate k_text
        valid_k_values = {K_TEXT_SIMPLE, K_TEXT_MODERATE, K_TEXT_COMPLEX}
        if output.k_text not in valid_k_values:
            raise ValueError(
                f"Invalid k_text {output.k_text}. "
                f"Must be one of: {valid_k_values}"
            )
        
        return {
            'is_ai_ml_related': True,
            'complexity': output.complexity,
            'k_text': output.k_text,
            'reasoning': output.reasoning
        }
    
    def _get_fallback_result(self, query: str, error_msg: str = "") -> Dict[str, any]:
        """
        Generate fallback result when analysis fails.
        
        Single Responsibility: Fallback handling.
        
        Args:
            query: Original query
            error_msg: Optional error message for logging
        
        Returns:
            Default complexity analysis result
        """
        if error_msg:
            logging.warning(f"Using fallback: {error_msg}")
        
        return {
            'is_ai_ml_related': True,  # Assume on-topic for fallback (Generator will validate)
            'complexity': DEFAULT_COMPLEXITY,
            'k_text': DEFAULT_K_TEXT,
            'reasoning': 'Fallback due to analysis error',
            'query': query,
            'fallback': True
        }
    
    def analyze(self, query: str) -> Dict[str, any]:
        """
        Analyze query complexity and determine optimal k_text.
        
        Main pipeline:
        1. Validate query
        2. Call LLM classifier
        3. Parse response
        4. Return structured result
        
        Args:
            query: User query string
        
        Returns:
            Dict with:
                - complexity: 'simple' | 'moderate' | 'complex'
                - k_text: 3 | 4 | 5
                - reasoning: Explanation of classification
                - query: Original query (for reference)
                - fallback: True if fallback used (optional)
        """
        import time
        start_time = time.perf_counter()
        
        try:
            # Step 1: Validate input
            sanitized_query = self._validate_query(query)
            logging.info(f"Analyzing query complexity: '{sanitized_query[:50]}...'")
            
            # Step 2: Build messages
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"Query: {sanitized_query}")
            ]
            
            # Step 3: Call LLM with structured output
            output = self.llm.invoke(messages)
            
            if not output:
                return self._get_fallback_result(query, "Empty LLM response")
            
            logging.info(
                f"LLM response: is_ai_ml_related={output.is_ai_ml_related}, "
                f"complexity={output.complexity}, k_text={output.k_text}"
            )
            
            # Step 4: Validate output
            result = self._validate_output(output)
            result['query'] = sanitized_query
            result['fallback'] = False
            
            # Log based on topic relevance
            if not result['is_ai_ml_related']:
                logging.warning(f"⚠ Off-topic query detected: '{sanitized_query[:50]}...'")
            else:
                logging.info(
                    f"✓ Complexity: {result['complexity']} → k_text={result['k_text']}"
                )
            
            # Log performance
            analysis_time = time.perf_counter() - start_time
            logging.info(f"Analyzed in {analysis_time:.3f}s (complexity: {result['complexity']}, k_text: {result['k_text']})")
            
            return result
        
        except ValueError as e:
            logging.error(f"Validation error: {e}")
            return self._get_fallback_result(query, f"Validation: {e}")
        
        except ConnectionError as e:
            logging.error(f"LLM connection error: {e}")
            return self._get_fallback_result(query, f"Connection: {e}")
        
        except TimeoutError as e:
            logging.error(f"LLM timeout: {e}")
            return self._get_fallback_result(query, f"Timeout: {e}")
        
        except Exception as e:
            logging.error(f"Unexpected error: {type(e).__name__} - {e}")
            return self._get_fallback_result(query, f"Error: {type(e).__name__}")


def test_analyzer():
    """Test analyzer with various query types aligned with RAG agent capabilities."""
    print("=" * 70)
    print("🧪 Testing Query Complexity Analyzer (RAG-optimized)")
    print("=" * 70)
    print()
    
    analyzer = QueryComplexityAnalyzer()
    
    test_queries = [
        # SIMPLE queries (k=3)
        ("what is ResNet", "Expected: simple, k=3 (single concept definition)"),
        ("show transformer architecture", "Expected: simple, k=3 (single diagram)"),
        ("define convolutional layer", "Expected: simple, k=3 (basic definition)"),
        ("explain gradient descent", "Expected: simple, k=3 (single algorithm)"),
        
        # MODERATE queries (k=4)
        ("compare ResNet and VGG", "Expected: moderate, k=4 (binary comparison)"),
        ("how does attention mechanism work", "Expected: moderate, k=4 (mechanism = HOW)"),
        ("explain self-attention and its benefits", "Expected: moderate, k=4 (multi-aspect)"),
        ("what's the difference between CNN and RNN", "Expected: moderate, k=4 (2 concepts)"),
        
        # COMPLEX queries (k=5)
        ("compare ResNet, VGG, and Inception", "Expected: complex, k=5 (3+ concepts)"),
        ("why does ResNet solve vanishing gradients better", "Expected: complex, k=5 (causal WHY)"),
        ("explain all types of attention in transformers", "Expected: complex, k=5 (comprehensive)"),
        ("how do residual connections improve training and inference", "Expected: complex, k=5 (theory+practice)"),
        
        # OFF-TOPIC queries (should be detected)
        ("how to cook pasta", "Expected: is_ai_ml_related=false (cooking)"),
        ("who won the football game", "Expected: is_ai_ml_related=false (sports)"),
        ("what is the capital of France", "Expected: is_ai_ml_related=false (geography)"),
        
        # EDGE cases
        ("", "Expected: error/fallback (empty query)"),
        ("a", "Expected: error/fallback (too short)"),
        ("attention", "Expected: simple/moderate, k=3-4 (ambiguous, default to simple)")
    ]
    
    success_count = 0
    total_count = len([q for q, _ in test_queries if q])  # Exclude error cases
    
    for query, expected in test_queries:
        print(f"\n{'─' * 70}")
        print(f"Query: '{query}'")
        print(f"Expected: {expected}")
        print(f"{'─' * 70}")
        
        try:
            result = analyzer.analyze(query)
            
            print(f"✓ AI/ML related: {result.get('is_ai_ml_related', 'N/A')}")
            print(f"✓ Complexity: {result['complexity']}")
            print(f"✓ k_text: {result['k_text']}")
            print(f"✓ Reasoning: {result['reasoning']}")
            
            if result.get('fallback'):
                print("⚠️  FALLBACK USED")
            elif not result.get('is_ai_ml_related'):
                print("⚠️  OFF-TOPIC DETECTED")
            else:
                success_count += 1
        
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    print("=" * 70)
    print(f"✅ Testing complete! Success rate: {success_count}/{total_count}")
    print("=" * 70)


if __name__ == "__main__":
    test_analyzer()
