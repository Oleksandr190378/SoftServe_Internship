"""
RAG Answer Generator with Groq LLM.

Generates grounded answers with citations from retrieved context.
Enforces topic validation and "I don't know" behavior.
"""

import logging
import re
from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# OpenAI GPT-5 Nano configuration
MODEL_NAME = "gpt-5-nano"
TEMPERATURE = 0.1  # Low for grounded, factual answer10
MAX_TOKENS = 15000  # Reasoning (up to 8000) + answer (up to 4000) (model supports up to 128K)
REASONING_EFFORT = "medium"  # Balance between speed and quality (can use up to 8000 tokens for reasoning)

# System prompt enforcing grounding and citations
SYSTEM_PROMPT = """You are an expert AI/ML course assistant specializing in deep learning and machine learning.

CRITICAL RULES:
1. TOPIC VALIDATION: You can ONLY answer questions about:
   - Deep Learning (neural networks, CNNs, RNNs, Transformers, etc.)
   - Machine Learning (algorithms, optimization, training, etc.)
   - Computer Vision, NLP, model architectures
   
   If query is NOT about AI/ML/DL → respond EXACTLY:
   "I can only assist with deep learning and machine learning topics. Please ask about neural networks, architectures, training methods, or related AI/ML concepts."

2. GROUNDED ANSWERS ONLY:
   - Use ONLY information from the provided context
   - NEVER use external knowledge or make assumptions
   - If context doesn't contain answer → respond EXACTLY:
   "I don't have enough information in my knowledge base to answer this question. The retrieved context doesn't contain relevant details about [topic]."

3. ANSWER LENGTH & STRUCTURE:
   **Target length based on question complexity:**
   - Simple definitions/concepts: ~150-200 words
   - Architectural explanations: ~200-280 words  
   - Complex mechanisms/comparisons: ~280-350 words
   
   **Write naturally in flowing prose (3 paragraphs). DO NOT label paragraphs.**
   
   Start with the core concept and key formula. Then explain technical implementation details with specific dimensions and parameters. Finally present experimental evidence and figure analysis. Connect ideas smoothly without section markers.
   
   **For visual queries (show, diagram, illustrate, architecture):**
   - Focus on describing what's visible in figures: layout, components, connections, data flow
   - Reference figure parts: "left side shows X, right side shows Y"
   - Use formulas sparingly unless specifically asked for mathematical details
   
   **FORBIDDEN:**
   - ❌ "Paragraph 1:", "Paragraph 2:", "Para X:", "First paragraph:", "Opening:", "Middle:", "Closing:"
   - ❌ Any numbered or labeled sections in the answer
   - ❌ Bullet points or lists
   - ❌ Redundant summary paragraphs ("Hence...", "Thus...", "In conclusion...")
   
   **REQUIRED:**
   - ✅ Write answer as continuous prose starting immediately with content
   - ✅ 3 natural paragraphs with smooth transitions
   - ✅ Each paragraph ~80-100 words
   - ✅ Integrate figures naturally: "as shown in Figure 2 [A]"
   - ✅ Cite sources inline: [1], [2], [A]

4. MANDATORY CITATIONS:
   - For text sources: use [1], [2], [3] based on chunk numbers
   - For images: use [A], [B], [C] based on image labels
   
   **Figure references (IMPORTANT):**
   - ALWAYS mention Figure number: "Figure 2 [A]", "Figure 7 [B]"
   - For composite figures with multiple parts: "Figure 2 [A] (left side shows X, right side shows Y)"
   - If same figure used multiple times: consistent labeling (e.g., [A] for Figure 2, even if retrieved twice)
   - Integrate figure references naturally in explanation, not as separate note
   
   **Citation density:**
   - Every major claim needs [1], [2], etc.
   - Reference figures when explaining visual concepts
   - Example: "Multi-head attention uses h parallel heads [1], as shown in Figure 2 [A] where the right side illustrates concatenation."

5. IMAGE USAGE:
   - HIGH confidence (1.0) = explicitly referenced → use freely
   - MEDIUM confidence (0.6-0.7) = semantically verified → use but acknowledge
   - LOW confidence (0.5+) = fallback → use sparingly, mention uncertainty
   
   When using MEDIUM/LOW: "Based on the retrieved diagram [A]..." or "The figure [A] suggests..."

6. RESPONSE FORMAT:
   Answer: <structured answer with inline citations>
   
   Sources: <list all citations: [1], [2], [A]>
   
   Reasoning: <1-2 sentences: which sources used, confidence levels, any sources ignored and why>

EXAMPLE GOOD RESPONSE:
Answer: Residual connections are skip connections that add the input x directly to the output of stacked layers, resulting in x + F(x) [1]. Each residual block uses identity shortcuts bypassing 3×3 convolutions without adding extra parameters, where the shortcut is connected to each pair of 3×3 layers [1].

This design solves the degradation problem where very deep plain networks have higher training error as depth increases [2]. The skip connections provide a direct gradient flow path during backpropagation, stabilizing training. In practice, ResNets use h = 3n shortcuts for architectures with n residual blocks, with dimensions handled through zero-padding or projections when needed [1][3].

Empirically, 110-layer ResNets achieve lower training and testing error than plain counterparts on CIFAR-10 [3]. Figure 7 [A] demonstrates this through layer response analysis: ResNets maintain stable activations with lower standard deviation across layers compared to plain networks, confirming improved optimization dynamics. This enables substantially deeper models (110+ layers) with better performance [2][3].

Sources: [1], [2], [3], [A]

Reasoning: Used chunks 1-3 for technical details and CIFAR-10 results. Image A (MEDIUM, 0.82) shows layer stability supporting optimization claims.
"""


class RAGGenerator:
    """
    Answer generator for RAG pipeline.
    
    Features:
    - OpenAI GPT-5 Nano integration (400K context, 128K output, reasoning support)
    - Grounded answer generation with mandatory citations
    - Topic validation (AI/ML only)
    - "I don't know" behavior when context insufficient
    - Confidence-aware image usage
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize generator with OpenAI GPT-5 Nano.
        
        Args:
            openai_api_key: Optional API key (uses .env if not provided)
        """
        logging.info(f"Initializing RAGGenerator with {MODEL_NAME}")
        
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            reasoning_effort=REASONING_EFFORT,
            api_key=openai_api_key
        )
        
        self.system_prompt = SYSTEM_PROMPT
        logging.info("Generator ready")
    
    def format_context_for_llm(self, llm_input: Dict) -> str:
        """
        Format retrieval results into structured context for LLM.
        
        Creates numbered text sources [1], [2], [3] and lettered image sources [A], [B], [C].
        
        Args:
            llm_input: Output from retriever.prepare_for_llm()
                {query, text_chunks, images, metadata}
        
        Returns:
            Formatted context string with labels
        """
        context_parts = []
        
        # Header
        context_parts.append(f"Query: {llm_input['query']}\n")
        
        # Text chunks with numbered labels
        if llm_input['text_chunks']:
            context_parts.append("=" * 70)
            context_parts.append("TEXT CONTEXT:")
            context_parts.append("=" * 70)
            
            for i, chunk in enumerate(llm_input['text_chunks'], 1):
                chunk_text = (
                    f"\n[{i}] chunk_id: {chunk['chunk_id']}\n"
                    f"    Source: {chunk['source']} (page {chunk['page']})\n"
                    f"    Has figure references: {chunk['has_figure_references']}\n"
                )
                
                if chunk['related_image_ids']:
                    chunk_text += f"    Related images: {', '.join(chunk['related_image_ids'])}\n"
                
                chunk_text += f"    Text: {chunk['text']}\n"
                context_parts.append(chunk_text)
        
        # Images with lettered labels
        if llm_input['images']:
            context_parts.append("\n" + "=" * 70)
            context_parts.append("IMAGE CONTEXT:")
            context_parts.append("=" * 70)
            
            for i, img in enumerate(llm_input['images'], 1):
                label = chr(64 + i)  # A, B, C...
                
                # Extract Figure number from caption if present
                import re
                caption = img['caption']
                figure_match = re.search(r'Figure \d+', caption)
                figure_label = f" ({figure_match.group()})" if figure_match else ""
                
                img_text = (
                    f"\n[{label}]{figure_label} image_id: {img['image_id']}\n"
                    f"    Filename: {img['filename']}\n"
                    f"    Page: {img['page']}\n"
                    f"    Confidence: {img['confidence']} (similarity: {img['similarity']})\n"
                    f"    Reason: {img['reason']}\n"
                    f"    Caption: {img['caption']}\n"
                )
                context_parts.append(img_text)
        
        # Metadata summary
        context_parts.append("\n" + "=" * 70)
        context_parts.append("METADATA:")
        context_parts.append("=" * 70)
        meta = llm_input['metadata']
        context_parts.append(
            f"Total text chunks: {meta['num_text_chunks']}\n"
            f"Total images: {meta['num_images']}\n"
            f"  HIGH confidence: {meta['high_confidence_images']}\n"
            f"  MEDIUM confidence: {meta['medium_confidence_images']}\n"
            f"  LOW confidence: {meta['low_confidence_images']}\n"
        )
        
        # Instructions
        context_parts.append("\n" + "=" * 70)
        context_parts.append("INSTRUCTIONS:")
        context_parts.append("=" * 70)
        context_parts.append("""
1. Check if query is about deep learning/machine learning
2. Analyze if the provided context is sufficient to answer
3. Generate answer using ONLY the context above
4. Use citations: [1], [2] for text chunks; [A], [B] for images
5. Explain your reasoning (which sources used, why)
6. If using MEDIUM/LOW confidence images, acknowledge uncertainty
""")
        
        return "\n".join(context_parts)
    
    def parse_response(self, response_text: str, llm_input: Dict) -> Dict:
        """
        Parse LLM response into structured output.
        
        Expected format:
            Answer: <text with citations>
            Sources: [1], [2], [A]
            Reasoning: <explanation>
        
        Args:
            response_text: Raw LLM response
            llm_input: Original input for validation
        
        Returns:
            Dict with answer, citations, reasoning, flags
        """
        # Extract sections using regex
        answer_match = re.search(r'Answer:\s*(.+?)(?=\n\s*Sources:|$)', response_text, re.DOTALL)
        sources_match = re.search(r'Sources:\s*(.+?)(?=\n\s*Reasoning:|$)', response_text, re.DOTALL)
        reasoning_match = re.search(r'Reasoning:\s*(.+?)$', response_text, re.DOTALL)
        
        answer = answer_match.group(1).strip() if answer_match else response_text.strip()
        sources_text = sources_match.group(1).strip() if sources_match else ""
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Extract chunk citations [1], [2], [3]
        chunk_citations = re.findall(r'\[(\d+)\]', sources_text)
        chunk_ids = []
        for idx in chunk_citations:
            idx_int = int(idx) - 1
            if 0 <= idx_int < len(llm_input['text_chunks']):
                chunk_ids.append(llm_input['text_chunks'][idx_int]['chunk_id'])
        
        # Extract image citations [A], [B], [C]
        image_citations = re.findall(r'\[([A-Z])\]', sources_text)
        image_ids = []
        for letter in image_citations:
            idx = ord(letter) - 65  # A=0, B=1, C=2
            if 0 <= idx < len(llm_input['images']):
                image_ids.append(llm_input['images'][idx]['image_id'])
        
        # Check for off-topic or insufficient context responses
        answer_lower = answer.lower()
        is_off_topic = "only assist with deep learning and machine learning" in answer_lower
        is_insufficient = "don't have enough information" in answer_lower or "doesn't contain relevant" in answer_lower
        
        return {
            'answer': answer,
            'cited_chunks': chunk_ids,
            'cited_images': image_ids,
            'sources_text': sources_text,
            'reasoning': reasoning,
            'is_off_topic': is_off_topic,
            'is_insufficient_context': is_insufficient,
            'raw_response': response_text
        }
    
    def generate(self, llm_input: Dict) -> Dict:
        """
        Generate answer from retrieval results.
        
        Full pipeline:
        1. Format context for LLM
        2. Create messages (system + user)
        3. Call Groq LLM
        4. Parse response
        5. Return structured result
        
        Args:
            llm_input: Output from retriever.prepare_for_llm()
        
        Returns:
            Dict with answer, citations, reasoning, metadata
        """
        logging.info(f"Generating answer for query: '{llm_input['query']}'")
        
        # Format context
        formatted_context = self.format_context_for_llm(llm_input)
        
        # Log context size
        context_chars = len(formatted_context)
        context_tokens_estimate = context_chars // 4  # Rough estimate
        logging.info(f"Context size: {context_chars} chars (~{context_tokens_estimate} tokens)")
        
        # Build messages
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=formatted_context)
        ]
        
        # Call LLM
        logging.info("Calling OpenAI GPT-5 Nano...")
        try:
            response = self.llm.invoke(messages)
            
            # Extract content (handle AIMessage structure)
            if hasattr(response, 'content'):
                response_text = response.content or ""
            else:
                response_text = str(response)
            
            logging.info(f"LLM response received ({len(response_text)} chars)")
            
            # Log metadata if available
            if hasattr(response, 'response_metadata'):
                token_usage = response.response_metadata.get('token_usage', {})
                logging.info(f"Tokens: {token_usage.get('total_tokens', 'unknown')} "
                           f"(reasoning: {token_usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)})")
            
        except Exception as e:
            logging.error(f"Error calling LLM: {e}")
            response_text = f"Error generating response: {str(e)}"
        
        # Parse response
        result = self.parse_response(response_text, llm_input)
        
        # Add metadata
        result['query'] = llm_input['query']
        result['num_chunks_retrieved'] = llm_input['metadata']['num_text_chunks']
        result['num_images_retrieved'] = llm_input['metadata']['num_images']
        
        # Log result
        if result['is_off_topic']:
            logging.warning("Query flagged as off-topic")
        elif result['is_insufficient_context']:
            logging.warning("Insufficient context to answer")
        else:
            logging.info(f"Answer generated: {len(result['cited_chunks'])} chunks, {len(result['cited_images'])} images cited")
        
        return result


def test_generator():
    """Test generator with different scenarios."""
    from retriever import MultimodalRetriever
    
    print("=" * 70)
    print("🧪 Testing RAG Generator")
    print("=" * 70)
    print()
    
    # Initialize
    retriever = MultimodalRetriever()
    generator = RAGGenerator()
    
    # Test scenarios
    test_queries = [
        ("show encoder decoder architecture", "Visual query (should use images)"),
        ("explain gradient descent algorithm", "Conceptual query (no images expected)"),
        ("how to cook pasta", "Off-topic query (should reject)")
    ]
    
    for query, description in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Query: {query}")
        print(f"Description: {description}")
        print(f"{'=' * 70}")
        
        # Retrieve
        text_chunks, verified_images = retriever.retrieve_with_verification(query, k_text=3)
        llm_input = retriever.prepare_for_llm(query, text_chunks, verified_images)
        
        print(f"\nRetrieved: {len(text_chunks)} chunks, {len(verified_images)} images")
        
        # Generate
        result = generator.generate(llm_input)
        
        # Display
        print(f"\n📝 Answer:")
        print(result['answer'])
        
        if result['is_off_topic']:
            print("\n⚠️  OFF-TOPIC DETECTED")
        elif result['is_insufficient_context']:
            print("\n⚠️  INSUFFICIENT CONTEXT")
        else:
            print(f"\n📚 Sources: {result['sources_text']}")
            print(f"\n💭 Reasoning: {result['reasoning']}")
            print(f"\n✅ Cited chunks: {result['cited_chunks']}")
            print(f"✅ Cited images: {result['cited_images']}")
        
        print()
    
    print("=" * 70)
    print("✅ Testing complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_generator()
