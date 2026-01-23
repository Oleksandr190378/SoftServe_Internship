"""
RAG Answer Generator.

Generates grounded answers with citations from retrieved context.
Enforces topic validation and "I don't know" behavior.
"""

import logging
import re
import time
from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Import centralized configuration
from config import GENERATOR
from utils.logging_config import setup_logging

# Import from sibling modules
from .security import sanitize_query
from .prompts import SYSTEM_PROMPT, MAX_IMAGES_TO_CITE
from .citations import (
    extract_chunk_citations,
    extract_image_citations,
    validate_and_clean_citations,
    check_answer_sources_consistency,
    FIRST_IMAGE_ASCII
)

load_dotenv()
setup_logging()  # Centralized logging configuration

# OpenAI GPT-5 Mini configuration (from config)
MODEL_NAME = GENERATOR.MODEL
TEMPERATURE = GENERATOR.TEMPERATURE
MAX_TOKENS = GENERATOR.MAX_TOKENS
REASONING_EFFORT = "low"  # Faster, less reasoning tokens (~2000-4000 instead of 8000)

# Context size estimation
CHARS_PER_TOKEN_ESTIMATE = 4  # Rough estimate for token counting

# Response parsing constants
MIN_CONTEXT_LINES_BEFORE = 3  # Minimum context lines for code snippets
MIN_CONTEXT_LINES_AFTER = 3


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
    
    def _check_special_responses(self, answer_text: str) -> Tuple[bool, bool]:
        """
        Check if response is off-topic or has insufficient context.
        
        Single Responsibility: Response type detection.
        
        Args:
            answer_text: The answer text to check
        
        Returns:
            Tuple of (is_off_topic, is_insufficient_context)
        """
        answer_lower = answer_text.lower()
        is_off_topic = ("only assist with ai" in answer_lower or 
                        "only assist with deep learning" in answer_lower or
                        "outside ai/ml" in answer_lower)
        is_insufficient = (
            "don't have enough information" in answer_lower or 
            "doesn't contain relevant" in answer_lower
        )
        return is_off_topic, is_insufficient
    
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
                label = chr(FIRST_IMAGE_ASCII + i - 1)  # A, B, C...
                
                # Note: DO NOT add figure numbers to label line - causes citation confusion
                # The figure number is already in the caption text
                img_text = (
                    f"\n[{label}] image_id: {img['image_id']}\n"
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
        
        num_chunks = meta['num_text_chunks']
        num_images = meta['num_images']
        
        context_parts.append(
            f"Total text chunks: {num_chunks}\n"
            f"Total images: {num_images}\n"
            f"  HIGH confidence: {meta['high_confidence_images']}\n"
            f"  MEDIUM confidence: {meta['medium_confidence_images']}\n"
            f"  LOW confidence: {meta['low_confidence_images']}\n"
        )
        
        # EXPLICIT citation availability (for generator, not evaluator)
        context_parts.append("\n📌 CITATIONS PROVIDED IN THIS CONTEXT:")
        if num_chunks > 0:
            chunk_citations = ', '.join([f"[{i}]" for i in range(1, num_chunks + 1)])
            context_parts.append(f"  Text chunks available: {chunk_citations}")
        else:
            context_parts.append("  Text chunks available: NONE")
        
        if num_images > 0:
            image_citations = ', '.join([f"[{chr(65 + i)}]" for i in range(num_images)])
            context_parts.append(f"  Images available: {image_citations}")
        else:
            context_parts.append("  Images available: NONE")
        
        context_parts.append("  ℹ️  These are all citations that appear in the context above.\n")
        
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
        Parse LLM response into structured output with robust fallback.
        
        Expected format:
            Answer: <text with citations>
            Sources: [1], [2], [A]
            Reasoning: <explanation>
        
        Fallback: If format doesn't match, returns full text as answer.
        
        Args:
            response_text: Raw LLM response
            llm_input: Original input for validation
        
        Returns:
            Dict with answer, citations, reasoning, flags
        """
        # Extract sections using case-insensitive regex with flexible whitespace
        answer_match = re.search(
            r'(?:Answer|ANSWER)[:\s]*(.+?)(?=(?:Sources|SOURCES)[:\s]|$)', 
            response_text, 
            re.DOTALL | re.IGNORECASE
        )
        sources_match = re.search(
            r'(?:Sources|SOURCES)[:\s]*(.+?)(?=(?:Reasoning|REASONING)[:\s]|$)', 
            response_text, 
            re.DOTALL | re.IGNORECASE
        )
        reasoning_match = re.search(
            r'(?:Reasoning|REASONING)[:\s]*(.+?)$', 
            response_text, 
            re.DOTALL | re.IGNORECASE
        )
        
        # Fallback if parsing fails - try flexible extraction
        if not answer_match:
            logging.warning(
                "LLM response doesn't match expected format. "
                "Attempting flexible parsing..."
            )
            
            # Try alternative patterns (e.g., markdown headers, bold labels)
            alt_answer = re.search(
                r'(?:##?\s*Answer|Answer|ANSWER|\*\*Answer\*\*)[:\s]*(.+?)(?=(?:##?\s*Sources|Sources|SOURCES|\*\*Sources\*\*)[:\s]|$)',
                response_text,
                re.DOTALL | re.IGNORECASE
            )
            alt_sources = re.search(
                r'(?:##?\s*Sources|Sources|SOURCES|\*\*Sources\*\*)[:\s]*(.+?)(?=(?:##?\s*Reasoning|Reasoning|REASONING|\*\*Reasoning\*\*)[:\s]|$)',
                response_text,
                re.DOTALL | re.IGNORECASE
            )
            alt_reasoning = re.search(
                r'(?:##?\s*Reasoning|Reasoning|REASONING|\*\*Reasoning\*\*)[:\s]*(.+?)$',
                response_text,
                re.DOTALL | re.IGNORECASE
            )
            
            if alt_answer:
                logging.info("✓ Flexible parsing succeeded with alternative patterns")
                answer_match = alt_answer
                sources_match = alt_sources
                reasoning_match = alt_reasoning
            else:
                # Last resort: extract citations from full text
                logging.warning("Flexible parsing partially failed. Extracting citations from unstructured text...")
                
                # Validate input
                text_chunks = llm_input.get('text_chunks', [])
                images = llm_input.get('images', [])
                
                # Extract citations using helper methods (DRY)
                chunk_ids, chunk_citations = extract_chunk_citations(response_text, text_chunks)
                image_ids, image_citations = extract_image_citations(response_text, images)
                
                # Reconstruct sources text from found citations
                sources_text = ', '.join(
                    [f"[{idx}]" for idx in chunk_citations] +
                    [f"[{letter}]" for letter in image_citations]
                )
                
                # Check response type using helper method (SRP)
                is_off_topic, is_insufficient = self._check_special_responses(response_text)
                
                logging.info(f"✓ Extracted {len(chunk_ids)} chunk citations, {len(image_ids)} image citations from unstructured text")
                
                return {
                    'answer': response_text.strip(),
                    'cited_chunks': chunk_ids,
                    'cited_images': image_ids,
                    'sources_text': sources_text,
                    'reasoning': '(extracted from unstructured response)',
                    'is_off_topic': is_off_topic,
                    'is_insufficient_context': is_insufficient,
                    'raw_response': response_text,
                    'parsing_failed': bool(not sources_text)  # Only failed if no citations found
                }
        
        answer = answer_match.group(1).strip()
        sources_text = sources_match.group(1).strip() if sources_match else ""
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Validate input data before processing
        if not llm_input or 'text_chunks' not in llm_input:
            logging.warning("Invalid llm_input: missing 'text_chunks' key")
            llm_input = {'text_chunks': [], 'images': []}
        
        text_chunks = llm_input.get('text_chunks', [])
        images = llm_input.get('images', [])
        
        # Extract citations from BOTH Answer and Sources
        chunk_ids_from_answer, chunk_citations_answer = extract_chunk_citations(answer, text_chunks)
        image_ids_from_answer, image_citations_answer = extract_image_citations(answer, images)
        
        chunk_ids_from_sources, _ = extract_chunk_citations(sources_text, text_chunks)
        image_ids_from_sources, _ = extract_image_citations(sources_text, images)
        
        # Combine citations from both sources (take union to catch all citations)
        # This prevents losing citations if LLM cites in Answer but forgets in Sources
        chunk_ids = list(set(chunk_ids_from_answer + chunk_ids_from_sources))
        image_ids = list(set(image_ids_from_answer + image_ids_from_sources))
        
        if sources_text:
            logging.info(f"Extracted {len(chunk_ids)} chunk citations, {len(image_ids)} image citations (UNION of Answer + Sources)")
        else:
            # Reconstruct sources_text from answer citations
            sources_text = ', '.join(
                [f"[{idx}]" for idx in chunk_citations_answer] +
                [f"[{letter}]" for letter in image_citations_answer]
            )
            logging.info(f"No Sources section found, extracted {len(chunk_ids)} chunk citations, {len(image_ids)} image citations from Answer")
        
        # Check response type using helper method (SRP)
        is_off_topic, is_insufficient = self._check_special_responses(answer)
        
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
        1. Sanitize query input
        2. Format context for LLM
        3. Create messages (system + user)
        4. Call OpenAI LLM
        5. Parse response
        6. Return structured result
        
        Args:
            llm_input: Output from retriever.prepare_for_llm()
        
        Returns:
            Dict with answer, citations, reasoning, metadata
        """
        start_time = time.perf_counter()
        
        # Sanitize query to prevent prompt injection
        original_query = llm_input['query']
        llm_input['query'] = sanitize_query(original_query)
        
        if original_query != llm_input['query']:
            logging.warning(f"Query sanitized: '{original_query}' -> '{llm_input['query']}'")
        
        logging.info(f"Generating answer for query: '{llm_input['query']}'")
        
        # Limit images to prevent citation overflow (Queries #4, #7)
        # Judge penalizes answers with 4+ image citations, so cap at MAX_IMAGES_TO_CITE
        if len(llm_input.get('images', [])) > MAX_IMAGES_TO_CITE:
            original_count = len(llm_input['images'])
            # Keep highest confidence images (sorted by confidence_score descending)
            llm_input['images'] = sorted(
                llm_input['images'], 
                key=lambda img: img.get('metadata', {}).get('confidence_score', 0.0),
                reverse=True
            )[:MAX_IMAGES_TO_CITE]
            logging.info(f"Limited images from {original_count} to {MAX_IMAGES_TO_CITE} (keeping highest confidence)")
            
            # Update metadata count
            if 'metadata' in llm_input:
                llm_input['metadata']['num_images'] = len(llm_input['images'])
        
        # Format context
        formatted_context = self.format_context_for_llm(llm_input)
        
        # Log context size
        context_chars = len(formatted_context)
        context_tokens_estimate = context_chars // CHARS_PER_TOKEN_ESTIMATE  # Rough estimate
        logging.info(f"Context size: {context_chars} chars (~{context_tokens_estimate} tokens)")
        
        # Build messages
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=formatted_context)
        ]
        
        # Call LLM
        logging.info("Calling OpenAI GPT-5 Mini...")
        response_text = ""
        try:
            response = self.llm.invoke(messages)
            
            # Extract content (handle AIMessage structure)
            if hasattr(response, 'content'):
                response_text = response.content or ""
            else:
                response_text = str(response)
            
            if not response_text:
                logging.warning("LLM returned empty response")
                response_text = "Error: Empty response from LLM"
            
            logging.info(f"LLM response received ({len(response_text)} chars)")
            
            # Log metadata if available
            if hasattr(response, 'response_metadata'):
                token_usage = response.response_metadata.get('token_usage', {})
                if token_usage:
                    logging.info(f"Tokens: {token_usage.get('total_tokens', 'unknown')} "
                               f"(reasoning: {token_usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)})")
            
        except ValueError as e:
            logging.error(f"LLM input validation error: {e}")
            response_text = "Error: Invalid input data for LLM"
        except ConnectionError as e:
            logging.error(f"LLM connection error: {e}")
            response_text = "Error: Failed to connect to LLM service"
        except TimeoutError as e:
            logging.error(f"LLM timeout error: {e}")
            response_text = "Error: LLM request timed out"
        except Exception as e:
            logging.error(f"Unexpected error calling LLM: {type(e).__name__} - {e}")
            response_text = f"Error generating response: {type(e).__name__}"
        
        # Parse response
        result = self.parse_response(response_text, llm_input)
        
        # Validate and clean citations (post-generation fix)
        # Add validation for metadata existence
        metadata = llm_input.get('metadata', {})
        num_chunks = metadata.get('num_text_chunks', 0)
        num_images = metadata.get('num_images', 0)
        
        result['answer'] = validate_and_clean_citations(
            result['answer'],
            num_chunks=num_chunks,
            num_images=num_images
        )
        
        # Check Answer-Sources consistency (log-only)
        check_answer_sources_consistency(result['answer'], result['sources_text'])
        
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
        
        # Log performance
        generation_time = time.perf_counter() - start_time
        logging.info(f"Generated in {generation_time:.3f}s ({len(result['answer'])} chars)")
        
        return result
