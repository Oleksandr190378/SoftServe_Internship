"""
Streamlit UI for AI/ML Course Assistant - Multimodal RAG System.

Features:
- Query input with visual feedback
- Answer display with citations
- Text sources panel with metadata
- Image gallery with confidence badges
- Debug view for retrieval inspection
"""

import streamlit as st
from pathlib import Path
import sys
import json

# Add parent directory to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag.retriever import MultimodalRetriever
from rag.generator import RAGGenerator

# Page config
st.set_page_config(
    page_title="AI/ML Course Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
IMAGES_DIR = BASE_DIR / "data" / "processed" / "images"
IMAGES_METADATA_FILE = BASE_DIR / "data" / "processed" / "images_metadata.json"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .confidence-badge-high {
        background-color: #4CAF50;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .confidence-badge-medium {
        background-color: #2196F3;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .confidence-badge-low {
        background-color: #FF9800;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .citation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_retriever():
    """Load retriever (cached)."""
    return MultimodalRetriever()


@st.cache_resource
def load_generator():
    """Load generator (cached)."""
    return RAGGenerator()


@st.cache_data
def load_images_metadata():
    """Load images metadata for file path mapping."""
    if IMAGES_METADATA_FILE.exists():
        with open(IMAGES_METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def get_image_path(image_id: str) -> Path:
    """Get image file path from image_id.
    
    Handles multiple image_id formats:
    - PDF: arxiv_1706_03762_embedded_001 -> paper_id: arxiv_1706_03762
    - PDF: arxiv_1409_3215_vector_006_01 -> paper_id: arxiv_1409_3215
    - JSON: realpython_numpy-tutorial_web_004 -> paper_id: realpython_numpy-tutorial
    - JSON: medium_agents-plan-tasks_web_001 -> paper_id: medium_agents-plan-tasks
    """
    # Extract paper_id by removing suffix patterns
    # Pattern 1: _embedded_XXX (PDFs - raster images)
    if '_embedded_' in image_id:
        paper_id = image_id.rsplit('_embedded_', 1)[0]
    # Pattern 2: _vector_XXX (PDFs - vector images)
    elif '_vector_' in image_id:
        paper_id = image_id.rsplit('_vector_', 1)[0]
    # Pattern 3: _web_XXX (JSON sources)
    elif '_web_' in image_id:
        paper_id = image_id.rsplit('_web_', 1)[0]
    # Fallback: use full image_id
    else:
        paper_id = image_id
    
    # Try PNG in paper subfolder
    png_path = IMAGES_DIR / paper_id / f"{image_id}.png"
    if png_path.exists():
        return png_path
    
    # Try JPG in paper subfolder
    jpg_path = IMAGES_DIR / paper_id / f"{image_id}.jpg"
    if jpg_path.exists():
        return jpg_path
    
    # Fallback: try direct path (backward compatibility)
    png_path_direct = IMAGES_DIR / f"{image_id}.png"
    if png_path_direct.exists():
        return png_path_direct
    
    jpg_path_direct = IMAGES_DIR / f"{image_id}.jpg"
    if jpg_path_direct.exists():
        return jpg_path_direct
    
    return None


def get_confidence_badge_html(confidence: str, similarity: float) -> str:
    """Generate HTML badge for confidence level."""
    if confidence == 'HIGH':
        badge_class = 'confidence-badge-high'
        icon = '🟢'
    elif confidence == 'MEDIUM':
        badge_class = 'confidence-badge-medium'
        icon = '🔵'
    else:
        badge_class = 'confidence-badge-low'
        icon = '🟠'
    
    return f'<span class="{badge_class}">{icon} {confidence} ({similarity:.3f})</span>'


def display_answer_section(result: dict, llm_input: dict = None):
    """Display answer with citations and inline cited images."""
    st.markdown("### 📝 Answer")
    
    if result['is_off_topic']:
        st.warning("⚠️ **Off-topic Query**")
        st.info(result['answer'])
        return
    
    if result['is_insufficient_context']:
        st.warning("⚠️ **Insufficient Context**")
        st.info(result['answer'])
        return
    
    # Main answer
    st.markdown(result['answer'])
    
    # Display cited images inline (right after answer)
    if result['cited_images'] and llm_input and llm_input['images']:
        st.markdown("---")
        st.markdown("#### 🖼️ Referenced Images")
        
        # Filter only cited images
        cited_imgs = [img for img in llm_input['images'] if img['image_id'] in result['cited_images']]
        
        # Display in grid (2 columns for better visibility)
        cols = st.columns(min(2, len(cited_imgs)))
        
        for i, img_data in enumerate(cited_imgs):
            img_id = img_data['image_id']
            
            with cols[i % len(cols)]:
                # Get image path
                img_path = get_image_path(img_id)
                
                if img_path and img_path.exists():
                    # Image label (extract figure number from caption if exists)
                    label_letter = chr(65 + i)  # A, B, C...
                    fig_match = None
                    if 'Figure' in img_data['caption']:
                        import re
                        fig_match = re.search(r'Figure \d+', img_data['caption'])
                    
                    if fig_match:
                        st.markdown(f"**[{label_letter}] {fig_match.group()}**")
                    else:
                        st.markdown(f"**[{label_letter}] Image {i+1}**")
                    
                    # Display image (fixed width for better layout)
                    st.image(str(img_path), width=400)
                    
                    # Metadata
                    st.caption(f"Page {img_data['page']}")
                    
                    # Short caption preview
                    with st.expander("📖 Image Description"):
                        st.text_area(
                            "Caption",
                            value=img_data['caption'],
                            height=150,
                            key=f"answer_caption_{img_id}",
                            label_visibility="collapsed"
                        )
                else:
                    st.error(f"⚠️ Image not found: {img_id}")
    
    # Citations summary
    if result['cited_chunks'] or result['cited_images']:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Text Sources Cited", len(result['cited_chunks']))
        with col2:
            st.metric("Images Cited", len(result['cited_images']))


def display_sources_section(result: dict, llm_input: dict):
    """Display text sources with metadata."""
    st.markdown("### 📚 Text Sources")
    
    if not llm_input['text_chunks']:
        st.info("No text sources retrieved.")
        return
    
    for i, chunk in enumerate(llm_input['text_chunks'], 1):
        chunk_id = chunk['chunk_id']
        is_cited = chunk_id in result['cited_chunks']
        
        # Citation marker
        cite_marker = "✅ **CITED**" if is_cited else "Retrieved"
        
        with st.expander(f"[{i}] {chunk_id} - {cite_marker}", expanded=is_cited):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Source:** {chunk['source']}")
                st.markdown(f"**Page:** {chunk['page']}")
            
            with col2:
                if chunk['has_figure_references']:
                    st.success("📊 Has figure refs")
                if chunk['related_image_ids']:
                    st.info(f"🖼️ {len(chunk['related_image_ids'])} related images")
            
            # Chunk text
            st.markdown("**Text:**")
            st.text_area(
                label="Chunk content",
                value=chunk['text'],
                height=200,
                key=f"chunk_{i}",
                label_visibility="collapsed"
            )


def display_images_section(result: dict, llm_input: dict):
    """Display images with confidence badges."""
    st.markdown("### 🖼️ Images")
    
    if not llm_input['images']:
        st.info("No images retrieved.")
        return
    
    images_metadata = load_images_metadata()
    
    # Create grid
    cols = st.columns(3)
    
    for i, img_data in enumerate(llm_input['images']):
        img_id = img_data['image_id']
        is_cited = img_id in result['cited_images']
        
        with cols[i % 3]:
            # Get image path
            img_path = get_image_path(img_id)
            
            if img_path and img_path.exists():
                # Image label
                label_letter = chr(65 + i)  # A, B, C...
                
                # Display image (smaller in grid)
                st.image(str(img_path), width=250)
                
                # Citation marker with label
                if is_cited:
                    st.success(f"✅ **[{label_letter}] CITED**")
                else:
                    st.info(f"[{label_letter}] Retrieved but not used")
                
                # Metadata
                st.caption(f"**ID:** {img_id}")
                st.caption(f"**Page:** {img_data['page']}")
                
                # Expand for full caption
                with st.expander("📖 Image Description"):
                    st.markdown(f"**Reason:** {img_data['reason']}")
                    st.text_area(
                        "Full caption",
                        value=img_data['caption'],
                        height=200,
                        key=f"caption_{img_id}",
                        label_visibility="collapsed"
                    )
            else:
                st.error(f"⚠️ Image file not found: {img_id}")


def display_debug_section(result: dict, llm_input: dict):
    """Display debug information."""
    st.markdown("### 🐛 Debug Information")
    
    # Retrieval mode
    num_images = llm_input['metadata']['num_images']
    if num_images > 0:
        retrieval_mode = f"Multimodal ({num_images} images verified)"
    else:
        retrieval_mode = "Text-only (no images found)"
    
    st.info(f"**Retrieval Mode:** {retrieval_mode}")
    
    # Metadata
    with st.expander("📊 Retrieval Metadata", expanded=False):
        st.json(llm_input['metadata'])
    
    # Reasoning
    if result.get('reasoning'):
        with st.expander("💭 LLM Reasoning", expanded=True):
            st.markdown(result['reasoning'])
    
    # Raw response
    with st.expander("🔍 Raw LLM Response", expanded=False):
        st.code(result.get('raw_response', ''), language='text')
    
    # Similarity scores table with confidence badges
    if llm_input['images']:
        with st.expander("📈 Image Confidence & Similarity Scores", expanded=False):
            for img in llm_input['images']:
                cited = "✅" if img['image_id'] in result['cited_images'] else "❌"
                
                # Create columns for better layout
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.markdown(f"**{img['image_id']}**")
                
                with col2:
                    badge_html = get_confidence_badge_html(
                        img['confidence'],
                        img['similarity']
                    )
                    st.markdown(badge_html, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"{cited} Cited" if cited == "✅" else "Not used")
                
                st.markdown("---")


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<div class="main-header">🤖 AI/ML Course Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ask questions about Deep Learning & Machine Learning</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar settings
    with st.sidebar:
        st.title("⚙️ Settings")
        
        st.markdown("---")
        st.markdown("### Retrieval Settings")
        k_text = st.slider(
            "Number of text chunks",
            min_value=2,
            max_value=5,
            value=3,
            help="How many text chunks to retrieve"
        )
        
        st.markdown("---")
        st.markdown("### Display Settings")
        show_debug = st.checkbox(
            "Show Debug View",
            value=False,
            help="Display retrieval metadata and LLM reasoning"
        )
        
        st.markdown("---")
        st.markdown("### Sample Queries")
        st.markdown("""
        - `show encoder decoder architecture`
        - `explain residual connections in ResNet`
        - `what is attention mechanism`
        - `compare VGG and ResNet`
        - `how does multi-head attention work`
        """)
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        **Multimodal RAG System**
        
        - 🔍 Retrieves text + images
        - 🤖 OpenAI GPT-5 Nano
        - 📊 Confidence-based verification
        - 🎯 Citation-grounded answers
        """)
    
    # Main query interface
    query = st.text_input(
        "Your Question:",
        placeholder="e.g., Show the Transformer architecture",
        help="Ask about deep learning concepts, architectures, or methods"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        submit = st.button("🔍 Ask", type="primary", width="stretch")
    with col2:
        if st.button("🗑️ Clear", width="stretch"):
            st.rerun()
    
    # Process query
    if submit and query:
        try:
            # Load models
            retriever = load_retriever()
            generator = load_generator()
            
            # Retrieval
            with st.spinner("🔍 Retrieving relevant content..."):
                text_chunks, verified_images = retriever.retrieve_with_verification(
                    query=query,
                    k_text=k_text
                )
                llm_input = retriever.prepare_for_llm(query, text_chunks, verified_images)
            
            # Generation
            with st.spinner("💭 Generating answer..."):
                result = generator.generate(llm_input)
            
            # Display results
            st.markdown("---")
            
            # Answer section with inline cited images
            display_answer_section(result, llm_input)
            
            st.markdown("---")
            
            # Sources and images in columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                display_sources_section(result, llm_input)
            
            with col2:
                display_images_section(result, llm_input)
            
            # Debug section
            if show_debug:
                st.markdown("---")
                display_debug_section(result, llm_input)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    elif submit and not query:
        st.warning("⚠️ Please enter a question.")


if __name__ == "__main__":
    main()
