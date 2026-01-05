"""
Download papers from arXiv API.

This script downloads AI/ML papers from arXiv and saves them as PDFs.
It focuses on Machine Learning, AI, and Computer Vision categories.

Usage:
    python download_arxiv.py --num-papers 10 --categories cs.LG,cs.AI
    
API:
    download_papers(paper_ids) -> List[str]  # Returns downloaded doc_ids
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import argparse

try:
    import arxiv
except ImportError:
    logging.error("arxiv library not installed. Run: pip install arxiv")
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

DEFAULT_CATEGORIES = ["cs.LG", "cs.AI", "cs.CV"]
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "papers"
DEFAULT_NUM_PAPERS = 10

CURATED_PAPERS = [
    # === LLMs & Transformers (6) ===
    "1810.04805",  # BERT - Bidirectional attention, pre-training
    "2005.14165",  # GPT-3 - Few-shot learning, scaling laws
    "1906.08237",  # XLNet - Permutation language modeling
    "2203.02155",  # InstructGPT - RLHF (Reinforcement Learning from Human Feedback)
    "2001.08361",  # Scaling Laws - Neural language model scaling
    "1907.11692",  # RoBERTa - Robustly Optimized BERT Approach
    
    # === Computer Vision (12) ===
    "1506.02640",  # YOLO - Real-time object detection
    "2010.11929",  # ViT (Vision Transformer) - Patches as tokens
    "1703.06870",  # Mask R-CNN - Instance segmentation
    "1905.11946",  # EfficientNet - Compound scaling
    "2103.14030",  # Swin Transformer - Hierarchical vision transformer
    "1608.06993",  # DenseNet - Densely connected convolutional networks
    "1611.05431",  # ResNeXt - Aggregated residual transformations
    "1409.0473",   # Network in Network - 1x1 convolutions, global average pooling
    "1409.4842",   # Inception (GoogLeNet) - Multi-scale convolutions
    "1505.04597",  # U-Net - Convolutional networks for biomedical image segmentation
    "1607.06450",  # Layer Normalization - Training stabilization
    "1704.04861",  # MobileNets - Efficient CNNs for mobile vision
    
    # === Multimodal & RAG (2) ===
    "2103.00020",  # CLIP - Vision-language alignment
    "2005.11401",  # RAG - Retrieval-Augmented Generation
    
    # === Generative Models (4) ===
    "2006.11239",  # DDPM - Denoising Diffusion Probabilistic Models
    "1406.2661",   # GANs - Generative Adversarial Networks
    "1312.6114",   # VAE - Variational Autoencoders
    "2112.10752",  # Stable Diffusion - High-resolution image synthesis
    
    # === Optimization & Regularization (4) ===
    "1502.03167",  # Batch Normalization
    "1207.0580",   # Dropout
    "2106.09685",  # LoRA - Low-Rank Adaptation
    "1711.05101",  # AdamW - Adam with decoupled weight decay
    
    # === Recurrent & Sequence Models (2) ===
    "1411.1784",   # GRU - Gated Recurrent Units (empirical evaluation)
    "1409.3215",   # Seq2Seq - Sequence to Sequence Learning with Neural Networks
    
    # === Reinforcement Learning (2) ===
    "1312.5602",   # DQN - Playing Atari with Deep RL
    "1707.06347",  # PPO - Proximal Policy Optimization
    
    # === Graph Neural Networks (1) ===
    "1609.02907",  # GCN - Semi-Supervised Classification with Graph Convolutional Networks
]

CURATED_PAPERS_FULL = [
    # Золотий фонд (вже завантажені)
    "1706.03762", "1512.03385", "1409.1556",
    
    # Фундаментальні архітектури
    "1502.03167", "1207.0580", "1406.2661", "1312.6114", "1409.0473", "1411.1784",
    "1810.04805", "2005.14165", "1906.08237", "1704.05526", "2001.08361",
    
    # Computer Vision
    "1506.02640", "1504.08083", "1703.06870", "1905.11946", "2010.11929",
    "1611.05431", "2004.10934", "2103.14030", "1911.09070", "1802.02611",
    
    # Мультимодальність та Генеративні моделі
    "2103.00020", "2006.11239", "2112.10752", "1909.11059",
    
    # Оптимізація, RL та Ефективність
    "1607.06450", "1711.05101", "1707.06347", "2203.02155", "1312.5602",
    "2106.09685", "2305.14314", "2005.11401", "1808.05377", "1706.02515", "1609.02907"
]


def download_curated_papers(output_dir: Path, num_papers: int = 10) -> List[Dict]:
    """
    Download curated list of important AI/ML papers.
    
    Args:
        output_dir: Directory to save PDFs
        num_papers: Number of papers to download
        
    Returns:
        List of paper metadata dictionaries
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    papers_metadata = []
    
    paper_ids = CURATED_PAPERS[:num_papers]
    
    logging.info(f"Downloading {len(paper_ids)} curated papers from arXiv")
    
    for idx, paper_id in enumerate(paper_ids, 1):
        try:
            search = arxiv.Search(id_list=[paper_id])
            paper = next(search.results())

            metadata = {
                "doc_id": f"arxiv_{paper_id.replace('.', '_')}",
                "arxiv_id": paper_id,
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "abstract": paper.summary,
                "published": paper.published.isoformat(),
                "categories": paper.categories,
                "pdf_url": paper.pdf_url,
                "source_type": "arxiv",
                "downloaded_at": datetime.now().isoformat(),
            }

            pdf_path = output_dir / f"{metadata['doc_id']}.pdf"
            
            if pdf_path.exists():
                logging.info(f"  [{idx}/{len(paper_ids)}] Already exists: {paper.title[:60]}...")
            else:
                logging.info(f"  [{idx}/{len(paper_ids)}] Downloading: {paper.title[:60]}...")
                paper.download_pdf(filename=str(pdf_path))
                time.sleep(2)  
            
            metadata["pdf_path"] = str(pdf_path.absolute())
            papers_metadata.append(metadata)
            
        except Exception as e:
            logging.error(f"  Error downloading {paper_id}: {e}")
            continue
    
    return papers_metadata


def search_and_download_papers(
    query: str,
    categories: List[str],
    output_dir: Path,
    max_results: int = 10
) -> List[Dict]:
    """
    Search arXiv and download papers matching query.
    
    Args:
        query: Search query
        categories: List of arXiv categories (e.g., ['cs.LG', 'cs.AI'])
        output_dir: Directory to save PDFs
        max_results: Maximum number of papers to download
        
    Returns:
        List of paper metadata dictionaries
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    papers_metadata = []
    
    category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
    full_query = f"{query} AND ({category_filter})"
    
    logging.info(f"Searching arXiv with query: {full_query}")
    logging.info(f"Max results: {max_results}")

    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    
    for idx, paper in enumerate(search.results(), 1):
        try:
            arxiv_id = paper.get_short_id()
            safe_id = arxiv_id.replace(".", "_").replace("/", "_")

            metadata = {
                "doc_id": f"arxiv_{safe_id}",
                "arxiv_id": arxiv_id,
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "abstract": paper.summary,
                "published": paper.published.isoformat(),
                "categories": paper.categories,
                "pdf_url": paper.pdf_url,
                "source_type": "arxiv",
                "downloaded_at": datetime.now().isoformat(),
            }

            pdf_path = output_dir / f"{metadata['doc_id']}.pdf"
            
            if pdf_path.exists():
                logging.info(f"  [{idx}/{max_results}] Already exists: {paper.title[:60]}...")
            else:
                logging.info(f"  [{idx}/{max_results}] Downloading: {paper.title[:60]}...")
                paper.download_pdf(filename=str(pdf_path))
                time.sleep(3)  
            
            metadata["pdf_path"] = str(pdf_path.absolute())
            papers_metadata.append(metadata)
            
        except Exception as e:
            logging.error(f"  Error downloading paper {idx}: {e}")
            continue
    
    return papers_metadata


def save_metadata(papers_metadata: List[Dict], output_dir: Path):
    """Save papers metadata to JSON file."""
    metadata_path = output_dir / "papers_metadata.json"
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(papers_metadata, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Saved metadata to: {metadata_path}")
    return metadata_path


def download_papers(
    paper_ids: Optional[List[str]] = None,
    mode: str = "curated",
    query: str = "deep learning OR neural network",
    categories: List[str] = None,
    output_dir: Path = None,
    max_papers: int = 10
) -> List[str]:
    """
    Download arXiv papers and return list of doc_ids.
    
    This is the main API function for use by run_pipeline.py.
    
    Args:
        paper_ids: Specific paper IDs to download (for curated mode)
        mode: 'curated' or 'search'
        query: Search query (for search mode)
        categories: arXiv categories (default: cs.LG, cs.AI, cs.CV)
        output_dir: Output directory (default: data/raw/papers)
        max_papers: Maximum papers to download
        
    Returns:
        List of downloaded doc_ids (e.g., ['arxiv_1706_03762', 'arxiv_1512_03385'])
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    if categories is None:
        categories = DEFAULT_CATEGORIES
    
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "curated":
        if paper_ids:
            global CURATED_PAPERS
            original_list = CURATED_PAPERS
            CURATED_PAPERS = paper_ids
            papers_metadata = download_curated_papers(output_dir, len(paper_ids))
            CURATED_PAPERS = original_list
        else:
            papers_metadata = download_curated_papers(output_dir, max_papers)
    else:
        papers_metadata = search_and_download_papers(
            query=query,
            categories=categories,
            output_dir=output_dir,
            max_results=max_papers
        )

    if papers_metadata:
        save_metadata(papers_metadata, output_dir)

    doc_ids = [paper["doc_id"] for paper in papers_metadata]
    logging.info(f"Downloaded {len(doc_ids)} papers: {doc_ids}")
    
    return doc_ids


def main():
    """CLI interface for downloading arXiv papers."""
    parser = argparse.ArgumentParser(
        description="Download AI/ML papers from arXiv"
    )
    parser.add_argument(
        "--num-papers",
        type=int,
        default=DEFAULT_NUM_PAPERS,
        help=f"Number of papers to download (default: {DEFAULT_NUM_PAPERS})"
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(DEFAULT_CATEGORIES),
        help=f"Comma-separated arXiv categories (default: {','.join(DEFAULT_CATEGORIES)})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for PDFs"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["curated", "search"],
        default="curated",
        help="Download mode: 'curated' for famous papers, 'search' for search query"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="deep learning OR neural network",
        help="Search query (only for 'search' mode)"
    )
    parser.add_argument(
        "--paper-ids",
        type=str,
        help="Comma-separated paper IDs (e.g., '1706.03762,1512.03385')"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    categories = args.categories.split(",")
    paper_ids = args.paper_ids.split(",") if args.paper_ids else None

    print("=" * 70)
    print("📚 arXiv Papers Downloader - AI/ML Course Assistant")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Output directory: {output_dir}")
    print(f"Number of papers: {args.num_papers}")
    if paper_ids:
        print(f"Paper IDs: {paper_ids}")
    print("=" * 70)
    print()

    doc_ids = download_papers(
        paper_ids=paper_ids,
        mode=args.mode,
        query=args.query,
        categories=categories,
        output_dir=output_dir,
        max_papers=args.num_papers
    )

    if doc_ids:
        print()
        print("=" * 70)
        print(f"✅ Successfully downloaded {len(doc_ids)} papers!")
        print("=" * 70)
        print()
        print("📊 Summary:")
        print(f"  - PDFs saved to: {output_dir}")
        print(f"  - Metadata saved to: {output_dir / 'papers_metadata.json'}")
        print(f"  - Doc IDs: {', '.join(doc_ids)}")
        print()
        print("🔜 Next steps:") 
        print("  1. Review downloaded papers")
        print("  2. Run: python run_pipeline.py process --doc-ids <ids>")
        print("=" * 70)
    else:
        print()
        print("⚠️  No papers were downloaded. Check your query or try again later.")


if __name__ == "__main__":
    main()
