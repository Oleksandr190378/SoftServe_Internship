"""
Download articles from Medium and TowardsDataScience.

This script downloads practical ML/DL articles focusing on industry use cases,
best practices, common pitfalls, and real-world insights.

Usage:
    python download_medium.py --num-articles 7
    
API:
    download_articles(article_urls) -> List[str]  # Returns downloaded doc_ids
"""

import os
import json
import time
import logging
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urlparse
import argparse

try:
    from bs4 import BeautifulSoup
except ImportError:
    logging.error("BeautifulSoup not installed. Run: pip install beautifulsoup4")
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "medium"
DEFAULT_NUM_ARTICLES = 10

# Top-10 curated Medium/TDS articles - VERIFIED URLs from recent publications (Jan 2026)
# Coverage: RAG evaluation, chunking, agents, transformers, agentic AI, LLM optimization, ethics
# Focus: Direct relevance to multimodal RAG systems with practical insights
CURATED_ARTICLES = [
    {
        "url": "https://towardsdatascience.com/doing-evals-on-a-bloated-rag-pipeline/",
        "slug": "running-evals-rag-pipeline",
        "title": "Running Evals on a Bloated RAG Pipeline",
        "topic": "RAG Evaluation",
        "description": "Comparing metrics across datasets and models - 71 min deep dive!"
    },
    {
        "url": "https://towardsdatascience.com/chunk-size-as-an-experimental-variable-in-rag-systems/",
        "slug": "chunk-size-rag-systems",
        "title": "Chunk Size as an Experimental Variable in RAG Systems",
        "topic": "RAG Chunking",
        "description": "Understanding retrieval by experimenting with chunk sizes - our exact problem!"
    },
    {
        "url": "https://towardsdatascience.com/how-agents-plan-tasks-with-to-do-lists/",
        "slug": "agents-plan-tasks",
        "title": "How Agents Plan Tasks with To-Do Lists",
        "topic": "Agentic AI",
        "description": "Agentic planning and task management in LangChain"
    },
    {
        "url": "https://towardsdatascience.com/production-ready-llms-made-simple-with-nemo-agent-toolkit/",
        "slug": "production-llms-nemo",
        "title": "Production-Ready LLMs Made Simple with the NeMo Agent Toolkit",
        "topic": "Production LLMs",
        "description": "From simple chat to multi-agent reasoning and REST APIs"
    },
    {
        "url": "https://towardsdatascience.com/the-machine-learning-advent-calendar-day-24-transformers-for-text-in-excel/",
        "slug": "transformers-text-excel",
        "title": "Transformers for Text: Self-Attention Explained",
        "topic": "Transformers",
        "description": "Step-by-step look at self-attention and word embeddings"
    },
    {
        "url": "https://towardsdatascience.com/why-map-and-mrr-fail-for-search-ranking-and-what-to-use-instead/",
        "slug": "map-mrr-search-ranking",
        "title": "Why MAP and MRR Fail for Search Ranking",
        "topic": "Search Metrics",
        "description": "NDCG, ERR for graded relevance - better than MRR for RAG!"
    },
    {
        "url": "https://towardsdatascience.com/understanding-vibe-proving-part-1/",
        "slug": "vibe-proving-llms",
        "title": "Understanding Vibe Proving",
        "topic": "LLM Reasoning",
        "description": "Making LLMs reason with verifiable step-by-step logic"
    },
    {
        "url": "https://towardsdatascience.com/the-geometry-of-laziness-what-angles-reveal-about-ai-hallucinations/",
        "slug": "geometry-ai-hallucinations",
        "title": "The Geometry of Laziness: AI Hallucinations",
        "topic": "LLM Reliability",
        "description": "What angles reveal about hallucinations - critical for RAG trust"
    },
    {
        "url": "https://towardsdatascience.com/understanding-the-generative-ai-user/",
        "slug": "generative-ai-user",
        "title": "Understanding the Generative AI User",
        "topic": "UX & Product",
        "description": "What regular users think about AI - important for course assistant UX"
    },
    {
        "url": "https://towardsdatascience.com/the-machine-learning-advent-calendar-bonus-2-gradient-descent-variants-in-excel/",
        "slug": "gradient-descent-variants",
        "title": "Gradient Descent Variants: Momentum, RMSProp, Adam",
        "topic": "Optimization",
        "description": "Visual comparison of optimization algorithms"
    }
]


def download_article(url: str, slug: str, output_dir: Path) -> Optional[Dict]:
    """
    Download and parse a single Medium/TDS article.
    
    Args:
        url: Article URL
        slug: Short identifier for the article
        output_dir: Directory to save content
        
    Returns:
        Article metadata dictionary or None if failed
    """
    try:
        # Send GET request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # TowardsDataScience uses different selectors than classic Medium
        # Try multiple selectors to find main content
        article_body = None
        
        # Try 1: article tag (standard Medium)
        article_body = soup.find('article')
        
        # Try 2: main content div (TDS/WordPress)
        if not article_body:
            article_body = soup.find('div', class_='article-content')
        
        # Try 3: post-content class (WordPress-based TDS)
        if not article_body:
            article_body = soup.find('div', class_='post-content')
            
        # Try 4: entry-content (common WordPress class)
        if not article_body:
            article_body = soup.find('div', class_='entry-content')
        
        # Try 5: Just get all paragraphs if nothing else works
        if not article_body:
            logging.warning(f"  Using fallback: extracting all paragraphs from {url}")
            article_body = soup.find('body')
        
        if not article_body:
            logging.error(f"  Could not find article in {url}")
            return None
        
        # Extract title
        title_elem = soup.find('h1')
        title = title_elem.get_text(strip=True) if title_elem else slug
        
        # Extract paragraphs (clean out spam sections)
        # Remove "Recommended stories", "Follow the author", comments sections
        spam_keywords = ['recommended', 'follow', 'subscribe', 'sign up', 'get started', 'read more stories']
        
        paragraphs = article_body.find_all('p')
        clean_paragraphs = []
        for p in paragraphs:
            text = p.get_text(strip=True).lower()
            # Skip spam sections
            if any(keyword in text for keyword in spam_keywords) and len(text) < 100:
                continue
            if p.get_text(strip=True):  # Non-empty
                clean_paragraphs.append(p.get_text(strip=True))
        
        text_content = '\n\n'.join(clean_paragraphs)
        
        # Extract code blocks
        code_blocks = article_body.find_all(['pre', 'code'])
        code_content = []
        for idx, code in enumerate(code_blocks, 1):
            code_text = code.get_text(strip=True)
            if code_text and len(code_text) > 10:  # Skip very short snippets
                code_content.append({
                    'index': idx,
                    'code': code_text
                })
        
        # Extract headings (for structure)
        headings = article_body.find_all(['h1', 'h2', 'h3', 'h4'])
        structure = []
        for heading in headings:
            level = heading.name
            text = heading.get_text(strip=True)
            if text:
                structure.append({'level': level, 'text': text})
        
        # Extract images with captions (CRITICAL for multimodal RAG!)
        images = article_body.find_all('img')
        image_list = []
        for idx, img in enumerate(images, 1):
            img_url = img.get('src', '')
            alt_text = img.get('alt', '')
            
            # Try to find image caption (Medium uses figcaption)
            caption = ''
            parent_figure = img.find_parent('figure')
            if parent_figure:
                figcaption = parent_figure.find('figcaption')
                if figcaption:
                    caption = figcaption.get_text(strip=True)
            
            if img_url and ('medium.com' in img_url or 'githubusercontent.com' in img_url):
                image_list.append({
                    'index': idx,
                    'url': img_url,
                    'alt_text': alt_text,
                    'caption': caption  # Image caption is gold for multimodal RAG!
                })
        
        # Create metadata
        metadata = {
            'doc_id': f'medium_{slug}',
            'url': url,
            'slug': slug,
            'title': title,
            'source_type': 'medium',
            'downloaded_at': datetime.now().isoformat(),
            'content': {
                'text': text_content,
                'structure': structure,
                'code_blocks': code_content,
                'images': image_list
            },
            'stats': {
                'text_length': len(text_content),
                'code_blocks': len(code_content),
                'images': len(image_list),
                'headings': len(structure)
            }
        }
        
        # Save as JSON
        article_dir = output_dir / slug
        article_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = article_dir / f"{slug}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logging.info(f"  Saved: {json_path}")
        
        return metadata
        
    except Exception as e:
        logging.error(f"  Error downloading {url}: {e}")
        return None


def download_curated_articles(output_dir: Path, num_articles: int = 7) -> List[Dict]:
    """
    Download curated list of Medium/TDS articles.
    
    Args:
        output_dir: Directory to save articles
        num_articles: Number of articles to download
        
    Returns:
        List of article metadata dictionaries
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    articles_metadata = []
    
    articles = CURATED_ARTICLES[:num_articles]
    
    logging.info(f"Downloading {len(articles)} curated Medium/TDS articles")
    
    for idx, article in enumerate(articles, 1):
        url = article['url']
        slug = article['slug']
        title = article['title']
        
        # Check if already exists
        article_dir = output_dir / slug
        json_path = article_dir / f"{slug}.json"
        
        if json_path.exists():
            logging.info(f"  [{idx}/{len(articles)}] Already exists: {title[:60]}...")
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                articles_metadata.append(metadata)
        else:
            logging.info(f"  [{idx}/{len(articles)}] Downloading: {title[:60]}...")
            metadata = download_article(url, slug, output_dir)
            if metadata:
                articles_metadata.append(metadata)
            time.sleep(3)  # Be extra polite with Medium rate limiting
    
    return articles_metadata


def save_summary_metadata(articles_metadata: List[Dict], output_dir: Path):
    """Save summary metadata for all articles."""
    metadata_path = output_dir / "articles_metadata.json"
    
    summary = {
        'total_articles': len(articles_metadata),
        'downloaded_at': datetime.now().isoformat(),
        'articles': [
            {
                'doc_id': article['doc_id'],
                'title': article['title'],
                'url': article['url'],
                'topic': next((a['topic'] for a in CURATED_ARTICLES if a['slug'] == article['slug']), 'Unknown'),
                'stats': article['stats']
            }
            for article in articles_metadata
        ]
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Saved summary metadata to: {metadata_path}")
    return metadata_path


def download_articles(
    article_urls: Optional[List[str]] = None,
    output_dir: Path = None,
    max_articles: int = 7
) -> List[str]:
    """
    Download Medium/TDS articles and return list of doc_ids.
    
    This is the main API function for use by run_pipeline.py.
    
    Args:
        article_urls: Specific article URLs to download (optional)
        output_dir: Output directory (default: data/raw/medium)
        max_articles: Maximum articles to download
        
    Returns:
        List of downloaded doc_ids (e.g., ['medium_learning_rates', 'medium_cnn_guide'])
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download curated articles
    articles_metadata = download_curated_articles(output_dir, max_articles)
    
    # Save summary metadata
    if articles_metadata:
        save_summary_metadata(articles_metadata, output_dir)
    
    # Return list of doc_ids for tracking
    doc_ids = [article['doc_id'] for article in articles_metadata]
    logging.info(f"Downloaded {len(doc_ids)} articles: {doc_ids}")
    
    return doc_ids


def main():
    """CLI interface for downloading Medium/TDS articles."""
    parser = argparse.ArgumentParser(
        description="Download ML/DL articles from Medium and TowardsDataScience"
    )
    parser.add_argument(
        "--num-articles",
        type=int,
        default=DEFAULT_NUM_ARTICLES,
        help=f"Number of articles to download (default: {DEFAULT_NUM_ARTICLES})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for articles"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    # CLI output
    print("=" * 70)
    print("📰 Medium/TDS Articles Downloader - AI/ML Course Assistant")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Number of articles: {args.num_articles}")
    print("=" * 70)
    print()
    
    # Use API function
    doc_ids = download_articles(
        output_dir=output_dir,
        max_articles=args.num_articles
    )

    if doc_ids:
        print()
        print("=" * 70)
        print(f"✅ Successfully downloaded {len(doc_ids)} articles!")
        print("=" * 70)
        print()
        print("📊 Summary:")
        print(f"  - Articles saved to: {output_dir}")
        print(f"  - Metadata saved to: {output_dir / 'articles_metadata.json'}")
        print(f"  - Doc IDs: {', '.join(doc_ids)}")
        print()
        print("📚 Topics covered:")
        for article in CURATED_ARTICLES[:args.num_articles]:
            print(f"  - {article['topic']}: {article['title'][:50]}...")
        print()
        print("🔜 Next steps:")
        print("  1. Review downloaded articles")
        print("  2. Run: python run_pipeline.py process --doc-ids <ids>")
        print("=" * 70)
    else:
        print()
        print("⚠️  No articles were downloaded. Check your connection or try again later.")


if __name__ == "__main__":
    main()
