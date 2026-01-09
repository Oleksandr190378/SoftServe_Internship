"""
Download tutorials from RealPython.com.

This script downloads practical ML/DL tutorials with code examples and visualizations.
Focuses on implementation details, best practices, and Python-specific guides.

Usage:
    python download_realpython.py --num-articles 10
    
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

from ingest.utils import save_articles_metadata

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "realpython"
DEFAULT_NUM_ARTICLES = 10

CURATED_ARTICLES = [
    {
        "url": "https://realpython.com/python-ai-neural-network/",
        "slug": "python-ai-neural-network",
        "title": "Python AI: How to Build a Neural Network & Make Predictions",
        "topic": "Neural Networks Basics",
        "description": "Forward/Backward propagation, activation functions, from scratch implementation"
    },
    {
        "url": "https://realpython.com/generative-adversarial-networks/",
        "slug": "generative-adversarial-networks",
        "title": "Generative Adversarial Networks: Build Your First Models",
        "topic": "GANs",
        "description": "Generator vs Discriminator, MNIST generation, training visualization"
    },
    {
        "url": "https://realpython.com/python-keras-text-classification/",
        "slug": "python-keras-text-classification",
        "title": "Practical Text Classification With Python and Keras",
        "topic": "NLP",
        "description": "Word embeddings, CNN for text, overfitting detection"
    },
    {
        "url": "https://realpython.com/pytorch-vs-tensorflow/",
        "slug": "pytorch-vs-tensorflow",
        "title": "PyTorch vs TensorFlow for Your Python Deep Learning Project",
        "topic": "Frameworks",
        "description": "Ecosystem comparison, API differences, use cases"
    },
    {
        "url": "https://realpython.com/gradient-descent-algorithm-python/",
        "slug": "gradient-descent-algorithm-python",
        "title": "Stochastic Gradient Descent Algorithm With Python and NumPy",
        "topic": "Optimization",
        "description": "Loss landscapes, gradient descent visualization, convergence"
    },
    {
        "url": "https://realpython.com/face-recognition-with-python/",
        "slug": "face-recognition-with-python",
        "title": "Build Your Own Face Recognition Tool With Python",
        "topic": "Computer Vision",
        "description": "Face detection, bounding boxes, face embeddings"
    },
    {
        "url": "https://realpython.com/python-machine-learning/",
        "slug": "python-machine-learning",
        "title": "Python Machine Learning: Scikit-Learn Tutorial",
        "topic": "Classical ML",
        "description": "Traditional ML algorithms, model training, evaluation metrics"
    },
    {
        "url": "https://realpython.com/image-processing-with-the-python-pillow-library/",
        "slug": "image-processing-pillow",
        "title": "Image Processing With the Python Pillow Library",
        "topic": "Image Processing",
        "description": "Image manipulation, filters, preprocessing for ML/DL"
    },
    {
        "url": "https://realpython.com/pandas-python-explore-dataset/",
        "slug": "pandas-explore-dataset",
        "title": "Pandas for Data Science: A Practical Guide",
        "topic": "Data Preprocessing",
        "description": "Data loading, cleaning, transformation for ML workflows"
    },
    {
        "url": "https://realpython.com/numpy-tutorial/",
        "slug": "numpy-tutorial",
        "title": "NumPy Tutorial: Your First Steps Into Data Science",
        "topic": "NumPy Fundamentals",
        "description": "Arrays, vectorization, mathematical operations for ML"
    }
]


def download_article(url: str, slug: str, output_dir: Path) -> Optional[Dict]:
    """
    Download and parse a single RealPython article.
    
    Args:
        url: Article URL
        slug: Short identifier for the article
        output_dir: Directory to save content
        
    Returns:
        Article metadata dictionary or None if failed
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        article_body = soup.find(class_="article-body")
        if not article_body:
            logging.error(f"  Could not find article-body in {url}")
            return None

        title_elem = soup.find('h1')
        title = title_elem.get_text(strip=True) if title_elem else slug

        paragraphs = article_body.find_all('p')
        text_content = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

        code_blocks = article_body.find_all('pre')
        code_content = []
        for idx, code in enumerate(code_blocks, 1):
            code_text = code.get_text(strip=True)
            language = 'python'  
            code_content.append({
                'index': idx,
                'language': language,
                'code': code_text
            })

        headings = article_body.find_all(['h2', 'h3', 'h4'])
        structure = []
        for heading in headings:
            level = heading.name
            text = heading.get_text(strip=True)
            structure.append({'level': level, 'text': text})

        images = article_body.find_all('img')
        image_list = []
        for idx, img in enumerate(images, 1):
            img_url = img.get('src', '')
            alt_text = img.get('alt', '')
            if img_url:
                image_list.append({
                    'index': idx,
                    'url': img_url,
                    'alt_text': alt_text
                })

        metadata = {
            'doc_id': f'realpython_{slug}',
            'url': url,
            'slug': slug,
            'title': title,
            'source_type': 'realpython',
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


def download_curated_articles(output_dir: Path, num_articles: int = 10) -> List[Dict]:
    """
    Download curated list of RealPython tutorials.
    
    Args:
        output_dir: Directory to save articles
        num_articles: Number of articles to download
        
    Returns:
        List of article metadata dictionaries
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    articles_metadata = []
    
    articles = CURATED_ARTICLES[:num_articles]
    
    logging.info(f"Downloading {len(articles)} curated RealPython tutorials")
    
    for idx, article in enumerate(articles, 1):
        url = article['url']
        slug = article['slug']
        title = article['title']
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
            time.sleep(2)  
    
    return articles_metadata


def download_articles(
    article_urls: Optional[List[str]] = None,
    output_dir: Path = None,
    max_articles: int = 10
) -> List[str]:
    """
    Download RealPython articles and return list of doc_ids.
    
    This is the main API function for use by run_pipeline.py.
    
    Args:
        article_urls: Specific article URLs to download (optional)
        output_dir: Output directory (default: data/raw/realpython)
        max_articles: Maximum articles to download
        
    Returns:
        List of downloaded doc_ids (e.g., ['realpython_neural_network', 'realpython_gans'])
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)

    articles_metadata = download_curated_articles(output_dir, max_articles)

    if articles_metadata:
        save_articles_metadata(articles_metadata, output_dir)

    doc_ids = [article['doc_id'] for article in articles_metadata]
    logging.info(f"Downloaded {len(doc_ids)} articles: {doc_ids}")
    
    return doc_ids


def main():
    """CLI interface for downloading RealPython tutorials."""
    parser = argparse.ArgumentParser(
        description="Download practical ML/DL tutorials from RealPython"
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

    print("=" * 70)
    print("🐍 RealPython Tutorials Downloader - AI/ML Course Assistant")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Number of articles: {args.num_articles}")
    print("=" * 70)
    print()

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
