"""
Shared utility functions for ingest scripts.

Extracted common logic to follow DRY principle and reduce code duplication
across download_arxiv.py, download_realpython.py, and download_medium.py.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime


def save_articles_metadata(
    articles_metadata: List[Dict],
    output_dir: Path,
    curated_articles: List[Dict],
    filename: str = "articles_metadata.json"
) -> Path:
    """
    Save summary metadata for articles (RealPython/Medium).
    
    Builds a summary with total count, download timestamp, and article details.
    Extracts topic from curated_articles list by matching slug.
    
    Args:
        articles_metadata: List of article metadata dicts with full details
        output_dir: Directory to save metadata JSON
        curated_articles: List of curated articles with topic information
        filename: Output filename (default: articles_metadata.json)
    
    Returns:
        Path to saved metadata file
    """
    metadata_path = output_dir / filename
    
    summary = {
        'total_articles': len(articles_metadata),
        'downloaded_at': datetime.now().isoformat(),
        'articles': [
            {
                'doc_id': article['doc_id'],
                'title': article['title'],
                'url': article['url'],
                'topic': next(
                    (a['topic'] for a in curated_articles if a['slug'] == article['slug']),
                    'Unknown'
                ),
                'stats': article['stats']
            }
            for article in articles_metadata
        ]
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Saved summary metadata to: {metadata_path}")
    return metadata_path


def save_papers_metadata(
    papers_metadata: List[Dict],
    output_dir: Path,
    filename: str = "papers_metadata.json"
) -> Path:
    """
    Save papers metadata for arXiv papers.
    
    Simpler than articles - just saves the full metadata list.
    
    Args:
        papers_metadata: List of paper metadata dicts
        output_dir: Directory to save metadata JSON
        filename: Output filename (default: papers_metadata.json)
    
    Returns:
        Path to saved metadata file
    """
    metadata_path = output_dir / filename
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(papers_metadata, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Saved metadata to: {metadata_path}")
    return metadata_path
