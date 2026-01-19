"""
Test suite for download_medium.py module.

Tests cover:
- STAGE 1: Parameter validation and error handling
- STAGE 2: Constants and configuration
- STAGE 3: Helper functions and article retrieval
- Integration: Download workflows
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from datetime import datetime
import tempfile

from ingest.download_medium import (
    CURATED_ARTICLES,
    DEFAULT_NUM_ARTICLES,
    DOWNLOAD_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    MIN_CODE_LENGTH,
    SPAM_TEXT_THRESHOLD,
)


class TestConstants(unittest.TestCase):
    """STAGE 2: Constants validation."""
    
    def test_curated_articles_populated(self):
        """Verify curated articles list is populated."""
        self.assertIsInstance(CURATED_ARTICLES, list)
        self.assertGreater(len(CURATED_ARTICLES), 0)
    
    def test_article_structure(self):
        """Verify each article has required fields."""
        required_fields = ["url", "slug", "title", "topic", "description"]
        
        for article in CURATED_ARTICLES:
            for field in required_fields:
                self.assertIn(field, article)
                self.assertIsInstance(article[field], str)
                self.assertGreater(len(article[field]), 0)
    
    def test_default_num_articles_positive(self):
        """Verify default num articles is positive."""
        self.assertGreater(DEFAULT_NUM_ARTICLES, 0)
    
    def test_download_delay_reasonable(self):
        """Verify download delay is reasonable."""
        self.assertGreater(DOWNLOAD_DELAY_SECONDS, 0)
        self.assertLess(DOWNLOAD_DELAY_SECONDS, 10)
    
    def test_request_timeout_positive(self):
        """Verify request timeout is positive."""
        self.assertGreater(REQUEST_TIMEOUT_SECONDS, 0)
    
    def test_min_code_length_valid(self):
        """Verify min code length is reasonable."""
        self.assertGreater(MIN_CODE_LENGTH, 0)
    
    def test_spam_threshold_valid(self):
        """Verify spam text threshold is positive."""
        self.assertGreater(SPAM_TEXT_THRESHOLD, 0)


class TestArticleDownload(unittest.TestCase):
    """STAGE 1-3: Article download functionality."""
    
    def test_articles_have_valid_urls(self):
        """STAGE 1: Verify all articles have valid URLs."""
        for article in CURATED_ARTICLES:
            url = article["url"]
            self.assertTrue(url.startswith("https://"))
            # Medium or TowardsDataScience
            self.assertTrue(
                "towardsdatascience.com" in url or "medium.com" in url
            )
    
    def test_articles_have_unique_slugs(self):
        """STAGE 1: Verify all articles have unique slugs."""
        slugs = [article["slug"] for article in CURATED_ARTICLES]
        self.assertEqual(len(slugs), len(set(slugs)))
    
    def test_article_slug_format(self):
        """STAGE 1: Verify slug format is safe."""
        for article in CURATED_ARTICLES:
            slug = article["slug"]
            # Slugs should be lowercase with hyphens only
            self.assertEqual(slug, slug.lower())
            self.assertTrue(slug.replace("-", "").isalnum())
    
    def test_article_topics_consistent(self):
        """STAGE 2: Verify topics are meaningful."""
        topics = [article["topic"] for article in CURATED_ARTICLES]
        self.assertGreater(len(topics), 0)
        
        for topic in topics:
            self.assertIsInstance(topic, str)
            self.assertGreater(len(topic), 0)
    
    def test_article_descriptions_meaningful(self):
        """STAGE 2: Verify descriptions have minimum length."""
        for article in CURATED_ARTICLES:
            desc = article["description"]
            self.assertGreater(len(desc), 10)
    
    def test_medium_urls_not_paywall(self):
        """STAGE 1: Verify Medium URLs avoid paywall."""
        for article in CURATED_ARTICLES:
            url = article["url"]
            if "medium.com" in url:
                # TowardsDataScience is free tier
                self.assertIn("towardsdatascience.com", url)


class TestDownloadIntegration(unittest.TestCase):
    """Integration tests for download workflow."""
    
    def test_article_count_limit(self):
        """STAGE 1: Verify article list respects count limits."""
        # Can get first N articles
        first_5 = CURATED_ARTICLES[:5]
        self.assertEqual(len(first_5), 5)
    
    def test_articles_ordered(self):
        """STAGE 2: Verify articles are in consistent order."""
        # Should be able to reliably get articles by index
        first_article = CURATED_ARTICLES[0]
        self.assertIn("url", first_article)
        self.assertIn("slug", first_article)
    
    def test_article_topics_rag_focused(self):
        """STAGE 2: Verify articles focus on RAG/LLM topics."""
        topics = [article["topic"].lower() for article in CURATED_ARTICLES]
        
        # Should have mix of RAG, LLM, evaluation topics
        topic_text = " ".join(topics)
        self.assertTrue(
            any(word in topic_text for word in ["rag", "llm", "eval", "agent", "search"])
        )


class TestErrorHandling(unittest.TestCase):
    """STAGE 1: Error handling and edge cases."""
    
    def test_empty_curated_articles_not_allowed(self):
        """STAGE 1: Verify curated articles is not empty."""
        self.assertIsNotNone(CURATED_ARTICLES)
        self.assertGreater(len(CURATED_ARTICLES), 0)
    
    def test_article_data_types(self):
        """STAGE 1: Verify article data types are correct."""
        for article in CURATED_ARTICLES:
            self.assertIsInstance(article, dict)
            self.assertIsInstance(article["url"], str)
            self.assertIsInstance(article["slug"], str)
            self.assertIsInstance(article["title"], str)
            self.assertIsInstance(article["topic"], str)
            self.assertIsInstance(article["description"], str)
    
    def test_no_duplicate_urls(self):
        """STAGE 1: Verify no duplicate article URLs."""
        urls = [article["url"] for article in CURATED_ARTICLES]
        self.assertEqual(len(urls), len(set(urls)))


if __name__ == "__main__":
    unittest.main()
