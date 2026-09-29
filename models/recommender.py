"""
News Article Recommendation Engine
CSC4352 Final Project

This module implements the core recommendation logic using
content-based filtering with semantic embeddings.
"""

import numpy as np
from typing import List, Dict, Optional, Set
from pathlib import Path
from dataclasses import dataclass
import json

from .embeddings import EmbeddingGenerator
from .vector_db import VectorDatabase, SearchResult


@dataclass
class Recommendation:
    """A recommended article with metadata."""
    article_id: str
    title: str
    content: str
    category: str
    source: str
    similarity_score: float
    published_date: str = ""
    url: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'article_id': self.article_id,
            'title': self.title,
            'content': self.content,
            'category': self.category,
            'source': self.source,
            'similarity_score': self.similarity_score,
            'published_date': self.published_date,
            'url': self.url
        }


class RecommendationEngine:
    """
    Content-based recommendation engine for news articles.
    
    This engine uses transformer embeddings and FAISS similarity search
    to recommend articles based on semantic similarity.
    
    Algorithm:
    1. User reads/selects an article
    2. Get the article's embedding
    3. Query FAISS for k most similar articles
    4. Filter out already-read articles
    5. Apply diversity and recency adjustments
    6. Return top-N recommendations
    
    Attributes:
        embedding_generator: Transformer model for generating embeddings
        vector_db: FAISS index for similarity search
        articles_db: Dictionary mapping article_id to article data
    """
    
    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        embedding_dim: int = 768,
        device: Optional[str] = None
    ):
        """
        Initialize the recommendation engine.
        
        Args:
            model_name: Pretrained transformer model name
            embedding_dim: Embedding dimension
            device: Device for model inference
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(
            model_name=model_name,
            device=device,
            pooling_strategy='mean'
        )
        
        self.vector_db = VectorDatabase(
            dimension=embedding_dim,
            index_type='flat_ip'
        )
        
        # Article data storage
        self.articles_db: Dict[str, Dict] = {}
        
        # User reading history
        self.user_history: Set[str] = set()
        
        # Embeddings cache
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        
        print("Recommendation engine initialized!")
    
    def index_articles(self, articles: List[Dict], batch_size: int = 16):
        """
        Index a collection of articles for recommendation.
        
        This generates embeddings for all articles and adds them
        to the FAISS index.
        
        Args:
            articles: List of article dictionaries with required fields
            batch_size: Batch size for embedding generation
        """
        print(f"Indexing {len(articles)} articles...")
        
        # Store article data
        for article in articles:
            article_id = article['article_id']
            self.articles_db[article_id] = article
        
        # Prepare texts for embedding
        texts = []
        article_ids = []
        
        for article in articles:
            # Combine title and content for richer embeddings
            text = f"{article['title']} [SEP] {article['content']}"
            texts.append(text)
            article_ids.append(article['article_id'])
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings_batch(
            texts,
            batch_size=batch_size,
            normalize=True,
            show_progress=True
        )
        
        # Cache embeddings
        for article_id, embedding in zip(article_ids, embeddings):
            self._embeddings_cache[article_id] = embedding
        
        # Add to FAISS index
        self.vector_db.add_embeddings(embeddings, article_ids)
        
        print(f"Successfully indexed {len(articles)} articles!")
    
    def get_recommendations(
        self,
        article_id: str,
        n_recommendations: int = 10,
        exclude_read: bool = True,
        diversity_weight: float = 0.1,
        category_filter: Optional[str] = None
    ) -> List[Recommendation]:
        """
        Get article recommendations based on a seed article.
        
        Args:
            article_id: ID of the article to base recommendations on
            n_recommendations: Number of recommendations to return
            exclude_read: Whether to exclude articles in user history
            diversity_weight: Weight for source diversity (0-1)
            category_filter: Optional category to filter results
            
        Returns:
            List of Recommendation objects
        """
        # Get seed article embedding
        if article_id in self._embeddings_cache:
            query_embedding = self._embeddings_cache[article_id]
        else:
            # Generate embedding if not cached
            article = self.articles_db.get(article_id)
            if not article:
                raise ValueError(f"Article {article_id} not found")
            
            text = f"{article['title']} [SEP] {article['content']}"
            query_embedding = self.embedding_generator.generate_embedding(text)
            self._embeddings_cache[article_id] = query_embedding
        
        # Determine articles to exclude
        exclude_ids = [article_id]  # Always exclude the seed article
        if exclude_read:
            exclude_ids.extend(list(self.user_history))
        
        # Query FAISS for similar articles (get extra for filtering)
        search_results = self.vector_db.search(
            query_embedding,
            k=n_recommendations * 3,  # Get extra results for filtering
            exclude_ids=exclude_ids
        )
        
        # Build recommendations with diversity scoring
        recommendations = []
        seen_sources = set()
        
        for result in search_results:
            article = self.articles_db.get(result.article_id)
            if not article:
                continue
            
            # Category filter
            if category_filter and article.get('category') != category_filter:
                continue
            
            # Adjust score for source diversity
            adjusted_score = result.score
            if article.get('source') in seen_sources:
                adjusted_score *= (1 - diversity_weight)
            
            recommendation = Recommendation(
                article_id=result.article_id,
                title=article['title'],
                content=article['content'],
                category=article.get('category', 'Unknown'),
                source=article.get('source', 'Unknown'),
                similarity_score=adjusted_score,
                published_date=article.get('published_date', ''),
                url=article.get('url', '')
            )
            
            recommendations.append(recommendation)
            seen_sources.add(article.get('source'))
            
            if len(recommendations) >= n_recommendations:
                break
        
        # Sort by adjusted score
        recommendations.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return recommendations[:n_recommendations]
    
    def search_by_query(
        self,
        query_text: str,
        n_results: int = 10,
        category_filter: Optional[str] = None
    ) -> List[Recommendation]:
        """
        Search for articles similar to a text query.
        
        This allows users to search using natural language queries.
        
        Args:
            query_text: Search query text
            n_results: Number of results to return
            category_filter: Optional category filter
            
        Returns:
            List of Recommendation objects
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        
        # Search
        search_results = self.vector_db.search(
            query_embedding,
            k=n_results * 2
        )
        
        # Build recommendations
        recommendations = []
        
        for result in search_results:
            article = self.articles_db.get(result.article_id)
            if not article:
                continue
            
            if category_filter and article.get('category') != category_filter:
                continue
            
            recommendation = Recommendation(
                article_id=result.article_id,
                title=article['title'],
                content=article['content'],
                category=article.get('category', 'Unknown'),
                source=article.get('source', 'Unknown'),
                similarity_score=result.score,
                published_date=article.get('published_date', ''),
                url=article.get('url', '')
            )
            
            recommendations.append(recommendation)
            
            if len(recommendations) >= n_results:
                break
        
        return recommendations
    
    def mark_as_read(self, article_id: str):
        """Mark an article as read."""
        self.user_history.add(article_id)
    
    def clear_history(self):
        """Clear user reading history."""
        self.user_history.clear()
    
    def get_user_profile_embedding(self) -> Optional[np.ndarray]:
        """
        Generate a user profile embedding based on reading history.
        
        This is the average of all read article embeddings.
        
        Returns:
            User profile embedding or None if no history
        """
        if not self.user_history:
            return None
        
        embeddings = []
        for article_id in self.user_history:
            if article_id in self._embeddings_cache:
                embeddings.append(self._embeddings_cache[article_id])
        
        if not embeddings:
            return None
        
        # Average embeddings
        profile = np.mean(embeddings, axis=0)
        
        # Normalize
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
        
        return profile
    
    def get_personalized_recommendations(
        self,
        n_recommendations: int = 10,
        content_weight: float = 0.5
    ) -> List[Recommendation]:
        """
        Get personalized recommendations based on user profile.
        
        Combines user profile similarity with content diversity.
        
        Args:
            n_recommendations: Number of recommendations
            content_weight: Weight for content similarity vs user profile
            
        Returns:
            List of personalized recommendations
        """
        profile = self.get_user_profile_embedding()
        
        if profile is None:
            # No history - return random diverse articles
            all_articles = list(self.articles_db.values())
            return [
                Recommendation(
                    article_id=a['article_id'],
                    title=a['title'],
                    content=a['content'],
                    category=a.get('category', 'Unknown'),
                    source=a.get('source', 'Unknown'),
                    similarity_score=0.5,
                    published_date=a.get('published_date', ''),
                    url=a.get('url', '')
                )
                for a in all_articles[:n_recommendations]
            ]
        
        # Search using user profile
        search_results = self.vector_db.search(
            profile,
            k=n_recommendations * 2,
            exclude_ids=list(self.user_history)
        )
        
        # Build recommendations
        recommendations = []
        
        for result in search_results:
            article = self.articles_db.get(result.article_id)
            if not article:
                continue
            
            recommendation = Recommendation(
                article_id=result.article_id,
                title=article['title'],
                content=article['content'],
                category=article.get('category', 'Unknown'),
                source=article.get('source', 'Unknown'),
                similarity_score=result.score,
                published_date=article.get('published_date', ''),
                url=article.get('url', '')
            )
            
            recommendations.append(recommendation)
            
            if len(recommendations) >= n_recommendations:
                break
        
        return recommendations
    
    def save(self, directory: Path):
        """Save the engine state to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save vector database
        self.vector_db.save(directory / "vector_db")
        
        # Save articles database
        with open(directory / "articles.json", 'w') as f:
            json.dump(self.articles_db, f)
        
        # Save embeddings cache
        np.savez_compressed(
            directory / "embeddings_cache.npz",
            **{k: v for k, v in self._embeddings_cache.items()}
        )
        
        print(f"Saved engine state to {directory}")
    
    @classmethod
    def load(cls, directory: Path, model_name: str = 'distilbert-base-uncased') -> 'RecommendationEngine':
        """Load engine state from disk."""
        directory = Path(directory)
        
        engine = cls(model_name=model_name)
        
        # Load vector database
        engine.vector_db = VectorDatabase.load(directory / "vector_db")
        
        # Load articles database
        with open(directory / "articles.json", 'r') as f:
            engine.articles_db = json.load(f)
        
        # Load embeddings cache
        cache_data = np.load(directory / "embeddings_cache.npz")
        engine._embeddings_cache = {k: cache_data[k] for k in cache_data.files}
        
        print(f"Loaded engine state from {directory}")
        return engine


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Recommendation Engine")
    print("="*60)
    
    # Import data loader
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.data_loader import load_sample_data
    
    # Load sample data
    dataset = load_sample_data()
    articles = [article.to_dict() for article in dataset]
    
    # Initialize engine
    engine = RecommendationEngine()
    
    # Index articles
    engine.index_articles(articles)
    
    # Get recommendations for first article
    first_article = articles[0]
    print(f"\nGetting recommendations for: {first_article['title']}")
    
    recommendations = engine.get_recommendations(
        first_article['article_id'],
        n_recommendations=5
    )
    
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.title}")
        print(f"   Category: {rec.category}")
        print(f"   Similarity: {rec.similarity_score:.2%}")
    
    # Test search
    print("\n" + "-"*40)
    print("Testing search by query...")
    
    results = engine.search_by_query("artificial intelligence technology", n_results=3)
    
    print("\nSearch results for 'artificial intelligence technology':")
    for i, rec in enumerate(results, 1):
        print(f"\n{i}. {rec.title}")
        print(f"   Similarity: {rec.similarity_score:.2%}")
