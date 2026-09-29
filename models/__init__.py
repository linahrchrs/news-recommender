"""
Model components for the News Article Recommender System
"""

# Import with error handling for missing dependencies
try:
    from .embeddings import EmbeddingGenerator, get_embeddings
except ImportError as e:
    EmbeddingGenerator = None
    get_embeddings = None
    print(f"Note: Embedding module not available - {e}")

try:
    from .vector_db import VectorDatabase, SearchResult, cosine_similarity
except ImportError as e:
    VectorDatabase = None
    SearchResult = None
    cosine_similarity = None
    print(f"Note: Vector DB module not available - {e}")

try:
    from .recommender import RecommendationEngine, Recommendation
except ImportError as e:
    RecommendationEngine = None
    Recommendation = None
    print(f"Note: Recommender module not available - {e}")

try:
    from .clustering import TopicClusterer, ClusterResult, create_cluster_visualization
except ImportError as e:
    TopicClusterer = None
    ClusterResult = None
    create_cluster_visualization = None
    print(f"Note: Clustering module not available - {e}")

__all__ = [
    'EmbeddingGenerator',
    'get_embeddings',
    'VectorDatabase',
    'SearchResult',
    'cosine_similarity',
    'RecommendationEngine',
    'Recommendation',
    'TopicClusterer',
    'ClusterResult',
    'create_cluster_visualization'
]
