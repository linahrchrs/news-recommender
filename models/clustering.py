"""
Topic Clustering and Visualization
CSC4352 Final Project - Advanced Features

This module implements topic clustering using K-means and HDBSCAN,
with visualization using t-SNE and UMAP for dimensionality reduction.

Features:
- K-means clustering for predefined number of topics
- HDBSCAN for automatic topic discovery
- t-SNE visualization of article clusters
- UMAP visualization (faster alternative to t-SNE)
- Interactive Plotly visualizations
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

# Optional imports with fallbacks
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Note: scikit-learn not installed. Install with: pip install scikit-learn")

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Note: HDBSCAN not installed. Install with: pip install hdbscan")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Note: UMAP not installed. Install with: pip install umap-learn")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: Plotly not installed. Install with: pip install plotly")


@dataclass
class ClusterResult:
    """Container for clustering results."""
    labels: np.ndarray  # Cluster label for each article
    n_clusters: int
    cluster_centers: Optional[np.ndarray] = None  # For K-means
    silhouette: Optional[float] = None
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """Get the size of each cluster."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


class TopicClusterer:
    """
    Clusters news articles into topics based on their embeddings.
    
    Supports two clustering algorithms:
    - K-means: When you know the number of topics
    - HDBSCAN: For automatic topic discovery
    """
    
    def __init__(self, embeddings: np.ndarray, article_titles: List[str] = None):
        """
        Initialize the clusterer.
        
        Args:
            embeddings: Article embeddings of shape (n_articles, embedding_dim)
            article_titles: Optional list of article titles for visualization
        """
        self.embeddings = embeddings
        self.article_titles = article_titles or [f"Article {i}" for i in range(len(embeddings))]
        self.labels = None
        self.reduced_embeddings = None
    
    def cluster_kmeans(self, n_clusters: int = 10, random_state: int = 42) -> ClusterResult:
        """
        Cluster articles using K-means algorithm.
        
        K-means partitions articles into exactly n_clusters groups
        by minimizing within-cluster variance.
        
        Args:
            n_clusters: Number of clusters/topics to create
            random_state: Random seed for reproducibility
            
        Returns:
            ClusterResult with labels and cluster centers
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        
        print(f"Clustering {len(self.embeddings)} articles into {n_clusters} topics using K-means...")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        
        labels = kmeans.fit_predict(self.embeddings)
        self.labels = labels
        
        # Calculate silhouette score (measure of cluster quality)
        silhouette = silhouette_score(self.embeddings, labels)
        
        result = ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            cluster_centers=kmeans.cluster_centers_,
            silhouette=silhouette
        )
        
        print(f"Clustering complete! Silhouette score: {silhouette:.3f}")
        print(f"Cluster sizes: {result.get_cluster_sizes()}")
        
        return result
    
    def cluster_hdbscan(
        self,
        min_cluster_size: int = 5,
        min_samples: int = 3
    ) -> ClusterResult:
        """
        Cluster articles using HDBSCAN algorithm.
        
        HDBSCAN automatically determines the number of clusters
        and can identify noise points that don't belong to any cluster.
        
        Args:
            min_cluster_size: Minimum number of articles to form a cluster
            min_samples: Number of samples in neighborhood for core points
            
        Returns:
            ClusterResult with labels (-1 indicates noise)
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan required: pip install hdbscan")
        
        print(f"Clustering {len(self.embeddings)} articles using HDBSCAN...")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean'
        )
        
        labels = clusterer.fit_predict(self.embeddings)
        self.labels = labels
        
        # Count clusters (excluding noise label -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        # Calculate silhouette score (excluding noise points)
        if n_clusters > 1:
            mask = labels != -1
            silhouette = silhouette_score(self.embeddings[mask], labels[mask])
        else:
            silhouette = None
        
        result = ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            silhouette=silhouette
        )
        
        print(f"Found {n_clusters} clusters with {n_noise} noise points")
        if silhouette:
            print(f"Silhouette score: {silhouette:.3f}")
        
        return result
    
    def find_optimal_k(self, k_range: range = range(2, 15)) -> Tuple[int, List[float]]:
        """
        Find optimal number of clusters using silhouette score.
        
        Args:
            k_range: Range of k values to try
            
        Returns:
            Tuple of (optimal_k, list of silhouette scores)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        
        print("Finding optimal number of clusters...")
        
        scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings)
            score = silhouette_score(self.embeddings, labels)
            scores.append(score)
            print(f"  k={k}: silhouette={score:.3f}")
        
        optimal_k = list(k_range)[np.argmax(scores)]
        print(f"\nOptimal k: {optimal_k} (silhouette={max(scores):.3f})")
        
        return optimal_k, scores
    
    def reduce_dimensions_tsne(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Reduce embeddings to 2D/3D using t-SNE for visualization.
        
        t-SNE preserves local structure and creates visually
        meaningful clusters in low dimensions.
        
        Args:
            n_components: Number of dimensions (2 or 3)
            perplexity: Related to number of nearest neighbors
            random_state: Random seed for reproducibility
            
        Returns:
            Reduced embeddings of shape (n_articles, n_components)
        """
        if not TSNE_AVAILABLE:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        
        print(f"Reducing {self.embeddings.shape[1]} dimensions to {n_components} using t-SNE...")
        print("(This may take a few minutes for large datasets)")
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, len(self.embeddings) - 1),
            random_state=random_state,
            n_iter=1000
        )
        
        self.reduced_embeddings = tsne.fit_transform(self.embeddings)
        print("t-SNE reduction complete!")
        
        return self.reduced_embeddings
    
    def reduce_dimensions_umap(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Reduce embeddings to 2D/3D using UMAP for visualization.
        
        UMAP is faster than t-SNE and often produces better
        global structure preservation.
        
        Args:
            n_components: Number of dimensions (2 or 3)
            n_neighbors: Number of neighbors for local structure
            min_dist: Minimum distance between points
            random_state: Random seed for reproducibility
            
        Returns:
            Reduced embeddings of shape (n_articles, n_components)
        """
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn required: pip install umap-learn")
        
        print(f"Reducing {self.embeddings.shape[1]} dimensions to {n_components} using UMAP...")
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(n_neighbors, len(self.embeddings) - 1),
            min_dist=min_dist,
            random_state=random_state
        )
        
        self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        print("UMAP reduction complete!")
        
        return self.reduced_embeddings
    
    def visualize_clusters_2d(
        self,
        labels: np.ndarray = None,
        categories: List[str] = None,
        title: str = "Article Clusters",
        method: str = "umap"
    ):
        """
        Create an interactive 2D visualization of article clusters.
        
        Args:
            labels: Cluster labels (uses self.labels if None)
            categories: Optional category labels for coloring
            title: Plot title
            method: Dimensionality reduction method ('umap' or 'tsne')
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required: pip install plotly")
        
        # Reduce dimensions if not already done
        if self.reduced_embeddings is None:
            if method == "umap" and UMAP_AVAILABLE:
                self.reduce_dimensions_umap(n_components=2)
            elif TSNE_AVAILABLE:
                self.reduce_dimensions_tsne(n_components=2)
            else:
                raise ImportError("Need either umap-learn or scikit-learn for visualization")
        
        # Use provided labels or stored labels
        if labels is None:
            labels = self.labels if self.labels is not None else np.zeros(len(self.embeddings))
        
        # Create color labels
        color_labels = categories if categories else [f"Cluster {l}" for l in labels]
        
        # Create DataFrame for Plotly
        import pandas as pd
        df = pd.DataFrame({
            'x': self.reduced_embeddings[:, 0],
            'y': self.reduced_embeddings[:, 1],
            'cluster': color_labels,
            'title': self.article_titles
        })
        
        # Create interactive scatter plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='cluster',
            hover_data=['title'],
            title=title,
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'}
        )
        
        fig.update_layout(
            width=900,
            height=700,
            template='plotly_dark'
        )
        
        return fig
    
    def visualize_clusters_3d(
        self,
        labels: np.ndarray = None,
        categories: List[str] = None,
        title: str = "Article Clusters (3D)",
        method: str = "umap"
    ):
        """
        Create an interactive 3D visualization of article clusters.
        
        Args:
            labels: Cluster labels
            categories: Optional category labels
            title: Plot title
            method: Dimensionality reduction method
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required: pip install plotly")
        
        # Reduce to 3D
        if method == "umap" and UMAP_AVAILABLE:
            reduced = self.reduce_dimensions_umap(n_components=3)
        elif TSNE_AVAILABLE:
            reduced = self.reduce_dimensions_tsne(n_components=3)
        else:
            raise ImportError("Need umap-learn or scikit-learn")
        
        if labels is None:
            labels = self.labels if self.labels is not None else np.zeros(len(self.embeddings))
        
        color_labels = categories if categories else [f"Cluster {l}" for l in labels]
        
        import pandas as pd
        df = pd.DataFrame({
            'x': reduced[:, 0],
            'y': reduced[:, 1],
            'z': reduced[:, 2],
            'cluster': color_labels,
            'title': self.article_titles
        })
        
        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            color='cluster',
            hover_data=['title'],
            title=title
        )
        
        fig.update_layout(
            width=900,
            height=700,
            template='plotly_dark'
        )
        
        return fig
    
    def get_cluster_topics(
        self,
        labels: np.ndarray,
        top_n: int = 5
    ) -> Dict[int, List[str]]:
        """
        Get representative article titles for each cluster.
        
        Args:
            labels: Cluster labels
            top_n: Number of representative articles per cluster
            
        Returns:
            Dictionary mapping cluster ID to list of representative titles
        """
        topics = {}
        
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise cluster
                continue
            
            # Get articles in this cluster
            mask = labels == cluster_id
            cluster_titles = [self.article_titles[i] for i in range(len(labels)) if mask[i]]
            
            # Return first N titles as representatives
            topics[cluster_id] = cluster_titles[:top_n]
        
        return topics


def create_cluster_visualization(
    embeddings: np.ndarray,
    titles: List[str],
    categories: List[str] = None,
    n_clusters: int = None,
    method: str = "kmeans",
    viz_method: str = "umap"
) -> Tuple[ClusterResult, any]:
    """
    Convenience function to cluster and visualize articles.
    
    Args:
        embeddings: Article embeddings
        titles: Article titles
        categories: Optional category labels
        n_clusters: Number of clusters (None for auto with HDBSCAN)
        method: Clustering method ('kmeans' or 'hdbscan')
        viz_method: Visualization method ('umap' or 'tsne')
        
    Returns:
        Tuple of (ClusterResult, Plotly figure)
    """
    clusterer = TopicClusterer(embeddings, titles)
    
    # Cluster
    if method == "kmeans":
        if n_clusters is None:
            n_clusters, _ = clusterer.find_optimal_k()
        result = clusterer.cluster_kmeans(n_clusters=n_clusters)
    else:
        result = clusterer.cluster_hdbscan()
    
    # Visualize
    fig = clusterer.visualize_clusters_2d(
        labels=result.labels,
        categories=categories,
        method=viz_method
    )
    
    return result, fig


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Topic Clustering Demo")
    print("="*60)
    
    # Create sample embeddings
    np.random.seed(42)
    n_articles = 100
    embedding_dim = 768
    
    # Create 4 clusters of embeddings
    embeddings = np.vstack([
        np.random.randn(25, embedding_dim) + np.array([1, 0, 0] + [0]*(embedding_dim-3)),
        np.random.randn(25, embedding_dim) + np.array([0, 1, 0] + [0]*(embedding_dim-3)),
        np.random.randn(25, embedding_dim) + np.array([0, 0, 1] + [0]*(embedding_dim-3)),
        np.random.randn(25, embedding_dim) + np.array([-1, -1, -1] + [0]*(embedding_dim-3)),
    ]).astype(np.float32)
    
    titles = [f"Article {i}" for i in range(n_articles)]
    
    # Test clustering
    clusterer = TopicClusterer(embeddings, titles)
    
    # K-means
    print("\nK-means Clustering:")
    result = clusterer.cluster_kmeans(n_clusters=4)
    print(f"Cluster sizes: {result.get_cluster_sizes()}")
    
    # Get representative articles
    topics = clusterer.get_cluster_topics(result.labels, top_n=3)
    print(f"\nCluster representatives:")
    for cluster_id, articles in topics.items():
        print(f"  Cluster {cluster_id}: {articles}")
