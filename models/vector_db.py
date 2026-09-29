"""
FAISS Vector Database for Similarity Search
CSC4352 Final Project

This module implements efficient similarity search using FAISS
(Facebook AI Similarity Search) for finding related news articles.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not installed. Run: pip install faiss-cpu")


@dataclass
class SearchResult:
    """Container for a single search result."""
    index: int
    article_id: str
    score: float
    
    def __repr__(self):
        return f"SearchResult(idx={self.index}, id={self.article_id}, score={self.score:.4f})"


class VectorDatabase:
    """
    FAISS-based vector database for efficient similarity search.
    
    This class manages the FAISS index for storing article embeddings
    and performing fast nearest neighbor searches.
    
    Key Concepts:
    - IndexFlatIP: Exact inner product (cosine similarity for normalized vectors)
    - IndexFlatL2: Exact L2 distance
    - IndexIVFFlat: Approximate search using inverted file structure
    
    For this project, we use IndexFlatIP since article embeddings are normalized,
    making inner product equivalent to cosine similarity.
    """
    
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = 'flat_ip',
        nlist: int = 100  # For IVF index: number of clusters
    ):
        """
        Initialize the vector database.
        
        Args:
            dimension: Embedding dimension (768 for BERT/DistilBERT)
            index_type: Type of FAISS index ('flat_ip', 'flat_l2', 'ivf_flat')
            nlist: Number of clusters for IVF index
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS required. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        
        # Initialize index
        self.index = self._create_index()
        
        # Mapping from FAISS index to article IDs
        self.id_mapping: List[str] = []
        
        # Store for retrieval
        self._is_trained = index_type == 'flat_ip' or index_type == 'flat_l2'
        
        print(f"Initialized VectorDatabase:")
        print(f"  Index type: {index_type}")
        print(f"  Dimension: {dimension}")
    
    def _create_index(self) -> faiss.Index:
        """Create the appropriate FAISS index based on configuration."""
        if self.index_type == 'flat_ip':
            # Exact inner product search
            # For normalized vectors, this is equivalent to cosine similarity
            return faiss.IndexFlatIP(self.dimension)
        
        elif self.index_type == 'flat_l2':
            # Exact L2 distance search
            return faiss.IndexFlatL2(self.dimension)
        
        elif self.index_type == 'ivf_flat':
            # Approximate search with inverted file index
            # Requires training on a subset of data first
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            return index
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def train(self, embeddings: np.ndarray):
        """
        Train the index (required for IVF indices).
        
        Args:
            embeddings: Training embeddings of shape (n_samples, dimension)
        """
        if self.index_type in ['flat_ip', 'flat_l2']:
            print("Flat indices don't require training.")
            return
        
        print(f"Training index on {embeddings.shape[0]} vectors...")
        embeddings = embeddings.astype(np.float32)
        self.index.train(embeddings)
        self._is_trained = True
        print("Training complete!")
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        article_ids: List[str]
    ):
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Array of shape (n_articles, dimension)
            article_ids: List of article IDs corresponding to embeddings
        """
        if not self._is_trained and self.index_type == 'ivf_flat':
            raise RuntimeError("IVF index must be trained before adding vectors. Call train() first.")
        
        if len(article_ids) != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {embeddings.shape[0]} embeddings but {len(article_ids)} IDs")
        
        # Ensure float32 for FAISS
        embeddings = embeddings.astype(np.float32)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store ID mapping
        self.id_mapping.extend(article_ids)
        
        print(f"Added {len(article_ids)} vectors. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        exclude_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Search for the k most similar articles.
        
        Args:
            query_embedding: Query vector of shape (dimension,) or (1, dimension)
            k: Number of results to return
            exclude_ids: Article IDs to exclude from results (e.g., already read)
            
        Returns:
            List of SearchResult objects ordered by similarity (descending)
        """
        # Reshape query if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Request more results if we need to filter
        search_k = k + len(exclude_ids) if exclude_ids else k
        search_k = min(search_k, self.index.ntotal)
        
        # Perform search
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Flatten results (single query)
        scores = scores[0]
        indices = indices[0]
        
        # Build results
        results = []
        exclude_set = set(exclude_ids) if exclude_ids else set()
        
        for idx, score in zip(indices, scores):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
                
            article_id = self.id_mapping[idx]
            
            if article_id in exclude_set:
                continue
            
            results.append(SearchResult(
                index=int(idx),
                article_id=article_id,
                score=float(score)
            ))
            
            if len(results) >= k:
                break
        
        return results
    
    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 10
    ) -> List[List[SearchResult]]:
        """
        Search for multiple queries at once.
        
        Args:
            query_embeddings: Array of shape (n_queries, dimension)
            k: Number of results per query
            
        Returns:
            List of result lists, one per query
        """
        query_embeddings = query_embeddings.astype(np.float32)
        
        scores, indices = self.index.search(query_embeddings, k)
        
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for idx, score in zip(query_indices, query_scores):
                if idx < 0:
                    continue
                results.append(SearchResult(
                    index=int(idx),
                    article_id=self.id_mapping[idx],
                    score=float(score)
                ))
            all_results.append(results)
        
        return all_results
    
    def save(self, filepath: Union[str, Path]):
        """
        Save the index and mappings to disk.
        
        Args:
            filepath: Base path (will create .index and .mapping files)
        """
        filepath = Path(filepath)
        
        # Save FAISS index
        index_path = filepath.with_suffix('.index')
        faiss.write_index(self.index, str(index_path))
        
        # Save ID mapping
        mapping_path = filepath.with_suffix('.mapping.npy')
        np.save(mapping_path, np.array(self.id_mapping))
        
        print(f"Saved index to {index_path}")
        print(f"Saved mapping to {mapping_path}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'VectorDatabase':
        """
        Load index and mappings from disk.
        
        Args:
            filepath: Base path (expects .index and .mapping files)
            
        Returns:
            Loaded VectorDatabase instance
        """
        filepath = Path(filepath)
        
        # Load FAISS index
        index_path = filepath.with_suffix('.index')
        index = faiss.read_index(str(index_path))
        
        # Load ID mapping
        mapping_path = filepath.with_suffix('.mapping.npy')
        id_mapping = np.load(mapping_path, allow_pickle=True).tolist()
        
        # Create instance
        db = cls(dimension=index.d)
        db.index = index
        db.id_mapping = id_mapping
        db._is_trained = True
        
        print(f"Loaded index with {db.index.ntotal} vectors")
        return db
    
    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Mathematical definition:
        similarity(A, B) = (A · B) / (||A|| × ||B||)
    
    For normalized vectors, this simplifies to just the dot product.
    
    Args:
        a, b: Input vectors
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Vector Database")
    print("="*60)
    
    # Create sample embeddings
    np.random.seed(42)
    n_articles = 100
    dimension = 768
    
    # Generate random normalized embeddings
    embeddings = np.random.randn(n_articles, dimension).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Generate article IDs
    article_ids = [f"article_{i:04d}" for i in range(n_articles)]
    
    # Create database
    db = VectorDatabase(dimension=dimension, index_type='flat_ip')
    
    # Add embeddings
    db.add_embeddings(embeddings, article_ids)
    
    # Test search
    query = embeddings[0]  # Use first article as query
    results = db.search(query, k=5, exclude_ids=['article_0000'])
    
    print(f"\nSearch results for article_0000:")
    for r in results:
        print(f"  {r}")
    
    # Test save/load
    print("\nTesting save/load...")
    db.save("/tmp/test_db")
    loaded_db = VectorDatabase.load("/tmp/test_db")
    
    print(f"Loaded database with {loaded_db.size} vectors")
