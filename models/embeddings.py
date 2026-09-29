"""
Transformer-Based Embedding Generation
CSC4352 Final Project

This module handles the generation of semantic embeddings for news articles
using pretrained transformer models (BERT/DistilBERT).
"""

import numpy as np
from typing import List, Optional, Union
from pathlib import Path

# Import with error handling for optional dependencies
TRANSFORMERS_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not installed. Run: pip install torch")

try:
    from transformers import (
        DistilBertTokenizer,
        DistilBertModel,
        BertTokenizer,
        BertModel,
        AutoTokenizer,
        AutoModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers library not installed. Run: pip install transformers")
    # Define placeholder classes to prevent NameError
    DistilBertTokenizer = None
    DistilBertModel = None
    BertTokenizer = None
    BertModel = None
    AutoTokenizer = None
    AutoModel = None

try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable


class EmbeddingGenerator:
    """
    Generates semantic embeddings for text using transformer models.
    
    This class wraps pretrained BERT/DistilBERT models to create
    768-dimensional vector representations of text that capture
    semantic meaning.
    
    Attributes:
        model_name: Name of the pretrained model
        device: Device to run inference on (cpu/cuda)
        max_length: Maximum sequence length for tokenization
        pooling_strategy: How to aggregate token embeddings ('cls', 'mean', 'max')
    """
    
    # Model mapping - only used when transformers is available
    SUPPORTED_MODELS = {
        'distilbert-base-uncased': 'distilbert',
        'distilbert-base-cased': 'distilbert',
        'bert-base-uncased': 'bert',
        'bert-base-cased': 'bert',
    }
    
    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        device: Optional[str] = None,
        max_length: int = 512,
        pooling_strategy: str = 'mean'
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of pretrained model from HuggingFace
            device: 'cpu' or 'cuda' (auto-detected if None)
            max_length: Maximum token sequence length (BERT max is 512)
            pooling_strategy: 'cls' for [CLS] token, 'mean' for average pooling, 'max' for max pooling
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install with:\n"
                "  pip install transformers torch"
            )
        
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch required. Install with:\n"
                "  pip install torch"
            )
        
        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Initializing EmbeddingGenerator...")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Pooling: {pooling_strategy}")
        
        # Load tokenizer and model
        self._load_model()
        
        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.hidden_size
        print(f"  Embedding dimension: {self.embedding_dim}")
    
    def _load_model(self):
        """Load the pretrained tokenizer and model."""
        print(f"Loading {self.model_name}...")
        
        # Use AutoTokenizer and AutoModel for flexibility
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
    
    def _pool_embeddings(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool token embeddings into a single vector per sequence.
        
        Args:
            last_hidden_state: Model output shape (batch, seq_len, hidden_dim)
            attention_mask: Mask indicating real tokens vs padding
            
        Returns:
            Pooled embeddings of shape (batch, hidden_dim)
        """
        if self.pooling_strategy == 'cls':
            # Use the [CLS] token embedding (first token)
            return last_hidden_state[:, 0, :]
        
        elif self.pooling_strategy == 'mean':
            # Mean pooling: average over non-padded tokens
            # Expand attention mask to match hidden state dimensions
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # Avoid division by zero
            return sum_embeddings / sum_mask
        
        elif self.pooling_strategy == 'max':
            # Max pooling over sequence dimension
            # Set padding tokens to very negative value so they're not selected
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            last_hidden_state[mask_expanded == 0] = -1e9
            return torch.max(last_hidden_state, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    @torch.no_grad()
    def generate_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string
            normalize: If True, L2 normalize the output vector
            
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.model(**inputs)
        
        # Pool embeddings
        embedding = self._pool_embeddings(
            outputs.last_hidden_state,
            inputs['attention_mask']
        )
        
        # Convert to numpy
        embedding = embedding.cpu().numpy().squeeze()
        
        # Normalize if requested
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    @torch.no_grad()
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input text strings
            batch_size: Number of texts to process at once
            normalize: If True, L2 normalize output vectors
            show_progress: If True, display progress bar
            
        Returns:
            Embeddings array of shape (num_texts, embedding_dim)
        """
        all_embeddings = []
        
        # Create batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Generating embeddings")
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Pool embeddings
            embeddings = self._pool_embeddings(
                outputs.last_hidden_state,
                inputs['attention_mask']
            )
            
            # Convert to numpy
            batch_embeddings = embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
            all_embeddings = all_embeddings / norms
        
        return all_embeddings
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        filepath: Union[str, Path],
        article_ids: Optional[List[str]] = None
    ):
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Embedding array
            filepath: Path to save to (.npz format)
            article_ids: Optional list of article IDs for reference
        """
        filepath = Path(filepath)
        
        save_dict = {'embeddings': embeddings}
        if article_ids is not None:
            save_dict['article_ids'] = np.array(article_ids)
        
        np.savez_compressed(filepath, **save_dict)
        print(f"Saved embeddings to {filepath}")
    
    @staticmethod
    def load_embeddings(filepath: Union[str, Path]) -> tuple:
        """
        Load embeddings from disk.
        
        Args:
            filepath: Path to .npz file
            
        Returns:
            Tuple of (embeddings array, article_ids array or None)
        """
        filepath = Path(filepath)
        data = np.load(filepath)
        
        embeddings = data['embeddings']
        article_ids = data.get('article_ids', None)
        
        print(f"Loaded {embeddings.shape[0]} embeddings from {filepath}")
        return embeddings, article_ids


# Convenience function for quick embedding generation
def get_embeddings(
    texts: List[str],
    model_name: str = 'distilbert-base-uncased',
    batch_size: int = 16,
    normalize: bool = True
) -> np.ndarray:
    """
    Quick function to generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        model_name: Pretrained model name
        batch_size: Processing batch size
        normalize: Whether to L2 normalize
        
    Returns:
        Embeddings array
    """
    generator = EmbeddingGenerator(model_name=model_name)
    return generator.generate_embeddings_batch(
        texts,
        batch_size=batch_size,
        normalize=normalize
    )


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Embedding Generator")
    print("="*60)
    
    # Initialize generator
    generator = EmbeddingGenerator(
        model_name='distilbert-base-uncased',
        pooling_strategy='mean'
    )
    
    # Test texts
    test_texts = [
        "Breaking news: Stock market reaches all-time high",
        "Scientists discover new species in Amazon rainforest",
        "Local team wins championship in overtime thriller",
        "Tech company announces revolutionary AI product",
    ]
    
    # Generate embeddings
    embeddings = generator.generate_embeddings_batch(test_texts, batch_size=2)
    
    print(f"\nGenerated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    
    # Compute similarity matrix
    print("\nCosine Similarity Matrix:")
    similarities = np.dot(embeddings, embeddings.T)
    
    for i, text in enumerate(test_texts):
        print(f"\n{i}: {text[:50]}...")
        for j in range(len(test_texts)):
            print(f"  -> {j}: {similarities[i, j]:.4f}")
