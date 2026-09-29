"""
Configuration settings for the News Article Recommender System
CSC4352 Final Project
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model Configuration
MODEL_CONFIG = {
    "model_name": "distilbert-base-uncased",  # Start with DistilBERT for speed
    "embedding_dim": 768,
    "max_length": 512,  # BERT's maximum sequence length
    "batch_size": 16,
    "pooling_strategy": "mean",  # Options: 'cls', 'mean', 'max'
}

# FAISS Configuration
FAISS_CONFIG = {
    "index_type": "IndexFlatIP",  # Inner product for cosine similarity
    "index_path": str(MODELS_DIR / "news_articles.index"),
    "top_k": 20,  # Number of similar articles to retrieve
}

# Dataset Configuration
DATASET_CONFIG = {
    "ag_news": {
        "name": "ag_news",
        "categories": ["World", "Sports", "Business", "Sci/Tech"],
        "max_articles": 10000,  # Limit for faster processing during development
    },
    "bbc_news": {
        "categories": ["business", "entertainment", "politics", "sport", "tech"],
    }
}

# Recommendation Settings
RECOMMENDATION_CONFIG = {
    "num_recommendations": 10,
    "diversity_penalty": 0.1,  # Penalize same-source articles
    "recency_boost": 0.05,  # Boost newer articles
    "min_similarity_threshold": 0.3,
}

# UI Configuration
UI_CONFIG = {
    "page_title": "📰 News Recommender",
    "page_icon": "📰",
    "layout": "wide",
    "articles_per_page": 20,
}
