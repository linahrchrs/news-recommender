"""
Data loading and management modules

Supported Datasets:
- MIND: Microsoft News Dataset
- AG News: 120,000 articles, 4 categories
- BBC News: 2,225 articles, 5 categories
"""

from .data_loader import (
    NewsArticle,
    NewsDataset,
    load_ag_news,
    load_bbc_news,
    load_bbc_news_csv,
    load_mind_dataset,
    load_all_datasets,
    load_sample_data,
    generate_article_id,
    download_mind_demo
)

__all__ = [
    'NewsArticle',
    'NewsDataset', 
    'load_ag_news',
    'load_bbc_news',
    'load_bbc_news_csv',
    'load_mind_dataset',
    'load_all_datasets',
    'load_sample_data',
    'generate_article_id',
    'download_mind_demo'
]
