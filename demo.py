"""
News Article Recommender System - Component Demonstration
CSC4352 Final Project

This script demonstrates all major components of the recommendation system.
Run this to verify your installation and understand how each part works.

Usage: python demo.py
"""

import numpy as np
from pathlib import Path
import sys

# Ensure proper imports
sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies():
    """Check if all required dependencies are installed."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import faiss
    except ImportError:
        missing.append("faiss-cpu")
    
    try:
        import streamlit
    except ImportError:
        missing.append("streamlit")
    
    try:
        import tqdm
    except ImportError:
        missing.append("tqdm")
    
    return missing


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_preprocessing():
    """Demonstrate the text preprocessing pipeline."""
    print_header("1. TEXT PREPROCESSING PIPELINE")
    
    from utils.preprocessing import TextPreprocessor, combine_title_content
    
    preprocessor = TextPreprocessor(lowercase=False)
    
    # Test cases
    test_cases = [
        ("HTML Content", "<p>Breaking <b>News</b>: Stock market &amp; economy update!</p>"),
        ("URLs/Emails", "Visit https://news.com for more. Contact: news@example.com"),
        ("Special Chars", "AI Breakthrough!!! 🚀 #trending @tech_news $$$ profits"),
        ("Whitespace", "  Multiple   spaces   and\n\nnewlines  here  "),
    ]
    
    for name, text in test_cases:
        cleaned = preprocessor.preprocess(text)
        print(f"  {name}:")
        print(f"    Original: {text}")
        print(f"    Cleaned:  {cleaned}")
        print()
    
    # Demonstrate title + content combination
    title = "Tech Giants Report Record Earnings"
    content = "Major technology companies reported better-than-expected quarterly results."
    combined = combine_title_content(title, content)
    print(f"  Combined Text (for embedding):")
    print(f"    Title: {title}")
    print(f"    Content: {content}")
    print(f"    Combined: {combined}")


def demo_data_loading():
    """Demonstrate data loading capabilities."""
    print_header("2. DATA LOADING")
    
    from data.data_loader import load_sample_data, NewsDataset, NewsArticle
    
    # Load sample dataset
    dataset = load_sample_data()
    
    print(f"  Dataset Statistics:")
    print(f"    Total articles: {len(dataset)}")
    print(f"    Categories: {', '.join(dataset.get_categories())}")
    print(f"    Sources: {', '.join(dataset.get_sources())}")
    
    # Show sample article
    article = dataset[0]
    print(f"\n  Sample Article:")
    print(f"    ID: {article.article_id}")
    print(f"    Title: {article.title}")
    print(f"    Category: {article.category}")
    print(f"    Source: {article.source}")
    print(f"    Content: {article.content[:150]}...")
    
    # Demonstrate filtering
    tech_articles = dataset.filter_by_category("Sci/Tech")
    print(f"\n  Filtered by 'Sci/Tech': {len(tech_articles)} articles")


def demo_embeddings():
    """Demonstrate transformer embedding generation."""
    print_header("3. TRANSFORMER EMBEDDINGS")
    
    # Check dependencies first
    missing = check_dependencies()
    if 'torch' in missing or 'transformers' in missing:
        print("  ❌ Missing required dependencies!")
        print("  Please install them first:")
        print()
        print("    pip install torch transformers")
        print()
        print("  After installing, run this demo again.")
        return False
    
    from models.embeddings import EmbeddingGenerator
    
    print("  Loading DistilBERT model (this may take a moment)...")
    generator = EmbeddingGenerator(
        model_name='distilbert-base-uncased',
        pooling_strategy='mean'
    )
    
    # Sample texts for embedding
    texts = [
        "Stock market reaches all-time high amid economic optimism",
        "Tech companies report strong quarterly earnings",
        "Scientists discover high-temperature superconductor breakthrough",
        "World Cup final draws record television audience",
    ]
    
    print(f"\n  Generating embeddings for {len(texts)} texts...")
    embeddings = generator.generate_embeddings_batch(texts, batch_size=2, show_progress=False)
    
    print(f"\n  Embedding Shape: {embeddings.shape}")
    print(f"    - {embeddings.shape[0]} texts")
    print(f"    - {embeddings.shape[1]} dimensions per embedding")
    
    # Compute and display similarity matrix
    print(f"\n  Cosine Similarity Matrix:")
    similarity_matrix = np.dot(embeddings, embeddings.T)
    
    # Print header
    print("     ", end="")
    for i in range(len(texts)):
        print(f"  [{i}]  ", end="")
    print()
    
    # Print matrix
    for i, text in enumerate(texts):
        print(f"  [{i}]", end="")
        for j in range(len(texts)):
            sim = similarity_matrix[i, j]
            print(f"  {sim:.3f}", end="")
        print(f"  {text[:40]}...")
    
    print("\n  Interpretation:")
    print("    - Diagonal values are 1.0 (identical)")
    print("    - Higher values = more semantically similar")
    print("    - Notice: Finance articles [0,1] are more similar to each other")
    
    return True


def demo_vector_database():
    """Demonstrate FAISS vector database operations."""
    print_header("4. FAISS VECTOR DATABASE")
    
    # Check for FAISS
    try:
        import faiss
    except ImportError:
        print("  ❌ FAISS not installed!")
        print("  Please install it first:")
        print()
        print("    pip install faiss-cpu")
        print()
        print("  After installing, run this demo again.")
        return False
    
    from models.vector_db import VectorDatabase, cosine_similarity
    
    # Create sample embeddings
    np.random.seed(42)
    n_vectors = 100
    dimension = 768
    
    # Generate normalized random vectors
    embeddings = np.random.randn(n_vectors, dimension).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    article_ids = [f"article_{i:04d}" for i in range(n_vectors)]
    
    # Create database
    print("  Creating FAISS index...")
    db = VectorDatabase(dimension=dimension, index_type='flat_ip')
    
    # Add vectors
    db.add_embeddings(embeddings, article_ids)
    print(f"  Added {db.size} vectors to index")
    
    # Perform search
    query = embeddings[0]  # Use first vector as query
    results = db.search(query, k=5, exclude_ids=['article_0000'])
    
    print(f"\n  Search Results (query: article_0000):")
    for r in results:
        print(f"    {r.article_id}: score = {r.score:.4f}")
    
    # Demonstrate save/load
    print("\n  Testing save/load functionality...")
    db.save("/tmp/demo_db")
    loaded_db = VectorDatabase.load("/tmp/demo_db")
    print(f"  Loaded database with {loaded_db.size} vectors")
    
    return True


def demo_recommendation_engine():
    """Demonstrate the full recommendation pipeline."""
    print_header("5. RECOMMENDATION ENGINE")
    
    # Check all dependencies
    missing = check_dependencies()
    required = ['torch', 'transformers', 'faiss-cpu']
    missing_required = [m for m in missing if m in required or m == 'faiss-cpu']
    
    if missing_required:
        print("  ❌ Missing required dependencies!")
        print("  Please install them first:")
        print()
        print(f"    pip install {' '.join(missing_required)}")
        print()
        print("  After installing, run this demo again.")
        return False
    
    from data.data_loader import load_sample_data
    from models.recommender import RecommendationEngine
    
    # Load data
    print("  Loading sample dataset...")
    dataset = load_sample_data()
    articles = [article.to_dict() for article in dataset]
    
    # Initialize engine
    print("  Initializing recommendation engine...")
    engine = RecommendationEngine(
        model_name='distilbert-base-uncased',
        embedding_dim=768
    )
    
    # Index articles
    print("  Indexing articles...")
    engine.index_articles(articles, batch_size=8)
    
    # Get recommendations for first article
    seed_article = articles[0]
    print(f"\n  Finding recommendations for:")
    print(f"    Title: {seed_article['title']}")
    print(f"    Category: {seed_article['category']}")
    
    recommendations = engine.get_recommendations(
        seed_article['article_id'],
        n_recommendations=5
    )
    
    print(f"\n  Top 5 Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n    {i}. {rec.title}")
        print(f"       Category: {rec.category}")
        print(f"       Similarity: {rec.similarity_score:.2%}")
    
    # Demonstrate search
    print("\n  " + "-" * 50)
    query = "artificial intelligence technology"
    print(f"\n  Semantic Search: '{query}'")
    
    search_results = engine.search_by_query(query, n_results=3)
    
    print(f"\n  Search Results:")
    for i, rec in enumerate(search_results, 1):
        print(f"    {i}. {rec.title} ({rec.similarity_score:.2%})")
    
    return True


def demo_full_pipeline():
    """Run a complete end-to-end demonstration."""
    print_header("RUNNING FULL DEMONSTRATION")
    
    print("This demo will show all components of the News Recommender System.\n")
    print("Components:")
    print("  1. Text Preprocessing Pipeline")
    print("  2. Data Loading and Management")
    print("  3. Transformer Embedding Generation")
    print("  4. FAISS Vector Database")
    print("  5. Recommendation Engine")
    
    input("\nPress Enter to continue...")
    
    demo_preprocessing()
    input("\nPress Enter for next section...")
    
    demo_data_loading()
    input("\nPress Enter for next section...")
    
    demo_embeddings()
    input("\nPress Enter for next section...")
    
    demo_vector_database()
    input("\nPress Enter for next section...")
    
    demo_recommendation_engine()
    
    print_header("DEMONSTRATION COMPLETE")
    print("  All components are working correctly!")
    print("  Run 'streamlit run app.py' to launch the web interface.")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="News Recommender System Demo")
    parser.add_argument(
        '--component',
        choices=['preprocess', 'data', 'embeddings', 'faiss', 'recommend', 'all'],
        default='all',
        help='Which component to demonstrate'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode with pauses'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Only check dependencies'
    )
    
    args = parser.parse_args()
    
    # Check dependencies first
    print_header("DEPENDENCY CHECK")
    missing = check_dependencies()
    
    if missing:
        print("  ⚠️  Missing dependencies detected:")
        for dep in missing:
            print(f"      - {dep}")
        print()
        print("  To install all dependencies, run:")
        print()
        print(f"      pip install {' '.join(missing)}")
        print()
        print("  Or install from requirements.txt:")
        print()
        print("      pip install -r requirements.txt")
        print()
        
        if args.check:
            sys.exit(1)
        
        print("  Continuing with available components...\n")
    else:
        print("  ✅ All dependencies installed!")
    
    if args.check:
        sys.exit(0)
    
    if args.component == 'all':
        if args.interactive:
            demo_full_pipeline()
        else:
            # Run demos that don't need missing deps
            demo_preprocessing()
            demo_data_loading()
            
            if 'torch' not in missing and 'transformers' not in missing:
                demo_embeddings()
            else:
                print_header("3. TRANSFORMER EMBEDDINGS")
                print("  ⏭️  Skipped - missing torch/transformers")
            
            if 'faiss-cpu' not in missing:
                demo_vector_database()
            else:
                print_header("4. FAISS VECTOR DATABASE")
                print("  ⏭️  Skipped - missing faiss-cpu")
            
            if not missing:
                demo_recommendation_engine()
            else:
                print_header("5. RECOMMENDATION ENGINE")
                print("  ⏭️  Skipped - missing dependencies")
            
            print_header("DEMONSTRATION COMPLETE")
            if missing:
                print(f"  ⚠️  Some demos were skipped due to missing: {', '.join(missing)}")
                print(f"  Install with: pip install {' '.join(missing)}")
            else:
                print("  ✅ All components working correctly!")
                print("  Run 'streamlit run app.py' to launch the web interface.")
    elif args.component == 'preprocess':
        demo_preprocessing()
    elif args.component == 'data':
        demo_data_loading()
    elif args.component == 'embeddings':
        demo_embeddings()
    elif args.component == 'faiss':
        demo_vector_database()
    elif args.component == 'recommend':
        demo_recommendation_engine()
