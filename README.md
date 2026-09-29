#  News Article Recommender System

**CSC4352 Final Project - NLP + Transformers**

An intelligent news recommendation system that leverages transformer-based models (DistilBERT) and FAISS vector search to provide semantically-aware article recommendations.

---

##  Project Overview

This system demonstrates modern NLP techniques for content recommendation:

- **Transformer Embeddings**: Uses DistilBERT to create 768-dimensional semantic representations of articles
- **Vector Search**: FAISS enables fast similarity search across thousands of articles
- **Content-Based Filtering**: Recommends articles based on semantic similarity
- **Interactive UI**: Streamlit-powered web interface for browsing, searching, and getting recommendations

---

##  System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                           │
│                  (Streamlit Frontend)                       │
├─────────────────────────────────────────────────────────────┤
│                 RECOMMENDATION ENGINE                       │
│    ┌──────────────┬──────────────────┬───────────────┐     │
│    │    Query     │    Similarity    │   Ranking &   │     │
│    │  Processing  │   Computation    │   Filtering   │     │
│    └──────────────┴──────────────────┴───────────────┘     │
├─────────────────────────────────────────────────────────────┤
│                   EMBEDDING LAYER                           │
│              (DistilBERT Transformer)                       │
│    ┌──────────────┬──────────────────┬───────────────┐     │
│    │  WordPiece   │     Model        │   768-dim     │     │
│    │  Tokenizer   │    Encoder       │   Vector      │     │
│    └──────────────┴──────────────────┴───────────────┘     │
├─────────────────────────────────────────────────────────────┤
│                   VECTOR DATABASE                           │
│                (FAISS Index Store)                          │
├─────────────────────────────────────────────────────────────┤
│                    DATA STORAGE                             │
│               (News Article Database)                       │
└─────────────────────────────────────────────────────────────┘
```

---

##  Project Structure

```
news-recommender/
├── app.py                 # Streamlit web application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── data/
│   ├── __init__.py
│   └── data_loader.py     # Dataset loading utilities
│
├── models/
│   ├── __init__.py
│   ├── embeddings.py      # Transformer embedding generation
│   ├── vector_db.py       # FAISS vector database
│   └── recommender.py     # Recommendation engine
│
└── utils/
    ├── __init__.py
    └── preprocessing.py   # Text preprocessing pipeline
```

---

##  Quick Start

### 1. Install Dependencies

```bash
cd news-recommender
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

### 3. Open in Browser

Navigate to `http://localhost:8501`

---

##  Supported Datasets

### Sample Data (Default)
- 15 hand-crafted articles across 5 categories
- Perfect for testing and demonstration

### AG News
- 120,000 news articles
- 4 categories: World, Sports, Business, Sci/Tech
- Loaded via HuggingFace Datasets

### Custom Data
You can add your own dataset by implementing a loader in `data/data_loader.py`:

```python
def load_custom_data() -> NewsDataset:
    articles = [
        NewsArticle(
            article_id="unique_id",
            title="Article Title",
            content="Article content...",
            category="Category",
            source="Source Name",
            published_date="2024-01-01"
        ),
        # ... more articles
    ]
    return NewsDataset(articles)
```

---

##  Core Components

### 1. Text Preprocessing (`utils/preprocessing.py`)

```python
from utils import TextPreprocessor

preprocessor = TextPreprocessor(lowercase=False)
clean_text = preprocessor.preprocess(raw_text)
```

Features:
- HTML tag removal
- URL and email filtering
- Unicode normalization
- Whitespace normalization

### 2. Embedding Generation (`models/embeddings.py`)

```python
from models import EmbeddingGenerator

generator = EmbeddingGenerator(
    model_name='distilbert-base-uncased',
    pooling_strategy='mean'  # Options: 'cls', 'mean', 'max'
)

# Single text
embedding = generator.generate_embedding("Article text here")

# Batch processing
embeddings = generator.generate_embeddings_batch(texts, batch_size=16)
```

### 3. Vector Database (`models/vector_db.py`)

```python
from models import VectorDatabase

db = VectorDatabase(dimension=768, index_type='flat_ip')

# Add vectors
db.add_embeddings(embeddings, article_ids)

# Search
results = db.search(query_embedding, k=10)
```

Index Types:
- `flat_ip`: Exact inner product (recommended for cosine similarity)
- `flat_l2`: Exact L2 distance
- `ivf_flat`: Approximate search (faster for large datasets)

### 4. Recommendation Engine (`models/recommender.py`)

```python
from models import RecommendationEngine

engine = RecommendationEngine()

# Index articles
engine.index_articles(articles)

# Get recommendations
recommendations = engine.get_recommendations(
    article_id="article_001",
    n_recommendations=10
)

# Search by query
results = engine.search_by_query("artificial intelligence news")

# Personalized recommendations (based on reading history)
engine.mark_as_read("article_001")
personalized = engine.get_personalized_recommendations()
```

---

##  Mathematical Foundation

### Cosine Similarity

The core similarity metric:
similarity(A, B) = (A · B) / (||A|| × ||B||)


Where:
- `A · B` is the dot product
- `||A||` is the L2 norm (magnitude)

For normalized vectors, this simplifies to just the dot product, which is why we use `IndexFlatIP` in FAISS.

### Embedding Pooling Strategies

1. **CLS Token** (`cls`): Use the [CLS] token's representation
2. **Mean Pooling** (`mean`): Average all token embeddings (recommended)
3. **Max Pooling** (`max`): Take maximum values across dimensions


##  UI Features

### Browse Mode
- Category filtering
- Pagination
- Click-to-recommend

### Search Mode
- Natural language queries
- Semantic understanding
- Configurable result count

### Personalized Mode
- Based on reading history
- Requires 2+ read articles
- User profile embedding



##  Testing

Run individual components:

```bash
# Test preprocessing
python -m utils.preprocessing

# Test embeddings
python -m models.embeddings

# Test vector database
python -m models.vector_db

# Test recommendation engine
python -m models.recommender
```

