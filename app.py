"""
News Article Recommender - Streamlit Web Interface
CSC4352 Final Project

A modern, interactive web interface for the news recommendation system.
Run with: streamlit run app.py

Features:
- Browse articles from MIND, AG News, BBC News datasets
- Semantic search using DistilBERT embeddings
- Content-based recommendations
- Personalized recommendations based on reading history
- Topic clustering visualization (K-means, HDBSCAN)
"""

import streamlit as st
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import (
    load_sample_data, 
    load_ag_news, 
    load_bbc_news,
    load_bbc_news_csv,
    load_mind_dataset,
    load_all_datasets,
    NewsDataset
)
from models.recommender import RecommendationEngine, Recommendation
from config import UI_CONFIG, MODEL_CONFIG, RECOMMENDATION_CONFIG

# Optional imports for clustering visualization
try:
    from models.clustering import TopicClusterer, create_cluster_visualization
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title=UI_CONFIG['page_title'],
    page_icon=UI_CONFIG['page_icon'],
    layout=UI_CONFIG['layout'],
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS Styling
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Article cards */
    .article-card {
        background: linear-gradient(145deg, #1e2a3a, #243447);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .article-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 179, 237, 0.4);
        box-shadow: 0 12px 40px rgba(99, 179, 237, 0.15);
    }
    
    .article-title {
        color: #e2e8f0;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 8px;
        line-height: 1.4;
    }
    
    .article-meta {
        display: flex;
        gap: 16px;
        margin-bottom: 12px;
        flex-wrap: wrap;
    }
    
    .article-category {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .article-source {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    
    .article-content {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.7;
        margin-top: 12px;
    }
    
    .article-full-content {
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.8;
        margin-top: 16px;
        padding: 16px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 8px;
    }
    
    .similarity-score {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 12px;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(145deg, #2d3748, #1a202c);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 4px;
    }
    
    /* Section headers */
    .rec-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 24px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'engine' not in st.session_state:
        st.session_state.engine = None
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'selected_article' not in st.session_state:
        st.session_state.selected_article = None
    if 'reading_history' not in st.session_state:
        st.session_state.reading_history = []
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'browse'
    if 'show_similar_for' not in st.session_state:
        st.session_state.show_similar_for = None
    if 'expanded_article' not in st.session_state:
        st.session_state.expanded_article = None


# ============================================================================
# Data Loading with Caching
# ============================================================================

@st.cache_resource
def load_engine(data_source: str, max_articles: int):
    """Load and cache the recommendation engine."""
    # Load dataset based on selection
    if data_source == "Sample Data (15 articles)":
        dataset = load_sample_data()
    elif data_source == "AG News":
        dataset = load_ag_news(max_articles=max_articles)
    else:
        dataset = load_sample_data()
    
    # Initialize recommendation engine
    engine = RecommendationEngine()
    # Convert NewsArticle objects to dictionaries for indexing
    articles_dict = [article.to_dict() for article in dataset.articles]
    engine.index_articles(articles_dict)
    
    return engine, dataset


# ============================================================================
# UI Components
# ============================================================================

def render_article_card(article, show_full=False, similarity_score=None):
    """Render an article card."""
    if show_full:
        content_display = article.content
    else:
        content_display = article.content[:250] + "..." if len(article.content) > 250 else article.content
    
    similarity_html = ""
    if similarity_score is not None:
        similarity_pct = similarity_score * 100
        similarity_html = f'<div class="similarity-score"> {similarity_pct:.1f}% Match</div>'
    
    st.markdown(f"""
    <div class="article-card">
        <div class="article-title">{article.title}</div>
        <div class="article-meta">
            <span class="article-category">{article.category}</span>
            <span class="article-source"> {article.source}</span>
            <span class="article-source"> {article.published_date}</span>
        </div>
        <div class="article-content">{content_display}</div>
        {similarity_html}
    </div>
    """, unsafe_allow_html=True)


def render_stats(dataset):
    """Render dataset statistics."""
    categories = dataset.get_categories()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(dataset)}</div>
            <div class="stat-label">Total Articles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(categories)}</div>
            <div class="stat-label">Categories</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(st.session_state.reading_history)}</div>
            <div class="stat-label">Articles Read</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# Main Views
# ============================================================================

def render_browse_view(engine, dataset):
    """Render the browse articles view."""
    st.markdown('<div class="rec-header"> Browse Articles</div>', unsafe_allow_html=True)
    
    # Filters
    col1, col2 = st.columns([2, 1])
    with col1:
        categories = ["All"] + sorted(dataset.get_categories())
        selected_category = st.selectbox("Filter by category", categories)
    
    # Filter articles
    if selected_category == "All":
        articles = dataset.articles
    else:
        articles = [a for a in dataset.articles if a.category == selected_category]
    
    st.caption(f"Showing {len(articles)} articles")
    
    # Check if we need to show similar articles
    if st.session_state.show_similar_for:
        render_similar_articles(engine, st.session_state.show_similar_for)
        if st.button("← Back to Browse", key="back_to_browse"):
            st.session_state.show_similar_for = None
            st.rerun()
        return
    
    # Pagination
    articles_per_page = 10
    total_pages = max(1, (len(articles) + articles_per_page - 1) // articles_per_page)
    
    if total_pages > 1:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    else:
        page = 1
    
    start_idx = (page - 1) * articles_per_page
    end_idx = start_idx + articles_per_page
    page_articles = articles[start_idx:end_idx]
    
    # Display articles
    for idx, article in enumerate(page_articles):
        global_idx = start_idx + idx
        
        # Create expander for each article
        with st.expander(f" {article.title}", expanded=False):
            # Show full article content
            st.markdown(f"""
            <div class="article-meta" style="margin-bottom: 16px;">
                <span class="article-category">{article.category}</span>
                <span class="article-source"> {article.source}</span>
                <span class="article-source"> {article.published_date}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Full Article:**")
            st.write(article.content)
            
            st.markdown("---")
            
            # Action buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button(" Find Similar", key=f"similar_{global_idx}_{article.article_id}", type="primary"):
                    # Mark as read
                    if article.article_id not in st.session_state.reading_history:
                        st.session_state.reading_history.append(article.article_id)
                        engine.mark_as_read(article.article_id)
                    # Show similar articles
                    st.session_state.show_similar_for = article
                    st.rerun()
            
            with col2:
                if article.article_id in st.session_state.reading_history:
                    st.success("✓ Read")
                else:
                    if st.button(" Mark as Read", key=f"read_{global_idx}_{article.article_id}"):
                        st.session_state.reading_history.append(article.article_id)
                        engine.mark_as_read(article.article_id)
                        st.rerun()


def render_similar_articles(engine, article):
    """Render similar articles for a selected article."""
    st.markdown('<div class="rec-header"> Similar Articles</div>', unsafe_allow_html=True)
    
    # Show the source article
    st.markdown("**You selected:**")
    render_article_card(article, show_full=False)
    
    st.markdown("---")
    st.markdown("**Similar articles based on semantic similarity:**")
    
    # Get recommendations
    recommendations = engine.get_recommendations(
        article.article_id,
        n_recommendations=RECOMMENDATION_CONFIG['num_recommendations']
    )
    
    if recommendations:
        for rec_idx, rec in enumerate(recommendations):
            with st.expander(f" {rec.similarity_score*100:.1f}% Match - {rec.title}", expanded=(rec_idx == 0)):
                st.markdown(f"""
                <div class="article-meta" style="margin-bottom: 16px;">
                    <span class="article-category">{rec.category}</span>
                    <span class="article-source"> {rec.source}</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.write(rec.content)
                
                # Mark as read button
                if rec.article_id not in st.session_state.reading_history:
                    if st.button(" Mark as Read", key=f"rec_read_{rec_idx}_{rec.article_id}"):
                        st.session_state.reading_history.append(rec.article_id)
                        engine.mark_as_read(rec.article_id)
                        st.rerun()
                else:
                    st.success("✓ Already read")
    else:
        st.info("No similar articles found.")


def render_search_view(engine):
    """Render the search view."""
    st.markdown('<div class="rec-header"> Semantic Search</div>', unsafe_allow_html=True)
    st.caption("Search using natural language - the AI understands meaning, not just keywords!")
    
    query = st.text_input(
        "Search articles",
        placeholder="e.g., 'latest developments in artificial intelligence' or 'stock market news'",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        num_results = st.slider("Number of results", 5, 20, 10)
    with col2:
        search_button = st.button(" Search", type="primary", use_container_width=True)
    
    if query and search_button:
        with st.spinner("Searching..."):
            results = engine.search_by_query(query, n_results=num_results)
        
        if results:
            st.success(f"Found **{len(results)}** relevant articles")
            
            for rec_idx, rec in enumerate(results):
                with st.expander(f" {rec.similarity_score*100:.1f}% Match - {rec.title}", expanded=(rec_idx < 3)):
                    st.markdown(f"""
                    <div class="article-meta" style="margin-bottom: 16px;">
                        <span class="article-category">{rec.category}</span>
                        <span class="article-source"> {rec.source}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write(rec.content)
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if rec.article_id not in st.session_state.reading_history:
                            if st.button(" Mark as Read", key=f"search_read_{rec_idx}_{rec.article_id}"):
                                st.session_state.reading_history.append(rec.article_id)
                                engine.mark_as_read(rec.article_id)
                                st.rerun()
                        else:
                            st.success("✓ Read")
        else:
            st.warning("No matching articles found. Try different keywords.")


def render_personalized_view(engine):
    """Render the personalized recommendations view."""
    st.markdown('<div class="rec-header"> Personalized For You</div>', unsafe_allow_html=True)
    
    if len(st.session_state.reading_history) < 2:
        st.info(" Read at least 2 articles to get personalized recommendations!")
        st.caption("Go to 'Browse Articles', expand articles to read them, and click 'Mark as Read' or 'Find Similar'.")
        
        # Show reading history
        if st.session_state.reading_history:
            st.markdown(f"**Articles read so far:** {len(st.session_state.reading_history)}")
        return
    
    st.caption(f"Based on your reading history ({len(st.session_state.reading_history)} articles)")
    
    # Show what articles were read
    with st.expander(" Your Reading History"):
        for article_id in st.session_state.reading_history[-10:]:  # Show last 10
            st.write(f"• {article_id}")
    
    num_recs = st.slider("Number of recommendations", 5, 20, 10)
    
    if st.button(" Get Recommendations", type="primary"):
        with st.spinner("Analyzing your preferences..."):
            recommendations = engine.get_personalized_recommendations(n_recommendations=num_recs)
        
        if recommendations:
            st.success(f"Found **{len(recommendations)}** personalized recommendations!")
            
            for rec_idx, rec in enumerate(recommendations):
                with st.expander(f" {rec.similarity_score*100:.1f}% Match - {rec.title}", expanded=(rec_idx < 3)):
                    st.markdown(f"""
                    <div class="article-meta" style="margin-bottom: 16px;">
                        <span class="article-category">{rec.category}</span>
                        <span class="article-source"> {rec.source}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write(rec.content)
        else:
            st.info("No personalized recommendations available yet. Keep reading!")


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Render the sidebar with controls."""
    st.sidebar.title(" News Recommender")
    st.sidebar.markdown("---")
    
    # Data source selection
    st.sidebar.subheader(" Data Source")
    data_source = st.sidebar.selectbox(
        "Select dataset",
        ["Sample Data (15 articles)", "AG News"],
        help="Choose which news dataset to use"
    )
    
    # Number of articles (for AG News)
    if data_source == "AG News":
        max_articles = st.sidebar.slider(
            "Max articles to load",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="More articles = better recommendations but slower loading"
        )
    else:
        max_articles = 15
    
    # Initialize button
    if st.sidebar.button(" Initialize System", type="primary", use_container_width=True):
        with st.spinner("Loading model and indexing articles..."):
            engine, dataset = load_engine(data_source, max_articles)
            st.session_state.engine = engine
            st.session_state.dataset = dataset
            st.session_state.initialized = True
            st.session_state.reading_history = []
            st.session_state.show_similar_for = None
        st.sidebar.success("System ready!")
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.subheader(" Navigation")
    
    if st.sidebar.button(" Browse Articles", use_container_width=True,
                         type="primary" if st.session_state.current_view == 'browse' else "secondary"):
        st.session_state.current_view = 'browse'
        st.session_state.show_similar_for = None
        st.rerun()
    
    if st.sidebar.button(" Search", use_container_width=True,
                         type="primary" if st.session_state.current_view == 'search' else "secondary"):
        st.session_state.current_view = 'search'
        st.rerun()
    
    if st.sidebar.button(" Personalized", use_container_width=True,
                         type="primary" if st.session_state.current_view == 'personalized' else "secondary"):
        st.session_state.current_view = 'personalized'
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Reading history
    st.sidebar.subheader(" Reading History")
    st.sidebar.metric("Articles Read", len(st.session_state.reading_history))
    
    if st.session_state.reading_history:
        if st.sidebar.button(" Clear History", use_container_width=True):
            st.session_state.reading_history = []
            if st.session_state.engine:
                st.session_state.engine.reading_history = set()
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Info
    with st.sidebar.expander(" How to Use"):
        st.markdown("""
        1. **Initialize** the system with your chosen dataset
        2. **Browse** articles and click to expand and read them
        3. **Find Similar** to discover related articles
        4. **Mark as Read** to track your reading
        5. **Search** using natural language queries
        6. Get **Personalized** recommendations after reading 2+ articles
        """)
    
    with st.sidebar.expander(" About"):
        st.markdown("""
        **Technology Stack:**
        -  DistilBERT embeddings (768-dim)
        -  FAISS vector search
        -  Streamlit UI
        
        **CSC4352 Final Project**
        """)


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    
    # Main content area
    if not st.session_state.initialized:
        # Welcome screen
        st.title(" News Article Recommender")
        st.markdown("""
        Welcome to the **AI-Powered News Recommendation System**!
        
        This system uses **DistilBERT** transformer embeddings and **FAISS** vector search 
        to find semantically similar articles based on actual meaning, not just keywords.
        
        ### Getting Started
        1. Select a dataset from the sidebar
        2. Click **"Initialize System"** to load the model
        3. Browse, search, and discover articles!
        
        ### Features
        -  **Browse** - Explore articles by category
        -  **Semantic Search** - Search using natural language
        -  **Find Similar** - Discover related articles
        -  **Personalized** - Get recommendations based on your reading history
        """)
        
        # Show sample architecture
        with st.expander("🏗️ System Architecture"):
            st.markdown("""
            ```
            ┌─────────────────────────────────────────┐
            │           User Interface (Streamlit)    │
            ├─────────────────────────────────────────┤
            │         Recommendation Engine           │
            │    (Query Processing + Ranking)         │
            ├─────────────────────────────────────────┤
            │           Embedding Layer               │
            │     (DistilBERT - 768 dimensions)       │
            ├─────────────────────────────────────────┤
            │          Vector Database                │
            │         (FAISS IndexFlatIP)             │
            ├─────────────────────────────────────────┤
            │           Data Storage                  │
            │    (MIND / AG News / BBC News)          │
            └─────────────────────────────────────────┘
            ```
            """)
        return
    
    # Get engine and dataset from session state
    engine = st.session_state.engine
    dataset = st.session_state.dataset
    
    # Show stats
    render_stats(dataset)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render current view
    if st.session_state.current_view == 'browse':
        render_browse_view(engine, dataset)
    elif st.session_state.current_view == 'search':
        render_search_view(engine)
    elif st.session_state.current_view == 'personalized':
        render_personalized_view(engine)


if __name__ == "__main__":
    main()