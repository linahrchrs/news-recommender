# src/streamlit_app.py
import streamlit as st
from src.search import Searcher
import json, numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(layout="wide", page_title="News Recommender")

@st.cache_resource
def load_searcher():
    return Searcher()

@st.cache_resource
def load_embedder():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

searcher = load_searcher()
embedder = load_embedder()

st.title("News Article Recommender")

col1, col2 = st.columns([2,3])

with col1:
    st.subheader("Find by text")
    query = st.text_area("Enter a text or paste article", height=120)
    if st.button("Search similar"):
        if query.strip():
            qvec = embedder.encode([query], normalize_embeddings=True)[0]
            results = searcher.query_topk(qvec, k=10)
            for r in results:
                st.write(f"**{r['meta'].get('title','(no title)')}** — {r['meta'].get('source','')}, score: {r['score']:.3f}")
                st.write(r['meta'].get('url',''))
                st.markdown("---")

with col2:
    st.subheader("Browse articles")
    # show a random article or let user pick by id
    ids = list(searcher.meta.keys())
    choice = st.selectbox("Pick an article to see recommendations", options=ids[:200])  # limit for UI
    if choice:
        meta = searcher.meta[choice]
        st.markdown(f"### {meta.get('title')}")
        st.write(meta.get('content')[:1000]+"...")
        if st.button("Recommend similar", key="rec"):
            embed = embedder.encode([meta.get('title','') + " . " + meta.get('content','')], normalize_embeddings=True)[0]
            recs = searcher.query_topk(embed, k=10, exclude_ids={choice})
            for r in recs:
                st.write(f"- {r['meta'].get('title')} ({r['score']:.3f}) — {r['meta'].get('source')}")
