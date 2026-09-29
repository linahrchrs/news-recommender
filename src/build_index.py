# src/build_index.py
import faiss
import numpy as np
from pathlib import Path
import json

EMB_PATH = "data/embeddings/ag_news_embeddings.npy"
OUT_INDEX = "faiss_index/news_articles.index"


def build_index_flat(emb, out_index_path):
    """Exact search — best for small/medium datasets (< 1M vectors)."""
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product for cosine
    index.add(emb)
    faiss.write_index(index, out_index_path)
    print("Saved FLAT index to:", out_index_path)


def build_index_ivfflat(emb, out_index_path, nlist=1024):
    """IVFFlat — scalable for large datasets (1M+ vectors)."""
    dim = emb.shape[1]

    # quantizer = coarse centroids
    quantizer = faiss.IndexFlatIP(dim)

    index = faiss.IndexIVFFlat(
        quantizer,
        dim,
        nlist,                      # number of clusters
        faiss.METRIC_INNER_PRODUCT
    )

    # IMPORTANT: train before adding vectors
    print("Training IVF index...")
    index.train(emb)

    print("Adding vectors to IVF index...")
    index.add(emb)

    faiss.write_index(index, out_index_path)
    print("Saved IVFFlat index to:", out_index_path)


if __name__ == "__main__":
    Path("faiss_index").mkdir(parents=True, exist_ok=True)

    emb = np.load(EMB_PATH).astype("float32")

    # ----- USE ONE OF THE TWO OPTIONS -----

    # (1) FLAT INDEX (default, smaller datasets)
    # build_index_flat(emb, OUT_INDEX)

    # (2) IVF INDEX (use this if dataset is large)
    build_index_ivfflat(emb, OUT_INDEX, nlist=1024)
