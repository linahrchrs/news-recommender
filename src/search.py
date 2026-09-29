# src/search.py
import faiss, numpy as np, json
from pathlib import Path
from datetime import datetime

INDEX_PATH = "faiss_index/news_articles.index"
EMB_PATH = "data/embeddings/ag_news_embeddings.npy"
ID_MAP = "data/embeddings/ag_news_id_map.json"
PROCESSED_PATH = "data/processed/ag_news.jsonl"

class Searcher:
    def __init__(self, index_path=INDEX_PATH, id_map_path=ID_MAP):
        self.index = faiss.read_index(index_path)
        self.id_map = json.load(open(id_map_path,'r',encoding='utf-8'))

        # load article metadata into dict for fast lookup
        self.meta = {}
        with open(PROCESSED_PATH,'r',encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.meta[obj['article_id']] = obj

    def query_topk(self, query_vec, k=10, exclude_ids=None):
        # query_vec must be normalized float32 1xdim
        if exclude_ids is None:
            exclude_ids = set()
        query = np.asarray(query_vec, dtype='float32').reshape(1, -1)
        scores, indices = self.index.search(query, k+len(exclude_ids))
        scores = scores[0]
        inds = indices[0]
        results = []
        for score, idx in zip(scores, inds):
            if idx < 0:
                continue
            article_id = self.id_map[idx]
            if article_id in exclude_ids:
                continue
            meta = self.meta.get(article_id, {})
            results.append({"article_id": article_id, "score": float(score), "meta": meta})
            if len(results) >= k:
                break
        return results

    def text_to_vec(self, text, embedder_fn):
        # embedder_fn must return normalized vector (1D numpy)
        return embedder_fn(text)

# Example quick usage:
# searcher = Searcher()
# qvec = your_embedder.encode("Some query text", normalize=True)
# results = searcher.query_topk(qvec, k=10, exclude_ids={"ag_news_10"})
