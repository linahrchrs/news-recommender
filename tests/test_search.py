# tests/test_search.py
from src.search import Searcher
import numpy as np
s = Searcher()
# use first embedding as query
emb = np.load("data/embeddings/ag_news_embeddings.npy")
q = emb[0]
res = s.query_topk(q, k=5, exclude_ids=set())
assert len(res) >= 1
print("search test passed")
