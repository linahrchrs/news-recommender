# tests/test_embedding.py
import numpy as np
emb = np.load("data/embeddings/ag_news_embeddings.npy")
assert emb.ndim == 2
norms = np.linalg.norm(emb, axis=1)
assert np.all(norms > 0.9) and np.all(norms < 1.1)
print("embedding tests passed")
