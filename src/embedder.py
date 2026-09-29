# src/embedder.py
import json
from pathlib import Path
from transformers import DistilBertTokenizerFast, DistilBertModel
import torch
import numpy as np
from tqdm import tqdm

DATA_PATH = Path("data/processed/ag_news.jsonl")   # example
OUT_EMB = Path("data/embeddings/ag_news_embeddings.npy")
OUT_MAP = Path("data/embeddings/ag_news_id_map.json")

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
POOLING = "mean"  # options: 'cls', 'mean', 'max'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
model = DistilBertModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def pool_output(outputs, attention_mask, strategy="cls"):
    last_hidden = outputs.last_hidden_state  # (batch, seq_len, dim)
    if strategy == "cls":
        return last_hidden[:,0,:]
    elif strategy == "mean":
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = (last_hidden * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts
    elif strategy == "max":
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size())
        last_hidden[~mask.bool()] = -1e9
        return last_hidden.max(1).values
    else:
        raise ValueError(strategy)

def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms==0] = 1e-9
    return vecs / norms

def generate_embeddings(jsonl_path, out_emb_path, out_map_path):
    ids = []
    texts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            text = (obj.get("title","") + " . " + obj.get("content","")).strip()
            ids.append(obj["article_id"])
            texts.append(text)

    all_embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
        pooled = pool_output(outputs, enc["attention_mask"], strategy=POOLING)
        pooled = pooled.cpu().numpy()
        all_embs.append(pooled)

    all_embs = np.vstack(all_embs)
    all_embs = normalize(all_embs)
    np.save(out_emb_path, all_embs)
    with open(out_map_path, 'w', encoding='utf-8') as f:
        json.dump(ids, f)
    print("Saved embeddings:", out_emb_path, "and id map:", out_map_path)

if __name__ == "__main__":
    Path("data/embeddings").mkdir(parents=True, exist_ok=True)
    generate_embeddings(DATA_PATH, OUT_EMB, OUT_MAP)
