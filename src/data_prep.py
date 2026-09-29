# src/data_prep.py
import re
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import html

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)                   # strip HTML
    text = re.sub(r"\s+", " ", text)                      # normalize whitespace
    text = re.sub(r"[^\x00-\x7F]+"," ", text)             # remove non-ascii (or keep)
    text = text.strip()
    return text

def convert_csv_to_jsonl(csv_path, source_name, out_name=None, title_col='title', content_col='content', id_col=None, date_col='published_date', url_col='url', category_col='category'):
    df = pd.read_csv(csv_path)
    out_path = OUT_DIR / (out_name or (Path(csv_path).stem + ".jsonl"))
    with open(out_path, 'w', encoding='utf-8') as fout:
        for i, row in df.iterrows():
            article_id = str(row[id_col]) if id_col and not pd.isna(row[id_col]) else f"{source_name}_{i}"
            title = clean_text(row.get(title_col, "") or "")
            content = clean_text(row.get(content_col, "") or "")
            category = row.get(category_col, "") if category_col in row.index else ""
            published_date = row.get(date_col, "") if date_col in row.index else ""
            url = row.get(url_col, "") if url_col in row.index else ""
            obj = {
                "article_id": article_id,
                "title": title,
                "content": content,
                "category": category,
                "source": source_name,
                "published_date": published_date,
                "url": url
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print("Wrote", out_path)

if __name__ == "__main__":
    # Example: adapt columns depending on dataset
    convert_csv_to_jsonl("data/raw/ag_news.csv", source_name="ag_news", out_name="ag_news.jsonl", title_col='title', content_col='description')
