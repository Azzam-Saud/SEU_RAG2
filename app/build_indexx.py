import os
import pickle
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from loaders import extract_txt_from_files, extract_word, extract_excel
from chunking import chunk_policy_qna_articles

BASE_DIR = r"D:\Azzam\Personal_Projects\SEU\filtered_data"
DIR1 = os.path.join(BASE_DIR, "Word_Excel")
DIR2 = os.path.join(BASE_DIR, "txt")

INDEX_PATH = "faiss.index"
META_PATH = "faiss_meta.pkl"

def safe_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def safe_chunk(text, max_chars=3000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

records = []

def process_dir(folder):
    if not os.path.exists(folder):
        return
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        ext = fname.split(".")[-1].lower()

        if ext == "txt":
            recs = extract_txt_from_files(path, fname)
        elif ext == "docx":
            recs = extract_word(path, fname)
        elif ext in ["xlsx", "xls"]:
            recs = extract_excel(path, fname)
        else:
            continue
        records.extend(recs)

process_dir(DIR1)
process_dir(DIR2)

# -------- Chunking --------
chunks = []
for rec in records:
    parts = safe_chunk(rec["text"]) if rec["type"] == "excel" \
            else chunk_policy_qna_articles(rec["text"])

    for i, ch in enumerate(parts):
        chunks.append({
            "id": f"{rec['id']}__chunk_{i}",
            "text": ch
        })

texts = [c["text"] for c in chunks]

# -------- Embeddings --------
model = SentenceTransformer("intfloat/multilingual-e5-base")

embs = model.encode(
    ["passage: " + t for t in texts],
    normalize_embeddings=True,
    show_progress_bar=True
)

embs = np.asarray(embs).astype("float32")

# -------- FAISS --------
dim = embs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embs)

faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump(chunks, f)

print(f"FAISS index built successfuily! ")