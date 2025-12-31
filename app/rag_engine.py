import faiss
import pickle
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from app.config import OPENAI_API_KEY

FAISS_INDEX_PATH = "app/data/faiss.index"
FAISS_META_PATH = "app/data/faiss_meta.pkl"

@lru_cache
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-base")

@lru_cache
def load_faiss():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta["texts"]

client = OpenAI(api_key=OPENAI_API_KEY)

# ========= Search =========
def search(query, k=10):
    model = load_model()
    index, texts = load_faiss()

    q_emb = model.encode(
        ["query: " + query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)

    return [
        {"score": float(D[0][i]), "text": texts[idx]}
        for i, idx in enumerate(I[0])
    ]

# ========= RAG =========
def rag_llm_answer(query: str):
    results = search(query)
    context = "\n\n".join(r["text"] for r in results)

    prompt = f"""
أجب فقط من النص التالي.
إذا لم تجد الإجابة قل: لا أعلم.

السؤال:
{query}

السياق:
{context}

الإجابة:
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return res.choices[0].message.content.strip()
