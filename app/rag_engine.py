import faiss
import pickle
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from app.config import OPENAI_API_KEY


# ---------- OpenAI ----------
client = OpenAI(api_key=OPENAI_API_KEY)


# ---------- FAISS + Model (Lazy Loaded) ----------
@lru_cache(maxsize=1)
def get_resources():
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )
    index = faiss.read_index("faiss.index")
    meta = pickle.load(open("meta.pkl", "rb"))
    return model, index, meta


# ---------- OpenAI response parser ----------
def extract_text(response) -> str:
    text = ""
    for item in response.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    text += c.text
    return text.strip() if text.strip() else "لا أعلم."


# ---------- RAG ----------
def rag_llm_answer(query: str) -> str:
    model, index, meta = get_resources()
    texts = meta["texts"]

    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    D, I = index.search(q_emb, 5)
    context = "\n\n".join(texts[i] for i in I[0])

    prompt = f"""
أجب فقط من النص التالي.
إذا لم تجد الإجابة قل: لا أعلم.

السؤال:
{query}

السياق:
{context}

الإجابة:
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return extract_text(response)
