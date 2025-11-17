from sentence_transformers import SentenceTransformer
import numpy as np

# Absolutely the best open-source embedding model for academic retrieval
_EMBED_MODEL = SentenceTransformer("BAAI/bge-large-en-v1.5")

def embed_texts(text_list):
    if isinstance(text_list, str):
        text_list = [text_list]

    embeddings = _EMBED_MODEL.encode(
        text_list,
        convert_to_numpy=True,
        normalize_embeddings=True  # improves cosine similarity performance
    )

    return embeddings.astype("float32")
