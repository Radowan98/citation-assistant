import os
import json
import numpy as np
import faiss

from backend.embeddings import embed_texts


def load_sentence_index(index_dir="data/sentence_index"):
    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "meta.json")

    if not os.path.exists(index_path):
        os.makedirs(index_dir, exist_ok=True)
        index = faiss.IndexFlatL2(1024)  # BGE-large = 1024 dims
        meta = []
        save_sentence_index(index, meta, index_dir)
        return index, meta

    index = faiss.read_index(index_path)
    meta = json.load(open(meta_path))
    return index, meta


def save_sentence_index(index, meta, index_dir="data/sentence_index"):
    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "meta.json")

    faiss.write_index(index, index_path)
    json.dump(meta, open(meta_path, "w"), indent=2)


def add_sentences(paper_id, sentences, index_dir="data/sentence_index"):
    """
    Add sentence embeddings to the FAISS index.
    """
    index, meta = load_sentence_index(index_dir)

    # Embed sentences
    vectors = embed_texts(sentences)  # shape (n, 1024)
    vectors = vectors.astype("float32")

    index.add(vectors)

    for i, _ in enumerate(sentences):
        meta.append({
            "paper_id": paper_id,
            "sentence_id": i,
            "text": sentences[i]
        })

    save_sentence_index(index, meta, index_dir)

    return len(sentences)


def search_sentences(query, top_k=5, index_dir="data/sentence_index"):
    """
    Query → embedding → FAISS → retrieve top_k sentences.
    """
    index, meta = load_sentence_index(index_dir)
    vec = embed_texts([query]).astype("float32")

    distances, idxs = index.search(vec, top_k)

    results = []
    for dist, idx in zip(distances[0], idxs[0]):
        if idx < len(meta):
            entry = meta[idx]
            entry["distance"] = float(dist)
            results.append(entry)

    return results
