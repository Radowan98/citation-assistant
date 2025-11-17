import os
import json
import faiss
import numpy as np

from backend.embeddings import embed_texts


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def init_para_index(index_dir: str, dim: int):
    """
    Create an empty FAISS index for paragraph embeddings.
    """
    _ensure_dir(index_dir)

    index_path = os.path.join(index_dir, "para.index")
    meta_path = os.path.join(index_dir, "para_meta.json")

    if os.path.exists(index_path) and os.path.exists(meta_path):
        return

    index = faiss.IndexFlatL2(dim)

    faiss.write_index(index, index_path)
    with open(meta_path, "w") as f:
        json.dump([], f)


def load_para_index(index_dir: str):
    index_path = os.path.join(index_dir, "para.index")
    meta_path = os.path.join(index_dir, "para_meta.json")

    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("Paragraph index not initialised.")

    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    return index, meta


def save_para_index(index, meta, index_dir: str):
    index_path = os.path.join(index_dir, "para.index")
    meta_path = os.path.join(index_dir, "para_meta.json")

    faiss.write_index(index, index_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def add_paragraphs(entries, index_dir: str):
    """
    entries: list of dicts:
      { "paper_id": ..., "para_id": ..., "text": ... }

    Adds embeddings + metadata to paragraph FAISS index.
    """
    texts = [e["text"] for e in entries]

    # embed using BGE-Large
    vectors = embed_texts(texts).astype("float32")
    dim = vectors.shape[1]

    # ensure index exists
    init_para_index(index_dir, dim)

    index, meta = load_para_index(index_dir)

    # add vectors
    index.add(vectors)

    # add metadata
    for e in entries:
        meta.append({
            "paper_id": e["paper_id"],
            "para_id": e["para_id"],
            "text": e["text"],
        })

    # save
    save_para_index(index, meta, index_dir)


def search_paragraphs(query_text, index_dir: str, top_k: int = 5):
    """Return top K most relevant paragraphs."""
    index, meta = load_para_index(index_dir)

    q_vec = embed_texts([query_text])[0].astype("float32").reshape(1, -1)

    scores, ids = index.search(q_vec, top_k)

    results = []
    for s, idx in zip(scores[0], ids[0]):
        if idx >= 0 and idx < len(meta):
            results.append({
                "score": float(s),
                "paper_id": meta[idx]["paper_id"],
                "para_id": meta[idx]["para_id"],
                "text": meta[idx]["text"],
            })

    return results
