import os
import json
import faiss
import numpy as np


EMBED_DIM = 1024  # BGE-large-en-v1.5


# ------------------------------------------------------
# Load or create FAISS profile index
# ------------------------------------------------------
def load_profile_index(index_dir):
    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "meta.json")

    # If index does not exist, create a new one
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        os.makedirs(index_dir, exist_ok=True)

        index = faiss.IndexFlatL2(EMBED_DIM)
        meta = []
        save_profile_index(index, meta, index_dir)
        return index, meta

    index = faiss.read_index(index_path)
    meta = json.load(open(meta_path))
    return index, meta


# ------------------------------------------------------
# Save FAISS + metadata
# ------------------------------------------------------
def save_profile_index(index, meta, index_dir):
    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "meta.json")

    faiss.write_index(index, index_path)
    json.dump(meta, open(meta_path, "w"), indent=2)


# ------------------------------------------------------
# Add a single profile embedding
# ------------------------------------------------------
def add_profile_entry(paper_id, summary, vector, index_dir):
    index, meta = load_profile_index(index_dir)

    vec = np.array([vector], dtype="float32")
    index.add(vec)

    meta.append({
        "paper_id": paper_id,
        "summary": summary
    })

    save_profile_index(index, meta, index_dir)


# ------------------------------------------------------
# Search profile summaries
# ------------------------------------------------------
def search_profiles(query_text, index_dir, top_k=5, embed_fn=None):
    """
    query_text: string
    embed_fn: function to embed (text -> vector)
    """
    if embed_fn is None:
        from backend.embeddings import embed_texts
        embed_fn = embed_texts

    # Build query vector
    q_emb = embed_fn([query_text])[0].astype("float32").reshape(1, -1)

    # Load index
    index, meta = load_profile_index(index_dir)

    if index.ntotal == 0:
        return []

    # Search
    distances, indices = index.search(q_emb, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(meta):
            results.append({
                "paper_id": meta[idx]["paper_id"],
                "summary": meta[idx]["summary"],
                "distance": float(distances[0][i])
            })

    return results
