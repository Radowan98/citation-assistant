import os
import json
import faiss
import numpy as np


def load_index(index_dir: str):
    """
    Load FAISS index if exists, otherwise return None.
    We no longer assume dimension or create index here.
    """
    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "meta.json")

    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        return None, []

    index = faiss.read_index(index_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    return index, meta


def save_index(index, meta, index_dir: str):
    """Save FAISS index + metadata."""
    os.makedirs(index_dir, exist_ok=True)

    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "meta.json")

    faiss.write_index(index, index_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def create_new_index(dim: int):
    """
    Create a new FAISS index with the correct dimension.
    Always use L2 for semantic search.
    """
    return faiss.IndexFlatL2(dim)


def add_vectors(index, vectors: np.ndarray, meta_entries: list, index_dir: str):
    """
    Add vectors to FAISS. Create new index if needed.
    Automatically verifies correct dimensions.
    """

    # Load existing index OR None
    existing_index, existing_meta = load_index(index_dir)

    dim = vectors.shape[1]

    if existing_index is None:
        # create a new index
        index = create_new_index(dim)
        meta = []
    else:
        # existing index
        index = existing_index
        meta = existing_meta

        # dimension check
        if index.d != dim:
            raise ValueError(
                f"Embedding dimension mismatch: index.d={index.d}, vectors={dim}. "
                f"You need to delete the old index at: {index_dir}"
            )

    # Add vectors
    index.add(vectors)

    # Update metadata
    meta.extend(meta_entries)

    # save index + metadata
    save_index(index, meta, index_dir)


def search_index(query_vector: np.ndarray, index_dir: str, top_k=5):
    """Search FAISS index."""
    index, meta = load_index(index_dir)

    if index is None:
        return []

    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    distances, indices = index.search(query_vector.astype("float32"), top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(meta):
            results.append({
                "distance": float(distances[0][i]),
                "meta": meta[idx],
            })

    return results
