import torch
from sentence_transformers import CrossEncoder

# Best lightweight cross-encoder for deep relevance scoring
_RERANK_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def score_pairs(query, texts):
    """
    Compute semantic relevance scores between one query and a list of texts
    using a cross-encoder.
    
    query: string
    texts: list of strings
    returns: list of floats (higher = more relevant)
    """
    if isinstance(texts, str):
        texts = [texts]

    # Prepare input pairs
    pairs = [(query, t) for t in texts]

    # Cross-encoder outputs relevance scores
    scores = _RERANK_MODEL.predict(pairs)

    # Convert to Python floats for JSON exporting later
    scores = [float(s) for s in scores]

    return scores


def rerank(query, items, text_key="text"):
    """
    Rerank a list of dict items based on semantic relevance.
    
    items: [{ "text": "...", <other keys> }, ...]
    text_key: which field to use for reranking
    
    returns: same items, sorted by relevance desc
    """
    texts = [item[text_key] for item in items]

    scores = score_pairs(query, texts)

    # attach scores
    for item, score in zip(items, scores):
        item["rerank_score"] = score

    # sort by score (descending)
    return sorted(items, key=lambda x: x["rerank_score"], reverse=True)
