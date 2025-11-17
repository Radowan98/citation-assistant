import json

from backend.profile_index import search_profiles
from backend.paragraph_index import search_paragraphs
from backend.paper_store import load_all_papers
from backend.reranker import rerank
from backend.ollama_client import ask_ollama
from backend.highlight import highlight_evidence


# -------------------------------------------------------------
# Metadata
# -------------------------------------------------------------

def load_metadata(meta_path="data/library/papers.json"):
    papers = load_all_papers(meta_path)
    return {p["id"]: p for p in papers}


# -------------------------------------------------------------
# Retrieval
# -------------------------------------------------------------

def get_profile_candidates(query, top_k=5):
    try:
        return search_profiles(query, "data/profile_index", top_k=top_k)
    except FileNotFoundError:
        return []


def get_paragraph_candidates(query, top_k=8):
    try:
        return search_paragraphs(query, "data/paragraph_index", top_k=top_k)
    except FileNotFoundError:
        return []


# -------------------------------------------------------------
# Fusion of profile-level + paragraph-level hits
# -------------------------------------------------------------

def fuse_candidates(profile_hits, para_hits):
    fused = {}

    # Document-level
    for p in profile_hits:
        pid = p["paper_id"]
        fused.setdefault(pid, {
            "paper_id": pid,
            "title": p.get("title", ""),
            "profile_score": p["score"],
            "best_para": None,
            "best_para_score": float("inf"),
        })

    # Paragraph-level
    for ph in para_hits:
        pid = ph["paper_id"]
        fused.setdefault(pid, {
            "paper_id": pid,
            "title": "",
            "profile_score": float("inf"),
            "best_para": None,
            "best_para_score": float("inf"),
        })

        if ph["score"] < fused[pid]["best_para_score"]:
            fused[pid]["best_para_score"] = ph["score"]
            fused[pid]["best_para"] = ph

    # Sort by (paragraph first, then document profile)
    fused_list = list(fused.values())
    fused_list.sort(key=lambda x: (x["best_para_score"], x["profile_score"]))

    return fused_list


# -------------------------------------------------------------
# Cross-encoder reranking
# -------------------------------------------------------------

def rerank_final(query, fused_list):
    items = []
    for f in fused_list:
        text = f["best_para"]["text"] if f["best_para"] else f["title"]
        items.append({
            "paper_id": f["paper_id"],
            "evidence_text": text,
            "title": f["title"],
        })

    return rerank(query, items, text_key="evidence_text")


# -------------------------------------------------------------
# Qwen explanation
# -------------------------------------------------------------

def get_relevance_explanation(query, evidence_text):
    if not evidence_text.strip():
        return ""

    prompt = f"""
You are an expert researcher. Explain clearly and concisely how the following
evidence supports or relates to the user's research query.

Focus on:
- the conceptual connection
- the shared problem framing
- why this evidence is relevant for citation

Avoid repeating the query; focus on reasoning.

Query:
{query}

Evidence from the paper:
{evidence_text}

Provide a precise explanation in 3–5 sentences.
"""

    try:
        return ask_ollama("qwen2.5:7b-instruct", prompt).strip()
    except Exception:
        return ""


# -------------------------------------------------------------
# Main Citation Suggestion Function
# -------------------------------------------------------------

def suggest_papers(query, top_k=5):
    # 1. Search both indexes
    profile_hits = get_profile_candidates(query, top_k=top_k)
    para_hits = get_paragraph_candidates(query, top_k=top_k * 2)

    if not profile_hits and not para_hits:
        return []

    # 2. Fuse
    fused = fuse_candidates(profile_hits, para_hits)

    # 3. Cross-encoder reranker
    reranked = rerank_final(query, fused[: top_k * 2])

    # 4. Metadata
    meta = load_metadata()

    results = []
    for r in reranked[:top_k]:
        pid = r["paper_id"]
        paper = meta.get(pid, {})

        # 5. Evidence selection
        evidence = r["evidence_text"]

        # If evidence is too weak (e.g., just the title), fallback to profile summary/introduction
        if len(evidence.split()) < 6:
            try:
                prof_path = f"data/profiles/{pid}.json"
                with open(prof_path, "r") as f:
                    prof = json.load(f)

                fallback = (
                    prof.get("summary")
                    or prof["sections"].get("introduction", "")
                )

                if fallback:
                    evidence = fallback

            except Exception:
                pass

        # 6. Explanation
        explanation = get_relevance_explanation(query, evidence)

        # 7. Highlight
        highlighted = highlight_evidence(query, evidence)

        # 8. Final structured result
        results.append({
            "paper_id": pid,
            "title": paper.get("title", ""),
            "bibtex": paper.get("bibtex", ""),
            "evidence": evidence,
            "highlighted_evidence": highlighted,
            "why_relevant": explanation,
            "score": r["rerank_score"],
        })

    return results
