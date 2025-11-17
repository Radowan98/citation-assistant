import yaml
import json
import re
import numpy as np
import ollama

from backend.sentence_index import search_sentences
from backend.paper_store import load_all_papers
from backend.embeddings import embed_texts
from backend.index_manager import search_index
from backend.reranker import rerank


CONFIG = yaml.safe_load(open("config.yaml"))


def llm_extract_evidence(query, sentence):
    prompt = f"""
You are an expert academic research assistant.

Your task: Given a query and an evidence passage, identify the precise span
inside the evidence that supports the query.

Return STRICT JSON with fields:
- "highlighted": the evidence passage with the matching span highlighted using **bold**
- "reason": one sentence explaining why the span supports the query

Requirements:
- If the match is conceptual (not literal), still highlight it
- Never change wording, only wrap the matching span in ** **
- If nothing matches, return the original passage as "highlighted"

QUERY:
{query}

EVIDENCE:
{sentence}
"""

    try:
        res = ollama.chat(
            model="qwen2.5:7b-instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        out = res["message"]["content"]
        json_str = re.search(r"\{.*\}", out, re.DOTALL)
        if json_str:
            return json.loads(json_str.group())
        return {"highlighted": sentence, "reason": ""}
    except Exception:
        return {"highlighted": sentence, "reason": ""}


def suggest_papers(query, top_k=5, sent_k=50, chunk_k=50, max_evidences=40, max_evidences_per_paper=5):
    try:
        q_emb = embed_texts([query])[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
    except Exception:
        q_emb = None

    sent_hits = search_sentences(
        query,
        top_k=sent_k,
        index_dir=CONFIG["paths"]["sentence_index"]
    )

    chunk_hits = []
    if q_emb is not None:
        chunk_hits = search_index(
            q_emb,
            index_dir=CONFIG["paths"]["index_dir"],
            top_k=chunk_k
        )

    items = []

    for h in sent_hits:
        text = h.get("text", "")
        if not text:
            continue
        items.append({
            "text": text,
            "paper_id": h.get("paper_id"),
            "source": "sentence",
            "base_distance": h.get("distance", 0.0),
        })

    for h in chunk_hits:
        meta = h.get("meta", {})
        text = meta.get("text", "")
        if not text:
            continue
        items.append({
            "text": text,
            "paper_id": meta.get("paper_id"),
            "source": "chunk",
            "base_distance": h.get("distance", 0.0),
        })

    if not items:
        return []

    ranked_items = rerank(query, items, text_key="text")
    ranked_items = ranked_items[:max_evidences]

    grouped = {}

    for it in ranked_items:
        pid = it.get("paper_id")
        if not pid:
            continue

        if pid not in grouped:
            grouped[pid] = {
                "title": "",
                "bibtex": "",
                "evidences": [],
                "highlighted_evidences": [],
                "evidence_reasons": [],
                "scores": [],
            }

        if len(grouped[pid]["evidences"]) >= max_evidences_per_paper:
            continue

        text = it["text"]
        score = float(it.get("rerank_score", 0.0))

        hl_obj = llm_extract_evidence(query, text)

        grouped[pid]["evidences"].append(text)
        grouped[pid]["highlighted_evidences"].append(hl_obj.get("highlighted", text))
        grouped[pid]["evidence_reasons"].append(hl_obj.get("reason", ""))
        grouped[pid]["scores"].append(score)

    papers = load_all_papers(CONFIG["paths"]["papers_json"])
    meta_map = {p["id"]: p for p in papers}

    for pid in grouped:
        if pid in meta_map:
            grouped[pid]["title"] = meta_map[pid].get("title", "")
            grouped[pid]["bibtex"] = meta_map[pid].get("bibtex", "")

    sorted_papers = sorted(
        grouped.items(),
        key=lambda x: max(x[1]["scores"]) if x[1]["scores"] else 0.0,
        reverse=True
    )

    return sorted_papers[:top_k]
