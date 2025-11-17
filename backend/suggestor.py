import yaml
from backend.embeddings import embed_texts
from backend.index_manager import search_index
from backend.paper_store import get_bibtex_for_paper
from backend.ollama_client import ask_ollama

config = yaml.safe_load(open("config.yaml"))


# ------------------------------------------------------------
# STEP 1: SIMPLE RETRIEVAL (FAISS)
# ------------------------------------------------------------
def get_similar_papers(query_text, index_dir="data/index", top_k=5):
    vec = embed_texts([query_text])[0]
    raw_results = search_index(vec, index_dir, top_k=top_k)

    paper_ids = []
    seen = set()

    for r in raw_results:
        pid = r["meta"]["paper_id"]
        if pid not in seen:
            paper_ids.append(pid)
            seen.add(pid)

    return paper_ids


# ------------------------------------------------------------
# STEP 2: BASIC (non-LLM) SUGGESTION
# ------------------------------------------------------------
def suggest_basic(query_text, top_k=5):
    paper_ids = get_similar_papers(query_text, "data/index", top_k)

    results = []
    for pid in paper_ids:
        bib = get_bibtex_for_paper(pid, config["paths"]["papers_json"])
        results.append({
            "paper_id": pid,
            "bibtex": bib
        })

    return results


# ------------------------------------------------------------
# STEP 3: EXPLAIN WHY A PAPER IS RELEVANT
# ------------------------------------------------------------
def explain_citation(query_text, paper_title, bibtex, model="qwen2.5:7b"):
    prompt = f"""
You are helping a researcher select citations.

User's text:
{query_text}

Paper title:
{paper_title}

BibTeX:
{bibtex}

Explain in 2–3 sentences why this paper is relevant.
"""
    return ask_ollama(model, prompt).strip()


# ------------------------------------------------------------
# STEP 4: LLM RE-RANKING
# ------------------------------------------------------------
def rerank_with_llm(query_text, candidates, model="qwen2.5:7b"):
    """
    candidates = [
        {"paper_id":..., "title":..., "bibtex":...},
        ...
    ]
    Returns re-ordered list.
    """

    papers_text = ""
    for i, c in enumerate(candidates):
        papers_text += f"""
[{i+1}]
Title: {c['title']}
BibTeX: {c['bibtex']}
"""

    prompt = f"""
You are ranking research papers by conceptual relevance.

User query:
\"\"\"{query_text}\"\"\"

Candidate papers:
{papers_text}

Rank the papers from MOST relevant to LEAST relevant.
Return ONLY a JSON list of indices, like:
[2, 1, 3]
"""

    response = ask_ollama(model, prompt).strip()

    try:
        order = eval(response)
        order = [int(i)-1 for i in order if 0 < int(i) <= len(candidates)]
        return [candidates[i] for i in order]
    except:
        # fail-safe
        return candidates


# ------------------------------------------------------------
# STEP 5: FINAL SMART SUGGESTION PIPELINE
# ------------------------------------------------------------
def suggest_smart(query_text, top_k=5, model="qwen2.5:7b"):
    """
    1. Retrieve using embeddings
    2. Extract bib & title
    3. LLM re-ranking
    4. Generate explanation
    """

    # Step 1: base retrieval
    basic_results = suggest_basic(query_text, top_k)

    # Step 2: construct candidate list
    candidates = []
    for r in basic_results:
        bib = r["bibtex"]

        # Extract title from BibTeX
        title = "Unknown Title"
        try:
            title = bib.split("title={")[1].split("}")[0]
        except:
            pass

        candidates.append({
            "paper_id": r["paper_id"],
            "title": title,
            "bibtex": bib
        })

    # Step 3: LLM re-ranking
    ranked = rerank_with_llm(query_text, candidates, model=model)

    # Step 4: Add LLM explanation
    final = []
    for c in ranked:
        explanation = explain_citation(query_text, c["title"], c["bibtex"], model=model)
        c["explanation"] = explanation
        final.append(c)

    return final
