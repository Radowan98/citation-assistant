import os
import json
import yaml
from tqdm import tqdm

from backend.pdf_parser import extract_text
from backend.sentence_splitter import split_into_sentences
from backend.embeddings import embed_texts
from backend.profile_index import add_profile_entry
from backend.paper_store import load_all_papers


# ------------------------------
# Section extractor (simple)
# ------------------------------
def extract_sections(text: str):
    """
    Split text into simple sections using keyword heuristics.
    Not perfect, but works well enough for semantic profiles.
    """
    lowered = text.lower()

    sections = {
        "introduction": "",
        "method": "",
        "results": "",
        "conclusion": "",
    }

    # Simple segmentation by keywords
    intro_idx = lowered.find("introduction")
    meth_idx  = lowered.find("method")
    res_idx   = lowered.find("result")
    concl_idx = lowered.find("conclusion")

    # introduction → method
    if intro_idx != -1 and meth_idx != -1:
        sections["introduction"] = text[intro_idx:meth_idx]

    # method → results
    if meth_idx != -1 and res_idx != -1:
        sections["method"] = text[meth_idx:res_idx]

    # results → conclusion
    if res_idx != -1 and concl_idx != -1:
        sections["results"] = text[res_idx:concl_idx]

    # conclusion → end
    if concl_idx != -1:
        sections["conclusion"] = text[concl_idx:]

    return sections


# ------------------------------
# Summary builder
# ------------------------------
def build_summary(text: str):
    """
    Summary is first 5 sentences (good enough for document-level embedding).
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return ""
    summary = " ".join(sentences[:5])
    return summary


# ------------------------------
# Main builder
# ------------------------------
def main():
    config = yaml.safe_load(open("config.yaml"))
    papers = load_all_papers(config["paths"]["papers_json"])

    profiles_dir = config["paths"]["profiles_dir"]
    index_dir = config["paths"]["profile_index"]

    os.makedirs(profiles_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    print(f"📘 Building semantic profiles for {len(papers)} papers...\n")

    for p in tqdm(papers):
        pid = p["id"]

        # Locate original PDF
        pdf_path_candidates = [
            f"{pid}.pdf",
            os.path.join("pdfs", f"{pid}.pdf"),
            os.path.join("papers", f"{pid}.pdf"),
        ]

        pdf_path = None
        for c in pdf_path_candidates:
            if os.path.exists(c):
                pdf_path = c
                break

        if pdf_path is None:
            print(f"⚠️ PDF not found for paper ID {pid}, skipping.")
            continue

        # Extract cleaned text
        text = extract_text(pdf_path)
        if not text.strip():
            print(f"⚠️ Empty text extracted for {pid}, skipping.")
            continue

        # Build sections + summary
        sections = extract_sections(text)
        summary = build_summary(text)

        # Save profile JSON
        profile_path = os.path.join(profiles_dir, f"{pid}.json")
        json.dump(
            {"summary": summary, "sections": sections},
            open(profile_path, "w"),
            indent=2
        )

        # Build embedding for summary
        emb = embed_texts([summary])[0]  # shape (1024,)
        add_profile_entry(pid, summary, emb, index_dir)  # adds to FAISS + meta.json

    print("\n✅ Done! All profiles built and indexed.")


if __name__ == "__main__":
    main()
