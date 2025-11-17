import streamlit as st
import tempfile
import os
import yaml

from backend.suggestor_v3 import suggest_papers
from backend.pdf_parser import extract_text
from backend.bibtex_parser import parse_bibtex_str
from backend.duplicate_detector import compute_hash, is_duplicate
from backend.paper_store import add_paper_metadata, load_all_papers
from backend.utils import process_pdf_text_into_faiss
from backend.profile_index import add_profile_entry
from backend.embeddings import embed_texts
from backend.sentence_splitter import split_into_sentences
from backend.sentence_index import add_sentences


CONFIG = yaml.safe_load(open("config.yaml"))

st.set_page_config(page_title="Local Citation Assistant", layout="wide")

with st.sidebar:
    st.title("📚 Local Citation Assistant")
    menu = st.radio("Menu", ["Search", "Add Paper", "Library"])
    st.markdown("---")


if menu == "Search":
    st.title("🔎 Search Your Personal Research Library")

    query = st.text_area(
        "Your statement / query",
        placeholder="e.g., Cross-project vulnerability detection remains challenging due to domain shift...",
        height=130,
    )

    top_k = st.slider("Number of papers to return", 1, 10, 5)

    results = None

    if st.button("Find References", type="primary"):
        if not query.strip():
            st.warning("Please enter text to search.")
        else:
            with st.spinner("Searching your library..."):
                try:
                    results = suggest_papers(query, top_k=top_k)
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    results = []

    st.markdown("---")

    if results is None:
        st.info("No results yet.")
    elif not results:
        st.info("No relevant results found.")
    else:
        st.subheader("📚 Suggested References")
        for paper_id, paper_data in results:
            title = paper_data.get("title") or "(Untitled paper)"
            bib = paper_data.get("bibtex", "")
            evidences = paper_data.get("highlighted_evidences", [])
            reasons = paper_data.get("evidence_reasons", [])
            scores = paper_data.get("scores", [])

            label = f"{title} · ID: {paper_id}"
            with st.expander(label):
                if evidences:
                    tabs = st.tabs([f"Evidence {i+1}" for i in range(len(evidences))])
                    for i, tab in enumerate(tabs):
                        with tab:
                            st.markdown(evidences[i])
                            if i < len(reasons) and reasons[i]:
                                st.caption("Why this matches: " + reasons[i])
                            if i < len(scores):
                                st.caption(f"Relevance score: {scores[i]:.4f}")
                else:
                    st.write("No evidence snippets available.")

                st.markdown("### 📄 BibTeX")
                st.code(bib or "_No BibTeX available._", language="text")


elif menu == "Add Paper":
    st.title("📥 Add a New Paper to Your Library")

    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    bib_file = st.file_uploader("Upload BibTeX (.bib or .txt)", type=["bib", "txt"])

    if st.button("Add Paper", type="primary"):
        if pdf_file is None or bib_file is None:
            st.warning("Please upload BOTH a PDF and a BibTeX file.")
        else:
            with st.spinner("Processing paper..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(pdf_file.read())
                    pdf_path = tmp_pdf.name

                with tempfile.NamedTemporaryFile(delete=False, suffix=".bib") as tmp_bib:
                    tmp_bib.write(bib_file.read())
                    bib_path = tmp_bib.name

                try:
                    text = extract_text(pdf_path)
                    if not text or not text.strip():
                        st.error("❌ Could not extract text from PDF.")
                        st.stop()

                    h = compute_hash(text)
                    if is_duplicate(h, CONFIG["paths"]["hashes_json"]):
                        st.error("❌ This paper already exists in your library.")
                        st.stop()

                    bib_str = open(bib_path).read()
                    entry = parse_bibtex_str(bib_str)
                    paper_id = entry.get("ID", f"paper_{h[:8]}")

                    meta = {
                        "id": paper_id,
                        "title": entry.get("title", ""),
                        "year": entry.get("year", ""),
                        "hash": h,
                        "bibtex": bib_str,
                    }

                    add_paper_metadata(
                        meta,
                        CONFIG["paths"]["papers_json"],
                        CONFIG["paths"]["hashes_json"],
                    )

                    process_pdf_text_into_faiss(
                        paper_id,
                        text,
                        CONFIG["paths"]["index_dir"],
                    )

                    sentences = split_into_sentences(text)

                    if sentences:
                        os.makedirs(CONFIG["paths"]["sentence_index"], exist_ok=True)
                        add_sentences(paper_id, sentences, CONFIG["paths"]["sentence_index"])
                        summary = " ".join(sentences[:5])
                        if summary.strip():
                            v = embed_texts([summary])[0]
                            add_profile_entry(
                                paper_id,
                                summary,
                                v,
                                CONFIG["paths"]["profile_index"],
                            )

                    st.success(f"✅ Paper added successfully: {paper_id}")

                finally:
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    if os.path.exists(bib_path):
                        os.remove(bib_path)


elif menu == "Library":
    st.title("📚 Your Paper Library")

    papers = load_all_papers(CONFIG["paths"]["papers_json"])

    if not papers:
        st.info("Your library is empty.")
    else:
        st.write(f"Total papers: **{len(papers)}**")
        st.markdown("---")

        for p in papers:
            title = p.get("title", "(Untitled)")
            paper_id = p.get("id", "")
            year = p.get("year", "")

            header = f"{title} ({year}) · {paper_id}"
            with st.expander(header):
                st.markdown(f"**Title:** {title}")
                st.markdown(f"**Year:** {year}")
                st.markdown(f"**ID:** `{paper_id}`")
                st.markdown("### BibTeX")
                st.code(p.get("bibtex", ""), language="text")
