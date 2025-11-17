import os
import argparse
import yaml

from backend.pdf_parser import extract_text
from backend.bibtex_parser import parse_bibtex_str
from backend.duplicate_detector import compute_hash, is_duplicate
from backend.paper_store import add_paper_metadata

from backend.utils import process_pdf_text_into_faiss  # paragraph index
from backend.sentence_splitter import split_into_sentences
from backend.sentence_index import add_sentences


# Load config
config = yaml.safe_load(open("config.yaml"))


def main(pdf_path, bib_path):
    # ---------------------------------------------------------
    # 1. Extract text from PDF
    # ---------------------------------------------------------
    print("📄 Extracting text from PDF...")
    text = extract_text(pdf_path)

    if not text or len(text.strip()) < 50:
        print("❌ Error: PDF extraction returned empty or too little text.")
        return

    # ---------------------------------------------------------
    # 2. Compute hash & check duplicates
    # ---------------------------------------------------------
    h = compute_hash(text)

    if is_duplicate(h, config["paths"]["hashes_json"]):
        print("❌ Paper already exists in your library.")
        return

    # ---------------------------------------------------------
    # 3. Load BibTeX
    # ---------------------------------------------------------
    with open(bib_path, "r") as f:
        bib_str = f.read()

    entry = parse_bibtex_str(bib_str)

    # ---------------------------------------------------------
    # 4. Build metadata entry
    # ---------------------------------------------------------
    meta = {
        "id": entry.get("ID", os.path.basename(pdf_path)),
        "title": entry.get("title", ""),
        "year": entry.get("year", ""),
        "hash": h,
        "bibtex": bib_str,
    }

    # Save metadata
    add_paper_metadata(
        meta,
        config["paths"]["papers_json"],
        config["paths"]["hashes_json"],
    )

    print(f"📘 Saved metadata for: {meta['id']}")

    # ---------------------------------------------------------
    # 5. Paragraph-level indexing
    # ---------------------------------------------------------
    print("🔍 Indexing paragraph chunks...")
    chunks_added = process_pdf_text_into_faiss(
        meta["id"],
        text,
        config["paths"]["index_dir"],  # usually data/index
    )
    print(f"📚 Added {chunks_added} paragraph chunks for {meta['id']}")

    # ---------------------------------------------------------
    # 6. Sentence-level indexing (BEST accuracy)
    # ---------------------------------------------------------
    print("✂️ Splitting into sentences...")
    sentences = split_into_sentences(text)

    print("🧠 Embedding sentences...")
    num_sent = add_sentences(
        meta["id"],
        sentences,
        "data/sentence_index",
    )
    print(f"📝 Added {num_sent} sentence embeddings for {meta['id']}")

    # ---------------------------------------------------------
    # 7. Done!
    # ---------------------------------------------------------
    print(f"✅ Paper fully processed and indexed: {meta['id']}")


# ---------------------------------------------------------
# CLI ENTRY
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--bib", required=True)
    args = parser.parse_args()
    main(args.pdf, args.bib)
