import json
import os


def _safe_load_json(path):
    """Load JSON file or return empty list if missing."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def add_paper_metadata(meta: dict, papers_path: str, hashes_path: str):
    """Save metadata + hash to JSON files."""

    # Load existing metadata
    papers = _safe_load_json(papers_path)
    hashes = _safe_load_json(hashes_path)

    # Append new entries
    papers.append(meta)
    hashes.append(meta["hash"])

    # Save updated metadata
    with open(papers_path, "w") as f:
        json.dump(papers, f, indent=2)

    with open(hashes_path, "w") as f:
        json.dump(hashes, f, indent=2)


def get_bibtex_for_paper(paper_id: str, papers_path: str):
    """Return the BibTeX string for a given paper ID."""
    papers = _safe_load_json(papers_path)

    for p in papers:
        if p["id"] == paper_id:
            return p.get("bibtex", None)

    return None


def load_all_papers(papers_path: str):
    """Load full paper metadata list."""
    return _safe_load_json(papers_path)
