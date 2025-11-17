import re


def split_into_paragraphs(text, min_chars: int = 150):
    """
    Split full paper text into reasonably clean paragraphs.

    - Splits on blank lines / multiple newlines
    - Strips whitespace
    - Filters out very short fragments (e.g. headings, single lines)
    - Returns list of paragraph strings
    """
    # Normalise newlines
    t = text.replace("\r", "\n")

    # Split on 2+ newlines -> paragraph blocks
    raw_paragraphs = re.split(r"\n\s*\n", t)

    paragraphs = []
    for p in raw_paragraphs:
        p_clean = p.strip()

        # Skip useless fragments
        if len(p_clean) < min_chars:
            continue

        # Collapse internal excessive whitespace
        p_clean = re.sub(r"\s+", " ", p_clean)

        paragraphs.append(p_clean)

    return paragraphs


def make_paragraph_entries(paper_id: str, text: str, min_chars: int = 150):
    """
    Turn full text into a list of paragraph entries with IDs.

    Returns:
        [
          { "paper_id": ..., "para_id": 0, "text": "..." },
          { "paper_id": ..., "para_id": 1, "text": "..." },
          ...
        ]
    """
    paras = split_into_paragraphs(text, min_chars=min_chars)

    entries = []
    for i, p in enumerate(paras):
        entries.append({
            "paper_id": paper_id,
            "para_id": i,
            "text": p,
        })

    return entries
