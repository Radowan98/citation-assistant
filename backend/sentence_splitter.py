import re

def split_into_sentences(text: str):
    """
    Reliable sentence splitter using simple heuristics.
    No heavy NLP libraries needed.

    Works well for academic PDFs.
    """
    if not text:
        return []

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Split on ., ?, !
    raw = re.split(r'(?<=[.!?])\s+', text)

    # Clean & filter
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) > 20:  # avoid junk or extremely short lines
            sentences.append(s)

    return sentences
