import re

def highlight_evidence(query: str, text: str):
    """
    Simple semantic highlight:
    - Split query into content words
    - Highlight matches in evidence paragraph
    """
    if not query.strip() or not text.strip():
        return text

    # Extract meaningful words (remove stop words)
    words = re.findall(r"\b[a-zA-Z]{4,}\b", query.lower())

    highlighted = text

    for w in words:
        pattern = re.compile(rf"\b({re.escape(w)})\b", re.IGNORECASE)
        highlighted = pattern.sub(r"**\1**", highlighted)

    return highlighted
