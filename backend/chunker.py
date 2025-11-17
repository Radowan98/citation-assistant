def chunk_text(text, max_chars=500):
    """
    Split long text into small chunks.
    Each chunk is at most `max_chars` characters.
    Very simple version — we improve later.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end

    return chunks
