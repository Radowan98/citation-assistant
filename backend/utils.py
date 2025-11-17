from backend.chunker import chunk_text
from backend.embeddings import embed_texts
from backend.index_manager import add_vectors


def chunk_and_embed(text):
    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    return chunks, vectors


def add_chunks_to_faiss(paper_id, chunks, vectors, index_dir):
    meta_entries = [
        {"paper_id": paper_id, "chunk_id": i, "text": chunks[i]}
        for i in range(len(chunks))
    ]
    add_vectors(None, vectors, meta_entries, index_dir)


def process_pdf_text_into_faiss(paper_id, text, index_dir):
    chunks, vectors = chunk_and_embed(text)
    add_chunks_to_faiss(paper_id, chunks, vectors, index_dir)
    return len(chunks)
