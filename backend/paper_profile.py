import re
import json
from backend.embeddings import embed_texts
from backend.ollama_client import ask_ollama


def extract_sections(text):
    """
    Very robust section extractor for academic PDFs converted to text.
    Splits text into sections based on common patterns.
    Returns dict: { section_name : section_text }
    """

    # Normalize
    t = text.replace("\r", "")
    lines = t.split("\n")

    sections = {}
    current = None
    buffer = []

    header_pattern = re.compile(r"^\s*(\d+\.?\d*)?\s*(abstract|introduction|background|related work|methods?|methodology|experiments?|results?|discussion|conclusion|references?)\s*$", re.I)

    for line in lines:
        if header_pattern.match(line.strip()):
            # save previous
            if current and buffer:
                sections[current] = "\n".join(buffer).strip()
                buffer = []

            # new header
            current = header_pattern.match(line.strip()).group(2).lower()

        else:
            if current:
                buffer.append(line)

    # final section
    if current and buffer:
        sections[current] = "\n".join(buffer).strip()

    return sections


def build_paper_profile(paper_id, text, title="", abstract=""):
    """
    Full paper semantic profile:
    - extracts sections
    - generates summary
    - generates contributions
    - embeds the global semantic meaning
    - saves a JSON profile
    """

    sections = extract_sections(text)

    intro = sections.get("introduction", "")[:5000]  # limit size
    conclusion = sections.get("conclusion", "")[:5000]

    # Build the base context for LLM
    llm_context = f"""
Title: {title}

Abstract:
{abstract}

Introduction:
{intro}

Conclusion:
{conclusion}
    """

    # Summaries & contributions with Qwen
    summary = ask_ollama(
        "qwen2.5:7b-instruct",
        f"Summarise the key ideas of the following paper in 6-8 sentences:\n\n{llm_context}"
    )

    contributions = ask_ollama(
        "qwen2.5:7b-instruct",
        f"Extract the key contributions from this paper as short bullet points:\n\n{llm_context}"
    )

    # Build semantic profile text
    profile_text = f"""
Title: {title}

Abstract:
{abstract}

Introduction:
{intro}

Conclusion:
{conclusion}

Summary:
{summary}

Contributions:
{contributions}
"""

    # Embed the semantic profile
    embedding = embed_texts([profile_text])[0].tolist()

    profile = {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "summary": summary,
        "contributions": contributions,
        "sections": {
            "introduction": intro,
            "conclusion": conclusion
        },
        "profile_text": profile_text,
        "embedding": embedding
    }

    return profile
