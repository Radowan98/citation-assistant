from pdfminer.high_level import extract_text as pdfminer_extract
import re

def remove_references(text: str) -> str:
    """
    Remove the references/bibliography section from PDF text.
    """
    patterns = [
        r'\bREFERENCES\b',
        r'\bReferences\b',
        r'\bREFERENCE\b',
        r'\bReference\b',
        r'\bBIBLIOGRAPHY\b',
        r'\bBibliography\b',
    ]

    for p in patterns:
        match = re.search(p, text)
        if match:
            return text[:match.start()].strip()

    return text


def extract_text(pdf_path: str) -> str:
    """
    Extract text using pdfminer and remove references section.
    """
    try:
        full_text = pdfminer_extract(pdf_path)
        cleaned = remove_references(full_text)
        return cleaned
    except Exception as e:
        print("PDF extraction failed:", e)
        return ""
