# placeholder
import bibtexparser

def parse_bibtex_str(bib_str):
    parsed = bibtexparser.loads(bib_str)
    return parsed.entries[0]  # return the first (and usually only) entry

