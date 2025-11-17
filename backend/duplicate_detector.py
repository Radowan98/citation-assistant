# placeholder
import hashlib
import json

def compute_hash(text: str) -> str:
    """Return a SHA-256 hash of the PDF text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def is_duplicate(hash_value: str, hashes_json_path: str) -> bool:
    """Check if this hash already exists in hashes.json."""
    with open(hashes_json_path, "r") as f:
        hashes = json.load(f)
    return hash_value in hashes

