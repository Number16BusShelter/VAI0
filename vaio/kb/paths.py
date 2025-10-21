from __future__ import annotations
from pathlib import Path
import hashlib

# Repo root: two levels up from this file (vaio/kb/…)
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_KB_DIR = REPO_ROOT / "knowledge" / "default"
print("DEFAULT_KB_DIR", DEFAULT_KB_DIR)
DATA_DIR = REPO_ROOT / "data" / "kb"    # chroma persistence root

DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CHROMA_DIR = "data/kb"

def ensure_default_dirs():
    DEFAULT_KB_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def kb_collection_name(kb_dir: Path) -> str:
    """Stable per-knowledge-dir collection name."""
    key = str(kb_dir.resolve()).encode("utf-8")
    return "kb_" + hashlib.md5(key).hexdigest()[:16]
