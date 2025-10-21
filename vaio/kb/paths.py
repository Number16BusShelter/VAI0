from __future__ import annotations
from pathlib import Path

# repo root = VAI0/
REPO_ROOT = Path(__file__).resolve().parents[2]

# All KB lives under <repo_root>/kb
BASE_KB_ROOT = REPO_ROOT / "kb"
DEFAULT_KB_DIR = BASE_KB_ROOT / "default"
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CHROMA_DIR = "data/kb"

def ensure_default_dirs() -> None:
    DEFAULT_KB_DIR.mkdir(parents=True, exist_ok=True)
