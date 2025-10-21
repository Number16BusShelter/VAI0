# paths.py
from __future__ import annotations
from pathlib import Path

# repo root = VAIO/
REPO_ROOT = Path(__file__).resolve().parents[2]

# Knowledge source directories (user-provided content)
BASE_KB_ROOT = REPO_ROOT / "knowledge"
DEFAULT_KB_DIR = BASE_KB_ROOT / "default"

# ChromaDB persistence directory (centralized)
DATA_ROOT = REPO_ROOT / "data" / "kb"
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def ensure_default_dirs() -> None:
    """Create necessary directories if they don't exist"""
    DEFAULT_KB_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)