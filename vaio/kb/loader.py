# loader.py  (replace the whole file with this or update the shown parts)

from __future__ import annotations
from pathlib import Path
import json, csv
from typing import List
from llama_index.core import Document

try:
    import yaml
except ImportError:
    yaml = None

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

def default_kb_root() -> Path:
    return Path("data/kb").resolve()

def kb_path(name: str) -> Path:
    root = default_kb_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / name


# -----------------------------
# Ignore rules
# -----------------------------
IGNORE_NAMES = {
    ".DS_Store", ".gitkeep", ".gitignore", ".env",
    "__pycache__", "node_modules", "tmp", "venv",
}
IGNORE_EXTS = {
    ".db", ".sqlite", ".lock", ".log", ".bak", ".tmp", ".old",
}
SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".json", ".csv", ".yml", ".yaml"}

def _category_for(path: Path) -> str:
    p = str(path).lower()
    if "/reference/" in p or "gia" in p or "guide" in p:
        return "reference"
    if "/marketing/" in p:
        return "marketing"
    return "marketing"

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _read_pdf(path: Path) -> str:
    if not PdfReader:
        print(f"⚠️ PDF support not available (PyPDF2 missing): {path}")
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        print(f"⚠️ Could not read PDF {path}: {e}")
        return ""

def _read_json(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Could not read JSON {path}: {e}")
        return ""

def _read_csv(path: Path) -> str:
    try:
        rows = []
        with open(path, newline="", encoding="utf-8", errors="ignore") as f:
            for row in csv.reader(f):
                rows.append(", ".join(row))
        return "\n".join(rows)
    except Exception as e:
        print(f"⚠️ Could not read CSV {path}: {e}")
        return ""

def _read_yaml(path: Path) -> str:
    if not yaml:
        return _read_text(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return _read_text(path)

def read_file(path: Path) -> str:
    """Public single-file reader with format routing."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"): return _read_text(path)
    if suffix == ".pdf":          return _read_pdf(path)
    if suffix == ".json":         return _read_json(path)
    if suffix == ".csv":          return _read_csv(path)
    if suffix in (".yml", ".yaml"): return _read_yaml(path)
    # Fallback for unknown but text-ish files
    return _read_text(path)

def _should_skip(path: Path) -> bool:
    if not path.is_file():
        return True
    if path.name.startswith("."):
        return path.name != ".DS_Store" and True or False  # handled below by set
    if path.name in IGNORE_NAMES:
        return True
    if path.suffix.lower() in IGNORE_EXTS:
        return True
    if path.name == ".DS_Store":
        return True
    # if you want to strictly enforce supported types:
    if path.suffix and path.suffix.lower() not in SUPPORTED_EXTS:
        return True
    return False

def load_documents(kb_path: Path) -> List[Document]:
    """
    Load supported docs from either:
      - a directory (recursively), or
      - a single file path.
    """
    docs: List[Document] = []
    if not kb_path.exists():
        return docs

    files: List[Path] = []
    if kb_path.is_file():
        files = [kb_path]
    else:
        files = [p for p in kb_path.rglob("*") if p.is_file()]

    kept = 0
    for p in sorted(files):
        if _should_skip(p):
            continue
        text = read_file(p).strip()
        if not text:
            continue
        kept += 1
        docs.append(Document(
            text=text,
            metadata={
                "source": str(p),
                "category": _category_for(p),
            },
        ))
    print(f"📚 Loaded {len(docs)} documents from {kb_path}")
    return docs
