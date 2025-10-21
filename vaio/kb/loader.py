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


# ────────────────────────────────
# 📂 Category detection
# ────────────────────────────────
def _category_for(path: Path) -> str:
    p = str(path).lower()
    if "/reference/" in p or "gia" in p or "guide" in p:
        return "reference"
    if "/marketing/" in p:
        return "marketing"
    return "marketing"


# ────────────────────────────────
# 🧾 File Readers
# ────────────────────────────────
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    if not PdfReader:
        print(f"⚠️ PDF support not available (PyPDF2 missing): {path}")
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"⚠️ Could not read PDF {path}: {e}")
        return ""


def _read_json(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        # flatten JSON
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


def _read_any_supported(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return _read_text(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".json":
        return _read_json(path)
    if suffix == ".csv":
        return _read_csv(path)
    if suffix in (".yml", ".yaml"):
        return _read_yaml(path)
    # fallback
    return _read_text(path)


# ────────────────────────────────
# 📘 Document loader
# ────────────────────────────────
def load_documents(kb_dir: Path) -> List[Document]:
    """
    Recursively load supported documents from a knowledge directory.
    Automatically assigns metadata: source, category.
    """
    docs: List[Document] = []
    if not kb_dir.exists():
        return docs

    for p in sorted(kb_dir.rglob("*")):
        if not p.is_file():
            continue
        text = _read_any_supported(p)
        if not text.strip():
            continue

        docs.append(Document(
            text=text,
            metadata={
                "source": str(p),
                "category": _category_for(p),
            },
        ))
    print(f"📚 Loaded {len(docs)} documents from {kb_dir}")
    return docs
