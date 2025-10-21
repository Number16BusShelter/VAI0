from __future__ import annotations
from pathlib import Path
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from .store import get_index
from .paths import ensure_default_dirs, DEFAULT_KB_DIR
from .store import build_index, get_index, collection_stats
from .loader import load_documents
from vaio.core.utils import load_meta, save_meta  # reuse existing meta IO

# query.py  (replace the build_kb_for_video + helper with the following)

from .loader import load_documents, read_file  # import read_file

IGNORE_NAMES = {
    ".DS_Store", ".gitkeep", ".gitignore", ".env",
    "__pycache__", "node_modules", "tmp", "venv",
}
IGNORE_EXTENSIONS = {
    ".db", ".sqlite", ".lock", ".log", ".bak", ".tmp", ".old",
}

def _iter_valid_files(kb_dir: Path) -> list[Path]:
    files = []
    for f in kb_dir.glob("**/*"):
        if not f.is_file():
            continue
        if f.name.startswith(".") or f.name in IGNORE_NAMES:
            continue
        if f.suffix.lower() in IGNORE_EXTENSIONS:
            continue
        files.append(f)
    return files

def _docs_from_files(files: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for f in files:
        try:
            text = read_file(f).strip()
            if not text:
                continue
            docs.append(Document(
                text=text,
                metadata={"source": str(f), "category": "marketing"},
            ))
        except Exception as e:
            print(f"⚠️ Skipped {f.name}: {e}")
    return docs


# ────────────────────────────────
# 🔍 Internal KB resolution
# ────────────────────────────────
def _resolve_kb_dir_for_video(video_path: Path) -> Path | None:
    from .paths import DEFAULT_KB_DIR
    ensure_default_dirs()
    meta = load_meta(video_path)
    kb_value = meta.get("knowledge", "__unset__")
    if kb_value == "__unset__":
        return DEFAULT_KB_DIR
    if kb_value in (None, "", "null"):
        return None
    return Path(kb_value)


def set_kb_dir_for_video(video_path: Path, kb_dir: Path | None):
    meta = load_meta(video_path)
    meta["knowledge"] = None if kb_dir is None else str(Path(kb_dir).resolve())
    save_meta(video_path, meta)


# ────────────────────────────────
# 🧱 Core KB operations
# ────────────────────────────────
def load_kb_if_available(video_path: Path) -> VectorStoreIndex | None:
    """
    Try to load an existing LlamaIndex/Chroma knowledge base for this project.
    Returns VectorStoreIndex or None if not found.
    """
    kb_dir = _resolve_kb_dir_for_video(video_path)
    if kb_dir is None:
        return None

    try:
        index = get_index(kb_dir)
        return index
    except Exception as e:
        print(f"⚠️ No existing KB index found for {kb_dir}: {e}")
        return None


def iter_knowledge_files(kb_dir: Path):
    """Yield valid knowledge files, skipping system and hidden files."""
    for path in kb_dir.glob("**/*"):
        if not path.is_file():
            continue
        # Skip hidden/system files
        if path.name.startswith(".") or path.suffix.lower() in {".db", ".sqlite", ".lock"}:
            continue
        yield path
        

def build_kb_for_video(video_path: Path, kb_dir: Path | None = None) -> dict:
    """
    Build (or rebuild) the KB for this project video.
    If kb_dir is None: resolve from meta (or default).
    If resolved is None: KB disabled (no-op).
    """
    kb = kb_dir if kb_dir is not None else _resolve_kb_dir_for_video(video_path)
    if kb is None:
        return {"status": "disabled", "count": 0, "kb": None}

    kb = Path(kb)
    kb.mkdir(parents=True, exist_ok=True)

    # 1) filter at the filesystem level
    valid_files = _iter_valid_files(kb)
    if not valid_files:
        print(f"⚠️ No valid documents found in {kb}")
        return {"status": "empty", "count": 0, "kb": str(kb)}

    print("📂 Found {} valid knowledge files:".format(len(valid_files)))
    for vf in valid_files:
        try:
            print(f"  - {vf.relative_to(kb)}")
        except Exception:
            print(f"  - {vf}")

    # 2) load exactly these files (NO parent directory scanning)
    docs = _docs_from_files(valid_files)
    if not docs:
        print(f"⚠️ No readable documents after filtering in {kb}")
        return {"status": "empty", "count": 0, "kb": str(kb)}

    # 3) build index
    build_index(kb, docs)
    stats = collection_stats(kb)
    return {"status": "built", "count": stats["count"], "kb": str(kb)}



def load_documents_from_list(files: list[Path]):
    """
    Legacy shim (kept for compatibility) — now loads unique parent directories.
    """
    parent_dirs = sorted({f.parent for f in files})
    docs = []
    for d in parent_dirs:
        try:
            docs.extend(load_documents(d))
        except Exception as e:
            print(f"⚠️ Skipped directory {d}: {e}")
    return docs

# ────────────────────────────────
def retrieve(video_path: Path, query: str, top_k: int = 3) -> list[dict]:
    kb_dir = _resolve_kb_dir_for_video(video_path)
    if kb_dir is None:
        return []

    try:
        index: VectorStoreIndex = get_index(kb_dir)
        retriever = index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)
        # keep structured output for metadata
        return [
            {
                "text": r.text,
                "source": r.metadata.get("source", "unknown"),
                "score": getattr(r, "score", None),
            }
            for r in results
        ]
    except Exception as e:
        print(f"⚠️ Retrieval failed: {e}")
        return []


# ────────────────────────────────
# 🧩 Context injection
# ────────────────────────────────
# vaio/kb/__init__.py

def _filters_for_task(task: str) -> MetadataFilters | None:
    if task in ("title", "desc"):
        return MetadataFilters(filters=[ExactMatchFilter(key="category", value="marketing")])
    if task in ("translate",):
        # Let translations see reference material too
        # return None  # or filter to multiple categories if you want
        return MetadataFilters(filters=[])  # no restriction
    return None

def retrieve(video_path: Path, query: str, top_k: int = 3) -> list[dict]:
    kb_dir = _resolve_kb_dir_for_video(video_path)
    if kb_dir is None:
        return []

    try:
        index: VectorStoreIndex = get_index(kb_dir)
        retriever = index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)

        return [
            {
                "text": r.text,
                "source": r.metadata.get("source", "unknown"),
                "score": getattr(r, "score", None),
            }
            for r in results
        ]

    except Exception as e:
        print(f"⚠️ Retrieval failed: {e}")
        return []


def inject_context(video_path: Path, user_prompt: str, task: str = "desc") -> str:
    """
    Retrieves relevant KB snippets and appends them to the LLM prompt.
    Applies confidence threshold + source tracing for debugging clarity.
    """
    kb_dir = _resolve_kb_dir_for_video(video_path)
    if kb_dir is None:
        return user_prompt

    try:
        query = f"{task} context: {user_prompt[:400]}"
        results = retrieve(video_path, query, top_k=5)
        if not results:
            return user_prompt

        # Confidence threshold filter
        filtered = [
            r for r in results if r["score"] is None or r["score"] > 0.35
        ]
        if not filtered:
            return user_prompt

        # Annotate each source
        context_blocks = [
            f"[{r['source']}] {r['text']}" for r in filtered if r.get("text")
        ]
        context_text = "\n\n".join(context_blocks)

        print(f"📚 Injected {len(filtered)} KB snippets into prompt.")
        return f"{user_prompt}\n\n---\nContext (from KB):\n{context_text}\n---"

    except Exception as e:
        print(f"⚠️ Context injection failed: {e}")
        return user_prompt


def build_if_needed(video_path: Path):
    """
    Automatically build the KB if it exists but has no indexed documents.
    """
    from .store import collection_stats
    from .store import build_index
    from .loader import load_documents

    kb_dir = _resolve_kb_dir_for_video(video_path)
    if kb_dir is None:
        return

    stats = collection_stats(kb_dir)
    if stats["count"] == 0 and any(kb_dir.iterdir()):
        docs = load_documents(kb_dir)
        build_index(kb_dir, docs)