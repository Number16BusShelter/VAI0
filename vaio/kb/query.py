from __future__ import annotations
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from .store import get_index
from .paths import ensure_default_dirs, DEFAULT_KB_DIR
from .store import build_index, get_index, collection_stats
from .loader import load_documents
from vaio.core.utils import load_meta, save_meta  # reuse existing meta IO

# ────────────────────────────────
# 📂 Path resolution
# ────────────────────────────────
def _resolve_kb_dir_for_video(video_path: Path) -> Path | None:
    """
    Decide which knowledge dir to use.
    - If project meta (<project>.vaio.json) has "knowledge": use that (or None).
    - Else use default <repo_root>/knowledge/default (auto-create).
    """
    ensure_default_dirs()
    meta = load_meta(video_path)  # existing VAIO helper
    kb_value = meta.get("knowledge", "__unset__")
    if kb_value == "__unset__":
        # Not set yet: default to <repo_root>/knowledge/default
        return DEFAULT_KB_DIR
    if kb_value in (None, "", "null"):
        # Explicitly disabled
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
    docs = load_documents(kb)
    if not docs:
        print(f"⚠️ No documents found in {kb}")
        return {"status": "empty", "count": 0, "kb": str(kb)}

    build_index(kb, docs)
    stats = collection_stats(kb)
    return {"status": "built", "count": stats["count"], "kb": str(kb)}


def retrieve(video_path: Path, query: str, top_k: int = 3) -> list[str]:
    """
    Retrieve top-K relevant text chunks from the knowledge base.
    Returns a list of text snippets.
    """
    kb = _resolve_kb_dir_for_video(video_path)
    if kb is None:
        return []
    index: VectorStoreIndex = get_index(kb)
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)
    return [r.text for r in results]


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

def retrieve(video_path: Path, query: str, top_k: int = 3) -> list[str]:
    kb_dir = _resolve_kb_dir_for_video(video_path)
    if kb_dir is None:
        return []
    try:
        index: VectorStoreIndex = get_index(kb_dir)
        retriever = index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)
        return [r.text for r in results]
    except Exception as e:
        print(f"⚠️ Retrieval failed: {e}")
        return []


def inject_context(video_path: Path, user_prompt: str, task: str = "desc") -> str:
    """
    Retrieve relevant knowledge chunks based on the user prompt and task.
    Uses LlamaIndex retriever abstraction for compatibility with all Chroma versions.
    """
    kb_dir = _resolve_kb_dir_for_video(video_path)
    if kb_dir is None:
        return user_prompt

    try:
        index = get_index(kb_dir)
        retriever = index.as_retriever(similarity_top_k=3)
        query = f"{task} context: {user_prompt[:400]}"
        results = retriever.retrieve(query)

        if not results:
            return user_prompt

        context_text = "\n\n".join(r.text for r in results if r.text)
        return f"{user_prompt}\n\n---\nContext (from KB):\n{context_text}\n---"

    except Exception as e:
        print(f"⚠️ Context injection failed: {e}")
        return user_prompt


    
def build_if_needed(video_path: Path):
    """
    Optional helper: on first use, if collection is empty but kb_dir has files,
    build the index automatically.
    """
    kb = _resolve_kb_dir_for_video(video_path)
    if kb is None:
        return
    from .store import collection_stats
    stats = collection_stats(kb)
    if stats["count"] == 0 and any(kb.iterdir()):
        build_kb_for_video(video_path, kb)
