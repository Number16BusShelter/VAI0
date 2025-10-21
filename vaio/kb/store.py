from __future__ import annotations
from pathlib import Path
from typing import Iterable, Union, Optional, List

import chromadb
from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore

from .paths import BASE_KB_ROOT, DEFAULT_KB_DIR, ensure_default_dirs
import os
import torch
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings
from typing import Optional
from llama_index.core import Settings, VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from .paths import DEFAULT_EMBED_MODEL
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Use your existing local model
Settings.embed_model = HuggingFaceEmbedding(
    model_name=DEFAULT_EMBED_MODEL
)
Settings.llm = None  # Disable any default LLM calls

_EMBED_MODEL_INITIALIZED = False

# ────────────────────────────────
# 📍 Directory setup
# ────────────────────────────────
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "kb"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

_EMB_MODEL = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR = Path("data/kb_cache")

# ────────────────────────────────
# 📦 LOCAL CHROMA INITIALIZATION
# ────────────────────────────────

def init_embed_model(model_name: str | None = None):
    """
    Always initialize a local HuggingFace embedding model.
    Prevents LlamaIndex from defaulting to OpenAI.
    """
    global _EMBED_MODEL_INITIALIZED

    # 🧱 Avoid triggering LlamaIndex's default OpenAI loader
    if _EMBED_MODEL_INITIALIZED or "embed_model" in Settings.__dict__:
        return

    model = model_name or DEFAULT_EMBED_MODEL

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Explicitly clear OpenAI usage
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["LLAMA_INDEX_USE_LOCAL_EMBEDDINGS_ONLY"] = "1"

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=model,
        device=device,
    )

    _EMBED_MODEL_INITIALIZED = True
    print(f"🧠 Using local embedding model: {model} [{device}]")


import re

def sanitize_collection_name(name: str) -> str:
    """
    Ensure collection name fits ChromaDB requirements:
    3–512 chars, only [a-zA-Z0-9._-], must start/end with alnum.
    """
    base = Path(name).stem  # drop path parts if any
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)
    base = base.strip("._-")
    if len(base) < 3:
        base = f"kb_{base or 'default'}"
    return base[:128]  # keep under limit


# -----------------------------------------------------------------------------
# Persist path resolver
# -----------------------------------------------------------------------------
def _persist_path(kb: Union[str, Path, None]) -> Path:
    """
    Resolve the on-disk directory where Chroma should persist.
    Rules:
      - None -> DEFAULT_KB_DIR  (<repo_root>/kb/default)
      - Path -> use AS-IS (absolute or relative to CWD)
      - "default" -> <repo_root>/kb/default
      - any other string (e.g., "clientA") -> <repo_root>/kb/<string>
      - strings with slashes (e.g., "kb/default") are treated as explicit paths.
    """
    ensure_default_dirs()

    if kb is None:
        return DEFAULT_KB_DIR

    if isinstance(kb, Path):
        return kb

    # string
    s = kb.strip()
    p = Path(s)
    if p.is_absolute() or len(p.parts) > 1:
        return p  # treat as explicit path like "kb/default"
    if s == "default":
        return DEFAULT_KB_DIR
    return BASE_KB_ROOT / s  # e.g., "clientA" -> <repo_root>/kb/clientA



def get_chroma_collection(persist: Union[str, Path, None], collection_name: str = "vaio_kb"):
    """
    Open (or create) a Chroma collection at the given persist directory.
    """
    persist_dir = _persist_path(persist)
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(name=collection_name)


def _get_local_client(kb_dir: Path) -> chromadb.Client:
    """
    Store Chroma data in global data/kb/<kb_name> directory,
    separate from the original knowledge source files.
    """
    kb_name = kb_dir.name or "default"
    chroma_path = DATA_ROOT / kb_name
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_path))


def debug_list_docs(kb_dir: Path, limit: int = 20):
    """
    Print the current stored documents in the Chroma collection.
    Useful for verifying duplicates or inspecting chunks.
    """
    try:
        client = _get_local_client(kb_dir)
        coll = client.get_or_create_collection("vaio_kb")

        count = coll.count()
        print(f"📊 Collection: {coll.name} | Total: {count}")

        # Retrieve raw entries from Chroma
        data = coll.get(include=["documents", "metadatas", "embeddings"])

        # Some clients paginate, so limit manually
        for i, doc_text in enumerate(data.get("documents", [])[:limit]):
            meta = data.get("metadatas", [{}])[i]
            print(f"\n🧩 Doc #{i+1}")
            print(f"  Metadata: {meta}")
            print(f"  Text: {doc_text[:300]!r}...")
    except Exception as e:
        print(f"⚠️ Debug list failed: {e}")



# ────────────────────────────────
# 🧠 BUILD INDEX
# ────────────────────────────────
def build_index(kb_name_or_dir: Union[str, Path, None], documents: Iterable[Document]) -> VectorStoreIndex:
    """
    Build or extend an index at the resolved persist path.
    Accepts: "default" | "clientName" | Path("/.../kb/default") | None
    """
    ensure_default_dirs()
    persist_dir = _persist_path(kb_name_or_dir)

    collection = get_chroma_collection(persist_dir)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_ctx,
        show_progress=True,
    )
    print(f"✅ Built KB index at {persist_dir} (collection={collection.name}, count={collection.count()})")
    return index


# ────────────────────────────────
# 🧭 GET EXISTING INDEX
# ────────────────────────────────
def open_index(kb_name_or_dir: Union[str, Path, None]) -> VectorStoreIndex:
    """
    Open an existing index at the resolved persist path.
    """
    persist_dir = _persist_path(kb_name_or_dir)
    collection = get_chroma_collection(persist_dir)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_ctx)


def get_index(kb_dir: Path) -> VectorStoreIndex:
    client = _get_local_client(kb_dir)
    collection = client.get_or_create_collection("vaio_kb")
    vs = ChromaVectorStore(
        chroma_collection=collection,
        client=client,
    )
    storage = StorageContext.from_defaults(vector_store=vs)
    return VectorStoreIndex.from_vector_store(
        vs,
        storage_context=storage,
        embed_model=_EMB_MODEL,
    )



# ────────────────────────────────
# 📊 STATS
# ────────────────────────────────

def collection_stats(kb_name_or_dir: Union[str, Path, None]) -> dict:
    persist_dir = _persist_path(kb_name_or_dir)
    collection = get_chroma_collection(persist_dir)
    return {
        "collection": collection.name,
        "count": collection.count(),
        "dir": str(persist_dir),
    }

def clear_index(kb_name_or_dir: Union[str, Path, None]) -> None:
    persist_dir = _persist_path(kb_name_or_dir)
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection("vaio_kb")
        print(f"🧹 Deleted collection 'vaio_kb' at {persist_dir}")
    except Exception:
        pass


def debug_list_docs(kb_name_or_dir: Union[str, Path, None], limit: int = 20) -> None:
    persist_dir = _persist_path(kb_name_or_dir)
    collection = get_chroma_collection(persist_dir)
    ids = collection.get(include=["documents", "metadatas"]).get("ids", [])
    print(f"📚 Collection={collection.name} docs={len(ids)} dir={persist_dir}")
