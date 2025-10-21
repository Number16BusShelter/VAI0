# store.py (replace the entire file)
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Union, Optional, List
import re
import os
import torch
import chromadb
from chromadb import PersistentClient

from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from .paths import REPO_ROOT, DATA_ROOT, DEFAULT_EMBED_MODEL, ensure_default_dirs

# ────────────────────────────────
# 🧠 EMBEDDING MODEL SETUP
# ────────────────────────────────
_EMBED_MODEL_INITIALIZED = False

def init_embed_model(model_name: str | None = None):
    """Initialize local HuggingFace embedding model"""
    global _EMBED_MODEL_INITIALIZED

    if _EMBED_MODEL_INITIALIZED:
        return

    model = model_name or DEFAULT_EMBED_MODEL

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Ensure local embeddings only
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["LLAMA_INDEX_USE_LOCAL_EMBEDDINGS_ONLY"] = "1"

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=model,
        device=device,
    )

    _EMBED_MODEL_INITIALIZED = True
    print(f"🧠 Using local embedding model: {model} [{device}]")

# ────────────────────────────────
# 📍 UNIFIED PERSISTENCE RESOLUTION
# ────────────────────────────────

def _resolve_kb_identifier(kb_identifier: Union[str, Path, None]) -> str:
    """
    Convert any KB identifier to a consistent string name for ChromaDB.
    
    Rules:
    - None -> "default"
    - Path -> use the stem (filename without extension) or "default" if empty
    - "default" or empty string -> "default"
    - Other strings -> sanitized version
    """
    if kb_identifier is None:
        return "default"
    
    if isinstance(kb_identifier, Path):
        # For Path objects, use the directory name or "default"
        if kb_identifier.name and kb_identifier.name != ".":
            return sanitize_collection_name(kb_identifier.name)
        return "default"
    
    # String handling
    kb_str = str(kb_identifier).strip()
    if not kb_str or kb_str.lower() == "default":
        return "default"
    
    return sanitize_collection_name(kb_str)

def _get_persist_path(kb_identifier: Union[str, Path, None]) -> Path:
    """
    Get the ChromaDB persistence path for a KB identifier.
    All ChromaDB data goes under DATA_ROOT (repo_root/data/kb)
    """
    kb_name = _resolve_kb_identifier(kb_identifier)
    persist_path = DATA_ROOT / kb_name
    persist_path.mkdir(parents=True, exist_ok=True)
    return persist_path

def sanitize_collection_name(name: str) -> str:
    """Ensure collection name fits ChromaDB requirements"""
    base = Path(name).stem
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)
    base = base.strip("._-")
    if len(base) < 3:
        base = f"kb_{base or 'default'}"
    return base[:128]

# ────────────────────────────────
# 🗄️ CHROMA DB OPERATIONS
# ────────────────────────────────

def get_chroma_collection(kb_identifier: Union[str, Path, None]) -> chromadb.Collection:
    """Get or create ChromaDB collection for the given KB identifier"""
    persist_path = _get_persist_path(kb_identifier)
    kb_name = _resolve_kb_identifier(kb_identifier)
    
    client = PersistentClient(path=str(persist_path))
    collection_name = sanitize_collection_name(f"vaio_kb_{kb_name}")
    
    return client.get_or_create_collection(name=collection_name)

# ────────────────────────────────
# 🧠 INDEX OPERATIONS
# ────────────────────────────────

def build_index(kb_identifier: Union[str, Path, None], documents: Iterable[Document]) -> VectorStoreIndex:
    """Build or rebuild index for the given KB identifier"""
    ensure_default_dirs()
    init_embed_model()
    
    collection = get_chroma_collection(kb_identifier)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_ctx,
        show_progress=True,
    )
    
    kb_name = _resolve_kb_identifier(kb_identifier)
    print(f"✅ Built KB index '{kb_name}' (docs={collection.count()})")
    return index

def open_index(kb_identifier: Union[str, Path, None]) -> VectorStoreIndex:
    """Open existing index for the given KB identifier"""
    ensure_default_dirs()
    init_embed_model()
    
    collection = get_chroma_collection(kb_identifier)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_ctx)

def get_index(kb_dir: Path) -> VectorStoreIndex:
    """Legacy compatibility wrapper"""
    return open_index(kb_dir)

# ────────────────────────────────
# 📊 UTILITIES
# ────────────────────────────────

def collection_stats(kb_identifier: Union[str, Path, None]) -> dict:
    """Get statistics for a KB collection"""
    collection = get_chroma_collection(kb_identifier)
    persist_path = _get_persist_path(kb_identifier)
    
    return {
        "collection": collection.name,
        "count": collection.count(),
        "kb_name": _resolve_kb_identifier(kb_identifier),
        "persist_path": str(persist_path),
    }

def clear_index(kb_identifier: Union[str, Path, None]) -> None:
    """Clear index for the given KB identifier"""
    persist_path = _get_persist_path(kb_identifier)
    kb_name = _resolve_kb_identifier(kb_identifier)
    collection_name = sanitize_collection_name(f"vaio_kb_{kb_name}")
    
    client = PersistentClient(path=str(persist_path))
    try:
        client.delete_collection(collection_name)
        print(f"🧹 Deleted collection '{collection_name}'")
    except Exception as e:
        print(f"⚠️ Could not delete collection: {e}")

def debug_list_docs(kb_identifier: Union[str, Path, None], limit: int = 20) -> None:
    """Debug function to list documents in a collection"""
    try:
        collection = get_chroma_collection(kb_identifier)
        count = collection.count()
        print(f"📊 Collection: {collection.name} | Total: {count}")

        data = collection.get(include=["documents", "metadatas"], limit=limit)
        
        for i, doc_text in enumerate(data.get("documents", [])[:limit]):
            meta = data.get("metadatas", [{}])[i]
            print(f"\n🧩 Doc #{i+1}")
            print(f"  Metadata: {meta}")
            print(f"  Text: {doc_text[:300]!r}...")
    except Exception as e:
        print(f"⚠️ Debug list failed: {e}")