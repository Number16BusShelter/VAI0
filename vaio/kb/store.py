from __future__ import annotations
from pathlib import Path
import chromadb
from chromadb import PersistentClient
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from .paths import DATA_DIR, kb_collection_name

# Embedding model: fast, widely used, multilingual-friendly enough
_EMB_MODEL = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ────────────────────────────────
# 📦 LOCAL CHROMA INITIALIZATION
# ────────────────────────────────
def _get_local_client(path: Path) -> chromadb.Client:
    """
    Always return a *local* Chroma persistent client.
    This avoids HttpClient(None) errors and keeps data in local directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


# ────────────────────────────────
# 🧠 BUILD INDEX
# ────────────────────────────────
def build_index(kb_dir: Path, documents: list[Document]):
    client = _get_local_client(kb_dir / "chroma")  # ensures folder exists
    vs = ChromaVectorStore(
        chroma_collection=client.get_or_create_collection("vaio_kb"),
        client=client,
    )
    storage = StorageContext.from_defaults(vector_store=vs)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage,
        embed_model=_EMB_MODEL,
        show_progress=True,
    )
    print(f"✅ Built index with {len(documents)} documents → {kb_dir}")


# ────────────────────────────────
# 🧭 GET EXISTING INDEX
# ────────────────────────────────
def get_index(kb_dir: Path) -> VectorStoreIndex:
    client = _get_local_client(kb_dir / "chroma")
    vs = ChromaVectorStore(
        chroma_collection=client.get_or_create_collection("vaio_kb"),
        client=client,
    )
    storage = StorageContext.from_defaults(vector_store=vs)
    return VectorStoreIndex.from_vector_store(
        vs,
        storage_context=storage,
        embed_model=_EMB_MODEL,
    )


# ────────────────────────────────
# 🧹 CLEAR / RESET INDEX
# ────────────────────────────────
def clear_index(kb_dir: Path):
    """
    Completely remove the local Chroma collection for this KB.
    This does not delete the documents themselves — only the vector index.
    Safe to re-run build afterwards.
    """
    try:
        client = _get_local_client(kb_dir / "chroma")

        # Chroma 0.4.x / 0.5.x compatibility
        existing_names = [c.name for c in client.list_collections()]
        if "vaio_kb" in existing_names:
            client.delete_collection("vaio_kb")
            print(f"🗑️  Deleted collection 'vaio_kb' in {kb_dir}/chroma")
        else:
            print(f"ℹ️  No existing collection 'vaio_kb' found in {kb_dir}/chroma")

    except Exception as e:
        print(f"⚠️  Failed to clear index for {kb_dir}: {e}")

# ────────────────────────────────
# 📊 STATS
# ────────────────────────────────
def collection_stats(kb_dir: Path) -> dict:
    try:
        client = _get_local_client(kb_dir / "chroma")
        coll = client.get_or_create_collection("vaio_kb")
        return {"count": coll.count()}
    except Exception as e:
        print(f"⚠️ Could not load stats: {e}")
        return {"count": 0}

