from __future__ import annotations
from pathlib import Path
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings
from typing import Optional
from llama_index.core import Settings, VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from .paths import DATA_DIR, kb_collection_name, DEFAULT_CHROMA_DIR, DEFAULT_EMBED_MODEL

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
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .paths import DEFAULT_EMBED_MODEL

def init_embed_model(model_name: str | None = None, device: str | None = None):
    """
    Initialize a local embedding model (HuggingFace) instead of OpenAI.
    """
    from torch import backends

    # pick MPS or CPU automatically for Apple Silicon
    device = device or ("mps" if backends.mps.is_available() else "cpu")
    model = model_name or DEFAULT_EMBED_MODEL

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=model,
        device=device,
    )

    print(f"🧠 Using local embedding model: {model} [{device}]")

def get_chroma_collection(kb_name: str, base_dir: str = DEFAULT_CHROMA_DIR):
    persist_dir = str(Path(base_dir) / kb_name)
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(allow_reset=True)  # optional; helpful in dev
    )
    # One collection per KB name
    return client.get_or_create_collection(name=kb_name, metadata={"kb_name": kb_name})


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
def build_index(kb_name_or_dir, documents):
    from .store import init_embed_model
    init_embed_model()
    """
    Build or update a ChromaDB index for the given KB and documents.

    Accepts either:
      - kb_name (str)
      - kb_dir (Path or list[str])
      - or legacy list input (['knowledge/default'])
    """

    # Normalize input type
    if isinstance(kb_name_or_dir, (list, tuple)):
        kb_name = str(kb_name_or_dir[0])
    elif isinstance(kb_name_or_dir, Path):
        kb_name = kb_name_or_dir.stem
    elif isinstance(kb_name_or_dir, str):
        kb_name = kb_name_or_dir
    else:
        raise TypeError(f"Unsupported type for kb_name_or_dir: {type(kb_name_or_dir)}")

    # Ensure embedding model initialized
    if not hasattr(Settings, "embed_model") or Settings.embed_model is None:
        init_embed_model()

    collection = get_chroma_collection(kb_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    # Build or extend the index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_ctx,
        show_progress=True,
    )

    count = collection.count()
    print(f"✅ Built KB index '{kb_name}' with {count} chunks stored.")
    return index


# ────────────────────────────────
# 🧭 GET EXISTING INDEX
# ────────────────────────────────
def open_index(kb_name: str) -> VectorStoreIndex:
    collection = get_chroma_collection(kb_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    # build an "empty" index object bound to the existing store
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
# 🧹 CLEAR / RESET INDEX
# ────────────────────────────────
def clear_index(kb_dir: Path):
    try:
        client = _get_local_client(kb_dir)
        names = [c.name for c in client.list_collections()]
        if "vaio_kb" in names:
            client.delete_collection("vaio_kb")
            print(f"🗑️  Deleted collection for KB: {kb_dir.name}")
        else:
            print(f"ℹ️  No collection found for {kb_dir.name}.")
    except Exception as e:
        print(f"⚠️  Failed to clear index for {kb_dir.name}: {e}")
# ────────────────────────────────
# 📊 STATS
# ────────────────────────────────
def collection_stats(kb_dir: Path) -> dict:
    print(kb_dir)
    """
    Return basic info about the Chroma collection for a given KB.
    """
    try:
        client = _get_local_client(kb_dir)
        coll = client.get_or_create_collection("vaio_kb")
        print(coll)
        count = coll.count()
        storage_path = (Path(client._settings.persist_path)  # type: ignore
                        if hasattr(client, "_settings") else None)
        return {
            "collection": coll.name,
            "count": count,
            "storage": str(storage_path or "unknown"),
        }
    except Exception as e:
        print(f"⚠️ Could not read stats: {e}")
        return {
            "collection": "vaio_kb",
            "count": 0,
            "storage": "unavailable",
        }
