from __future__ import annotations
from pathlib import Path
import chromadb
from chromadb import PersistentClient
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from .paths import DATA_DIR, kb_collection_name

# ────────────────────────────────
# 📍 Directory setup
# ────────────────────────────────
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "kb"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

_EMB_MODEL = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ────────────────────────────────
# 📦 LOCAL CHROMA INITIALIZATION
# ────────────────────────────────
def _get_local_client(kb_dir: Path) -> chromadb.Client:
    """
    Store Chroma data in global data/kb/<kb_name> directory,
    separate from the original knowledge source files.
    """
    kb_name = kb_dir.name or "default"
    chroma_path = DATA_ROOT / kb_name
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_path))



# ────────────────────────────────
# 🧠 BUILD INDEX
# ────────────────────────────────
def build_index(kb_dir: Path, documents: list[Document]):
    client = _get_local_client(kb_dir)
    collection = client.get_or_create_collection("vaio_kb")

    vs = ChromaVectorStore(
        chroma_collection=collection,
        client=client,
    )

    storage = StorageContext.from_defaults(vector_store=vs)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage,
        embed_model=_EMB_MODEL,
        show_progress=True,
    )

    print(f"✅ Built index with {len(documents)} docs → {DATA_ROOT / kb_dir.name}")

# ────────────────────────────────
# 🧭 GET EXISTING INDEX
# ────────────────────────────────
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
    try:
        client = _get_local_client(kb_dir)
        coll = client.get_or_create_collection("vaio_kb")
        return {"count": coll.count()}
    except Exception as e:
        print(f"⚠️ Could not read stats: {e}")
        return {"count": 0}

