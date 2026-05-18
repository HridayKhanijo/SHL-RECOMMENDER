"""
app/core/index.py — Embedding + FAISS index management.

WHY: Pre-computing embeddings at startup means each /chat call does
only a fast ANN (approximate nearest-neighbor) lookup rather than
re-embedding the whole catalog every time.

DESIGN DECISION:
- sentence-transformers/all-MiniLM-L6-v2 is tiny (80 MB), fast (CPU ok),
  and has strong recall on short HR queries. Switch to a larger model
  if recall on holdout traces is insufficient.
- FAISS IndexFlatIP (inner product on L2-normalized vectors) = cosine
  similarity without extra deps.
- We persist the index to disk so container restarts are fast.

TRADEOFF: FAISS is in-memory. For a catalog with <1 000 entries this is
fine. At 100 k+ entries, switch to ChromaDB or pgvector.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
CATALOG_PATH = DATA_DIR / "shl_catalog.json"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
META_PATH = DATA_DIR / "catalog_meta.pkl"

# Lazy globals — populated once at startup
_index = None
_metadata: list[dict] = []
_embedder = None


def _load_embedder():
    """Load sentence-transformer model. Done once at startup."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info(f"Loading embedding model: {model_name}")
        _embedder = SentenceTransformer(model_name)
    return _embedder


def _make_document_text(item: dict) -> str:
    """
    Construct the text we embed for each assessment.

    DESIGN DECISION: We combine name + type label + description into one
    rich string. This lets semantic search match "personality test for
    managers" → OPQ even when the query uses no exact keywords.

    Order matters: name first because it's highest signal.
    """
    parts = [
        item.get("name", ""),
        item.get("test_type_label", ""),
        item.get("description", ""),
        " ".join(item.get("all_types", [])),
    ]
    return " | ".join(p for p in parts if p).strip()


def build_or_load_index():
    """
    Entry point called at FastAPI lifespan startup.
    Loads from disk if fresh, otherwise builds from catalog JSON.
    """
    global _index, _metadata

    if INDEX_PATH.exists() and META_PATH.exists():
        logger.info("Loading cached FAISS index from disk …")
        _index, _metadata = _load_from_disk()
        logger.info(f"Loaded {len(_metadata)} items from cache.")
        return

    logger.info("No cached index found. Building from catalog …")
    _index, _metadata = _build_index()


def _load_from_disk():
    import faiss
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def _build_index():
    import faiss

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalog not found at {CATALOG_PATH}. "
            "Run: python scripts/scrape_catalog.py"
        )

    with open(CATALOG_PATH) as f:
        catalog = json.load(f)

    if not catalog:
        raise ValueError("Catalog is empty.")

    logger.info(f"Embedding {len(catalog)} assessments …")
    embedder = _load_embedder()

    texts = [_make_document_text(item) for item in catalog]
    # batch=True for speed; normalize=True → cosine via IndexFlatIP
    embeddings = embedder.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized = cosine
    index.add(embeddings)

    # Persist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(catalog, f)

    logger.info(f"Index built: {index.ntotal} vectors, dim={dim}")
    return index, catalog


def semantic_search(query: str, top_k: int = 15) -> list[dict]:
    """
    Encode the query and return top_k catalog items by cosine similarity.

    DESIGN DECISION: We return 15 candidates and let the hybrid ranker
    re-rank to the final 1–10. Over-fetching improves recall when the
    LLM query is slightly off.
    """
    global _index, _metadata

    if _index is None or not _metadata:
        raise RuntimeError("Index not loaded. Call build_or_load_index() first.")

    embedder = _load_embedder()
    q_vec = embedder.encode(
        [query], normalize_embeddings=True
    ).astype("float32")

    scores, indices = _index.search(q_vec, min(top_k, _index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = dict(_metadata[idx])  # copy so caller can't mutate cache
        item["_score_semantic"] = float(score)
        results.append(item)

    return results


def get_by_name(name: str) -> Optional[dict]:
    """Exact (case-insensitive) name lookup — used for comparison queries."""
    name_lower = name.lower().strip()
    for item in _metadata:
        if item["name"].lower().strip() == name_lower:
            return item
    # Fuzzy fallback: partial match
    for item in _metadata:
        if name_lower in item["name"].lower():
            return item
    return None


def get_all_metadata() -> list[dict]:
    return list(_metadata)
