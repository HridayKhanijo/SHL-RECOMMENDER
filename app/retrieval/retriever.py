"""
app/retrieval/retriever.py — Hybrid retrieval layer.

WHY: Pure semantic search misses exact keyword matches (e.g., "Java 8").
Pure BM25 misses synonyms (e.g., "personality" ≈ "behaviour"). Hybrid
= best of both worlds.

DESIGN DECISION:
- Semantic: FAISS cosine similarity (see core/index.py)
- Keyword: BM25 via rank_bm25 (lightweight, no server needed)
- Fusion: Reciprocal Rank Fusion (RRF) — parameter-free, robust,
  widely used in production IR systems (Cormack et al. 2009).
- Final re-ranking: type-filter boost if the user specifies a test type.

TRADEOFF: BM25 is built in-memory at startup from the same catalog.
For <1 000 items this is trivial. At scale, use Elasticsearch BM25.
"""

import logging
import re
from typing import Optional

from app.core.index import semantic_search, get_all_metadata

logger = logging.getLogger(__name__)

# RRF constant — standard value; 60 recommended by literature
RRF_K = 60

# Lazy BM25 index
_bm25 = None
_bm25_corpus: list[dict] = []


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


def _get_bm25():
    """Build BM25 index on first call (lazy)."""
    global _bm25, _bm25_corpus
    if _bm25 is not None:
        return _bm25, _bm25_corpus

    from rank_bm25 import BM25Okapi

    catalog = get_all_metadata()
    if not catalog:
        raise RuntimeError("Catalog not loaded — call build_or_load_index() first.")

    _bm25_corpus = catalog
    tokenized = [
        _tokenize(
            f"{item['name']} {item.get('test_type_label','')} {item.get('description','')}"
        )
        for item in catalog
    ]
    _bm25 = BM25Okapi(tokenized)
    logger.info(f"BM25 index built over {len(catalog)} documents.")
    return _bm25, _bm25_corpus


def keyword_search(query: str, top_k: int = 15) -> list[dict]:
    """BM25 keyword retrieval."""
    bm25, corpus = _get_bm25()
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)

    ranked = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:top_k]

    results = []
    for idx, score in ranked:
        item = dict(corpus[idx])
        item["_score_bm25"] = float(score)
        results.append(item)
    return results


def rrf_fuse(
    semantic_hits: list[dict],
    keyword_hits: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """
    Reciprocal Rank Fusion of two ranked lists.

    score(d) = Σ 1 / (k + rank(d))

    Items appear in at most one of the two lists when urls differ,
    but we merge on url to avoid duplicates.
    """
    scores: dict[str, float] = {}
    items_by_url: dict[str, dict] = {}

    for rank, item in enumerate(semantic_hits, start=1):
        url = item["url"]
        scores[url] = scores.get(url, 0) + 1 / (k + rank)
        items_by_url[url] = item

    for rank, item in enumerate(keyword_hits, start=1):
        url = item["url"]
        scores[url] = scores.get(url, 0) + 1 / (k + rank)
        if url not in items_by_url:
            items_by_url[url] = item

    fused = sorted(items_by_url.values(), key=lambda x: scores[x["url"]], reverse=True)
    for item in fused:
        item["_score_rrf"] = scores[item["url"]]
    return fused


def apply_type_filter_boost(
    items: list[dict],
    required_types: list[str],
    boost: float = 0.2,
) -> list[dict]:
    """
    Boost assessments whose type matches user-requested types.

    DESIGN DECISION: We boost rather than hard-filter so the system
    degrades gracefully when no exact type exists.
    """
    if not required_types:
        return items

    req = {t.upper() for t in required_types}
    for item in items:
        all_types = {t.upper() for t in item.get("all_types", [item.get("test_type", "")])}
        if req & all_types:  # intersection
            item["_score_rrf"] = item.get("_score_rrf", 0) + boost

    return sorted(items, key=lambda x: x.get("_score_rrf", 0), reverse=True)


def retrieve(
    query: str,
    top_k: int = 10,
    required_types: Optional[list[str]] = None,
) -> list[dict]:
    """
    Main retrieval interface. Returns top_k assessments.

    Pipeline:
      1. Semantic search (FAISS cosine)
      2. Keyword search (BM25)
      3. RRF fusion
      4. Type-filter boost (optional)
      5. Truncate to top_k
    """
    sem = semantic_search(query, top_k=20)
    kw = keyword_search(query, top_k=20)
    fused = rrf_fuse(sem, kw)

    if required_types:
        fused = apply_type_filter_boost(fused, required_types)

    return fused[:top_k]
