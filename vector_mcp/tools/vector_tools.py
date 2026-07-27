"""
Vector search tools for the Vector MCP server.

All tools use a shared ChromaDB collection backed by HuggingFace
sentence-transformer embeddings.  The collection is seeded at startup
from public APIs (arXiv, Wikipedia) and can be extended at runtime.

Tools:
  - semantic_search          — find documents by semantic similarity
  - get_vector_db_stats      — collection statistics
  - add_text_to_vector_db    — add a custom text snippet
  - populate_from_arxiv      — fetch + embed arXiv papers for a topic
  - populate_from_wikipedia  — fetch + embed Wikipedia article summaries
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote, urlencode

import feedparser
import httpx

if TYPE_CHECKING:
    import chromadb

logger = logging.getLogger(__name__)

# Lazy globals (initialised by init_vector_db() called from start.py)
_collection: "chromadb.Collection | None" = None
_embedding_function = None

ARXIV_API = os.environ.get("ARXIV_API_URL", "https://export.arxiv.org/api/query")
WIKIPEDIA_API = os.environ.get("WIKIPEDIA_API_URL", "https://en.wikipedia.org/api/rest_v1")
_HEADERS = {"User-Agent": "ai-platform-demo/1.0 (vector-mcp)"}


# ---------------------------------------------------------------------------
# Initialisation (called once at server startup)
# ---------------------------------------------------------------------------

def init_vector_db(db_path: str, embed_model: str) -> None:
    """Initialise ChromaDB collection with HuggingFace sentence-transformer embeddings.

    Args:
        db_path:     Absolute path for persistent ChromaDB storage.
        embed_model: HuggingFace model name, e.g. 'sentence-transformers/all-MiniLM-L6-v2'.
    """
    global _collection, _embedding_function

    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    logger.info("Loading embedding model: %s", embed_model)
    # Strip the 'sentence-transformers/' prefix if present (chromadb expects bare names)
    model_name = embed_model.replace("sentence-transformers/", "")
    _embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)

    logger.info("Opening ChromaDB at: %s", db_path)
    Path(db_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    _collection = client.get_or_create_collection(
        name="research_documents",
        embedding_function=_embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("ChromaDB collection 'research_documents' ready. Count: %d", _collection.count())


def _get_collection():
    if _collection is None:
        raise RuntimeError("Vector DB not initialised. Call init_vector_db() first.")
    return _collection


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_doc_id(source: str, identifier: str) -> str:
    """Create a stable document ID from source and identifier."""
    import hashlib
    raw = f"{source}:{identifier}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


async def _arxiv_papers(query: str, max_results: int = 10) -> list[dict]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": min(max_results, 25),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urlencode(params)}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, headers=_HEADERS)
        resp.raise_for_status()
    feed = feedparser.parse(resp.text)
    papers = []
    for entry in feed.entries:
        papers.append({
            "id": entry.get("id", "").split("/abs/")[-1],
            "title": entry.get("title", "").replace("\n", " ").strip(),
            "summary": entry.get("summary", "")[:600].replace("\n", " ").strip(),
            "published": entry.get("published", "")[:10],
            "authors": ", ".join(a["name"] for a in entry.get("authors", [])[:3]),
            "url": entry.get("link", ""),
            "categories": " ".join(t["term"] for t in entry.get("tags", [])[:3]),
        })
    return papers


async def _wikipedia_summary(title: str) -> dict | None:
    encoded = quote(title.replace(" ", "_"))
    url = f"{WIKIPEDIA_API}/page/summary/{encoded}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=_HEADERS)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
        data = resp.json()
        return {
            "title": data.get("title", title),
            "extract": data.get("extract", "")[:800],
            "description": data.get("description", ""),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

async def semantic_search(query: str, n_results: int = 5, source_filter: str = "") -> str:
    """Search the vector database using semantic (embedding) similarity.

    Unlike keyword search, semantic search understands meaning — e.g. searching
    "deep learning" will also surface documents about 'neural networks' and
    'backpropagation'.  Documents are ranked by cosine similarity to the query.

    Args:
        query:         Natural-language search query.
        n_results:     Number of results to return (default 5, max 20).
        source_filter: Optional — filter by source: 'arxiv', 'wikipedia', or '' for all.

    Returns:
        JSON with semantically similar documents ranked by relevance score.
    """
    try:
        col = _get_collection()
        n = min(n_results, 20)
        where = {"source": source_filter} if source_filter else None
        kwargs: dict = {"query_texts": [query], "n_results": min(n, col.count() or 1)}
        if where:
            kwargs["where"] = where
        results = col.query(**kwargs)

        docs = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]

        for doc_id, dist, meta, doc_text in zip(ids, distances, metas, documents):
            docs.append({
                "id": doc_id,
                "relevance_score": round(1.0 - float(dist), 4),  # cosine → similarity
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "topic": meta.get("topic", ""),
                "url": meta.get("url", ""),
                "excerpt": doc_text[:300] if doc_text else "",
                "published": meta.get("published", ""),
            })

        return json.dumps({
            "query": query,
            "count": len(docs),
            "source_filter": source_filter or "all",
            "results": docs,
        })
    except Exception as exc:
        logger.exception("Error in semantic_search")
        return json.dumps({"error": str(exc), "status": "failed"})


async def get_vector_db_stats() -> str:
    """Return statistics about the vector database collection.

    Shows the total number of indexed documents, breakdown by source
    (arXiv vs. Wikipedia vs. custom), and the embedding model in use.

    Returns:
        JSON with collection statistics.
    """
    try:
        col = _get_collection()
        total = col.count()
        # Sample a few to get source breakdown
        if total > 0:
            sample = col.get(limit=min(total, 500), include=["metadatas"])
            sources: dict[str, int] = {}
            topics: dict[str, int] = {}
            for meta in sample.get("metadatas", []):
                src = meta.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1
                topic = meta.get("topic", "")
                if topic:
                    topics[topic] = topics.get(topic, 0) + 1
            top_topics = sorted(topics.items(), key=lambda x: -x[1])[:10]
        else:
            sources = {}
            top_topics = []

        return json.dumps({
            "total_documents": total,
            "sources": sources,
            "top_topics": [{"topic": t, "count": c} for t, c in top_topics],
            "embedding_model": os.environ.get("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            "db_path": os.environ.get("VECTOR_DB_PATH", "data/vector_db"),
            "status": "ready",
        })
    except Exception as exc:
        logger.exception("Error in get_vector_db_stats")
        return json.dumps({"error": str(exc), "status": "failed"})


async def add_text_to_vector_db(
    text: str,
    title: str,
    topic: str = "",
    source: str = "custom",
    url: str = "",
) -> str:
    """Add a custom text snippet to the vector database.

    Embeds the text with the HuggingFace model and stores it in ChromaDB.
    Use this to persist research notes, findings, or any text that should
    be retrievable via semantic search.

    Args:
        text:   The text content to embed and store (up to ~2000 chars).
        title:  Short title or heading for this document.
        topic:  Optional topic tag (e.g. 'machine learning').
        source: Source label (default 'custom').
        url:    Optional source URL.

    Returns:
        JSON confirming the document was added.
    """
    try:
        col = _get_collection()
        doc_id = _make_doc_id(source, title + text[:50])
        col.upsert(
            ids=[doc_id],
            documents=[text[:2000]],
            metadatas=[{
                "title": title[:200],
                "source": source,
                "topic": topic,
                "url": url,
                "published": "",
            }],
        )
        return json.dumps({
            "id": doc_id,
            "title": title,
            "status": "added",
            "total_docs": col.count(),
        })
    except Exception as exc:
        logger.exception("Error in add_text_to_vector_db")
        return json.dumps({"error": str(exc), "status": "failed"})


async def populate_from_arxiv(topic: str, max_papers: int = 15) -> str:
    """Fetch arXiv papers for a topic and add them to the vector database.

    Downloads recent papers from arXiv, embeds their titles + abstracts
    using the HuggingFace model, and indexes them in ChromaDB for semantic
    search.  This is the primary way to populate the vector DB with
    up-to-date scientific literature.

    Args:
        topic:      Research topic to search on arXiv (e.g. 'transformer architecture').
        max_papers: Number of papers to fetch and index (default 15, max 30).

    Returns:
        JSON with the number of papers added and sample titles.
    """
    try:
        col = _get_collection()
        papers = await _arxiv_papers(topic, max_results=min(max_papers, 30))
        if not papers:
            return json.dumps({"topic": topic, "added": 0, "message": "No papers found on arXiv."})

        ids, documents, metadatas = [], [], []
        for p in papers:
            doc_text = f"{p['title']}. {p['summary']}"
            doc_id = _make_doc_id("arxiv", p["id"])
            ids.append(doc_id)
            documents.append(doc_text)
            metadatas.append({
                "title": p["title"],
                "source": "arxiv",
                "topic": topic,
                "url": p["url"],
                "published": p["published"],
                "authors": p["authors"],
                "categories": p["categories"],
            })

        col.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Populated %d arXiv papers for topic '%s'", len(papers), topic)

        return json.dumps({
            "topic": topic,
            "added": len(papers),
            "total_docs": col.count(),
            "sample_titles": [p["title"] for p in papers[:3]],
            "source": "arXiv",
        })
    except Exception as exc:
        logger.exception("Error in populate_from_arxiv")
        return json.dumps({"error": str(exc), "status": "failed"})


async def populate_from_wikipedia(topics: str) -> str:
    """Fetch Wikipedia article summaries and add them to the vector database.

    Downloads summary sections for the given topics from the Wikipedia REST API,
    embeds them, and indexes them in ChromaDB.  Complements the arXiv collection
    with encyclopaedic background knowledge.

    Args:
        topics: Comma-separated list of Wikipedia article titles / topics,
                e.g. 'Transformer (deep learning), BERT (language model), GPT-3'.

    Returns:
        JSON with the number of articles added and any titles not found.
    """
    try:
        col = _get_collection()
        topic_list = [t.strip() for t in topics.split(",") if t.strip()]
        added, not_found = [], []

        for title in topic_list:
            article = await _wikipedia_summary(title)
            if not article:
                not_found.append(title)
                continue
            doc_text = f"{article['title']}: {article['extract']}"
            doc_id = _make_doc_id("wikipedia", article["title"])
            col.upsert(
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[{
                    "title": article["title"],
                    "source": "wikipedia",
                    "topic": title,
                    "url": article["url"],
                    "published": "",
                    "description": article.get("description", ""),
                }],
            )
            added.append(article["title"])

        logger.info("Populated %d Wikipedia articles", len(added))
        return json.dumps({
            "added": len(added),
            "added_titles": added,
            "not_found": not_found,
            "total_docs": col.count(),
            "source": "Wikipedia",
        })
    except Exception as exc:
        logger.exception("Error in populate_from_wikipedia")
        return json.dumps({"error": str(exc), "status": "failed"})
