"""
Analytics tools — powered by public APIs (no database required).

Replaces the original SQLite/PostgreSQL-backed tools with live data from:
  - arXiv API  (https://export.arxiv.org/api/query)  — academic papers (no key)
  - OpenAlex   (https://api.openalex.org)             — open scholarly graph (no key)

These implement the same MCP interface as before so agents need no changes.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode, quote_plus

import feedparser
import httpx

logger = logging.getLogger(__name__)

ARXIV_API = os.environ.get("ARXIV_API_URL", "https://export.arxiv.org/api/query")
OPENALEX_API = os.environ.get("OPENALEX_API_URL", "https://api.openalex.org")
OPENALEX_EMAIL = os.environ.get("OPENALEX_EMAIL", "demo@example.com")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _arxiv_category_for(topic: str) -> str:
    """Map a plain-English topic to an arXiv category hint."""
    topic_lower = topic.lower()
    if any(k in topic_lower for k in ("machine learning", "ml", "deep learning", "neural")):
        return "cs.LG"
    if any(k in topic_lower for k in ("nlp", "natural language", "llm", "language model")):
        return "cs.CL"
    if any(k in topic_lower for k in ("vision", "image", "cv", "computer vision")):
        return "cs.CV"
    if any(k in topic_lower for k in ("ai", "artificial intelligence", "agent")):
        return "cs.AI"
    if any(k in topic_lower for k in ("robotics", "robot")):
        return "cs.RO"
    return "cs.AI"


async def _arxiv_search(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    raw_query: bool = False,
) -> list[dict]:
    """Fetch papers from arXiv API and return parsed list.

    Args:
        raw_query: If True, use the query string as-is (for category filters
                   like ``cat:cs.AI``).  If False, prepend ``all:`` for
                   full-text search.
    """
    import asyncio

    search_query = query if raw_query else f"all:{query}"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": min(max_results, 50),
        "sortBy": sort_by,          # relevance | lastUpdatedDate | submittedDate
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urlencode(params)}"
    # Retry with back-off on 429 (arXiv rate-limits burst requests)
    for attempt in range(3):
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers={"User-Agent": "ai-platform-demo/1.0"})
        if resp.status_code == 429:
            wait = 10 * (attempt + 1)
            logger.warning("arXiv rate-limited (429). Retrying in %ds (attempt %d/3).", wait, attempt + 1)
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        raise RuntimeError("arXiv API returned 429 after 3 retries. Please try again later.")
    feed = feedparser.parse(resp.text)
    papers = []
    for entry in feed.entries:
        papers.append({
            "id": entry.get("id", ""),
            "title": entry.get("title", "").replace("\n", " ").strip(),
            "authors": [a["name"] for a in entry.get("authors", [])[:5]],
            "summary": entry.get("summary", "")[:400].replace("\n", " ").strip(),
            "published": entry.get("published", "")[:10],
            "updated": entry.get("updated", "")[:10],
            "categories": [t["term"] for t in entry.get("tags", [])],
        })
    return papers


async def _openalex_search(query: str, max_results: int = 10) -> list[dict]:
    """Search OpenAlex works API."""
    params = {
        "search": query,
        "per_page": min(max_results, 50),
        "mailto": OPENALEX_EMAIL,
    }
    url = f"{OPENALEX_API}/works?{urlencode(params)}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, headers={"User-Agent": "ai-platform-demo/1.0"})
        resp.raise_for_status()
    data = resp.json()
    works = []
    for item in data.get("results", []):
        works.append({
            "id": item.get("id", ""),
            "title": item.get("title", ""),
            "publication_year": item.get("publication_year"),
            "cited_by_count": item.get("cited_by_count", 0),
            "open_access": item.get("open_access", {}).get("is_oa", False),
            "primary_topic": (item.get("primary_topic") or {}).get("display_name", ""),
            "doi": item.get("doi", ""),
        })
    return works


# ---------------------------------------------------------------------------
# Public MCP tools  (same names as the original SQLite tools)
# ---------------------------------------------------------------------------

async def get_topic_statistics(topic: str = "artificial intelligence") -> str:
    """Get research statistics for a topic using OpenAlex and arXiv APIs.

    Queries both OpenAlex (publication counts, citation stats) and arXiv
    (recent paper count) to give a comprehensive view of research activity.

    Args:
        topic: Research topic or field (default: "artificial intelligence").

    Returns:
        JSON with publication statistics, citation data, and open-access info.
    """
    try:
        # OpenAlex concept / field stats
        concept_url = (
            f"{OPENALEX_API}/concepts"
            f"?search={quote_plus(topic)}&per_page=3&mailto={OPENALEX_EMAIL}"
        )
        async with httpx.AsyncClient(timeout=15.0) as client:
            c_resp = await client.get(concept_url, headers={"User-Agent": "ai-platform-demo/1.0"})
            c_resp.raise_for_status()
        concepts = c_resp.json().get("results", [])

        concept_stats = []
        for c in concepts[:3]:
            concept_stats.append({
                "concept": c.get("display_name", ""),
                "level": c.get("level"),
                "works_count": c.get("works_count", 0),
                "cited_by_count": c.get("cited_by_count", 0),
            })

        # arXiv — quick count via search
        arxiv_papers = await _arxiv_search(topic, max_results=5, sort_by="submittedDate", raw_query=False)

        return json.dumps({
            "topic": topic,
            "openalex_concepts": concept_stats,
            "recent_arxiv_papers": len(arxiv_papers),
            "arxiv_sample_titles": [p["title"] for p in arxiv_papers[:3]],
            "data_sources": ["OpenAlex", "arXiv"],
        })

    except Exception as exc:
        logger.exception("Error in get_topic_statistics")
        return json.dumps({"error": str(exc), "status": "failed"})


async def get_recent_findings(limit: int = 10, topic: str = "artificial intelligence") -> str:
    """Return the most recent research papers from arXiv for a given topic.

    Replaces the DB-backed 'recent findings' with live arXiv data, so the
    information is always up-to-date.

    Args:
        limit: Number of papers to return (default 10, max 25).
        topic: Research topic to filter by (default: "artificial intelligence").

    Returns:
        JSON with the latest arXiv papers for the topic.
    """
    try:
        papers = await _arxiv_search(topic, max_results=min(limit, 25), sort_by="submittedDate")
        return json.dumps({
            "topic": topic,
            "count": len(papers),
            "source": "arXiv API",
            "papers": papers,
        })
    except Exception as exc:
        logger.exception("Error in get_recent_findings")
        return json.dumps({"error": str(exc), "status": "failed"})


async def search_findings_by_keyword(keyword: str) -> str:
    """Search academic papers by keyword across arXiv and OpenAlex.

    Args:
        keyword: Search keyword or phrase.

    Returns:
        JSON with matching papers from arXiv and OpenAlex (up to 10 each).
    """
    try:
        arxiv_papers, openalex_works = await _run_parallel_search(keyword)
        return json.dumps({
            "keyword": keyword,
            "arxiv_count": len(arxiv_papers),
            "openalex_count": len(openalex_works),
            "arxiv_papers": arxiv_papers,
            "openalex_works": openalex_works,
            "data_sources": ["arXiv", "OpenAlex"],
        })
    except Exception as exc:
        logger.exception("Error in search_findings_by_keyword")
        return json.dumps({"error": str(exc), "status": "failed"})


async def _run_parallel_search(keyword: str) -> tuple[list, list]:
    """Run arXiv and OpenAlex searches concurrently."""
    import asyncio
    arxiv_task = asyncio.create_task(_arxiv_search(keyword, max_results=8))
    openalex_task = asyncio.create_task(_openalex_search(keyword, max_results=8))
    arxiv_papers = await arxiv_task
    openalex_works = await openalex_task
    return arxiv_papers, openalex_works


async def get_trending_topics(days: int = 7) -> str:
    """Identify trending research topics based on recent arXiv submissions.

    Fetches the most-recently submitted papers in AI/ML and groups them
    by their arXiv category to surface active research areas.

    Args:
        days: Look-back window in days (default 7, max 30).

    Returns:
        JSON with trending topics / categories and representative papers.
    """
    try:
        days = min(days, 30)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y%m%d")

        # Fetch recent AI papers from arXiv using category query syntax
        papers = await _arxiv_search(
            "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV",
            max_results=30,
            sort_by="submittedDate",
            raw_query=True,
        )

        # Group by primary arXiv category
        category_map: dict[str, list[str]] = {}
        for p in papers:
            cats = p.get("categories", [])
            primary = cats[0] if cats else "unknown"
            category_map.setdefault(primary, []).append(p["title"])

        trending = [
            {"category": cat, "paper_count": len(titles), "sample_titles": titles[:2]}
            for cat, titles in sorted(category_map.items(), key=lambda x: -len(x[1]))
        ]

        return json.dumps({
            "window_days": days,
            "cutoff_date": cutoff,
            "total_papers_sampled": len(papers),
            "trending_topics": trending,
            "data_source": "arXiv API",
        })

    except Exception as exc:
        logger.exception("Error in get_trending_topics")
        return json.dumps({"error": str(exc), "status": "failed"})


async def search_openalex_works(query: str, max_results: int = 10) -> str:
    """Search the OpenAlex open scholarly graph for academic works.

    OpenAlex indexes 200M+ scholarly works with citation, authorship, and
    concept metadata — all freely accessible with no API key required.

    Args:
        query:       Search query (title keywords, author, DOI, etc.).
        max_results: Maximum results to return (default 10, max 25).

    Returns:
        JSON with matching academic works and their metadata.
    """
    try:
        works = await _openalex_search(query, max_results=min(max_results, 25))
        return json.dumps({
            "query": query,
            "count": len(works),
            "source": "OpenAlex",
            "works": works,
        })
    except Exception as exc:
        logger.exception("Error in search_openalex_works")
        return json.dumps({"error": str(exc), "status": "failed"})
