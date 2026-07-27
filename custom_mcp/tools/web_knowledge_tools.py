"""
Web-knowledge tools for the Custom MCP server.

Fetches information from public, no-key-required APIs:
  - Wikipedia REST API   (https://en.wikipedia.org/api/rest_v1/)

arXiv tools have been moved to Toolbox MCP (toolbox_mcp/tools.yaml), which
runs them as native HTTP tools via the MCP Toolbox binary.
"""

from __future__ import annotations

import json
import logging
import os
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

WIKIPEDIA_API = os.environ.get("WIKIPEDIA_API_URL", "https://en.wikipedia.org/api/rest_v1")
# Wikipedia moved the search endpoint to v2 of the MediaWiki REST API
WIKIPEDIA_SEARCH_API = "https://en.wikipedia.org/w/rest.php/v1"

_HEADERS = {"User-Agent": "ai-platform-demo/1.0 (research assistant)"}


# ---------------------------------------------------------------------------
# Wikipedia tools
# ---------------------------------------------------------------------------

async def search_wikipedia(query: str, limit: int = 5) -> str:
    """Search Wikipedia for articles matching a query.

    Uses the Wikipedia REST API (no API key required) to find relevant
    articles and return their titles, summaries, and URLs.

    Args:
        query: Search term or phrase.
        limit: Maximum number of results to return (default 5, max 10).

    Returns:
        JSON with a list of matching Wikipedia articles (title, summary, url).
    """
    limit = min(limit, 10)
    # Wikipedia moved page search to /w/rest.php/v1 (the old /api/rest_v1/page/search returns 404)
    url = f"{WIKIPEDIA_SEARCH_API}/search/page?q={quote(query)}&limit={limit}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=_HEADERS)
            resp.raise_for_status()
        data = resp.json()
        pages = data.get("pages", [])
        results = []
        for p in pages:
            results.append({
                "title": p.get("title", ""),
                "excerpt": p.get("excerpt", "").replace("<span class=\"searchmatch\">", "").replace("</span>", ""),
                "description": p.get("description", ""),
                "url": f"https://en.wikipedia.org/wiki/{quote(p.get('title', '').replace(' ', '_'))}",
            })
        return json.dumps({
            "query": query,
            "count": len(results),
            "source": "Wikipedia",
            "articles": results,
        })
    except Exception as exc:
        logger.exception("Error in search_wikipedia")
        return json.dumps({"error": str(exc), "status": "failed"})


async def get_wikipedia_summary(title: str) -> str:
    """Retrieve a Wikipedia article summary (intro section + key facts).

    Uses the Wikipedia page-summary endpoint which returns a concise,
    human-readable overview of any Wikipedia article.

    Args:
        title: Exact or approximate Wikipedia article title.

    Returns:
        JSON with the article title, extract (plain text), thumbnail info,
        and canonical URL.
    """
    encoded = quote(title.replace(" ", "_"))
    url = f"{WIKIPEDIA_API}/page/summary/{encoded}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=_HEADERS)
            if resp.status_code == 404:
                # Try a search fallback
                search_result = await search_wikipedia(title, limit=1)
                search_data = json.loads(search_result)
                if search_data.get("articles"):
                    first = search_data["articles"][0]["title"]
                    encoded = quote(first.replace(" ", "_"))
                    resp = await client.get(
                        f"{WIKIPEDIA_API}/page/summary/{encoded}", headers=_HEADERS
                    )
            resp.raise_for_status()
        data = resp.json()
        return json.dumps({
            "title": data.get("title", title),
            "description": data.get("description", ""),
            "extract": data.get("extract", "")[:1500],
            "thumbnail": (data.get("thumbnail") or {}).get("source", ""),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "source": "Wikipedia",
        })
    except Exception as exc:
        logger.exception("Error in get_wikipedia_summary")
        return json.dumps({"error": str(exc), "status": "failed"})


