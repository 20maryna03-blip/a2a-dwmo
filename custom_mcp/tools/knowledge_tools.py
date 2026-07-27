"""
Knowledge base tools for the Custom MCP server.

These tools allow the Research Agent to persist, search, and retrieve
research findings from a shared SQLite database.  Each tool is an
``async def`` function registered with FastMCP via ``mcp.tool()``.

Database schema (auto-created on first use):

    findings(id, topic, content, source, tags TEXT (JSON), created_at)
"""

import json
import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _db_path() -> Path:
    """Resolve the SQLite database path from env or project default."""
    raw = os.environ.get("RESEARCH_DB_PATH", "data/research.db")
    p = Path(raw)
    if not p.is_absolute():
        # Resolve relative to project root (2 levels up from custom_mcp/tools/)
        p = Path(__file__).resolve().parents[2] / raw
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _connect() -> sqlite3.Connection:
    """Return a connection with Row factory and auto-created schema."""
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS findings (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            topic      TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            source     TEXT    DEFAULT 'AI Research',
            tags       TEXT    DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

async def save_finding(
    topic: str,
    content: str,
    source: str = "AI Research",
    tags: str = "[]",
) -> str:
    """Save a research finding to the knowledge base.

    Args:
        topic:   Research topic category (e.g. "AI regulation — EU AI Act").
        content: The finding content — keep focused, ≤ 300 words.
        source:  Origin of the information (default: "AI Research").
        tags:    JSON-encoded array of keyword tags, e.g. ``'["AI","EU"]'``.

    Returns:
        JSON with the new finding's id and a confirmation message.
    """
    if not topic.strip() or not content.strip():
        return json.dumps({"error": "topic and content must not be empty", "status": "failed"})

    try:
        # Validate tags JSON
        parsed_tags = json.loads(tags) if tags else []
        if not isinstance(parsed_tags, list):
            parsed_tags = []
        safe_tags = json.dumps(parsed_tags)
    except json.JSONDecodeError:
        safe_tags = "[]"

    try:
        with _connect() as conn:
            cursor = conn.execute(
                "INSERT INTO findings (topic, content, source, tags) VALUES (?, ?, ?, ?)",
                (topic.strip(), content.strip(), source.strip(), safe_tags),
            )
            finding_id = cursor.lastrowid
            conn.commit()

        logger.info("Saved finding id=%d topic='%s'", finding_id, topic)
        return json.dumps({
            "id": finding_id,
            "topic": topic,
            "status": "saved",
            "message": f"Finding saved with id {finding_id}.",
        })

    except Exception as exc:
        logger.exception("Error saving finding")
        return json.dumps({"error": str(exc), "status": "failed"})


async def search_knowledge_base(
    query: str,
    topic: str | None = None,
    limit: int = 10,
) -> str:
    """Search the knowledge base for relevant findings.

    Args:
        query: Keyword search string.
        topic: Optional topic filter (partial match, case-insensitive).
        limit: Maximum results to return (default 10).

    Returns:
        JSON with matching findings list and total count.
    """
    try:
        like_query = f"%{query}%"
        with _connect() as conn:
            if topic:
                rows = conn.execute(
                    """
                    SELECT id, topic, content, source, tags, created_at
                      FROM findings
                     WHERE topic LIKE ?
                       AND (content LIKE ? OR tags LIKE ?)
                     ORDER BY created_at DESC
                     LIMIT ?
                    """,
                    (f"%{topic}%", like_query, like_query, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, topic, content, source, tags, created_at
                      FROM findings
                     WHERE content LIKE ? OR topic LIKE ? OR tags LIKE ?
                     ORDER BY created_at DESC
                     LIMIT ?
                    """,
                    (like_query, like_query, like_query, limit),
                ).fetchall()

        results = [dict(r) for r in rows]
        logger.info("search '%s' → %d results", query, len(results))
        return json.dumps({"query": query, "count": len(results), "findings": results})

    except Exception as exc:
        logger.exception("Error searching knowledge base")
        return json.dumps({"error": str(exc), "status": "failed"})


async def list_topics() -> str:
    """List all research topics with their finding counts and latest activity date.

    Returns:
        JSON with a sorted list of topics and metadata.
    """
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT topic,
                       COUNT(*)       AS count,
                       MAX(created_at) AS latest
                  FROM findings
                 GROUP BY topic
                 ORDER BY count DESC
                """
            ).fetchall()

        topics = [dict(r) for r in rows]
        return json.dumps({"total_topics": len(topics), "topics": topics})

    except Exception as exc:
        logger.exception("Error listing topics")
        return json.dumps({"error": str(exc), "status": "failed"})


async def get_findings_by_topic(topic: str, limit: int = 20) -> str:
    """Retrieve all findings for a specific topic (partial match).

    Args:
        topic: Topic name or partial name.
        limit: Maximum findings to return (default 20).

    Returns:
        JSON with findings list for the topic.
    """
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT id, topic, content, source, tags, created_at
                  FROM findings
                 WHERE topic LIKE ?
                 ORDER BY created_at DESC
                 LIMIT ?
                """,
                (f"%{topic}%", limit),
            ).fetchall()

        results = [dict(r) for r in rows]
        return json.dumps({"topic": topic, "count": len(results), "findings": results})

    except Exception as exc:
        logger.exception("Error getting findings by topic")
        return json.dumps({"error": str(exc), "status": "failed"})
