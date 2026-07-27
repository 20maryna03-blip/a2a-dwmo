"""
Report generation tools for the Custom MCP server.

These tools build Markdown reports and simple aggregates directly from
the shared SQLite knowledge base, giving agents a ready-made narrative
without needing the Analyst Agent.
"""

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def _db_path() -> Path:
    raw = __import__("os").environ.get("RESEARCH_DB_PATH", "data/research.db")
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / raw
    return p


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


async def generate_summary_report(topic: str) -> str:
    """Generate a structured Markdown research report for a topic.

    Pulls all stored findings for the topic and formats them as an
    executive-style report with key insights and source attribution.

    Args:
        topic: Topic name (partial match supported).

    Returns:
        JSON with ``report`` (Markdown string) and ``finding_count``.
    """
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT content, source, tags, created_at
                  FROM findings
                 WHERE topic LIKE ?
                 ORDER BY created_at DESC
                """,
                (f"%{topic}%",),
            ).fetchall()

        if not rows:
            return json.dumps({
                "topic": topic,
                "finding_count": 0,
                "report": f"No findings available for topic: **{topic}**. "
                          "Run the Research Agent first to gather information.",
            })

        findings = [dict(r) for r in rows]
        date_range = f"{findings[-1]['created_at'][:10]} → {findings[0]['created_at'][:10]}"

        lines = [
            f"# Research Report: {topic}",
            f"\n**Findings**: {len(findings)}  |  **Period**: {date_range}",
            "\n---\n",
            "## Key Findings\n",
        ]

        for i, f in enumerate(findings[:10], 1):
            try:
                tag_list = json.loads(f.get("tags") or "[]")
                tag_str = "  " + " ".join(f"`{t}`" for t in tag_list) if tag_list else ""
            except (json.JSONDecodeError, TypeError):
                tag_str = ""

            lines += [
                f"### {i}. Finding{tag_str}",
                f"{f['content']}",
                f"*Source: {f['source']} | {f['created_at'][:10]}*\n",
            ]

        if len(findings) > 10:
            lines.append(f"\n*…and {len(findings) - 10} additional finding(s) not shown.*")

        report = "\n".join(lines)
        return json.dumps({"topic": topic, "finding_count": len(findings), "report": report})

    except Exception as exc:
        logger.exception("Error generating report")
        return json.dumps({"error": str(exc), "status": "failed"})


async def count_findings(topic: str | None = None) -> str:
    """Count findings, optionally filtered by topic.

    Args:
        topic: Optional partial topic name filter.

    Returns:
        JSON with count details.
    """
    try:
        with _connect() as conn:
            if topic:
                count = conn.execute(
                    "SELECT COUNT(*) FROM findings WHERE topic LIKE ?",
                    (f"%{topic}%",),
                ).fetchone()[0]
                return json.dumps({"topic": topic, "count": count})
            else:
                total = conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
                n_topics = conn.execute("SELECT COUNT(DISTINCT topic) FROM findings").fetchone()[0]
                return json.dumps({"total_findings": total, "total_topics": n_topics})

    except Exception as exc:
        logger.exception("Error counting findings")
        return json.dumps({"error": str(exc), "status": "failed"})
