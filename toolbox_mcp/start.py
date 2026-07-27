"""
MCPToolbox MCP Server — Research Analytics (Public-API Edition)
===============================================================
FastMCP server that exposes live academic-paper analytics via public APIs:
  - arXiv  (https://export.arxiv.org/api/query)  — no key required
  - OpenAlex (https://api.openalex.org)           — no key required

Tools provided:
  - get_topic_statistics       — paper counts & citation stats for a topic
  - get_recent_findings        — latest arXiv papers for a topic
  - search_findings_by_keyword — full-text search across arXiv + OpenAlex
  - get_trending_topics        — most active arXiv categories in last N days
  - search_openalex_works      — OpenAlex scholarly graph search

Start:
    python start.py              # default port 8005
    python start.py --port 8105  # custom port
    python start.py --help
"""

import errno
import logging
import socket
import sys
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from config import Config  # noqa: E402
from tools.analytics_tools import (  # noqa: E402
    get_recent_findings,
    get_topic_statistics,
    get_trending_topics,
    search_findings_by_keyword,
    search_openalex_works,
)

from fastmcp import FastMCP  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402

_config = Config()

logging.basicConfig(
    level=_config.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="toolbox-analytics-mcp",
    instructions=(
        "Academic research analytics server powered by arXiv and OpenAlex public APIs. "
        "No database or API key required — all data is fetched live. "
        "Use get_topic_statistics for overview stats, get_recent_findings for the "
        "latest papers on a topic, search_findings_by_keyword for keyword-based "
        "cross-source search, get_trending_topics to identify active arXiv research "
        "areas, and search_openalex_works to query the OpenAlex scholarly graph."
    ),
)

mcp.tool()(get_topic_statistics)
mcp.tool()(get_recent_findings)
mcp.tool()(search_findings_by_keyword)
mcp.tool()(get_trending_topics)
mcp.tool()(search_openalex_works)


@mcp.custom_route("/", methods=["GET"])
async def healthcheck(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "server": "toolbox-analytics-mcp",
        "mode": "public-api (arXiv + OpenAlex)",
        "port": _config.TOOLBOX_PORT,
    })


# ---------------------------------------------------------------------------
# Port guard
# ---------------------------------------------------------------------------
def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                logger.error("Port %d is already in use.", port)
                return True
            raise


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
@click.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host")
@click.option("--port", default=_config.TOOLBOX_PORT, show_default=True, help="Listen port")
def main(host: str, port: int) -> None:
    """Start the MCPToolbox analytics MCP server (public-API edition)."""
    if _is_port_in_use(host, port):
        sys.exit(1)

    logger.info(
        "Starting Toolbox MCP (public-API) on %s:%d  →  http://%s:%d/mcp",
        host, port, host, port,
    )

    try:
        mcp.run(
            transport="streamable-http",
            host=host,
            port=port,
            path="/mcp",
            stateless_http=True,
        )
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            logger.error("Port %d already in use.", port)
        else:
            raise
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
