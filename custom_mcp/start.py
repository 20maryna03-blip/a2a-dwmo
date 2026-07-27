"""
Custom MCP Server — Knowledge Management + Web Knowledge + HuggingFace
=======================================================================
FastMCP server that exposes three groups of tools to research agents:

 1. Knowledge Base tools (SQLite — local persistence):
      save_finding, search_knowledge_base, list_topics,
      get_findings_by_topic, generate_summary_report, count_findings

 2. Web Knowledge tools (public APIs — no key required):
      search_wikipedia, get_wikipedia_summary,
      fetch_arxiv_papers, get_arxiv_paper_details

 3. HuggingFace tools (Inference API — HF_API_KEY required):
      analyze_image  (BLIP — multimodal image captioning)
      summarize_text (BART — abstractive summarization)
      classify_text  (BART-MNLI — zero-shot classification)
      generate_text_with_hf (Mistral-7B — open-source LLM)

Start:
    python start.py              # default port 8004
    python start.py --port 8104  # custom port
    python start.py --help
"""

import errno
import logging
import socket
import sys
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Path setup — allows imports from project root and this component
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from config import Config  # noqa: E402
from tools.knowledge_tools import (  # noqa: E402
    get_findings_by_topic,
    list_topics,
    save_finding,
    search_knowledge_base,
)
from tools.report_tools import count_findings, generate_summary_report  # noqa: E402
from tools.web_knowledge_tools import (  # noqa: E402
    fetch_arxiv_papers,
    get_arxiv_paper_details,
    get_wikipedia_summary,
    search_wikipedia,
)
from tools.hf_tools import (  # noqa: E402
    analyze_image,
    classify_text,
    generate_text_with_hf,
    summarize_text,
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
    name="knowledge-management-mcp",
    instructions=(
        "Comprehensive knowledge management server with three capability groups:\n"
        "1. LOCAL KNOWLEDGE BASE — save_finding, search_knowledge_base, list_topics, "
        "   get_findings_by_topic, generate_summary_report, count_findings\n"
        "2. WEB KNOWLEDGE (free APIs) — search_wikipedia, get_wikipedia_summary, "
        "   fetch_arxiv_papers, get_arxiv_paper_details\n"
        "3. HUGGINGFACE AI (requires HF_API_KEY) — analyze_image (BLIP multimodal), "
        "   summarize_text (BART), classify_text (zero-shot), generate_text_with_hf (Mistral-7B)\n\n"
        "Workflow: Use web/HF tools to gather information, then save_finding to persist it."
    ),
)

# Group 1 — Local knowledge base
mcp.tool()(save_finding)
mcp.tool()(search_knowledge_base)
mcp.tool()(list_topics)
mcp.tool()(get_findings_by_topic)
mcp.tool()(generate_summary_report)
mcp.tool()(count_findings)

# Group 2 — Web knowledge (public APIs)
mcp.tool()(search_wikipedia)
mcp.tool()(get_wikipedia_summary)
mcp.tool()(fetch_arxiv_papers)
mcp.tool()(get_arxiv_paper_details)

# Group 3 — HuggingFace AI tools (multimodal + NLP)
mcp.tool()(analyze_image)
mcp.tool()(summarize_text)
mcp.tool()(classify_text)
mcp.tool()(generate_text_with_hf)


@mcp.custom_route("/", methods=["GET"])
async def healthcheck(request: Request) -> JSONResponse:
    """Health check — required by proxies and orchestrators."""
    return JSONResponse({
        "status": "ok",
        "server": "knowledge-management-mcp",
        "port": _config.MCP_SERVER_PORT,
        "tool_groups": ["knowledge_base", "web_knowledge", "huggingface"],
    })


# ---------------------------------------------------------------------------
# Port guard helper
# ---------------------------------------------------------------------------
def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                logger.error("Port %d is already in use. Change MCP_SERVER_PORT or free the port.", port)
                return True
            raise


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
@click.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host")
@click.option("--port", default=_config.MCP_SERVER_PORT, show_default=True, help="Listen port")
def main(host: str, port: int) -> None:
    """Start the Custom MCP Knowledge Management server."""
    if _is_port_in_use(host, port):
        sys.exit(1)

    logger.info("Starting Custom MCP server on %s:%d  →  http://%s:%d/mcp", host, port, host, port)

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
