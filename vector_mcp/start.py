"""
Vector MCP Server — Semantic Search with HuggingFace Embeddings + ChromaDB
===========================================================================
FastMCP server that provides semantic (vector) search over a local ChromaDB
collection populated from public APIs.

Architecture:
  - Embeddings: HuggingFace sentence-transformers (runs locally, no GPU needed)
  - Vector DB:  ChromaDB (persistent local storage)
  - Data seed:  arXiv API + Wikipedia REST API (no keys required)

On first start, the server automatically seeds the vector DB with papers
from arXiv for the topics defined in SEED_TOPICS (.env).

Tools provided:
  - semantic_search          — similarity search by natural-language query
  - get_vector_db_stats      — collection info (doc count, sources, topics)
  - add_text_to_vector_db    — index a custom text snippet
  - populate_from_arxiv      — fetch + embed arXiv papers for any topic
  - populate_from_wikipedia  — fetch + embed Wikipedia article summaries

Start:
    python start.py              # default port 8006
    python start.py --port 8106
    python start.py --no-seed    # skip auto-seeding (faster cold start)
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
from tools.vector_tools import (  # noqa: E402
    add_text_to_vector_db,
    get_vector_db_stats,
    init_vector_db,
    populate_from_arxiv,
    populate_from_wikipedia,
    semantic_search,
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
# DB path resolution
# ---------------------------------------------------------------------------

def _resolve_db_path() -> str:
    raw = _config.VECTOR_DB_PATH
    p = Path(raw)
    if not p.is_absolute():
        p = _PROJECT_ROOT / raw
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="vector-search-mcp",
    instructions=(
        "Semantic vector search server powered by HuggingFace sentence-transformer "
        "embeddings and a local ChromaDB persistent store.  The collection is "
        "pre-seeded with arXiv research papers and Wikipedia articles.\n\n"
        "Primary tool: semantic_search(query) — finds the most semantically "
        "similar documents to a natural-language query (better than keyword search).\n\n"
        "Data management: populate_from_arxiv(topic) and populate_from_wikipedia(topics) "
        "fetch and embed new documents from public APIs.  add_text_to_vector_db() "
        "lets you index any custom text.  Use get_vector_db_stats() to inspect "
        "collection contents."
    ),
)

mcp.tool()(semantic_search)
mcp.tool()(get_vector_db_stats)
mcp.tool()(add_text_to_vector_db)
mcp.tool()(populate_from_arxiv)
mcp.tool()(populate_from_wikipedia)


@mcp.custom_route("/", methods=["GET"])
async def healthcheck(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "server": "vector-search-mcp",
        "embedding_model": _config.HF_EMBED_MODEL,
        "db_path": _config.VECTOR_DB_PATH,
        "port": _config.VECTOR_MCP_PORT,
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
# Auto-seeding
# ---------------------------------------------------------------------------

async def _auto_seed() -> None:
    """Seed the vector DB from arXiv + Wikipedia on first start."""
    import json
    import asyncio

    stats_raw = await get_vector_db_stats()
    stats = json.loads(stats_raw)
    existing = stats.get("total_documents", 0)

    if existing > 0:
        logger.info("Vector DB already has %d documents — skipping auto-seed.", existing)
        return

    logger.info("Auto-seeding vector DB from arXiv and Wikipedia …")
    topics = [t.strip() for t in _config.SEED_TOPICS.split(",") if t.strip()]
    n = _config.SEED_PAPERS_PER_TOPIC

    for topic in topics:
        logger.info("  Seeding arXiv papers for topic: %s", topic)
        result = await populate_from_arxiv(topic, max_papers=n)
        r = json.loads(result)
        logger.info("  Added %d arXiv papers for '%s'", r.get("added", 0), topic)

    # Seed a selection of foundational Wikipedia AI articles
    wiki_topics = (
        "Transformer (deep learning),BERT (language model),GPT-3,"
        "Reinforcement learning,Federated learning,Diffusion model,"
        "Large language model,Generative adversarial network"
    )
    logger.info("  Seeding Wikipedia articles …")
    result = await populate_from_wikipedia(wiki_topics)
    r = json.loads(result)
    logger.info("  Added %d Wikipedia articles", r.get("added", 0))

    stats_raw = await get_vector_db_stats()
    stats = json.loads(stats_raw)
    logger.info(
        "Auto-seed complete. Total documents in vector DB: %d",
        stats.get("total_documents", 0),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host")
@click.option("--port", default=_config.VECTOR_MCP_PORT, show_default=True, help="Listen port")
@click.option(
    "--no-seed",
    is_flag=True,
    default=False,
    help="Skip auto-seeding the vector DB at startup.",
)
def main(host: str, port: int, no_seed: bool) -> None:
    """Start the Vector MCP semantic search server."""
    if _is_port_in_use(host, port):
        sys.exit(1)

    db_path = _resolve_db_path()
    logger.info("Initialising vector DB at %s with model %s", db_path, _config.HF_EMBED_MODEL)
    init_vector_db(db_path, _config.HF_EMBED_MODEL)

    if not no_seed:
        # Run async seeding in a sync context before the server loop starts
        import asyncio
        asyncio.run(_auto_seed())

    logger.info(
        "Starting Vector MCP server on %s:%d  →  http://%s:%d/mcp",
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
