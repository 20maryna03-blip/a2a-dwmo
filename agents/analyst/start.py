"""
Data Analyst A2A Agent — Start Script
======================================
Port 8003 (default)

Exposes a fully A2A-compliant FastAPI application.  The agent uses LangGraph
with a ReAct loop and queries the MCPToolbox analytics server for data.

AgentCard endpoints:
  GET  /.well-known/agent-card.json  — agent discovery
  POST /a2a                          — JSON-RPC 2.0 message handling
  GET  /                             — health check

Start:
    python start.py              # default port 8003
    python start.py --port 8103
    python start.py --help
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import httpx
import uvicorn
from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from starlette.responses import JSONResponse

# ---------------------------------------------------------------------------
# Path setup — project root for shared modules; this dir for local imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]   # agents/analyst/ → project root
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from config import Config          # noqa: E402
from executor import AnalystExecutor  # noqa: E402

_config = Config()

logging.basicConfig(
    level=_config.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host")
@click.option("--port", default=_config.ANALYST_PORT, show_default=True, help="Listen port")
def main(host: str, port: int) -> None:
    """Start the Data Analyst A2A agent."""

    public_url = _config.ANALYST_URL or f"http://localhost:{port}"

    capabilities = AgentCapabilities(streaming=True, push_notifications=True)
    skill = AgentSkill(
        id="data_analyst",
        name="Data Analyst",
        description=(
            "Specialist agent for querying and analysing research data. "
            "Provides statistics, trend analysis, keyword searches, and "
            "data-driven insights from the research findings database."
        ),
        tags=["data", "analytics", "sql", "trends", "statistics"],
        examples=[
            "What are the most researched topics?",
            "Show me trends from the last 7 days.",
            "Search findings about AI regulation.",
            "Which topics have the most recent activity?",
        ],
    )
    agent_card = AgentCard(
        name="Data Analyst Agent",
        description=(
            "A2A agent that analyses the research knowledge base using MCPToolbox-powered "
            "SQL tools.  Surfaces statistics, trends, and data-driven insights."
        ),
        url=public_url,
        version="1.0.0",
        default_input_modes=AnalystExecutor.SUPPORTED_CONTENT_TYPES
            if hasattr(AnalystExecutor, "SUPPORTED_CONTENT_TYPES")
            else ["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
    )

    httpx_client = httpx.AsyncClient()
    push_config_store = InMemoryPushNotificationConfigStore()
    push_sender = BasePushNotificationSender(
        httpx_client=httpx_client,
        config_store=push_config_store,
    )
    request_handler = DefaultRequestHandler(
        agent_executor=AnalystExecutor(),
        task_store=InMemoryTaskStore(),
        push_config_store=push_config_store,
        push_sender=push_sender,
    )

    server = A2AFastAPIApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    ).build(
        agent_card_url="/.well-known/agent-card.json",
        rpc_url="/a2a",
    )

    server.add_route(
        "/",
        lambda r: JSONResponse({"status": "ok", "agent": "data-analyst", "port": port}),
        methods=["GET"],
    )

    logger.info("Starting Data Analyst agent on %s:%d  →  %s", host, port, public_url)
    uvicorn.run(server, host=host, port=port, log_level=_config.LOG_LEVEL.lower())


if __name__ == "__main__":
    main()
