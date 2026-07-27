"""
Research A2A Agent — Start Script
===================================
Port 8002 (default)

The Research Agent gathers information on any topic and persists findings
to the shared knowledge base via the Custom MCP server (port 8004).

AgentCard endpoints:
  GET  /.well-known/agent-card.json
  POST /a2a
  GET  /

Start:
    python start.py              # default port 8002
    python start.py --port 8102
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
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from config import Config              # noqa: E402
from executor import ResearcherExecutor  # noqa: E402

_config = Config()

logging.basicConfig(
    level=_config.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host")
@click.option("--port", default=_config.RESEARCHER_PORT, show_default=True, help="Listen port")
def main(host: str, port: int) -> None:
    """Start the Research A2A agent."""

    public_url = _config.RESEARCHER_URL or f"http://localhost:{port}"

    capabilities = AgentCapabilities(streaming=True, push_notifications=True)
    skill = AgentSkill(
        id="knowledge_researcher",
        name="Knowledge Researcher",
        description=(
            "Specialist agent that researches any topic, generates structured findings, "
            "and saves them to the knowledge base.  Checks for existing knowledge before "
            "researching to avoid duplication."
        ),
        tags=["research", "knowledge", "information-gathering"],
        examples=[
            "Research the impact of AI on healthcare.",
            "Gather information about quantum computing trends.",
            "What do we know about sustainable energy storage?",
        ],
    )
    agent_card = AgentCard(
        name="Research Agent",
        description=(
            "A2A agent that researches topics and persists findings via the Custom MCP "
            "knowledge management server.  Uses chain-of-thought prompting to ensure "
            "systematic, comprehensive coverage of each topic."
        ),
        url=public_url,
        version="1.0.0",
        default_input_modes=["text"],
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
        agent_executor=ResearcherExecutor(),
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
        lambda r: JSONResponse({"status": "ok", "agent": "researcher", "port": port}),
        methods=["GET"],
    )

    logger.info("Starting Research agent on %s:%d  →  %s", host, port, public_url)
    uvicorn.run(server, host=host, port=port, log_level=_config.LOG_LEVEL.lower())


if __name__ == "__main__":
    main()
