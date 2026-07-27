"""
Orchestrator A2A Agent — Start Script
========================================
Port 8001 (default)

The Orchestrator is the user-facing entry point.  It coordinates the
Research Agent (8002) and Data Analyst Agent (8003) via A2A JSON-RPC calls,
synthesising their outputs into a coherent, structured response.

AgentCard endpoints:
  GET  /.well-known/agent-card.json
  POST /a2a
  GET  /

Start:
    python start.py              # default port 8001
    python start.py --port 8101
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

from config import Config                  # noqa: E402
from executor import OrchestratorExecutor  # noqa: E402

_config = Config()

logging.basicConfig(
    level=_config.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host")
@click.option("--port", default=_config.ORCHESTRATOR_PORT, show_default=True, help="Listen port")
def main(host: str, port: int) -> None:
    """Start the Orchestrator A2A agent."""

    public_url = _config.ORCHESTRATOR_URL or f"http://localhost:{port}"

    capabilities = AgentCapabilities(streaming=True, push_notifications=True)
    skill = AgentSkill(
        id="research_coordinator",
        name="Research Coordinator",
        description=(
            "Coordinates research and analysis tasks across specialist agents. "
            "Delegates information gathering to the Research Agent and data analysis "
            "to the Analyst Agent, then synthesises a comprehensive answer."
        ),
        tags=["orchestration", "research", "multi-agent", "coordinator"],
        examples=[
            "Research AI regulation trends and give me a data-backed summary.",
            "What is the current state of quantum computing and what does our data show?",
            "Gather information about sustainable energy and analyse what we already know.",
        ],
    )
    agent_card = AgentCard(
        name="Research Orchestrator Agent",
        description=(
            "A2A multi-agent coordinator.  Delegates to specialist Research and Analyst "
            "agents, then synthesises their outputs into structured, data-backed answers."
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
        agent_executor=OrchestratorExecutor(),
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
        lambda r: JSONResponse({
            "status": "ok",
            "agent": "orchestrator",
            "port": port,
            "downstream": {
                "researcher": _config.RESEARCHER_AGENT_URL,
                "analyst": _config.ANALYST_AGENT_URL,
            },
        }),
        methods=["GET"],
    )

    logger.info(
        "Starting Orchestrator agent on %s:%d  →  %s\n"
        "  researcher: %s\n  analyst:    %s",
        host, port, public_url,
        _config.RESEARCHER_AGENT_URL,
        _config.ANALYST_AGENT_URL,
    )
    uvicorn.run(server, host=host, port=port, log_level=_config.LOG_LEVEL.lower())


if __name__ == "__main__":
    main()
