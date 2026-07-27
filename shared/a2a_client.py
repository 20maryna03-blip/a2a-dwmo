"""
Shared A2A client utilities.

Provides helpers for:
  - Fetching an agent's AgentCard (discovery)
  - Sending a synchronous message/send JSON-RPC call
  - Building a LangChain StructuredTool that wraps an A2A agent

These utilities are used by the Orchestrator agent and the demo_client.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent Card discovery
# ---------------------------------------------------------------------------

async def get_agent_card(base_url: str) -> dict[str, Any]:
    """Fetch and return the agent's AgentCard JSON.

    Args:
        base_url: Base URL of the A2A agent (e.g. ``http://localhost:8002``).

    Returns:
        The parsed AgentCard dict.
    """
    url = f"{base_url.rstrip('/')}/.well-known/agent-card.json"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Synchronous A2A message/send call
# ---------------------------------------------------------------------------

async def call_a2a_agent(
    base_url: str,
    query: str,
    context_id: str | None = None,
    timeout: float = 600.0,
) -> str:
    """Send a message to an A2A agent and return the text response.

    Uses the ``message/send`` JSON-RPC method which blocks until the task
    is complete and returns the final artifact text.

    Args:
        base_url:   Base URL of the target A2A agent.
        query:      User query / instruction text.
        context_id: Optional session context ID for multi-turn conversations.
        timeout:    HTTP request timeout in seconds.

    Returns:
        The agent's text response, extracted from artifacts or status message.

    Raises:
        RuntimeError: If the A2A server returns a JSON-RPC error.
        httpx.HTTPError: On network or HTTP-level failures.
    """
    rpc_url = f"{base_url.rstrip('/')}/a2a"
    ctx = context_id or str(uuid.uuid4())

    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "id": str(uuid.uuid4()),
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "messageId": str(uuid.uuid4()),
                "contextId": ctx,
                "parts": [{"kind": "text", "text": query}],
            }
        },
    }

    logger.debug("A2A → %s | query: %s…", rpc_url, query[:60])

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(rpc_url, json=payload)
        response.raise_for_status()
        data = response.json()

    if "error" in data:
        raise RuntimeError(f"A2A error from {base_url}: {data['error']}")

    result = data.get("result", {})

    # Primary: extract from artifacts
    for artifact in result.get("artifacts", []):
        for part in artifact.get("parts", []):
            if isinstance(part, dict) and part.get("kind") == "text":
                text = part["text"]
                logger.debug("A2A ← %s | %d chars", base_url, len(text))
                return text

    # Fallback: extract from status message
    status = result.get("status", {})
    if isinstance(status, dict):
        msg = status.get("message", {})
        if isinstance(msg, dict):
            for part in msg.get("parts", []):
                if isinstance(part, dict) and part.get("kind") == "text":
                    return part["text"]

    # Last resort: return raw JSON
    return json.dumps(result)


# ---------------------------------------------------------------------------
# LangChain tool builder
# ---------------------------------------------------------------------------

class _AgentToolInput(BaseModel):
    query: str = Field(description="The task or question to send to the agent")


def build_agent_tool(
    name: str,
    description: str,
    base_url: str,
    context_id: str | None = None,
) -> StructuredTool:
    """Build a LangChain StructuredTool that calls an A2A agent.

    The tool is suitable for use in a LangGraph ReAct agent's tool list.

    Args:
        name:        Tool name (snake_case, no spaces).
        description: Description shown to the LLM.
        base_url:    Target A2A agent base URL.
        context_id:  Optional shared session context.

    Returns:
        A StructuredTool wrapping ``call_a2a_agent``.
    """
    ctx = context_id or str(uuid.uuid4())

    async def _invoke(query: str) -> str:
        return await call_a2a_agent(base_url, query, context_id=ctx)

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=name,
        description=description,
        args_schema=_AgentToolInput,
    )
