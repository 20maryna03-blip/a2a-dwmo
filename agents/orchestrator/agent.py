"""
Orchestrator Agent
==================
A LangGraph ReAct agent that coordinates the Research Agent and the Data
Analyst Agent via the A2A protocol.

Each downstream agent is wrapped as a LangChain ``StructuredTool`` that
makes an A2A ``message/send`` JSON-RPC call and returns the agent's text
response.  The orchestrator can therefore call both agents in a single
LangGraph reasoning loop.

Prompt engineering:
  - System prompt from the registry (versioned Jinja2 template)
  - Role-based multi-agent coordination instructions
  - Output format specification for structured final answers
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterable
from typing import Any

import httpx
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from prompts import PromptLoader
from config import Config

logger = logging.getLogger(__name__)
_config = Config()
_prompt_loader = PromptLoader()


def _build_llm():
    """Return the active LLM backend: HuggingFace > Ollama > OpenAI."""
    # 1. HuggingFace Inference API (OpenAI-compatible endpoint)
    if _config.USE_HF_LLM and _config.HF_API_KEY and not _config.HF_API_KEY.startswith("hf_your"):
        logger.info("Using HuggingFace LLM backend: %s", _config.HF_MODEL)
        return ChatOpenAI(
            model=_config.HF_MODEL,
            base_url=_config.HF_BASE_URL,
            api_key=_config.HF_API_KEY,
            temperature=_config.OPENAI_TEMPERATURE,
            max_tokens=1024,
        )

    # 2. Ollama (local — no API key needed)
    if _config.USE_OLLAMA:
        from langchain_ollama import ChatOllama
        ollama_host = _config.OLLAMA_BASE_URL.rstrip("/").removesuffix("/v1")
        logger.info("Using Ollama LLM backend: %s @ %s", _config.OLLAMA_MODEL, ollama_host)
        return ChatOllama(
            model=_config.OLLAMA_MODEL,
            base_url=ollama_host,
            temperature=_config.OPENAI_TEMPERATURE,
        )

    # 3. OpenAI fallback
    logger.info("Using OpenAI LLM backend: %s", _config.OPENAI_MODEL)
    return ChatOpenAI(
        model=_config.OPENAI_MODEL,
        api_key=_config.OPENAI_API_KEY,
        temperature=_config.OPENAI_TEMPERATURE,
    )


# ---------------------------------------------------------------------------
# A2A client helper — calls another agent via JSON-RPC 2.0
# ---------------------------------------------------------------------------

async def _call_a2a_agent(base_url: str, query: str, context_id: str) -> str:
    """POST a message to an A2A agent and return the response text."""
    rpc_url = f"{base_url.rstrip('/')}/a2a"
    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "id": str(uuid.uuid4()),
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "messageId": str(uuid.uuid4()),
                "contextId": context_id,
                "parts": [{"kind": "text", "text": query}],
            }
        },
    }

    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(rpc_url, json=payload)
        response.raise_for_status()
        data = response.json()

    if "error" in data:
        raise RuntimeError(f"A2A error from {base_url}: {data['error']}")

    result = data.get("result", {})

    # Extract text from artifacts (primary path: executor calls updater.add_artifact)
    for artifact in result.get("artifacts", []):
        for part in artifact.get("parts", []):
            if isinstance(part, dict) and part.get("kind") == "text":
                return part["text"]

    # Fallback: extract from status message
    status = result.get("status", {})
    if isinstance(status, dict):
        msg = status.get("message", {})
        if isinstance(msg, dict):
            for part in msg.get("parts", []):
                if isinstance(part, dict) and part.get("kind") == "text":
                    return part["text"]

    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool input schemas (Pydantic) — gives the LLM clear parameter guidance
# ---------------------------------------------------------------------------

class ResearcherInput(BaseModel):
    query: str = Field(description="Research task or topic for the Research Agent to investigate")


class AnalystInput(BaseModel):
    query: str = Field(description="Analysis question for the Data Analyst Agent to answer using database tools")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class OrchestratorAgent:
    """LangGraph coordinator that delegates to specialist A2A agents."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, context_id: str) -> None:
        self._context_id = context_id
        self._llm = _build_llm()
        self._system_prompt = _prompt_loader.load("orchestrator_system")

    def _build_tools(self) -> list:
        """Wrap each downstream A2A agent as a LangChain tool."""
        ctx = self._context_id

        async def _call_researcher(query: str) -> str:
            """Delegate a research task to the Research Agent.
            The Research Agent gathers information and saves findings to the
            knowledge base.  Returns a structured research summary."""
            logger.info("Calling Research Agent: %s", query[:80])
            return await _call_a2a_agent(_config.RESEARCHER_AGENT_URL, query, ctx)

        async def _call_analyst(query: str) -> str:
            """Delegate a data analysis task to the Data Analyst Agent.
            The Analyst queries the research database and returns statistics,
            trends, and data-driven insights."""
            logger.info("Calling Analyst Agent: %s", query[:80])
            return await _call_a2a_agent(_config.ANALYST_AGENT_URL, query, ctx)

        researcher_tool = StructuredTool.from_function(
            coroutine=_call_researcher,
            name="researcher",
            description=(
                "Send a research task to the Research Agent. "
                "Use this to gather NEW information on a topic. "
                "The agent will search the knowledge base, identify gaps, "
                "research them, and save findings."
            ),
            args_schema=ResearcherInput,
        )

        analyst_tool = StructuredTool.from_function(
            coroutine=_call_analyst,
            name="analyst",
            description=(
                "Send an analysis question to the Data Analyst Agent. "
                "Use this to get STATISTICS, TRENDS, or INSIGHTS from the "
                "existing research database.  Best used after the researcher "
                "has already gathered data."
            ),
            args_schema=AnalystInput,
        )

        return [researcher_tool, analyst_tool]

    def _make_prompt(self):
        """Return a dynamic prompt callable.

        On the first agent turn (before any tool results exist in the
        conversation) an extra reminder is appended to the system message so
        that small models (e.g. Llama-3.1-8B) reliably call tools instead of
        answering directly from training knowledge.
        """
        sys_text = self._system_prompt

        def prompt(state: dict) -> list:
            messages = state.get("messages", [])
            has_tool_result = any(isinstance(m, ToolMessage) for m in messages)
            if has_tool_result:
                content = sys_text
            else:
                content = (
                    sys_text
                    + "\n\n**ACTION REQUIRED**: Call `researcher` (and/or `analyst`) "
                    "NOW. Do NOT write a final answer yet."
                )
            return [SystemMessage(content=content)] + list(messages)

        return prompt

    async def stream(
        self,
        query: str,
        context_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        """Run the orchestration workflow and stream status dicts."""
        tools = self._build_tools()

        graph = create_react_agent(
            self._llm,
            tools,
            prompt=self._make_prompt(),
        )
        run_cfg = {"recursion_limit": 10}

        final_answer = ""

        try:
            async for chunk in graph.astream(
                {"messages": [("human", query)]},
                config={"configurable": {"thread_id": context_id}, **run_cfg},
                stream_mode="updates",
            ):
                for node_name, node_output in chunk.items():
                    for msg in node_output.get("messages", []):
                        if isinstance(msg, AIMessage) and msg.content:
                            content = str(msg.content)
                            final_answer = content
                            if node_name != "__end__":
                                yield {
                                    "is_task_complete": False,
                                    "require_user_input": False,
                                    "content": content,
                                }

        except Exception:
            logger.exception("Error during orchestrator agent streaming")
            raise

        yield {
            "is_task_complete": True,
            "require_user_input": False,
            "content": final_answer or "I was unable to complete the research task.",
        }

    async def stream_verbose(
        self,
        query: str,
        context_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        """Stream raw LangGraph step events for chain-of-thought display."""
        from langchain_core.messages import ToolMessage

        tools = self._build_tools()
        graph = create_react_agent(self._llm, tools, prompt=self._system_prompt)
        run_cfg = {"recursion_limit": 6}

        async for chunk in graph.astream(
            {"messages": [("human", query)]},
            config={"configurable": {"thread_id": context_id}, **run_cfg},
            stream_mode="updates",
        ):
            for node_name, node_output in chunk.items():
                for msg in node_output.get("messages", []):
                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                args_preview = str(tc.get("args", {}))[:120]
                                yield {
                                    "step_type": "tool_call",
                                    "node": node_name,
                                    "content": f"{tc['name']}({args_preview})",
                                    "raw": msg,
                                }
                        if msg.content:
                            yield {
                                "step_type": "think",
                                "node": node_name,
                                "content": str(msg.content),
                                "raw": msg,
                            }
                    elif isinstance(msg, ToolMessage):
                        result_preview = str(msg.content)[:300]
                        yield {
                            "step_type": "tool_result",
                            "node": node_name,
                            "content": result_preview,
                            "raw": msg,
                        }
