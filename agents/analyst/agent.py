"""
Data Analyst Agent
==================
A LangGraph ReAct agent that queries the research database via MCPToolbox
MCP tools to surface statistics, trends, and data-driven insights.

The agent's system prompt is loaded from the prompt registry (with few-shot
examples auto-injected), ensuring consistent, well-engineered behaviour.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterable
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Project root must be in sys.path (set by start.py)
from prompts import PromptLoader
from config import Config

logger = logging.getLogger(__name__)
_config = Config()
_prompt_loader = PromptLoader()


class AnalystAgent:
    """LangGraph ReAct agent backed by MCPToolbox analytics tools."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self) -> None:
        if _config.USE_OLLAMA:
            from langchain_ollama import ChatOllama
            ollama_host = _config.OLLAMA_BASE_URL.rstrip("/").removesuffix("/v1")
            logger.info("Using Ollama LLM backend: %s @ %s", _config.OLLAMA_MODEL, ollama_host)
            self._llm = ChatOllama(
                model=_config.OLLAMA_MODEL,
                base_url=ollama_host,
                temperature=_config.OPENAI_TEMPERATURE,
            )
        else:
            logger.info("Using OpenAI LLM backend: %s", _config.OPENAI_MODEL)
            self._llm = ChatOpenAI(
                model=_config.OPENAI_MODEL,
                api_key=_config.OPENAI_API_KEY,
                temperature=_config.OPENAI_TEMPERATURE,
            )
        self._system_prompt = _prompt_loader.load("analyst_system")

    # ------------------------------------------------------------------
    # Tool loading
    # ------------------------------------------------------------------

    @property
    def _mcp_client(self) -> MultiServerMCPClient:
        return MultiServerMCPClient({
            "toolbox_mcp": {
                "url": _config.TOOLBOX_MCP_URL,
                "transport": "streamable_http",
            },
        })

    async def _get_tools(self) -> list:
        return await self._mcp_client.get_tools()

    # ------------------------------------------------------------------
    # Streaming interface (consumed by AgentExecutor)
    # ------------------------------------------------------------------

    async def stream(
        self,
        query: str,
        context_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        """Run the agent and yield status dicts compatible with the A2A executor.

        Yields intermediate working updates, then a final completed item.
        """
        tools = await self._get_tools()

        graph = create_react_agent(
            self._llm,
            tools,
            prompt=self._system_prompt,
        )
        run_cfg = {"recursion_limit": 6}

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
                            # Stream intermediate updates (tool calls / reasoning)
                            if node_name != "__end__":
                                yield {
                                    "is_task_complete": False,
                                    "require_user_input": False,
                                    "content": content,
                                }

        except Exception:
            logger.exception("Error during analyst agent streaming")
            raise

        yield {
            "is_task_complete": True,
            "require_user_input": False,
            "content": final_answer or "Analysis complete. No data found for the given query.",
        }

    async def stream_verbose(
        self,
        query: str,
        context_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        """Stream raw LangGraph step events for chain-of-thought display."""
        from langchain_core.messages import ToolMessage

        tools = await self._get_tools()
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
