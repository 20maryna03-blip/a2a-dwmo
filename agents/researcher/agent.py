"""
Research Agent
==============
A LangGraph ReAct agent that gathers, synthesises, and persists research
findings via two MCP servers:

  1. Custom MCP (port 8004) — knowledge base (SQLite), Wikipedia, arXiv,
     and HuggingFace multimodal tools (image analysis, summarization, LLM).
  2. Vector MCP  (port 8006) — semantic search over a HuggingFace-embedded
     ChromaDB collection pre-seeded from arXiv and Wikipedia.

LLM backend:
  - Primary:     OpenAI  (OPENAI_API_KEY)
  - Alternative: HuggingFace Inference API (HF_API_KEY + HF_MODEL)
    Set USE_HF_LLM=true in .env to use the HuggingFace backend.

The agent's system prompt is loaded from the prompt registry (Jinja2 template).
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterable
from typing import Any

from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from prompts import PromptLoader
from config import Config

logger = logging.getLogger(__name__)
_config = Config()
_prompt_loader = PromptLoader()


def _build_llm():
    """Return the active LLM backend: Ollama > HuggingFace > OpenAI."""
    from langchain_openai import ChatOpenAI

    # 1. Ollama (local — preferred when USE_OLLAMA=true)
    if _config.USE_OLLAMA:
        from langchain_ollama import ChatOllama
        # OLLAMA_BASE_URL is "http://localhost:11434/v1" — strip the /v1 suffix for ChatOllama
        ollama_host = _config.OLLAMA_BASE_URL.rstrip("/").removesuffix("/v1")
        logger.info("Using Ollama LLM backend: %s @ %s", _config.OLLAMA_MODEL, ollama_host)
        return ChatOllama(
            model=_config.OLLAMA_MODEL,
            base_url=ollama_host,
            temperature=_config.OPENAI_TEMPERATURE,
        )

    # 2. HuggingFace Inference API
    use_hf = os.environ.get("USE_HF_LLM", "false").lower() == "true"
    if use_hf and _config.HF_API_KEY and not _config.HF_API_KEY.startswith("hf-your"):
        try:
            from langchain_huggingface import HuggingFaceEndpoint
            logger.info("Using HuggingFace LLM backend: %s", _config.HF_MODEL)
            return HuggingFaceEndpoint(
                repo_id=_config.HF_MODEL,
                huggingfacehub_api_token=_config.HF_API_KEY,
                temperature=_config.OPENAI_TEMPERATURE,
                max_new_tokens=1024,
                task="text-generation",
            )
        except Exception as exc:
            logger.warning("HuggingFace LLM init failed (%s), falling back to OpenAI.", exc)

    # 3. OpenAI
    logger.info("Using OpenAI LLM backend: %s", _config.OPENAI_MODEL)
    return ChatOpenAI(
        model=_config.OPENAI_MODEL,
        api_key=_config.OPENAI_API_KEY,
        temperature=_config.OPENAI_TEMPERATURE,
    )


class ResearchAgent:
    """LangGraph ReAct agent with Custom MCP + Vector MCP tool access.

    Tool groups available to the agent:
      - Custom MCP: save_finding, search_knowledge_base, list_topics,
        get_findings_by_topic, generate_summary_report, count_findings,
        search_wikipedia, get_wikipedia_summary, fetch_arxiv_papers,
        get_arxiv_paper_details, analyze_image (BLIP multimodal),
        summarize_text (BART), classify_text (zero-shot), generate_text_with_hf
      - Vector MCP: semantic_search, get_vector_db_stats, add_text_to_vector_db,
        populate_from_arxiv, populate_from_wikipedia
    """

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self) -> None:
        self._llm = _build_llm()
        self._system_prompt = _prompt_loader.load("researcher_system")

    # ------------------------------------------------------------------
    # Tool loading via MCP servers
    # ------------------------------------------------------------------

    @property
    def _mcp_client(self) -> MultiServerMCPClient:
        servers: dict[str, dict] = {
            "knowledge_mcp": {
                "url": _config.CUSTOM_MCP_URL,
                "transport": "streamable_http",
            },
        }
        # Attach vector MCP if configured
        vector_url = _config.VECTOR_MCP_URL
        if vector_url:
            servers["vector_mcp"] = {
                "url": vector_url,
                "transport": "streamable_http",
            }
        return MultiServerMCPClient(servers)

    async def _get_tools(self) -> list:
        try:
            return await self._mcp_client.get_tools()
        except Exception as exc:
            # Vector MCP may not be running — fall back to knowledge MCP only
            logger.warning("Could not load all MCP tools (%s). Retrying with knowledge MCP only.", exc)
            fallback = MultiServerMCPClient({
                "knowledge_mcp": {
                    "url": _config.CUSTOM_MCP_URL,
                    "transport": "streamable_http",
                }
            })
            return await fallback.get_tools()

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------

    async def stream(
        self,
        query: str,
        context_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        """Run the research workflow and stream status dicts.

        The agent:
        1. Performs semantic search in the vector DB for related content.
        2. Searches Wikipedia and arXiv for fresh information.
        3. Optionally uses HuggingFace tools (image analysis, summarization).
        4. Saves each distinct finding via the Custom MCP save_finding tool.
        5. Returns a structured summary.
        """
        tools = await self._get_tools()

        graph = create_react_agent(
            self._llm,
            tools,
            prompt=self._system_prompt,
        )
        # Limit tool-call iterations to keep CPU inference time manageable
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
                            if node_name != "__end__":
                                yield {
                                    "is_task_complete": False,
                                    "require_user_input": False,
                                    "content": content,
                                }

        except Exception:
            logger.exception("Error during researcher agent streaming")
            raise

        yield {
            "is_task_complete": True,
            "require_user_input": False,
            "content": final_answer or "Research complete. No new findings were generated.",
        }

    async def stream_verbose(
        self,
        query: str,
        context_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        """Stream raw LangGraph step events for chain-of-thought display.

        Yields dicts with keys:
          - step_type: "think" | "tool_call" | "tool_result" | "final"
          - node:      LangGraph node name ("agent" | "tools")
          - content:   Human-readable description of the step
          - raw:       Original message object (for further inspection)
        """
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
                        # Tool-call decisions (LLM picks a tool)
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                args_preview = str(tc.get("args", {}))[:120]
                                yield {
                                    "step_type": "tool_call",
                                    "node": node_name,
                                    "content": f"{tc['name']}({args_preview})",
                                    "raw": msg,
                                }
                        # Reasoning text (LLM prose)
                        if msg.content:
                            yield {
                                "step_type": "think",
                                "node": node_name,
                                "content": str(msg.content),
                                "raw": msg,
                            }
                    elif isinstance(msg, ToolMessage):
                        # Tool execution result
                        result_preview = str(msg.content)[:300]
                        yield {
                            "step_type": "tool_result",
                            "node": node_name,
                            "content": result_preview,
                            "raw": msg,
                        }
