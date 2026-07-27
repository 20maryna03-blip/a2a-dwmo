"""
AI Platform Demo — End-to-End Client
======================================
Demonstrates the full multi-agent platform by sending queries through
the Orchestrator and observing how the A2A agents collaborate.

Prerequisites:
  1. Copy .env.template to .env and fill in at minimum OPENAI_API_KEY.
     Optionally set HF_API_KEY for HuggingFace multimodal tools.
  2. Install dependencies:  pip install -e .
  3. Start all components:  ./start_all.sh
  4. Run this script:       python demo_client.py

Demo scenarios:
  1. Service health check
  2. Agent discovery (AgentCard)
  3. Research task (Researcher Agent — Wikipedia + arXiv + Vector search)
  4. Analysis task (Analyst Agent — arXiv + OpenAlex analytics)
  5. HuggingFace tools (image analysis, summarization, text generation)
  6. Semantic vector search (Vector MCP — ChromaDB + HF embeddings)
  7. Full pipeline (Orchestrator coordinates all agents)
  8. Prompt engineering showcase
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx

from shared.a2a_client import call_a2a_agent, get_agent_card

ORCHESTRATOR_URL = "http://localhost:8001"
RESEARCHER_URL   = "http://localhost:8002"
ANALYST_URL      = "http://localhost:8003"
CUSTOM_MCP_URL   = "http://localhost:8004"
TOOLBOX_MCP_URL  = "http://localhost:8005"
VECTOR_MCP_URL   = "http://localhost:8006"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    line = "═" * 70
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def _section(label: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")


def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=80, initial_indent=prefix, subsequent_indent=prefix)


async def _health_check(name: str, url: str) -> bool:
    """Return True if the service responds to GET /."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url)
            ok = r.status_code < 400
            status = "OK" if ok else f"HTTP {r.status_code}"
    except Exception as exc:
        ok = False
        status = str(exc)
    symbol = "✓" if ok else "✗"
    print(f"  {symbol}  {name:<28} {url}  [{status}]")
    return ok


async def _call_mcp_tool(mcp_url: str, tool_name: str, args: dict) -> dict:
    """Direct MCP tool call for demo purposes (bypasses agent layer)."""
    # Use httpx to POST to the MCP server's tool endpoint
    payload = {"tool": tool_name, "arguments": args}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{mcp_url.rstrip('/mcp')}/mcp", json=payload)
        resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

async def demo_agent_discovery() -> None:
    """Show agent cards for all three A2A agents."""
    _banner("DEMO 1 — Agent Discovery (/.well-known/agent-card.json)")

    for name, url in [
        ("Orchestrator", ORCHESTRATOR_URL),
        ("Researcher",   RESEARCHER_URL),
        ("Analyst",      ANALYST_URL),
    ]:
        try:
            card = await get_agent_card(url)
            print(f"\n  {name} Agent Card:")
            print(f"    name:        {card.get('name')}")
            print(f"    description: {card.get('description', '')[:80]}…")
            skills = card.get("skills", [])
            for skill in skills:
                print(f"    skill:       {skill.get('id')} — {skill.get('description', '')[:60]}…")
        except Exception as exc:
            print(f"\n  ✗  {name}: {exc}")


async def demo_research_task() -> None:
    """Send a research task directly to the Research Agent."""
    _banner("DEMO 2 — Research Agent (Wikipedia + arXiv + Vector Search, port 8002)")

    query = (
        "Research the key advantages and limitations of transformer-based "
        "large language models for scientific research. Use Wikipedia for "
        "background, arXiv for recent papers, and semantic search for related work."
    )
    print(f"\n  Query → Research Agent:\n{_wrap(query)}\n")

    try:
        response = await call_a2a_agent(RESEARCHER_URL, query, timeout=600.0)
        print("  Response:")
        for line in response.split("\n"):
            print(f"    {line}")
    except Exception as exc:
        print(f"  ✗  Error: {exc}")
        print("     Is the Research Agent running?  python agents/researcher/start.py")


async def demo_analysis_task() -> None:
    """Send an analysis task directly to the Analyst Agent (arXiv + OpenAlex)."""
    _banner("DEMO 3 — Analyst Agent (arXiv + OpenAlex analytics, port 8003)")

    query = (
        "What are the trending research topics in AI this week? "
        "Search for recent arXiv papers and give me statistics on the most active areas."
    )
    print(f"\n  Query → Analyst Agent:\n{_wrap(query)}\n")

    try:
        response = await call_a2a_agent(ANALYST_URL, query, timeout=600.0)
        print("  Response:")
        for line in response.split("\n"):
            print(f"    {line}")
    except Exception as exc:
        print(f"  ✗  Error: {exc}")
        print("     Is the Analyst Agent running?  python agents/analyst/start.py")


async def demo_huggingface_tools() -> None:
    """Demo HuggingFace multimodal tools via the Research Agent."""
    _banner("DEMO 4 — HuggingFace Multimodal Tools (Custom MCP, port 8004)")
    print()
    print("  This demo asks the Research Agent to use HuggingFace tools:")
    print("    • analyze_image   — BLIP multimodal image captioning")
    print("    • summarize_text  — BART abstractive summarization")
    print("    • classify_text   — zero-shot text classification")
    print("    • generate_text_with_hf — Mistral-7B open-source LLM")
    print()
    print("  NOTE: Requires HF_API_KEY in .env. Tools gracefully report")
    print("        'unavailable' if the key is not set.")
    print()

    query = (
        "Please do the following:\n"
        "1. Classify this text using zero-shot classification with labels "
        "'machine learning, cybersecurity, climate change, healthcare': "
        "'Attention mechanisms in transformer models have revolutionised NLP.'\n"
        "2. Summarize this text: 'Large language models are neural networks trained "
        "on vast amounts of text data. They learn statistical patterns in language "
        "and can generate coherent, contextually relevant text. Applications include "
        "code generation, summarization, translation, and question answering.'\n"
        "3. Generate a short explanation of federated learning using the HuggingFace LLM."
    )
    print(f"  Query → Research Agent:\n{_wrap(query)}\n")

    try:
        response = await call_a2a_agent(RESEARCHER_URL, query, timeout=600.0)
        print("  Response:")
        for line in response.split("\n"):
            print(f"    {line}")
    except Exception as exc:
        print(f"  ✗  Error: {exc}")


async def demo_vector_search() -> None:
    """Demo semantic search via the Vector MCP server."""
    _banner("DEMO 5 — Semantic Vector Search (HuggingFace Embeddings + ChromaDB, port 8006)")
    print()
    print("  The Vector MCP server is pre-seeded with arXiv papers and Wikipedia")
    print("  articles on AI/ML topics, embedded via sentence-transformers/all-MiniLM-L6-v2.")
    print()

    query = (
        "Use semantic search to find papers about 'attention mechanisms in transformers'. "
        "Then also check what research topics are indexed in the vector database "
        "and add a note about your findings."
    )
    print(f"  Query → Research Agent:\n{_wrap(query)}\n")

    try:
        response = await call_a2a_agent(RESEARCHER_URL, query, timeout=600.0)
        print("  Response:")
        for line in response.split("\n"):
            print(f"    {line}")
    except Exception as exc:
        print(f"  ✗  Error: {exc}")
        print("     Is the Vector MCP running?  python vector_mcp/start.py")


async def demo_full_pipeline() -> None:
    """Send a full research-and-analyse query through the Orchestrator."""
    _banner("DEMO 6 — Full Pipeline via Orchestrator (A2A coordination, port 8001)")

    query = (
        "I need a comprehensive overview of multimodal AI systems in 2024. "
        "Please: (1) research the topic using Wikipedia, arXiv, and semantic search; "
        "(2) analyse trending arXiv papers and OpenAlex statistics on this topic; "
        "(3) synthesise a data-backed executive summary with key trends and findings."
    )
    print(f"\n  Query → Orchestrator Agent:\n{_wrap(query)}\n")
    print("  (The orchestrator coordinates Research Agent and Analyst Agent via A2A…)\n")

    try:
        response = await call_a2a_agent(ORCHESTRATOR_URL, query, timeout=900.0)
        print("  Final synthesised response:")
        print()
        for line in response.split("\n"):
            print(f"    {line}")
    except Exception as exc:
        print(f"  ✗  Error: {exc}")
        print("     Is the Orchestrator running?  python agents/orchestrator/start.py")


async def demo_prompt_engineering() -> None:
    """Show the prompt registry and render a sample prompt."""
    _banner("DEMO 7 — Prompt Engineering (registry + Jinja2 templates)")

    from prompts import PromptLoader

    loader = PromptLoader()

    _section("Registered prompts")
    for name in loader.list_prompts():
        meta = loader.get_metadata(name)
        print(f"  • {name:<30} v{meta['latest_version']}  tags={meta['tags']}")

    _section("Rendered researcher_system prompt (first 600 chars)")
    rendered = loader.load("researcher_system", include_examples=False)
    print()
    print(rendered[:600])
    print("  …")

    _section("Rendered report_format prompt with context variables")
    report_prompt = loader.load(
        "report_format",
        context={"topic": "Multimodal AI", "version": "1.0", "finding_count": 15},
        include_examples=False,
    )
    print()
    print(report_prompt[:500])
    print("  …")


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

async def check_all_services() -> bool:
    """Check all services are up before running demos."""
    _banner("SERVICE HEALTH CHECK")
    print()

    results = await asyncio.gather(
        _health_check("Custom MCP (Knowledge+HF)", CUSTOM_MCP_URL),
        _health_check("Toolbox MCP (arXiv+OpenAlex)", TOOLBOX_MCP_URL),
        _health_check("Vector MCP (ChromaDB+HF embed)", VECTOR_MCP_URL),
        _health_check("Analyst Agent",     ANALYST_URL),
        _health_check("Researcher Agent",  RESEARCHER_URL),
        _health_check("Orchestrator",      ORCHESTRATOR_URL),
    )
    all_ok = all(results)
    if not all_ok:
        print("\n  Some services are down. Run ./start_all.sh to start all components.")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("\n" + "█" * 70)
    print("  AI PLATFORM DEMO — Multi-Agent Research Platform")
    print("  A2A Agents · Custom MCP · Toolbox MCP · Vector MCP")
    print("  HuggingFace Embeddings · ChromaDB · Multimodal LLM")
    print("█" * 70)

    all_up = await check_all_services()

    # Prompt engineering demo always works (no network needed)
    await demo_prompt_engineering()

    if not all_up:
        print("\n  Skipping agent demos — start all services first.")
        print("  Run: ./start_all.sh")
        return

    await demo_agent_discovery()
    await demo_research_task()
    await demo_analysis_task()
    await demo_huggingface_tools()
    await demo_vector_search()
    await demo_full_pipeline()

    _banner("DEMO COMPLETE")
    print()
    print("  Architecture summary:")
    print("    Orchestrator (OpenAI GPT)")
    print("    ├── Researcher Agent (OpenAI or HuggingFace Mistral-7B)")
    print("    │     ├── Custom MCP  — SQLite + Wikipedia + arXiv + HF multimodal")
    print("    │     └── Vector MCP  — HuggingFace all-MiniLM-L6-v2 + ChromaDB")
    print("    └── Analyst Agent  (OpenAI GPT)")
    print("          └── Toolbox MCP — arXiv + OpenAlex live analytics")
    print()
    print("  Data sources (all public, no API keys required):")
    print("    • arXiv        https://export.arxiv.org/api/query")
    print("    • OpenAlex     https://api.openalex.org")
    print("    • Wikipedia    https://en.wikipedia.org/api/rest_v1")
    print()
    print("  HuggingFace models (HF_API_KEY required):")
    print("    • Salesforce/blip-image-captioning-large  (multimodal / vision)")
    print("    • facebook/bart-large-cnn                 (summarization)")
    print("    • facebook/bart-large-mnli                (zero-shot classification)")
    print("    • mistralai/Mistral-7B-Instruct-v0.1      (text generation LLM)")
    print("    • sentence-transformers/all-MiniLM-L6-v2  (embeddings — runs locally)")


if __name__ == "__main__":
    asyncio.run(main())
