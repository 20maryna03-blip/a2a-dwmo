"""
Quick test runner for the AI Platform Demo.

Runs a series of targeted calls — from fast MCP tool tests (seconds)
to full agent calls (minutes on CPU).  Each test prints pass/fail and
elapsed time so you can judge what works without waiting for the full
demo_client.py suite.

Usage:
    python run_tests.py              # run all tests
    python run_tests.py --fast       # MCP + health checks only (< 30 s)
    python run_tests.py --agents     # agent calls only (slow, ~2–5 min each)
    python run_tests.py --cot        # chain-of-thought: live step-by-step display
    python run_tests.py --cot --agent researcher   # COT for one agent only
    python run_tests.py --cot --agent analyst
    python run_tests.py --cot --agent orchestrator
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path


def _load_module(rel_path: str):
    """Load a Python module by file path, bypassing sys.path namespace conflicts."""
    root = Path(__file__).parent
    path = root / rel_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
from shared.a2a_client import call_a2a_agent, get_agent_card

PASS = "PASS"
FAIL = "FAIL"

ORCHESTRATOR_URL = "http://localhost:8001"
RESEARCHER_URL   = "http://localhost:8002"
ANALYST_URL      = "http://localhost:8003"
CUSTOM_MCP_URL   = "http://localhost:8004"
TOOLBOX_MCP_URL  = "http://localhost:8005"
VECTOR_MCP_URL   = "http://localhost:8006"

results: list[tuple[str, str, float, str]] = []   # (test, status, elapsed, detail)


def _log(label: str, status: str, elapsed: float, detail: str = "") -> None:
    icon = "✓" if status == PASS else "✗"
    print(f"  {icon} [{elapsed:5.1f}s]  {label}")
    if detail:
        for line in detail.strip().splitlines()[:4]:
            print(f"              {line}")
    results.append((label, status, elapsed, detail))


async def _http_get(url: str, timeout: float = 5.0) -> tuple[int, dict | str]:
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(url)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text


async def _mcp_tool(mcp_url: str, tool: str, args: dict, timeout: float = 30.0) -> dict:
    """Call a FastMCP tool via SSE JSON-RPC."""
    payload = {
        "jsonrpc": "2.0", "id": "1", "method": "tools/call",
        "params": {"name": tool, "arguments": args},
    }
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(f"{mcp_url.rstrip('/')}", json=payload)
        r.raise_for_status()
        data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    content = data.get("result", {}).get("content", [])
    if content and isinstance(content[0], dict):
        return json.loads(content[0].get("text", "{}"))
    return data


# ─────────────────────────────────────────────────────────────
# FAST tests: health + MCP tools (< 30 s total)
# ─────────────────────────────────────────────────────────────

async def test_health_all() -> None:
    """All 6 services respond on their ports."""
    print("\n── Health checks ──────────────────────────────────────────")
    services = [
        ("Ollama :11434",        "http://localhost:11434/api/version"),
        ("Orchestrator :8001",   ORCHESTRATOR_URL),
        ("Researcher :8002",     RESEARCHER_URL),
        ("Analyst :8003",        ANALYST_URL),
        ("Custom MCP :8004",     CUSTOM_MCP_URL),
        ("Toolbox MCP :8005",    TOOLBOX_MCP_URL),
        ("Vector MCP :8006",     VECTOR_MCP_URL),
    ]
    for name, url in services:
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(url)
            _log(name, PASS if r.status_code < 400 else FAIL, time.time() - t0,
                 f"HTTP {r.status_code}")
        except Exception as e:
            _log(name, FAIL, time.time() - t0, str(e))


async def test_agent_cards() -> None:
    """Agent discovery — AgentCard JSON from each agent."""
    print("\n── Agent discovery ────────────────────────────────────────")
    for name, url in [("Orchestrator", ORCHESTRATOR_URL),
                      ("Researcher",   RESEARCHER_URL),
                      ("Analyst",      ANALYST_URL)]:
        t0 = time.time()
        try:
            card = await get_agent_card(url)
            _log(f"AgentCard:{name}", PASS, time.time() - t0,
                 f"name={card.get('name')!r}  version={card.get('version')!r}")
        except Exception as e:
            _log(f"AgentCard:{name}", FAIL, time.time() - t0, str(e))


async def test_wikipedia_search() -> None:
    """Custom MCP — Wikipedia search tool."""
    print("\n── MCP tool: Wikipedia search ─────────────────────────────")
    t0 = time.time()
    try:
        wk = _load_module("custom_mcp/tools/web_knowledge_tools.py")
        raw = await wk.search_wikipedia("transformer neural network", limit=3)
        d = json.loads(raw)
        assert d["count"] > 0, "no results"
        _log("search_wikipedia", PASS, time.time() - t0,
             f"count={d['count']}  first={d['articles'][0]['title']!r}")
    except Exception as e:
        _log("search_wikipedia", FAIL, time.time() - t0, str(e))


async def test_arxiv_search() -> None:
    """Toolbox MCP — search_papers via live MCP Toolbox binary HTTP endpoint."""
    print("\n── MCP tool: arXiv search (Toolbox binary) ────────────────")
    t0 = time.time()
    try:
        payload = {
            "jsonrpc": "2.0", "id": "t-arxiv", "method": "tools/call",
            "params": {"name": "search_papers",
                       "arguments": {"search_query": "all:large language models", "max_results": 3}},
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "http://localhost:8005/mcp",
                json=payload,
                headers={"Content-Type": "application/json",
                         "Accept": "application/json, text/event-stream"},
            )
        resp.raise_for_status()
        body = resp.text
        data_line = next((l[5:] for l in body.splitlines() if l.startswith("data:")), body)
        result = json.loads(data_line)
        content = result.get("result", {}).get("content", [{}])[0].get("text", "")
        assert "entry" in content or "xml" in content.lower(), f"unexpected: {content[:100]}"
        _log("search_papers (arXiv via Toolbox)", PASS, time.time() - t0,
             f"got {len(content)} chars of Atom XML")
    except Exception as e:
        _log("search_papers (arXiv via Toolbox)", FAIL, time.time() - t0, str(e))


async def test_vector_search() -> None:
    """Vector MCP — semantic search via live HTTP endpoint."""
    print("\n── MCP tool: Vector semantic search ───────────────────────")
    t0 = time.time()
    try:
        # Call the running Vector MCP server via its /mcp endpoint (JSON-RPC)
        payload = {
            "jsonrpc": "2.0", "id": "t1", "method": "tools/call",
            "params": {"name": "semantic_search",
                       "arguments": {"query": "attention mechanism transformer", "n_results": 3}},
        }
        headers = {"Accept": "application/json, text/event-stream"}
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.post(VECTOR_MCP_URL + "/mcp", json=payload, headers=headers)
            r.raise_for_status()
        # FastMCP responds with SSE (text/event-stream); extract the data: line
        raw_body = r.text
        data_line = next(
            (ln[len("data:"):].strip() for ln in raw_body.splitlines() if ln.startswith("data:")),
            None,
        )
        if data_line is None:
            raise RuntimeError(f"No data line in SSE response: {raw_body[:200]}")
        data = json.loads(data_line)
        if "error" in data:
            raise RuntimeError(data["error"])
        content = data.get("result", {}).get("content", [])
        text = content[0].get("text", "{}") if content else "{}"
        d = json.loads(text)
        total = d.get("total_results", 0)
        first = d["results"][0]["title"][:50] if d.get("results") else "(db empty — still seeding?)"
        _log("semantic_search (ChromaDB)", PASS, time.time() - t0,
             f"total={total}  first={first!r}")
    except Exception as e:
        _log("semantic_search", FAIL, time.time() - t0, str(e))


# ─────────────────────────────────────────────────────────────
# SLOW tests: A2A agent calls (2–5 min each on CPU)
# ─────────────────────────────────────────────────────────────

async def test_researcher_simple() -> None:
    """Researcher Agent — simple factual question (1–2 LLM calls)."""
    print("\n── Agent: Researcher (simple) ─────────────────────────────")
    print("   (llama3.2:1b on CPU — expect 1–3 min)")
    t0 = time.time()
    try:
        result = await call_a2a_agent(
            RESEARCHER_URL,
            "In two sentences, what is the attention mechanism in transformers?",
            timeout=600.0,
        )
        _log("Researcher: attention mechanism", PASS, time.time() - t0, result[:200])
    except Exception as e:
        _log("Researcher: attention mechanism", FAIL, time.time() - t0, str(e))


async def test_researcher_with_tools() -> None:
    """Researcher Agent — query that triggers Wikipedia + arXiv tool calls."""
    print("\n── Agent: Researcher (with tools) ─────────────────────────")
    print("   (llama3.2:1b + MCP tools — expect 3–6 min)")
    t0 = time.time()
    try:
        result = await call_a2a_agent(
            RESEARCHER_URL,
            "Search Wikipedia for 'BERT language model' and summarise what you find in 3 bullet points.",
            timeout=600.0,
        )
        _log("Researcher: Wikipedia BERT lookup", PASS, time.time() - t0, result[:250])
    except Exception as e:
        _log("Researcher: Wikipedia BERT lookup", FAIL, time.time() - t0, str(e))


async def test_analyst_trending() -> None:
    """Analyst Agent — arXiv trending topics query."""
    print("\n── Agent: Analyst (arXiv trending) ────────────────────────")
    print("   (llama3.2:1b + Toolbox MCP — expect 3–8 min)")
    t0 = time.time()
    try:
        result = await call_a2a_agent(
            ANALYST_URL,
            "List the top 3 most-cited AI papers from arXiv this month. Be brief.",
            timeout=600.0,
        )
        _log("Analyst: arXiv trending topics", PASS, time.time() - t0, result[:250])
    except Exception as e:
        _log("Analyst: arXiv trending topics", FAIL, time.time() - t0, str(e))


async def test_orchestrator_pipeline() -> None:
    """Orchestrator — delegates research + analysis across both sub-agents."""
    print("\n── Agent: Orchestrator (full pipeline) ─────────────────────")
    print("   (coordinates Researcher + Analyst — expect 5–15 min)")
    t0 = time.time()
    try:
        result = await call_a2a_agent(
            ORCHESTRATOR_URL,
            "Give me a one-paragraph overview of recent advances in large language models, "
            "drawing on both Wikipedia background and recent arXiv papers.",
            timeout=900.0,
        )
        _log("Orchestrator: LLM overview", PASS, time.time() - t0, result[:300])
    except Exception as e:
        _log("Orchestrator: LLM overview", FAIL, time.time() - t0, str(e))


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

def print_summary() -> None:
    print("\n" + "═" * 60)
    print("  RESULTS SUMMARY")
    print("═" * 60)
    passed = [r for r in results if r[1] == PASS]
    failed = [r for r in results if r[1] == FAIL]
    for label, status, elapsed, _ in results:
        icon = "✓" if status == PASS else "✗"
        print(f"  {icon} {label:<40} {elapsed:6.1f}s")
    print("─" * 60)
    print(f"  {len(passed)} passed  /  {len(failed)} failed  /  {len(results)} total")
    print()


# ─────────────────────────────────────────────────────────────
# CHAIN-OF-THOUGHT mode — live step-by-step display
# ─────────────────────────────────────────────────────────────

# ANSI colour codes (fall back gracefully if terminal doesn't support them)
_C = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "dim":    "\033[2m",
    "cyan":   "\033[36m",
    "yellow": "\033[33m",
    "green":  "\033[32m",
    "blue":   "\033[34m",
    "magenta":"\033[35m",
    "red":    "\033[31m",
}

def _cprint(color: str, prefix: str, text: str) -> None:
    c = _C.get(color, "")
    r = _C["reset"]
    b = _C["bold"]
    # Wrap long lines at 100 chars
    import textwrap
    lines = text.strip().splitlines() or [""]
    first = True
    for line in lines:
        for wrapped in textwrap.wrap(line, width=96) or [""]:
            pad = f"  {b}{c}{prefix}{r}  " if first else " " * (len(prefix) + 4)
            print(f"{pad}{wrapped}")
            first = False


def _cot_header(agent_name: str, query: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {_C['bold']}CHAIN OF THOUGHT — {agent_name}{_C['reset']}")
    print(f"  Query: {query[:80]}")
    print("═" * 70)


def _print_step(step: dict, step_num: int) -> None:
    stype = step["step_type"]
    content = step["content"]
    node = step.get("node", "")

    if stype == "tool_call":
        # Split "tool_name({'arg': 'val'})" for display
        paren = content.find("(")
        tool_name = content[:paren] if paren != -1 else content
        args_str  = content[paren:] if paren != -1 else ""
        print(f"\n  {_C['bold']}{_C['yellow']}[Step {step_num}] TOOL CALL{_C['reset']}")
        _cprint("yellow", "  tool ▶", tool_name)
        if args_str:
            _cprint("dim",    "  args  ", args_str)

    elif stype == "tool_result":
        print(f"\n  {_C['bold']}{_C['cyan']}[Step {step_num}] TOOL RESULT{_C['reset']}")
        _cprint("cyan", "  ◀ out ", content)

    elif stype == "think":
        print(f"\n  {_C['bold']}{_C['green']}[Step {step_num}] THINK ({node}){_C['reset']}")
        _cprint("green", "  ✎     ", content)

    elif stype == "final":
        print(f"\n  {_C['bold']}{_C['magenta']}[FINAL ANSWER]{_C['reset']}")
        _cprint("magenta", "  ★     ", content)


async def run_cot_agent(agent_name: str, query: str) -> None:
    """Instantiate an agent directly and stream its chain of thought."""
    import importlib
    import os
    root = Path(__file__).parent
    os.chdir(root)

    # --- isolate sys.path for this agent ---
    # Remove all other agent directories that may have been inserted by a previous run
    all_agent_dirs = [str(root / "agents" / n) for n in ("researcher", "analyst", "orchestrator")]
    for d in all_agent_dirs:
        while d in sys.path:
            sys.path.remove(d)

    # Remove stale cached modules that differ per agent (agent.py, config.py, executor.py)
    for cached in list(sys.modules.keys()):
        if cached in ("agent", "config", "executor"):
            del sys.modules[cached]

    # Insert the correct agent directory at the front
    agent_dir = str(root / "agents" / agent_name)
    shared_dir = str(root / "shared")
    for d in [agent_dir, shared_dir, str(root)]:
        if d not in sys.path:
            sys.path.insert(0, d)

    _cot_header(agent_name.upper(), query)

    t0 = time.time()
    step_num = 0

    try:
        # Use importlib to force a fresh load from the correct file,
        # bypassing any stale sys.modules entry for "agent".
        agent_mod = importlib.import_module("agent")

        if agent_name == "researcher":
            agent_obj = agent_mod.ResearchAgent()
        elif agent_name == "analyst":
            agent_obj = agent_mod.AnalystAgent()
        elif agent_name == "orchestrator":
            agent_obj = agent_mod.OrchestratorAgent(context_id=str(time.time()))
        else:
            print(f"  Unknown agent: {agent_name!r}")
            return

        stream_fn = agent_obj.stream_verbose
        ctx = str(time.time())
        print(f"\n  {_C['dim']}Starting… (each LLM call takes ~50 s on CPU){_C['reset']}\n")

        async for step in stream_fn(query, ctx):
            step_num += 1
            _print_step(step, step_num)

        elapsed = time.time() - t0
        print(f"\n{'─' * 70}")
        print(f"  Completed in {elapsed:.0f}s  |  {step_num} steps")

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  {_C['red']}ERROR after {elapsed:.0f}s: {exc}{_C['reset']}")
        import traceback; traceback.print_exc()


async def run_cot(agent_filter: str | None = None) -> None:
    """Run COT mode for one or all agents."""
    scenarios = [
        ("researcher",   "Search Wikipedia for 'BERT language model' and give me 3 key facts."),
        ("analyst",      "What are the top trending AI topics on arXiv this week? Be brief."),
        ("orchestrator", "Give me a short overview of large language models using both research and analysis."),
    ]
    for agent_name, query in scenarios:
        if agent_filter and agent_name != agent_filter:
            continue
        await run_cot_agent(agent_name, query)
        print()


async def run_fast() -> None:
    """MCP + health checks only — completes in < 30 s."""
    # Run from the project root so relative imports work
    import os; os.chdir(Path(__file__).parent)
    await test_health_all()
    await test_agent_cards()
    await test_wikipedia_search()
    await test_arxiv_search()
    await test_vector_search()
    print_summary()


async def run_agents() -> None:
    """Agent calls only (slow)."""
    import os; os.chdir(Path(__file__).parent)
    await test_researcher_simple()
    await test_researcher_with_tools()
    await test_analyst_trending()
    await test_orchestrator_pipeline()
    print_summary()


async def run_all() -> None:
    import os; os.chdir(Path(__file__).parent)
    await test_health_all()
    await test_agent_cards()
    await test_wikipedia_search()
    await test_arxiv_search()
    await test_vector_search()
    await test_researcher_simple()
    await test_researcher_with_tools()
    await test_analyst_trending()
    await test_orchestrator_pipeline()
    print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Platform Demo test runner")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fast",   action="store_true", help="Health + MCP tool tests only (< 30 s)")
    group.add_argument("--agents", action="store_true", help="Agent A2A calls only (slow, CPU)")
    group.add_argument("--cot",    action="store_true", help="Chain-of-thought: live step-by-step display")
    parser.add_argument("--agent", choices=["researcher", "analyst", "orchestrator"],
                        help="Filter --cot to one agent only")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  AI Platform Demo — Test Runner")
    print("  Model: llama3.2:1b via Ollama (local CPU)")
    print("═" * 60)

    if args.fast:
        asyncio.run(run_fast())
    elif args.agents:
        asyncio.run(run_agents())
    elif args.cot:
        asyncio.run(run_cot(agent_filter=args.agent))
    else:
        asyncio.run(run_all())
