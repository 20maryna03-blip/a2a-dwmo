# AI Platform Demo — Project Notes

## Architecture
Multi-agent A2A platform with 3 agents and 3 MCP tool servers:

| Service | Port | Description |
|---|---|---|
| Orchestrator Agent | 8001 | LangGraph ReAct agent, delegates to specialists |
| Researcher Agent | 8002 | LangGraph ReAct agent, uses Custom MCP + Vector MCP |
| Analyst Agent | 8003 | LangGraph ReAct agent, uses Toolbox MCP |
| Custom MCP | 8004 | Wikipedia + arXiv + HuggingFace tools |
| Toolbox MCP | 8005 | arXiv + OpenAlex analytics |
| Vector MCP | 8006 | HuggingFace embeddings + ChromaDB semantic search |

## Commands

```bash
# Install dependencies
pip install -e .

# Start all services (Ollama must be running first if USE_OLLAMA=true)
./start_all.sh

# Stop all services
./stop_all.sh

# Run demo scenarios
python demo_client.py
```

## LLM Backend

**Ollama (default, recommended)**
- `USE_OLLAMA=true` in `.env`
- `OLLAMA_MODEL=llama3.2:1b` (fast on CPU, ~50s/call)
- `OLLAMA_MODEL=qwen2.5:3b` (higher quality, ~120s/call on CPU)
- Ollama must be running: `ollama serve`
- Pull model: `ollama pull llama3.2:1b`
- Both models support full tool/function calling required by LangGraph ReAct agents.

**OpenAI (optional)**
- Set `USE_OLLAMA=false` and `OPENAI_API_KEY=sk-...` in `.env`
- Note: blocked on some corporate networks (Dell/Zscaler proxy)

## Known Issues / Quirks

### CPU inference speed
- `llama3.2:1b`: ~50s per LLM call on CPU
- `qwen2.5:3b`: ~120s per LLM call on CPU
- A multi-step agent call (3-5 LLM rounds) takes 2.5–10 minutes
- Client timeouts set to 600s (agents) and 900s (orchestrator)

### arXiv rate limiting
- arXiv API enforces rate limits on burst requests (returns HTTP 429)
- The analytics tools retry up to 3 times with 10/20/30s back-off
- If rate-limited, the agent will return a graceful error message

### Wikipedia API (v2 search)
- Wikipedia retired the `GET /api/rest_v1/page/search` endpoint (returns 404)
- Search now uses `GET /w/rest.php/v1/search/page` (v2 MediaWiki REST API)
- Summary endpoint (`/api/rest_v1/page/summary/{title}`) is still v1

### Config / .env loading
- Agent configs use absolute path resolution for `.env`:
  `_ENV_FILE = Path(__file__).parent.parent.parent / ".env"`
- This ensures correct loading regardless of the working directory.
- All three agent configs (researcher, analyst, orchestrator) use this pattern.

### Ollama setup on WSL2 (no GPU)
- Binary extracted from GitHub release tarball (no sudo needed)
- Placed in `~/miniforge3/bin/ollama`
- Companion libs in `~/miniforge3/lib/ollama/`
- Start server: `OLLAMA_MODELS=~/.ollama/models ollama serve`
- `start_all.sh` auto-starts Ollama if `USE_OLLAMA=true` and it's not running

### LangGraph + ChatOllama
- Use `langchain-ollama` package (`ChatOllama` class), NOT `ChatOpenAI` with Ollama base_url
- `langchain-core 1.5.x` routes `ainvoke` through `_astream` when tools are bound,
  which breaks with `ChatOpenAI` + Ollama but works correctly with `ChatOllama`
- `recursion_limit=6` set on `graph.astream` config to cap tool-call iterations

## Verification

```bash
# Health check all services
python -c "
import asyncio, httpx
async def main():
    async with httpx.AsyncClient(timeout=5) as c:
        for n,p in [('Orchestrator',8001),('Researcher',8002),('Analyst',8003),
                    ('Custom MCP',8004),('Toolbox MCP',8005),('Vector MCP',8006)]:
            try: r=await c.get(f'http://localhost:{p}'); print(f'OK   {n} [{r.status_code}]')
            except Exception as e: print(f'FAIL {n}: {e}')
asyncio.run(main())
"

# Test Researcher agent (expect ~2 min on CPU)
python -c "
import asyncio, sys
sys.path.insert(0, '.')
from shared.a2a_client import call_a2a_agent
async def main():
    r = await call_a2a_agent('http://localhost:8002', 'What is a transformer model?', timeout=600.0)
    print(r[:300])
asyncio.run(main())
"
```
