# AI Platform Demo — Project Notes

## Architecture
Multi-agent A2A platform with 3 agents and 3 MCP tool servers:

| Service | Port | Description |
|---|---|---|
| Orchestrator Agent | 8001 | LangGraph ReAct agent, delegates to specialists |
| Researcher Agent | 8002 | LangGraph ReAct agent, uses Custom MCP + Vector MCP |
| Analyst Agent | 8003 | LangGraph ReAct agent, uses Toolbox MCP |
| Custom MCP | 8004 | Wikipedia + HuggingFace tools + SQLite knowledge base |
| Toolbox MCP | 8005 | **MCP Toolbox binary** — native HTTP tools: arXiv + OpenAlex |
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

**HuggingFace Inference API (recommended)**
- `USE_HF_LLM=true` + `HF_API_KEY=hf_...` in `.env`
- `HF_MODEL=meta-llama/Llama-3.1-8B-Instruct` (free tier, tool calling supported)
- `HF_MODEL=meta-llama/Llama-3.3-70B-Instruct` (HF Pro — better quality)
- Uses HF's OpenAI-compatible endpoint (`https://api-inference.huggingface.co/v1/`)
- All three agents use `ChatOpenAI` pointed at HF — full `.bind_tools()` support
- Get a free token: https://huggingface.co/settings/tokens

**Ollama (local, no API key)**
- `USE_HF_LLM=false` + `USE_OLLAMA=true` in `.env`
- `OLLAMA_MODEL=llama3.2:1b` (~50s/call on CPU); `qwen2.5:3b` for higher quality
- Requires Ollama running: `OLLAMA_MODELS=~/.ollama/models ollama serve`
- Slow on CPU without GPU — use HF backend instead when possible

**OpenAI (fallback)**
- `USE_HF_LLM=false` + `USE_OLLAMA=false` + `OPENAI_API_KEY=sk-...` in `.env`
- Note: blocked on some corporate networks (Dell/Zscaler proxy)

## Known Issues / Quirks

### CPU inference speed
- `llama3.2:1b`: ~50s per LLM call on CPU
- `qwen2.5:3b`: ~120s per LLM call on CPU
- A multi-step agent call (3-5 LLM rounds) takes 2.5–10 minutes
- Client timeouts set to 600s (agents) and 900s (orchestrator)

### Toolbox MCP — MCP Toolbox binary (genai-toolbox)
- Toolbox MCP is the real `genai-toolbox` binary from `googleapis/mcp-toolbox`
- Binary location: `bin/toolbox` in the project root (committed via Git LFS); override with `TOOLBOX_BIN` in `.env`
- Config: `toolbox_mcp/tools.yaml` — multi-document YAML with HTTP sources + tools
- Sources: `arxiv` (`https://export.arxiv.org/api`) and `openalex` (`https://api.openalex.org`)
- Tools: `search_papers`, `get_trending_ai_papers`, `search_openalex`
- No Python/FastMCP — the binary handles HTTP calls natively
- Docs: https://mcp-toolbox.dev/integrations/http/tools/http-tool/

### arXiv responses (Atom XML)
- arXiv returns Atom XML feed; the toolbox binary passes the raw response to the LLM
- The LLM (llama3.2:1b) is expected to parse the XML — works in practice
- arXiv enforces rate limits (HTTP 429); the binary does not auto-retry, so space out calls

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
