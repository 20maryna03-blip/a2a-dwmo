# AI Platform Demo — Multi-Agent Research Platform

A production-style, end-to-end demo of a multi-agent AI research platform built with:

- **A2A Protocol** (Google) — inter-agent communication
- **LangGraph** — ReAct agent orchestration
- **FastMCP** — three MCP servers (Custom, Toolbox, Vector)
- **HuggingFace** — embeddings, multimodal (BLIP vision), open-source LLM (Mistral-7B)
- **ChromaDB** — local persistent vector database
- **Public APIs** — arXiv, OpenAlex, Wikipedia (no keys required)
- **Prompt Engineering** — versioned Jinja2 templates with few-shot examples

---

## Architecture

```
User → Orchestrator Agent (OpenAI GPT, port 8001)
            ├── Researcher Agent (OpenAI or HuggingFace Mistral-7B, port 8002)
            │     ├── Custom MCP  (port 8004) ── knowledge base + Wikipedia + arXiv + HF multimodal
            │     └── Vector MCP  (port 8006) ── HuggingFace embeddings + ChromaDB semantic search
            └── Analyst Agent   (OpenAI GPT, port 8003)
                  └── Toolbox MCP (port 8005) ── arXiv + OpenAlex live analytics
```

### Components

| Component | Port | Description |
|---|---|---|
| Orchestrator Agent | 8001 | A2A coordinator — delegates to Researcher and Analyst |
| Researcher Agent | 8002 | Gathers knowledge via Custom MCP + Vector MCP |
| Analyst Agent | 8003 | Analytics queries via Toolbox MCP |
| Custom MCP | 8004 | Knowledge base + Wikipedia + arXiv + HuggingFace tools |
| Toolbox MCP | 8005 | arXiv + OpenAlex academic analytics (no DB needed) |
| Vector MCP | 8006 | Semantic search (HuggingFace embeddings + ChromaDB) |

---

## Public APIs Used (no keys required)

| API | Used by | What it provides |
|---|---|---|
| [arXiv](https://export.arxiv.org/api/query) | Toolbox MCP, Custom MCP, Vector MCP | Research paper search + trending topics |
| [OpenAlex](https://api.openalex.org) | Toolbox MCP | Scholarly graph — 200M+ works, citation stats |
| [Wikipedia REST API](https://en.wikipedia.org/api/rest_v1) | Custom MCP, Vector MCP | Article summaries, encyclopaedic knowledge |

---

## HuggingFace Models

| Model | Used by | Capability |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Vector MCP | Local embeddings for ChromaDB (no GPU, no key) |
| `Salesforce/blip-image-captioning-large` | Custom MCP | Multimodal image captioning (requires HF_API_KEY) |
| `facebook/bart-large-cnn` | Custom MCP | Abstractive text summarization (requires HF_API_KEY) |
| `facebook/bart-large-mnli` | Custom MCP | Zero-shot text classification (requires HF_API_KEY) |
| `mistralai/Mistral-7B-Instruct-v0.1` | Researcher Agent (optional) | Open-source LLM backend (requires HF_API_KEY) |

---

## MCP Tool Reference

### Custom MCP (port 8004) — 14 tools in 3 groups

**Group 1 — Local Knowledge Base** (SQLite, no network needed)
- `save_finding(topic, content, source, tags)` — persist a research finding
- `search_knowledge_base(query, topic, limit)` — keyword search findings
- `list_topics()` — list all research topics
- `get_findings_by_topic(topic, limit)` — retrieve findings for a topic
- `generate_summary_report(topic)` — generate a Markdown report
- `count_findings(topic)` — count findings (total or per-topic)

**Group 2 — Web Knowledge** (public APIs, no key)
- `search_wikipedia(query, limit)` — search Wikipedia articles
- `get_wikipedia_summary(title)` — get a Wikipedia article summary
- `fetch_arxiv_papers(query, max_results)` — search arXiv papers
- `get_arxiv_paper_details(arxiv_id)` — full details for one paper

**Group 3 — HuggingFace AI** (requires `HF_API_KEY`)
- `analyze_image(image_url)` — BLIP multimodal image captioning
- `summarize_text(text, max_length)` — BART abstractive summarization
- `classify_text(text, candidate_labels)` — zero-shot classification
- `generate_text_with_hf(prompt, max_new_tokens)` — Mistral-7B LLM generation

### Toolbox MCP (port 8005) — 5 tools

- `get_topic_statistics(topic)` — OpenAlex concept stats + arXiv paper count
- `get_recent_findings(limit, topic)` — latest arXiv papers for a topic
- `search_findings_by_keyword(keyword)` — cross-source search (arXiv + OpenAlex)
- `get_trending_topics(days)` — trending arXiv categories in last N days
- `search_openalex_works(query, max_results)` — OpenAlex scholarly graph search

### Vector MCP (port 8006) — 5 tools

- `semantic_search(query, n_results, source_filter)` — cosine similarity search
- `get_vector_db_stats()` — collection info (doc count, sources, top topics)
- `add_text_to_vector_db(text, title, topic, source, url)` — index custom text
- `populate_from_arxiv(topic, max_papers)` — fetch + embed arXiv papers
- `populate_from_wikipedia(topics)` — fetch + embed Wikipedia summaries

---

## Quick Start

### 1. Prerequisites

```bash
python >= 3.11
pip install -e .
```

### 2. Choose your LLM backend

**Option A — Ollama (recommended, runs fully offline, no API key)**

```bash
# Install Ollama (Linux/WSL2)
curl -fsSL https://ollama.com/install.sh | sh
# Pull a small model with tool-calling support (~1.3 GB)
ollama pull llama3.2:1b
```

Then in `.env`:
```
USE_OLLAMA=true
OLLAMA_MODEL=llama3.2:1b
```

> **CPU performance note:** `llama3.2:1b` generates ~0.6 words/s on CPU.
> Expect 2–5 min per agent call. For faster results pull `qwen2.5:3b` and set
> `OLLAMA_MODEL=qwen2.5:3b` (higher quality, ~2 min/call on a modern laptop).
> Ollama auto-detects GPU if available and runs at full speed.

**Option B — OpenAI**

```bash
cp .env.template .env
# Set:
#   USE_OLLAMA=false
#   OPENAI_API_KEY=sk-...
#   HF_API_KEY=hf-...    (optional — enables HuggingFace multimodal tools)
```

### 3. Start all components

```bash
./start_all.sh
```

> **First run note:** The Vector MCP downloads the `all-MiniLM-L6-v2` embedding
> model (~80 MB) and seeds ChromaDB with arXiv + Wikipedia data. This takes
> 30–60 seconds. Subsequent starts are instant (data is cached locally).

### 4. Run the demo

```bash
python demo_client.py
```

### 5. Stop all components

```bash
./stop_all.sh
```

---

## Manual Component Start

```bash
# MCP servers (start first)
python custom_mcp/start.py --port 8004
python toolbox_mcp/start.py --port 8005
python vector_mcp/start.py --port 8006       # --no-seed to skip auto-seeding

# A2A agents (start after MCP servers)
python agents/analyst/start.py --port 8003
python agents/researcher/start.py --port 8002
python agents/orchestrator/start.py --port 8001
```

---

## Vector MCP — How It Works

The Vector MCP creates a **local semantic search engine** for research papers
and encyclopaedic articles:

1. **Embedding model** — `sentence-transformers/all-MiniLM-L6-v2` runs locally
   via the `sentence-transformers` library. No GPU or API key needed.

2. **Vector database** — ChromaDB stores embeddings persistently at
   `data/vector_db/` (cosine similarity index).

3. **Auto-seeding** — On first start, the server fetches papers from arXiv and
   articles from Wikipedia for topics defined in `SEED_TOPICS` (.env):
   ```
   SEED_TOPICS=artificial intelligence,machine learning,natural language processing
   ```

4. **On-demand expansion** — Use `populate_from_arxiv(topic)` or
   `populate_from_wikipedia(topics)` to add more content at any time.

---

## HuggingFace Multimodal Integration

The Researcher Agent has access to multimodal tools via the Custom MCP:

```
# Analyze an image
analyze_image("https://example.com/chart.png")
→ "a bar chart showing accuracy metrics for three transformer models"

# Summarize long text
summarize_text("Large language models are neural networks trained on...")
→ "Large language models are neural networks that learn from vast text data..."

# Zero-shot classify
classify_text("Attention mechanisms have improved NLP", "machine learning,cybersecurity")
→ { "top_label": "machine learning", "score": 0.97 }

# Use Mistral-7B (set USE_HF_LLM=true to use as the agent's main LLM)
generate_text_with_hf("Explain federated learning in one paragraph")
→ "Federated learning is a machine learning approach..."
```

---

## Prompt Engineering

Prompts are managed via a versioned registry with Jinja2 templates and
few-shot examples:

```bash
prompts/
  registry.py                    # PromptRegistry — version control
  loader.py                      # PromptLoader  — render + inject examples
  templates/
    orchestrator_system.j2       # Orchestrator coordination instructions
    researcher_system.j2         # Researcher with tool guidance
    analyst_system.j2            # Analyst with query guidance
    report_format.j2             # Report generation format
  examples/
    researcher_system_examples.json
    analyst_system_examples.json
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `USE_OLLAMA` | `true` | Route all agents through local Ollama (recommended) |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `OLLAMA_MODEL` | `llama3.2:1b` | Ollama model name (must be pulled first) |
| `OPENAI_API_KEY` | — | OpenAI API key (used when `USE_OLLAMA=false`) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model |
| `HF_API_KEY` | — | HuggingFace Hub token (optional — for multimodal tools) |
| `HF_MODEL` | `mistralai/Mistral-7B-Instruct-v0.1` | HF text-gen model (requires Pro plan) |
| `HF_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (runs locally) |
| `HF_VISION_MODEL` | `Salesforce/blip-image-captioning-large` | Vision/multimodal model |
| `VECTOR_DB_PATH` | `data/vector_db` | ChromaDB storage path |
| `SEED_TOPICS` | `artificial intelligence,...` | Auto-seed topics for Vector MCP |
| `CUSTOM_MCP_URL` | `http://localhost:8004/mcp` | Custom MCP endpoint |
| `TOOLBOX_MCP_URL` | `http://localhost:8005/mcp` | Toolbox MCP endpoint |
| `VECTOR_MCP_URL` | `http://localhost:8006/mcp` | Vector MCP endpoint |
