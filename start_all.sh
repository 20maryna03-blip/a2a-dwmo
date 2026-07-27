#!/usr/bin/env bash
# =============================================================================
# Start all AI Platform Demo components
# Each component runs in its own background process; PIDs are saved to .pids/
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS_DIR="$ROOT/.pids"
LOGS_DIR="$ROOT/.logs"
mkdir -p "$PIDS_DIR" "$LOGS_DIR"

# Load environment if .env exists
if [[ -f "$ROOT/.env" ]]; then
    set -a; source "$ROOT/.env"; set +a
    echo "Loaded .env"
else
    echo "WARNING: .env not found. Copy .env.template to .env and fill in your values."
fi

start_component() {
    local name="$1"
    local script="$2"
    local port="$3"
    local extra_args="${4:-}"
    local log="$LOGS_DIR/${name}.log"
    local pid_file="$PIDS_DIR/${name}.pid"

    echo "Starting $name on port $port..."
    PYTHONPATH="$ROOT" python "$ROOT/$script" --port "$port" $extra_args \
        > "$log" 2>&1 &
    echo $! > "$pid_file"
    echo "  PID=$(cat "$pid_file")  log=$log"
    sleep 1
}

start_toolbox_mcp() {
    local port="${TOOLBOX_PORT:-8005}"
    local log="$LOGS_DIR/toolbox_mcp.log"
    local pid_file="$PIDS_DIR/toolbox_mcp.pid"
    local bin="${TOOLBOX_BIN:-$ROOT/bin/toolbox}"
    local config="$ROOT/toolbox_mcp/tools.yaml"

    echo "Starting Toolbox MCP (binary) on port $port..."
    if [[ ! -x "$bin" ]]; then
        echo "  ERROR: toolbox binary not found at $bin"
        echo "  Set TOOLBOX_BIN in .env to the correct path."
        return 1
    fi

    "$bin" \
        --config "$config" \
        --address "0.0.0.0" \
        --port "$port" \
        > "$log" 2>&1 &
    echo $! > "$pid_file"
    echo "  PID=$(cat "$pid_file")  log=$log"
    sleep 1
}

# ---------------------------------------------------------------------------
# Start order:
#   0. Ollama (local LLM — started if USE_OLLAMA=true and not already running)
#   1. MCP servers (agents connect to MCPs at startup)
#   2. Vector MCP (may take extra time to load the embedding model + seed DB)
#   3. A2A agents
# ---------------------------------------------------------------------------

# --- Ollama ---
if [[ "${USE_OLLAMA:-false}" == "true" ]]; then
    if ! curl -sf http://localhost:11434/api/version > /dev/null 2>&1; then
        echo "Starting Ollama server..."
        OLLAMA_MODELS="${HOME}/.ollama/models" ollama serve >> "$LOGS_DIR/ollama.log" 2>&1 &
        echo $! > "$PIDS_DIR/ollama.pid"
        for i in $(seq 1 10); do
            sleep 1
            curl -sf http://localhost:11434/api/version > /dev/null 2>&1 && break
        done
        echo "  Ollama ready at http://localhost:11434"
    else
        echo "  Ollama already running at http://localhost:11434"
    fi
fi

start_component "custom_mcp"      "custom_mcp/start.py"         8004
start_toolbox_mcp

echo ""
echo "Starting Vector MCP (loads HuggingFace embedding model + seeds ChromaDB)..."
echo "This may take 30–60 s on first run while the model is downloaded."
start_component "vector_mcp"      "vector_mcp/start.py"         8006
sleep 10  # Vector MCP needs extra time for model init + DB seeding

sleep 2   # Give other MCP servers time to be ready

start_component "analyst"         "agents/analyst/start.py"     8003
start_component "researcher"      "agents/researcher/start.py"  8002
sleep 2   # Give specialist agents time to be ready

start_component "orchestrator"    "agents/orchestrator/start.py" 8001

echo ""
echo "All components started!"
echo ""
echo "  Orchestrator  http://localhost:8001  (A2A coordinator)"
echo "  Researcher    http://localhost:8002  (A2A research agent — Ollama / OpenAI / HuggingFace)"
echo "  Analyst       http://localhost:8003  (A2A data analyst)"
echo "  Custom MCP    http://localhost:8004  (FastMCP: knowledge base + Wikipedia + HF tools)"
echo "  Toolbox MCP   http://localhost:8005  (MCP Toolbox binary: arXiv + OpenAlex HTTP tools)"
echo "  Vector MCP    http://localhost:8006  (FastMCP: HuggingFace embeddings + ChromaDB semantic search)"
echo ""
echo "Run demo: python demo_client.py"
echo "Stop all: ./stop_all.sh"
