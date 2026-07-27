#!/usr/bin/env bash
# =============================================================================
# Stop all AI Platform Demo components
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS_DIR="$ROOT/.pids"

if [[ ! -d "$PIDS_DIR" ]]; then
    echo "No .pids directory found. Nothing to stop."
    exit 0
fi

stop_component() {
    local name="$1"
    local pid_file="$PIDS_DIR/${name}.pid"

    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $name (PID $pid)..."
            kill "$pid"
        else
            echo "$name (PID $pid) is not running."
        fi
        rm -f "$pid_file"
    else
        echo "$name: no PID file found."
    fi
}

stop_component "orchestrator"
stop_component "researcher"
stop_component "analyst"
stop_component "toolbox_mcp"
stop_component "custom_mcp"
stop_component "vector_mcp"
stop_component "ollama"

echo "All components stopped."
