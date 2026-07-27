"""
Toolbox MCP — binary launcher
==============================
Starts the MCP Toolbox for Databases binary (genai-toolbox) with the HTTP
tools defined in tools.yaml.

The binary is the real MCP Toolbox server — no FastMCP, no custom Python
HTTP logic. It reads tools.yaml, connects to the HTTP sources defined there,
and exposes each tool over MCP (streamable-HTTP on /mcp).

Binary:   $TOOLBOX_BIN  (default: ../../images/mlp/bin/toolbox)
Config:   tools.yaml    (next to this file)
Port:     $TOOLBOX_PORT (default: 8005)

Usage:
    python start.py           # uses env vars
    python start.py --port 8105
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TOOLS_YAML = _HERE / "tools.yaml"

# Default: bin/toolbox in the project root (copied from images/mlp/bin/toolbox)
_DEFAULT_BIN = Path(__file__).resolve().parents[1] / "bin" / "toolbox"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch MCP Toolbox HTTP server")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on (overrides TOOLBOX_PORT)")
    parser.add_argument("--address", default="0.0.0.0", help="Bind address (default 0.0.0.0)")
    args = parser.parse_args()

    toolbox_bin = Path(os.environ.get("TOOLBOX_BIN", str(_DEFAULT_BIN)))
    if not toolbox_bin.exists():
        print(
            f"ERROR: toolbox binary not found at {toolbox_bin}\n"
            "Set TOOLBOX_BIN env var to the correct path.",
            file=sys.stderr,
        )
        sys.exit(1)

    port = args.port or int(os.environ.get("TOOLBOX_PORT", "8005"))

    cmd = [
        str(toolbox_bin),
        "--config", str(_TOOLS_YAML),
        "--address", args.address,
        "--port", str(port),
    ]

    print(f"Starting MCP Toolbox binary on {args.address}:{port}")
    print(f"  binary : {toolbox_bin}")
    print(f"  config : {_TOOLS_YAML}")
    print(f"  MCP URL: http://{args.address}:{port}/mcp")
    print(f"  UI    : http://{args.address}:{port}/ui")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
