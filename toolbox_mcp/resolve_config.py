"""
resolve_config.py
=================
One-shot script that fills ``${VAR}`` / ``${VAR:-default}`` placeholders in
``tools.yaml`` from environment variables (or DataRobot MLOPS_RUNTIME_PARAM_*
prefixed parameters).

This is the same pattern used in the teradata-mcp blueprint.

Usage (called automatically by start.py when --use-toolbox is set):
    python -m resolve_config          # reads/writes tools.yaml in current dir
    python -m resolve_config --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _resolve(text: str) -> str:
    """Replace all ${VAR} and ${VAR:-default} placeholders in *text*."""

    def _replace(m: re.Match) -> str:  # type: ignore[type-arg]
        var_name: str = m.group(1)
        default: str | None = m.group(2)

        # Support DataRobot MLOPS_RUNTIME_PARAM_ prefix
        value = (
            os.environ.get(var_name)
            or os.environ.get(f"MLOPS_RUNTIME_PARAM_{var_name}")
        )

        if value is not None:
            return value
        if default is not None:
            return default
        print(f"WARNING: no value for ${{{var_name}}} and no default defined.", file=sys.stderr)
        return m.group(0)  # leave placeholder intact

    return _PLACEHOLDER_RE.sub(_replace, text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Resolve placeholder variables in tools.yaml")
    parser.add_argument(
        "--tools-file",
        default=str(Path(__file__).parent / "tools.yaml"),
        help="Path to the tools.yaml file (default: tools.yaml next to this script)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved output without writing")
    args = parser.parse_args(argv)

    src = Path(args.tools_file)
    if not src.exists():
        sys.exit(f"ERROR: tools file not found: {src}")

    raw = src.read_text()
    resolved = _resolve(raw)

    if args.dry_run:
        print(resolved)
        return

    src.write_text(resolved)
    print(f"Resolved placeholders in {src}")


if __name__ == "__main__":
    main()
