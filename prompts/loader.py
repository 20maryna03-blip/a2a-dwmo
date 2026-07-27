"""
Prompt Loader — renders Jinja2 templates from the registry with optional
few-shot examples and arbitrary context variables.

Design decisions:
- Templates live in prompts/templates/ as *.j2 files.
- Few-shot examples live in prompts/examples/ as JSON files named
  ``{prompt_name}_examples.json``.  They are auto-injected into the render
  context under the ``examples`` key when ``include_examples=True``.
- Context variables are passed as ``**kwargs`` so templates can reference
  them directly (e.g. ``{{ topic }}``, ``{{ agent_name }}``).
"""

import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .registry import PROMPT_REGISTRY

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_EXAMPLES_DIR = Path(__file__).parent / "examples"


class PromptLoader:
    """Renders versioned Jinja2 prompt templates with optional few-shot injection."""

    def __init__(self) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        prompt_name: str,
        version: str | None = None,
        context: dict[str, Any] | None = None,
        include_examples: bool = True,
    ) -> str:
        """Load and render a prompt template.

        Args:
            prompt_name: Key in PROMPT_REGISTRY (e.g. ``"researcher_system"``).
            version: Explicit version string; defaults to the entry's ``latest``.
            context: Extra Jinja2 template variables (merged with examples).
            include_examples: If True, auto-injects few-shot examples from the
                matching ``examples/{prompt_name}_examples.json`` file.

        Returns:
            Rendered prompt string ready for use as a system prompt.
        """
        entry = PROMPT_REGISTRY[prompt_name]
        prompt_version = entry.get(version)

        ctx: dict[str, Any] = dict(context or {})
        # Always provide `examples` so StrictUndefined templates never raise on
        # {% if examples %} — override with real data only when requested.
        ctx.setdefault("examples", [])

        if include_examples:
            examples_file = _EXAMPLES_DIR / f"{prompt_name}_examples.json"
            if examples_file.exists():
                with open(examples_file) as f:
                    ctx["examples"] = json.load(f)
                logger.debug("Injected %d examples into '%s'", len(ctx["examples"]), prompt_name)

        template = self._env.get_template(prompt_version.template_file)
        rendered = template.render(**ctx)
        logger.debug("Rendered prompt '%s' v%s (%d chars)", prompt_name, prompt_version.version, len(rendered))
        return rendered

    def list_prompts(self) -> list[str]:
        """Return the names of all registered prompts."""
        return list(PROMPT_REGISTRY.keys())

    def get_metadata(self, prompt_name: str) -> dict[str, Any]:
        """Return metadata dict for a prompt (name, versions, description, tags)."""
        entry = PROMPT_REGISTRY[prompt_name]
        latest = entry.get()
        return {
            "name": entry.name,
            "latest_version": entry.latest,
            "versions": list(entry.versions.keys()),
            "description": latest.description,
            "tags": latest.tags,
        }
