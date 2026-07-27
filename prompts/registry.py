"""
Prompt Registry — versioned catalog of all prompt templates.

Each entry tracks available versions, the latest default, and metadata
(description, tags). This allows iterative prompt improvements without
breaking existing callers.

Adding a new version::

    PROMPT_REGISTRY["researcher_system"].versions["v2"] = PromptVersion(
        version="v2",
        description="Improved CoT instructions with structured steps",
        template_file="researcher_system_v2.j2",
        tags=["researcher", "cot", "v2"],
    )
    PROMPT_REGISTRY["researcher_system"].latest = "v2"
"""

from dataclasses import dataclass, field


@dataclass
class PromptVersion:
    """Metadata for one version of a prompt."""

    version: str
    description: str
    template_file: str
    tags: list[str] = field(default_factory=list)


@dataclass
class PromptEntry:
    """A named prompt with multiple available versions."""

    name: str
    versions: dict[str, PromptVersion] = field(default_factory=dict)
    latest: str = "v1"

    def get(self, version: str | None = None) -> PromptVersion:
        """Return the requested version (defaults to latest)."""
        v = version or self.latest
        if v not in self.versions:
            available = list(self.versions.keys())
            raise KeyError(f"Version '{v}' not found for prompt '{self.name}'. Available: {available}")
        return self.versions[v]


# ---------------------------------------------------------------------------
# Registry definition
# ---------------------------------------------------------------------------

PROMPT_REGISTRY: dict[str, PromptEntry] = {
    "orchestrator_system": PromptEntry(
        name="orchestrator_system",
        latest="v1",
        versions={
            "v1": PromptVersion(
                version="v1",
                description="Orchestrator system prompt — delegates to researcher and analyst agents",
                template_file="orchestrator_system.j2",
                tags=["orchestrator", "coordinator", "multi-agent"],
            ),
        },
    ),
    "researcher_system": PromptEntry(
        name="researcher_system",
        latest="v1",
        versions={
            "v1": PromptVersion(
                version="v1",
                description="Researcher system prompt — gathers & saves findings via Custom MCP",
                template_file="researcher_system.j2",
                tags=["researcher", "knowledge", "chain-of-thought"],
            ),
        },
    ),
    "analyst_system": PromptEntry(
        name="analyst_system",
        latest="v1",
        versions={
            "v1": PromptVersion(
                version="v1",
                description="Analyst system prompt — queries DB via MCPToolbox and produces insights",
                template_file="analyst_system.j2",
                tags=["analyst", "data", "sql", "chain-of-thought"],
            ),
        },
    ),
    "report_format": PromptEntry(
        name="report_format",
        latest="v1",
        versions={
            "v1": PromptVersion(
                version="v1",
                description="Structured report generation prompt with JSON/Markdown output spec",
                template_file="report_format.j2",
                tags=["report", "output-format", "structured"],
            ),
        },
    ),
}
