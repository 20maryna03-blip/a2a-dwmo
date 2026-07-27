"""Configuration for the MCPToolbox MCP server (public-API edition)."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    TOOLBOX_PORT: int = Field(default=8005, description="Port for the Toolbox MCP server")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # Public API endpoints (no credentials required)
    ARXIV_API_URL: str = Field(
        default="https://export.arxiv.org/api/query",
        description="arXiv Atom API base URL",
    )
    OPENALEX_API_URL: str = Field(
        default="https://api.openalex.org",
        description="OpenAlex open scholarly graph base URL",
    )
    OPENALEX_EMAIL: str = Field(
        default="demo@example.com",
        description="Email sent with OpenAlex requests (increases rate limits)",
    )
