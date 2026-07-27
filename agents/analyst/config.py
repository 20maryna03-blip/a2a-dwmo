"""Configuration for the Data Analyst A2A agent."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

_ENV_FILE = Path(__file__).parent.parent.parent / ".env"


class Config(BaseSettings):
    model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8", "extra": "ignore"}

    # LLM — OpenAI
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    OPENAI_TEMPERATURE: float = Field(default=0.1, description="LLM temperature")

    # LLM — Ollama (local, no API key needed)
    USE_OLLAMA: bool = Field(default=False, description="Use local Ollama instead of OpenAI")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434/v1", description="Ollama OpenAI-compatible base URL")
    OLLAMA_MODEL: str = Field(default="qwen2.5:3b", description="Ollama model to use")

    # Agent server
    ANALYST_PORT: int = Field(default=8003, description="Analyst agent port")
    ANALYST_URL: str = Field(default="http://localhost:8003", description="Analyst public URL (in AgentCard)")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # MCPToolbox MCP server
    TOOLBOX_MCP_URL: str = Field(
        default="http://localhost:8005/mcp",
        description="URL of the MCPToolbox analytics MCP server",
    )
