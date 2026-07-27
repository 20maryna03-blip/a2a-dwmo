"""Configuration for the Research A2A agent."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# Resolve .env relative to this file's location (project root is two levels up)
_ENV_FILE = Path(__file__).parent.parent.parent / ".env"


class Config(BaseSettings):
    model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8", "extra": "ignore"}

    # LLM — HuggingFace Inference API (recommended, OpenAI-compatible endpoint)
    USE_HF_LLM: bool = Field(default=False, description="Use HuggingFace Inference API as LLM backend")
    HF_API_KEY: str = Field(default="", description="HuggingFace Hub token (hf_...)")
    HF_MODEL: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="HuggingFace model ID — must support tool calling",
    )
    HF_BASE_URL: str = Field(
        default="https://api-inference.huggingface.co/v1/",
        description="HuggingFace OpenAI-compatible inference endpoint",
    )

    # LLM — OpenAI (fallback)
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    OPENAI_TEMPERATURE: float = Field(default=0.2, description="LLM temperature")

    # LLM — Ollama (local, no API key needed)
    USE_OLLAMA: bool = Field(default=False, description="Use local Ollama instead of OpenAI")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434/v1", description="Ollama OpenAI-compatible base URL")
    OLLAMA_MODEL: str = Field(default="qwen2.5:3b", description="Ollama model to use")

    # Agent server
    RESEARCHER_PORT: int = Field(default=8002, description="Researcher agent port")
    RESEARCHER_URL: str = Field(
        default="http://localhost:8002",
        description="Researcher public URL (in AgentCard)",
    )
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # MCP servers
    CUSTOM_MCP_URL: str = Field(
        default="http://localhost:8004/mcp",
        description="Custom MCP server (knowledge base + web + HuggingFace tools)",
    )
    VECTOR_MCP_URL: str = Field(
        default="http://localhost:8006/mcp",
        description="Vector MCP server (semantic search, HuggingFace embeddings + ChromaDB)",
    )
